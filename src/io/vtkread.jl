module VTKRead

# Reader for .vtr/.pvd output written by Output.write_vtk,

using LightXML
using CodecZlib

export VTRFile, VTRSeries, read_vtr, read_pvd, read_series, cellcentres

struct VTRFile
    z::Vector{Float64} # axial node coords (nz + 1)
    r::Vector{Float64} # radial node coords (nr + 1)
    data::Dict{String,Matrix{Float64}} # field -> (nz, nr) cell values
end

struct VTRSeries
    t::Vector{Float64} # nt timestep values
    z::Vector{Float64} # axial node coords (nz + 1)
    r::Vector{Float64} # radial node coords (nr + 1)
    data::Dict{String,Array{Float64,3}} # field -> (nz, nr, nt)
end

Base.getindex(s::VTRSeries, f::AbstractString) = s.data[f]
Base.keys(s::VTRSeries) = keys(s.data)

# cell centres from node coords
cellcentres(x::AbstractVector) = @views 0.5 .* (x[1:end-1] .+ x[2:end])

function _findseq(hay::Vector{UInt8}, needle::Vector{UInt8}, from::Int=1)
    n, m = length(hay), length(needle)
    m == 0 && return from
    @inbounds for i = from:(n-m+1)
        ok = true
        for j = 1:m
            hay[i+j-1] == needle[j] || (ok = false; break)
        end
        ok && return i
    end
    return nothing
end

function _split_appended(bytes::Vector{UInt8})
    i = _findseq(bytes, Vector{UInt8}(codeunits("<AppendedData")))
    i === nothing && error("no <AppendedData> section (only raw-appended .vtr is supported)")
    j = findnext(==(UInt8('_')), bytes, i)
    j === nothing && error("<AppendedData> present but no '_' payload marker")
    return String(@view bytes[1:i-1]) * "</VTKFile>", j + 1
end

function _read_array(bytes::Vector{UInt8}, base::Int, offset::Int)
    p = base + offset
    word(k) = reinterpret(UInt64, @view bytes[p+8(k-1):p+8k-1])[1]
    nblocks = Int(word(1))
    nblocks == 0 && return Float64[]
    csizes = [Int(word(3 + k)) for k = 1:nblocks]
    q = p + 8 * (3 + nblocks)
    raw = UInt8[]
    for c in csizes
        append!(raw, transcode(ZlibDecompressor, bytes[q:q+c-1]))
        q += c
    end
    length(raw) % 8 == 0 ||
        error("decompressed array is $(length(raw)) bytes, not a multiple of 8")
    return collect(reinterpret(Float64, raw))
end

function read_vtr(path::AbstractString)
    bytes = read(path)
    xmlstr, base = _split_appended(bytes)
    doc = parse_string(xmlstr)
    try
        vtk = root(doc)
        attribute(vtk, "byte_order") == "LittleEndian" ||
            error("only LittleEndian files are supported")
        attribute(vtk, "header_type") == "UInt64" ||
            error("only UInt64 header_type is supported")
        cmp = attribute(vtk, "compressor")
        (cmp === nothing || cmp == "vtkZLibDataCompressor") ||
            error("unsupported compressor $cmp")

        piece = find_element(find_element(vtk, "RectilinearGrid"), "Piece")
        ext = parse.(Int, split(attribute(piece, "Extent")))
        nz, nr = ext[2] - ext[1], ext[4] - ext[3]

        coords = find_element(piece, "Coordinates")
        off(e) = parse(Int, attribute(e, "offset"))
        cmap = Dict(attribute(e, "Name") => off(e) for e in child_elements(coords))
        z = _read_array(bytes, base, cmap["x"])
        r = _read_array(bytes, base, cmap["y"])

        data = Dict{String,Matrix{Float64}}()
        for e in child_elements(find_element(piece, "CellData"))
            v = _read_array(bytes, base, off(e))
            length(v) == nz * nr ||
                error("field $(attribute(e, "Name")): $(length(v)) values, expected $(nz*nr)")
            # VTK cell order is x-fastest -> (axial, radial)
            data[attribute(e, "Name")] = reshape(v, nz, nr)
        end
        return VTRFile(z, r, data)
    finally
        free(doc)
    end
end

# (times, absolute file paths), ordered by time
function read_pvd(path::AbstractString)
    doc = parse_file(path)
    try
        coll = find_element(root(doc), "Collection")
        coll === nothing && error("$path is not a VTK Collection")
        t = Float64[]
        files = String[]
        for ds in child_elements(coll)
            push!(t, parse(Float64, attribute(ds, "timestep")))
            push!(files, joinpath(dirname(path), attribute(ds, "file")))
        end
        p = sortperm(t)
        return t[p], files[p]
    finally
        free(doc)
    end
end

function read_series(pvd::AbstractString)
    t, files = read_pvd(pvd)
    isempty(files) && error("$pvd lists no datasets")
    first_file = read_vtr(files[1])
    nz, nr = size(first(values(first_file.data)))
    names = collect(keys(first_file.data))
    data = Dict(n => Array{Float64,3}(undef, nz, nr, length(t)) for n in names)
    for (n, path) in enumerate(files)
        f = n == 1 ? first_file : read_vtr(path)
        for name in names
            size(f.data[name]) == (nz, nr) ||
                error("$path: field $name is $(size(f.data[name])), expected $((nz, nr))")
            data[name][:, :, n] = f.data[name]
        end
    end
    return VTRSeries(t, first_file.z, first_file.r, data)
end

end
