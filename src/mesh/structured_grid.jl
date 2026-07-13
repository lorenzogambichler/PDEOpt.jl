module StructuredMesh

using SparseArrays

# Implements basic tensor product grid:
# Connectivity
# Geometry
# Dof map (for multiple fields)
# Sparsity pattern 

## Connectivity

export StructuredGrid, ncells, cellindex, cellij,
    FaceSet, interior_faces, BoundaryFaceSet, boundary_faces


struct StructuredGrid
    z::Vector{Float64} # axial node coords (nz + 1)
    r::Vector{Float64} # radial node coords (nr + 1) 
    nz::Int # ncells axial 
    nr::Int # ncells radial 
end

# For non-uniform grid (refinement), call StrcutredGrid with custom node vectors
function StructuredGrid(Lz::Float64, Lr::Float64, nz::Int, nr::Int)
    z = collect(range(0.0, Lz; length = nz + 1)) # axial node vector
    r = collect(range(0.0, Lr; length = nr + 1)) # radial node vector
    return StructuredGrid(z, r, nz, nr)
end

ncells(g::StructuredGrid) = g.nz * g.nr

# radial j varies fastest
cellindex(g::StructuredGrid, i::Int, j::Int) = (i - 1) * g.nr + j

function cellij(g::StructuredGrid, c::Int)
    i, j = divrem(c - 1, g.nr)
    return (i + 1, j + 1)
end

# Split faces by axis, axis determines normal
struct FaceSet
    owner::Vector{Int} 
    neighbor::Vector{Int}
    axis::Int # 1(axial), 2 (radial)
end

# +flux -> owner, −flux -> neighbor 
function interior_faces(g::StructuredGrid)
    # Axial faces, between (i, j) and (i+1, j), i = 1:nz-1, ∀j.
    na = (g.nz - 1) * g.nr
    ao = Vector{Int}(undef, na)
    an = Vector{Int}(undef, na)
    k = 0
    for i in 1:g.nz-1, j in 1:g.nr
        k += 1
        ao[k] = cellindex(g, i, j)
        an[k] = cellindex(g, i + 1, j)
    end
    axial = FaceSet(ao, an, 1)

    # Radial faces, between (i, j) and (i, j+1), j = 1:nr-1, ∀i. 
    nrd = g.nz * (g.nr - 1)
    ro = Vector{Int}(undef, nrd)
    rn = Vector{Int}(undef, nrd)
    k = 0
    for i in 1:g.nz, j in 1:g.nr-1
        k += 1
        ro[k] = cellindex(g, i, j)
        rn[k] = cellindex(g, i, j + 1)
    end
    radial = FaceSet(ro, rn, 2)

    return axial, radial
end

# Axis, side determine normal vector
struct BoundaryFaceSet
    cells::Vector{Int}
    axis::Int # 1 (axial), 2 (radial)
    side::Int # -1 (low), +1 (high)
end

function boundary_faces(g::StructuredGrid)
    inlet = BoundaryFaceSet([cellindex(g, 1, j) for j in 1:g.nr], 1, -1) # z = 0
    outlet = BoundaryFaceSet([cellindex(g, g.nz, j) for j in 1:g.nr], 1, +1) # z = L
    center = BoundaryFaceSet([cellindex(g, i, 1) for i in 1:g.nz], 2, -1) # r = 0 (symmetry)
    wall = BoundaryFaceSet([cellindex(g, i, g.nr) for i in 1:g.nz], 2, +1) # r = R
    return (; inlet, outlet, center, wall) # named tuple
end

## Geometry

export Geometry, cellvolume, cellcentroid, FaceGeometry, BoundaryFaceGeometry

struct Geometry
    dz::Vector{Float64} # axial widths (nz)
    dr::Vector{Float64} # radial widths (nr)
    zc::Vector{Float64} # axial centroid (nz), midpoint
    rc::Vector{Float64} # radial centroid (nr), area-weighted
    rbar::Vector{Float64} # arithmetic mid-radius (nr)
end

function Geometry(g::StructuredGrid)
    z = g.z
    r = g.r
    dz = diff(z)
    dr = diff(r)
    zc = @. 0.5 * (z[1:end-1] + z[2:end])
    rbar = @. 0.5 * (r[1:end-1] + r[2:end])
    rc = @. rbar + dr^2 / (12.0 * rbar)
    return Geometry(dz, dr, zc, rc, rbar)
end

cellvolume(geom::Geometry, i::Int, j::Int) = geom.rbar[j] * geom.dr[j] * geom.dz[i]
cellcentroid(geom::Geometry, i::Int, j::Int) = (geom.zc[i], geom.rc[j])

struct FaceGeometry
    area::Vector{Float64}
    dist::Vector{Float64} # owner -> neighbor centroid dist
    wf::Vector{Float64} # interp weight for owner
end

function FaceGeometry(g::StructuredGrid, geom::Geometry, fs::FaceSet)
    n = length(fs.owner)
    area = Vector{Float64}(undef, n)
    dist = Vector{Float64}(undef, n)
    wf = Vector{Float64}(undef, n)
    for f in 1:n
        i, j = cellij(g, fs.owner[f])
        if fs.axis == 1
            # Axial face (i,j), (i+1,j) 
            area[f] = geom.rbar[j] * geom.dr[j]
            do_ = 0.5 * geom.dz[i] # owner centroid -> face
            dn_ = 0.5 * geom.dz[i+1] # neighbor centroid -> face
            dist[f] = geom.zc[i+1] - geom.zc[i]
            wf[f] = dn_ / (do_ + dn_)
        else
            # Radial face (i,j), (i,j+1) 
            area[f] = g.r[j+1] * geom.dz[i] # outer 
            do_ = g.r[j+1] - geom.rc[j]
            dn_ = geom.rc[j+1] - g.r[j+1]
            dist[f] = geom.rc[j+1] - geom.rc[j]
            wf[f] = dn_ / (do_ + dn_)
        end
    end
    return FaceGeometry(area, dist, wf)
end

struct BoundaryFaceGeometry
    area::Vector{Float64}
    dist::Vector{Float64}
end

function BoundaryFaceGeometry(g::StructuredGrid, geom::Geometry, bf::BoundaryFaceSet)
    n = length(bf.cells)
    area = Vector{Float64}(undef, n)
    dist = Vector{Float64}(undef, n)
    for k in 1:n
        i, j = cellij(g, bf.cells[k])
        if bf.axis == 1
            # Inlet, outlet (axial)
            area[k] = geom.rbar[j] * geom.dr[j]
            dist[k] = 0.5 * geom.dz[i]
        else
            # Center, wall (radial) 
            rf = bf.side == -1 ? g.r[j] : g.r[j+1] # 0 flux center
            area[k] = rf * geom.dz[i]
            dist[k] = bf.side == -1 ? (geom.rc[j] - g.r[j]) : (g.r[j+1] - geom.rc[j])
        end
    end
    return BoundaryFaceGeometry(area, dist)
end

## DOF map (multi-field)

export DofMap, ndof, fieldindex, dof, celldof, fielddof

# Cell interleaved, e.g. (C1, T1, C2, T2, ...)
# Every axial coord forms memory block (nr*nfields dofs) 
# avoid ferrite names
struct DofMap{N}
    ncells::Int
    fields::NTuple{N,Symbol} # e.g. (:C, :T)
end
DofMap(ncells::Int, fields::Symbol...) = DofMap(ncells, fields) 

ndof(dm::DofMap{N}) where {N} = N * dm.ncells

# all examples for fields dm = DofMap(10, :C, :T)

# local ordering index of field in cell
fieldindex(dm::DofMap, field::Symbol) = findfirst(==(field), dm.fields)
# e.g. 1 for :C , 2 for :T 

# global dof for specific field of specific cell
dof(dm::DofMap{N}, cell::Int, f::Int) where {N} = (cell - 1) * N + f
dof(dm::DofMap, cell::Int, field::Symbol) = dof(dm, cell, fieldindex(dm, field))
# e.g. 20 for dof(dm, 10, :T)

# global dof range for all fields of specific cell 
celldof(dm::DofMap{N}, cell::Int) where {N} = ((cell - 1) * N + 1):(cell * N)
# e.g 19:20 for celldof(dm ,10)

# global dof range for specific field over all cells
fielddof(dm::DofMap{N}, field::Symbol) where {N} = fieldindex(dm, field):N:ndof(dm)
# e.g. 1:2:19 for feilddof(dm, :C)

## Sparsity pattern

export sparsity_pattern

# for K, but use copy for diagonal M and Jr 
function sparsity_pattern(g::StructuredGrid, dm::DofMap;
    coupling::Union{Nothing,AbstractMatrix{Bool}}=nothing, # field coupling inside cell (reaction)
    face_coupling::Union{Nothing,AbstractMatrix{Bool}}=nothing) # field coupling across faces (transport)

    nf = length(dm.fields)
    nc = dm.ncells
    n = ndof(dm)
    coupling = isnothing(coupling) ? trues(nf, nf) : coupling # default is full coupling inside cell
    face_coupling = (isnothing(face_coupling) ? 
                    [a == b for a in 1:nf, b in 1:nf] : face_coupling) # default is only field coupling 

    axial, radial = interior_faces(g)
    nfaces = length(axial.owner) + length(radial.owner)
    nnz_max = nc * count(coupling) + nfaces * count(face_coupling) * 2
    I = Vector{Int}(undef, nnz_max)
    J = Vector{Int}(undef, nnz_max)
    k = 0

    # within-cell field coupling (reaction block + all diagonals)
    for c in 1:nc
        for fb in 1:nf, fa in 1:nf
            coupling[fa, fb] || continue
            k += 1
            I[k] = dof(dm, c, fa)
            J[k] = dof(dm, c, fb)
        end
    end

    # across-face coupling (transport), same-field by default, both directions
    for fs in (axial, radial)
        for e in 1:length(fs.owner)
            o = fs.owner[e]
            nb = fs.neighbor[e]
            for fb in 1:nf, fa in 1:nf
                face_coupling[fa, fb] || continue
                go = dof(dm, o, fa)
                gn = dof(dm, nb, fb)
                k += 1
                I[k] = go
                J[k] = gn
                k += 1
                I[k] = gn
                J[k] = go
            end
        end
    end

    # build with nonzeros, then zero 
    S = sparse(view(I, 1:k), view(J, 1:k), ones(k), n, n)
    fill!(nonzeros(S), 0.0)
    return S
end

end 
