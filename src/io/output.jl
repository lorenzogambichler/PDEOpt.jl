module Output

# VTK/CSV writers for generic time grid, use dense_output to generate Y

using WriteVTK
using ..StructuredMesh
using ..Problem
using ..CrankNicolson

export write_vtk, write_control

# Y[:, n] is state at times[n]
function write_vtk(path::String, prob::ProblemCache, Y::AbstractMatrix,
    times::AbstractVector)
    grid = prob.grid
    nz, nr = grid.nz, grid.nr
    length(times) <= size(Y, 2) ||
        error("$(length(times)) times but Y has only $(size(Y, 2)) columns")
    mkpath(dirname(path))
    paraview_collection(path) do pvd
        for n in eachindex(times)
            vtk_grid("$(path)_$(n)", grid.z, grid.r) do vtk
                for field in prob.dm.fields
                    fd = fielddof(prob.dm, field)
                    # cell-index order
                    vtk[string(field), VTKCellData()] = permutedims(reshape(Y[fd, n], nr, nz))
                end
                pvd[times[n]] = vtk
            end
        end
    end
end

# uniform grid -> cols 1:N at t = 0, Δt, …, (N-1)Δt
write_vtk(path::String, prob::ProblemCache, Y::AbstractMatrix, Δt::Real, N::Int) =
    write_vtk(path, prob, Y, range(0.0; step=Float64(Δt), length=N))

write_vtk(path::String, cache::CNCache) =
    write_vtk(path, cache.prob, cache.y, cache.Δt, cache.N)

# csv for piecewise-constant control (u[k] const on element k, left edge t = (k-1)Δt)
function write_control(path::String, u::AbstractVector, Δt::Float64, N::Int=length(u))
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "t,Tw")
        for n = 1:N
            println(io, (n - 1) * Δt, ",", u[n])
        end
    end
end

end
