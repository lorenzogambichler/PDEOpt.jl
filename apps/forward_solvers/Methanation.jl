using LinearAlgebra
using SparseArrays
using WriteVTK
using PDEOpt

# Discretization cache: 
# operators, mesh, physics model
struct ProblemCache{Tgrid, Tgeom, Tdm, Tmodel}
    M::SparseMatrixCSC{Float64,Int}
    K::SparseMatrixCSC{Float64,Int}
    Jr::SparseMatrixCSC{Float64,Int} 
    r::Vector{Float64} # global reaction source
    re::Vector{Float64} # local reaction source 
    Jre::Matrix{Float64}
    f_in::Vector{Float64} # inlet forcing
    f_wall::Vector{Float64} # wall forcing
    grid::Tgrid
    geom::Tgeom
    dm::Tdm
    model::Tmodel # AbstractModel
end

function write_vtk(path::String, cache::CNCache)
    prob = cache.prob
    grid = prob.grid
    nz, nr = grid.nz, grid.nr
    Cdofs, Tdofs = fielddof(prob.dm, :C), fielddof(prob.dm, :T)
    mkpath(dirname(path))
    paraview_collection(path) do pvd
        for n = 1:cache.N
            t = (n - 1) * cache.Δt
            # cell-index order, reshape (nr,nz) then transpose -> [i,j]
            Cgrid = permutedims(reshape(cache.y[Cdofs, n], nr, nz))
            Tgrid = permutedims(reshape(cache.y[Tdofs, n], nr, nz))
            vtk_grid("$(path)_$(n)", grid.z, grid.r) do vtk
                vtk["C", VTKCellData()] = Cgrid
                vtk["T", VTKCellData()] = Tgrid
                pvd[t] = vtk
            end
        end
    end
end

# Initial condition: 
# Constant at inlet value at t=0
function set_ic!(cache::CNCache)
    prob = cache.prob
    for field in prob.dm.fields
        @views cache.y[fielddof(prob.dm, field), 1] .= inlet_mod(prob.model, field, 0.0)
    end
    return cache
end

function setup_problem(tf::Float64)
    # StructuredGrid, Geometry
    L, R = 2.0, 0.5
    nz, nr = 100, 25
    grid = StructuredGrid(L, R, nz, nr)
    geom = Geometry(grid)

    # FaceSets, FaceGeometries
    fs_axial, fs_radial = interior_faces(grid)
    fg_axial = FaceGeometry(grid, geom, fs_axial)
    fg_radial = FaceGeometry(grid, geom, fs_radial)

    # BoundaryFaceSets, BoundaryFaceGeometries
    bfs = boundary_faces(grid)
    bfg_inlet = BoundaryFaceGeometry(grid, geom, bfs.inlet)
    bfg_outlet = BoundaryFaceGeometry(grid, geom, bfs.outlet)
    bfg_wall = BoundaryFaceGeometry(grid, geom, bfs.wall)

    # Fields + DofMap
    fields = (:ρCH4, :ρCO, :ρCO2, :ρH2O, ρH2, ρN2, :T) # p algebraic
    dm = DofMap(ncells(grid), fields)

    # Coefficients
    # TODO 
    
    # Boundary conditions
    # Inlet
    g = (C = r -> 1.0, T = r -> 1.0)
    t_ramp = 0.5 * tf
    C_in(t) = 1.0
    T_in(t) = 100.0 + (500.0 - 100.0) * clamp(t / t_ramp, 0.0, 1.0) # 100K -> 500K
    y_in = (C = C_in, T = T_in)
    # Wall
    T_wall(t) = 100.0 

    # Model
    react = LHHW()
    model = ADRModel(fields, D, β, react, U, T_wall, g, y_in)

    # Matrices, shared pattern
    K = sparsity_pattern(grid, dm)
    M = copy(K)
    Jr = copy(K)

    # Assembly 
    f_in = zeros(ndof(dm))
    f_wall = zeros(ndof(dm))
    assemble_global(M, K, ((fs_axial,fg_axial),(fs_radial,fg_radial)), grid, geom, dm, model)
    assemble_inlet_outlet!(K, f_in, dm, geom, grid, model,
        (bfs.inlet, bfg_inlet), (bfs.outlet, bfg_outlet))
    assemble_wall!(K, f_wall, dm, model, bfs.wall, bfg_wall)

    # Reaction: global source + local cell scratch (source, jac)
    nf = length(dm.fields)
    r = zeros(ndof(dm))
    re = zeros(nf)
    Jre = zeros(nf, nf)

    return ProblemCache(M, K, Jr, r, re, Jre, f_in, f_wall, grid, geom, dm, model)
end

function main()
    # Time-stepping
    tf = 2.5
    Δt = 0.01
    N = round(Int, tf / Δt) + 1

    # Caches
    prob = setup_problem(tf)
    cache = CNCache(prob, Δt, N)

    # IC
    set_ic!(cache)

    cn_solve!(cache, (y, WithJac) -> assemble_react_global!(prob, y, WithJac),
        (b, t) -> assemble_boundary_global!(prob, b, t), frozen_newton()) # newton!, chord!, shamanskii()
    write_vtk("apps/forward_solvers/results/Methanation/Methanation", cache)
end
