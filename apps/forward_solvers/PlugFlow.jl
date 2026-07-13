using LinearAlgebra
using SparseArrays
using WriteVTK
using PDEOpt

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

function setup_problem()
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
    fields = (:C, :T)
    dm = DofMap(ncells(grid), fields)

    # Coefficients
    # D[field, axis], e.g. [D1_ax, D1_rad; ...; λ_ax, λ_rad]
    D = [0.0 1e-2; 0.0 5e-2] # m^2/s, W/(mK)
    β = [2.0, 0.0] # m/s, axial advection
    k0 = 1e7 # 1/s
    Ea = 40e3 # J/mol
    ΔHr = -120.0 # J/mol
    T_floor = 0.1 # K
    U = 50.0 # W/(m^2K), 1/(1/α_inner + d_w/λ_w + 1/α_outer)

    # Boundary conditions
    # Inlet
    g = (C = r -> 1.0, T = r -> 1.0)
    t_ramp = 1.0
    C_in(t) = 1.0
    T_in(t) = 100.0 + (500.0 - 100.0) * clamp(t / t_ramp, 0.0, 1.0) # 100K -> 500K
    y_in = (C = C_in, T = T_in)
    # Wall
    T_wall(t) = 100.0 

    # Model
    react = Arrhenius(k0, Ea, ΔHr, T_floor)
    model = PlugFlowModel(fields, D, β, react, U, T_wall, g, y_in)

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
    tf = 3.0
    Δt = 0.01
    N = round(Int, tf / Δt) + 1

    # Caches
    prob = setup_problem()
    cache = CNCache(prob, Δt, N)

    # IC
    set_ic!(cache)
    
    # Solve and write results
    @time cn_solve!(cache, (y, WithJac) -> assemble_react!(prob, y, WithJac),
        (b, t) -> assemble_boundary!(prob, b, t), shamanskii()) # newton!, chord!, shamanskii()
    write_vtk("apps/forward_solvers/results/PlugFlow/PlugFlow", cache)
end
