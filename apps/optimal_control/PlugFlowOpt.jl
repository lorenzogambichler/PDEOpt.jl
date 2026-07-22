ENV["OMP_NUM_THREADS"] = "4"
using LinearAlgebra
using SparseArrays
using WriteVTK
using PDEOpt

# includet separately (not in PDEOpt)
# avoid loading IPOPT and JuMP for other apps
includet("../../src/optimization/jump_ipopt.jl")

function write_vtk(path::String, prob::ProblemCache, y::AbstractMatrix, Δt::Float64, N::Int)
    grid = prob.grid
    nz, nr = grid.nz, grid.nr
    Cdofs, Tdofs = fielddof(prob.dm, :C), fielddof(prob.dm, :T)
    mkpath(dirname(path))
    paraview_collection(path) do pvd
        for n = 1:N
            t = (n - 1) * Δt
            Cgrid = permutedims(reshape(y[Cdofs, n], nr, nz))
            Tgrid = permutedims(reshape(y[Tdofs, n], nr, nz))
            vtk_grid("$(path)_$(n)", grid.z, grid.r) do vtk
                vtk["C", VTKCellData()] = Cgrid
                vtk["T", VTKCellData()] = Tgrid
                pvd[t] = vtk
            end
        end
    end
end

write_vtk(path::String, cache::CNCache) =
    write_vtk(path, cache.prob, cache.y, cache.Δt, cache.N)

# csv for control
function write_control(path::String, u::AbstractVector, Δt::Float64, N::Int)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "t,Tw")
        for n = 1:N
            println(io, (n - 1) * Δt, ",", u[n])
        end
    end
end

function setup_problem()
    # StructGrid2D, Geometry
    L, R = 2.5, 0.1
    nz, nr = 50, 5
    grid = StructGrid2D(L, R, nz, nr)
    geom = Geometry2D(grid)

    # FaceSets, FaceGeometries
    (fs_axial, fs_radial) = interior_faces(grid)
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
    D = [0.0 1e-4; 0.0 1e-3] # m^2/s, W/(mK)
    β = [2.0, 0.0] # m/s, axial advection
    k0 = 5e5 # 1/s
    Ea = 40e3 # J/mol
    ΔHr = -250 # J/mol
    T_floor = 0.1 # K
    U = 50.0 # W/(m^2K), 1/(1/α_inner + d_w/λ_w + 1/α_outer)

    # Boundary conditions
    # Inlet
    g = (C = r -> 1.0, T = r -> 1.0)
    #t_ramp = 1.0
    C_in(t) = 1.0
    T_in(t) = 200.0
    #T_in(t) = 100.0 + (500.0 - 100.0) * clamp(t / t_ramp, 0.0, 1.0) # 100K -> 500K
    y_in = (C = C_in, T = T_in)
    # Wall
    T_wall(t) = 300.0

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
    assemble_mass!(M, grid, geom, dm)
    assemble_transport!(K, ((fs_axial,fg_axial),(fs_radial,fg_radial)), dm, model)
    assemble_inlet_outlet!(K, f_in, dm, geom, grid, model,
        (bfs.inlet, bfg_inlet), (bfs.outlet, bfg_outlet))
    assemble_wall!(K, f_wall, dm, model, bfs.wall, bfg_wall)

    # Reaction
    nf = length(dm.fields)
    r = zeros(ndof(dm))
    re = zeros(nf)
    Jre = zeros(nf, nf)

    return ProblemCache(M, K, Jr, r, re, Jre, f_in, f_wall, grid, geom, dm, model)
end

function main()
    # Time-stepping params
    tf = 2.0
    Δt = 0.005
    N = round(Int, tf / Δt) + 1

    # NLP params
    Tw_min = 200.0
    Tw_max = 500.0
    Tmax = 500.0
    γ = 10.0

    # Problem
    prob = setup_problem()

    # Integrator
    cn = CNCache(prob, Δt, N)
    set_ic!(cn, :T, inlet_mod(prob.model, :T, 0.0))
    set_ic!(cn, :C, inlet_mod(prob.model, :C, 0.0))
    
    # Solve
    @time cn_solve!(cn, (y, WithJac) -> assemble_react!(prob, y, WithJac),
        (b, t) -> assemble_boundary!(prob, b, t), shamanskii()) # newton!, chord!, shamanskii()
    #println("Factorizations: ", cn.stats.nfact)

    # NLP
    opt = OptCache(γ, Tmax, Tw_min, Tw_max, cn.y[:, 1], cn.y)
    model, vars = build_nlp(prob, prob.model, opt, Δt, N)
    res = solve_nlp!(model, vars)

    # Visualize
    resultsdir = joinpath(@__DIR__, "results/PlugFlow")
    write_vtk(joinpath(resultsdir, "opt"), prob, res.y, Δt, N)
    write_vtk(joinpath(resultsdir, "forward"), cn)
    write_control(joinpath(resultsdir, "control.csv"), res.u, Δt, N)

    return res
end
