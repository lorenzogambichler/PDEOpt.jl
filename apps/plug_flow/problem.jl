const FIELDS = (:C, :T)

function setup_problem(; nz=50, nr=5, L=2.5, R=0.1, T_wall=t->100.0)
    # StructGrid2D, Geometry
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
    dm = DofMap(ncells(grid), FIELDS)

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
    C_in(t) = 1.0
    T_in(t) = 200.0
    y_in = (C = C_in, T = T_in)

    # Model
    react = Arrhenius(k0, Ea, ΔHr, T_floor)
    model = PlugFlowModel(FIELDS, D, β, react, U, T_wall, g, y_in)

    # Matrices, shared pattern
    K = sparsity_pattern(grid, dm)
    M = copy(K)
    Jr = copy(K)

    # Assembly
    f_in = zeros(ndof(dm))
    f_wall = zeros(ndof(dm))
    assemble_mass!(M, grid, geom, dm)
    assemble_transport!(K, ((fs_axial, fg_axial), (fs_radial, fg_radial)), dm, model)
    assemble_inlet_outlet!(K, f_in, dm, geom, grid, model,
        (bfs.inlet, bfg_inlet), (bfs.outlet, bfg_outlet))
    assemble_wall!(K, f_wall, dm, model, bfs.wall, bfg_wall)

    # Reaction: global source + local cell scratch (source, jac)
    nf = length(FIELDS)
    r = zeros(ndof(dm))
    re = zeros(nf)
    Jre = zeros(nf, nf)

    return ProblemCache(M, K, Jr, r, re, Jre, f_in, f_wall, grid, geom, dm, model)
end

# CN forward solve, IC at the inlet state
function cn_forward(Twfun, tf::Float64, Δt::Float64; solver=shamanskii())
    prob = setup_problem(T_wall=Twfun)
    N = round(Int, tf / Δt) + 1
    cache = CNCache(prob, Δt, N)
    for field in prob.dm.fields
        set_ic!(cache, field, inlet_mod(prob.model, field, 0.0))
    end
    cn_solve!(cache, (y, WithJac) -> assemble_react!(prob, y, WithJac),
        (b, t) -> assemble_boundary!(prob, b, t), solver) # newton!, chord!, shamanskii()
    return cache
end
