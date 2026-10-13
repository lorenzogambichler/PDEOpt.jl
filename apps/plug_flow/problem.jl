const FIELDS = (:C, :T)

function setup_problem(; nz=100, L=1.0, T_wall=600.0)
    # StructGrid2D, Geometry
    grid = StructGrid1D(L, nz)
    geom = Geometry1D(grid)

    # FaceSets, FaceGeometries
    (fs,) = interior_faces(grid)
    fg = FaceGeometry(grid, geom, fs)

    # BoundaryFaceSets, BoundaryFaceGeometries
    bfs = boundary_faces(grid)
    bfg_inlet = BoundaryFaceGeometry(grid, geom, bfs.inlet)
    bfg_outlet = BoundaryFaceGeometry(grid, geom, bfs.outlet)

    # Fields + DofMap
    dm = DofMap(ncells(grid), FIELDS)

    # Coefficients
    # D[field], e.g. [D1, D2, ..., λ]
    D = [1e-4, 1e-5] # m^2/s, W/(mK)
    β = 2.0 # m/s, axial advection
    k0 = 5e5 # 1/s
    Ea = 40e3 # J/mol
    ΔHr = -250 # J/mol
    U = 50.0 # W/(m^2K), 1/(1/α_inner + d_w/λ_w + 1/α_outer)
    d = 1.0 # m, diameter of reactor tube
    ρcp = 1.0 # J/(m^3K), TODO realistic value (retune ΔHr, U, λ)

    # Inlet Dirichlet BCs
    y_in = (C = t->1.0, T = t->300.0)
    Tw = T_wall isa Function ? T_wall : t -> T_wall

    # Model
    react = Arrhenius(k0, Ea, ΔHr)
    model = PlugFlowModel(FIELDS, D, β, ρcp, react, U, d, Tw, y_in)

    # Matrices, shared pattern
    K = sparsity_pattern(grid, dm)
    M = copy(K)
    Jr = copy(K)

    # Assembly
    assemble_mass!(M, grid, geom, dm, model)
    assemble_transport!(K, ((fs, fg),), dm, model)
    f_in = zeros(ndof(dm))
    assemble_inlet_outlet!(K, f_in, dm, grid, model,
        (bfs.inlet, bfg_inlet), (bfs.outlet, bfg_outlet))
    f_wall = zeros(ndof(dm))
    assemble_wall!(K, f_wall, dm, model, geom)

    # Reaction: global source + local cell scratch (source, jac)
    nf = length(FIELDS)
    r = zeros(ndof(dm))
    re = zeros(nf)
    Jre = zeros(nf, nf)

    prob = ProblemCache(M, K, Jr, r, re, Jre, f_in, f_wall, grid, geom, dm, model)

    y0 = zeros(ndof(dm))
    for f in FIELDS
        @views y0[fielddof(dm, f)] .= inlet_mod(model, f, 0.0)
    end

    return prob, (fs, fg), y0
end

# CN forward solve, IC at the inlet state
function cn_forward(Twfun, tf::Float64, Δt::Float64; solver=shamanskii(),
    recon::Symbol=:vanalbada)
    prob, (fs, fg), y0 = setup_problem(T_wall=Twfun)
    N = round(Int, tf / Δt) + 1
    cache = CNCache(prob, Δt, N)
    for field in prob.dm.fields
        set_ic!(cache, field, inlet_mod(prob.model, field, 0.0))
    end

    # Deferred correction for 2nd order (inexact newton)
    sc = antidiff_sc(prob.model, view(y0, celldof(prob.dm, 1)))
    ad = recon === :upwind1 ? nothing : AntiDiffusion(prob.grid, prob.geom, fs, fg, sc)
    react! = function (y, WithJac)
        assemble_react!(prob, y, WithJac)
        isnothing(ad) ||
            assemble_antidiffusion!(prob, nothing, ad.st, ad.fs, ad.fg, y, ad.εf)
        return prob
    end

    cn_solve!(cache, react!,
        (b, t) -> assemble_boundary!(prob, b, t), solver) # newton!, chord!, shamanskii()
    return cache
end
