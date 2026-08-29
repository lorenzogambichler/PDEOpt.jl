const SPECIES = (:CH4, :CO, :CO2, :H2O, :H2, :N2)

function setup_problem(; nz=25, nr=3, L=5.0, R=0.01, Tw=650.0, ratio=8.0)
    grid = StructGrid2D(L, R, nz, nr; ratio=ratio)
    geom = Geometry2D(grid)

    (fs_axial, fs_radial) = interior_faces(grid)
    fg_axial = FaceGeometry(grid, geom, fs_axial)
    fg_radial = FaceGeometry(grid, geom, fs_radial)

    bfs = boundary_faces(grid)
    bfg_inlet = BoundaryFaceGeometry(grid, geom, bfs.inlet)
    bfg_outlet = BoundaryFaceGeometry(grid, geom, bfs.outlet)
    bfg_wall = BoundaryFaceGeometry(grid, geom, bfs.wall)

    fields = (SPECIES..., :T)
    dm = DofMap(ncells(grid), fields)

    vz = 1.0
    T0, p0, Rgas = 400.0, 5.0e5, 8.314
    Mα = (CH4=0.016043, CO=0.028010, CO2=0.044010, H2O=0.018015, H2=0.0020159, N2=0.0280134)
    xin = (CH4=1e-5, CO=1e-5, CO2=0.2, H2O=1e-5, H2=0.8, N2=1e-5)
    cmol = p0 / (Rgas * T0)
    ρin = NamedTuple{SPECIES}(ntuple(i -> xin[SPECIES[i]] * cmol * Mα[SPECIES[i]], length(SPECIES)))

    T_in(t) = 400.0
    g = (CH4=r->1.0, CO=r->1.0, CO2=r->1.0, H2O=r->1.0, H2=r->1.0, N2=r->1.0, T=r->1.0)
    y_in = (CH4=t->ρin.CH4, CO=t->ρin.CO, CO2=t->ρin.CO2, H2O=t->ρin.H2O,
        H2=t->ρin.H2, N2=t->ρin.N2, T=T_in)
    T_wall = Tw isa Function ? Tw : t -> Tw

    model = MethanationModel(fields; g=g, y_in=y_in, T_wall=T_wall,
        vz=vz, kw=120.0, Rrad=R)

    K = sparsity_pattern(grid, dm)
    M = copy(K)
    Jr = copy(K)

    f_in = zeros(ndof(dm))
    f_wall = zeros(ndof(dm))
    K_bc = copy(K)
    fill!(K_bc.nzval, 0.0)
    assemble_inlet_outlet!(K_bc, f_in, dm, geom, grid, model,
        (bfs.inlet, bfg_inlet), (bfs.outlet, bfg_outlet); fields=SPECIES)
    assemble_wall!(K_bc, f_wall, dm, model, bfs.wall, bfg_wall)

    nf = length(fields)
    r = zeros(ndof(dm))
    re = zeros(nf)
    Jre = zeros(nf, nf)
    prob = ProblemCache(M, K, Jr, r, re, Jre, f_in, f_wall, grid, geom, dm, model)

    props = MethanationProps(nspecies(model), ncells(grid))
    dirs = ((fs_axial, fg_axial), (fs_radial, fg_radial))
    ebnd = ((bfs.inlet, bfg_inlet), (bfs.outlet, bfg_outlet))
    sa = StateAssembly(prob, props, K_bc, dirs, ebnd)

    y0 = zeros(ndof(dm))
    for field in fields
        @views y0[fielddof(dm, field)] .= inlet_mod(model, field, 0.0)
    end
    sa(y0)

    return prob, sa, y0
end

# CN forward solve for given Tw(t)
function cn_forward(Twfun, tf::Float64, Δt_cn::Float64; nz=25, nr=3, ratio=8.0,
    hook=(y, prob) -> nothing, recon::Symbol=:vanalbada)
    prob, sa, y0 = setup_problem(; nz=nz, nr=nr, ratio=ratio, Tw=Twfun)
    N = round(Int, tf / Δt_cn) + 1
    cache = CNCache(prob, Δt_cn, N)
    for field in prob.dm.fields
        set_ic!(cache, field, inlet_mod(prob.model, field, 0.0))
    end

    # Deferred correction for 2nd order (inexat newton)
    ad = recon === :upwind1 ? nothing : AntiDiffusion(sa, y0)
    react! = function (y, WithJac)
        assemble_react!(prob, y, WithJac)
        isnothing(ad) ||
            assemble_antidiffusion!(prob, sa.props, ad.st, ad.fs, ad.fg, y, ad.εf)
        return prob
    end

    cn_solve!(cache, react!,
        (b, t) -> assemble_boundary!(prob, b, t),
        shamanskii(refactor_every=20);
        reassemble! = y -> (hook(y, prob); sa(y)))
    return cache, sa
end
