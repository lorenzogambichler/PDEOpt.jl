ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", "4")
using LinearAlgebra
using SparseArrays
using PDEOpt

includet("../../src/optimization/methanation_adnlp.jl")

const SPECIES = (:CH4, :CO, :CO2, :H2O, :H2, :N2)

function setup_problem(; nz=25, nr=3, L=5.0, R=0.01, Tw=650.0, vz=1.0)
    grid = StructGrid2D(L, R, nz, nr)
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

# Piecewise-constant control
zoh(u::AbstractVector, Δt::Float64) =
    t -> u[clamp(floor(Int, t / Δt) + 1, 1, length(u))]

# CN forward solve for given Tw(t)
function cn_forward(Twfun, tf::Float64, Δt_cn::Float64; nz=25, nr=3, hook=(y, prob) -> nothing,
    recon::Symbol=:vanalbada)
    prob, sa, y0 = setup_problem(; nz=nz, nr=nr, Tw=Twfun)
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

# Initial guess
function forward_guess(tf, ts, Δt, Ne; Δt_cn=0.25, Tmax=750.0, Tw_min=300.0, Tw_max=650.0,
    δ=10.0, recon::Symbol=:vanalbada)

    function forward(u)
        uvec = u isa Real ? fill(float(u), Ne) : u # vector (bisect_shape), scalar (bisect_const)
        cache, _ = cn_forward(zoh(uvec, Δt), tf, Δt_cn; recon=recon)
        Z = resample(cache.y, Δt_cn, ts)
        return Z, maximum(view(Z, fielddof(cache.prob.dm, :T), :))
    end

    #g = bisect_shape(forward, Ne; Tmax=Tmax, Tw_min=Tw_min, Tw_max=Tw_max, δ=δ)
    g = bisect_const(forward, Ne; Tmax=Tmax, Tw_min=Tw_min, Tw_max=Tw_max, δ=δ)
    g.conv || @warn "initial guess bisection did not converge" g.Tpk g.Tw
    return g.Z, g.u
end

function main(; recon::Symbol=:vanalbada)
    # Params
    tf = 750.0
    Ne = 30 # finite elements
    s = 3 # radau IIA stages
    Δt = tf / Ne
    Δt_resim = 0.1
    Δt_plot = 2.5 # for VTK
    Tw_min, Tw_max, Tmax = 300.0, 650.0, 750.0
    γ = 0.01
    nz_resim, nr_resim = 75, 5

    # Problem
    prob, sa, y0 = setup_problem()

    # OCP
    co2_in = co2_inflow(sa) # inlet flowrate CO2
    ocp = MethanationOCP(sa, RadauIIA(s), Δt, Ne, y0; Tw_min=Tw_min, Tw_max=Tw_max,
        Tmax=Tmax, γ=γ, co2_in=co2_in, recon=recon)

    # Initial guesses
    Zguess, uguess = forward_guess(tf, timegrid(ocp), Δt, Ne;
        Tmax=Tmax, Tw_min=Tw_min, Tw_max=Tw_max, recon=recon)
    x0 = vcat(vec(Zguess), uguess)

    # Check ϵ
    Td = fielddof(prob.dm, :T)
    Tg, idx = findmax(view(Zguess, Td, :))
    zhot = prob.geom.zc[cellij(prob.grid, idx[1])[1]]
    spmin = minimum(view(Zguess, setdiff(1:ndof(prob.dm), Td), :))
    println("recon = :", recon)
    println("init guess: T_max = ", round(Tg; digits=1), " K, z_max = ",
        round(zhot; digits=3), " m, ρ_min = ", round(spmin; sigdigits=3))
    spmin < 0 && @warn "negative species density in CN guess -> limiter ε too large" spmin

    # Solve
    res = solve_ocp(ocp, x0; print_level=5, max_iter=200, tol=1e-5, exact_hessian=false, show_time=true, 
        limited_memory_max_history=75)

    # Print res
    Xguess = outlet_conv(ocp, Zguess)
    Xopt = outlet_conv(ocp, res.Z)
    println("status: ", res.stats.status)
    println("nvars = ", nvars(ocp), ", ncons = ", ncons(ocp))
    println("X_CO2(tf):  guess -> ", round(Xguess[end]; digits=4),
        ", opt -> ", round(Xopt[end]; digits=4))
    println("mean X_CO2: guess -> ", round(mean_conv(ocp, Zguess); digits=4),
        ", opt -> ", round(mean_conv(ocp, res.Z); digits=4))
    println("Tw: ", round.(res.u; digits=1))

    # Re-sim with CN, validate constraints on fine grid
    res_cn, sa_cn = cn_forward(zoh(res.u, Δt), tf, Δt_resim; nz=nz_resim, nr=nr_resim, recon=recon)
    guess_cn, _ = cn_forward(zoh(uguess, Δt), tf, Δt_resim; nz=nz_resim, nr=nr_resim, recon=recon)

    # Compare (re-sim on fine grid -> own dm/sa)
    Td_cn = fielddof(res_cn.prob.dm, :T)
    Tpk_cn = maximum(view(res_cn.y, Td_cn, :))
    Xbar_cn = mean_conv(ocp, outlet_conv(sa_cn, resample(res_cn.y, Δt_resim, timegrid(ocp))))
    println("grid: NLP ", prob.grid.nz, " by ", prob.grid.nr,
        " -> re-sim ", res_cn.prob.grid.nz, " by ", res_cn.prob.grid.nr)
    println("T_max: (Δt_cn = ", Δt_resim, ") -> ", round(Tpk_cn; digits=2),
        " K, NLP -> ", round(maximum(view(res.Z, Td, :)); digits=2), " K (constraint: ", Tmax, " K)")
    println("mean X_CO2: (Δt_cn = ", Δt_resim, ") -> ", round(Xbar_cn; digits=4),
        ", NLP -> ", round(mean_conv(ocp, res.Z); digits=4))

    # Save res
    ts_plot = collect(0.0:Δt_plot:tf)
    resultsdir = joinpath(@__DIR__, "results/Methanation/")
    write_vtk(joinpath(resultsdir, "opt_state"), res_cn.prob, dense_output(res_cn, ts_plot), ts_plot)
    write_vtk(joinpath(resultsdir, "guess_state"), guess_cn.prob, dense_output(guess_cn, ts_plot), ts_plot)
    write_control(joinpath(resultsdir, "opt_control.csv"), res.u, Δt, Ne)
    write_control(joinpath(resultsdir, "guess_control.csv"), uguess, Δt, Ne)
end
