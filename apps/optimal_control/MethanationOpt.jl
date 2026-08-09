ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", "4")
using LinearAlgebra
using SparseArrays
using WriteVTK
using PDEOpt

includet("../../src/optimization/methanation_adnlp.jl")

const SPECIES = (:CH4, :CO, :CO2, :H2O, :H2, :N2)

function write_state(path::String, prob::ProblemCache, y::AbstractMatrix, Δt::Float64, N::Int)
    grid = prob.grid
    nz, nr = grid.nz, grid.nr
    mkpath(dirname(path))
    paraview_collection(path) do pvd
        for n = 1:N
            t = (n - 1) * Δt
            vtk_grid("$(path)_$(n)", grid.z, grid.r) do vtk
                for field in prob.dm.fields
                    fd = fielddof(prob.dm, field)
                    # cell-index order
                    vtk[string(field), VTKCellData()] = permutedims(reshape(y[fd, n], nr, nz))
                end
                pvd[t] = vtk
            end
        end
    end
end

function write_control(path::String, u::AbstractVector, Δt::Float64, N::Int)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "t,Tw")
        for n = 1:N
            println(io, (n - 1) * Δt, ",", u[n])
        end
    end
end

function setup_problem(; nz=25, nr=4, L=5.0, R=0.01, Tw=650.0, vz=1.0)
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

# Linear interpolation of Y (Δt grid -> ts nodes)
function resample(Y::AbstractMatrix, Δt::Float64, ts::AbstractVector)
    n, N = size(Y)
    Z = Matrix{Float64}(undef, n, length(ts))
    for (m, t) in enumerate(ts)
        x = clamp(t / Δt, 0.0, float(N - 1))
        k = min(floor(Int, x), N - 2)
        θ = x - k
        @views @. Z[:, m] = (1 - θ) * Y[:, k+1] + θ * Y[:, k+2]
    end
    return Z
end

# CN forward solve for given Tw(t)
function cn_forward(Twfun, tf::Float64, Δt_cn::Float64; hook=(y, prob) -> nothing,
    recon::Symbol=:vanalbada)
    prob, sa, y0 = setup_problem(; Tw=Twfun)
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
    return cache
end

# Feasible init guess (CN), resample onto collocation grid
# 1) feedback law generates continuous Tw(t), extract piecewise constant u
# 2) re-simulate with piecewise constant u
function forward_guess(tf, ts, Δt, Ne; Δt_cn=0.25, Tw_min=300.0, Tw_max=650.0,
    Tset=680.0, Kp=10.0, τ=10.0, recon::Symbol=:vanalbada)
    peak = Ref(0.0)
    Tw = Ref(Tw_max)
    tprev = Ref(-1.0)
    log = Tuple{Float64,Float64}[]
    function Twctl(t)
        if t > tprev[]
            raw = clamp(Tw_max - Kp * max(0.0, peak[] - Tset), Tw_min, Tw_max)
            a = min(1.0, (t - max(tprev[], 0.0)) / τ)
            Tw[] += a * (raw - Tw[])
            tprev[] = t
            push!(log, (t, Tw[]))
        end
        return Tw[]
    end

    # 1) Continuous Tw(t)
    cn_forward(Twctl, tf, Δt_cn; recon=recon,
        hook=(y, prob) -> (peak[] = maximum(@view y[fielddof(prob.dm, :T)])))

    u = fill(Tw_max, Ne)
    for k in 1:Ne
        v = [w for (t, w) in log if (k - 1) * Δt <= t < k * Δt]
        isempty(v) && error("no Tw samples in element $k; need Δt_cn ($Δt_cn) < Δt ($Δt)")
        u[k] = sum(v) / length(v)
    end

    # 2) Re-simulate
    uf(t) = u[clamp(floor(Int, t / Δt) + 1, 1, Ne)]
    return resample(cn_forward(uf, tf, Δt_cn; recon=recon).y, Δt_cn, ts), u
end

function main(; recon::Symbol=:vanalbada)
    # Params
    tf = 750.0
    Ne = 30 # finite elements
    s = 3 # radau IIA stages
    Δt = tf / Ne # ideally 25s
    Tw_nom = 600.0
    Tw_min, Tw_max, Tmax = 300.0, 650.0, 750.0
    γ = 0.01

    # Problem
    prob, sa, y0 = setup_problem(; Tw=Tw_nom)

    # OCP
    co2_in = co2_inflow(sa) # inlet flowrate CO2
    ocp = MethanationOCP(sa, RadauIIA(s), Δt, Ne, y0; Tw_min=Tw_min, Tw_max=Tw_max,
        Tmax=Tmax, γ=γ, co2_in=co2_in, recon=recon)

    # Initial guesses
    Zguess, uguess = forward_guess(tf, timegrid(ocp), Δt, Ne;
        Tw_min=Tw_min, Tw_max=Tw_max, recon=recon)
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
    Xguess = co2_conv(ocp, Zguess)
    Xopt = co2_conv(ocp, res.Z)
    println("status: ", res.stats.status)
    println("nvars = ", nvars(ocp), ", ncons = ", ncons(ocp))
    println("X_CO2(tf):  guess -> ", round(Xguess[end]; digits=4),
        ", opt -> ", round(Xopt[end]; digits=4))
    println("mean X_CO2: guess -> ", round(mean_co2_conv(ocp, Zguess); digits=4),
        ", opt -> ", round(mean_co2_conv(ocp, res.Z); digits=4))
    println("Tw: ", round.(res.u; digits=1))

    # Save res
    resultsdir = joinpath(@__DIR__, "results/Methanation/")
    write_state(joinpath(resultsdir, "opt_state"), prob, res.Z, Δt, Ne+1)
    write_state(joinpath(resultsdir, "guess_state"), prob, Zguess, Δt, Ne+1)
    write_control(joinpath(resultsdir, "opt_control.csv"), res.u, Δt, Ne)
    write_control(joinpath(resultsdir, "guess_control.csv"), uguess, Δt, Ne)
end
