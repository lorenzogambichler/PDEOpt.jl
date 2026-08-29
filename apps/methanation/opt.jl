ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", "4") # set OpenMP threads
using LinearAlgebra
using SparseArrays
using PDEOpt
using Revise

includet("problem.jl") # setup_problem, cn_forward
includet("../../src/optimization/methanation/methanation_ocp.jl")

# Piecewise-constant control
zoh(u::AbstractVector, Δt::Float64) =  t -> u[clamp(floor(Int, t / Δt) + 1, 1, length(u))]

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

# solve_ocp with memory sampler, needs julia -t 2 (main, sampler)
function solve_profiled(ocp, x0, logpath::String; full::Bool=true, kwargs...)
    mkpath(dirname(logpath))

    res, tr = memtrace() do
        solve_ocp(ocp, x0; output_file=logpath, file_print_level=5, kwargs...)
    end

    println()
    try
        print_timing(read_ipopt_timing(logpath); full=full)
    catch e
        @warn "could not parse the ipopt timing table" e # don't lose res
    end
    display(tr)
    return res
end

function main(; recon::Symbol=:vanalbada, profile::Bool=false, solver::Symbol=:IPOPT)
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
    resultsdir = joinpath(@__DIR__, "results", "opt")

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

    Td = fielddof(prob.dm, :T)
    Tg, idx = findmax(view(Zguess, Td, :))
    zhot = prob.geom.zc[cellij(prob.grid, idx[1])[1]]
    spmin = minimum(view(Zguess, setdiff(1:ndof(prob.dm), Td), :))
    println("recon = :", recon)
    println("init guess: T_max = ", round(Tg; digits=1), " K, z_max = ",
        round(zhot; digits=3), " m, ρ_min = ", round(spmin; sigdigits=3))
    spmin < 0 && @warn "negative species density in CN guess -> limiter ε too large" spmin

    # Solve
    if solver == :IPOPT
        opts = (print_level=5, max_iter=200, tol=1e-5, exact_hessian=false,
            show_time=true, limited_memory_max_history=75)
        res = profile ? solve_profiled(ocp, x0, joinpath(resultsdir, "ipopt/ipopt.log"); opts...) :
            solve_ocp(ocp, x0; opts...) # TODO add solver argument -> dispatch?
    elseif solver == :MadNLP
        # TODO
        opts = ()
        res = profile ? (@warn "profiling not implemented for MadNLP, running standard solve" profile) : 
            solve_ocp(ocp, x0, opts...) # TODO solver argument
        @error "MadNLP is currently not implemented, use solver=:IPOPT" solver
    else 
        @warn "$(solver) is not implemented, defaulting to IPOPT" solver
        opts = (print_level=5, max_iter=200, tol=1e-5, exact_hessian=false,
            show_time=true, limited_memory_max_history=75)
        res = profile ? solve_profiled(ocp, x0, joinpath(resultsdir, "ipopt/ipopt.log"); opts...) :
        solve_ocp(ocp, x0; opts...)
    end

    # Print res
    Xguess = outlet_conv(ocp, Zguess)
    Xopt = outlet_conv(ocp, res.Z)
    println("status: ", res.stats.status)
    println("X_CO2_final: guess -> ", round(Xguess[end]; digits=4), ", opt -> ", round(Xopt[end]; digits=4))
    println("X_CO2_mean:  guess -> ", round(mean_conv(ocp, Zguess); digits=4), ", opt -> ", round(mean_conv(ocp, res.Z); digits=4))

    # Re-sim with CN
    res_cn_coarse, sa_cn_coarse = cn_forward(zoh(res.u, Δt), tf, Δt_resim; recon=recon)
    res_cn, sa_cn = cn_forward(zoh(res.u, Δt), tf, Δt_resim; nz=nz_resim, nr=nr_resim, recon=recon)
    guess_cn_coarse, _ = cn_forward(zoh(uguess, Δt), tf, Δt_resim; recon=recon)
    guess_cn, _ = cn_forward(zoh(uguess, Δt), tf, Δt_resim; nz=nz_resim, nr=nr_resim, recon=recon)

    # Transcription check (Δt -> Δt_resim)
    Td_cc = fielddof(res_cn_coarse.prob.dm, :T)
    Tpk_cc = maximum(view(res_cn_coarse.y, Td_cc, :))
    Xbar_cc = mean_conv(ocp, outlet_conv(sa_cn_coarse,
        resample(res_cn_coarse.y, Δt_resim, timegrid(ocp))))
    println("re-sim on coarse grid (", res_cn_coarse.prob.grid.nz, " by ", res_cn_coarse.prob.grid.nr, ", Δt_cn = ", Δt_resim, "):")
    println("  T_max = ", round(Tpk_cc; digits=2))
    println("  X_CO2_mean = ", round(Xbar_cc; digits=4))

    # Feasibility check (coarse -> fine grid, own dm/sa)
    Td_cn = fielddof(res_cn.prob.dm, :T)
    Tpk_cn = maximum(view(res_cn.y, Td_cn, :))
    Xbar_cn = mean_conv(ocp, outlet_conv(sa_cn, resample(res_cn.y, Δt_resim, timegrid(ocp))))
    println("re-sim on fine grid (", res_cn.prob.grid.nz, " by ", res_cn.prob.grid.nr, ", Δt_cn = ", Δt_resim, "):")
    println("  T_max = ", round(Tpk_cn; digits=2))
    println("  X_CO2_mean = ", round(Xbar_cn; digits=4))

    # Save res
    ts_plot = collect(0.0:Δt_plot:tf)
    write_vtk(joinpath(resultsdir, "opt_coarse_state"), res_cn_coarse.prob, dense_output(res_cn_coarse, ts_plot), ts_plot)
    write_vtk(joinpath(resultsdir, "guess_coarse_state"), guess_cn_coarse.prob, dense_output(guess_cn_coarse, ts_plot), ts_plot)
    write_vtk(joinpath(resultsdir, "opt_fine_state"), res_cn.prob, dense_output(res_cn, ts_plot), ts_plot)
    write_vtk(joinpath(resultsdir, "guess_fine_state"), guess_cn.prob, dense_output(guess_cn, ts_plot), ts_plot)
    write_control(joinpath(resultsdir, "opt_control.csv"), res.u, Δt, Ne)
    write_control(joinpath(resultsdir, "guess_control.csv"), uguess, Δt, Ne)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end