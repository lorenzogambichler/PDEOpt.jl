ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", "4") # set OpenMP threads
using LinearAlgebra
using SparseArrays
using PDEOpt
using Revise

includet("problem.jl") # setup_problem, cn_forward

# Piecewise-constant control
zoh(u::AbstractVector, Δt::Float64) = t -> u[clamp(floor(Int, t / Δt) + 1, 1, length(u))]

# Initial guess, largest constant Tw s.t. max T < Tmax - δ
function forward_guess(tf, ts, Δt, Ne; Δt_cn=0.0025, Tmax, Tw_min, Tw_max, δ=10.0,
    recon::Symbol=:vanalbada)

    function forward(u)
        uvec = u isa Real ? fill(float(u), Ne) : u
        cache = cn_forward(zoh(uvec, Δt), tf, Δt_cn; recon=recon, solver=newton!)
        Y = resample(cache.y, Δt_cn, ts)
        return Y, maximum(view(Y, fielddof(cache.prob.dm, :T), :))
    end

    g = bisect_const(forward, Ne; Tmax=Tmax, Tw_min=Tw_min, Tw_max=Tw_max, δ=δ)
    g.conv || @warn "initial guess bisection did not converge" g.Tpk g.Tw
    return g.Y, g.u
end

function main(; recon::Symbol=:vanalbada, save::Bool=false)
    # Params
    tf = 5.0
    Ne = 30 # finite elements
    s = 3 # radau IIA stages
    Δt = tf / Ne
    Δt_resim = 0.0025
    Tw_min, Tw_max = 300.0, 600.0
    Tmin, Tmax = 280.0, 400.0
    γ = 0.01
    resultsdir = joinpath(@__DIR__, "results", "opt")

    # Problem
    prob, _, y0 = setup_problem()

    # OCP
    ocp = PlugFlowOCP(prob, y0; Δt=Δt, Ne=Ne, s=s, Tw_min=Tw_min, Tw_max=Tw_max,
        Tmin=Tmin, Tmax=Tmax, γ=γ, recon=recon)

    # Initial guess
    Yguess, uguess = forward_guess(tf, stage_times(ocp.lay), Δt, Ne;
        Tmax=Tmax, Tw_min=Tw_min, Tw_max=Tw_max, recon=recon)
    x0 = pack(ocp.lay, Yguess, uguess)

    # Solve
    # J = mean(1 - X) ~ 0.03 -> obj_scaling_factor for O(1)
    opts = (print_level=5, max_iter=300, tol=1e-6, limited_memory_max_history=50,
        obj_scaling_factor=100.0)
    res = solve_ocp(ocp, x0; opts...)

    # Print res
    println("status: ", res.stats.status)
    println("X_C_mean: guess -> ", round(mean_conv(prob, ocp.lay, Yguess); digits=4),
        ", opt -> ", round(mean_conv(prob, ocp.lay, res.Y); digits=4))
    println("T_max (colloc): opt -> ", round(maximum(view(res.Y, fielddof(prob.dm, :T), :)); digits=2))

    # Re-sim with CN (transcription check)
    res_cn = cn_forward(zoh(vec(res.U), Δt), tf, Δt_resim; recon=recon, solver=newton!)
    Tpk_cn = maximum(view(res_cn.y, fielddof(res_cn.prob.dm, :T), :))
    Xbar_cn = mean_conv(prob, ocp.lay, resample(res_cn.y, Δt_resim, stage_times(ocp.lay)))
    println("re-sim (Δt_cn = ", Δt_resim, "):")
    println("  T_max = ", round(Tpk_cn; digits=2))
    println("  X_C_mean = ", round(Xbar_cn; digits=4))

    # Save res (write_vtk is 2D only for now)
    if save
        write_control(joinpath(resultsdir, "opt_control.csv"), vec(res.U), Δt, Ne)
        write_control(joinpath(resultsdir, "guess_control.csv"), uguess, Δt, Ne)
    end
    return res, ocp
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
