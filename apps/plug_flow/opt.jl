ENV["OMP_NUM_THREADS"] = get(ENV, "OMP_NUM_THREADS", "4") # set OpenMP threads
using LinearAlgebra
using SparseArrays
using PDEOpt
using Revise

includet("problem.jl") # setup_problem, cn_forward
includet("../../src/optimization/plug_flow/jump_model.jl")

function main(; tf=2.0, Δt=0.005, Tw_min=200.0, Tw_max=500.0, Tmax=500.0, γ=10.0)
    N = round(Int, tf / Δt) + 1
    resultsdir = joinpath(@__DIR__, "results", "opt")

    # CN forward solve -> initial guess
    Tw = t -> Tw_min
    @time cn = cn_forward(Tw, tf, Δt)
    prob = cn.prob

    # NLP
    opt = OptCache(γ, Tmax, Tw_min, Tw_max, cn.y[:, 1], cn.y)
    model, vars = build_nlp(prob, prob.model, opt, Δt, N)
    res = solve_nlp!(model, vars)

    # Save res
    write_vtk(joinpath(resultsdir, "opt_state"), prob, res.y, Δt, N)
    write_vtk(joinpath(resultsdir, "guess_state"), cn)
    write_control(joinpath(resultsdir, "control.csv"), res.u, Δt, N)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
