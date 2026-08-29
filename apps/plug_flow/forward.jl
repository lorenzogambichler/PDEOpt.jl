using LinearAlgebra
using SparseArrays
using PDEOpt
using Revise

includet("problem.jl") # setup_problem, cn_forward

function main(; tf=3.0, Δt=0.01)
    resultsdir = joinpath(@__DIR__, "results", "forward")

    t_ramp = 1.0
    Tw = t-> 100.0 + (500.0 - 100.0) * clamp(t / t_ramp, 0.0, 1.0)
    @time cache = cn_forward(Tw, tf, Δt)

    write_vtk(joinpath(resultsdir, "state"), cache)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
