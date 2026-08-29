using LinearAlgebra
using SparseArrays
using PDEOpt
using Printf
using Revise

includet("problem.jl")

# CN forward simulation at given wall temperature
function main(; tf=1000.0, Δt=0.5, nz=150, nr=7, ratio=1.0,
    recon::Symbol=:vanalbada)

    resultsdir = joinpath(@__DIR__, "results", "forward")

    Tw = t -> 650.0 # constant Tw

    @time cache, _ = cn_forward(Tw, tf, Δt; nz=nz, nr=nr, ratio=ratio, recon=recon)

    write_vtk(joinpath(resultsdir, "state"), cache)
    @printf("%9s %7s %9s\n", "f-evals", "jacs", "steps")
    @printf("%9d %7d %9d\n", cache.stats.nf, cache.stats.nfact, cache.N - 1)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
