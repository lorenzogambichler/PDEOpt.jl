using LinearAlgebra, SparseArrays, PDEOpt, Random

const APPDIR = "/home/lorenzog/Documents/julia/PDEOpt_julia_DtO/apps/optimal_control"

includet(p) = Base.include(Main, normpath(joinpath(APPDIR, p)))
include(joinpath(APPDIR, "MethanationOpt.jl"))

mb(x) = round(x / 2^20, digits=1)
mark(tag) = println(rpad(tag, 28), " rss = ", mb(Sys.maxrss()), " MB   julia-heap = ",
                    mb(Base.gc_live_bytes()), " MB")

mark("baseline")
prob, sa, y0 = setup_problem()
ocp = MethanationOCP(sa, RadauIIA(3), 25.0, 30, y0;
    Tw_min=300.0, Tw_max=650.0, Tmax=750.0, γ=0.01, co2_in=co2_inflow(sa))
mark("after ocp")

t = @elapsed H = OCPHessian(ocp)
mark("after OCPHessian")
println("  hess_structure: $(round(t,digits=1)) s,  nnzh = $(nnzh(H))  (nvar = $(nvars(ocp)))")

nv, nc = nvars(ocp), ncons(ocp)
Random.seed!(1)
z = 0.2 .+ 0.6 .* rand(nv); λ = randn(nc)
vals = zeros(nnzh(H))
hess_values!(vals, H, z, λ)
t = @elapsed hess_values!(vals, H, z, λ)
mark("after hess_values!")
println("  hess_values!: $(round(t,digits=2)) s/call,  ",
        count(!iszero, vals), " nonzero of ", length(vals))

# fill-in of the KKT system, H block vs diagonal (L-BFGS) block
Hs = sparse(H.rows, H.cols, ones(nnzh(H)), nv, nv)
Hs = Hs + Hs' - spdiagm(0 => diag(Hs))
patJ = jac_pattern(ocp, scale_z(ocp, vcat(vec(zeros(ocp.n, ncols(ocp))), zeros(30))))
mark("after jac_pattern")
println("  nnz(H sym) = ", nnz(Hs), ",  nnz(J) = ", nnz(patJ))

using AMD
for (tag, Hblk) in (("exact H", Hs), ("L-BFGS (diag)", spdiagm(0 => ones(nv))))
    K = [Hblk sparse(patJ') ; patJ spdiagm(0 => ones(nc))]
    p = AMD.amd(K)
    F = cholesky(Symmetric(Float64.(K[p, p]) + 1e6I, :L); check=false)
    println(rpad("  KKT $tag:", 22), " nnz(K) = ", rpad(nnz(K), 10),
            " nnz(L) = ", nnz(sparse(F.L)), "  -> L ≈ ", mb(nnz(sparse(F.L)) * 12), " MB")
end
mark("done")