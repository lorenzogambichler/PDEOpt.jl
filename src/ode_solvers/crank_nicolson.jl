module CrankNicolson

using LinearAlgebra

export cn_solve!

# Crank–Nicolson method for  M ẏ + K y = r(y) + τ(t)·f_in
# Either Newton or Quasi-Newton for solve!
function cn_solve!(cache, react!, solve!; maxiter::Int=30, tol::Float64=1e-8)
    A, R, Δt = cache.A, cache.R, cache.Δt
    b, F, δ = cache.temp1, cache.temp2, cache.temp3

    # jacobian! uses cache.Jr argument, but is already assembled by residual!
    residual!(Fo, x) = (react!(x); mul!(Fo, A, x); @. Fo += -0.5 * Δt * cache.fr + b)
    jacobian!(_) = (@. cache.J.nzval = A.nzval - 0.5 * Δt * cache.Jr.nzval; lu!(cache.fact, cache.J))
    linsolve!(d, Fi) = ldiv!(d, cache.fact, Fi)

    for n = 1:cache.N-1
        yk, ykp1 = view(cache.y, :, n), view(cache.y, :, n + 1)
        τavg = 0.5 * (cache.τ((n - 1) * Δt) + cache.τ(n * Δt)) # averaged over [tk, tkp1]

        # b = -R·yk - (Δt/2)·r(yk) - Δt·τavg·f_in  (constant over step)
        react!(yk)
        mul!(b, R, yk)
        @. b = -b - 0.5 * Δt * cache.fr - Δt * τavg * cache.f_in

        copyto!(ykp1, yk) # initial guess
        solve!(ykp1, residual!, jacobian!, linsolve!, F, δ; maxiter=maxiter, tol=tol) 
    end
    return cache.y
end

end
