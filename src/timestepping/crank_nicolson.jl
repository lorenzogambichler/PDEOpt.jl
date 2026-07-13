module CrankNicolson

using LinearAlgebra
using SparseArrays

export cn_solve!, CNCache

mutable struct CNStats
    nf::Int # function evals (residual calls)
    nfact::Int # factorizations
end
CNStats() = CNStats(0, 0)
reset!(s::CNStats) = (s.nf = 0; s.nfact = 0; s)

# Crank–Nicolson state for given ProblemCache
struct CNCache{TP,Tfact}
    prob::TP # ProblemCache
    A::SparseMatrixCSC{Float64,Int} # M + Δt/2·K
    B::SparseMatrixCSC{Float64,Int} # M − Δt/2·K
    J::SparseMatrixCSC{Float64,Int} # Newton Jac A − Δt/2·Jr
    fact::Tfact # lu(J)
    y::Matrix{Float64} 
    temp1::Vector{Float64}
    temp2::Vector{Float64}
    temp3::Vector{Float64}
    b::Vector{Float64} # boundary forcing vector
    Δt::Float64
    N::Int
    stats::CNStats
end

function CNCache(prob, Δt::Float64, N::Int)
    stats = CNStats()
    A = copy(prob.K)
    B = copy(prob.K)
    @. A.nzval = prob.M.nzval + 0.5 * Δt * prob.K.nzval # implicit
    @. B.nzval = prob.M.nzval - 0.5 * Δt * prob.K.nzval # explicit
    J = copy(A) # Newton Jac, same pattern as A and prob.Jr
    fact = lu(J)
    n = size(A, 1)
    y = zeros(n, N)
    return CNCache(prob, A, B, J, fact, y, zeros(n), zeros(n), zeros(n), zeros(n), Δt, N, stats)
end

# Crank–Nicolson for  M ẏ + K y = r(y) + b(t)
# react!(x) assembles reaction (prob.r, prob.Jr)
# boundary!(b, t) fills boundary forcing vector b(t) at t 
# solve!, see solvers/nonlinear.jl
function cn_solve!(cache::CNCache, react!, boundary!, solve!; maxiter::Int=30, tol::Float64=1e-8)
    reset!(cache.stats) # reset stats (for consecutive solves)
    A, B, Δt, b = cache.A, cache.B, cache.Δt, cache.b
    prob = cache.prob
    h, F, s = cache.temp1, cache.temp2, cache.temp3

    # F(ykp1) = A⋅ykp1 - (Δt/2)⋅r(ykp1) + h(yk)
    # h(yk) = -B⋅yk - (Δt/2)⋅(r(yk) + bkp1 + bk)
    # A = M + Δt/2⋅K
    # B = M - Δt/2⋅K

    function residual!(F, ykp1) 
        react!(ykp1, Val(false)) # r(ykp1)
        mul!(F, A, ykp1)
        @. F += -0.5 * Δt * prob.r + h
        cache.stats.nf += 1
    end
    function jacobian!(ykp1)
        react!(ykp1, Val(true)) # (r(ykp1), Jr(ykp1))
        @. cache.J.nzval = A.nzval - 0.5 * Δt * prob.Jr.nzval
        lu!(cache.fact, cache.J)
        cache.stats.nfact += 1
    end
    linsolve!(s, F) = ldiv!(s, cache.fact, F)

    for k = 1:cache.N-1
        yk = view(cache.y, :, k)
        ykp1 = view(cache.y, :, k + 1)

        # h(yk) = -B·yk - (Δt/2)·r(yk) - (Δt/2)·(bk + bkp1) 
        react!(yk, Val(false)) 
        mul!(h, B, yk)
        @. h = -h - 0.5 * Δt * prob.r
        boundary!(b, (k - 1) * Δt) 
        @. h -= 0.5 * Δt * b # -(Δt/2)⋅bk
        boundary!(b, k * Δt) 
        @. h -= 0.5 * Δt * b # -(Δt/2)⋅bkp1

        copyto!(ykp1, yk) # initial guess
        solve!(ykp1, residual!, jacobian!, linsolve!, F, s; maxiter=maxiter, tol=tol)
    end
    return cache.y
end

end
