module Collocation

# Orthogonal collocation on finite elements — Radau IIA
#
# A collocation method with abscissae c ∈ [0,1]^s has a Butcher matrix A fixed by
# requiring the stage conditions to be exact for polynomials of degree < s:
#
#   ∑_j a_ij c_j^(k-1) = c_i^k / k,   k = 1..s      ⟺   A V = W
#
# with V[j,k] = c_j^(k-1) and W[i,k] = c_i^k / k. Radau IIA takes c to be the roots
# of the right Radau polynomial, giving classical order 2s-1, stage order s,
# L-stability, and c_s = 1 (stiff accuracy) so that b = A[s,:] and the last stage
# is the step endpoint.
#
# D = A^(-1) is the collocation differentiation matrix: if p is the degree-s
# polynomial with p(t_{k-1}) = y_{k-1} and p(t_{k-1} + c_i Δt) = Y^i, then
#
#   Δt p'(t_{k-1} + c_i Δt) = ∑_j d_ij (Y^j - y_{k-1}).
#
# The D form is the one to use when the mass matrix is state dependent: it states
# M(y) ẏ = f(y,t) pointwise at each collocation node, so M and f are evaluated at
# the same argument, and cross-stage coupling enters only through M.

using LinearAlgebra

export RadauIIA, nstages, stage_time, quadrature_weights

struct RadauIIA
    s::Int
    c::Vector{Float64}  # abscissae on [0,1], c[s] == 1
    A::Matrix{Float64}  # Butcher matrix
    b::Vector{Float64}  # quadrature weights == A[s,:], order 2s-1
    D::Matrix{Float64}  # A^(-1), differentiation matrix
    d0::Vector{Float64} # D*1, row sums (so ∑_j d_ij - d0_i = 0)
end

# Roots of the right Radau polynomial on [0,1]
function radau_points(s::Int)
    s == 1 && return [1.0]
    s == 2 && return [1 / 3, 1.0]
    s == 3 && return [(4 - sqrt(6)) / 10, (4 + sqrt(6)) / 10, 1.0]
    throw(ArgumentError("Radau IIA abscissae tabulated for s ∈ 1:3, got s = $s"))
end

function RadauIIA(s::Int=3)
    c = radau_points(s)
    V = [c[j]^(k - 1) for j in 1:s, k in 1:s]
    W = [c[i]^k / k for i in 1:s, k in 1:s]
    A = W / V # A·V = W

    # Collocation consistency: row sums of A are the abscissae
    @assert isapprox(vec(sum(A, dims=2)), c; atol=1e-12) "Radau IIA($s): ∑_j a_ij ≠ c_i"
    @assert c[s] == 1.0 "Radau IIA($s): not stiffly accurate"

    D = inv(A)
    return RadauIIA(s, c, A, A[s, :], D, D * ones(s))
end

nstages(tab::RadauIIA) = tab.s
quadrature_weights(tab::RadauIIA) = tab.b

# Time of stage i in element k (elements numbered from 1, t = 0 at the left of element 1)
stage_time(tab::RadauIIA, k::Int, i::Int, Δt::Real) = (k - 1 + tab.c[i]) * Δt

end
