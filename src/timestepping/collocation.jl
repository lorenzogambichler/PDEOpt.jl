module Collocation

# Radau IIA (OCFE)

using LinearAlgebra

export RadauIIA, nstages, stage_time, quadrature_weights

struct RadauIIA
    s::Int
    c::Vector{Float64} # Abscissae
    A::Matrix{Float64} # Butcher matrix
    b::Vector{Float64} # A[s,:]
    D::Matrix{Float64} # inv(A)
    d0::Vector{Float64} # ∑_j d_ij - d0_i = 0
end

function radau_points(s::Int)
    if s == 1 
        return [1.0]
    elseif s == 2
        return [1 / 3, 1.0]
    elseif s == 3
        return [(4 - sqrt(6)) / 10, (4 + sqrt(6)) / 10, 1.0]
    end
end

function RadauIIA(s::Int=3)
    c = radau_points(s)
    V = [c[j]^(k - 1) for j in 1:s, k in 1:s]
    W = [c[i]^k / k for i in 1:s, k in 1:s]
    A = W / V

    # Check consistency
    @assert isapprox(vec(sum(A, dims=2)), c; atol=1e-12) "Radau IIA($s): ∑_j a_ij ≠ c_i"
    @assert c[s] == 1.0 "Radau IIA($s): not stiffly accurate"

    D = inv(A)
    return RadauIIA(s, c, A, A[s, :], D, D * ones(s))
end

nstages(tab::RadauIIA) = tab.s
quadrature_weights(tab::RadauIIA) = tab.b

# Time of stage i in element k
stage_time(tab::RadauIIA, k::Int, i::Int, Δt::Real) = (k - 1 + tab.c[i]) * Δt

end
