module Collocation

# Radau IIA (OCFE)

using LinearAlgebra
import ..dense_output

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

# Lagrange basis at τ for the s+1 local nodes
function _lagrange!(w::Vector{Float64}, nodes::Vector{Float64}, τ::Float64)
    m = length(nodes)
    @inbounds for i in 1:m
        p = 1.0
        for j in 1:m
            j == i && continue
            p *= (τ - nodes[j]) / (nodes[i] - nodes[j])
        end
        w[i] = p
    end
    return w
end

# Collocation dense output
function dense_output(tab::RadauIIA, Δt::Float64, Z::AbstractMatrix, ts::AbstractVector)
    s = tab.s
    n, ncol = size(Z)
    Ne, rest = divrem(ncol - 1, s)
    (rest == 0 && Ne >= 1) ||
        error("Z has $ncol columns, expected 1 + s*Ne for s = $s")
    nodes = vcat(0.0, tab.c)
    w = Vector{Float64}(undef, s + 1)
    Y = Matrix{Float64}(undef, n, length(ts))
    for (m, t) in enumerate(ts)
        x = clamp(t / Δt, 0.0, float(Ne))
        k = min(floor(Int, x), Ne - 1) # 0-based element
        _lagrange!(w, nodes, x - k)
        col0 = 1 + k * s # left endpoint of element
        @views @. Y[:, m] = w[1] * Z[:, col0]
        for i in 1:s
            @views @. Y[:, m] += w[i+1] * Z[:, col0+i]
        end
    end
    return Y
end

end
