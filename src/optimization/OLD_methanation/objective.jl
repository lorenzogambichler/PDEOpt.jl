# min mean CO2 outlet fraction over [0,tf], equiv to max mean CO2 conversion
# Radau quad weights b, ∑_i b_i = 1
function ocp_obj(ocp::MethanationOCP, z)
    n, s, Ne = ocp.n, ocp.s, ocp.Ne
    nz = n * ncols(ocp)
    Z = reshape(view(z, 1:nz), n, ncols(ocp))
    u = view(z, nz+1:nz+Ne)

    J_conv = zero(eltype(z))
    for k in 1:Ne, i in 1:s
        col = stagecol(ocp, k, i)
        co2_out = zero(eltype(z))
        for j in eachindex(ocp.co2_dofs)
            d = ocp.co2_dofs[j]
            co2_out += ocp.co2_w[j] * (ocp.y_off[d] + ocp.sy[d] * Z[d, col])
        end
        J_conv += ocp.tab.b[i] * (co2_out / ocp.co2_in)
    end

    # piecewise-constant control
    J_reg = zero(eltype(z))
    for k in 1:Ne-1
        J_reg += (u[k+1] - u[k])^2
    end
    return J_conv / Ne + ocp.γ * Ne * J_reg
end

# Analytic grad
function ocp_grad!(g::AbstractVector, ocp::MethanationOCP, z::AbstractVector)
    n, s, Ne = ocp.n, ocp.s, ocp.Ne
    nz = n * ncols(ocp)
    fill!(g, zero(eltype(g)))

    # ∂f/∂Z[d,col(k,i)] = b_i·co2_w[j]·sy[d] / (co2_in·Ne)
    for k in 1:Ne, i in 1:s
        base = (stagecol(ocp, k, i) - 1) * n
        bi = ocp.tab.b[i] / (ocp.co2_in * Ne)
        for j in eachindex(ocp.co2_dofs)
            d = ocp.co2_dofs[j]
            g[base+d] = bi * ocp.co2_w[j] * ocp.sy[d]
        end
    end

    # ∂f/∂u[k] = 2γ·Ne·[(u_k − u_{k−1}) − (u_{k+1} − u_k)]
    q = 2 * ocp.γ * Ne
    u = view(z, nz+1:nz+Ne)
    for k in 1:Ne
        gk = zero(eltype(g))
        if k > 1
            (gk += u[k] - u[k-1])
        end
        if k < Ne 
            (gk -= u[k+1] - u[k])
        end
        g[nz+k] = q * gk
    end
    return g
end

struct OCPGradient{TO} <: ADNLPModels.ADBackend
    ocp::TO
end
ADNLPModels.gradient!(b::OCPGradient, g, f, x) = ocp_grad!(g, b.ocp, x)
