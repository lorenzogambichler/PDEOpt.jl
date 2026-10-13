struct PlugFlowCost{TL, TS, TI}
    lay::TL # CollocationLayout
    scl::TS # Scaling
    c_dofs::Vector{Int} # outlet dofs
    c_w::Vector{Float64} # β * A
    c_in::TI # inlet t-dependent
    γ::Float64 # control regularization
end

function PlugFlowCost(prob, lay, scl, γ)
    c_dofs, c_w = plug_flow_outlet(prob, :C)
    c_in(t) = inflow(prob, :C, t)
    return PlugFlowCost(lay, scl, c_dofs, c_w, c_in, γ)
end

# J(x) = 1/Ne(∑ₖ∑ᵢbᵢ·(∑ⱼwⱼ(offⱼ + syⱼ*Ykiⱼ))/c_in(tᵢ)) + γ⋅Ne(∑ₖ(ukp1 - uk)^2)
function plug_flow_cost(pfc::PlugFlowCost, x)
    Ne, s = pfc.lay.Ne, pfc.lay.coll.s

    J_out = zero(eltype(x))
    for k in 1:Ne, i in 1:s
        ti = stage_time(pfc.lay.coll, k, i, pfc.lay.Δt)
        Yi = stage(pfc.lay, x, k, i)
        c_out = zero(eltype(x))
        for j in eachindex(pfc.c_dofs)
            d = pfc.c_dofs[j]
            c_out += pfc.c_w[j] * (pfc.scl.y_off[d] + pfc.scl.sy[d] * Yi[d])
        end
        J_out += pfc.lay.coll.b[i] * (c_out / pfc.c_in(ti))
    end

    J_reg = zero(eltype(x))
    for k in 1:Ne-1
        uk = control(pfc.lay, x, k)
        ukp1 = control(pfc.lay, x, k+1)
        J_reg += (ukp1[1] - uk[1])^2
    end

    return J_out / Ne + pfc.γ * Ne * J_reg
end

function plug_flow_cost_grad!(g, pfc::PlugFlowCost, x)
    Ne, s = pfc.lay.Ne, pfc.lay.coll.s
    fill!(g, zero(eltype(x)))

    # ∂f/∂Y^ki[d] = b[i]·c_w[j]·sy[d] / (c_in·Ne)
    for k in 1:Ne, i in 1:s
        gi = view(g, stage_range(pfc.lay, k, i))
        ti = stage_time(pfc.lay.coll, k, i, pfc.lay.Δt)
        for j in eachindex(pfc.c_dofs)
            d = pfc.c_dofs[j]
            gi[d] = pfc.lay.coll.b[i] * pfc.c_w[j] * pfc.scl.sy[d] / (Ne * pfc.c_in(ti))
        end
    end

    # ∂f/∂u[k] = 2γ·Ne·[(u_k − u_{k−1}) − (u_{k+1} − u_k)]
    q = 2 * pfc.γ * Ne
    for k in 1:Ne-1
        gk = view(g, control_range(pfc.lay, k))
        gkp1 = view(g, control_range(pfc.lay, k+1))
        uk = control(pfc.lay, x, k)
        ukp1 = control(pfc.lay, x, k+1)

        gk[1] -= q * (ukp1[1] - uk[1])
        gkp1[1] += q * (ukp1[1] - uk[1])
    end

    return g
end
