struct MethanationCost{TSA, TL, TS, TI}
    sa::TSA # StateAssembly
    lay::TL #CollocationLayout
    scl::TS # Scaling
    co2_dofs::Vector{Int} # outlet dofs
    co2_w::Vector{Float64} # vel_z * A
    co2_in::TI # inlet t-dependent
    γ::Float64 # control regularization
end

# TODO 
function MethanationCost(sa, lay, scl, γ)
    dm, m, f_in= sa.prob.dm, sa.prob.model, sa.prob.f_in
    bfs_out, bfg_out = sa.ebnd[2]

    co2_dofs = [dof(dm, c, :CO2) for c in bfs_out.cells]
    co2_w = velocity(m, 1) .* copy(bfg_out.area)
    co2_in(t) = sum(@view f_in[fielddof(dm, :CO2)]) * inlet_mod(m, :CO2, t)
    return MethanationCost(sa, lay, scl, co2_dofs, co2_w, co2_in, γ)
end

# stage cost (x) -> (ℓ(x), ∇ℓ(x))
# J(x) = 1/Ne(∑ₖ(∑ᵢbᵢ/co2_in(∑ⱼwⱼ(offⱼ + syⱼ*Ykiⱼ)) + γ⋅Ne(∑ₖ(ukp1 - uk)^2)
function methanation_cost(mc::MethanationCost, x)
    Ne, s = mc.lay.Ne, mc.lay.coll.s

    J_conv = zero(eltype(x))
    for k in 1:Ne, i in 1:s
        ti = stage_time(mc.lay.coll, k, i, mc.lay.Δt)
        Yi = stage(mc.lay, x, k, i)
        co2_out = zero(eltype(x))
        for j in eachindex(mc.co2_dofs)
            d = mc.co2_dofs[j]
            co2_out += mc.co2_w[j] * (mc.scl.y_off[d] + mc.scl.sy[d] * Yi[d])
        end
        J_conv += mc.lay.coll.b[i] * (co2_out / mc.co2_in(ti))
    end

    J_reg = zero(eltype(x))
    for k in 1:Ne-1
        uk = control(mc.lay, x, k)
        ukp1 = control(mc.lay, x, k+1)
        J_reg += (ukp1[1] - uk[1])^2
    end

    val = J_conv / Ne + mc.γ * Ne * J_reg
    return val
end

function methanation_cost_grad!(g, mc::MethanationCost, x)
    Ne, s = mc.lay.Ne, mc.lay.coll.s
    fill!(g, zero(eltype(x)))
    
    # ∂f/∂Y^ki[d] = b[i]·co2_w[j]·sy[d] / (co2_in·Ne)
    for k in 1:Ne, i in 1:s
        gi = view(g, stage_range(mc.lay, k, i))
        ti = stage_time(mc.lay.coll, k, i, mc.lay.Δt)
        for j in eachindex(mc.co2_dofs)
            d = mc.co2_dofs[j]
            gi[d] = mc.lay.coll.b[i] * mc.co2_w[j] * mc.scl.sy[d] / (Ne * mc.co2_in(ti))
        end
    end

    # ∂f/∂u[k] = 2γ·Ne·[(u_k − u_{k−1}) − (u_{k+1} − u_k)]
    q = 2 * mc.γ * Ne
    for k in 1:Ne-1
        gk = view(g, control_range(mc.lay, k))
        gkp1 = view(g, control_range(mc.lay, k+1))
        uk = control(mc.lay, x, k)
        ukp1 = control(mc.lay, x, k+1)

        gk[1] -= q * (ukp1[1] - uk[1])
        gkp1[1]+= q * (ukp1[1] - uk[1])
    end

    return g
end