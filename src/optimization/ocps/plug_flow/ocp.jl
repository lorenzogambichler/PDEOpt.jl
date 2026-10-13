struct PlugFlowOCP{TL, TS, TB, TO, TR} <: AbstractOCP
    lay::TL # CollocationLayout
    scl::TS # Scaling
    bnd::TB # Bounds
    obj::TO # Objective
    rsd::TR # Residual
    y0::Vector{Float64} # Init state
end

function PlugFlowOCP(prob, y0; Ne, Δt, s=3, Tmin, Tmax, Tw_max, Tw_min, γ=0.01,
    recon=:vanalbada, ad_rel=1e-3)
    lay = plug_flow_layout(prob, Ne, Δt, s)
    scl = plug_flow_scaling(prob; Tmin, Tmax, Tw_min, Tw_max)
    bnd = plug_flow_bounds(prob; Tmin, Tmax, Tw_min, Tw_max)
    rsd = plug_flow_residual(prob, lay, scl, y0; Tmin, Tmax, recon, ad_rel)
    obj = plug_flow_objective(prob, lay, scl; γ)
    return PlugFlowOCP(lay, scl, bnd, obj, rsd, copy(y0))
end

# Radau IIA
function plug_flow_layout(prob, Ne, Δt, s)
    coll = RadauIIA(s)
    ny = ndof(prob.dm)
    nu = 1 # wall temp
    return CollocationLayout(Ne, Δt, ny, nu, coll)
end

# C -> inlet conc, T -> [Tmin, Tmax], Tw -> [Tw_min, Tw_max], rows -> cell capacity
function plug_flow_scaling(prob; Tmin, Tmax, Tw_min, Tw_max)
    dm, m, grid, geom = prob.dm, prob.model, prob.grid, prob.geom
    ΔT = Tmax - Tmin
    iC, iT = fieldindex(dm, :C), fieldindex(dm, :T)

    sy = zeros(ndof(dm))
    y_off = zeros(ndof(dm))
    sc = zeros(ndof(dm))
    @inbounds for c in 1:ncells(grid)
        i, j = cellij(grid, c)
        V = cellvolume(geom, i, j)

        d = dof(dm, c, iC)
        sy[d] = inlet_mod(m, :C, 0.0)
        sc[d] = 1 / (capacity(m, iC) * V * sy[d])

        d = dof(dm, c, iT)
        sy[d] = ΔT
        y_off[d] = Tmin
        sc[d] = 1 / (capacity(m, iT) * V * ΔT)
    end
    return Scaling(sy, y_off, [Tw_max - Tw_min], [Tw_min], sc)
end

# C floor, T and Tw box (unscaled)
function plug_flow_bounds(prob; Tmin, Tmax, Tw_min, Tw_max, c_floor=1e-8)
    dm, m = prob.dm, prob.model

    y_low = fill(-Inf, ndof(dm))
    y_up = fill(Inf, ndof(dm))
    y_low[fielddof(dm, :C)] .= c_floor * inlet_mod(m, :C, 0.0)
    Td = fielddof(dm, :T)
    y_low[Td] .= Tmin
    y_up[Td] .= Tmax
    return Bounds(y_low, y_up, [Tw_min], [Tw_max])
end

plug_flow_residual(prob, lay, scl, y0; Tmin, Tmax, recon, ad_rel) =
    CollocationResidual(lay, scl, PlugFlowDynamics(prob, y0; Tmin, Tmax, recon, ad_rel))

function plug_flow_objective(prob, lay, scl; γ)
    pfc = PlugFlowCost(prob, lay, scl, γ)
    return Objective(x -> plug_flow_cost(pfc, x),
        (g, x) -> plug_flow_cost_grad!(g, pfc, x))
end
