struct MethanationOCP{TCL, TO, TR, TB, TS} <: AbstractOCP
    lay::TCL # CollocationLayout
    scl::TS # Scaling
    bnd::TB # Bounds
    obj::TO # Objective
    rsd::TR # Residual
    y0::Vector{Float64} # Init state
end

function MethanationOCP(sa, y0; Ne, Δt, s=3, Tmin, Tmax, Tw_max, Tw_min, γ=0.01, 
    recon=:vanalbada, ad_rel=1e-8)
    lay = methanation_layout(sa.prob, Ne, Δt, s)
    scl = methanation_scaling(sa.prob, y0; Tmin, Tmax, Tw_min, Tw_max)
    bnd = methanation_bounds(sa.prob, y0; Tmin, Tmax, Tw_min, Tw_max)
    rsd = methanation_residual(sa, lay, scl, y0; Tmin, Tmax, recon, ad_rel)
    obj = methanation_objective(sa, lay, scl; γ)
    return MethanationOCP(lay, scl, bnd, obj, rsd, copy(y0))
end

# Inlet density of each species, (ctot·M_α)
function methanation_rho_in(prob, y0::AbstractVector)
    m, dm = prob.model, prob.dm
    nsp = nspecies(m)
    ctot = sum(y0[dof(dm, 1, α)] / m.M[α] for α in 1:nsp) # inlet conc
    return [ctot * m.M[α] for α in 1:nsp]
end

# ρ -> inlet density, T -> [Tmin, Tmax], Tw -> [Tw_min, Tw_max], rows -> cell capacity
function methanation_scaling(prob, y0::AbstractVector; Tmin, Tmax, Tw_min, Tw_max)
    dm, m, grid, geom = prob.dm, prob.model, prob.grid, prob.geom
    nsp = nspecies(m)
    ρin = methanation_rho_in(prob, y0)
    ΔT = Tmax - Tmin
    ρcp_ref = (1 - m.ε) * m.ρcat * m.cp_cat

    sy = zeros(ndof(dm))
    y_off = zeros(ndof(dm))
    sc = zeros(ndof(dm))
    @inbounds for c in 1:ncells(grid)
        i, j = cellij(grid, c)
        V = cellvolume(geom, i, j)
        for α in 1:nsp
            d = dof(dm, c, α)
            sy[d] = ρin[α]
            sc[d] = 1 / (m.ε * V * sy[d])
        end
        d = dof(dm, c, nsp + 1)
        sy[d] = ΔT
        y_off[d] = Tmin
        sc[d] = 1 / (ρcp_ref * V * ΔT)
    end
    return Scaling(sy, y_off, [Tw_max - Tw_min], [Tw_min], sc)
end

# Species floor, T and Tw box (unscaled)
function methanation_bounds(prob, y0::AbstractVector; Tmin, Tmax, Tw_min, Tw_max,
    x_floor=1e-8, x_floor_H2=1e-3)
    dm, m = prob.dm, prob.model
    nsp = nspecies(m)
    ρin = methanation_rho_in(prob, y0)

    y_low = fill(-Inf, ndof(dm))
    y_up = fill(Inf, ndof(dm))
    for α in 1:nsp
        field = dm.fields[α]
        xf = field === :H2 ? x_floor_H2 : x_floor
        y_low[fielddof(dm, field)] .= xf * ρin[α]
    end
    Td = fielddof(dm, dm.fields[nsp+1])
    y_low[Td] .= Tmin
    y_up[Td] .= Tmax
    return Bounds(y_low, y_up, [Tw_min], [Tw_max])
end

# Radau IIA
function methanation_layout(prob, Ne, Δt, s)
    dm, m = prob.dm, prob.model
    coll = RadauIIA(s)
    ny = ndof(dm)
    nu = 1 # wall temp
    return CollocationLayout(Ne, Δt, ny, nu, coll)
end

methanation_residual(sa, lay, scl, y0; Tmin, Tmax, recon, ad_rel) =
    CollocationResidual(lay, scl, MethanationDynamics(sa, y0; Tmin, Tmax, recon, ad_rel))

function methanation_objective(sa, lay, scl; γ)
    mc = MethanationCost(sa, lay, scl, γ)
    return Objective(x -> methanation_cost(mc, x), 
        (g, x) -> methanation_cost_grad!(g, mc, x))
end