# first order, T_floor gurad for low T
struct Arrhenius <: Reaction
    k0::Float64 # 1/s
    Ea::Float64 # J/mol
    ΔHr::Float64 # J/mol
    T_floor::Float64 # K
end

# No T-safeguard for opt
function reaction_source(react::Arrhenius, C, T)
    r = react.k0 * exp(-react.Ea / (8.314 * T)) * C
    return (-r, -react.ΔHr * r)
end

function reaction!(react::Arrhenius, re, yc)
    C, T = yc[1], yc[2]
    re[1], re[2] = reaction_source(react, C, max(T, react.T_floor)) # (sC, sT)
    return re
end

function reaction_jac!(react::Arrhenius, Jre, yc)
    C, T = yc[1], yc[2]
    e = react.k0 * exp(-react.Ea / (8.314 * max(T, react.T_floor)))
    R = e * C
    RT = T > react.T_floor ? R * react.Ea / (8.314 * T^2) : 0.0 # max()-term
    Jre[1, 1] = -e # ∂sC/∂C
    Jre[1, 2] = -RT # ∂sC/∂T
    Jre[2, 1] = -react.ΔHr * e # ∂sT/∂C
    Jre[2, 2] = -react.ΔHr * RT # ∂sT/∂T
    return Jre
end

# Advection-diffusion-reaction with constant transport
struct PlugFlowModel{TF, TD, TR, F1, F2, F3} <: AbstractModel
    fields::TF
    # Transport
    D::TD          # D[field, axis], constant
    β::Vector{Float64}
    # Reaction
    react::TR
    # Wall
    U::Float64 # W/(m²K)
    T_wall::F3 # time modulation
    # Inlet
    g::F1 # spatial profile per field
    y_in::F2 # time modulation per field
end

fields(m::PlugFlowModel) = m.fields

# Constant transport -> assemble once (state_dependent = false)
diffusivity(m::PlugFlowModel, f::Int, axis::Int) = m.D[f, axis]
velocity(m::PlugFlowModel, axis::Int) = m.β[axis]

inlet_value(m::PlugFlowModel, field::Symbol, r::Real) = m.g[field](r)
wall_coeff(m::PlugFlowModel) = m.U
wall_field(::PlugFlowModel) = :T

inlet_mod(m::PlugFlowModel, field::Symbol, t::Real) = m.y_in[field](t)
wall_mod(m::PlugFlowModel, t::Real) = m.T_wall(t)

reaction!(m::PlugFlowModel, re, yc) = reaction!(m.react, re, yc)
# analytic jac -> ignore the finite diff args
reaction_jac!(m::PlugFlowModel, Jre, yc, fbase, fpert, ypert) = reaction_jac!(m.react, Jre, yc)
