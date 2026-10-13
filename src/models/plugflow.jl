struct Arrhenius <: Reaction
    k0::Float64 # 1/s
    Ea::Float64 # J/mol
    ΔHr::Float64 # J/mol
end

# C consumed by first order reaction
function reaction_source(react::Arrhenius, C, T)
    r = react.k0 * exp(-react.Ea / (8.314 * T)) * C
    return (-r, -react.ΔHr * r)
end

function reaction!(react::Arrhenius, re, yc)
    C, T = yc[1], yc[2]
    re[1], re[2] = reaction_source(react, C, T) # (sC, sT)
    return re
end

function reaction_jac!(react::Arrhenius, Jre, yc)
    C, T = yc[1], yc[2]
    e = react.k0 * exp(-react.Ea / (8.314 * T))
    R = e * C
    RT = R * react.Ea / (8.314 * T^2)
    Jre[1, 1] = -e # ∂sC/∂C
    Jre[1, 2] = -RT # ∂sC/∂T
    Jre[2, 1] = -react.ΔHr * e # ∂sT/∂C
    Jre[2, 2] = -react.ΔHr * RT # ∂sT/∂T
    return Jre
end

# Advection-diffusion-reaction with constant transport
struct PlugFlowModel{TF, TD, TR, F2, F3} <: AbstractModel
    fields::TF
    # Transport
    D::TD
    β::Float64
    ρcp::Float64 # J/(m³K)
    # Reaction
    react::TR
    # Wall
    U::Float64 # W/(m²K)
    d::Float64 # m
    T_wall::F3 # time modulation
    # Inlet
    y_in::F2 # time modulation per field
end

fields(m::PlugFlowModel) = m.fields

# Constant transport -> assemble once (state_dependent = false)
diffusivity(m::PlugFlowModel, f::Int, ::Int) = m.D[f]
velocity(m::PlugFlowModel, ::Int) = m.β
capacity(m::PlugFlowModel, fi::Int) = m.fields[fi] === :T ? m.ρcp : 1.0
# props unused (constant coefficients)
face_velocity(m::PlugFlowModel, _, fi::Int, axis::Int, ::Int, ::Int, _) =
    capacity(m, fi) * velocity(m, axis)

# Per-field scaling for AntiDiffusion from a reference cell state yc (C, T)
antidiff_sc(::PlugFlowModel, yc::AbstractVector; ΔT::Float64=300.0) = [Float64(yc[1]), ΔT]

wall_coeff(m::PlugFlowModel) = 4*m.U/m.d
wall_field(::PlugFlowModel) = :T

inlet_mod(m::PlugFlowModel, field::Symbol, t::Real) = m.y_in[field](t)
wall_mod(m::PlugFlowModel, t::Real) = m.T_wall(t)

reaction!(m::PlugFlowModel, re, yc) = reaction!(m.react, re, yc)
# analytic jac -> ignore the finite diff args
reaction_jac!(m::PlugFlowModel, Jre, yc, fbase, fpert, ypert) = reaction_jac!(m.react, Jre, yc)
