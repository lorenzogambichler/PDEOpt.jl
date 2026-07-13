module Models

using ..Properties

include("reactions.jl")

export AbstractModel, ADRModel, PSAModel, Reaction, Arrhenius, LHHW, AdsorptionLDF, TDependentD,
    fields, transported, state_dependent, diffusivity, velocity, reaction!, reaction_jac!,
    inlet_value, wall_coeff, wall_field, inlet_mod, wall_mod

abstract type AbstractModel end

# Spatial transport (advection,diffusion) for field?
transported(::AbstractModel, ::Int) = true
# Reassemble transprt per step (state-dependent coefficients)?
# false -> constant K assembled once (Arrhenius),  true -> D(T)/velocity(p) per step
state_dependent(::AbstractModel) = false

## Advection–diffusion–reaction
struct ADRModel{TF, TD, TR, F1, F2, F3} <: AbstractModel
    fields::TF 
    # Transport
    D::TD 
    β::Vector{Real} 
    # Reaction 
    react::TR
    # Wall
    U::Real # W/(m²K)
    T_wall::F3 # time modulation
    # Inlet
    g::F1 # spatial profile per field
    y_in::F2 # time modulation per field
end

fields(m::ADRModel) = m.fields

# Constant diffusivity, D[field, axis]
diffusivity(m::ADRModel, f::Int, axis::Int) = m.D[f, axis]

# T-dependent diffusivity (properties.jl)
struct FullerD
    # TODO: per-field correlation parameters
end
function diffusivity(D::FullerD, f::Int, axis::Int, T::Real, p::Real)
    axis == 1 && return 0.0 # only readial diffusion
    Ma, Mb, va, vb = D.params[f]
    return fuller_D(T, p, Ma, Mb, va, vb)
end

# State-dependent diffusivity dispatched on coefficient type (typeof(m.D))
# constant D (Matrix) -> state_dependent = false
# correllation D -> state_dependent = true
_statedep(::AbstractMatrix) = false
_statedep(::FullerD) = true
state_dependent(m::ADRModel) = _statedep(m.D)
diffusivity(m::ADRModel, f::Int, axis::Int, T::Real, p::Real) = diffusivity(m.D, f, axis, T, p)

velocity(m::ADRModel, axis::Int) = m.β[axis]

inlet_value(m::ADRModel, field::Symbol, r::Real) = m.g[field](r)

wall_coeff(m::ADRModel) = m.U
wall_field(::ADRModel) = :T

inlet_mod(m::ADRModel, field::Symbol, t::Real) = m.y_in[field](t)
wall_mod(m::ADRModel, t::Real) = m.T_wall(t)

reaction!(m::ADRModel, re, yc) = reaction!(m.react, re, yc)
reaction_jac!(m::ADRModel, Jre, yc) = reaction_jac!(m.react, Jre, yc)

end
