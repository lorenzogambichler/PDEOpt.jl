module Models

using ..Properties

# Contains physics that assembly kernels need
abstract type AbstractModel end

# Spatial transport (advection, diffusion) for field fi?
transported(::AbstractModel, ::Int) = true

# Reassemble transport/capacity per step (state-dependent coefficients)?
state_dependent(::AbstractModel) = false

# Maps local cell state yc to volumetric source re (per field)
abstract type Reaction end

# Finite diff jac of reaction! closure (if no analytic jac available)
function reaction_jac_fd!(f!, Jre, yc, fbase, fpert, ypert; relstep::Float64=1e-7)
    n = length(yc)
    f!(fbase, yc) # base
    copyto!(ypert, yc)
    @inbounds for k in 1:n
        h = relstep * max(abs(yc[k]), 1.0)
        ypert[k] = yc[k] + h
        f!(fpert, ypert) # perturbed
        for a in 1:n
            Jre[a, k] = (fpert[a] - fbase[a]) / h
        end
        ypert[k] = yc[k]
    end
    return Jre
end

# Models
include("plugflow.jl")
include("methanation.jl")
include("psa.jl")

export AbstractModel, transported, state_dependent,
    # reactions
    Reaction, Arrhenius, reaction!, reaction_jac!, reaction_jac_fd!, reaction_source,
    # plug flow
    PlugFlowModel, fields, diffusivity, velocity, inlet_value,
    wall_coeff, wall_field, inlet_mod, wall_mod,
    # methanation
    MethanationModel, MethanationProps, properties!, nspecies,
    capacity, face_diffusivity, face_velocity,
    # PSA
    PSAModel, AdsorptionLDF
    # SMB
end
