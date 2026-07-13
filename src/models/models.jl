module Models

using ..Properties

# Model interface:
# Contains physics that assembly kernels need
# - transport coefficients (diffusivity, velocity)
# - reaction source (reaction!/reaction_jac!)
# - boundary data (inlet/wall)
# - per cell property pass (properties!, for state-dependent problems) 
# - capacity (mass-matrix) contribution
abstract type AbstractModel end

# Spatial transport (advection, diffusion) for field fi?
transported(::AbstractModel, ::Int) = true

# Reassemble transport/capacity per step (state-dependent coefficients)?
state_dependent(::AbstractModel) = false

# Reactions:
# Maps local cell state yc to volumetric source re (per field)
# reaction_jac! for local nf×nf jac Jre
# E.g. 
# - Arrhenius (plugflow.jl)
# - Xu–Froment (methanation.jl),
# - AdsorptionLDF (psa.jl)
abstract type Reaction end

# finite diff jac of reaction! closure (if no analytic jac available)
function reaction_jac_fd!(f!, Jre, yc, re; relstep::Float64=1e-7)
    n = length(yc)
    f!(re, yc)                      # base source
    r0 = copy(re)
    ytmp = collect(float.(yc))
    @inbounds for k in 1:n
        h = relstep * max(abs(ytmp[k]), 1.0)
        yk = ytmp[k]
        ytmp[k] = yk + h
        f!(re, ytmp)                # perturbed source
        for a in 1:n
            Jre[a, k] = (re[a] - r0[a]) / h
        end
        ytmp[k] = yk
    end
    f!(re, yc)                      # restore re to the base source
    return Jre
end

# Models
include("plugflow.jl")
include("methanation.jl")
include("psa.jl")

export AbstractModel, transported, state_dependent,
    # reactions
    Reaction, Arrhenius, reaction!, reaction_jac!, reaction_jac_fd!,
    # plug flow
    PlugFlowModel, fields, diffusivity, velocity, inlet_value,
    wall_coeff, wall_field, inlet_mod, wall_mod,
    # methanation
    MethanationModel, MethanationProps, properties!, nspecies,
    # PSA
    PSAModel, AdsorptionLDF
end
