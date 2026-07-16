struct AdsorptionLDF <: Reaction
    # TODO: LDF coefficients k_i, isotherm params, ε, ρ_s
end
reaction!(react::AdsorptionLDF, re, yc) = error("TODO: LDF + isotherm coupling")
reaction_jac!(react::AdsorptionLDF, Jre, yc) = error("TODO: LDF Jacobian")

struct PSAModel{TG, TL, TA} <: AbstractModel
    gas::TG # transported gas field symbols, e.g. (:CO2, :N2, :T)
    loading::TL # cell-local loading field symbols, e.g. (:qCO2, :qN2)
    ads::TA # AdsorptionLDF coupling
    # TODO: transport coeffs (D, velocity via Ergun), bed props (ε, ρ_s), BCs
end

# gas fields first, then loadings (loadings carry no spatial flux)
fields(m::PSAModel) = (m.gas..., m.loading...)
transported(m::PSAModel, f::Int) = f ≤ length(m.gas)

reaction!(m::PSAModel, re, yc) = reaction!(m.ads, re, yc)
reaction_jac!(m::PSAModel, Jre, yc, fbase, fpert, ypert) = reaction_jac!(m.ads, Jre, yc)

# TODO: diffusivity, velocity, inlet/wall BC methods (reuse ADR kernels)
