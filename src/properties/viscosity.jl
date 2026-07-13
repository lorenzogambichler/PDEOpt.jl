# Viscosity correlations and the shared Wilke/Wassiljewa interaction factor.
# Per-species μ_α(T), λ_α(T) come from VDI polynomial banks on the model.

# Wilke interaction factor φ_αβ (also used by the thermal-conductivity mixing):
#   φ_αβ = [1 + (μ_α/μ_β)^0.5 (M_β/M_α)^0.25]² / √(8 (1 + M_α/M_β))
function wilke_phi(μα, μβ, Mα, Mβ)
    num = (1 + sqrt(μα / μβ) * (Mβ / Mα)^0.25)^2
    return num / sqrt(8 * (1 + Mα / Mβ))
end

# Mixture dynamic viscosity (Wilke)  [Pa·s]
#   μ_mix = Σ_α x_α μ_α / Σ_β x_β φ_αβ
# x mole fractions, μ per-species viscosities, M molar masses.
function mix_viscosity(x, μ, M, nsp::Int)
    μmix = 0.0
    @inbounds for α in 1:nsp
        den = 0.0
        for β in 1:nsp
            den += x[β] * wilke_phi(μ[α], μ[β], M[α], M[β])
        end
        den > 0 && (μmix += x[α] * μ[α] / den)
    end
    return μmix
end
