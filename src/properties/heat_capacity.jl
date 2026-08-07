# Heat capacity

# cp,gas = Σ(ρ_α/ρ)cpα in J/(kg K), ρ = Σρα
function mix_cp(ρα, cpα::NTuple{N,Any}, ρ) where {N}
    s = zero(promote_type(eltype(ρα), eltype(cpα)))
    @inbounds for α in 1:N
        s += ρα[α] * cpα[α]
    end
    #return ρ > 0 ? s / ρ : zero(s)
    return s / max(ρ, 1e-30)
end

# (ρcp)^eff = (1−ε)·ρcat·cp,cat + ε·ρ·cp,gas in J/(m³ K)
eff_capacity(ε, ρcat, cp_cat, ρ, cp_gas) = (1 - ε) * ρcat * cp_cat + ε * ρ * cp_gas
