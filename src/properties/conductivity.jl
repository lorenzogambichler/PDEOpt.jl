# Effective radial conductivity λ^eff_r

# Wassiljewa
# λ_gas = Σ_α (c_α λ_α) / Σ_β (c_β⋅φ_αβ) in W/(m K), c_α = ρ_α/M_α 
function mix_conductivity(c::NTuple{N,Any}, λ, μ, M) where {N}
    Tv = promote_type(eltype(c), eltype(λ), eltype(μ))
    λmix = zero(Tv)
    @inbounds for α in 1:N
        den = zero(Tv)
        for β in 1:N
            den += c[β] * wilke_phi(μ[α], μ[β], M[α], M[β])
        end
        #den > 0 && (λmix += c[α] * λ[α] / den)
        λmix += c[α] * λ[α] / max(den, 1e-30)
    end
    return λmix
end

# Bauer–Schlünder
# λ_r = 2.27e-7·ε_cat/(2−ε_cat)·T³·dp in W/(m K)
lambda_rad(T, ε_cat, dp) = 2.27e-7 * ε_cat / (2 - ε_cat) * T^3 * dp

# Bauer–Schlünder
# λ^eff_r = λ_conv + λ_cond,r 
# λ_conv = 1.15⋅ρ⋅v_z⋅cp,gas ⋅dp / (8[2 − (1 − dp/R)²])
# λ_cond,r = (1 − √(1−ε))⋅(λ_gas + ε λ_r) + √(1−ε)⋅λ_rs
# TODO λ_rs (has negligible impact on solution)
function lambda_eff_r(T, ρ, cp_gas, λ_gas, ε, ε_cat, dp, R, vz; λ_rs::Float64=0.0)
    λ_conv = 1.15 * ρ * vz * cp_gas * dp / (8 * (2 - (1 - dp / R)^2))
    λ_r = lambda_rad(T, ε_cat, dp)
    λ_cond = (1 - sqrt(1 - ε)) * (λ_gas + ε * λ_r) + sqrt(1 - ε) * λ_rs
    return λ_conv + λ_cond
end
