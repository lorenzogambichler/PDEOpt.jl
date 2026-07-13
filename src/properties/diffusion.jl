# Gas-phase diffusion correlations for radial dispersion coefficient D^eff_r,α
# Eval order per cell: binary D_ij (Fuller) -> mixture-average D_r,α (Kee)
# -> effective radial dispersion D^eff_r,α (Tsotsas–Schlünder)

# Fuller binary diffusion coefficient in m²/s (standard SI form)
# D = 1.43e-7·T^1.75 / (p_bar·√M_AB·(υ_i^1/3 + υ_j^1/3)²),  M_AB = 2/(1/Mi+1/Mj)
# values (e.g. H2/CO2 ≈ 3.9e-5 m²/s at 550 K, 5 bar)
function fuller_Dij(T, p, Mi, Mj, υi, υj)
    Mab = 2 / (1 / (1e3 * Mi) + 1 / (1e3 * Mj)) # g/mol
    return 1.43e-7 * T^1.75 / ((p / 1e5) * sqrt(Mab) * (cbrt(υi) + cbrt(υj))^2)
end

# Mixture-averaged diffusion of α into remaining mixture (Kee et al.)
# D_r,α = Σ_{j≠α} (ρ_j/M_j) / Σ_{j≠α} (ρ_j/M_j)/D_αj
# c[j] = ρ_j/M_j molar concentration in mol/m³, Dbin[α,j] binary coeffs
function mixture_D(α::Int, c, Dbin, nsp::Int)
    num = 0.0
    den = 0.0
    @inbounds for j in 1:nsp
        j == α && continue
        num += c[j]
        den += c[j] / Dbin[α, j]
    end
    return den > 0 ? num / den : zero(num)
end

# Tsotsas–Schlünder effective radial dispersion (packed-bed cross-mixing)
# D^eff_r,α = (1 − √(1−ε))·D_r,α + v_z·dp/8
tsotsas_Deff(Dr, ε, vz, dp) = (1 - sqrt(1 - ε)) * Dr + vz * dp / 8
