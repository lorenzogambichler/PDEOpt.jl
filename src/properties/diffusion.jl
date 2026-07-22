# D^eff_r,α

# Fuller
# D_ij = 1.43e-7·T^1.75 / (p_bar·√M_AB·(υ_i^1/3 + υ_j^1/3)²) in m²/s, M_AB = 2/(1/Mi+1/Mj)
function fuller_Dij(T, p, Mi, Mj, υi, υj)
    return fuller_factor(Mi, Mj, υi, υj) * T^1.75 / (p / 1e5)
end

# Precompute state-independent part for D_ij
function fuller_factor(Mi, Mj, υi, υj)
    Mab = 2 / (1 / (1e3 * Mi) + 1 / (1e3 * Mj)) # g/mol
    return 1.43e-7 / (sqrt(Mab) * (cbrt(υi) + cbrt(υj))^2)
end

# Kee et al.
# D_r,α = Σ_{j≠α} (ρ_j/M_j) / Σ_{j≠α} (ρ_j/M_j)/D_αj
function mixture_D(α::Int, c, Dbin, nsp::Int)
    num = 0.0
    den = 0.0
    @inbounds for j in 1:nsp
        j == α && continue
        num += c[j]
        den += c[j] / Dbin[α, j]
    end
    #return den > 0 ? num / den : zero(num)
    return num / max(den, 1e-30)
end

# Tsotsas–Schlünder
# D^eff_r,α = (1 − √(1−ε))·D_r,α + v_z·dp/8
tsotsas_Deff(Dr, ε, vz, dp) = (1 - sqrt(1 - ε)) * Dr + vz * dp / 8
