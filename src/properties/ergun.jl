# Ergun
# dp/dz = −150·μ(1−ε)²/(ε³⋅dp²)·u + 1.75·ρ(1−ε)/(ε³⋅dp)·u|u|
# p[i] = p[i-1] + dpdz[i-1]·Δz
function ergun_dpdz(u, μ, ρ, ε, dp)
    visc = 150 * μ * (1 - ε)^2 / (ε^3 * dp^2) * u
    inert = 1.75 * ρ * (1 - ε) / (ε^3 * dp) * u * abs(u)
    return -(visc + inert)
end
