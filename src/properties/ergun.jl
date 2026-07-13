# Ergun equation — packed-bed axial pressure gradient dp/dz [Pa/m].
#   dp/dz = −[ 150·μ(1−ε)²/(ε³ dp²)·u  +  1.75·ρ(1−ε)/(ε³ dp)·u|u| ]
# Superficial velocity u = v_z is a fixed model input (paper: constant v_z),
# so this returns the gradient, not the velocity. Integrate it cumulatively
# down the axis from p_in to obtain the pressure profile p(z):
#   p[i] = p[i-1] + dpdz(cell i-1)·Δz    (dpdz < 0)
function ergun_dpdz(u, μ, ρ, ε, dp)
    visc = 150 * μ * (1 - ε)^2 / (ε^3 * dp^2) * u
    inert = 1.75 * ρ * (1 - ε) / (ε^3 * dp) * u * abs(u)
    return -(visc + inert)
end
