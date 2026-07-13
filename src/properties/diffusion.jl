# Fuller binary gas diffusion coefficient D(T, p)
fuller_Dij(T, p, Ma, Mb, va, vb) = 0.00143 * T^1.75 * sqrt(2/(1/Ma+1/Mb)) / (p*(cbrt(va)+cbrt(vb))^2)

# Tsotas/Schlünder radial diffusion coefficient for species i 
tsotas_Dri(T, p...) = error("TODO: Tsotas/Schlünder radial diffusion coeff")