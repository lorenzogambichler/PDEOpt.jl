# Reaction equilibrium const for Xu–Froment
# Xu–Froment, T in K
# R1: CH4 + H2O <-> CO + 3*H2, K1 in bar²
# R2: CO + H2O <-> CO2 + H2, K2 in -
# R3: CH4 + 2*H2O <-> CO2 + 4*H2, K3 = K1·K2 in bar²
Keq1(T) = exp(-26830.0 / T + 30.114)
Keq2(T) = exp(4400.0 / T - 4.036)
Keq3(T) = Keq1(T) * Keq2(T)
