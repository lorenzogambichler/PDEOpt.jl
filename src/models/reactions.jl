# Reaction template:
# - param struct
# - reaction! maps yc -> re per cell
# - reaction_jac! maps yc -> Jre per cell
abstract type Reaction end

struct Arrhenius <: Reaction
    k0::Float64 # 1/s
    Ea::Float64 # J/mol
    ΔHr::Float64 # J/mol
    T_floor::Float64 # K
end
function reaction!(react::Arrhenius, re, yc)
    C, T = yc[1], yc[2]
    R = react.k0 * exp(-react.Ea / (8.314 * max(T, react.T_floor))) * C
    re[1] = -R
    re[2] = -react.ΔHr * R
    return re
end
function reaction_jac!(react::Arrhenius, Jre, yc)
    C, T = yc[1], yc[2]
    e = react.k0 * exp(-react.Ea / (8.314 * max(T, react.T_floor)))
    R = e * C
    RT = T > react.T_floor ? R * react.Ea / (8.314 * T^2) : 0.0
    Jre[1, 1] = -e # ∂sC/∂C
    Jre[1, 2] = -RT # ∂sC/∂T
    Jre[2, 1] = -react.ΔHr * e # ∂sT/∂C
    Jre[2, 2] = -react.ΔHr * RT # ∂sT/∂T
    return Jre
end

# Langmuir–Hinshelwood–Hougen–Watson (skeleton).
# State-dependent rate parameters (rate constant, adsorption constants) are drawn
# from the Properties correlations, evaluated from the local state yc (T, p_i).
struct LHHW <: Reaction
    # TODO: k0, Ea, adsorption constants, stoichiometry ν, ΔHr, T_floor
end
function reaction!(react::LHHW, re, yc)
    # TODO: r = k(T)·(driving force) / (1 + Σ K_i(T)·p_i)^n ;  re .= ν·r ,  re[T] = -ΔHr·r
    error("TODO: LHHW rate law")
end
function reaction_jac!(react::LHHW, Jre, yc)
    # TODO: analytic quotient rule, or ForwardDiff fallback on reaction!
    error("TODO: LHHW Jacobian")
end