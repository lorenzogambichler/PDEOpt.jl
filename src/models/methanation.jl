### T-dependent property correlations 
# Polynomial in T per species, evalpoly(T, c) = c1 + c2 T + c3 T² + ...
struct Poly{N}
    c::NTuple{N,Float64}
end
(p::Poly)(T) = evalpoly(T, p.c)

# Placeholders -> constant values (~300–400 K) 
# TODO: replace tuples with VDI polynomial coeffs (cp,α, μ_α, λ_α)
const DEFAULT_CP = ntuple(i -> Poly((( 2200.0, 1040.0, 846.0, 1996.0, 14310.0, 1040.0)[i],)), 6) # J/(kg K)
const DEFAULT_MU = ntuple(i -> Poly((( 1.1e-5, 1.8e-5, 1.5e-5, 1.3e-5, 0.9e-5, 1.8e-5)[i],)), 6) # Pa·s
const DEFAULT_LAM = ntuple(i -> Poly((( 0.034, 0.025, 0.017, 0.025, 0.18, 0.026)[i],)), 6) # W/(m K)

### Model 
# 2D pseudo-homogeneous fixed-bed CO2 methanation (Bremer et al.)
# State per cell: ρ_α (CH4,CO,CO2,H2O,H2,N2) + T
# Transport, capacity and reaction coefficients state-dependent -> state_dependent(::MethanationModel) = true
# FIXED order: 1 CH4, 2 CO, 3 CO2, 4 H2O, 5 H2, 6 N2(inert), 7 T
struct MethanationModel{TF, TCP, TMU, TLA, F1, F2, F3} <: AbstractModel
    fields::TF
    # Species data
    M::NTuple{6,Float64} # molar mass, kg/mol
    υ::NTuple{6,Float64} # Fuller diffusion volume, -
    ν::NTuple{3,NTuple{6,Float64}} # stoichiometry ν[β][α]
    ΔHr::NTuple{3,Float64} # reaction enthalpy, J/mol
    # Kinetics (Xu–Froment, Table 2)
    A::NTuple{3,Float64} # pre-exponential b_β
    E::NTuple{3,Float64} # activation energy, J/mol
    Aads::NTuple{4,Float64} # adsorption pre-exp (CH4,CO,H2,H2O)
    ΔHads::NTuple{4,Float64} # adsorption enthalpy, J/mol
    χ::Float64 # effectiveness factor
    # Bed / geometry
    ρcat::Float64
    cp_cat::Float64
    ε::Float64
    ε_cat::Float64
    dp::Float64
    Rrad::Float64
    vz::Float64
    kw::Float64
    λ_rs::Float64
    # Properties
    cp_corr::TCP
    μ_corr::TMU
    λ_corr::TLA
    # Boundary data
    g::F1 # inlet spatial profile per field
    y_in::F2 # inlet time modulation per field
    T_wall::F3 # cooling-wall temperature T(t) (or T(t,z))
end

# Keyword constructor
function MethanationModel(fields; g, y_in, T_wall, vz, kw,
    ε=0.4, dp=0.002, Rrad=0.01, ρcat=2355.2, cp_cat=1107.0,
    ε_cat=0.4, χ=0.1, λ_rs=0.0,
    cp_corr=DEFAULT_CP, μ_corr=DEFAULT_MU, λ_corr=DEFAULT_LAM)
    M = (0.016043, 0.028010, 0.044010, 0.018015, 0.0020159, 0.0280134)
    υ = (25.14, 18.0, 26.9, 13.1, 6.12, 18.5) 
    ν = ((-1.0, 1.0, 0.0, -1.0, 3.0, 0.0), # R1: CH4 + H2O ⇌ CO + 3 H2
        (0.0, -1.0, 1.0, -1.0, 1.0, 0.0), # R2: CO + H2O ⇌ CO2 + H2
        (-1.0, 0.0, 1.0, -2.0, 4.0, 0.0)) # R3: CH4 + 2 H2O ⇌ CO2 + 4 H2
    ΔHr = (206.3e3, -41.1e3, 164.9e3)
    A = (5.176e15, 2.395e6, 1.250e15)
    E = (240.10e3, 67.13e3, 243.90e3)
    Aads = (8.15e-4, 1.008e-4, 7.50e-9, 2.17e5)
    ΔHads = (-38.28e3, -70.65e3, -82.90e3, 88.68e3)
    return MethanationModel(fields, M, υ, ν, ΔHr, A, E, Aads, ΔHads, χ,
        ρcat, cp_cat, ε, ε_cat, dp, Rrad, vz, kw, λ_rs,
        cp_corr, μ_corr, λ_corr, g, y_in, T_wall)
end

fields(m::MethanationModel) = m.fields
nspecies(m::MethanationModel) = length(m.M)
state_dependent(::MethanationModel) = true

velocity(m::MethanationModel, axis::Int) = axis == 1 ? m.vz : 0.0
wall_coeff(m::MethanationModel) = m.kw
wall_field(::MethanationModel) = :T
inlet_value(m::MethanationModel, field::Symbol, r::Real) = m.g[field](r)
inlet_mod(m::MethanationModel, field::Symbol, t::Real) = m.y_in[field](t)
wall_mod(m::MethanationModel, t::Real) = m.T_wall(t)

### Property cache
struct MethanationProps
    ρ::Vector{Float64} # mass density, kg/m³
    ρM::Vector{Float64} # molar density Σ ρ_α/M_α, mol/m³
    p::Vector{Float64} # total pressure, Pa
    x::Matrix{Float64} # mole fractions (nsp × ncell)
    c::Matrix{Float64} # molar conc ρ_α/M_α (nsp × ncell)
    Deff::Matrix{Float64} # effective radial dispersion D^eff_r,α (nsp × ncell)
    ρcp::Vector{Float64} # (ρcp)^eff -> energy capacity matrix
    ρcp_flow::Vector{Float64} # ρ·cp,gas -> energy advection coefficient
    λeff::Vector{Float64} # effective radial conductivity λ^eff_r
    # temporary (overwritten each cell)
    cpα::Vector{Float64}
    μα::Vector{Float64}
    λα::Vector{Float64}
    Dbin::Matrix{Float64} # binary diffusion (nsp × nsp)
end
function MethanationProps(nsp::Int, ncell::Int)
    z1() = zeros(ncell)
    z2() = zeros(nsp, ncell)
    MethanationProps(z1(), z1(), z1(), z2(), z2(), z2(), z1(), z1(), z1(),
        zeros(nsp), zeros(nsp), zeros(nsp), zeros(nsp, nsp))
end

### Properties
# Maps current y to properties
# 1) ρ, ρM, x, c -> 2) p (ideal gas) -> 3) cp, μ, λ per species -> 
# 4) cp_gas, (ρcp)^eff -> 5) λ_gas, λ^eff_r -> 6) D_ij, D_r,α, D^eff_r,α
function properties!(m::MethanationModel, props::MethanationProps, y::AbstractVector)
    nsp = nspecies(m)
    N = nsp + 1
    ncell = length(props.ρ)
    Rgas = 8.314
    @inbounds for cell in 1:ncell
        base = (cell - 1) * N
        T = y[base+N]
        ρview = view(y, base+1:base+nsp)

        # composition
        ρ = 0.0; ρM = 0.0
        for α in 1:nsp
            cα = ρview[α] / m.M[α]
            props.c[α, cell] = cα
            ρ += ρview[α]
            ρM += cα
        end
        props.ρ[cell] = ρ
        props.ρM[cell] = ρM
        for α in 1:nsp
            props.x[α, cell] = ρM > 0 ? props.c[α, cell] / ρM : 0.0
        end

        # pressure (ideal gas)
        # For Ergun profile, replace with axial sweep
        # (p[cell] = p_in + Σ ergun_dpdz(vz, μmix, ρ, ε, dp)·Δz)
        p = ρM * Rgas * T
        props.p[cell] = p

        # T-dependent properties (species)
        for α in 1:nsp
            props.cpα[α] = m.cp_corr[α](T)
            props.μα[α] = m.μ_corr[α](T)
            props.λα[α] = m.λ_corr[α](T)
        end

        # heat capacity
        cp_gas = mix_cp(ρview, props.cpα, ρ, nsp)
        props.ρcp[cell] = eff_capacity(m.ε, m.ρcat, m.cp_cat, ρ, cp_gas)
        props.ρcp_flow[cell] = ρ * cp_gas

        # conductivity
        λ_gas = mix_conductivity(view(props.c, :, cell), props.λα, props.μα, m.M, nsp)
        props.λeff[cell] = lambda_eff_r(T, ρ, cp_gas, λ_gas, m.ε, m.ε_cat,
            m.dp, m.Rrad, m.vz; λ_rs=m.λ_rs)

        # diffusion, binary -> mixture-average -> effective radial dispersion
        for i in 1:nsp, j in 1:nsp
            props.Dbin[i, j] = i == j ? 0.0 :
                fuller_Dij(T, p, m.M[i], m.M[j], m.υ[i], m.υ[j])
        end
        for α in 1:nsp
            Dr = mixture_D(α, view(props.c, :, cell), props.Dbin, nsp)
            props.Deff[α, cell] = tsotsas_Deff(Dr, m.ε, m.vz, m.dp)
        end
    end
    return props
end

# Effective radial dispersion, species α at cell
diffusivity(m::MethanationModel, props::MethanationProps, α::Int, cell::Int) =
    props.Deff[α, cell]

### Reaction (Xu-Froment LHHW) 
# Volumetric rates r̃_β in mol/(m³_cat s) from local state yc
function _rates(m::MethanationModel, yc)
    T = yc[nspecies(m)+1]
    RT = 8.314 * T
    # partial pressures in bar: p_α = (ρ_α/M_α)·R·T (ideal gas)
    pCH4 = yc[1] / m.M[1] * RT / 1e5
    pCO  = yc[2] / m.M[2] * RT / 1e5
    pCO2 = yc[3] / m.M[3] * RT / 1e5
    pH2O = yc[4] / m.M[4] * RT / 1e5
    pH2  = max(yc[5] / m.M[5] * RT / 1e5, 1e-12)

    b1 = m.A[1] * exp(-m.E[1] / RT)
    b2 = m.A[2] * exp(-m.E[2] / RT)
    b3 = m.A[3] * exp(-m.E[3] / RT)
    BCH4 = m.Aads[1] * exp(-m.ΔHads[1] / RT)
    BCO  = m.Aads[2] * exp(-m.ΔHads[2] / RT)
    BH2  = m.Aads[3] * exp(-m.ΔHads[3] / RT)
    BH2O = m.Aads[4] * exp(-m.ΔHads[4] / RT)
    DEN = 1 + BCO * pCO + BH2 * pH2 + BCH4 * pCH4 + BH2O * pH2O / pH2

    K1 = Keq1(T); K2 = Keq2(T); K3 = Keq3(T)
    r1 = b1 / pH2^2.5 * (pCH4 * pH2O - pH2^3 * pCO / K1) / DEN^2
    r2 = b2 / pH2     * (pCO * pH2O - pH2 * pCO2 / K2) / DEN^2
    r3 = b3 / pH2^3.5 * (pCH4 * pH2O^2 - pH2^4 * pCO2 / K3) / DEN^2

    f = m.χ * m.ρcat * 1000 / 3600 # kmol/(kgcat h) -> mol/(m³_cat s)
    return (r1 * f, r2 * f, r3 * f)
end

# Volumetric source per field: 
# species (1−ε)·M_α·Σ_β ν_αβ r̃_β, kg/(m³ s)
# energy −(1−ε)·Σ_β ΔH_β r̃_β, W/m³
function reaction!(m::MethanationModel, re, yc)
    nsp = nspecies(m)
    r̃ = _rates(m, yc)
    @inbounds for α in 1:nsp
        s = 0.0
        for β in 1:3
            s += m.ν[β][α] * r̃[β]
        end
        re[α] = (1 - m.ε) * m.M[α] * s
    end
    re[nsp+1] = -(1 - m.ε) * (m.ΔHr[1] * r̃[1] + m.ΔHr[2] * r̃[2] + m.ΔHr[3] * r̃[3])
    return re
end

# Finite-difference jac
# TODO: pass persistent scratch to avoid allocations or analytic/ForwardDiff jac
function reaction_jac!(m::MethanationModel, Jre, yc)
    re = zeros(eltype(Jre), length(yc))
    reaction_jac_fd!((r, y) -> reaction!(m, r, y), Jre, yc, re)
    return Jre
end
