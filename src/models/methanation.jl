# T-dependent property correlations 
# Polynomial in T, evalpoly(T, c) = c1 + c2 T + c3 T² + ...
struct Poly{N}
    c::NTuple{N,Float64}
end
(p::Poly)(T) = evalpoly(T, p.c)

# TODO: replace with VDI polynomial coeffs (cp,α, μ_α, λ_α)
const DEFAULT_CP = ntuple(i -> Poly((( 2200.0, 1040.0, 846.0, 1996.0, 14310.0, 1040.0)[i],)), 6) # J/(kg K)
const DEFAULT_MU = ntuple(i -> Poly((( 1.1e-5, 1.8e-5, 1.5e-5, 1.3e-5, 0.9e-5, 1.8e-5)[i],)), 6) # Pa·s
const DEFAULT_LAM = ntuple(i -> Poly((( 0.034, 0.025, 0.017, 0.025, 0.18, 0.026)[i],)), 6) # W/(m K)

# Model
# 2D pseudo-homogeneous fixed-bed CO2 methanation (Bremer et al.)
# State ρ_α (CH4,CO,CO2,H2O,H2,N2) + T (fixed order)
struct MethanationModel{TF, TCP, TMU, TLA, F1, F2, F3} <: AbstractModel
    fields::TF
    # Species data
    M::NTuple{6,Float64} # molar mass, kg/mol
    υ::NTuple{6,Float64} # Fuller diffusion volume, -
    Dfac::Matrix{Float64} # constant Fuller pair factor C_ij (nsp×nsp), diag 0
    ν::NTuple{3,NTuple{6,Float64}} # stoichiometry ν[β][α]
    ΔHr::NTuple{3,Float64} # reaction enthalpy, J/mol
    # Kinetics (Xu–Froment, Table 2)
    k0::NTuple{3,Float64} # pre-exponential b_β
    Ea::NTuple{3,Float64} # activation energy, J/mol
    k0ads::NTuple{4,Float64} # adsorption pre-exp (CH4,CO,H2,H2O)
    ΔHads::NTuple{4,Float64} # adsorption enthalpy, J/mol
    χ::Float64 # effectiveness factor
    # Cat bed
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
    # Boundary
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
    nsp = length(M)
    Dfac = [i == j ? 0.0 : fuller_factor(M[i], M[j], υ[i], υ[j]) for i in 1:nsp, j in 1:nsp]
    ν = ((-1.0, 1.0, 0.0, -1.0, 3.0, 0.0), # R1: CH4 + H2O ⇌ CO + 3 H2
        (0.0, -1.0, 1.0, -1.0, 1.0, 0.0), # R2: CO + H2O ⇌ CO2 + H2
        (-1.0, 0.0, 1.0, -2.0, 4.0, 0.0)) # R3: CH4 + 2 H2O ⇌ CO2 + 4 H2
    ΔHr = (206.3e3, -41.1e3, 164.9e3)
    k0 = (5.176e15, 2.395e6, 1.250e15)
    Ea = (240.10e3, 67.13e3, 243.90e3)
    k0ads = (8.15e-4, 1.008e-4, 7.50e-9, 2.17e5)
    ΔHads = (-38.28e3, -70.65e3, -82.90e3, 88.68e3)
    return MethanationModel(fields, M, υ, Dfac, ν, ΔHr, k0, Ea, k0ads, ΔHads, χ,
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

# Property cache
struct MethanationProps{T}
    # general
    ρ::Vector{T} # mass density, kg/m³
    ρM::Vector{T} # molar density Σ ρ_α/M_α, mol/m³
    p::Vector{T} # total pressure, Pa
    x::Matrix{T} # mole frac (nsp × ncells)
    c::Matrix{T} # molar conc ρ_α/M_α (nsp × ncells)
    Deff::Matrix{T} # eff rad dispersion D^eff_r,α (nsp × ncells)
    ρcp::Vector{T} # (ρcp)^eff
    ρcp_flow::Vector{T} # ρ·cp,gas
    λeff::Vector{T} # eff rad conductivity λ^eff_r
    # temporary (overwritten each cell)
    cpα::Vector{T}
    μα::Vector{T}
    λα::Vector{T}
    Dbin::Matrix{T} # binary diffusion (nsp × nsp)
end
function MethanationProps{T}(nsp::Int, ncells::Int) where {T}
    z1() = zeros(T, ncells)
    z2() = zeros(T, nsp, ncells)
    MethanationProps{T}(z1(), z1(), z1(), z2(), z2(), z2(), z1(), z1(), z1(),
        zeros(T, nsp), zeros(T, nsp), zeros(T, nsp), zeros(T, nsp, nsp))
end
MethanationProps(nsp::Int, ncells::Int) = MethanationProps{Float64}(nsp, ncells)

# Properties, y -> props
# 1) ρ, ρM, x, c -> 2) p (id. gas) -> 3) cp, μ, λ per species -> 
# 4) cp_gas, (ρcp)^eff -> 5) λ_gas, λ^eff_r -> 6) D_ij, D_r,α, D^eff_r,α
function properties!(m::MethanationModel, props::MethanationProps, y::AbstractVector)
    nsp = nspecies(m)
    N = nsp + 1 # T
    ncells = length(props.ρ)
    Rgas = 8.314
    Tv = eltype(y)
    @inbounds for cell in 1:ncells
        base = (cell - 1) * N
        T = y[base+N]
        ρview = view(y, base+1:base+nsp)

        # composition
        ρ = zero(Tv); ρM = zero(Tv)
        for α in 1:nsp
            cα = ρview[α] / m.M[α]
            props.c[α, cell] = cα
            ρ += ρview[α]
            ρM += cα
        end
        props.ρ[cell] = ρ
        props.ρM[cell] = ρM
        for α in 1:nsp
            #props.x[α, cell] = ρM > 0 ? props.c[α, cell] / ρM : 0.0
            props.x[α, cell] = props.c[α, cell] / max(ρM, 1e-30)
        end

        # pressure (id. gas)
        # For Ergun, replace with axial sweep
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
        T175 = T^1.75
        pbar = p / 1e5
        for i in 1:nsp, j in 1:nsp
            props.Dbin[i, j] = m.Dfac[i, j] * T175 / pbar
        end
        for α in 1:nsp
            Dr = mixture_D(α, view(props.c, :, cell), props.Dbin, nsp)
            props.Deff[α, cell] = tsotsas_Deff(Dr, m.ε, m.vz, m.dp)
        end
    end
    return props
end

# Effective rad dispersion
diffusivity(m::MethanationModel, props::MethanationProps, α::Int, cell::Int) =
    props.Deff[α, cell]

# Mass-matrix coeff (ρ_α -> ε, T -> (ρcp)^eff)
capacity(m::MethanationModel, props::MethanationProps, fi::Int, c::Int) =
    fi == nspecies(m) + 1 ? props.ρcp[c] : m.ε

# Face diffusion coeff (ρ -> Deff, T -> λeff)
function face_diffusivity(m::MethanationModel, props::MethanationProps,
    fi::Int, axis::Int, o::Int, n::Int, wf::Float64)
    axis == 1 && return 0.0
    if fi == nspecies(m) + 1
        return wf * props.λeff[o] + (1 - wf) * props.λeff[n]
    else
        return wf * diffusivity(m, props, fi, o) + (1 - wf) * diffusivity(m, props, fi, n)
    end
end

# Face advection coeff (ρ -> vz, T -> ρ·cp,gas·vz)
function face_velocity(m::MethanationModel, props::MethanationProps,
    fi::Int, axis::Int, o::Int, n::Int, wf::Float64)
    axis == 1 || return 0.0
    if fi == nspecies(m) + 1
        ρcp_flow = wf * props.ρcp_flow[o] + (1 - wf) * props.ρcp_flow[n]
        return ρcp_flow * m.vz
    else
        return m.vz
    end
end

# Reaction (Xu-Froment LHHW) 
# Volumetric rates r̃_β in mol/(m³_cat s) from local yc
function _rates(m::MethanationModel, yc)
    T = yc[nspecies(m)+1]
    RT = 8.314 * T
    # partial pressures in bar, p_α = (ρ_α/M_α)·R·T (ideal gas)
    pCH4 = yc[1] / m.M[1] * RT / 1e5
    pCO  = yc[2] / m.M[2] * RT / 1e5
    pCO2 = yc[3] / m.M[3] * RT / 1e5
    pH2O = yc[4] / m.M[4] * RT / 1e5
    pH2  = max(yc[5] / m.M[5] * RT / 1e5, 1e-12)

    b1 = m.k0[1] * exp(-m.Ea[1] / RT)
    b2 = m.k0[2] * exp(-m.Ea[2] / RT)
    b3 = m.k0[3] * exp(-m.Ea[3] / RT)
    BCH4 = m.k0ads[1] * exp(-m.ΔHads[1] / RT)
    BCO  = m.k0ads[2] * exp(-m.ΔHads[2] / RT)
    BH2  = m.k0ads[3] * exp(-m.ΔHads[3] / RT)
    BH2O = m.k0ads[4] * exp(-m.ΔHads[4] / RT)
    DEN = 1 + BCO * pCO + BH2 * pH2 + BCH4 * pCH4 + BH2O * pH2O / pH2

    K1 = Keq1(T); K2 = Keq2(T); K3 = Keq3(T)
    r1 = b1 / pH2^2.5 * (pCH4 * pH2O - pH2^3 * pCO / K1) / DEN^2
    r2 = b2 / pH2 * (pCO * pH2O - pH2 * pCO2 / K2) / DEN^2
    r3 = b3 / pH2^3.5 * (pCH4 * pH2O^2 - pH2^4 * pCO2 / K3) / DEN^2

    f = m.χ * m.ρcat * 1000 / 3600 # kmol/(kgcat h) -> mol/(m³_cat s)
    return (r1 * f, r2 * f, r3 * f)
end

# Volumetric source per field
# ρ -> (1−ε)·M_α·Σ_β ν_αβ r̃_β in kg/(m³ s)
# T -> −(1−ε)·Σ_β ΔH_β r̃_β in W/m³
function reaction!(m::MethanationModel, re, yc)
    nsp = nspecies(m)
    r̃ = _rates(m, yc)
    @inbounds for α in 1:nsp
        s = zero(eltype(re))
        for β in 1:3
            s += m.ν[β][α] * r̃[β]
        end
        re[α] = (1 - m.ε) * m.M[α] * s
    end
    re[nsp+1] = -(1 - m.ε) * (m.ΔHr[1] * r̃[1] + m.ΔHr[2] * r̃[2] + m.ΔHr[3] * r̃[3])
    return re
end

# Finite diff jac
# TODO: analytic/ForwardDiff jac
function reaction_jac!(m::MethanationModel, Jre, yc, fbase, fpert, ypert)
    reaction_jac_fd!((r, y) -> reaction!(m, r, y), Jre, yc, fbase, fpert, ypert)
    return Jre
end
