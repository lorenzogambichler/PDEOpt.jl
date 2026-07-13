module Properties

include("diffusion.jl")
include("viscosity.jl") # defines wilke_phi, used by conductivity.jl
include("conductivity.jl")
include("heat_capacity.jl")
include("ergun.jl")
include("equilibrium.jl")
include("isotherms.jl")

export 
    # diffusion
    fuller_Dij, mixture_D, tsotsas_Deff,
    # viscosity 
    wilke_phi, mix_viscosity,
    # conductivity
    mix_conductivity, lambda_rad, lambda_eff_r,
    # heat capacity
    mix_cp, eff_capacity,
    # pressure drop
    ergun_dpdz,
    # equilibrium
    Keq1, Keq2, Keq3,
    # isotherms
    langmuir

end
