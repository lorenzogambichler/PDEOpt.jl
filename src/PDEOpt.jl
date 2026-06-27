module PDEOpt

include("assembly/assemble_dg.jl")
include("solvers/nonlinear.jl")
include("ode_solvers/crank_nicolson.jl")

using .AssembleDG
using .NonlinearSolvers
using .CrankNicolson

# FEM assembly
export assemble_mass!, assemble_diffusion!, assemble_interface!, assemble_advection!,
    assemble_advection_interface!, assemble_advection_boundary!, assemble_inflow!,
    assemble_reaction!, getdistance, getdiameter

# Nonlinear + ODE solvers
export newton!, quasi_newton!, cn_solve!

end
