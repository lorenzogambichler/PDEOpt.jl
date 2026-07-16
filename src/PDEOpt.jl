module PDEOpt

include("mesh/structured_grid.jl")
include("properties/properties.jl")
include("models/models.jl")
include("problem/problem.jl")
function assemble_mass! end
include("assembly/assemble_dg.jl")
include("assembly/assemble_fvm.jl")
include("solvers/nonlinear.jl")
include("timestepping/crank_nicolson.jl")

using .StructuredMesh
using .Properties
using .Models
using .Problem
using .AssembleDG
using .AssembleFVM
using .NonlinearSolvers
using .CrankNicolson

# FEM assembly
export assemble_mass!, assemble_diffusion!, assemble_interface!, assemble_advection!,
    assemble_advection_interface!, assemble_advection_boundary!, assemble_inflow!,
    assemble_reaction!, getdistance, getdiameter

# Tensor grid connectivity, geometry, DOF map, sparsitiy pattern
export AbstractGrid, AbstractGeometry,
    StructGrid2D, ncells, cellindex, cellij,
    FaceSet, interior_faces, BoundaryFaceSet, boundary_faces,
    Geometry, cellvolume, cellcentroid, FaceGeometry, BoundaryFaceGeometry,
    DofMap, ndof, fieldindex, dof, celldof, fielddof, sparsity_pattern

# Physics models
export AbstractModel, PlugFlowModel, PSAModel, MethanationModel, MethanationProps,
    Reaction, Arrhenius, AdsorptionLDF,
    fields, transported, state_dependent, diffusivity, velocity, nspecies,
    reaction!, reaction_jac!, reaction_jac_fd!, properties!,
    inlet_value, wall_coeff, wall_field, inlet_mod, wall_mod

# Assembled-problem container
export ProblemCache

# FVM assembly
export advection_flux, diffusion_flux,
    assemble_transport!, assemble_inlet_outlet!, assemble_wall!,
    assemble_react!, assemble_boundary!, StateAssembly

# Nonlinear solvers
export newton!, chord!, shamanskii

# ODE solvers
export cn_solve!, CNCache

end
