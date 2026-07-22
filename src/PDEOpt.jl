module PDEOpt

include("mesh/structured_grid.jl")
include("properties/properties.jl")
include("models/models.jl")
include("problem/problem.jl")
#include("optimization/nlp.jl")
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
#using .NLP

# FEM assembly
export assemble_mass!, assemble_diffusion!, assemble_interface!, assemble_advection!,
    assemble_advection_interface!, assemble_advection_boundary!, assemble_inflow!,
    assemble_reaction!, getdistance, getdiameter

# FVM assembly
export advection_flux, diffusion_flux,
    assemble_transport!, assemble_inlet_outlet!, assemble_wall!,
    assemble_react!, assemble_boundary!, StateAssembly

# Tensor grid connectivity, geometry, DOF map, sparsitiy pattern
export AbstractGrid, AbstractGeometry,
    StructGrid2D, StructGrid1D, ncells, cellindex, cellij,
    FaceSet, interior_faces, BoundaryFaceSet, boundary_faces,
    Geometry1D, Geometry2D, cellvolume, cellcentroid, FaceGeometry, BoundaryFaceGeometry,
    DofMap, ndof, fieldindex, dof, celldof, fielddof, sparsity_pattern

# Physics models
export AbstractModel, PlugFlowModel, PSAModel, MethanationModel, MethanationProps,
    Reaction, Arrhenius, AdsorptionLDF,
    fields, transported, state_dependent, diffusivity, velocity, nspecies,
    reaction!, reaction_jac!, reaction_jac_fd!, reaction_source, properties!,
    inlet_value, wall_coeff, wall_field, inlet_mod, wall_mod

# Assembled-problem container
export ProblemCache

# Nonlinear solvers
export newton!, chord!, shamanskii

# ODE solvers
export cn_solve!, CNCache, set_ic!

# NLP
#export OptCache, build_nlp

end
