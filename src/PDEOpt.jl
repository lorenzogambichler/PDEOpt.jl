module PDEOpt

include("mesh/structured_grid.jl")
include("properties/properties.jl")
include("models/adr.jl")
include("assembly/assemble_dg.jl")
include("assembly/assemble_fvm.jl")
include("solvers/nonlinear.jl")
include("timestepping/crank_nicolson.jl")

using .StructuredMesh
using .Properties
using .Models
using .AssembleDG
using .AssembleFVM
using .NonlinearSolvers
using .CrankNicolson

# FEM assembly
export assemble_mass!, assemble_diffusion!, assemble_interface!, assemble_advection!,
    assemble_advection_interface!, assemble_advection_boundary!, assemble_inflow!,
    assemble_reaction!, getdistance, getdiameter

# Tensor grid connectivity, geometry, DOF map, sparsitiy pattern
export StructuredGrid, ncells, cellindex, cellij,
    FaceSet, interior_faces, BoundaryFaceSet, boundary_faces,
    Geometry, cellvolume, cellcentroid, FaceGeometry, BoundaryFaceGeometry,
    DofMap, ndof, fieldindex, dof, celldof, fielddof, sparsity_pattern

# Physics models
export AbstractModel, ADRModel, PSAModel, Reaction, Arrhenius, LHHW, AdsorptionLDF,
    TDependentD, inlet_mod

# FVM assembly
export advection_flux, diffusion_flux,
    assemble_global, assemble_transport!, assemble_inlet_outlet!, assemble_wall!,
    assemble_react!, assemble_boundary!

# Nonlinear solvers
export newton!, chord!, shamanskii

# ODE solvers
export cn_solve!, CNCache

end
