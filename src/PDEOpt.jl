module PDEOpt

include("mesh/structured_grid.jl")
include("assembly/assemble_dg.jl")
include("assembly/assemble_fvm.jl")
include("solvers/nonlinear.jl")
include("ode_solvers/crank_nicolson.jl")

using .StructuredMesh
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

# FVM assembly
export advection_flux, diffusion_flux, reaction!

# Nonlinear solvers
export newton!, quasi_newton!, FrozenNewton

# ODE solvers
export cn_solve!, CNCache

end
