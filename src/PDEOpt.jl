module PDEOpt

include("diagnostics/diagnostics.jl")
include("mesh/structured_grid.jl")
include("properties/properties.jl")
include("models/models.jl")
include("problem/problem.jl")
function assemble_mass! end
include("assembly/assemble_dg.jl")
include("assembly/assemble_fvm.jl")
include("solvers/nonlinear.jl")
function dense_output end
include("timestepping/crank_nicolson.jl")
include("timestepping/collocation.jl")
include("io/output.jl")
include("io/vtkread.jl")
include("optimization/optimization.jl")

using .Diagnostics
using .StructuredMesh
using .Properties
using .Models
using .Problem
using .AssembleDG
using .AssembleFVM
using .NonlinearSolvers
using .CrankNicolson
using .Collocation
using .Output
using .VTKRead
using .Optimization

# FEM assembly
export assemble_mass!, assemble_diffusion!, assemble_interface!, assemble_advection!,
    assemble_advection_interface!, assemble_advection_boundary!, assemble_inflow!,
    assemble_reaction!, getdistance, getdiameter

# FVM assembly
export advection_flux, diffusion_flux,
    assemble_transport!, assemble_inlet_outlet!, assemble_wall!,
    assemble_react!, assemble_boundary!, StateAssembly,
    face_flux, bnd_energy_kernel,
    vanalbada, antidiff_eps, AntiDiffusion, assemble_antidiffusion!

# Grid connectivity, geometry, DOF map, sparsitiy pattern
export AbstractGrid, AbstractGeometry,
    StructGrid2D, StructGrid1D, ncells, cellindex, cellij,
    FaceSet, interior_faces, BoundaryFaceSet, boundary_faces,
    Geometry1D, Geometry2D, cellvolume, cellcentroid, FaceGeometry, BoundaryFaceGeometry,
    UpwindStencil, graded_nodes,
    DofMap, ndof, fieldindex, dof, celldof, fielddof, sparsity_pattern

# Physics models
export AbstractModel, PlugFlowModel, PSAModel, MethanationModel, MethanationProps,
    Reaction, Arrhenius, AdsorptionLDF,
    fields, transported, state_dependent, diffusivity, velocity, nspecies,
    reaction!, reaction_jac!, reaction_jac_fd!, reaction_source, properties!, props_cell,
    nspecies_val, nfields_val, reaction_cell, capacity_cell, capacity,
    inlet_value, wall_coeff, wall_field, inlet_mod, wall_mod

# Assembled-problem container
export ProblemCache

# Nonlinear solvers
export newton!, chord!, shamanskii

# ODE solvers
export cn_solve!, CNCache, set_ic!, resample

# Collocation tableaus
export RadauIIA, nstages, stage_time, quadrature_weights

# Dense output
export dense_output

# Output
export write_vtk, write_control

# Reading results back
export VTRFile, VTRSeries, read_vtr, read_pvd, read_series, cellcentres

# Optimization (collocation transcription + OCPs)
export AbstractOCP,
    CollocationLayout, ncols, elsize, nvars, ncons,
    initial_range, stage_range, control_range, cons_range, left_range,
    initial, stage, control, left, stagecol, element_times, stage_times,
    pack, pack!, unpack,
    Scaling, scale_x, scale_x!, unscale_x, unscale_x!, scale_bounds,
    Bounds, assemble_bound_vecs,
    CollocationResidual, retype,
    Objective,
    MethanationOCP, MethanationDynamics, MethanationCost,
    methanation_layout, methanation_scaling, methanation_bounds,
    methanation_residual, methanation_objective,
    ZeroHessian, ObjectiveGradient, hsl_options, build_nlp, solve_ocp,
    inflow, co2_inflow, outlet_conv, mean_conv

# Profiling
export MemTrace, memtrace,
    IpoptTiming, parse_ipopt_timing, read_ipopt_timing, print_timing

end
