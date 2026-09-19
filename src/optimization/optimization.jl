module Optimization

using SparseArrays
using LinearAlgebra
using Serialization
using ADNLPModels
using NLPModelsIpopt
using HSL_jll
using ..Models
using ..Problem
using ..StructuredMesh
using ..AssembleFVM
using ..Collocation

abstract type AbstractOCP end

include("objective.jl")
include("ocp.jl")
include("residual.jl")
include("initial_guess.jl")
include("solve.jl")

export AbstractOCP,
    # Objective
    ocp_obj, obj_grad, OCPGradient,
    # OCP
    MethanationOCP, recon, _diagindex, ncols, stagecol, leftcol, nvars,
    ncons, timegrid, control_times, _scales, scale_z, unscale_z,
    # Residual
    _tcache, gettcache, ocp_cons!, _residual!, _residual_kernels!,
    # Initial guess
    bisect_shape, twshape, bisect_const,
    # Solve
    ZeroHessian, jac_pattern, build_ocp, hsl_options, solve_ocp
end