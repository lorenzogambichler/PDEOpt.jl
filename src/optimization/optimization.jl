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
using ..Collocation: AbstractCollocationMethod # not exported

abstract type AbstractOCP end

include("nlp/nlp.jl")

include("ocps/methanation/methanation.jl")
#include("ocps/plug_flow/plug_flow.jl")

export AbstractOCP,
    # Layout
    CollocationLayout, ncols, elsize, nvars, ncons,
    initial_range, stage_range, control_range, cons_range, left_range,
    initial, stage, control, left, stagecol, element_times, stage_times,
    pack, pack!, unpack,
    # Scaling, bounds
    Scaling, scale_x, scale_x!, unscale_x, unscale_x!, scale_bounds,
    Bounds, assemble_bound_vecs,
    # Residual
    CollocationResidual, retype,
    # Objective
    Objective,
    # Methanation
    MethanationOCP, MethanationDynamics, MethanationCost,
    methanation_layout, methanation_scaling, methanation_bounds,
    methanation_residual, methanation_objective,
    ZeroHessian, ObjectiveGradient, hsl_options, build_nlp, solve_ocp,
    inflow, co2_inflow, outlet_conv, mean_conv
end
