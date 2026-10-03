# Full-space DtO (ADNLPModels for sparse AD) for MethanationModel
# Radau IIA (OCFE)

# z = [y0; Y11; Y12; Y13; Y21; Y22; …; YNe2; YNe3; u1; …; uNe]

# TODO: maybe make this a package instead of using Revise?

using PDEOpt
using SparseArrays
using LinearAlgebra
using ADNLPModels
using NLPModelsIpopt
using HSL_jll
using Serialization
using Printf
using Revise
import ForwardDiff

includet("initial_guess.jl") # bisection guess from CN forward solves (standalone)
includet("ocp.jl") # MethanationOCP (layout, indexing, scaling)
includet("residual.jl") # collocation residual (global assembly + kernel form)
includet("objective.jl") # objective + analytic gradient
includet("postprocess.jl") # inlet/outlet flows, conversion
includet("analytic_hessian.jl") # exact hessian from local kernels
includet("solve.jl") # NLP construction + Ipopt driver
