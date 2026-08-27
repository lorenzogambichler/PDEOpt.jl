# Full-space DtO (ADNLPModels for sparse AD) for MethanationModel
# Radau IIA (OCFE)

# z = [y0; Y11; Y12; Y13; Y21; Y22; …; YNe2; YNe3; u1; …; uNe]

using PDEOpt
using SparseArrays
using LinearAlgebra
using ADNLPModels
using NLPModelsIpopt
using HSL_jll
using Serialization
using Printf
import ForwardDiff

include("initial_guess.jl") # bisection guess from CN forward solves (standalone)
include("ocp.jl") # MethanationOCP (layout, indexing, scaling)
include("residual.jl") # collocation residual (global assembly + kernel form)
include("objective.jl") # objective + analytic gradient
include("postprocess.jl") # inlet/outlet flows, conversion
include("analytic_hessian.jl") # exact hessian from local kernels
include("solve.jl") # NLP construction + Ipopt driver
