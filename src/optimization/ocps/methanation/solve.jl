# Prevent BFGS hessian from being densely populated with zeros
struct ZeroHessian <: ADNLPModels.ADBackend end
ZeroHessian(args...; kwargs...) = ZeroHessian()
ADNLPModels.get_nln_nnzh(::ZeroHessian, nvar) = 0

# Analytic gradient from Objective.cost_grad!
struct ObjectiveGradient{TO} <: ADNLPModels.ADBackend
    obj::TO # Objective
end
ADNLPModels.gradient!(b::ObjectiveGradient, g, f, x) = b.obj.cost_grad!(g, x)

function hsl_options(linear_solver::String)
    startswith(linear_solver, "ma") || return (;)
    if !isfile(HSL_jll.libhsl_path)
        @warn "libhsl not found, using MUMPS" HSL_jll.libhsl_path
        return (;)
    end
    opts = (hsllib=HSL_jll.libhsl_path, linear_solver=linear_solver)
    if linear_solver == "ma97"
        return merge(opts, (ma97_order="metis", ma97_scaling="mc64", ma97_nemin=8))
        # keep nemin low when ram is limiting factor
    elseif linear_solver == "ma57"
        return merge(opts, (ma57_order="metis", ma57_scaling="none", ma57_nemin=8))
    else
        return opts
    end
end

# x0 physical -> NLP scaled
function build_nlp(ocp::MethanationOCP, x0::AbstractVector; hessian::Symbol=:lbfgs)
    hessian === :lbfgs || error("hessian=:$hessian not implemented yet, use :lbfgs")
    lay, scl = ocp.lay, ocp.scl
    length(x0) == nvars(lay) ||
        throw(DimensionMismatch("x0 has length $(length(x0)), expected $(nvars(lay))"))

    x0sc = scale_x(lay, scl, x0)
    lvar, uvar = scale_bounds(lay, scl, assemble_bound_vecs(lay, ocp.bnd, ocp.y0)...)
    lcon = zeros(ncons(lay))
    ucon = zeros(ncons(lay))
    c! = (cx, x) -> ocp.rsd(cx, x) # ADNLPModels needs function (not functor)

    return ADNLPModel!(ocp.obj.cost, x0sc, lvar, uvar, c!, lcon, ucon;
        gradient_backend=ObjectiveGradient(ocp.obj), hessian_backend=ZeroHessian)
end

function solve_ocp(ocp::MethanationOCP, x0::AbstractVector; linear_solver::String="ma97", 
    hessian::Symbol=:lbfgs, kwargs...)
    nlp = build_nlp(ocp, x0; hessian=hessian)
    # kwargs override
    opts = merge((hessian_approximation="limited-memory",
            mu_strategy="adaptive", acceptable_tol=1e-4, acceptable_iter=3,
            bound_relax_factor=0.0, bound_push=1e-6, bound_frac=1e-6,
            print_timing_statistics="yes"),
        hsl_options(linear_solver), values(kwargs))
    stats = ipopt(nlp; opts...)

    # scaled sol -> physical trajectory
    x = unscale_x(ocp.lay, ocp.scl, stats.solution)
    Y, U = unpack(ocp.lay, x)
    return (Y=Y, U=U, t=stage_times(ocp.lay), tu=element_times(ocp.lay), stats=stats)
end
