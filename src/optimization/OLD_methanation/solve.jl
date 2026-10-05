# Prevent BFGS hessian from being densely populated with zeros
struct ZeroHessian <: ADNLPModels.ADBackend end
ZeroHessian(args...; kwargs...) = ZeroHessian()
ADNLPModels.get_nln_nnzh(::ZeroHessian, nvar) = 0

# jac sparsity pattern
function jac_pattern(ocp::MethanationOCP, z0::Vector{Float64}; cache::String="")
    # recon must be in the key
    key = (nvars(ocp), ncons(ocp), ocp.n, ocp.s, ocp.Ne, recon(ocp))
    if !isempty(cache) && isfile(cache)
        stored = deserialize(cache)
        stored.key == key && return stored.J
        @warn "sparsity cache dimension mismatch, retracing" cache stored.key key
    end
    # Compute pattern
    cx = zeros(ncons(ocp))
    J = ADNLPModels.compute_jacobian_sparsity((c, z) -> ocp_cons!(ocp, c, z), cx, z0)
    isempty(cache) || serialize(cache, (key=key, J=J))
    return J
end

function build_ocp(ocp::MethanationOCP, x0::Vector{Float64};
    exact_hessian::Bool=false, sparsity_cache::String="", show_time::Bool=false)
    n, nc = ocp.n, ncols(ocp)
    nz = n * nc
    dm = ocp.sa.prob.dm
    nsp = nspecies(ocp.sa.prob.model)
    z0 = scale_z(ocp, x0)

    lvar = fill(-Inf, nvars(ocp))
    uvar = fill(Inf, nvars(ocp))
    Zl = reshape(view(lvar, 1:nz), n, nc)
    Zu = reshape(view(uvar, 1:nz), n, nc)
    for fi in 1:nsp # scaled ρ
        field = dm.fields[fi]
        fd = fielddof(dm, field)
        Zl[fd, :] .= field === :H2 ? ocp.x_floor_H2 : ocp.x_floor
    end
    Td = fielddof(dm, dm.fields[nsp+1]) # T path constr
    @views @. Zl[Td, :] = (ocp.Tmin - ocp.y_off[Td]) / ocp.sy[Td]
    @views @. Zu[Td, :] = (ocp.Tmax - ocp.y_off[Td]) / ocp.sy[Td]
    Zl[:, 1] .= view(z0, 1:n) # initial cond
    Zu[:, 1] .= view(z0, 1:n)
    lvar[nz+1:end] .= (ocp.Tw_min - ocp.u_off) / ocp.su
    uvar[nz+1:end] .= (ocp.Tw_max - ocp.u_off) / ocp.su

    lcon = zeros(ncons(ocp))
    ucon = zeros(ncons(ocp))

    f = z -> ocp_obj(ocp, z)
    c! = (cx, z) -> ocp_cons!(ocp, cx, z)

    gb = OCPGradient(ocp)

    exact_hessian || return ADNLPModel!(f, z0, lvar, uvar, c!, lcon, ucon;
        gradient_backend=gb, hessian_backend=ZeroHessian)

    isnothing(ocp.ad) || error("exact_hessian=true only valid with recon=:upwind1")

    # Jac from ADNLPModels (precomputed pattern -> no tracer)
    # Hess from local kernel assembly (analytic_hessian.jl)
    patJ = jac_pattern(ocp, z0; cache=sparsity_cache)
    nv, nc_ = nvars(ocp), ncons(ocp)
    tj = @elapsed jb = ADNLPModels.SparseADJacobian(nv, f, nc_, c!, patJ; x0=z0, show_time)
    show_time && println("  -> Jacobian backend: $(round(tj, digits=1)) s, $(nnz(patJ)) nnz.")
    inner = ADNLPModel!(f, z0, lvar, uvar, c!, lcon, ucon;
        gradient_backend=gb, jacobian_backend=jb, hessian_backend=ZeroHessian)
    th = @elapsed H = OCPHessian(ocp)
    show_time && println("  -> Hessian structure: $(round(th, digits=1)) s, $(nnzh(H)) nnz.")
    return KernelHessNLP(inner, H)
end

# HSL opts
function hsl_options(linear_solver::String)
    startswith(linear_solver, "ma") || return (;)
    if !isfile(HSL_jll.libhsl_path)
        @warn "libhsl not found, falling back to MUMPS" HSL_jll.libhsl_path
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

function solve_ocp(ocp::MethanationOCP, x0::Vector{Float64}; linear_solver::String="ma97",
    exact_hessian::Bool=false, sparsity_cache::String="", show_time::Bool=false, kwargs...)
    
    nlp = build_ocp(ocp, x0; exact_hessian, sparsity_cache, show_time)
    
    hess_opts = exact_hessian ? (;) : (hessian_approximation="limited-memory",)
    stats = ipopt(nlp; hess_opts..., mu_strategy="adaptive",
        acceptable_tol=1e-4, acceptable_iter=3,
        bound_relax_factor=0.0, bound_push=1e-6, bound_frac=1e-6,
        print_timing_statistics="yes",
        hsl_options(linear_solver)..., kwargs...)
    
    Z, u = unscale_z(ocp, stats.solution)
    return (Z=Z, u=u, stats=stats)
end
