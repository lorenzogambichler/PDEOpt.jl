# Full-space DtO (ADNLPModels for sparse AD) for MethanationModel

# Scaling to [0,1]:
# ρ_α = (c_tot,in·M_α)·ρ̂_α
# T = T_min + (Tmax − T_min)·T̂
# Tw = Tw_min + (Tw_max − Tw_min)·û

using PDEOpt
using SparseArrays
using LinearAlgebra
using ADNLPModels
using NLPModelsIpopt
using HSL_jll

# Prevent BFGS hessian from being densely populated with zeros
struct ZeroHessian <: ADNLPModels.ADBackend end
ZeroHessian(args...; kwargs...) = ZeroHessian()
ADNLPModels.get_nln_nnzh(::ZeroHessian, nvar) = 0

struct MethanationOCP{TSA}
    sa::TSA
    Δt::Float64
    N::Int
    n::Int
    y0::Vector{Float64}
    # Bounds
    Tw_min::Float64
    Tw_max::Float64
    Tmax::Float64
    T_min::Float64
    x_floor::Float64
    x_floor_H2::Float64
    # Scaling
    sy::Vector{Float64} # state scale
    y_off::Vector{Float64} # state offset (T only)
    sc::Vector{Float64} # constraint row scale
    su::Float64 # control scale
    u_off::Float64 # control offset
    # Objective
    γ::Float64
    co2_dofs::Vector{Int}
    co2_w::Vector{Float64} # vz·A
    co2_in::Float64
    caches::Dict{DataType,Any}
end

# Σ_j vz·A_j·g(r_j)·ρ_CO2,in
co2_inflow(sa, t::Real=0.0) =
    sum(@view sa.prob.f_in[fielddof(sa.prob.dm, :CO2)]) * inlet_mod(sa.prob.model, :CO2, t)

# Reference scales, sy/y_off (variables), sc (constraint rows)
function _scales(sa, y0::Vector{Float64}, T_min::Float64, Tmax::Float64)
    prob = sa.prob
    dm, m, grid, geom = prob.dm, prob.model, prob.grid, prob.geom
    nsp = nspecies(m)
    ctot = sum(y0[dof(dm, 1, α)] / m.M[α] for α in 1:nsp) # inlet conc
    ΔT = Tmax - T_min
    ρcp_ref = (1 - m.ε) * m.ρcat * m.cp_cat

    sy = zeros(ndof(dm))
    y_off = zeros(ndof(dm))
    sc = zeros(ndof(dm))
    for c in 1:ncells(grid)
        i, j = cellij(grid, c)
        V = cellvolume(geom, i, j)
        for α in 1:nsp
            d = dof(dm, c, α)
            sy[d] = ctot * m.M[α]
            sc[d] = 1 / (m.ε * V * sy[d])
        end
        d = dof(dm, c, nsp + 1)
        sy[d] = ΔT
        y_off[d] = T_min
        sc[d] = 1 / (ρcp_ref * V * ΔT)
    end
    return sy, y_off, sc
end

function MethanationOCP(sa, Δt::Float64, N::Int, y0::Vector{Float64};
    Tw_min, Tw_max, Tmax,
    T_min=min(Tw_min, minimum(@view y0[fielddof(sa.prob.dm, :T)])) - 50.0,
    x_floor=1e-8, x_floor_H2=1e-3, γ=0.0, co2_in=nothing)

    dm, m = sa.prob.dm, sa.prob.model
    sy, y_off, sc = _scales(sa, y0, Float64(T_min), Float64(Tmax))

    bfs_out, bfg_out = sa.ebnd[2]
    co2_dofs = [dof(dm, c, :CO2) for c in bfs_out.cells]
    co2_w = velocity(m, 1) .* copy(bfg_out.area)
    Fin = isnothing(co2_in) ? co2_inflow(sa) : Float64(co2_in)

    return MethanationOCP(sa, Δt, N, ndof(dm), copy(y0),
        Tw_min, Tw_max, Tmax, T_min, x_floor, x_floor_H2,
        sy, y_off, sc, Tw_max - Tw_min, Tw_min,
        γ, co2_dofs, co2_w, Fin, Dict{DataType,Any}())
end

function _tcache(ocp::MethanationOCP, ::Type{T}) where {T}
    p0 = ocp.sa.prob
    tosp(A) = SparseMatrixCSC(A.m, A.n, copy(A.colptr), copy(A.rowval), zeros(T, length(A.nzval)))
    nf = length(p0.dm.fields)
    prob = ProblemCache(tosp(p0.M), tosp(p0.K), tosp(p0.Jr),
        zeros(T, ocp.n), zeros(T, nf), zeros(T, nf, nf),
        T.(p0.f_in), T.(p0.f_wall), p0.grid, p0.geom, p0.dm, p0.model)
    props = MethanationProps{T}(nspecies(p0.model), ncells(p0.grid))
    saT = StateAssembly(prob, props, ocp.sa.K_bc, ocp.sa.dirs, ocp.sa.ebnd)
    buf = (ymid=zeros(T, ocp.n), Δy=zeros(T, ocp.n), Kv=zeros(T, ocp.n), b=zeros(T, ocp.n))
    return (prob=prob, sa=saT, buf=buf)
end

gettcache(ocp::MethanationOCP, ::Type{T}) where {T} = get!(() -> _tcache(ocp, T), ocp.caches, T)

# Physical to scaled
function scale_z(ocp::MethanationOCP, x::AbstractVector)
    n, N = ocp.n, ocp.N
    z = similar(x, Float64)
    Z = reshape(view(z, 1:n*N), n, N)
    X = reshape(view(x, 1:n*N), n, N)
    @. Z = (X - ocp.y_off) / ocp.sy
    @views @. z[n*N+1:end] = (x[n*N+1:end] - ocp.u_off) / ocp.su
    return z
end

# Scaled to physical
function unscale_z(ocp::MethanationOCP, z::AbstractVector)
    n, N = ocp.n, ocp.N
    Y = reshape(z[1:n*N], n, N)
    @. Y = ocp.y_off + ocp.sy * Y
    u = ocp.u_off .+ ocp.su .* z[n*N+1:end]
    return Y, u
end

function ocp_cons!(ocp::MethanationOCP, cx, z)
    c = gettcache(ocp, eltype(z))
    return _residual!(cx, z, ocp, c.prob, c.sa, c.buf)
end

function _residual!(cx, z, ocp::MethanationOCP, prob, saT, buf)
    n, N, Δt = ocp.n, ocp.N, ocp.Δt
    Y = reshape(view(z, 1:n*N), n, N)
    u = view(z, n*N+1:n*N+N)
    dm, m = prob.dm, prob.model
    ymid, Δy, Kv, b = buf.ymid, buf.Δy, buf.Kv, buf.b
    sy, y_off, sc = ocp.sy, ocp.y_off, ocp.sc

    # Implicit midpoint (phys state)
    for k in 1:N-1
        yk = view(Y, :, k)
        ykp1 = view(Y, :, k+1)
        @. ymid = y_off + sy * 0.5 * (yk + ykp1)
        @. Δy = sy * (ykp1 - yk)
        umid = ocp.u_off + ocp.su * 0.5 * (u[k] + u[k+1])
        tmid = (k - 0.5) * Δt

        saT(ymid)
        assemble_react!(prob, ymid, Val(false))

        fill!(b, zero(eltype(b)))
        for field in dm.fields
            fd = fielddof(dm, field)
            md = inlet_mod(m, field, tmid)
            @views @. b[fd] += prob.f_in[fd] * md
        end
        @. b += prob.f_wall * umid

        cb = view(cx, (k-1)*n+1:k*n)
        mul!(cb, prob.M, Δy)
        mul!(Kv, prob.K, ymid)
        @. cb = (cb + Δt * Kv - Δt * prob.r - Δt * b) * sc # row scaling
    end
    return cx
end

# min mean CO2 frac over [0,tf], equiv to max CO2 conversion
function ocp_obj(ocp::MethanationOCP, z)
    n, N = ocp.n, ocp.N
    Y = reshape(view(z, 1:n*N), n, N)
    u = view(z, n*N+1:n*N+N)

    J_conv = zero(eltype(z))
    for k in 1:N
        co2_out = zero(eltype(z))
        for j in eachindex(ocp.co2_dofs)
            d = ocp.co2_dofs[j]
            co2_out += ocp.co2_w[j] * (ocp.y_off[d] + ocp.sy[d] * Y[d, k])
        end
        wf = (k==1 || k==N) ? 0.5 : 1.0 # trapezoidal quad
        J_conv += wf * (co2_out/ocp.co2_in)
    end

    J_reg = zero(eltype(z))
    for k in 1:N-1
        J_reg += ((u[k+1] - u[k]) * (N - 1))^2
    end

    return 1/(N - 1) * (J_conv + ocp.γ * J_reg)
end

co2_conversion(ocp::MethanationOCP, Y::AbstractMatrix) =
    [1 - sum(ocp.co2_w[j] * Y[ocp.co2_dofs[j], k] for j in eachindex(ocp.co2_dofs)) / ocp.co2_in
     for k in 1:size(Y, 2)]

function build_ocp(ocp::MethanationOCP, x0::Vector{Float64})
    n, N = ocp.n, ocp.N
    nvar = n * N + N
    ncon = n * (N - 1)
    dm = ocp.sa.prob.dm
    nsp = nspecies(ocp.sa.prob.model)
    z0 = scale_z(ocp, x0)

    lvar = fill(-Inf, nvar)
    uvar = fill(Inf, nvar)
    Yl = reshape(view(lvar, 1:n*N), n, N)
    Yu = reshape(view(uvar, 1:n*N), n, N)
    for fi in 1:nsp
        field = dm.fields[fi]
        fd = fielddof(dm, field)
        Yl[fd, :] .= field === :H2 ? ocp.x_floor_H2 : ocp.x_floor # scaled ρ̂ (mole frac)
    end
    Td = fielddof(dm, dm.fields[nsp+1])
    @views @. Yl[Td, :] = (ocp.T_min - ocp.y_off[Td]) / ocp.sy[Td]
    @views @. Yu[Td, :] = (ocp.Tmax - ocp.y_off[Td]) / ocp.sy[Td]
    Yl[:, 1] .= view(z0, 1:n) # initial condition
    Yu[:, 1] .= view(z0, 1:n)
    lvar[n*N+1:end] .= (ocp.Tw_min - ocp.u_off) / ocp.su
    uvar[n*N+1:end] .= (ocp.Tw_max - ocp.u_off) / ocp.su

    lcon = zeros(ncon)
    ucon = zeros(ncon)

    return ADNLPModel!(z -> ocp_obj(ocp, z), z0, lvar, uvar,
        (cx, z) -> ocp_cons!(ocp, cx, z), lcon, ucon;
        hessian_backend = ZeroHessian)
end

# HSL opts
function hsl_options(linear_solver::String)
    startswith(linear_solver, "ma") || return (;)
    if !isfile(HSL_jll.libhsl_path)
        @warn "libhsl not found, falling back to MUMPS" HSL_jll.libhsl_path
        return (;)
    end
    opts = (hsllib=HSL_jll.libhsl_path, linear_solver=linear_solver)
    return linear_solver == "ma97" ?
           merge(opts, (ma97_order="metis", ma97_scaling="none")) : opts
end

function solve_ocp(ocp::MethanationOCP, x0::Vector{Float64};
    linear_solver::String="ma97", kwargs...)
    nlp = build_ocp(ocp, x0)
    stats = ipopt(nlp; hessian_approximation="limited-memory",
        bound_relax_factor=0.0, hsl_options(linear_solver)..., kwargs...)
    Y, u = unscale_z(ocp, stats.solution)
    return (Y=Y, u=u, stats=stats)
end
