# Full-space DtO (ADNLPModels for sparse AD) for MethanationModel
# Radau IIA collocation on finite elements, differentiation-matrix form.
#
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

struct MethanationOCP{TSA,TTab}
    sa::TSA
    tab::TTab # RadauIIA tableau
    Δt::Float64 # element length
    Ne::Int # number of finite elements
    s::Int # collocation points per element
    n::Int # spatial dof
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
    # Support sets. M and f_wall are allocated with K's pattern / full length so that
    # CN can share it, but assemble_mass! only writes the diagonal and f_wall only the
    # wall cells. Touching just the true support keeps the traced Jacobian from
    # inheriting those structural zeros as spurious dependencies.
    mdiag::Vector{Int} # nzval index of M[d,d]
    wall_dofs::Vector{Int} # support of f_wall
    caches::Dict{DataType,Any}
end

# nzval positions of the diagonal of a CSC matrix
function _diagindex(A::SparseMatrixCSC)
    idx = Vector{Int}(undef, size(A, 1))
    rv, cp = rowvals(A), A.colptr
    for d in eachindex(idx)
        p = searchsortedfirst(view(rv, cp[d]:cp[d+1]-1), d)
        (p <= cp[d+1] - cp[d] && rv[cp[d]+p-1] == d) ||
            error("mass matrix has no stored diagonal entry at dof $d")
        idx[d] = cp[d] + p - 1
    end
    return idx
end

# Trajectory columns: y_0 followed by s stage vectors per element.
# The left endpoint of element k is the last stage of element k−1 (stiff accuracy).
ncols(ocp::MethanationOCP) = ocp.s * ocp.Ne + 1
stagecol(ocp::MethanationOCP, k::Int, i::Int) = 1 + (k - 1) * ocp.s + i
leftcol(ocp::MethanationOCP, k::Int) = 1 + (k - 1) * ocp.s
nvars(ocp::MethanationOCP) = ocp.n * ncols(ocp) + ocp.Ne
ncons(ocp::MethanationOCP) = ocp.n * ocp.s * ocp.Ne

# Time of each trajectory column: t=0, then the collocation points of each element
function timegrid(ocp::MethanationOCP)
    t = zeros(ncols(ocp))
    for k in 1:ocp.Ne, i in 1:ocp.s
        t[stagecol(ocp, k, i)] = stage_time(ocp.tab, k, i, ocp.Δt)
    end
    return t
end

# Left edge of each element — the control is piecewise constant on [t_{k-1}, t_k)
control_times(ocp::MethanationOCP) = [(k - 1) * ocp.Δt for k in 1:ocp.Ne]

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

function MethanationOCP(sa, tab, Δt::Float64, Ne::Int, y0::Vector{Float64};
    Tw_min, Tw_max, Tmax,
    T_min=min(Tw_min, minimum(@view y0[fielddof(sa.prob.dm, :T)])) - 50.0,
    x_floor=1e-8, x_floor_H2=1e-3, γ=0.0, co2_in=nothing)

    dm, m = sa.prob.dm, sa.prob.model
    sy, y_off, sc = _scales(sa, y0, Float64(T_min), Float64(Tmax))

    bfs_out, bfg_out = sa.ebnd[2]
    co2_dofs = [dof(dm, c, :CO2) for c in bfs_out.cells]
    co2_w = velocity(m, 1) .* copy(bfg_out.area)
    Fin = isnothing(co2_in) ? co2_inflow(sa) : Float64(co2_in)

    mdiag = _diagindex(sa.prob.M)
    nnz(dropzeros(copy(sa.prob.M))) == ndof(dm) ||
        error("mass matrix is not diagonal — the diagonal-product residual is invalid")
    wall_dofs = findall(!iszero, sa.prob.f_wall)

    return MethanationOCP(sa, tab, Δt, Ne, nstages(tab), ndof(dm), copy(y0),
        Tw_min, Tw_max, Tmax, T_min, x_floor, x_floor_H2,
        sy, y_off, sc, Tw_max - Tw_min, Tw_min,
        γ, co2_dofs, co2_w, Fin, mdiag, wall_dofs, Dict{DataType,Any}())
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
    buf = (yp=zeros(T, ocp.n), w=zeros(T, ocp.n), Kv=zeros(T, ocp.n), b=zeros(T, ocp.n))
    return (prob=prob, sa=saT, buf=buf)
end

gettcache(ocp::MethanationOCP, ::Type{T}) where {T} = get!(() -> _tcache(ocp, T), ocp.caches, T)

# Physical to scaled
function scale_z(ocp::MethanationOCP, x::AbstractVector)
    nz = ocp.n * ncols(ocp)
    z = similar(x, Float64)
    Z = reshape(view(z, 1:nz), ocp.n, ncols(ocp))
    X = reshape(view(x, 1:nz), ocp.n, ncols(ocp))
    @. Z = (X - ocp.y_off) / ocp.sy
    @views @. z[nz+1:end] = (x[nz+1:end] - ocp.u_off) / ocp.su
    return z
end

# Scaled to physical
function unscale_z(ocp::MethanationOCP, z::AbstractVector)
    nz = ocp.n * ncols(ocp)
    Z = reshape(z[1:nz], ocp.n, ncols(ocp))
    @. Z = ocp.y_off + ocp.sy * Z
    u = ocp.u_off .+ ocp.su .* z[nz+1:end]
    return Z, u
end

function ocp_cons!(ocp::MethanationOCP, cx, z)
    c = gettcache(ocp, eltype(z))
    return _residual!(cx, z, ocp, c.prob, c.sa, c.buf)
end

# Radau IIA collocation, differentiation-matrix form. For element k, stage i:
#
#   M(Y_k^i)·∑_j d_ij (Y_k^j − y_{k−1}) + Δt·[K(Y_k^i)Y_k^i − r(Y_k^i) − b(t_k^i, u_k)] = 0
#
# with D = A^(-1). Because ∑_j d_ij − d0_i = 0 the state offsets cancel in the D
# combination, so it can be formed on scaled variables and rescaled by sy.
function _residual!(cx, z, ocp::MethanationOCP, prob, saT, buf)
    n, s, Ne, Δt = ocp.n, ocp.s, ocp.Ne, ocp.Δt
    nz = n * ncols(ocp)
    Z = reshape(view(z, 1:nz), n, ncols(ocp))
    u = view(z, nz+1:nz+Ne)
    tab = ocp.tab
    dm, m = prob.dm, prob.model
    yp, w, Kv, b = buf.yp, buf.w, buf.Kv, buf.b
    sy, y_off, sc = ocp.sy, ocp.y_off, ocp.sc

    for k in 1:Ne
        ykm1 = view(Z, :, leftcol(ocp, k))
        uk = ocp.u_off + ocp.su * u[k]

        for i in 1:s
            Yi = view(Z, :, stagecol(ocp, k, i))
            @. yp = y_off + sy * Yi # physical stage value

            # w = ∑_j d_ij (Y_k^j − y_{k−1}), physical units
            d0i = tab.d0[i]
            @. w = -d0i * ykm1
            for j in 1:s
                Yj = view(Z, :, stagecol(ocp, k, j))
                dij = tab.D[i, j]
                @. w += dij * Yj
            end
            @. w *= sy

            saT(yp) # M(Y_k^i), K(Y_k^i), f_in
            assemble_react!(prob, yp, Val(false)) # r(Y_k^i)

            ti = stage_time(tab, k, i, Δt)
            fill!(b, zero(eltype(b)))
            for field in dm.fields
                fd = fielddof(dm, field)
                md = inlet_mod(m, field, ti)
                @views @. b[fd] += prob.f_in[fd] * md
            end
            for d in ocp.wall_dofs # only the wall cells carry the control
                b[d] += prob.f_wall[d] * uk
            end

            row = ((k - 1) * s + i - 1) * n
            cb = view(cx, row+1:row+n)
            Mnz = nonzeros(prob.M) # M is diagonal; skip its structural zeros
            @views @. cb = Mnz[ocp.mdiag] * w
            mul!(Kv, prob.K, yp)
            @. cb = (cb + Δt * Kv - Δt * prob.r - Δt * b) * sc # row scaling
        end
    end
    return cx
end

# min mean CO2 outlet fraction over [0,tf], equiv to max mean CO2 conversion.
# Radau quadrature (weights b, exact to order 2s−1) on each element; ∑_i b_i = 1.
function ocp_obj(ocp::MethanationOCP, z)
    n, s, Ne = ocp.n, ocp.s, ocp.Ne
    nz = n * ncols(ocp)
    Z = reshape(view(z, 1:nz), n, ncols(ocp))
    u = view(z, nz+1:nz+Ne)

    J_conv = zero(eltype(z))
    for k in 1:Ne, i in 1:s
        col = stagecol(ocp, k, i)
        co2_out = zero(eltype(z))
        for j in eachindex(ocp.co2_dofs)
            d = ocp.co2_dofs[j]
            co2_out += ocp.co2_w[j] * (ocp.y_off[d] + ocp.sy[d] * Z[d, col])
        end
        J_conv += ocp.tab.b[i] * (co2_out / ocp.co2_in)
    end

    # piecewise-constant control: ∑(Δû)²·Ne is Δt-independent
    J_reg = zero(eltype(z))
    for k in 1:Ne-1
        J_reg += (u[k+1] - u[k])^2
    end

    return J_conv / Ne + ocp.γ * Ne * J_reg
end

# X_CO2 = 1 − ṅ_CO2,out/ṅ_CO2,in at every column of a physical trajectory
co2_conversion(ocp::MethanationOCP, Z::AbstractMatrix) =
    [1 - sum(ocp.co2_w[j] * Z[ocp.co2_dofs[j], col] for j in eachindex(ocp.co2_dofs)) / ocp.co2_in
     for col in 1:size(Z, 2)]

function build_ocp(ocp::MethanationOCP, x0::Vector{Float64})
    n, nc = ocp.n, ncols(ocp)
    nz = n * nc
    dm = ocp.sa.prob.dm
    nsp = nspecies(ocp.sa.prob.model)
    z0 = scale_z(ocp, x0)

    lvar = fill(-Inf, nvars(ocp))
    uvar = fill(Inf, nvars(ocp))
    Zl = reshape(view(lvar, 1:nz), n, nc)
    Zu = reshape(view(uvar, 1:nz), n, nc)
    for fi in 1:nsp # scaled ρ̂ (mole frac), all collocation points
        field = dm.fields[fi]
        fd = fielddof(dm, field)
        Zl[fd, :] .= field === :H2 ? ocp.x_floor_H2 : ocp.x_floor
    end
    Td = fielddof(dm, dm.fields[nsp+1]) # path bound on T, all collocation points
    @views @. Zl[Td, :] = (ocp.T_min - ocp.y_off[Td]) / ocp.sy[Td]
    @views @. Zu[Td, :] = (ocp.Tmax - ocp.y_off[Td]) / ocp.sy[Td]
    Zl[:, 1] .= view(z0, 1:n) # initial condition pins column 1
    Zu[:, 1] .= view(z0, 1:n)
    lvar[nz+1:end] .= (ocp.Tw_min - ocp.u_off) / ocp.su
    uvar[nz+1:end] .= (ocp.Tw_max - ocp.u_off) / ocp.su

    lcon = zeros(ncons(ocp))
    ucon = zeros(ncons(ocp))

    return ADNLPModel!(z -> ocp_obj(ocp, z), z0, lvar, uvar,
        (cx, z) -> ocp_cons!(ocp, cx, z), lcon, ucon;
        hessian_backend=ZeroHessian)
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
    # Ipopt's default bound_push (0.01) is absolute — max(1,|l|) — so with a species
    # floor of x_floor = 1e-8 every trace species below 0.01 (CH4, CO, H2O, N2: ~40%
    # of all variables) is shoved from ~1e-5 to 1e-2 before the first residual
    # evaluation. A 1000x composition error wrecks the starting point; push only far
    # enough to stay strictly interior.
    stats = ipopt(nlp; hessian_approximation="limited-memory",
        bound_relax_factor=0.0, bound_push=1e-6, bound_frac=1e-6,
        hsl_options(linear_solver)..., kwargs...)
    Z, u = unscale_z(ocp, stats.solution)
    return (Z=Z, u=u, stats=stats)
end
