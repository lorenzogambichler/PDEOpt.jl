# Full-space DtO (ADNLPModels for sparse AD) for MethanationModel

# Still work in progress!!

using PDEOpt
using SparseArrays
using LinearAlgebra
using ADNLPModels
using NLPModelsIpopt

# 
struct ZeroHessian <: ADNLPModels.ADBackend end
ZeroHessian(args...; kwargs...) = ZeroHessian()
ADNLPModels.get_nln_nnzh(::ZeroHessian, nvar) = 0

struct MethOCP{TSA}
    sa::TSA
    Δt::Float64
    N::Int
    n::Int
    y0::Vector{Float64}
    Tw_min::Float64
    Tw_max::Float64
    Tmax::Float64
    ρ_floor::Float64
    γ::Float64
    obj_scale::Float64
    ch4::Int
    outlet_cells::Vector{Int}
    outlet_area::Vector{Float64}
    caches::Dict{DataType,Any}
end

function MethOCP(sa, Δt::Float64, N::Int, y0::Vector{Float64};
    Tw_min, Tw_max, Tmax, ρ_floor=1e-8, γ=0.0, obj_scale=1.0)
    bfs_out, bfg_out = sa.ebnd[2]
    return MethOCP(sa, Δt, N, length(y0), copy(y0), Tw_min, Tw_max, Tmax,
        ρ_floor, γ, obj_scale, 1, copy(bfs_out.cells), copy(bfg_out.area), Dict{DataType,Any}())
end

function ch4_production(ocp::MethOCP, Y::AbstractMatrix)
    p0 = ocp.sa.prob
    vz = velocity(p0.model, 1)
    s = 0.0
    for k in 1:size(Y, 2), j in eachindex(ocp.outlet_cells)
        s += vz * ocp.outlet_area[j] * Y[dof(p0.dm, ocp.outlet_cells[j], ocp.ch4), k]
    end
    return s
end

function _tcache(ocp::MethOCP, ::Type{T}) where {T}
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

gettcache(ocp::MethOCP, ::Type{T}) where {T} = get!(() -> _tcache(ocp, T), ocp.caches, T)

function ocp_cons!(ocp::MethOCP, cx, z)
    c = gettcache(ocp, eltype(z))
    return _residual!(cx, z, ocp, c.prob, c.sa, c.buf)
end

function _residual!(cx, z, ocp::MethOCP, prob, saT, buf)
    n, N, Δt = ocp.n, ocp.N, ocp.Δt
    Y = reshape(view(z, 1:n*N), n, N)
    u = view(z, n*N+1:n*N+N)
    dm, m = prob.dm, prob.model
    ymid, Δy, Kv, b = buf.ymid, buf.Δy, buf.Kv, buf.b
    for k in 1:N-1
        yk = view(Y, :, k)
        ykp1 = view(Y, :, k+1)
        @. ymid = 0.5 * (yk + ykp1)
        @. Δy = ykp1 - yk
        umid = 0.5 * (u[k] + u[k+1])
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
        @. cb += Δt * Kv - Δt * prob.r - Δt * b
    end
    return cx
end

function ocp_obj(ocp::MethOCP, z)
    n, N = ocp.n, ocp.N
    p0 = ocp.sa.prob
    Y = reshape(view(z, 1:n*N), n, N)
    u = view(z, n*N+1:n*N+N)
    vz = velocity(p0.model, 1)
    prod = zero(eltype(z))
    for k in 1:N, j in eachindex(ocp.outlet_cells)
        cdof = dof(p0.dm, ocp.outlet_cells[j], ocp.ch4)
        prod += vz * ocp.outlet_area[j] * Y[cdof, k]
    end
    reg = zero(eltype(z))
    for k in 1:N-1
        reg += (u[k+1] - u[k])^2
    end
    return -ocp.obj_scale * prod + ocp.γ * reg
end

function build_ocp(ocp::MethOCP, x0::Vector{Float64})
    n, N = ocp.n, ocp.N
    nvar = n * N + N
    ncon = n * (N - 1)
    dm = ocp.sa.prob.dm
    nsp = nspecies(ocp.sa.prob.model)

    lvar = fill(-Inf, nvar)
    uvar = fill(Inf, nvar)
    Yl = reshape(view(lvar, 1:n*N), n, N)
    Yu = reshape(view(uvar, 1:n*N), n, N)
    for fi in 1:nsp
        fd = fielddof(dm, dm.fields[fi])
        Yl[fd, :] .= ocp.ρ_floor
    end
    Td = fielddof(dm, dm.fields[nsp+1])
    Yl[Td, :] .= 1.0
    Yu[Td, :] .= ocp.Tmax
    Yl[:, 1] .= ocp.y0
    Yu[:, 1] .= ocp.y0
    lvar[n*N+1:end] .= ocp.Tw_min
    uvar[n*N+1:end] .= ocp.Tw_max

    lcon = zeros(ncon)
    ucon = zeros(ncon)

    return ADNLPModel!(z -> ocp_obj(ocp, z), copy(x0), lvar, uvar,
        (cx, z) -> ocp_cons!(ocp, cx, z), lcon, ucon;
        hessian_backend = ZeroHessian)
end

function solve_ocp(ocp::MethOCP, x0::Vector{Float64}; kwargs...)
    nlp = build_ocp(ocp, x0)
    stats = ipopt(nlp; hessian_approximation="limited-memory", kwargs...)
    n, N = ocp.n, ocp.N
    z = stats.solution
    return (Y=reshape(z[1:n*N], n, N), u=z[n*N+1:end], stats=stats)
end
