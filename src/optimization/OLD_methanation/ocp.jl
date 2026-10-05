struct MethanationOCP{TSA,TTab,TAD}
    # Collocation
    sa::TSA
    tab::TTab # tableau
    Δt::Float64
    Ne::Int
    s::Int # stages
    n::Int # spatial dofs
    y0::Vector{Float64}
    # Bounds
    Tw_min::Float64
    Tw_max::Float64
    Tmax::Float64
    Tmin::Float64
    x_floor::Float64
    x_floor_H2::Float64
    # Scaling
    sy::Vector{Float64} # state scale
    y_off::Vector{Float64} # state offset (T only)
    sc::Vector{Float64} # constraint row scale
    su::Float64 # control scale
    u_off::Float64 # control offset
    # Objective
    γ::Float64 # reg scale
    co2_dofs::Vector{Int} # outlet
    co2_w::Vector{Float64} # vz·A
    co2_in::Float64
    # Sparsity
    mdiag::Vector{Int} # nzval indexes M[d,d]
    wall_dofs::Vector{Int} # f_wall dofs
    caches::Dict{DataType,Any}
    # Discretization
    ad::TAD # AntiDiffusion (nothing for 1st order)
end

recon(ocp::MethanationOCP) = isnothing(ocp.ad) ? :upwind1 : :vanalbada

# diagonal nzval positions CSC matrix
function _diagindex(A::SparseMatrixCSC)
    idx = Vector{Int}(undef, size(A, 1))
    rv, cp = rowvals(A), A.colptr
    for d in eachindex(idx)
        p = searchsortedfirst(view(rv, cp[d]:cp[d+1]-1), d)
        idx[d] = cp[d] + p - 1
    end
    return idx
end

ncols(ocp::MethanationOCP) = ocp.s * ocp.Ne + 1
stagecol(ocp::MethanationOCP, k::Int, i::Int) = 1 + (k - 1) * ocp.s + i
leftcol(ocp::MethanationOCP, k::Int) = 1 + (k - 1) * ocp.s
nvars(ocp::MethanationOCP) = ocp.n * ncols(ocp) + ocp.Ne
ncons(ocp::MethanationOCP) = ocp.n * ocp.s * ocp.Ne

# Time of each col
function timegrid(ocp::MethanationOCP)
    t = zeros(ncols(ocp))
    for k in 1:ocp.Ne
        for i in 1:ocp.s
            t[stagecol(ocp, k, i)] = stage_time(ocp.tab, k, i, ocp.Δt)
        end
    end
    return t
end

# Tw piecewise constant on el -> left edge of each el
control_times(ocp::MethanationOCP) = [(k - 1) * ocp.Δt for k in 1:ocp.Ne]

# Reference scales, sy/y_off (vars), sc (constr rows)
function _scales(sa, y0::Vector{Float64}, Tmin::Float64, Tmax::Float64)
    prob = sa.prob
    dm, m, grid, geom = prob.dm, prob.model, prob.grid, prob.geom
    nsp = nspecies(m)
    ctot = sum(y0[dof(dm, 1, α)] / m.M[α] for α in 1:nsp) # inlet conc
    ΔT = Tmax - Tmin
    ρcp_ref = (1 - m.ε) * m.ρcat * m.cp_cat

    sy = zeros(ndof(dm))
    y_off = zeros(ndof(dm))
    sc = zeros(ndof(dm))
    for c in 1:ncells(grid)
        i, j = cellij(grid, c)
        V = cellvolume(geom, i, j)
        for α in 1:nsp
            d = dof(dm, c, α)
            sy[d] = ctot * m.M[α] # ρ scaling
            sc[d] = 1 / (m.ε * V * sy[d])
        end
        d = dof(dm, c, nsp + 1)
        sy[d] = ΔT # T scaling
        y_off[d] = Tmin # T offset
        sc[d] = 1 / (ρcp_ref * V * ΔT)
    end
    return sy, y_off, sc
end

function MethanationOCP(sa, tab, Δt::Float64, Ne::Int, y0::Vector{Float64};
    Tw_min, Tw_max, Tmax, Tmin=min(Tw_min, minimum(@view y0[fielddof(sa.prob.dm, :T)])) - 50.0,
    x_floor=1e-8, x_floor_H2=1e-3, γ=0.0, co2_in=nothing,
    recon::Symbol=:vanalbada, ad_rel::Float64=1e-3)

    dm, m = sa.prob.dm, sa.prob.model
    sy, y_off, sc = _scales(sa, y0, Float64(Tmin), Float64(Tmax))

    recon in (:upwind1, :vanalbada) ||
        error("recon must be :upwind1 or :vanalbada, got :$recon")
    ad = recon === :upwind1 ? nothing :
         AntiDiffusion(sa, y0; ΔT=Float64(Tmax) - Float64(Tmin), rel=ad_rel)

    bfs_out, bfg_out = sa.ebnd[2]
    co2_dofs = [dof(dm, c, :CO2) for c in bfs_out.cells]
    co2_w = velocity(m, 1) .* copy(bfg_out.area)
    Fin = isnothing(co2_in) ? co2_inflow(sa) : Float64(co2_in)

    mdiag = _diagindex(sa.prob.M)
    nnz(dropzeros(copy(sa.prob.M))) == ndof(dm) ||
        error("mass matrix not diagonal")
    wall_dofs = findall(!iszero, sa.prob.f_wall)

    return MethanationOCP(sa, tab, Δt, Ne, nstages(tab), ndof(dm), copy(y0),
        Tw_min, Tw_max, Tmax, Tmin, x_floor, x_floor_H2,
        sy, y_off, sc, Tw_max - Tw_min, Tw_min,
        γ, co2_dofs, co2_w, Fin, mdiag, wall_dofs, Dict{DataType,Any}(), ad)
end

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
