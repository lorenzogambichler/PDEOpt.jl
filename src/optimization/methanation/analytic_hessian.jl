# Element-level exact Hessian of the Lagrangian for MethanationOCP
# ∇²L = σ∇²f + Σ_r λ_r ∇²c_r
# Assemble from local blocks instead of global AD sweep and use ForwardDiff on blocks

# global variable index of Z[d, col]
@inline gidx(ocp::MethanationOCP, d::Int, col::Int) = (col - 1) * ocp.n + d

# Cell kernel
# v = [Z[cd[1..nf], stage i] ; Z[dT, stage 1..s] ; Z[dT, leftcol]]
mutable struct CellHess{TM}
    m::TM
    nf::Int
    s::Int
    i::Int
    Δt::Float64
    vol::Float64
    y_off::Vector{Float64} # nf, cell offsets
    sy::Vector{Float64} # nf, cell scales
    ν::Vector{Float64} # nf, λ_R·sc for this (stage, cell)
    D::Vector{Float64} # s, tableau row i
    d0i::Float64
    syT::Float64
    βn::Float64 # 0 -> not inlet/outlet cell
    area::Float64
    rc::Float64
    mdT::Float64
end

function (K::CellHess)(v)
    m, nf, s = K.m, K.nf, K.s
    y_off, sy, ν = K.y_off, K.sy, K.ν
    yc = ntuple(a -> y_off[a] + sy[a] * v[a], nfields_val(m))
    re = reaction_cell(m, yc)
    P = props_cell(m, yc)

    # reaction source (all fields)
    acc = zero(eltype(v))
    @inbounds for a in 1:nf
        acc -= K.Δt * K.vol * ν[a] * re[a]
    end

    # T mass term (species capacity constant -> only T row)
    wT = zero(eltype(v))
    @inbounds for j in 1:s
        wT += K.D[j] * v[nf+j]
    end
    wT = K.syT * (wT - K.d0i * v[nf+s+1])
    acc += ν[nf] * K.vol * P.ρcp * wT

    # state-dependent energy advection at inlet/outlet cell
    if K.βn != 0.0
        Kd, fin = bnd_energy_kernel(m, yc, K.βn, K.area, K.rc)
        acc += K.Δt * ν[nf] * (Kd * yc[nf] - fin * K.mdT)
    end
    return acc
end

# Face kernel
# v = [Z[od[1..nf], stage i] ; Z[nd[1..nf], stage i]]
mutable struct FaceHess{TM}
    m::TM
    nf::Int
    axis::Int
    Δt::Float64
    area::Float64
    dist::Float64
    wf::Float64
    y_off_o::Vector{Float64}
    sy_o::Vector{Float64}
    y_off_n::Vector{Float64}
    sy_n::Vector{Float64}
    dν::Vector{Float64} # nf, ν_owner − ν_neighbor
end

function (K::FaceHess)(v)
    m, nf = K.m, K.nf
    nfv = nfields_val(m)
    yo = ntuple(a -> K.y_off_o[a] + K.sy_o[a] * v[a], nfv)
    yn = ntuple(a -> K.y_off_n[a] + K.sy_n[a] * v[nf+a], nfv)
    F = face_flux(m, yo, yn, K.axis, K.area, K.dist, K.wf)
    acc = zero(eltype(v))
    @inbounds for a in 1:nf
        acc += K.dν[a] * F[a]
    end
    return K.Δt * acc
end

# Boundary
function _bnd_cells(ocp::MethanationOCP)
    p0 = ocp.sa.prob
    m, grid, geom = p0.model, p0.grid, p0.geom
    nc = ncells(grid)
    βn = zeros(nc)
    area = zeros(nc)
    rc = zeros(nc)
    for (bfs, bfg) in ocp.sa.ebnd
        b = velocity(m, bfs.axis) * bfs.side
        for q in eachindex(bfs.cells)
            c = bfs.cells[q]
            iszero(βn[c]) || error("cell $c appears in two boundary face sets")
            βn[c] = b
            area[c] = bfg.area[q]
            _, cj = cellij(grid, c)
            rc[c] = geom.rc[cj]
        end
    end
    return (βn=βn, area=area, rc=rc)
end

# local -> global index maps
function _cell_gmap!(gv, ocp::MethanationOCP, k::Int, i::Int, c::Int, nf::Int)
    dm, s = ocp.sa.prob.dm, ocp.s
    cd = celldof(dm, c)
    dT = cd[nf]
    ci = stagecol(ocp, k, i)
    @inbounds for a in 1:nf
        gv[a] = gidx(ocp, cd[a], ci)
    end
    @inbounds for j in 1:s
        gv[nf+j] = gidx(ocp, dT, stagecol(ocp, k, j))
    end
    gv[nf+s+1] = gidx(ocp, dT, leftcol(ocp, k))
    return gv
end

function _face_gmap!(gv, ocp::MethanationOCP, k::Int, i::Int, o::Int, nb::Int, nf::Int)
    dm = ocp.sa.prob.dm
    od, nd = celldof(dm, o), celldof(dm, nb)
    ci = stagecol(ocp, k, i)
    @inbounds for a in 1:nf
        gv[a] = gidx(ocp, od[a], ci)
        gv[nf+a] = gidx(ocp, nd[a], ci)
    end
    return gv
end

# Structure
# LLocal (row,col) pairs in block order, save positions (hess_values! uses same block order)
function hess_structure(ocp::MethanationOCP)
    p0 = ocp.sa.prob
    dm, grid = p0.dm, p0.grid
    nf = length(dm.fields)
    n, s, Ne = ocp.n, ocp.s, ocp.Ne
    nvar = nvars(ocp)
    nzv = n * ncols(ocp)
    nlc, nlf = nf + s + 1, 2nf
    ncell = ncells(grid)
    nface = sum(length(fs.owner) for (fs, _) in ocp.sa.dirs)

    ncp = nlc * (nlc + 1) ÷ 2
    nfp = nlf * (nlf + 1) ÷ 2
    nobj = 2Ne - 1
    ntot = s * Ne * (ncell * ncp + nface * nfp) + nobj
    keys = Vector{Int}(undef, ntot)

    gv = Vector{Int}(undef, max(nlc, nlf))
    @inline pack(p, q) = (max(p, q) - 1) * nvar + min(p, q)

    t = 0
    for k in 1:Ne, i in 1:s
        for c in 1:ncell
            _cell_gmap!(gv, ocp, k, i, c, nf)
            for b in 1:nlc, a in b:nlc
                keys[t+=1] = pack(gv[a], gv[b])
            end
        end
        for (fs, _) in ocp.sa.dirs, e in eachindex(fs.owner)
            _face_gmap!(gv, ocp, k, i, fs.owner[e], fs.neighbor[e], nf)
            for b in 1:nlf, a in b:nlf
                keys[t+=1] = pack(gv[a], gv[b])
            end
        end
    end
    for k in 1:Ne # objective: γ(u_{k+1}−u_k)²
        keys[t+=1] = pack(nzv + k, nzv + k)
        k < Ne && (keys[t+=1] = pack(nzv + k + 1, nzv + k))
    end
    @assert t == ntot

    uk = unique!(sort(keys))
    pos = Vector{Int}(undef, ntot)
    @inbounds for q in 1:ntot
        pos[q] = searchsortedfirst(uk, keys[q])
    end

    rows = @. (uk - 1) ÷ nvar + 1
    cols = @. (uk - 1) % nvar + 1
    return (rows=rows, cols=cols, pos=pos, ncell=ncell, nface=nface)
end

struct OCPHessian{TM,TCC,TFC}
    ocp::MethanationOCP
    rows::Vector{Int}
    cols::Vector{Int}
    pos::Vector{Int}
    nf::Int
    nlc::Int
    nlf::Int
    ncell::Int
    nface::Int
    bnd::NamedTuple{(:βn, :area, :rc),NTuple{3,Vector{Float64}}}
    cellK::CellHess{TM}
    faceK::FaceHess{TM}
    vc::Vector{Float64}
    vf::Vector{Float64}
    Hc::Matrix{Float64}
    Hf::Matrix{Float64}
    cfgc::TCC
    cfgf::TFC
end

function OCPHessian(ocp::MethanationOCP)
    p0 = ocp.sa.prob
    m, dm = p0.model, p0.dm
    nf = length(dm.fields)
    s = ocp.s
    nlc, nlf = nf + s + 1, 2nf
    st = hess_structure(ocp)

    cellK = CellHess(m, nf, s, 1, ocp.Δt, 1.0,
        zeros(nf), zeros(nf), zeros(nf), zeros(s), 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    faceK = FaceHess(m, nf, 1, ocp.Δt, 1.0, 1.0, 0.5,
        zeros(nf), zeros(nf), zeros(nf), zeros(nf), zeros(nf))
    vc, vf = zeros(nlc), zeros(nlf)
    cfgc = ForwardDiff.HessianConfig(cellK, vc, ForwardDiff.Chunk{nlc}())
    cfgf = ForwardDiff.HessianConfig(faceK, vf, ForwardDiff.Chunk{nlf}())

    return OCPHessian(ocp, st.rows, st.cols, st.pos, nf, nlc, nlf, st.ncell, st.nface,
        _bnd_cells(ocp), cellK, faceK, vc, vf,
        zeros(nlc, nlc), zeros(nlf, nlf), cfgc, cfgf)
end

nnzh(H::OCPHessian) = length(H.rows)

# Values
function hess_values!(vals::Vector{Float64}, H::OCPHessian, z::Vector{Float64},
    λ::Vector{Float64}; obj_weight::Float64=1.0)
    ocp = H.ocp
    p0 = ocp.sa.prob
    m, dm, grid, geom = p0.model, p0.dm, p0.grid, p0.geom
    n, s, Ne, Δt = ocp.n, ocp.s, ocp.Ne, ocp.Δt
    nf, nlc, nlf = H.nf, H.nlc, H.nlf
    nzv = n * ncols(ocp)
    sy, y_off, sc = ocp.sy, ocp.y_off, ocp.sc
    tab = ocp.tab
    bnd = H.bnd
    ck, fk = H.cellK, H.faceK
    gv = Vector{Int}(undef, max(nlc, nlf))
    ν = zeros(n)

    fill!(vals, 0.0)
    t = 0

    for k in 1:Ne, i in 1:s
        rowbase = ((k - 1) * s + i - 1) * n
        @views @. ν = λ[rowbase+1:rowbase+n] * sc
        ti = stage_time(tab, k, i, Δt)
        mdT = inlet_mod(m, dm.fields[nf], ti)
        ck.i = i
        ck.d0i = tab.d0[i]
        ck.mdT = mdT
        @views ck.D .= tab.D[i, :]
        ci = stagecol(ocp, k, i)

        # cell blocks
        for c in 1:H.ncell
            cd = celldof(dm, c)
            cix, cjx = cellij(grid, c)
            ck.vol = cellvolume(geom, cix, cjx)
            @views ck.y_off .= y_off[cd]
            @views ck.sy .= sy[cd]
            @views ck.ν .= ν[cd]
            ck.syT = sy[cd[nf]]
            ck.βn = bnd.βn[c]
            ck.area = bnd.area[c]
            ck.rc = bnd.rc[c]
            _cell_gmap!(gv, ocp, k, i, c, nf)
            @inbounds for a in 1:nlc
                H.vc[a] = z[gv[a]]
            end
            ForwardDiff.hessian!(H.Hc, ck, H.vc, H.cfgc)
            # slots nf and nf+i alias one global variable -> pair lands on diag
            # full double sum needs (a,b) and (b,a)
            @inbounds for b in 1:nlc, a in b:nlc
                mult = (a != b && gv[a] == gv[b]) ? 2.0 : 1.0
                vals[H.pos[t+=1]] += mult * H.Hc[a, b]
            end
        end

        # face blocks
        for (fs, fg) in ocp.sa.dirs, e in eachindex(fs.owner)
            o, nb = fs.owner[e], fs.neighbor[e]
            od, nd = celldof(dm, o), celldof(dm, nb)
            fk.axis = fs.axis
            fk.area = fg.area[e]
            fk.dist = fg.dist[e]
            fk.wf = fg.wf[e]
            @views fk.y_off_o .= y_off[od]
            @views fk.sy_o .= sy[od]
            @views fk.y_off_n .= y_off[nd]
            @views fk.sy_n .= sy[nd]
            @views @. fk.dν = ν[od] - ν[nd]
            _face_gmap!(gv, ocp, k, i, o, nb, nf)
            @inbounds for a in 1:nlf
                H.vf[a] = z[gv[a]]
            end
            ForwardDiff.hessian!(H.Hf, fk, H.vf, H.cfgf)
            @inbounds for b in 1:nlf, a in b:nlf
                vals[H.pos[t+=1]] += H.Hf[a, b]
            end
        end
    end

    # objective: f = J_conv/Ne + γ·Ne·Σ(u_{k+1}−u_k)², J_conv affine in Z
    q = obj_weight * ocp.γ * Ne
    for k in 1:Ne
        vals[H.pos[t+=1]] += 2q * ((k > 1) + (k < Ne))
        k < Ne && (vals[H.pos[t+=1]] += -2q)
    end
    return vals
end

# NLPModel wrapper
const NLPM = ADNLPModels.NLPModels

struct KernelHessNLP{T,S,TI,TH} <: NLPM.AbstractNLPModel{T,S}
    meta::NLPM.NLPModelMeta{T,S}
    counters::NLPM.Counters
    inner::TI
    hess::TH
    vals::Vector{T}
end

function KernelHessNLP(inner::NLPM.AbstractNLPModel{T,S}, H::OCPHessian) where {T,S}
    im = inner.meta
    meta = NLPM.NLPModelMeta{T,S}(im.nvar; x0=im.x0, lvar=im.lvar, uvar=im.uvar,
        ncon=im.ncon, y0=im.y0, lcon=im.lcon, ucon=im.ucon,
        nnzj=im.nnzj, lin_nnzj=im.lin_nnzj, nln_nnzj=im.nln_nnzj,
        nnzh=nnzh(H), lin=im.lin, minimize=im.minimize, islp=im.islp,
        name="MethanationOCP (kernel Hessian)")
    return KernelHessNLP(meta, NLPM.Counters(), inner, H, zeros(T, nnzh(H)))
end

NLPM.obj(nlp::KernelHessNLP, x::AbstractVector) = NLPM.obj(nlp.inner, x)
NLPM.grad!(nlp::KernelHessNLP, x::AbstractVector, g::AbstractVector) =
    NLPM.grad!(nlp.inner, x, g)
NLPM.cons_nln!(nlp::KernelHessNLP, x::AbstractVector, c::AbstractVector) =
    NLPM.cons_nln!(nlp.inner, x, c)
NLPM.jac_nln_structure!(nlp::KernelHessNLP, r::AbstractVector{<:Integer},
    c::AbstractVector{<:Integer}) = NLPM.jac_nln_structure!(nlp.inner, r, c)
NLPM.jac_nln_coord!(nlp::KernelHessNLP, x::AbstractVector, v::AbstractVector) =
    NLPM.jac_nln_coord!(nlp.inner, x, v)
NLPM.jprod_nln!(nlp::KernelHessNLP, x::AbstractVector, v::AbstractVector,
    Jv::AbstractVector) = NLPM.jprod_nln!(nlp.inner, x, v, Jv)
NLPM.jtprod_nln!(nlp::KernelHessNLP, x::AbstractVector, v::AbstractVector,
    Jtv::AbstractVector) = NLPM.jtprod_nln!(nlp.inner, x, v, Jtv)

function NLPM.hess_structure!(nlp::KernelHessNLP, rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer})
    copyto!(rows, nlp.hess.rows)
    copyto!(cols, nlp.hess.cols)
    return rows, cols
end

function NLPM.hess_coord!(nlp::KernelHessNLP, x::AbstractVector, y::AbstractVector,
    vals::AbstractVector; obj_weight=1.0)
    hess_values!(nlp.vals, nlp.hess, collect(Float64, x), collect(Float64, y);
        obj_weight=Float64(obj_weight))
    copyto!(vals, nlp.vals)
    return vals
end

function NLPM.hess_coord!(nlp::KernelHessNLP, x::AbstractVector, vals::AbstractVector;
    obj_weight=1.0)
    return NLPM.hess_coord!(nlp, x, zeros(Float64, nlp.meta.ncon), vals;
        obj_weight=obj_weight)
end
