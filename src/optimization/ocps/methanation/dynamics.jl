# M(y)·ẏ − F(y,t,u) -> global assembly, independent of transcription
struct MethanationDynamics{T,TSA,TAD}
    sa::TSA # StateAssembly
    ad::TAD # AntiDiffusion
    mdiag::Vector{Int} # nzval indexes M[d,d]
    wall_dofs::Vector{Int} # f_wall dofs
    Kv::Vector{T} # scratch
    b::Vector{T} # scratch
end

function MethanationDynamics(sa, y0::AbstractVector; Tmin, Tmax,
    recon::Symbol=:vanalbada, ad_rel::Float64=1e-3)
    prob = sa.prob
    recon in (:upwind1, :vanalbada) ||
        error("recon must be :upwind1 or :vanalbada, got :$recon")
    ad = recon === :upwind1 ? nothing :
         AntiDiffusion(sa, collect(y0); ΔT=(Float64(Tmax)-Float64(Tmin)), rel=ad_rel)

    n = ndof(prob.dm)
    nnz(dropzeros(copy(prob.M))) == n || error("mass matrix not diagonal")
    return MethanationDynamics(sa, ad, _diagindex(prob.M), findall(!iszero, prob.f_wall),
        zeros(n), zeros(n))
end

function (dyn::MethanationDynamics)(res, y, yd, t, u)
    sa = dyn.sa
    prob = sa.prob
    dm, m = prob.dm, prob.model

    sa(y) # M(y), K(y), f_in
    assemble_react!(prob, y, Val(false)) # r(y)
    ad = dyn.ad
    isnothing(ad) ||
        assemble_antidiffusion!(prob, sa.props, ad.st, ad.fs, ad.fg, y, ad.εf)

    b = dyn.b
    fill!(b, zero(eltype(b)))
    for field in dm.fields
        fd = fielddof(dm, field)
        md = inlet_mod(m, field, t)
        @views @. b[fd] += prob.f_in[fd] * md
    end
    for d in dyn.wall_dofs # only wall cells carry control
        b[d] += prob.f_wall[d] * u[1]
    end

    Mnz = nonzeros(prob.M) # M diagonal -> skip structural zeros
    mul!(dyn.Kv, prob.K, y)
    @. res = Mnz[dyn.mdiag] * yd + dyn.Kv - prob.r - b
    return res
end

# copy for AD
function retype(dyn::MethanationDynamics, ::Type{T}) where {T}
    p0 = dyn.sa.prob
    tosp(A) = SparseMatrixCSC(A.m, A.n, copy(A.colptr), copy(A.rowval), zeros(T, length(A.nzval)))
    n, nf = ndof(p0.dm), length(p0.dm.fields)
    prob = ProblemCache(tosp(p0.M), tosp(p0.K), tosp(p0.Jr),
        zeros(T, n), zeros(T, nf), zeros(T, nf, nf),
        T.(p0.f_in), T.(p0.f_wall), p0.grid, p0.geom, p0.dm, p0.model)
    props = MethanationProps{T}(nspecies(p0.model), ncells(p0.grid))
    sa = StateAssembly(prob, props, dyn.sa.K_bc, dyn.sa.dirs, dyn.sa.ebnd)
    return MethanationDynamics(sa, dyn.ad, dyn.mdiag, dyn.wall_dofs,
        zeros(T, n), zeros(T, n))
end

