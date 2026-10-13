# M·ẏ − F(y,t,u) -> constant M, K (assembled once), only r(y) state-dependent
struct PlugFlowDynamics{T,TP,TAD}
    prob::TP # ProblemCache (eltype T, for r(y))
    ad::TAD # AntiDiffusion
    K::SparseMatrixCSC{Float64,Int} # constant transport
    mdiag::Vector{Float64} # diag(M), constant
    wall_dofs::Vector{Int} # f_wall dofs
    Kv::Vector{T} # scratch
    b::Vector{T} # scratch
end

function PlugFlowDynamics(prob, y0::AbstractVector; Tmin, Tmax,
    recon::Symbol=:vanalbada, ad_rel::Float64=1e-3)
    recon in (:upwind1, :vanalbada) ||
        error("recon must be :upwind1 or :vanalbada, got :$recon")
    ad = nothing
    if recon === :vanalbada
        (fs,) = interior_faces(prob.grid) # axial only
        fg = FaceGeometry(prob.grid, prob.geom, fs)
        sc = antidiff_sc(prob.model, collect(view(y0, celldof(prob.dm, 1)));
            ΔT=(Float64(Tmax) - Float64(Tmin)))
        ad = AntiDiffusion(prob.grid, prob.geom, fs, fg, sc; rel=ad_rel)
    end

    n = ndof(prob.dm)
    nnz(dropzeros(copy(prob.M))) == n || error("mass matrix not diagonal")
    return PlugFlowDynamics(prob, ad, copy(prob.K), Vector(diag(prob.M)),
        findall(!iszero, prob.f_wall), zeros(n), zeros(n))
end

function (dyn::PlugFlowDynamics)(res, y, yd, t, u)
    prob = dyn.prob
    dm, m = prob.dm, prob.model

    assemble_react!(prob, y, Val(false)) # r(y)
    ad = dyn.ad
    isnothing(ad) ||
        assemble_antidiffusion!(prob, nothing, ad.st, ad.fs, ad.fg, y, ad.εf)

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

    mul!(dyn.Kv, dyn.K, y)
    @. res = dyn.mdiag * yd + dyn.Kv - prob.r - b
    return res
end

# copy for AD (constant M, K stay Float64)
function retype(dyn::PlugFlowDynamics, ::Type{T}) where {T}
    p0 = dyn.prob
    n, nf = ndof(p0.dm), length(p0.dm.fields)
    sp(A) = SparseMatrixCSC{T,Int}(A)
    prob = ProblemCache(sp(p0.M), sp(p0.K), sp(p0.Jr),
        zeros(T, n), zeros(T, nf), zeros(T, nf, nf),
        T.(p0.f_in), T.(p0.f_wall), p0.grid, p0.geom, p0.dm, p0.model)
    return PlugFlowDynamics(prob, dyn.ad, dyn.K, dyn.mdiag, dyn.wall_dofs,
        zeros(T, n), zeros(T, n))
end
