using Ferrite
using LinearAlgebra
using SparseArrays
using WriteVTK
using PDEOpt

struct SolverCache{TLfact,Tfact,Tdh,Tcv,F1,F2,F3}
    L_fact::TLfact # LU of A 
    A::SparseMatrixCSC{Float64,Int} # M + Δt/2·K 
    R::SparseMatrixCSC{Float64,Int} # M - Δt/2·K 
    J::SparseMatrixCSC{Float64,Int} # Newton Jacobian
    Jr::SparseMatrixCSC{Float64,Int} # reaction Jacobian 
    fact::Tfact # LU of J
    Je::Matrix{Float64} # element reaction Jacobian 
    re::Vector{Float64} # element reaction residual
    fr::Vector{Float64} # global reaction residual buffer
    f_in::Vector{Float64} # unit inflow load ∫_∂Ω_in -(β·ν) g(x) y ds, scaled by τ(t)
    y::Matrix{Float64} 
    temp1::Vector{Float64}
    temp2::Vector{Float64}
    temp3::Vector{Float64}
    dh::Tdh
    cv::Tcv
    Δt::Float64
    N::Int64
    src::F1 # coupled source terms for C, T
    srcJac::F2 # source term Jacobian
    τ::F3 # inlet time modulation for g(x)⋅τ(t)
end

# Assemble constant block-diagonal M, K for :C and :T
function assemble_global(cv::CellValues, facetvalues::FacetValues,
    interfacevalues::InterfaceValues, M::SparseMatrixCSC, K::SparseMatrixCSC, dh::DofHandler,
    ∂Ω_out::AbstractSet{FacetIndex}, order::Int64, dim::Int64, β::Vec,
    D::SymmetricTensor{2,2,Float64,3}, λ::SymmetricTensor{2,2,Float64,3})

    # Allocate element
    nbf = getnbasefunctions(cv)
    Me = zeros(nbf, nbf)
    Ke = zeros(nbf, nbf)
    Ce = zeros(nbf, nbf)
    Cb = zeros(nbf, nbf)
    Ki = zeros(2nbf, 2nbf)
    Ci = zeros(2nbf, 2nbf)

    cr = dof_range(dh, :C) # local field dofs inside cell
    tr = dof_range(dh, :T) 
    assembler_M = start_assemble(M)
    assembler_K = start_assemble(K)

    # Volume terms, scattered into dof blocks per field (advection and diffusion in K)
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        dofs = celldofs(cell)
        cdofs, tdofs = @view(dofs[cr]), @view(dofs[tr])

        assemble_mass!(Me, cv)
        assemble!(assembler_M, cdofs, Me)
        assemble!(assembler_M, tdofs, Me)

        assemble_advection!(Ce, cv, β)
        assemble!(assembler_K, cdofs, Ce)
        assemble!(assembler_K, tdofs, Ce)

        assemble_diffusion!(Ke, cv, D)
        assemble!(assembler_K, cdofs, Ke)
        assemble_diffusion!(Ke, cv, λ)
        assemble!(assembler_K, tdofs, Ke)
    end

    # Interface terms (SIPG diffusion + upwind advection) per field 
    # Interface dofs are [here(C,T), there(C,T)], here/there pair for each field gathered explicitly
    off = 2nbf
    for ic in InterfaceIterator(dh)
        reinit!(interfacevalues, ic)
        hₑ = getdiameter(∩(getcoordinates(ic)...))
        μ = (1 + order)^dim / hₑ
        idofs = interfacedofs(ic)
        cidofs = vcat(idofs[cr], idofs[off .+ cr])
        tidofs = vcat(idofs[tr], idofs[off .+ tr])

        assemble_advection_interface!(Ci, interfacevalues, β)
        assemble!(assembler_K, cidofs, Ci)
        assemble!(assembler_K, tidofs, Ci)

        assemble_interface!(Ki, interfacevalues, μ, D)
        assemble!(assembler_K, cidofs, Ki)
        assemble_interface!(Ki, interfacevalues, μ, λ)
        assemble!(assembler_K, tidofs, Ki)
    end

    # Outflow advective flux per field (interior trace). Walls no-flux; inlet weak (load).
    for fc in FacetIterator(dh, ∂Ω_out)
        reinit!(facetvalues, fc)
        dofs = celldofs(fc)
        assemble_advection_boundary!(Cb, facetvalues, β)
        assemble!(assembler_K, @view(dofs[cr]), Cb)
        assemble!(assembler_K, @view(dofs[tr]), Cb)
    end
    return M, K
end

# TODO: Global diffusion fd = ... and Jacobian Jd = ...
# replaces assemble_diffusion in assemble_global
function assemble_diff_global!()
end


# Global coupled reaction residual fr and Jacobian Jr (CC/CT/TC/TT blocks) at (C,T)
function assemble_react_global!(Jr::SparseMatrixCSC, fr::Vector{Float64}, cache::SolverCache, y::AbstractVector)
    cr, tr = dof_range(cache.dh, :C), dof_range(cache.dh, :T)
    assembler = start_assemble(Jr, fr)
    for cell in CellIterator(cache.dh)
        reinit!(cache.cv, cell)
        dofs = celldofs(cell)
        Ce = @view y[@view dofs[cr]]
        Te = @view y[@view dofs[tr]]
        assemble_reaction!(cache.Je, cache.re, cache.cv, Ce, Te, cache.src, cache.srcJac)
        assemble!(assembler, dofs, cache.Je, cache.re)
    end
    return Jr, fr
end

# Global weak inflow into frange, f_in[frange] += ∫_∂Ω_in -(β·ν) g(x) v ds (additive)
function assemble_inflow_global!(f_in::Vector{Float64}, fv::FacetValues, dh::DofHandler,
    ∂Ω_in::AbstractSet{FacetIndex}, β::Vec, g, frange)
    fe = zeros(getnbasefunctions(fv))
    for fc in FacetIterator(dh, ∂Ω_in)
        reinit!(fv, fc)
        assemble_inflow!(fe, fv, β, g, getcoordinates(fc))
        @views f_in[celldofs(fc)[frange]] .+= fe
    end
    return f_in
end

# reaction Jacobian cache.Jr and residual cache.fr at y, passed to cn_solve!
react!(cache, y) = assemble_react_global!(cache.Jr, cache.fr, cache, y)

function write_vtk(FilePath::String, cache::SolverCache)
    pvd = paraview_collection(FilePath)
    for n = 1:cache.N
        t = (n - 1) * cache.Δt
        VTKGridFile("$(FilePath)_$(n)", cache.dh) do vtk
            write_solution(vtk, cache.dh, cache.y[:, n])  
            pvd[t] = vtk
        end
    end
    close(pvd)
end

# Arrhenius term R = Da·exp(-γ/T)·C  (T floored at ε > 0)
# use callable structs to avoid capture boxing
struct ArrheniusSrc # (C,T) -> (sC, sT)
    Da::Float64; γ::Float64; q::Float64; ε::Float64
end
function (a::ArrheniusSrc)(C, T)
    R = a.Da * exp(-a.γ / max(T, a.ε)) * C
    return (-R, a.q * R)
end

struct ArrheniusJac # (C,T) -> (∂sC/∂C, ∂sC/∂T, ∂sT/∂C, ∂sT/∂T)
    Da::Float64; γ::Float64; q::Float64; ε::Float64
end
function (a::ArrheniusJac)(C, T)
    e = a.Da * exp(-a.γ / max(T, a.ε)) # ∂R/∂C
    R = e * C
    RT = T > a.ε ? R * a.γ / T^2 : 0.0 # ∂R/∂T (0 in the floored region)
    return (-e, -RT, a.q * e, a.q * RT)
end

function setup_cache()
    # Time-stepping
    tf = 2.5
    Δt = 0.01
    N = round(Int64, tf / Δt) + 1

    # Parameters
    β = Vec((2.0, 0.0))
    D = SymmetricTensor{2,2,Float64}((0.0, 0.0, 1e-2)) # (ax, 0, rad)
    λ = SymmetricTensor{2,2,Float64}((0.0, 0.0, 1e-2))

    # Domain
    dim = 2
    nz = 40
    nr = 10
    grid = generate_grid(Quadrilateral, (nz, nr), Vec{2, Float64}((0.0, 0.0)), Vec{2, Float64}((2.0, 0.5)))
    topology = ExclusiveTopology(grid)

    # FE Space and quadrature
    order = 1
    ip = DiscontinuousLagrange{RefQuadrilateral,order}()
    qr = QuadratureRule{RefQuadrilateral}(2 * order)
    facet_qr = FacetQuadratureRule{RefQuadrilateral}(2 * order)

    cv = CellValues(qr, ip)
    facetvalues = FacetValues(facet_qr, ip)
    interfacevalues = InterfaceValues(facet_qr, ip)

    dh = DofHandler(grid)
    add!(dh, :C, ip)
    add!(dh, :T, ip)
    close!(dh)
    cr, tr = dof_range(dh, :C), dof_range(dh, :T)

    # block sparsity: same-field flux across interfaces; volume room for C–T reaction.
    K = allocate_matrix(dh; topology=topology, coupling=trues(2, 2), interface_coupling=[true false; false true])
    M = copy(K) # same sparsity pattern as K

    # inlets  y_in(x, t) = g(x)·τ_in(t)  (shared ramp)
    g_C(x) = 1.0
    g_T(x) = 1.0
    τ_in(t) = 2.0 * clamp(t / (0.5 * tf), -1.0, 1.0)

    # Outflow (Neumann)
    ∂Ω_out = getfacetset(grid, "right")
    # Weak inflow
    ∂Ω_in = getfacetset(grid, "left")

    M, K = assemble_global(cv, facetvalues, interfacevalues, M, K, dh, ∂Ω_out, order, dim, β, D, λ)

    # Weak inflow load per field, scaled by τ_in(t)
    f_in = zeros(ndofs(dh))
    assemble_inflow_global!(f_in, facetvalues, dh, ∂Ω_in, β, g_C, cr)
    assemble_inflow_global!(f_in, facetvalues, dh, ∂Ω_in, β, g_T, tr)

    # CN matrices, nzval to keep C–T blocks for coupled Jr
    A = copy(K)
    R = copy(K)
    @. A.nzval = M.nzval + 0.5 * Δt * K.nzval # implicit
    @. R.nzval = M.nzval - 0.5 * Δt * K.nzval # explicit
    J = copy(A) # Newton Jacobian
    Jr = copy(A) # reaction Jacobian buffer
    fact = lu(J) # cached symbolic factorizaation

    L_fact = lu(A)

    nbf = getnbasefunctions(cv)
    Je = zeros(2nbf, 2nbf) # coupled (C,T) element Jacobian
    re = zeros(2nbf)
    fr = zeros(ndofs(dh))

    temp1 = zeros(ndofs(dh))
    temp2 = zeros(ndofs(dh))
    temp3 = zeros(ndofs(dh))

    # Arrhenius reaction model (Da, γ, q, ε); see ArrheniusSrc/ArrheniusJac above.
    src = ArrheniusSrc(0.1, 1.0, 1.0, 1e-3)
    srcJac = ArrheniusJac(0.1, 1.0, 1.0, 1e-3)

    # Initial condition
    y = zeros(ndofs(dh), N)

    cache = SolverCache(L_fact, A, R, J, Jr, fact, Je, re, fr, f_in, y, temp1, temp2, temp3, dh, cv, Δt, N, src, srcJac, τ_in)
    return cache
end

function main()
    cache = setup_cache()
    cn_solve!(cache, y -> react!(cache, y), newton!) # or quasi_newton!
    write_vtk("tests/results/ADR_DG/ADR_DG", cache)
end