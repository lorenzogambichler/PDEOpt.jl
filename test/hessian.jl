using PDEOpt
using LinearAlgebra
using SparseArrays
using Serialization
using Random
using Printf
using CairoMakie
import ForwardDiff

include("../apps/methanation/opt.jl")

const RECON = :upwind1

function setup_case(; nz=4, nr=2, tf=150.0, Ne=2, s=3, Δt_cn=0.01, Tw_min=300.0, Tw_max=560.0, 
    Tmax=750.0, γ=0.01, Tw=500.0, recon=RECON)

    prob, sa, y0 = setup_problem(;nz=nz, nr=nr)
    
    Δt = tf/Ne
    ocp = MethanationOCP(sa, RadauIIA(s), Δt, Ne, y0; Tw_min=Tw_min, Tw_max=Tw_max,
        Tmax=Tmax, γ=γ, co2_in=co2_inflow(sa), recon=recon)

    cache, _ = cn_forward(t->Tw, tf, Δt_cn; recon=recon, nz=nz, nr=nr)
    Y = resample(cache.y, Δt_cn, timegrid(ocp))
    u = fill(Tw, Ne)
    x0 = vcat(vec(Y), u)
    z0 = scale_z(ocp, x0)

    data = (nz=nz, nr=nr, Ne=Ne, s=s, nvars=nvars(ocp), ncons=ncons(ocp))
    return (ocp=ocp, x0=x0, z0=z0, data=data)
end

function setup_nlps(case)
    lcon = ucon = zeros(ncons(case.ocp))
    f = z -> ocp_obj(case.ocp, z)
    c! = (cx, z) -> ocp_cons!(case.ocp, cx, z)

    nlp_ref = ADNLPModel!(f, case.z0, c!, lcon, ucon; jacobian_backend=ADNLPModels.EmptyADbackend)
    #nlp_kernel = build_ocp(case.ocp, case.x0; exact_hessian=true)

    return nlp_ref#, nlp_kernel
end

# Test
case = setup_case()
nlp_ref = setup_nlps(case)
n = nlp_ref.meta.nvar

r_ref = zeros(Int, nlp_ref.meta.nnzh)
c_ref = similar(r)
ADNLPModels.NLPModels.hess_structure!(nlp_ref, r_ref, c_ref)

H_lower = sparse(r_ref, c_ref, ones(length(r_ref)), n, n)
display(spy(H, colormap = :binary))




