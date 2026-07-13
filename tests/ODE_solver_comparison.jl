# Prototype / scoping experiment.
# Compare self-written Crank–Nicolson against adaptive OrdinaryDiffEq solvers for same problem
# M ẏ = f(y) = -K y + r(y) + b(t), ∂f/∂y = -K + Jr(y).

using LinearAlgebra
using SparseArrays
using Printf
using OrdinaryDiffEq
using OrdinaryDiffEqSDIRK

using PDEOpt
include("../apps/forward_solvers/PlugFlow.jl") # ProblemCache, setup_problem, set_ic!, write_vtk (assembly now in PDEOpt)

pushdisplay(TextDisplay(devnull))

# f and ∂f/∂u jac as functors
struct ODERHS{TP}
    prob::TP
    b::Vector{Float64} # boundary-forcing scratch
end
function (f::ODERHS)(dy, y, p, t)
    prob = f.prob
    mul!(dy, prob.K, y) # dy = Ky
    assemble_react_global!(prob, y, Val(false)) # prob.r = r(y), source only (no jac)
    assemble_boundary_global!(prob, f.b, t) # f.b = b(t)
    @. dy = -dy + prob.r + f.b # dy = -Ky + r(y) + b(t)
    return nothing
end

struct ODEJac{TP}
    prob::TP
end
function (j::ODEJac)(J, y, p, t)
    prob = j.prob
    assemble_react_global!(prob, y, Val(true)) # prob.Jr = ∂r/∂y at y
    @. J.nzval = -prob.K.nzval + prob.Jr.nzval # ∂f/∂y = -K + Jr, use shared pattern
    return nothing
end

# Time gradient ∂f/∂t = ∂b/∂t, (only boundary forcing depends explicitly on t)
# Analytical to avoid autodiff 
struct ODETgrad{TP}
    prob::TP
    b1::Vector{Float64}
    b2::Vector{Float64}
end
function (tg::ODETgrad)(dT, y, p, t)
    h = 1e-6 * max(abs(t), 1.0)
    assemble_boundary_global!(tg.prob, tg.b1, t)
    assemble_boundary_global!(tg.prob, tg.b2, t + h)
    @. dT = (tg.b2 - tg.b1) / h
    return nothing
end

function initial_state(prob::ProblemCache)
    y0 = zeros(ndof(prob.dm))
    for field in prob.dm.fields # each field constant at its inlet value
        @views y0[fielddof(prob.dm, field)] .= inlet_mod(prob.model, field, 0.0)
    end
    return y0
end

function build_ode(prob::ProblemCache, tf::Float64)
    y0 = initial_state(prob)
    Mmass = Diagonal(diag(prob.M)) 
    n = ndof(prob.dm)
    f = ODEFunction(ODERHS(prob, zeros(n)); mass_matrix = Mmass, jac = ODEJac(prob), 
        jac_prototype = copy(prob.K), tgrad = ODETgrad(prob, zeros(n), zeros(n)))
    return ODEProblem(f, y0, (0.0, tf))
end

function bestof(f; n = 3)
    f() # warm-up
    t = minimum(@elapsed(f()) for _ in 1:n)
    a = @allocated f()                        
    return (; time = t, allocs = a)
end

function main_compare()
    tf = 2.5
    Δt = 0.01
    N = round(Int, tf / Δt) + 1 

    prob = setup_problem(tf)

    # CN
    cache = CNCache(prob, Δt, N)
    run_cn() = (set_ic!(cache); cn_solve!(cache, (y, WithJac) -> assemble_react_global!(prob, y, WithJac), 
        (b, t) -> assemble_boundary_global!(prob, b, t), frozen_newton()))
    res_cn = bestof(run_cn)        
    t_cn = res_cn.time
    y_cn = copy(cache.y[:, end])

    # OrdinaryDiffEq
    ode = build_ode(prob, tf)
    reltol, abstol = 1e-3, 1e-4
    algs = ["Rodas5P"  => Rodas5P(), "KenCarp4" => KenCarp4(), "Trapezoid"   => Trapezoid()]

    @printf("\nCN fixed Δt = %.3g  ->  %d steps\n", Δt, N - 1)
    @printf("adaptive tolerances: reltol = %.0e, abstol = %.0e\n\n", reltol, abstol)
    @printf("%-11s %10s %9s %7s %9s %12s %12s\n",
        "method", "time [s]", "f-evals", "jacs", "steps", "‖⋅‖∞ vs CN", "allocs")
    @printf("%-11s %10.3f %9s %7s %9d %12s %12s\n",
        "CN", t_cn, cache.stats.nf, cache.stats.nfact, N - 1, "-", Base.format_bytes(res_cn.allocs))

    for (name, alg) in algs
        solve_it() = solve(ode, alg; reltol, abstol, save_everystep = false)
        res = bestof(solve_it)
        sol = solve_it()
        d   = norm(sol.u[end] .- y_cn, Inf)
        @printf("%-11s %10.3f %9d %7d %9d %12.2e %12s\n",
                name, res.time, sol.stats.nf, sol.stats.njacs, sol.stats.naccept, d,
                Base.format_bytes(res.allocs))
    end
    return nothing
end
