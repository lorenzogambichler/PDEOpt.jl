# Prototype / scoping experiment.
#
# Compare the hand-rolled Crank–Nicolson forward solve against a few stiff
# integrators from OrdinaryDiffEq.jl (Rosenbrock + ESDIRK) on the *same*
# FVM-discretised ADR problem.  Wall-clock only — this is a "should I bother"
# measurement, not a converged accuracy study.
#
# The PDE assembled in ADR_FVM.jl is        M ẏ + K y = r(y) + b(t),
# i.e. a mass-matrix ODE                     M ẏ = -K y + r(y) + b(t),
# with Jacobian                              ∂f/∂u = -K + Jr(u).
# Because K, M and Jr are all copies of the same sparsity_pattern, that
# Jacobian is a single nzval combination — no pattern juggling.

using LinearAlgebra
using SparseArrays
using Printf
using OrdinaryDiffEq
using OrdinaryDiffEqSDIRK   # KenCarp*/TRBDF2 — not re-exported by the meta-package (v7.1.1)

using PDEOpt
include("ADR_FVM.jl")   # ProblemCache, setup_problem, assemble_*!, CNCache, cn_solve!, set_ic!

# Swallow the debug `display(it)` inside quasi_newton!/newton! (it prints through
# the display stack, which redirect_stdout can't catch) without touching
# nonlinear.jl.  println/@printf write to stdout directly and are unaffected.
pushdisplay(TextDisplay(devnull))

# ---------------------------------------------------------------------------
# Reaction *source only* (no Jacobian) for the ODE right-hand side.
# assemble_react_global! also builds Jr on every call, which the RHS doesn't
# need — evaluating only the source keeps the f-eval comparison fair.
# ---------------------------------------------------------------------------
function assemble_react_r!(prob::ProblemCache, y::AbstractVector)
    fill!(prob.r, 0.0)
    for c in 1:ncells(prob.grid)
        i, j = cellij(prob.grid, c)
        vol = cellvolume(prob.geom, i, j)
        cd = celldof(prob.dm, c)              # (C dof, T dof)
        sC, sT = prob.src(y[cd[1]], y[cd[2]])
        prob.r[cd[1]] += sC * vol
        prob.r[cd[2]] += sT * vol
    end
    return prob.r
end

# ---------------------------------------------------------------------------
# f and ∂f/∂u as functors (structs) rather than closures, to avoid the
# closure-boxing that would otherwise skew the timing.
# ---------------------------------------------------------------------------
struct ODERHS{TP}
    prob::TP
    b::Vector{Float64}   # boundary-forcing scratch
end
function (f::ODERHS)(du, u, p, t)
    prob = f.prob
    mul!(du, prob.K, u)                        # du = K u
    assemble_react_r!(prob, u)                 # prob.r = r(u)
    assemble_boundary_global!(prob, f.b, t)    # f.b    = b(t)
    @. du = -du + prob.r + f.b                 # du = -K u + r(u) + b(t)
    return nothing
end

struct ODEJac{TP}
    prob::TP
end
function (j::ODEJac)(J, u, p, t)
    prob = j.prob
    assemble_react_global!(prob, u)            # prob.Jr = ∂r/∂u at u
    @. J.nzval = -prob.K.nzval + prob.Jr.nzval # ∂f/∂u = -K + Jr  (shared pattern)
    return nothing
end

# Time gradient ∂f/∂t = ∂b/∂t  (only the boundary forcing depends explicitly on t).
# Supplied analytically (one-sided FD of b) so Rosenbrock methods never autodiff
# through t — which would hit the `t::Float64` signature of assemble_boundary_global!.
struct ODETgrad{TP}
    prob::TP
    b1::Vector{Float64}
    b2::Vector{Float64}
end
function (tg::ODETgrad)(dT, u, p, t)
    h = 1e-6 * max(abs(t), 1.0)
    assemble_boundary_global!(tg.prob, tg.b1, t)
    assemble_boundary_global!(tg.prob, tg.b2, t + h)
    @. dT = (tg.b2 - tg.b1) / h
    return nothing
end

function initial_state(prob::ProblemCache)
    y0 = zeros(ndof(prob.dm))
    for field in prob.dm.fields                # each field constant at its inlet value
        @views y0[fielddof(prob.dm, field)] .= prob.y_in[field](0.0)
    end
    return y0
end

function build_ode(prob::ProblemCache, tf::Float64)
    y0 = initial_state(prob)
    Mmass = Diagonal(diag(prob.M))             # cell volumes; nonsingular -> plain ODE
    n = ndof(prob.dm)
    f = ODEFunction(ODERHS(prob, zeros(n));
                    mass_matrix   = Mmass,
                    jac           = ODEJac(prob),
                    jac_prototype = copy(prob.K),
                    tgrad         = ODETgrad(prob, zeros(n), zeros(n)))
    return ODEProblem(f, y0, (0.0, tf))
end

# warm up once (compile), then report the fastest of n runs plus the heap
# traffic of a warmed run (probed after compilation so compile-time allocations
# aren't counted).  Swap @allocated -> @allocations for a raw event count.
function bestof(f; n = 1)
    f()                                        # compile / warm up
    t = minimum(@elapsed(f()) for _ in 1:n)
    a = @allocated f()                         # heap bytes on a warmed run
    return (; time = t, allocs = a)
end

function main_compare()
    tf = 2.5
    Δt = 0.01
    N  = round(Int, tf / Δt) + 1               # CN's fixed step count + 1

    prob = setup_problem(tf)

    # --- CN baseline (their solver, quasi-Newton) ---
    cache = CNCache(prob, Δt, N)
    run_cn() = (set_ic!(cache);
                cn_solve!(cache,
                          y -> assemble_react_global!(prob, y),
                          (b, t) -> assemble_boundary_global!(prob, b, t),
                          FrozenNewton(refactor_every = 10)))
    res_cn = bestof(run_cn)        # display(it) noise suppressed via pushdisplay above
    t_cn   = res_cn.time
    y_cn   = copy(cache.y[:, end])

    # --- OrdinaryDiffEq stiff integrators (adaptive) ---
    ode = build_ode(prob, tf)
    reltol, abstol = 1e-3, 1e-4
    algs = ["Rodas5P"  => Rodas5P(),           # L-stable Rosenbrock-W, order 5
            "KenCarp4" => KenCarp4()]           # ESDIRK, order 4
            #"TRBDF2"   => TRBDF2(),            # ESDIRK, order 2 — "L-stable CN"
            #"FBDF"     => FBDF()]             # variable-order BDF (large-scale reference)

    @printf("\nCN fixed Δt = %.3g  ->  %d steps\n", Δt, N - 1)
    @printf("adaptive tolerances: reltol = %.0e, abstol = %.0e\n\n", reltol, abstol)
    @printf("%-11s %10s %9s %7s %9s %12s %12s\n",
            "method", "time [s]", "f-evals", "jacs", "steps", "‖Δ‖∞ vs CN", "allocs")
    @printf("%-11s %10.3f %9s %7s %9d %12s %12s\n",
            "CN", t_cn, "-", "-", N - 1, "-", Base.format_bytes(res_cn.allocs))

    for (name, alg) in algs
        solve_it() = solve(ode, alg; reltol, abstol, save_everystep = false)
        res = bestof(solve_it; n = 1)
        sol = solve_it()
        d   = norm(sol.u[end] .- y_cn, Inf)
        @printf("%-11s %10.3f %9d %7d %9d %12.2e %12s\n",
                name, res.time, sol.stats.nf, sol.stats.njacs, sol.stats.naccept, d,
                Base.format_bytes(res.allocs))
    end
    return nothing
end

main_compare()
