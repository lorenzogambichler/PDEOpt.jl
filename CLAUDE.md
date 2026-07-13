# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A hand-rolled framework for **PDE-constrained optimization** via the
**discretize-then-optimize (DtO)** route. Work is currently on the **forward
solver**; the optimization layer (adjoints, control) comes later, and that
future adjoint sweep is *why* the design keeps assembly explicit,
allocation-light, and free of hidden closures. Treat adjoint-friendliness and
factorization amortization as first-class design axes, not just runtime.

## Commands

No `Pkg.test`/`runtests.jl` harness — `tests/` are standalone scripts. Run
everything from the project environment (Julia ≥ 1.11).

```bash
# One-time: instantiate the pinned Manifest
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run the working demo (writes ParaView VTK to apps/.../results/)
julia --project=. -e 'include("apps/forward_solvers/PlugFlow.jl"); main()'

# Run a single test script
julia --project=. tests/ODE_solver_comparison.jl
```

For iterative work prefer a **persistent REPL** (`julia --project=.`, then
`include(...)`) — compile latency dominates otherwise. `Revise` is not a
dependency; re-`include` the changed source or restart after editing `src/`.

## The semidiscrete abstraction (the core contract)

Every model discretizes to the same ODE shape

```
M ẏ + K y = r(y) + b(t)
```

- **M** mass/capacity, **K** transport (diffusion + advection), **r(y)**
  nonlinear reaction source (needs a Jacobian), **b(t)** boundary forcing
  (inlet + wall).
- **Constant transport** (plug flow): `M`, `K` assembled **once**.
- **State-dependent** physics (methanation): coefficients depend on `y`, so it
  generalizes to `C(y) ẏ + K(y) y = r(y) + b(t)`, reassembled per step from a
  per-cell `properties!` pass. `state_dependent(model)` selects the path.

Time integration is Crank–Nicolson (`src/timestepping/crank_nicolson.jl`):
`A y_{k+1} = B y_k + Δt/2 (r_{k+1}+r_k) + Δt/2 (b_{k+1}+b_k)` with
`A = M + Δt/2 K`, `B = M − Δt/2 K`. The per-step nonlinear solve goes through a
**pluggable Newton family** (`src/solvers/nonlinear.jl`): `newton!` (refactor
every iter), `chord!` (freeze factorization within a step), `shamanskii` (a
**functor** holding state — reuse one factorization across steps, refresh on
scheduled interval or when contraction stalls). Solver choice = the
factorization-amortization knob that dominates cost at scale.

## Module layout and load order

`src/PDEOpt.jl` is the single module; **include order matters** (mesh →
properties → models → problem → assembly → solvers → timestepping) because each
`using ..Module` depends on the prior. The public surface is the `export` block
in `PDEOpt.jl` — update it when adding user-facing symbols.

- `src/mesh/` — `StructuredGrid` (cylindrical z×r), `Geometry`, `DofMap`,
  `sparsity_pattern`, face sets.
- `src/properties/` — **reusable** correlations (diffusion, viscosity,
  conductivity, heat capacity, Ergun, equilibrium, isotherms). Physics-agnostic.
- `src/models/` — **problem-dependent** physics, one file per project.
- `src/problem/` — `ProblemCache`, the assembled-state container.
- `src/assembly/` — local kernels + global scatter loops, FVM and DG backends.
- `src/solvers/`, `src/timestepping/` — nonlinear solvers, ODE integrators.
- `apps/forward_solvers/` — driver scripts (`setup_problem`/`main`, VTK output).
- `tests/` — assembly/solver check scripts.

## The model interface (how physics plugs in)

Assembly kernels **dispatch on the interface, never the concrete model type**.
Adding a project = adding one file in `src/models/` that implements the subset
of the interface it needs (see the `Models` module docstring in
`src/models/models.jl` for the full contract). Key methods: `fields`,
`transported`, `state_dependent`, `diffusivity`/`velocity` (constant transport),
`reaction!`/`reaction_jac!`, the `inlet_*`/`wall_*` BC accessors, and
`properties!` for state-dependent models.

State-dependent models (`methanation.jl`) fill a per-cell property struct
(`MethanationProps`) in one `properties!` pass ordered as a dependency DAG:
composition → pressure → cp/μ/λ → capacity, conductivity, dispersion. Transport
and capacity assembly then *read* that struct; `reaction!` stays self-contained.

Models present: `PlugFlowModel` (constant transport, `Arrhenius` — the working
fast path), `MethanationModel` (Xu–Froment LHHW kinetics, WIP), `PSAModel`
(skeleton). `reaction_jac!` currently defaults to a finite-difference Jacobian
(`reaction_jac_fd!`); an analytic/AD path is a planned replacement.

## Invariants to preserve

- **Allocation-light hot paths.** Assembly, `properties!`, and reaction kernels
  run per-cell per-step — thread caller-supplied scratch (as in
  `reaction_jac_fd!`, `MethanationProps`), don't allocate inside loops.
- **No hidden closures in the framework.** Closures captured in setup get boxed
  and defeat the explicit-DtO goal; use **functors/structs with call
  operators** for stateful pieces (see `shamanskii`).
- **Shared sparsity pattern.** `M`, `K`, `Jr`, and the CN `A`/`B`/`J` are all
  copies of one `sparsity_pattern(grid, dm)` and updated via `.nzval` in place —
  a sparse `+` prunes structurally-zero coupling blocks, silently changing the
  pattern. Never rebuild these with arithmetic that can drop entries.
- **Dispatch on the interface**, not `isa SomeModel`, so new models drop in.

## Status caveats

`apps/forward_solvers/Methanation.jl` is a **stale stub**: it references symbols
that no longer exist (`ADRModel`, `LHHW`, `frozen_newton`, `assemble_*_global!`)
left from a prior ADR design. `PlugFlow.jl` is the current working reference for
wiring a driver. `TODO.md` tracks the live worklist (state-dependent `K(y)`/`C(y)`
assembly, AD reaction Jacobian, Ergun pressure drop, the DtO adjoint sweep, 2nd-
order FVM). Physical constants and correlations in the models (VDI polynomials,
Xu–Froment params) include **placeholders marked `TODO`** — verify against a
cited source before trusting a run; validation is still open.
```
