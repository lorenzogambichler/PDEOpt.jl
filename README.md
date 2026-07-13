# PDEOpt

A hand-rolled framework for **PDE-constrained optimization** via the
discretize-then-optimize (DtO) route. Currently focused on the **forward
solver**; the optimization layer (adjoints, control) comes later, which is why
the design keeps assembly explicit, allocation-light, and free of hidden
closures.

## Semidiscrete abstraction

Every model discretizes to the same ODE shape

$$M\,\dot{y} + K\,y = r(y) + b(t)$$

- **M** — mass / capacity matrix (transient term)
- **K** — transport (diffusion + advection)
- **r(y)** — reaction source (nonlinear, needs a Jacobian)
- **b(t)** — boundary forcing (inlet + wall)

For **constant transport** (plug flow), `M` and `K` are assembled **once**. For
**state-dependent** physics (methanation) the coefficients depend on `y`, so the
system generalizes to a state-dependent capacity and transport operator

$$C(y)\,\dot{y} + K(y)\,y = r(y) + b(t),$$

reassembled per step from a per-cell property pass (see *Models* below).

## Time-stepping

Crank–Nicolson, solved with a Newton family:

$$A\,y_{k+1} = B\,y_k + \tfrac{\Delta t}{2}\big(r(y_{k+1}) + r(y_k)\big) + \tfrac{\Delta t}{2}\big(b_{k+1} + b_k\big)$$

with $A = M + \tfrac{\Delta t}{2}K$, $B = M - \tfrac{\Delta t}{2}K$. Nonlinear
solve via `newton!`, `chord!` (freeze factorization within a step), or
`shamanskii` (reuse one factorization across steps, refresh on stall) — the
factorization amortization that dominates cost at scale.

## Assembly backends

- **FVM** (`assemble_fvm.jl`) — structured cylindrical grid, upwind advection,
  central-difference radial diffusion, weak inflow / wall BCs. Used by the
  reactor apps.
- **DG** (`assemble_dg.jl`) — SIPG diffusion, upwind-flux advection.

Diffusion is symmetric, advection is not; both scatter into `K`.

## Models — the physics layer

`src/models/` holds one file per project. A model bundles transport, reaction,
and boundary data; the assembly kernels dispatch on the **interface**, never the
concrete type. See the `Models` module docstring for the full contract. Adding a
project = adding a file that implements the subset it needs.

| Model | File | Notes |
|---|---|---|
| `PlugFlowModel` | `plugflow.jl` | Constant transport, `Arrhenius` reaction — the fast path |
| `MethanationModel` | `methanation.jl` | State-dependent transport + capacity; Xu–Froment kinetics; `properties!` DAG pass filling `MethanationProps` |
| `PSAModel` | `psa.jl` | Adsorption (LDF + isotherm) — skeleton |

State-dependent models compute per-cell thermophysical properties in one
`properties!` pass (a dependency DAG: composition → pressure → cp/μ/λ →
capacity, conductivity, dispersion). Transport/capacity assembly reads the
resulting `MethanationProps`; `reaction!` stays self-contained.

## Directory map

- `src/models/`      — problem-dependent physics + the model interface
- `src/properties/`  — reusable correlations (diffusion, viscosity, conductivity, heat capacity, Ergun, equilibrium, isotherms)
- `src/assembly/`    — local kernels + global scatter loops (FVM, DG)
- `src/mesh/`        — structured grid, geometry, DOF map, sparsity, refinement
- `src/timestepping/`— ODE integrators (Crank–Nicolson)
- `src/solvers/`     — nonlinear solvers (Newton, chord, Shamanskii)
- `apps/`            — driver scripts (`PlugFlow.jl`, `Methanation.jl`)
- `tests/`           — assembly / solver checks

## Apps

- **`PlugFlow.jl`** — working 2D advection–diffusion–reaction demo.
- **`Methanation.jl`** — WIP: Bremer et al. CO₂ methanation start-up. Model
  physics (`MethanationModel`, `properties!`) is in place; state-dependent
  transport/capacity assembly and the driver are being wired.
