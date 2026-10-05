# General workflow

This document describes the general workflow for the direct transcription of an optimal control problem.

## Reusable and custom components

Currently, the `PDEOpt` module is fully generic, while the constents of 
[src/optimization/](../src/optimization/) are problem-specific and need to be re-implemented (or adapted) for each problem. This is subject to change. 

## Steps
For the methanation case, all the required steps are listed below:
- `setup_problem` -> ProblemCache, StateAssembly (step 1)
- `MethanationOCP` -> layout, bounds, scaling (steps 2-3)
- residual, objective, Jac/Hess (steps 4-6)
- initial guess, `solve_ocp` (staps 7-8)

## Step 1, Spatial discretization

The optimization layer requires:

```julia
prob, sa, y0 = setup_problem()
```

- `prob::ProblemCache` -> assembled operators $M$, $K$, `Jr`, boundary forcing
  `f_in`/`f_wall`, and mesh/dofmap/model
- `sa::StateAssembly` -> functor that rebuilds $M(y)$ and $K(y)$ at a given state $y$
- `y0` -> initial condition

See [apps/methanation/problem.jl](../apps/methanation/problem.jl)
for example.

The semi-discretized ODE takes the form

$$M(y)\,\dot y + K(y)\,y = r(y) + b(t, u),$$

with the control $u$ entering affinely through $b$.
The capacity matrix $M$ has to be diagonal and state-dependent branching in the differentiated code is not supported because of tracer-based sparsity detection.

## Step 2, Transcription

Radau IIA collocation (`RadauIIA(s)`) on `Ne` finite elements. The decision variable is

```
z = [y0; Y11 Y12 Y13; Y21 Y22 Y23; …; YNe1 YNe2 YNe3; u1; …; uNe].
```

One column vector per collocation stage and one constant control per element

Indexing is handled by helper functions in [ocp.jl](../src/optimization/methanation/ocp.jl):

| | |
| --- | --- |
| `ncols(ocp)` | `s*Ne + 1` |
| `leftcol(ocp, k)` | column of $y_{k-1}$ (left end of element `k`) |
| `stagecol(ocp, k, i)` | column of stage `i` on element `k` |
| `nvars(ocp)` | `n*ncols + Ne` |
| `ncons(ocp)` | `n*s*Ne` |
| `timegrid(ocp)` | physical time of every column |
| `control_times(ocp)` | left end time of every element |


## Step 3, OCP

`MethanationOCP` in [ocp.jl:70](../src/optimization/methanation/ocp.jl) stores all the OCP data.

Variables are scaled and constraint rows are normalized using

- `sy`, `y_off` -> scale and offset for every dof
  (species-scale $c_{tot} M_\alpha$, T-scale $\Delta T = T_{max} - T_{min}$, T-offset $T_{min}$)
- `su`, `u_off` -> control scale (from bounds)
- `sc` -> constraint scale for every row (species-scale $1/(\varepsilon V s_y)$, T-scale $1/(\rho c_p^{ref} V \Delta T)$)

and `_scales` derives all scaling factors from `y0`.

Bounds are set in `build_ocp` (scaled units). These inclide a positivity constraint on species
(`x_floor`, `x_floor_H2`), T path constraint
$T_{min} \le T \le T_{max}$, initial condition and control
box-bounds $[T_{w,min}, T_{w,max}]$.

## Step 4, Collocation residual

[residual.jl](../src/optimization/methanation/residual.jl) contains the constraint function
$c(z) = 0$, one block $c^{i,k}(z) \in \mathbb{R}^n$ per stage:

Global assembly through `_residual!` calls `saT(yp)` and `assemble_react!` per stage.

The kernel-based assembly `_residual_kernels!` constructs the same residual from cell and face kernels (`props_cell`, `reaction_cell`, `capacity_cell`, `face_flux`, `bnd_energy_kernel`) with no
global matrices. Used for AD.

`gettcache(ocp, T)` allows using `ForwardDiff.Dual` by retyping `ProblemCache`, `StateAssembly` and buffers.

## Step 5, Objective and gradient

[objective.jl](../src/optimization/methanation/objective.jl). Integration of the Lagrange term using Radau quadrature and a control
regularization term $\gamma N_e \sum_k (u_{k+1} - u_k)^2$.

The objective is linear in the state and quadratic in the control, so the gradient is
written out analytically in `ocp_grad!` as an `ADNLPModels.ADBackend`:

```julia
struct OCPGradient{TO} <: ADNLPModels.ADBackend
    ocp::TO
end
ADNLPModels.gradient!(b::OCPGradient, g, f, x) = ocp_grad!(g, b.ocp, x)
```

## Step 6, Jacobian and Hessian

### Jacobian 
Sparse AD via `ADNLPModels`, pattern computed once by
`compute_jacobian_sparsity`, cached by `jac_pattern` as `(nvars, ncons, n, s, Ne, recon)`.

### Hessian

Default is quasi-Newton (`exact_hessian=false`) -> `limited-memory` L-BFGS in IPOPT. Requires `ZeroHessian` backend for `get_nln_nnzh = 0`
to stop `ADNLPModels` from allocating a dense Hessian structure.

Exact Hessian (`exact_hessian=true`, [analytic_hessian.jl](../src/optimization/methanation/analytic_hessian.jl))
assembles the exact Hessian of the Lagrangian from local kernels -> `CellHess`, `FaceHess` inside `KernelHessNLP` that implements `hess_structure!` / `hess_coord!`. Requires
`recon = :upwind1` for now.

## Step 7, Initial guess

The presence of an ignition fold requires a good initial guess to achieve convergence.

[initial_guess.jl](../src/optimization/methanation/initial_guess.jl) uses a Crank-Nicolson forward solver inside a bisection loop to find a good control guess that respects the bounds.

- `bisect_const` -> largest constant $T_w$ with
  $\max T < T_{max} - \delta$
- `bisect_shape` -> use shape $s(\tau) \in [0,1]$, find largest feasible $\lambda$ in
  $u(t) = \lambda + (T_{w,max} - \lambda)s(t/t_f)$.

Forward solves use a separate time grid, so `resample` converts the guess to the OCP time grid

## Step 8, Solve

```julia
ocp = MethanationOCP(sa, RadauIIA(3), Δt, Ne, y0;
    Tw_min, Tw_max, Tmax, γ, co2_in, recon = :vanalbada)

res = solve_ocp(ocp, x0; print_level=5, max_iter=200, tol=1e-5,
    exact_hessian=false, limited_memory_max_history=75)
```

`build_ocp` to assemble the `ADNLPModel!`, `solve_ocp` to call IPOPT and unscale results `(Z, u, stats)`.

Uses HSL `ma97` by default with MUMPS fallback if `libhsl`
is missing. `hsl_options` sets METIS and low `nemin` to save memory.

Optional proflling using `solve_profiled`, needs `julia -t 2`.
