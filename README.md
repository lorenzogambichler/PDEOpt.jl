# PDEOpt

PDE-constrained optimization framework based on direct collocation (temporal) and FVM/DG-FEM (spatial).

## Governing equations

Currently, two types of models are implemented. Both with cyclindrical, radially symmetric domains $(z,r)$ and only axial advection (no axial dispersion) with and a cooling/heating wall at $r = R$.

**PlugFlow** (`plugflow.jl`): advection–diffusion–reaction, constant coefficients,
first-order Arrhenius kinetics.

$$\partial_t C + v_z\,\partial_z C = \frac{1}{r}\partial_r\big(r\,D_r\,\partial_r C\big) - k(T)\,C$$

$$\partial_t T + v_z\,\partial_z T = \frac{1}{r}\partial_r\big(r\,\lambda_r\,\partial_r T\big) - \Delta H_r\,k(T)\,C$$

$$k(T) = k_0\,\exp\!\left(-\frac{E_a}{R\,T}\right), \qquad -\lambda_r\,\partial_r T\big|_{r=R} = U\,\big(T - T_w(t)\big)$$

**Methanation** (`methanation.jl`): 2D pseudo-homogeneous fixed bed, state
$(\rho_i,\,T)$ with $i \in$ \{CH4, CO, CO2, H2O, H2, N2\}, Xu–Froment LHHW
kinetics over three reactions $j \in \{1,2,3\}$.

$$\varepsilon\,\partial_t \rho_i+ v_z\,\partial_z \rho_i = \frac{1}{r}\partial_r\big(r\,D^{\text{eff}}_{r,i}\,\partial_r \rho_i\big) + (1-\varepsilon)\,M_i \sum_j \nu_{i,j}\,\tilde{r}_j(\rho, T)$$

$$(\rho c_p)^{\text{eff}}\,\partial_t T + \rho\,c_{p,\text{gas}}\,v_z\,\partial_z T = \frac{1}{r}\partial_r\big(r\,\lambda^{\text{eff}}_r\,\partial_r T\big) - (1-\varepsilon) \sum_j \Delta H_j\,\tilde{r}_j(\rho, T)$$

$$p = \sum_i \frac{\rho_i}{M_i}\,R\,T, \qquad -\lambda^{\text{eff}}_r\,\partial_r T\big|_{r=R} = k_w\,\big(T - T_w(t)\big)$$

All transport coefficients $D^{\text{eff}}_{r,i}$, $\lambda^{\text{eff}}_r$,
$(\rho c_p)^{\text{eff}}$ depend on the state via `properties!`.

## Spatial discretization

- **FVM** (`assemble_fvm.jl`): Structured cylindrical grid, 1st order upwind scheme, central-difference radial diffusion, weak inflow/wall BCs, *self-written*.
- **DG** (`assemble_dg.jl`): SIPG diffusion, upwind-flux advection, makes use of *Ferrite.jl*.


The general discretized ODE form after spatial discretization is

$$M(y) \dot{y} + K(y) y = r(y) + b(u)$$

with
- M(y): Mass matrix (transient term)
- K(y): Transport matrix (diffusion + advection)
- r(y): Reaction source
- b(u): boundary forcing (BCs, control u)

## Temporal discretization

For the contruction of the NLP, the decision vector is
$$z = (y_1,\dots,y_N, u_1,\dots,u_N), \qquad y_k \in \mathbb{R}^n, u_k \in \mathbb{R}^m$$
and the ODE-constraint is formulated using different schemes, based on the problem definition:

- Trapezoidal collocation (`PlugFlowOpt.jl`)
$$0 = Ay_{k+1} - By_k - \tfrac{\Delta t}{2}\big(r(y_{k+1}) + r(y_k)\big) - \tfrac{\Delta t}{2}\big(b(t_{k+1},u_{k+1}) + b(t_k,u_k)\big)$$ with $$A = M + \tfrac{\Delta t}{2}K, \quad B = M - \tfrac{\Delta t}{2}K.$$

- Implicit midpoint collocation (`MethanationOpt.jl`)
$$0 = M\big(y_{k+1/2}\big)(y_{k+1}-y_k) + (\Delta t)K\big(y_{k+1/2}\big)y_{k+1/2} - (\Delta t) r\big(y_{k+1/2}\big) - (\Delta t)b\big(t_{k+1/2},u_{k+1/2}\big)$$ with $$y_{k+1/2} = \frac{y_k + y_{k+1}}{2}, \qquad u_{k+1/2} = \frac{u_k + u_{k+1}}{2}, \qquad t_{k+1/2} = \left(k - \tfrac12\right)\Delta t$$

## Simple example case (PlugFlowOpt.jl)

Wall-temperature control $T_w(t)$ of the plug-flow reactor, minimizing reactant concentration at the outlet (i.e. maximize conversion) with a rate penalty on the controls:
$$\min_{y,\,T_w} \; \sum_{k=1}^{N} \sum_{j \in \text{outlet}} v_z\,A_j\,C_{j,k} \; + \; \gamma \sum_{k=1}^{N-1} \big(\tilde{T}_{w,k+1} - \tilde{T}_{w,k}\big)^2$$

s.t. the trapezoidal collocation constraints and box bounds
$T \le T_{\max}$, $C \ge 10^{-4}$ (safeguard), $T_w \in [T_{w,\min}, T_{w,\max}]$.
All the decision variables are scaled for better conditioning and the penalty acts on the scaled control $\tilde{T}_w$, making $\gamma$ dimensionless.

**Setup:** JuMP + Ipopt with exact second derivatives, HSL MA97 as linear solver
running on multiple threads, using a Crank-Nicolson forward simulation as initial guess (`cn_solve!`).

**Size:** Grid is $50\cdot 5 =250$ cells with $2$ fields, meaning $n = 500$. $t_f = 2\ \mathrm{s}$,
$\Delta t = 0.005\ \mathrm{s}$, so $N = 401$. That results in $nN + N = 200901$ variables
and $nN = 200500$ equality constraints.

**Output:** `apps/optimal_control/results/PlugFlow/` — `opt_*.vtr` (optimized
trajectory) as ParaView collection and the $T_w$ profile as `control.csv`.

## Complex case (MethanationOpt.jl)
Technically runs, but is still work in progress. Requires NLP conditioning improvements and upgrade to HSL solvers, aswell as problem formulation tweaks (cost function, constraints, etc.). 

## Directory structure

- `src/models/`: Problem-dependent physics, model interface
- `src/properties/`: Correlations (diffusion, viscosity, conductivity, heat capacity, Ergun, equilibrium, isotherms)
- `src/assembly/`: Local kernels, global scatter loops (FVM, DG)
- `src/mesh/`: Structured grid, geometry, DOF map, sparsity
- `src/timestepping/`: ODE integrators (Crank–Nicolson)
- `src/solvers/`: Nonlinear solvers (Newton, Chord, Shamanskii)
- `src/optimization/`: Simultaneous NLP transcription (`jump_ipopt.jl`, `adnlp_ipopt.jl`)
- `apps/`: Driver scripts for simulation (`PlugFlow.jl`, `Methanation.jl`) and optimal control
- `tests/`: Assembly/solver checks
