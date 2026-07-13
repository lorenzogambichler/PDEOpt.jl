### Summary

Upwinding DG SIPG FEM advection-diffusion-reaction (ADR) solver

**Diffusion** ($\Delta y$): SIPG
- assemble_diffusion!, assemble_interface! 
- symmetric
- K

**Advection** ($-\nabla \cdot (\beta y)$): upwind flux
- assemble_advection!, assemble_advection_interface!
- non-symmetric
- K

**Reaction** ($r(y)$): 
- assemble_reaction!
- Newton, linearized

**Mass** ($\partial_t y$):
- assemble_mass!, cn_step_nonlinear!
- Newton
- M

**ODE**
- $M\dot{y} + Ky = r(y) + b(u)$
- Crank-Nicolson time-stepping
- $Ay_{k+1} = By_k + \frac{\Delta t}{2}(r(y_{k+1}) + r(y_k)) + \frac{\Delta t}{2}(b(u_{k+1}) + b(u_k))$
- $A = (M + \frac{\Delta t}{2}K)$, $B = (M - \frac{\Delta t}{2}K)$
- Newton, Quasi-Newton, Frozen Newton (reuse factorization over multiple steps)

**Structure**
- src/assembly/: local kernels, global scatter loops
- src/mesh/: Structured mesh helper functions, mesh refinement
- src/models/: Problem-dependent physics
- apps/: Driver scripts
- src/timestepping/: ODE solvers
- src/solvers/: Nonlinear Solvers (Newton)
