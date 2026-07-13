### TODOs
- $K(y)$ and $C(y)$ assembly in fvm_assembly.jl
- Bauer-Schlünder $\lambda_{rs}$
- reaction_jac! AD instead of finite differences
- Get Mathenetion.jl up to speed
- Lag diffusion coeffs in cn_solve
- axially varying wall T
- pressure drop/ergun instead of ideal gas
- validation?
- DtO adjoint backward sweep (first factorize every step, then use stale fact as precond and iterate)
- Adaptive FrozenNewton (monitor contraction rate $\theta_k = \frac{\lVert s_k \rVert}{\lVert s_{k-1}\rVert}$) 
- 2nd order FVM (TVD/MUSCL) -> How to integrate into framework? 
- refine_mesh.jl (r-adaptivity)

