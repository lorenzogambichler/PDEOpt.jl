### TODOs
- T-dependent reaction params, transport params
- Lag diffusion coeffs
- axially varying wall T
- pressure drop/ergun
- validation?
- DtO adjoint backward sweep (first factorize every step, then use stale fact as precond and iterate)
- Adaptive FrozenNewton (monitor contraction rate $\theta_k = \frac{\lVert s_k \rVert}{\lVert s_{k-1}\rVert}$) 
- 2nd order FVM (TVD/MUSCL) -> How to integrate into framework? 
- refine_mesh.jl (r-adaptivity)

