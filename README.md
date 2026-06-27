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
