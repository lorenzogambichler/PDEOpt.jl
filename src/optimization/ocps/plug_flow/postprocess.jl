# Inlet/outlet flow rates and conversion (constant transport)

# Outlet dofs of field and advective weights cap·β·A
function plug_flow_outlet(prob::ProblemCache, field::Symbol)
    grid, geom, dm, m = prob.grid, prob.geom, prob.dm, prob.model
    bfs_out = boundary_faces(grid).outlet
    bfg_out = BoundaryFaceGeometry(grid, geom, bfs_out)
    dofs = [dof(dm, c, field) for c in bfs_out.cells]
    w = capacity(m, fieldindex(dm, field)) * velocity(m, bfs_out.axis) .* bfg_out.area
    return dofs, w
end

# Σ_j f_in,j·y_in(t)
inflow(prob::ProblemCache, field::Symbol, t::Real=0.0) =
    sum(@view prob.f_in[fielddof(prob.dm, field)]) * inlet_mod(prob.model, field, t)

# X(t) = 1 − ṅ_out(t)/ṅ_in (outlet)
function outlet_conv(prob::ProblemCache, Y::AbstractMatrix, field::Symbol=:C)
    dofs, w = plug_flow_outlet(prob, field)
    Fin = inflow(prob, field)
    return [1 - dot(w, view(Y, dofs, col)) / Fin for col in axes(Y, 2)]
end

mean_conv(prob::ProblemCache, lay::CollocationLayout, Y::AbstractMatrix, field::Symbol=:C) =
    mean_conv(lay, outlet_conv(prob, Y, field))
