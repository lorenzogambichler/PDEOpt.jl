# Inlet/outlet flow rates and conversion for nlp solution

# Σ_j vz·A_j·g(r_j)·ρ_i,in
inflow(sa::StateAssembly, spec::Symbol, t::Real=0.0) =
    sum(@view sa.prob.f_in[fielddof(sa.prob.dm, spec)]) * inlet_mod(sa.prob.model, spec, t)

co2_inflow(sa, t::Real=0.0) = inflow(sa, :CO2, t)

# X_i(t) = 1 − ṅ_i,out(t)/ṅ_i,in (outlet)
# Y any grid as long as sa same
function outlet_conv(sa::StateAssembly, Y::AbstractMatrix, spec::Symbol=:CO2)
    bfs_out, bfg_out = sa.ebnd[2]
    dofs = [dof(sa.prob.dm, c, spec) for c in bfs_out.cells]
    w = velocity(sa.prob.model, 1) .* bfg_out.area
    Fin = inflow(sa, spec)
    return [1 - dot(w, view(Y, dofs, col)) / Fin for col in axes(Y, 2)]
end

# mean outlet X_i over [0,tf] (Radau quad)
# X on stage columns (stage_times(lay)), can be resampled
function mean_conv(lay::CollocationLayout, X::AbstractVector)
    length(X) == ncols(lay) || error("X has $(length(X)) cols, expected $(ncols(lay))")
    Xbar = 0.0
    for k in 1:lay.Ne, i in 1:lay.coll.s
        Xbar += lay.coll.b[i] * X[stagecol(lay, k, i)]
    end
    return Xbar / lay.Ne
end

mean_conv(sa::StateAssembly, lay::CollocationLayout, Y::AbstractMatrix, spec::Symbol=:CO2) =
    mean_conv(lay, outlet_conv(sa, Y, spec))
