# Inlet/outlet flow rates and conversion from a solved state
# Only needs a StateAssembly, so it also works on CN re-simulation output

# Σ_j vz·A_j·g(r_j)·ρ_i,in
inflow(sa, spec::Symbol, t::Real=0.0) =
    sum(@view sa.prob.f_in[fielddof(sa.prob.dm, spec)]) * inlet_mod(sa.prob.model, spec, t)

co2_inflow(sa, t::Real=0.0) = inflow(sa, :CO2, t)

# X_i(t) = 1 − ṅ_i,out(t)/ṅ_i,in (outlet)
# Y on any grid as long as sa same
function outlet_conv(sa::StateAssembly, Y::AbstractMatrix, spec::Symbol=:CO2)
    bfs_out, bfg_out = sa.ebnd[2]
    dofs = [dof(sa.prob.dm, c, spec) for c in bfs_out.cells]
    w = velocity(sa.prob.model, 1) .* bfg_out.area
    Fin = inflow(sa, spec)
    return [1 - dot(w, view(Y, dofs, col)) / Fin for col in axes(Y, 2)]
end

outlet_conv(ocp::MethanationOCP, Z::AbstractMatrix, spec::Symbol=:CO2) =
    outlet_conv(ocp.sa, Z, spec)

# mean outlet X_i over [0,tf] (Radau quad)
# X must be on ocp stage columns but can be resampled from any grid
function mean_conv(ocp::MethanationOCP, X::AbstractVector)
    length(X) == ncols(ocp) || error("X has $(length(X)) cols, expected $(ncols(ocp))")
    Xbar = 0.0
    for k in 1:ocp.Ne, i in 1:ocp.s
        Xbar += ocp.tab.b[i] * X[stagecol(ocp, k, i)]
    end
    return Xbar / ocp.Ne
end

mean_conv(ocp::MethanationOCP, Z::AbstractMatrix, spec::Symbol=:CO2) =
    mean_conv(ocp, outlet_conv(ocp.sa, Z, spec))
