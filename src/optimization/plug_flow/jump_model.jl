# Full-space DtO (symbolic, JuMP for AD)

using JuMP
using Ipopt
using HSL_jll
using PDEOpt

struct OptCache
    γ::Float64
    Tmax::Float64
    Tw_min::Float64
    Tw_max::Float64
    y0::Vector{Float64}
    y_guess::Matrix{Float64}
end

function build_nlp(prob::ProblemCache, m::PlugFlowModel, opt::OptCache, Δt::Float64, N::Int)
    # Params
    dm = prob.dm
    n = ndof(dm)
    A = prob.M .+ 0.5Δt .* prob.K
    B = prob.M .- 0.5Δt .* prob.K
    Cd = fielddof(dm, :C)
    Td = fielddof(dm, :T)

    model = Model(Ipopt.Optimizer)

    # IPOPT options
    set_optimizer_attribute(model, "tol", 1e-6)
    set_optimizer_attribute(model, "acceptable_tol", 1e-4)
    set_optimizer_attribute(model, "acceptable_obj_change_tol", 1e-6)
    set_optimizer_attribute(model, "acceptable_iter", 10)
    set_optimizer_attribute(model, "mu_strategy", "monotone")

    # HSL
    set_attribute(model, "hsllib", HSL_jll.libhsl_path)
    set_attribute(model, "linear_solver", "ma97")

    # HSL Multi-threading
    set_attribute(model, "ma97_scaling", "none") 
    set_attribute(model, "ma97_order", "metis")

    # Scale vars to O(1)
    sy = ones(n)
    for field in dm.fields
        fd = fielddof(dm, field)
        sy[fd] .= max(maximum(abs, opt.y_guess[fd, :]), 1e-8)
    end
    sTw = 0.5 * (opt.Tw_min + opt.Tw_max)

    # decision vars (scaled)
    @variable(model, yscaled[1:n, 1:N])
    set_lower_bound.(yscaled[Td, :], m.react.T_floor / sy[Td[1]])
    set_upper_bound.(yscaled[Td, :], opt.Tmax / sy[Td[1]])
    set_lower_bound.(yscaled[Cd, :], 1e-4) # degeneration safeguard for IPOPT
    set_start_value.(yscaled, opt.y_guess ./ sy)
    y = sy .* yscaled

    @variable(model, Twtil[1:N])
    set_lower_bound.(Twtil, opt.Tw_min / sTw)
    set_upper_bound.(Twtil, opt.Tw_max / sTw)
    set_start_value.(Twtil, (wall_mod(m, 0.0) / sTw))
    Tw = sTw .* Twtil

    # IC
    @constraint(model, y[:, 1] .== opt.y0)

    # Reaction
    r = Matrix{Any}(undef, n, N)
    fill!(r, 0.0)
    for k = 1:N
        for c = 1:ncells(prob.grid)
            i, j = cellij(prob.grid, c)
            vol  = cellvolume(prob.geom, i, j)
            dC = dof(dm, c, :C)
            dT = dof(dm, c, :T)
            sC, sT = reaction_source(m.react, y[dC, k], y[dT, k])
            r[dC, k] = vol * sC
            r[dT, k] = vol * sT
        end
    end

    # Inlet forcing
    # b[:,k] = f_in .* yin(k) + f_wall .* Tw(k);
    bfix  = zeros(n, N)
    for k in 1:N
        t = (k-1)*Δt
        for field in dm.fields
            fd = fielddof(dm, field)
            @views bfix[fd, k] .= prob.f_in[fd] .* inlet_mod(m, field, t)
        end
    end
    b(k) = bfix[:, k] .+ prob.f_wall .* Tw[k]

    # ODE constraint (CN)
    for k = 1:N-1
        @constraint(model,
            A * y[:, k+1] .- B * y[:, k]
            .- 0.5*Δt .* (r[:, k+1] .+ r[:, k])
            .- 0.5*Δt .* (b(k+1) .+ b(k)) .== 0)
    end

    # Objective
    bfs = boundary_faces(prob.grid)
    bfg_outlet = BoundaryFaceGeometry(prob.grid, prob.geom, bfs.outlet)
    outlet_cells = bfs.outlet.cells
    @objective(model, Min,
        sum(velocity(m, 1) * bfg_outlet.area[j] * y[dof(dm, c, :C), k]
        for (j, c) in enumerate(outlet_cells), k in 1:N)
        + opt.γ * sum((Twtil[k+1]-Twtil[k])^2 for k in 1:N-1))
    # TODO proper quadrature?

    return model, (; y, u = Tw)
end

function build_nlp(prob::ProblemCache, m::MethanationModel, opt::OptCache, Δt::Float64, N::Int)
    # TODO
end

function solve_nlp!(model, vars)
    optimize!(model)
    is_solved_and_feasible(model) ||
        @warn "NLP error" termination_status(model)
    return (; y = value.(vars.y), u = value.(vars.u),
        objective = objective_value(model),
        status = termination_status(model))
end

#end