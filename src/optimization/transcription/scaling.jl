struct Scaling
    # TODO
end

# Reference scales, sy/y_off (vars), sc (constr rows)
function _scales(sa, y0::Vector{Float64}, Tmin::Float64, Tmax::Float64)
    prob = sa.prob
    dm, m, grid, geom = prob.dm, prob.model, prob.grid, prob.geom
    nsp = nspecies(m)
    ctot = sum(y0[dof(dm, 1, α)] / m.M[α] for α in 1:nsp) # inlet conc
    ΔT = Tmax - Tmin
    ρcp_ref = (1 - m.ε) * m.ρcat * m.cp_cat

    sy = zeros(ndof(dm))
    y_off = zeros(ndof(dm))
    sc = zeros(ndof(dm))
    for c in 1:ncells(grid)
        i, j = cellij(grid, c)
        V = cellvolume(geom, i, j)
        for α in 1:nsp
            d = dof(dm, c, α)
            sy[d] = ctot * m.M[α] # ρ scaling
            sc[d] = 1 / (m.ε * V * sy[d])
        end
        d = dof(dm, c, nsp + 1)
        sy[d] = ΔT # T scaling
        y_off[d] = Tmin # T offset
        sc[d] = 1 / (ρcp_ref * V * ΔT)
    end
    return sy, y_off, sc
end

# Physical to scaled
function scale_z(ocp::MethanationOCP, x::AbstractVector)
    nz = ocp.n * ncols(ocp)
    z = similar(x, Float64)
    Z = reshape(view(z, 1:nz), ocp.n, ncols(ocp))
    X = reshape(view(x, 1:nz), ocp.n, ncols(ocp))
    @. Z = (X - ocp.y_off) / ocp.sy
    @views @. z[nz+1:end] = (x[nz+1:end] - ocp.u_off) / ocp.su
    return z
end

# Scaled to physical
function unscale_z(ocp::MethanationOCP, z::AbstractVector)
    nz = ocp.n * ncols(ocp)
    Z = reshape(z[1:nz], ocp.n, ncols(ocp))
    @. Z = ocp.y_off + ocp.sy * Z
    u = ocp.u_off .+ ocp.su .* z[nz+1:end]
    return Z, u
end

scale_z = nothing # TODO
unscale_z = nothing # TODO
