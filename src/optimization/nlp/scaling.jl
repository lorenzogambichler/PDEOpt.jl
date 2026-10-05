struct Scaling
    sy::Vector{Float64} # state scaling (ny)
    y_off::Vector{Float64} # state offset (ny)
    su::Vector{Float64} # controls scaling (nu)
    u_off::Vector{Float64} # controls offset (nu)
    sc::Vector{Float64} # constr scaling, per stage row block (ny)
end

function scale_x!(x_sc, lay::CollocationLayout, scl::Scaling, x)
    r = initial_range(lay)
    @views @. x_sc[r] = (x[r] - scl.y_off) / scl.sy
    for k in 1:lay.Ne
        for i in 1:lay.coll.s
            r = stage_range(lay, k, i)
            @views @. x_sc[r] = (x[r] - scl.y_off) / scl.sy # states
        end
        r = control_range(lay, k)
        @views @. x_sc[r] = (x[r] - scl.u_off) / scl.su # controls
    end
    return x_sc
end
scale_x(lay::CollocationLayout, scl::Scaling, x) = scale_x!(similar(x, Float64), lay, scl, x)

function unscale_x!(x, lay::CollocationLayout, scl::Scaling, x_sc)
    r = initial_range(lay)
    @views @. x[r] = scl.y_off + scl.sy * x_sc[r]
    for k in 1:lay.Ne
        for i in 1:lay.coll.s
            r = stage_range(lay, k, i)
            @views @. x[r] = scl.y_off + scl.sy * x_sc[r]
        end
        r = control_range(lay, k)
        @views @. x[r] = scl.u_off + scl.su * x_sc[r]
    end
    return x
end
unscale_x(lay::CollocationLayout, scl::Scaling, x_sc) = unscale_x!(similar(x_sc, Float64), lay, scl, x_sc)

scale_bounds(lay::CollocationLayout, scl::Scaling, lvar::AbstractVector, uvar::AbstractVector) = 
    (scale_x(lay, scl, lvar), scale_x(lay, scl, uvar))