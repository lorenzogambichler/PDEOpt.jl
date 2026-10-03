struct Bounds
    y_low::Vector{Float64} # (ny)
    y_up::Vector{Float64} # (ny)
    u_low::Vector{Float64} # (nu) 
    u_up::Vector{Float64} # (nu)
end

function assemble_bound_vecs(lay::CollocationLayout, bnd::Bounds, y0::AbstractVector)
    lvar = Vector{Float64}(undef, nvars(lay))
    uvar = Vector{Float64}(undef, nvars(lay))
    initial(lay, lvar) .= y0
    initial(lay, uvar) .= y0
    for k in 1:lay.Ne
        for i in 1:lay.coll.s
            stage(lay, lvar, k, i) .= bnd.y_low
            stage(lay, uvar, k, i) .= bnd.y_up
        end
        control(lay, lvar, k) .= bnd.u_low
        control(lay, uvar, k) .= bnd.u_up
    end
    return lvar, uvar
end