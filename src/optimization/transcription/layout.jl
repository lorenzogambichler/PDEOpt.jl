struct CollocationLayout{TC}
    Ne::Int
    Δt::Float64
    ny::Int # spatial
    nu::Int # control
    coll::TC # AbstractCollocationMethod
end

ncols(ocp::MethanationOCP) = ocp.s * ocp.Ne + 1
nvars(ocp::MethanationOCP) = ocp.n * ncols(ocp) + ocp.Ne
ncons(ocp::MethanationOCP) = ocp.n * ocp.s * ocp.Ne

leftcol(lay::CollocationLayout, k) = 1 + (k - 1) * lay.coll.s
stagecol(lay::CollocationLayout, k, i) = 1 + (k - 1) * lay.coll.s + i 

function timegrid(lay::CollocationLayout)
    t = zeros(lay.ncols)
    for k in 1:lay.Ne
        for i in 1:lay.coll.s
            t[stagecol(lay, k, i)] = stage_time(lay.coll.tab, k, i, lay.Δt)
        end
    end
    return t
end

control_times(lay::CollocationLayout) = [(k - 1) * lay.Δt for k in 1:lay.Ne]