function retype end

struct CollocationResidual{TL,TS,TD}
    lay::TL # CollocationLayout
    scl::TS # Scaling
    dyn::TD # dynamics
    caches::Dict{DataType,Any} # for each AD eltype
end

CollocationResidual(lay, scl, dyn) =
    CollocationResidual(lay, scl, dyn, Dict{DataType,Any}())

function _residual_cache(rsd::CollocationResidual, ::Type{T}) where {T}
    ny, nu = rsd.lay.ny, rsd.lay.nu
    # retype(dyn, T) -> copy of dyn for AD
    return (dyn=retype(rsd.dyn, T), yp=zeros(T, ny), yd=zeros(T, ny),
        res=zeros(T, ny), u=zeros(T, nu))
end

get_residual_cache(rsd::CollocationResidual, ::Type{T}) where {T} =
    get!(() -> _residual_cache(rsd, T), rsd.caches, T)

# c from Dict is ::Any -> function barrier
(rsd::CollocationResidual)(cx, x) =
    _residual!(cx, x, rsd.lay, rsd.scl, get_residual_cache(rsd, eltype(x)))

function _residual!(cx, x, lay, scl, c)
    coll = lay.coll
    s, Ne, Δt = coll.s, lay.Ne, lay.Δt
    dyn, yp, yd, res, u = c.dyn, c.yp, c.yd, c.res, c.u

    for k in 1:Ne
        ykm1 = left(lay, x, k)
        uk = control(lay, x, k)
        @. u = scl.u_off + scl.su * uk # physical control

        for i in 1:s
            Yi = stage(lay, x, k, i)
            @. yp = scl.y_off + scl.sy * Yi # physical stage val

            # ẏ(t_i) = (∑_j D_ij Y_k^j − d0_i y_{k−1})·sy/Δt
            d0i = coll.d0[i]
            @. yd = -d0i * ykm1
            for j in 1:s
                dij = coll.D[i, j]
                Yj = stage(lay, x, k, j)
                @. yd += dij * Yj
            end
            @. yd *= scl.sy / Δt

            # dyn(res, y, ẏ, t, u) -> res = M(y)·ẏ − F(y,t,u), physical units
            dyn(res, yp, yd, stage_time(coll, k, i, Δt), u)

            cb = view(cx, cons_range(lay, k, i))
            @. cb = Δt * scl.sc * res # row scaling
        end
    end
    return cx
end

