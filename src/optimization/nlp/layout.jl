struct CollocationLayout{TC<:AbstractCollocationMethod}
    Ne::Int # elements
    Δt::Float64 # timestep element
    ny::Int # states
    nu::Int # controls
    coll::TC # collocation
end

# x = [y0; Y11; ...; Y1s; u1; Y21; ...; Y2s; u2; ...; YNe1; ...; YNes; uNe] (interleaved)
ncols(lay::CollocationLayout) = lay.coll.s * lay.Ne + 1 # y0 + stages (k,i)
elsize(lay::CollocationLayout) = lay.coll.s * lay.ny + lay.nu # vars per element
nvars(lay::CollocationLayout) = lay.ny + lay.Ne * elsize(lay)
ncons(lay::CollocationLayout) = lay.ny * lay.coll.s * lay.Ne

# Index ranges for x
eloffset(lay::CollocationLayout, k) = lay.ny + (k - 1) * elsize(lay)
initial_range(lay::CollocationLayout) = 1:lay.ny
function stage_range(lay::CollocationLayout, k, i)
    off = eloffset(lay, k) + (i - 1) * lay.ny
    return off+1:off+lay.ny
end
function control_range(lay::CollocationLayout, k)
    off = eloffset(lay, k) + lay.coll.s * lay.ny
    return off+1:off+lay.nu
end
function cons_range(lay::CollocationLayout, k, i)
    off = ((k - 1) * lay.coll.s + i - 1) * lay.ny
    return off+1:off+lay.ny
end
# assumes stiffly accurate (e.g. Radau IIA)
left_range(lay::CollocationLayout, k) =
    k == 1 ? initial_range(lay) : stage_range(lay, k - 1, lay.coll.s)

initial(lay::CollocationLayout, x) = view(x, initial_range(lay))
stage(lay::CollocationLayout, x, k, i) = view(x, stage_range(lay, k, i))
control(lay::CollocationLayout, x, k) = view(x, control_range(lay, k))
left(lay::CollocationLayout, x, k) = view(x, left_range(lay, k))

stagecol(lay::CollocationLayout, k, i) = 1 + (k - 1) * lay.coll.s + i # col of stage (k,i) in traj grid

# Trajectory <-> x, Y (ny × ncols, col 1 = y0), U (nu × Ne)
function pack!(x::AbstractVector, lay::CollocationLayout, Y::AbstractMatrix, U::AbstractMatrix)
    size(Y) == (lay.ny, ncols(lay)) ||
        throw(DimensionMismatch("Y is $(size(Y)), expected $((lay.ny, ncols(lay)))"))
    size(U) == (lay.nu, lay.Ne) ||
        throw(DimensionMismatch("U is $(size(U)), expected $((lay.nu, lay.Ne))"))
    initial(lay, x) .= view(Y, :, 1)
    for k in 1:lay.Ne
        for i in 1:lay.coll.s
            stage(lay, x, k, i) .= view(Y, :, stagecol(lay, k, i))
        end
        control(lay, x, k) .= view(U, :, k)
    end
    return x
end
pack(lay::CollocationLayout, Y::AbstractMatrix, U::AbstractMatrix) =
    pack!(Vector{Float64}(undef, nvars(lay)), lay, Y, U)
pack(lay::CollocationLayout, Y::AbstractMatrix, u::AbstractVector) = # nu = 1
    pack(lay, Y, reshape(u, 1, :))

function unpack(lay::CollocationLayout, x::AbstractVector)
    Y = Matrix{Float64}(undef, lay.ny, ncols(lay))
    U = Matrix{Float64}(undef, lay.nu, lay.Ne)
    Y[:, 1] .= initial(lay, x)
    for k in 1:lay.Ne
        for i in 1:lay.coll.s
            Y[:, stagecol(lay, k, i)] .= stage(lay, x, k, i)
        end
        U[:, k] .= control(lay, x, k)
    end
    return (Y=Y, U=U)
end

element_times(lay::CollocationLayout) = [(k - 1) * lay.Δt for k in 1:lay.Ne]
function stage_times(lay::CollocationLayout)
    t = zeros(ncols(lay))
    for k in 1:lay.Ne
        for i in 1:lay.coll.s
            t[stagecol(lay, k, i)] = stage_time(lay.coll, k, i, lay.Δt)
        end
    end
    return t
end
