function _tcache(ocp::MethanationOCP, ::Type{T}) where {T}
    p0 = ocp.sa.prob
    tosp(A) = SparseMatrixCSC(A.m, A.n, copy(A.colptr), copy(A.rowval), zeros(T, length(A.nzval)))
    nf = length(p0.dm.fields)
    prob = ProblemCache(tosp(p0.M), tosp(p0.K), tosp(p0.Jr),
        zeros(T, ocp.n), zeros(T, nf), zeros(T, nf, nf),
        T.(p0.f_in), T.(p0.f_wall), p0.grid, p0.geom, p0.dm, p0.model)
    props = MethanationProps{T}(nspecies(p0.model), ncells(p0.grid))
    saT = StateAssembly(prob, props, ocp.sa.K_bc, ocp.sa.dirs, ocp.sa.ebnd)
    buf = (yp=zeros(T, ocp.n), w=zeros(T, ocp.n), Kv=zeros(T, ocp.n), b=zeros(T, ocp.n))
    return (prob=prob, sa=saT, buf=buf)
end

gettcache(ocp::MethanationOCP, ::Type{T}) where {T} = get!(() -> _tcache(ocp, T), ocp.caches, T)

function ocp_cons!(ocp::MethanationOCP, cx, z)
    c = gettcache(ocp, eltype(z))
    return _residual!(cx, z, ocp, c.prob, c.sa, c.buf)
end

# Radau IIA collocation 
function _residual!(cx, z, ocp::MethanationOCP, prob, saT, buf)
    n, s, Ne, Δt = ocp.n, ocp.s, ocp.Ne, ocp.Δt
    nz = n * ncols(ocp)
    Z = reshape(view(z, 1:nz), n, ncols(ocp))
    u = view(z, nz+1:nz+Ne)
    tab = ocp.tab
    dm, m = prob.dm, prob.model
    yp, w, Kv, b = buf.yp, buf.w, buf.Kv, buf.b
    sy, y_off, sc = ocp.sy, ocp.y_off, ocp.sc

    for k in 1:Ne
        ykm1 = view(Z, :, leftcol(ocp, k))
        uk = ocp.u_off + ocp.su * u[k]

        for i in 1:s
            Yi = view(Z, :, stagecol(ocp, k, i))
            @. yp = y_off + sy * Yi # physical stage val

            # w = ∑_j d_ij (Y_k^j − y_{k−1}), physical units
            d0i = tab.d0[i]
            @. w = -d0i * ykm1
            for j in 1:s
                Yj = view(Z, :, stagecol(ocp, k, j))
                dij = tab.D[i, j]
                @. w += dij * Yj
            end
            @. w *= sy

            saT(yp) # M(Y_k^i), K(Y_k^i), f_in
            assemble_react!(prob, yp, Val(false)) # r(Y_k^i)
            # 2nd-order axial correction rides in as a source (K left 1st order).
            # saT(yp) just refreshed props, so they are consistent with yp.
            ad = ocp.ad
            isnothing(ad) ||
                assemble_antidiffusion!(prob, saT.props, ad.st, ad.fs, ad.fg, yp, ad.εf)

            ti = stage_time(tab, k, i, Δt)
            fill!(b, zero(eltype(b)))
            for field in dm.fields
                fd = fielddof(dm, field)
                md = inlet_mod(m, field, ti)
                @views @. b[fd] += prob.f_in[fd] * md
            end
            for d in ocp.wall_dofs # only wall cells carry control
                b[d] += prob.f_wall[d] * uk
            end

            row = ((k - 1) * s + i - 1) * n
            cb = view(cx, row+1:row+n)
            Mnz = nonzeros(prob.M) # M diagonal -> skip structural zeros
            @views @. cb = Mnz[ocp.mdiag] * w
            mul!(Kv, prob.K, yp)
            @. cb = (cb + Δt * Kv - Δt * prob.r - Δt * b) * sc # row scaling
        end
    end
    return cx
end

# Kernel collocation residual (local)
# Same as _residual!, but with local kernels instead of global assembly
# Used only for jac/hess assembly
function _residual_kernels!(cx, z, ocp::MethanationOCP)
    sa = ocp.sa
    p0 = sa.prob
    m, dm, grid, geom = p0.model, p0.dm, p0.grid, p0.geom
    n, s, Ne, Δt = ocp.n, ocp.s, ocp.Ne, ocp.Δt
    nsp = nspecies(m)
    nf = nsp + 1
    fiT = nf
    nzv = n * ncols(ocp)
    Z = reshape(view(z, 1:nzv), n, ncols(ocp))
    u = view(z, nzv+1:nzv+Ne)
    tab = ocp.tab
    sy, y_off, sc = ocp.sy, ocp.y_off, ocp.sc
    Tv = eltype(z)

    yp = zeros(Tv, n)
    w = zeros(Tv, n)
    acc = zeros(Tv, n)
    Kbcv = zeros(Tv, n)

    for k in 1:Ne
        ykm1 = view(Z, :, leftcol(ocp, k))
        uk = ocp.u_off + ocp.su * u[k]

        for i in 1:s
            Yi = view(Z, :, stagecol(ocp, k, i))
            @. yp = y_off + sy * Yi

            d0i = tab.d0[i]
            @. w = -d0i * ykm1
            for j in 1:s
                Yj = view(Z, :, stagecol(ocp, k, j))
                dij = tab.D[i, j]
                @. w += dij * Yj
            end
            @. w *= sy

            ti = stage_time(tab, k, i, Δt)
            fill!(acc, zero(Tv))

            # cell kernels, M(y)·w and reaction source
            for c in 1:ncells(grid)
                ci, cj = cellij(grid, c)
                vol = cellvolume(geom, ci, cj)
                cd = celldof(dm, c)
                yc = view(yp, cd)
                P = props_cell(m, yc)
                re = reaction_cell(m, yc)
                for fi in 1:nf
                    d = cd[fi]
                    acc[d] += vol * capacity_cell(m, P, fi) * w[d] - Δt * vol * re[fi]
                end
            end

            # face kernels, state-dependent transport
            for (fs, fg) in sa.dirs
                for e in eachindex(fs.owner)
                    o, nb = fs.owner[e], fs.neighbor[e]
                    od, nd = celldof(dm, o), celldof(dm, nb)
                    F = face_flux(m, view(yp, od), view(yp, nd),
                        fs.axis, fg.area[e], fg.dist[e], fg.wf[e])
                    for fi in 1:nf
                        acc[od[fi]] += Δt * F[fi]
                        acc[nd[fi]] -= Δt * F[fi]
                    end
                end
            end

            # constant boundary transport (species inlet/outlet + wall), no Hessian
            mul!(Kbcv, sa.K_bc, yp)
            @. acc += Δt * Kbcv

            # state-dependent energy advection at inlet/outlet
            mdT = inlet_mod(m, dm.fields[fiT], ti)
            for (bfs, bfg) in sa.ebnd
                βn = velocity(m, bfs.axis) * bfs.side
                for q in eachindex(bfs.cells)
                    c = bfs.cells[q]
                    gc = dof(dm, c, fiT)
                    _, cj = cellij(grid, c)
                    Kd, fin = bnd_energy_kernel(m, view(yp, celldof(dm, c)),
                        βn, bfg.area[q], geom.rc[cj])
                    acc[gc] += Δt * (Kd * yp[gc] - fin * mdT)
                end
            end

            # constant species inlet forcing + wall control
            for fi in 1:nsp
                fd = fielddof(dm, dm.fields[fi])
                md = inlet_mod(m, dm.fields[fi], ti)
                @views @. acc[fd] -= Δt * p0.f_in[fd] * md
            end
            for d in ocp.wall_dofs
                acc[d] -= Δt * p0.f_wall[d] * uk
            end

            row = ((k - 1) * s + i - 1) * n
            @views @. cx[row+1:row+n] = acc * sc
        end
    end
    return cx
end
