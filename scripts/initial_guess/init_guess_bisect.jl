using LinearAlgebra, SparseArrays, PDEOpt, Random, Printf, Plots

include("../../apps/methanation/opt.jl")

# Bisection to find largest constant Tw s.t. max(Tguess) < Tmax - δ
# lo always feasible by construction
function bisect_const(tf, ts, Δt, Ne; Tmax, Δt_cn=0.25, Tw_min=300.0, Tw_max=650.0,
    δ=2.0, itmax=16, tol=0.05, recon::Symbol=:vanalbada, verbose::Bool=true)

    Ttgt = Tmax - δ
    function run(Tw)
        cache, _ = cn_forward(t -> Tw, tf, Δt_cn; recon=recon)
        Z = resample(cache.y, Δt_cn, ts)
        return Z, maximum(view(Z, fielddof(cache.prob.dm, :T), :))
    end

    Zh, Th = run(Tw_max)
    if Th <= Ttgt
        verbose && @printf("  Tw_max already feasible (max T %.2f K)\n", Th)
        return (Z=Zh, u=fill(Tw_max, Ne), Tpk=Th, Tw=Tw_max, it=1, conv=true)
    end
    Zl, Tl = run(Tw_min)
    if Tl > Ttgt
        @warn "even Tw_min breaches the cap -- infeasible for this reactor" Tl Ttgt
        return (Z=Zl, u=fill(Tw_min, Ne), Tpk=Tl, Tw=Tw_min, it=2, conv=false)
    end

    lo, hi, it = Tw_min, Tw_max, 2
    best = (Zl, Tw_min, Tl)
    while hi - lo > tol && it < itmax
        mid = 0.5 * (lo + hi)
        Zm, Tm = run(mid)
        it += 1
        verbose && @printf("  it %2d | Tw %8.3f | max T %8.3f K  %s\n",
            it, mid, Tm, Tm <= Ttgt ? "feasible" : "over")
        Tm <= Ttgt ? (lo = mid; best = (Zm, mid, Tm)) : (hi = mid)
    end
    Z, Tw, Tpk = best
    return (Z=Z, u=fill(Tw, Ne), Tpk=Tpk, Tw=Tw, it=it, conv=true)
end

function check_guess(; recon::Symbol=:vanalbada, tf=700.0, Ne=30, s=3,
    Tw_min=300.0, Tw_max=650.0, Tmax=750.0, γ=0.01, kwargs...)

    Δt = tf / Ne
    prob, sa, y0 = setup_problem()
    ocp = MethanationOCP(sa, RadauIIA(s), Δt, Ne, y0; Tw_min=Tw_min, Tw_max=Tw_max,
        Tmax=Tmax, γ=γ, co2_in=co2_inflow(sa), recon=recon)

    @printf("recon = :%s | tf = %.0f s | Ne = %d | Δt = %.2f s | s = %d | Tmax = %.1f K\n",
        recon, tf, Ne, Δt, s, Tmax)
    g = bisect_const(tf, timegrid(ocp), Δt, Ne; Tmax=Tmax, Tw_min=Tw_min, Tw_max=Tw_max,
        recon=recon, kwargs...)

    dm, Z = prob.dm, g.Z
    Td = fielddof(dm, :T)
    Tpk, idx = findmax(view(Z, Td, :))
    zhot = prob.geom.zc[cellij(prob.grid, idx[1])[1]]
    spmin = minimum(view(Z, setdiff(1:ndof(dm), Td), :))

    # Consistency of guess on collocation grid (inf_pr)
    x0 = vcat(vec(Z), g.u)
    c0 = zeros(ncons(ocp))
    ocp_cons!(ocp, c0, scale_z(ocp, x0))

    # Clipped consistency (inf_pr after clipping)
    Zc = copy(Z)
    @views Zc[Td, :] .= min.(Zc[Td, :], Tmax)
    c0c = zeros(ncons(ocp))
    ocp_cons!(ocp, c0c, scale_z(ocp, vcat(vec(Zc), g.u)))

    Xt = outlet_conv(ocp, Z)
    Xend, Xmean = Xt[end], mean_conv(ocp, Z)
    ignited = Xend > 0.5 # Ingition proxy

    println()
    @printf("  Tw (bisected)      %10.3f K\n", g.Tw)
    @printf("  max T (resampled)  %10.3f K, headroom %+8.3f\n", Tpk, Tmax - Tpk)
    @printf("  z_hot              %10.4f m\n", zhot)
    @printf("  min ρ              %10.3e %s\n", spmin, spmin < 0 ? "  <-- NEGATIVE" : "")
    @printf("  bound clip         %10.3f K\n", max(0.0, Tpk - Tmax))
    @printf("  ‖c(z0)‖∞           %10.3e, (clipped: %.3e)\n", norm(c0, Inf), norm(c0c, Inf))
    @printf("  X_CO2(tf)          %10.4f, mean %.4f, max %.4f\n", Xend, Xmean, maximum(Xt))
    @printf("  ignited            %10s, (X_CO2(tf) > 0.5)\n", ignited ? "yes" : "NO")

    spmin < 0 && @warn "negative species density in CN guess -> limiter ε too large" spmin
    Tpk > Tmax && @warn "guess breaches Tmax" Tpk Tmax

    return (ocp=ocp, prob=prob, sa=sa, y0=y0, Z=Z, u=g.u, Tw=g.Tw, Tpk=Tpk,
        cinf=norm(c0, Inf), cinf_clipped=norm(c0c, Inf), it=g.it, conv=g.conv,
        ignited=ignited, Xend=Xend, Xmean=Xmean)
end

# Control (piecewise constant on elements) and CO2 conversion
function plot_guess(res)
    ocp = res.ocp
    te = (0:ocp.Ne) .* ocp.Δt # element edges
    ts = timegrid(ocp)

    p1 = plot(te, vcat(res.u, res.u[end]); seriestype=:steppost, lw=2, legend=false,
        xlabel="t [s]", ylabel="Tw [K]", title="control")
    hline!(p1, [ocp.Tw_max]; ls=:dash, lw=1, c=:gray)
    p2 = plot(ts, outlet_conv(ocp, res.Z); lw=2, legend=false, ylims=(0, 1),
        xlabel="t [s]", ylabel="X_CO2 [-]", title="conversion")
    return plot(p1, p2; layout=(1, 2), size=(900, 360),
        left_margin=6Plots.mm, bottom_margin=6Plots.mm)
end

res = check_guess();
plot_guess(res)