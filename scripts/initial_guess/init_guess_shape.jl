using LinearAlgebra, SparseArrays, PDEOpt, Random, Printf, Plots

include("../../apps/methanation/opt.jl")

# Dynamic initial guess
# Fix shape s(τ) ∈ [0,1] and bisect λ, u(t; λ) = λ + (Tw_max - λ) * s(t/tf)
# ∂u/∂λ = 1 - s ≥ 0 -> peak is monotone in λ, bisection still applies

# Tw_max on [0,a], decrease on [a,b], hold on [b,c], increase on [c,d], Tw_max after
function twshape(τ; a=0.15, b=0.35, c=0.75, d=0.90)
    τ < a && return 1.0
    τ < b && return 1.0 - (τ - a) / (b - a)
    τ < c && return 0.0
    τ < d && return (τ - c) / (d - c)
    return 1.0
end

# Bisection to find largest λ s.t. max(Tguess) < Tmax - δ (for fixed shape)
function bisect_shape(tf, ts, Δt, Ne; Tmax, Δt_cn=0.25, Tw_min=300.0, Tw_max=650.0,
    δ=2.0, itmax=16, tol=0.05, a=0.15, b=0.35, c=1.0, d=1.0,
    recon::Symbol=:vanalbada, verbose::Bool=true)

    Ttgt = Tmax - δ
    sh = [twshape((k - 0.5) * Δt / tf; a=a, b=b, c=c, d=d) for k in 1:Ne]

    function run(λ)
        u = @. λ + (Tw_max - λ) * sh
        uf(t) = u[clamp(floor(Int, t / Δt) + 1, 1, Ne)]
        cache, _ = cn_forward(uf, tf, Δt_cn; recon=recon)
        Z = resample(cache.y, Δt_cn, ts)
        return Z, u, maximum(view(Z, fielddof(cache.prob.dm, :T), :))
    end

    Zh, uh, Th = run(Tw_max)
    if Th <= Ttgt
        verbose && @printf("  Tw_max already feasible (max T %.2f K)\n", Th)
        return (Z=Zh, u=uh, Tpk=Th, Tw=Tw_max, it=1, conv=true)
    end
    # lo not automatically feasible, where s = 1 the profile sits at Tw_max whatever λ is
    Zl, ul, Tl = run(Tw_min)
    if Tl > Ttgt
        verbose && @printf("  λ=Tw_min still breaches (max T %.2f K) -> shorten a or raise d\n", Tl)
        return (Z=Zl, u=ul, Tpk=Tl, Tw=Tw_min, it=2, conv=false)
    end

    lo, hi, it = Tw_min, Tw_max, 2
    best = (Zl, ul, Tw_min, Tl)
    while hi - lo > tol && it < itmax
        mid = 0.5 * (lo + hi)
        Zm, um, Tm = run(mid)
        it += 1
        verbose && @printf("  it %2d | λ %8.3f | max T %8.3f K  %s\n",
            it, mid, Tm, Tm <= Ttgt ? "feasible" : "over")
        Tm <= Ttgt ? (lo = mid; best = (Zm, um, mid, Tm)) : (hi = mid)
    end
    Z, u, λ, Tpk = best
    return (Z=Z, u=u, Tpk=Tpk, Tw=λ, it=it, conv=true)
end

function check_guess(; recon::Symbol=:vanalbada, tf=700.0, Ne=30, s=3,
    Tw_min=300.0, Tw_max=650.0, Tmax=750.0, γ=0.01, kwargs...)

    Δt = tf / Ne
    prob, sa, y0 = setup_problem()
    ocp = MethanationOCP(sa, RadauIIA(s), Δt, Ne, y0; Tw_min=Tw_min, Tw_max=Tw_max,
        Tmax=Tmax, γ=γ, co2_in=co2_inflow(sa), recon=recon)

    @printf("recon = :%s | tf = %.0f s | Ne = %d | Δt = %.2f s | s = %d | Tmax = %.1f K\n",
        recon, tf, Ne, Δt, s, Tmax)
    g = bisect_shape(tf, timegrid(ocp), Δt, Ne; Tmax=Tmax, Tw_min=Tw_min, Tw_max=Tw_max,
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
    Jreg = γ * Ne * sum(((g.u[k+1] - g.u[k]) / (Tw_max - Tw_min))^2 for k in 1:Ne-1)

    println()
    @printf("  λ (bisected)       %10.3f K\n", g.Tw)
    @printf("  max T (resampled)  %10.3f K, headroom %+8.3f\n", Tpk, Tmax - Tpk)
    @printf("  z_hot              %10.4f m\n", zhot)
    @printf("  min ρ              %10.3e %s\n", spmin, spmin < 0 ? "  <-- NEGATIVE" : "")
    @printf("  bound clip         %10.3f K\n", max(0.0, Tpk - Tmax))
    @printf("  ‖c(z0)‖∞           %10.3e, (clipped: %.3e)\n", norm(c0, Inf), norm(c0c, Inf))
    @printf("  X_CO2(tf)          %10.4f, mean %.4f, max %.4f\n", Xend, Xmean, maximum(Xt))
    @printf("  ignited            %10s, (X_CO2(tf) > 0.5)\n", ignited ? "yes" : "NO")
    @printf("  obj / Jreg         %10.5f, %.5f\n", ocp_obj(ocp, scale_z(ocp, x0)), Jreg)

    g.conv || @warn "shape infeasible at λ=Tw_min -> shorten a or raise d" Tpk
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
