using LinearAlgebra, SparseArrays, PDEOpt, Random, Printf, CairoMakie

include("../../apps/methanation/opt.jl")

# Original feedback law guess
# 1) P law on peak T -> continuous Tw(t), 2) average onto elements, re-simulate
# Cap only seen by pass 1, so pass 2 can overshoot
function feedback_guess(tf, ts, Δt, Ne; Tmax, Δt_cn=0.25, Tw_min=300.0, Tw_max=650.0,
    δ=2.0, Tset=680.0, Kp=10.0, τ=10.0, recon::Symbol=:vanalbada, verbose::Bool=true)

    peak = Ref(0.0)
    Tw = Ref(Tw_max)
    tprev = Ref(-1.0)
    log = Tuple{Float64,Float64}[]
    function Twctl(t)
        if t > tprev[]
            raw = clamp(Tw_max - Kp * max(0.0, peak[] - Tset), Tw_min, Tw_max)
            a = min(1.0, (t - max(tprev[], 0.0)) / τ)
            Tw[] += a * (raw - Tw[])
            tprev[] = t
            push!(log, (t, Tw[]))
        end
        return Tw[]
    end

    cn_forward(Twctl, tf, Δt_cn; recon=recon,
        hook=(y, prob) -> (peak[] = maximum(@view y[fielddof(prob.dm, :T)])))

    u = fill(Tw_max, Ne)
    for k in 1:Ne
        v = [w for (t, w) in log if (k - 1) * Δt <= t < k * Δt]
        isempty(v) && error("no Tw samples in element $k; need Δt_cn ($Δt_cn) < Δt ($Δt)")
        u[k] = sum(v) / length(v)
    end

    uf(t) = u[clamp(floor(Int, t / Δt) + 1, 1, Ne)]
    cache, _ = cn_forward(uf, tf, Δt_cn; recon=recon)
    Z = resample(cache.y, Δt_cn, ts)
    Tpk = maximum(view(Z, fielddof(cache.prob.dm, :T), :))
    conv = Tpk <= Tmax - δ
    verbose && @printf("  Tset %7.2f | max T %8.3f K  %s\n", Tset, Tpk, conv ? "feasible" : "over")
    return (Z=Z, u=u, Tpk=Tpk, Tset=Tset, it=2, conv=conv)
end

function check_guess(; recon::Symbol=:vanalbada, tf=700.0, Ne=30, s=3,
    Tw_min=300.0, Tw_max=650.0, Tmax=750.0, γ=0.01, kwargs...)

    Δt = tf / Ne
    prob, sa, y0 = setup_problem()
    ocp = MethanationOCP(sa, RadauIIA(s), Δt, Ne, y0; Tw_min=Tw_min, Tw_max=Tw_max,
        Tmax=Tmax, γ=γ, co2_in=co2_inflow(sa), recon=recon)

    @printf("recon = :%s | tf = %.0f s | Ne = %d | Δt = %.2f s | s = %d | Tmax = %.1f K\n",
        recon, tf, Ne, Δt, s, Tmax)
    g = feedback_guess(tf, timegrid(ocp), Δt, Ne; Tmax=Tmax, Tw_min=Tw_min, Tw_max=Tw_max,
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
    @printf("  Tset               %10.3f K\n", g.Tset)
    @printf("  max T (resampled)  %10.3f K, headroom %+8.3f\n", Tpk, Tmax - Tpk)
    @printf("  z_hot              %10.4f m\n", zhot)
    @printf("  min ρ              %10.3e %s\n", spmin, spmin < 0 ? "  <-- NEGATIVE" : "")
    @printf("  bound clip         %10.3f K\n", max(0.0, Tpk - Tmax))
    @printf("  ‖c(z0)‖∞           %10.3e, (clipped: %.3e)\n", norm(c0, Inf), norm(c0c, Inf))
    @printf("  X_CO2(tf)          %10.4f, mean %.4f, max %.4f\n", Xend, Xmean, maximum(Xt))
    @printf("  ignited            %10s, (X_CO2(tf) > 0.5)\n", ignited ? "yes" : "NO")
    @printf("  obj / Jreg         %10.5f, %.5f\n", ocp_obj(ocp, scale_z(ocp, x0)), Jreg)

    spmin < 0 && @warn "negative species density in CN guess -> limiter ε too large" spmin
    Tpk > Tmax && @warn "guess breaches Tmax" Tpk Tmax

    return (ocp=ocp, prob=prob, sa=sa, y0=y0, Z=Z, u=g.u, Tset=g.Tset, Tpk=Tpk,
        cinf=norm(c0, Inf), cinf_clipped=norm(c0c, Inf), it=g.it, conv=g.conv,
        ignited=ignited, Xend=Xend, Xmean=Xmean)
end

# Control (piecewise constant on elements) and CO2 conversion
function plot_guess(res)
    ocp = res.ocp
    te = (0:ocp.Ne) .* ocp.Δt # element edges
    ts = timegrid(ocp)

    fig = Figure(size=(900, 360))

    ax1 = Axis(fig[1, 1]; xlabel="t [s]", ylabel="Tw [K]", title="control")
    stairs!(ax1, te, vcat(res.u, res.u[end]); step=:post, linewidth=2)
    hlines!(ax1, [ocp.Tw_max]; linestyle=:dash, linewidth=1, color=:gray)

    ax2 = Axis(fig[1, 2]; xlabel="t [s]", ylabel="X_CO2 [-]", title="conversion")
    lines!(ax2, ts, outlet_conv(ocp, res.Z); linewidth=2)
    ylims!(ax2, 0, 1)

    return fig
end

res = check_guess(recon=:vanalbada);
plot_guess(res)
