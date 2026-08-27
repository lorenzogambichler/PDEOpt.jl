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
function bisect_shape(forward, Ne::Int; Tmax, Tw_min=300.0, Tw_max=650.0,
    δ=7.0, itmax=20, tol=0.05, a=0.15, b=0.35, c=0.75, d=0.90, verbose::Bool=true)
    Ttgt = Tmax - δ # δ absorbs error (CN, interpolation, etc.)
    sh = [twshape((k - 0.5) / Ne; a=a, b=b, c=c, d=d) for k in 1:Ne]

    function run(λ)
        u = @. λ + (Tw_max - λ) * sh
        Z, pk = forward(u)
        return Z, u, pk
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

# Bisection to find largest constant Tw s.t. max(Tguess) < Tmax - δ
# lo always feasible by construction
function bisect_const(forward, Ne::Int; Tmax, Tw_min=300.0, Tw_max=650.0,
    δ=7.0, itmax=20, tol=0.05, verbose::Bool=true)
    Ttgt = Tmax - δ # δ absorbs error (CN, interpolation, etc.) 

    Zh, Th = forward(Tw_max)
    if Th <= Ttgt
        verbose && @printf("  Tw_max already feasible (max T %.2f K)\n", Th)
        return (Z=Zh, u=fill(Tw_max, Ne), Tpk=Th, Tw=Tw_max, it=1, conv=true)
    end
    Zl, Tl = forward(Tw_min)
    if Tl > Ttgt
        @warn "even Tw_min breaches the cap -- infeasible for this reactor" Tl Ttgt
        return (Z=Zl, u=fill(Tw_min, Ne), Tpk=Tl, Tw=Tw_min, it=2, conv=false)
    end

    lo, hi, it = Tw_min, Tw_max, 2
    best = (Zl, Tw_min, Tl)
    while hi - lo > tol && it < itmax
        mid = 0.5 * (lo + hi)
        Zm, Tm = forward(mid)
        it += 1
        verbose && @printf("  it %2d | Tw %8.3f | max T %8.3f K  %s\n",
            it, mid, Tm, Tm <= Ttgt ? "feasible" : "over")
        Tm <= Ttgt ? (lo = mid; best = (Zm, mid, Tm)) : (hi = mid)
    end
    Z, Tw, Tpk = best
    return (Z=Z, u=fill(Tw, Ne), Tpk=Tpk, Tw=Tw, it=it, conv=true)
end
