# Figures for MethanationOpt.jl

using PDEOpt
using CairoMakie
using Printf

const APPDIR = @__DIR__
const RESULTS = joinpath(APPDIR, "results", "Methanation")
const FIGDIR = joinpath(RESULTS, "figures")

const TMAX = 750.0
const TW_MAX = 650.0

# dataviz categorical slots 1-3
const C_COARSE = colorant"#2a78d6"
const C_GUESS = colorant"#eb6834"
const C_FINE = colorant"#1baf7a"
const C_LIMIT = colorant"#e34948"
const C_GRID = colorant"#b8b7b2"

const CMAP_T = :thermal # same ramp wherever T is shown as a field

# Data

struct Case
    name::String
    color::CairoMakie.Colorant
    s::VTRSeries
    ut::Vector{Float64} # u switching times
    u::Vector{Float64}
end

function read_control(path::AbstractString)
    t = Float64[]
    u = Float64[]
    for r in readlines(path)[2:end]
        isempty(strip(r)) && continue
        a, b = split(strip(r), ',')
        push!(t, parse(Float64, a))
        push!(u, parse(Float64, b))
    end
    return t, u
end

function load(tag::String, name::String, color)
    s = read_series(joinpath(RESULTS, "$(tag)_state.pvd"))
    ctrl = startswith(tag, "opt") ? "opt_control.csv" :
           startswith(tag, "guess") ? "guess_control.csv" :
           error("no control file for tag $tag")
    ut, u = read_control(joinpath(RESULTS, ctrl))
    return Case(name, color, s, ut, u)
end

griddims(s::VTRSeries) = (length(s.z) - 1, length(s.r) - 1)

# Add grid specifics to curves
gridlabel(c::Case) = @sprintf("%s  (%d×%d)", c.name, griddims(c.s)...)

# Normalized annular ring areas (for radial average)
ringareas(s::VTRSeries) =
    (a = [pi * (s.r[j+1]^2 - s.r[j]^2) for j = 1:length(s.r)-1]; a ./ sum(a))

# area-weighted rad mean -> (nz, nt)
function radialmean(s::VTRSeries, f::AbstractString)
    A = ringareas(s)
    x = s[f]
    return dropdims(sum(x .* reshape(A, 1, :, 1); dims=2); dims=2)
end

# Tmax cell at each time
peak(s::VTRSeries, f::AbstractString) = vec(maximum(s[f]; dims=(1, 2)))

# Outlet conversion X_i(t) (needs uniform vz)
function outlet_conv(s::VTRSeries, f::AbstractString="CO2")
    A = ringareas(s)
    x = s[f]
    feed = sum(A .* x[1, :, 1])
    return [1 - sum(A .* x[end, :, n]) / feed for n in axes(x, 3)]
end

# cell edges from sample points (heatmaps)
function edges(c::AbstractVector)
    length(c) == 1 && return [c[1] - 0.5, c[1] + 0.5]
    e = similar(c, length(c) + 1)
    e[2:end-1] .= 0.5 .* (c[1:end-1] .+ c[2:end])
    e[1] = c[1] - (c[2] - c[1]) / 2
    e[end] = c[end] + (c[end] - c[end-1]) / 2
    return e
end

# ZOH u
function stairpoints(t::Vector{Float64}, u::Vector{Float64}, tf::Float64)
    return vcat(t, tf), vcat(u, u[end])
end

nearest(t::Vector{Float64}, τ) = argmin(abs.(t .- τ))

# Theme

function figtheme()
    Theme(
        fontsize=11,
        figure_padding=8,
        Axis=(
            xgridcolor=(C_GRID, 0.45), ygridcolor=(C_GRID, 0.45),
            xgridwidth=0.6, ygridwidth=0.6,
            leftspinevisible=true, bottomspinevisible=true,
            topspinevisible=false, rightspinevisible=false,
            spinewidth=0.8, xtickwidth=0.8, ytickwidth=0.8,
            xticksize=3, yticksize=3, titlesize=12, titlealign=:left,
        ),
        Legend=(framevisible=false, patchsize=(16, 2), rowgap=1, padding=(4, 4, 2, 2)),
        Lines=(linewidth=2,),
    )
end

# Figures

# Tw traj + Tpeak traj
function fig_control(cases::Vector{Case})
    fig = Figure(size=(620, 460))
    ax1 = Axis(fig[1, 1]; ylabel=L"$T_w$ \,/\,K",
        title="Control")
    ax2 = Axis(fig[2, 1]; xlabel=L"\mathrm{t}\, /\,\mathrm{s}", ylabel=L"$T_\mathrm{peak}$\, /\,K",
        title="Peak temperature")
    linkxaxes!(ax1, ax2)
    hidexdecorations!(ax1; grid=false)

    seen = Vector{Float64}[]
    for c in cases
        any(==(c.u), seen) && continue
        push!(seen, c.u)
        ts, us = stairpoints(c.ut, c.u, c.s.t[end])
        stairs!(ax1, ts, us; step=:post, color=c.color, linewidth=2,
            label=strip(replace(c.name, r"\s*\(.*\)$" => "")))
    end
    for c in cases
        lines!(ax2, c.s.t, peak(c.s, "T"); color=c.color, linewidth=2) #, label=gridlabel(c))
    end

    ulo, uhi = extrema(vcat((c.u for c in cases)...))
    pad = 0.15 * max(uhi - ulo, 1.0)
    ylims!(ax1, ulo - pad, max(uhi, TW_MAX) + pad)
    hlines!(ax1, TW_MAX; color=C_LIMIT, linestyle=:dash, linewidth=1.2)
    text!(ax1, cases[1].s.t[end], TW_MAX;
        text=@sprintf("%g K", TW_MAX), align=(:right, :top),
        color=C_LIMIT, fontsize=9)

    hlines!(ax2, TMAX; color=C_LIMIT, linestyle=:dash, linewidth=1.2)
    text!(ax2, 0.0, TMAX; text=" $(round(Int, TMAX)) K", align=(:left, :bottom),
        color=C_LIMIT, fontsize=9)

    axislegend(ax1; position=:lb)
    #axislegend(ax2; position=:lt)
    rowgap!(fig.layout, 6)
    return fig
end

# Peak temperature vs Tmax
function fig_maxT(cases::Vector{Case}; title="Peak temperature")
    fig = Figure(size=(620, 300))
    ax = Axis(fig[1, 1]; xlabel="t / s",
        ylabel=L"T_{\mathrm{peak}}\, /\, \mathrm{K}", title=title)

    for c in cases
        lines!(ax, c.s.t, peak(c.s, "T"); color=c.color, linewidth=2, label=gridlabel(c))
    end

    hlines!(ax, TMAX; color=C_LIMIT, linestyle=:dash, linewidth=1.2)
    text!(ax, 0.0, TMAX; text=" $(round(Int, TMAX)) K",
        align=(:left, :bottom), color=C_LIMIT, fontsize=9)

    length(cases) > 1 && axislegend(ax; position=:rb)
    return fig
end

# T(z, t)
function fig_spacetime(cases::Vector{Case}; field="T", unit="K")
    fig = Figure(size=(310 * length(cases) + 130, 300))
    maps = [radialmean(c.s, field) for c in cases]
    lo, hi = extrema(vcat(vec.(maps)...))

    local hm
    for (i, (c, M)) in enumerate(zip(cases, maps))
        ax = Axis(fig[1, i]; xlabel="t / s",
            ylabel=i == 1 ? "z / m" : "",
            title=c.name)
        i > 1 && hideydecorations!(ax; grid=false)
        hm = heatmap!(ax, edges(c.s.t), c.s.z, permutedims(M);
            colorrange=(lo, hi), colormap=CMAP_T)
        xlims!(ax, c.s.t[1], c.s.t[end])
        ylims!(ax, c.s.z[1], c.s.z[end])
    end
    Colorbar(fig[1, length(cases)+1], hm;
        label="$field / $unit (radially averaged)", width=10, ticksize=3)
    colgap!(fig.layout, 24)
    return fig
end

# T(z,t) snapshots
function fig_snapshots(cases::Vector{Case};
    times=(0.0, 187.5, 375.0, 562.5, 750.0), field="T", unit="K", panelheight=54)
    nrow = length(times)
    fig = Figure(size=(300 * length(cases) + 190, (panelheight + 16) * nrow + 74))

    idx = [nearest(c.s.t, τ) for c in cases, τ in times]
    lo, hi = extrema(vcat([vec(c.s[field][:, :, idx[i, :]])
                           for (i, c) in enumerate(cases)]...))

    local hm
    mid = cld(nrow, 2)
    for row = 1:nrow, (col, c) in enumerate(cases)
        ax = Axis(fig[row, col+1];
            xlabel=row == nrow ? "z / m" : "",
            ylabel=(col == 1 && row == mid) ? "r / mm" : "",
            yticks=[0, 5, 10], title=row == 1 ? c.name : "")
        col > 1 && hideydecorations!(ax)
        row < nrow && hidexdecorations!(ax)
        hm = heatmap!(ax, c.s.z, c.s.r .* 1e3, c.s[field][:, :, idx[col, row]];
            colorrange=(lo, hi), colormap=CMAP_T)
    end
    # tellheight=false so rows do not collapse to the label height
    for (row, τ) in enumerate(times)
        Label(fig[row, 1], @sprintf("t = %g s", τ);
            halign=:right, fontsize=10, tellheight=false)
        rowsize!(fig.layout, row, Fixed(panelheight))
    end
    Colorbar(fig[:, length(cases)+2], hm; label="$field / $unit",
        width=10, ticksize=3)
    colgap!(fig.layout, 20)
    rowgap!(fig.layout, 16)
    return fig
end

# Outlet conv
function fig_conversion(cases::Vector{Case}; title="Outlet conversion")
    fig = Figure(size=(620, 280))
    ax = Axis(fig[1, 1]; xlabel="t / s", ylabel=L"$X_{\mathrm{CO}_2}$ (outlet)",
        title=title)
    for c in cases
        X = outlet_conv(c.s)
        lines!(ax, c.s.t, X; color=c.color, linewidth=2, label=gridlabel(c))
        text!(ax, c.s.t[end], X[end]; text=@sprintf("  %.3f", X[end]),
            align=(:left, :center), color=c.color, fontsize=9)
    end
    xlims!(ax, cases[1].s.t[1], cases[1].s.t[end] * 1.08)
    axislegend(ax; position=:rb)
    return fig
end

# Radial profile at hotspot
function fig_radial(cases::Vector{Case})
    fig = Figure(size=(620, 340), figure_padding=(8, 18, 8, 8))
    R = cases[1].s.r[end] * 1e3
    ax = Axis(fig[1, 1]; xlabel="r / mm",
        ylabel="T / K", xticks=0:2.5:R,
        title="Radial profile at hot spot")
    for c in cases
        T = c.s["T"]
        _, I = findmax(T)
        iz, _, it = Tuple(I)
        rc = cellcentres(c.s.r) .* 1e3
        lines!(ax, rc, T[iz, :, it]; color=c.color, linewidth=2,
            label=@sprintf("%s  (%d×%d, z = %.2f m, t = %g s)",
                c.name, griddims(c.s)..., cellcentres(c.s.z)[iz], c.s.t[it]))
        scatter!(ax, rc, T[iz, :, it]; color=c.color, markersize=8)
    end
    xlims!(ax, 0, R)
    # below axis
    Legend(fig[2, 1], ax; tellheight=true, tellwidth=false)
    rowgap!(fig.layout, 4)
    return fig
end

# Main

function makefigures(; formats=("pdf", "png"))
    mkpath(FIGDIR)
    
    fine = load("opt_fine", "optimised (fine)", C_FINE)
    coarse = load("opt_coarse", "optimised (coarse)", C_COARSE)
    guess_coarse = load("guess_coarse", "initial guess", C_GUESS)
    guess_fine = load("guess_fine", "constant wall temperature", C_GUESS)

    figs = with_theme(figtheme()) do
        ["control" => fig_control([coarse, guess_coarse]),
            "spacetime" => fig_spacetime([coarse, fine]),
            "snapshots" => fig_snapshots([fine]),
            "conversion" => fig_conversion([coarse, guess_coarse];
                title="Outlet conversion (NLP grid)"),
            "conversion_validation" => fig_conversion([fine, coarse];
                title="Outlet conversion (coarse vs. fine grid)"),
            "radial" => fig_radial([coarse, fine]),
            "maxT" => fig_maxT([coarse]; title="Peak temperature (NLP grid)"),
            "maxT_validation" => fig_maxT([fine, coarse];
                title="Peak temperature (coarse vs. fine grid)"),
            "guess_snapshots" => fig_snapshots([guess_fine])]

    end
    for (name, fig) in figs, ext in formats
        path = joinpath(FIGDIR, "methanation_$name.$ext")
        save(path, fig)
        println("wrote ", relpath(path, dirname(APPDIR)))
    end
    return figs
end

if abspath(PROGRAM_FILE) == @__FILE__
    makefigures()
end
