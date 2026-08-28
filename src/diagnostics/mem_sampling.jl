const GiB = 1024^3

# resident set of process (@allocated/Profile.Allocs only gc heap)
function _rss()
    @static if Sys.islinux()
        return parse(Int, split(read("/proc/self/statm", String))[2]) * Sys.PAGESIZE
    else
        return Int(Sys.maxrss()) # peak
    end
end

struct MemTrace
    t::Vector{Float64}
    rss::Vector{Int} # resident set
    jl::Vector{Int} # gc live bytes
    peak::Int
end

# ma97
foreign(tr::MemTrace) = tr.rss .- tr.jl

# run f while sampling memory -> (result, MemTrace)
function memtrace(f; dt::Float64=0.1)
    Threads.nthreads() > 1 ||
        @warn "single-threaded: sampler will stall inside blocking ccalls, start julia with -t 2"

    t, rss, jl = Float64[], Int[], Int[]
    t0 = time()
    sample!() = (push!(t, time() - t0); push!(rss, _rss()); push!(jl, Base.gc_live_bytes()))

    stop = Threads.Atomic{Bool}(false)
    sample!()
    sampler = Threads.@spawn while !stop[]
        sample!()
        Libc.systemsleep(dt)
    end

    out = try
        f()
    finally
        stop[] = true
        wait(sampler)
        sample!()
    end
    return out, MemTrace(t, rss, jl, Int(Sys.maxrss()))
end

const _BLK = ('▁', '▂', '▃', '▄', '▅', '▆', '▇', '█')

# bucket-max sparkline
function _spark(v::AbstractVector, lo::Real, hi::Real, w::Int=48)
    isempty(v) && return ""
    n = min(w, length(v))
    hi <= lo && return String(fill(_BLK[1], n))
    s = Vector{Char}(undef, n)
    for k in 1:n
        i1 = 1 + fld((k - 1) * length(v), n)
        i2 = max(i1, fld(k * length(v), n))
        m = maximum(view(v, i1:i2))
        s[k] = _BLK[clamp(1 + floor(Int, 7 * (m - lo) / (hi - lo)), 1, 8)]
    end
    return String(s)
end

function Base.show(io::IO, ::MIME"text/plain", tr::MemTrace)
    fo = foreign(tr)
    @printf(io, "MemTrace: %.1f s, %d samples, peak rss %.2f GiB\n",
        tr.t[end], length(tr.t), tr.peak / GiB)
    hi = maximum(tr.rss) # common scale -> foreign + julia = rss
    for (name, v) in (("rss", tr.rss), ("foreign", fo), ("julia", tr.jl))
        @printf(io, "  %-7s %s %6.2f -> %6.2f GiB (max %.2f)\n",
            name, _spark(v, 0, hi), v[1] / GiB, v[end] / GiB, maximum(v) / GiB)
    end
end
