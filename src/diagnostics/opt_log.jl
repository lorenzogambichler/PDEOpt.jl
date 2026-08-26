# tee stdout at fd level, so ipopt's C-side output is caught too -> (result, log)
function capture_stdout(f; echo::Bool=true)
    Threads.nthreads() > 1 ||
        @warn "single-threaded: reader will stall inside blocking ccalls, start julia with -t 2"

    real = stdout
    pipe = Pipe()
    Base.link_pipe!(pipe; reader_supports_async=true, writer_supports_async=true)
    buf = IOBuffer()

    reader = Threads.@spawn while !eof(pipe)
        data = readavailable(pipe)
        write(buf, data)
        if echo
            write(real, data)
            flush(real)
        end
    end

    out = try
        redirect_stdout(pipe.in) do
            try
                f()
            finally
                # flush while fd 1 is still the pipe, libc buffers when stdout isn't a tty
                Base.Libc.flush_cstdio()
            end
        end
    finally
        close(pipe.in)
    end
    wait(reader)
    close(pipe)
    return out, String(take!(buf))
end

struct IpoptTiming
    cpu::Dict{String,Float64}
    wall::Dict{String,Float64}
    depth::Dict{String,Int} # indent of the row, i.e. level in the timing tree
    order::Vector{String}
    counts::Dict{String,Int} # eval counts from the summary block
    iters::Int
end

# "  LinearSystemFactorization....:  1093.560 (sys:  19.282 wall:  332.830)"
# the (sys: … wall: …) group is required, it keeps the convergence block out
const _ROW = r"^(\s*)(\S.*?)\.*:\s*([-\d.eE+]+)\s*\(sys:\s*[-\d.eE+]+\s+wall:\s*([-\d.eE+]+)\)"
const _CNT = r"^Number of (.+?) evaluations\s*=\s*(\d+)"
const _IT = r"^Number of Iterations\.*:\s*(\d+)"

# needs print_timing_statistics="yes"
function parse_ipopt_timing(log::AbstractString)
    cpu, wall, depth = Dict{String,Float64}(), Dict{String,Float64}(), Dict{String,Int}()
    order, counts, iters = String[], Dict{String,Int}(), 0

    for line in eachline(IOBuffer(log))
        m = match(_ROW, line)
        if !isnothing(m)
            key = strip(m[2])
            haskey(cpu, key) || push!(order, key)
            cpu[key] = parse(Float64, m[3])
            wall[key] = parse(Float64, m[4])
            depth[key] = length(m[1])
            continue
        end
        m = match(_CNT, line)
        isnothing(m) || (counts[lowercase(strip(m[1]))] = parse(Int, m[2]))
        m = match(_IT, line)
        isnothing(m) || (iters = parse(Int, m[1]))
    end
    isempty(order) && error("no timing table found -- run ipopt with print_timing_statistics=\"yes\"")
    return IpoptTiming(cpu, wall, depth, order, counts, iters)
end

read_ipopt_timing(path::AbstractString) = parse_ipopt_timing(read(path, String))

total(t::IpoptTiming) = get(t.wall, "OverallAlgorithm", maximum(values(t.wall)))

# disjoint leaves of the timing tree, the rest is ipopt's own vector algebra
# (label, timing row, eval-count row)
const _LEAVES = (
    ("ma97 factorization", "LinearSystemFactorization", ""),
    ("AD jacobian", "Equality constraint Jacobian", "equality constraint jacobian"),
    ("ma97 backsolve", "LinearSystemBackSolve", ""),
    ("residual", "Equality constraints", "equality constraint"),
    ("ma97 symbolic", "LinearSystemSymbolicFactorization", ""),
    ("ma97 scaling", "LinearSystemScaling", ""),
    ("hessian update", "UpdateHessian", ""),
    ("objective", "Objective function", "objective function"),
    ("gradient", "Objective function gradient", "objective gradient"),
    ("exact hessian", "Lagrangian Hessian", "lagrangian hessian"),
)

# full=true also dumps the raw timing tree, cpu/wall ratio is ma97's thread speedup
function print_timing(io::IO, t::IpoptTiming; full::Bool=false)
    tot = total(t)
    @printf(io, "Ipopt: %.1f s wall, %d iterations\n", tot, t.iters)

    acc = 0.0
    for (label, key, ckey) in _LEAVES
        w = get(t.wall, key, 0.0)
        w == 0.0 && continue
        acc += w
        n = get(t.counts, ckey, 0)
        @printf(io, "  %-20s %8.1f s %6.1f %%", label, w, 100w / tot)
        n > 0 ? @printf(io, "   %4d × %8.4f s\n", n, w / n) : println(io)
    end
    @printf(io, "  %-20s %8.1f s %6.1f %%\n", "other", tot - acc, 100 * (tot - acc) / tot)

    full || return nothing
    println(io)
    for key in t.order
        w = t.wall[key]
        @printf(io, "%s%-*s %9.3f s %6.1f %%  (cpu %9.3f, %.2f×)\n",
            " "^t.depth[key], 38 - t.depth[key], key, w, 100w / tot,
            t.cpu[key], w > 0 ? t.cpu[key] / w : 0.0)
    end
    return nothing
end

print_timing(t::IpoptTiming; full::Bool=false) = print_timing(stdout, t; full=full)
Base.show(io::IO, ::MIME"text/plain", t::IpoptTiming) = print_timing(io, t)
