module Diagnostics

# Runtime/memory profiling helpers
# memtrace samples from a second task -> start julia with -t 2

using Printf

include("mem_sampling.jl")
include("opt_log.jl")

export
    # memory sampling
    MemTrace, memtrace,
    # ipopt timing table
    IpoptTiming, parse_ipopt_timing, read_ipopt_timing, print_timing
end
