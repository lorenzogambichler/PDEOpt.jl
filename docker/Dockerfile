# syntax=docker/dockerfile:1.7
# docker build -f docker/Dockerfile -t pdeopt:0.1 \
#   --build-context hsl=/path/to/HSL/HSL_jll.jl.v2025.7.21/HSL_jll.jl.v2025.7.21 \


# julia_version in Manifest.toml
FROM julia:1.12.6-bookworm

ARG JULIA_CPU_TARGET=generic
ENV JULIA_CPU_TARGET=${JULIA_CPU_TARGET}

# Build-time depot (packages, precompile caches)
ENV JULIA_DEPOT_PATH=/opt/julia \
    JULIA_PROJECT=/app

WORKDIR /app

# HSL
COPY --from=hsl . /opt/HSL_jll

# Julia env
COPY Project.toml Manifest.toml ./

RUN mkdir -p src && printf 'module PDEOpt end\n' > src/PDEOpt.jl

RUN julia -e 'using Pkg; Pkg.develop(path="/opt/HSL_jll"); Pkg.instantiate(); Pkg.precompile()'

# Source
COPY src ./src
COPY apps ./apps

RUN julia -e 'using Pkg; Pkg.precompile()'

RUN julia -e 'using HSL_jll; @assert isfile(HSL_jll.libhsl_path) HSL_jll.libhsl_path; println("HSL ok: ", HSL_jll.libhsl_path)'

ENV JULIA_DEPOT_PATH=/tmp/julia-depot:/opt/julia

ENTRYPOINT ["julia", "--project=/app"]
CMD ["apps/methanation/opt.jl"]
