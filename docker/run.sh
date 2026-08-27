#!/usr/bin/env bash
# Run a driver script inside the pdeopt image, with results landing in the repo.
#
#   docker/run.sh                                       # methanation OCP, foreground
#   docker/run.sh apps/forward_solvers/Methanation.jl   # any other driver
#   DETACH=1 NAME=meth-01 docker/run.sh                 # long run, survives an SSH drop
#
# Env: IMAGE (default pdeopt:0.1), OMP_NUM_THREADS, JULIA_NUM_THREADS, DETACH, NAME
set -euo pipefail

IMAGE=${IMAGE:-pdeopt:0.1}
SCRIPT=${1:-apps/optimal_control/MethanationOpt.jl}

# MA97 is OpenMP-threaded. nproc counts hyperthreads, so on a shared machine set
# OMP_NUM_THREADS to the physical core count you are allowed to use.
OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(nproc)}
# 2 keeps the stdout reader in src/diagnostics/opt_log.jl from stalling on profiled runs.
JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-2}

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

# Bind mounts fail on a missing host path, and .dockerignore keeps these out of the image.
mkdir -p "$repo/apps/optimal_control/results" "$repo/apps/forward_solvers/results"

args=(
  # without this, everything written to the mounts is owned by root on the host
  --user "$(id -u):$(id -g)"
  -v "$repo/apps/optimal_control/results:/app/apps/optimal_control/results"
  -v "$repo/apps/forward_solvers/results:/app/apps/forward_solvers/results"
  -e "OMP_NUM_THREADS=$OMP_NUM_THREADS"
  -e "JULIA_NUM_THREADS=$JULIA_NUM_THREADS"
)

if [[ -n ${DETACH:-} ]]; then
  name=${NAME:-pdeopt-$(date +%Y%m%d-%H%M%S)}
  args+=(-d --name "$name")   # no --rm: it would discard the logs on exit
else
  args+=(--rm)
fi

docker run "${args[@]}" "$IMAGE" "$SCRIPT" "${@:2}"

if [[ -n ${DETACH:-} ]]; then
  echo
  echo "started ${name}"
  echo "  docker logs -f ${name}    # follow (Ctrl-C detaches, does not stop the run)"
  echo "  docker stats ${name}      # live memory, watch this if IPOPT gets large"
  echo "  docker wait ${name}       # block, print exit code (137 = OOM-killed)"
  echo "  docker rm ${name}         # clean up once finished"
fi
