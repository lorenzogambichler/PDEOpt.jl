#!/usr/bin/env bash
# Run a driver script inside the pdeopt image (results in the repo)
#   docker/run.sh (methanation OCP, foreground)
#   docker/run.sh apps/methanation/... (other driver)
#   DETACH=1 NAME=meth-01 docker/run.sh (long run, survives SSH drop)
# Env: IMAGE (default pdeopt:0.1), OMP_NUM_THREADS, JULIA_NUM_THREADS, DETACH, NAME
set -euo pipefail

IMAGE=${IMAGE:-pdeopt:0.1}
SCRIPT=${1:-apps/methanation/opt.jl}

OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(nproc)}
# 2 for src/diagnostics/opt_log.jl
JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-2}

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

mkdir -p "$repo/apps/methanation/results" "$repo/apps/plug_flow/results"

args=(
  --user "$(id -u):$(id -g)"
  -v "$repo/apps/methanation/results:/app/apps/methanation/results"
  -v "$repo/apps/plug_flow/results:/app/apps/plug_flow/results"
  -e "OMP_NUM_THREADS=$OMP_NUM_THREADS"
  -e "JULIA_NUM_THREADS=$JULIA_NUM_THREADS"
)

if [[ -n ${DETACH:-} ]]; then
  name=${NAME:-pdeopt-$(date +%Y%m%d-%H%M%S)}
  args+=(-d --name "$name") # --rm discards logs on exit
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
