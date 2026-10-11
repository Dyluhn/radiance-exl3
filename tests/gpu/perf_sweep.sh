#!/bin/sh
# perf_sweep.sh -- reading 1's sweep inside a run_gpu.sh session: perfmoe for each (op, tp, M), exl3r4d against
# libr4d on the same routing. Run as the session's host command:
#   tests/gpu/run_gpu.sh <pci> <lock> <logdir> -- tests/gpu/perf_sweep.sh <ceiling GB/s> [exl3r4d.so] [ops] [Ms]
# Paths are inside the container (/work = this repository). PERF_ARGS adds perfmoe arguments to every run.
set -eu
ceil=$1
so=${2:-build-gpu/radiance_home/kernels/exl3r4d.so}
ops=${3:-"gate_up:1 gate_up:2 down:1 down:2"}
Ms=${4:-"1 2 4 6 8 16"}
r4d=${R4D_SO:-/stage/opt/radiance/share/radiance/kernels/libr4d.so}
for opt in $ops; do
    op=${opt%:*} tp=${opt#*:}
    for M in $Ms; do
        docker exec -w /work "$EXL3_CONTAINER" build-gpu/kernels/exl3r4d/tests-gpu/perfmoe --exl3 "$so" \
            --r4d "$r4d" --op "$op" --tp "$tp" --M "$M" --ceiling "$ceil" ${PERF_ARGS:-}
    done
done
