#!/bin/sh
# ksweep.sh -- reading 5 inside a run_gpu.sh session: fix N, sweep K (gate_up: hidden; down: width), M fixed,
# exl3r4d and libr4d on the same routing. A line fit of time against weight bytes gives the streaming rate (slope)
# and the fixed cost a dispatch (intercept).
#   tests/gpu/run_gpu.sh <pci> <lock> <logdir> -- tests/gpu/ksweep.sh <op> <M> "<K values>" [exl3r4d.so]
set -eu
op=$1 M=$2 Ks=$3 so=${4:-build-gpu/radiance_home/kernels/exl3r4d.so}
r4d=${R4D_SO:-/stage/opt/radiance/share/radiance/kernels/libr4d.so}
for K in $Ks; do
    if [ "$op" = gate_up ]; then dims="--hidden $K"; else dims="--width $K"; fi
    printf 'KSWEEP %s K=%s\n' "$op" "$K"
    docker exec -w /work "$EXL3_CONTAINER" build-gpu/kernels/exl3r4d/tests-gpu/perfmoe --exl3 "$so" --r4d "$r4d" \
        --op "$op" --tp 1 --M "$M" $dims ${PERF_ARGS:-}
done
