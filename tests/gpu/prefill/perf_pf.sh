#!/bin/sh
# perf_pf.sh -- the M > 16 rows timed against libr4d's moe_gemm_w4a8 on the same routing and bytes (tools/perfmoe.cpp,
# P4's harness: 48 routing draws over 2 real-size pools, 500 ms warm-up, ABAB). Run as tests/gpu/run_gpu.sh's host
# command (paths inside the container: /work = this repository):
#   tests/gpu/run_gpu.sh <pci> <lock> <logdir> -- tests/gpu/prefill/perf_pf.sh <ceiling GB/s> "<so ...>" "<op:tp ...>" "<M ...>"
# Each (op, tp, M) runs every library given in turn (A B A B ... across the list), so a variant arm and the shipped one
# share the session's clocks. PERF_ARGS adds perfmoe arguments to every run.
set -eu
ceil=$1 sos=$2 opts=$3 Ms=$4
r4d=${R4D_SO:-/stage/opt/radiance/share/radiance/kernels/libr4d.so}
for opt in $opts; do
    op=${opt%:*} tp=${opt#*:}
    for M in $Ms; do
        for so in $sos; do
            echo "=== $so"
            docker exec -w /work "$EXL3_CONTAINER" build-gpu/kernels/exl3r4d/tests-gpu/perfmoe --exl3 "$so" \
                --r4d "$r4d" --op "$op" --tp "$tp" --M "$M" --ceiling "$ceil" --exl3-row "exl3r4d_moe_${op}_pf" ${PERF_ARGS:-}
        done
    done
done
