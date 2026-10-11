#!/bin/sh
# pmc.sh -- counter reading of the decode dispatches inside a run_gpu.sh session: rocprofv3 --pmc on a short
# perfmoe run (exl3r4d and libr4d on the same routing), one counter group a pass, per (op, tp, M).
#   tests/gpu/run_gpu.sh <pci> <lock> <logdir> -- tests/gpu/pmc.sh <out dir inside /work> "<op:tp:M ...>" [exl3r4d.so]
# Needs libdw.so.1 for rocprofv3 (absent from the build image): PMC_LIBDW names a directory holding it.
set -u
out=$1 points=$2 so=${3:-build-gpu/radiance_home/kernels/exl3r4d.so}
r4d=${R4D_SO:-/stage/opt/radiance/share/radiance/kernels/libr4d.so}
G1="SQ_WAVE_CYCLES SQ_BUSY_CYCLES SQ_WAIT_ANY SQ_WAIT_INST_ANY SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_TEX_LOAD SQ_INSTS_LDS"
G2="WAVE_DEP_WAIT WAVE_ISSUE_WAIT ValuPipeIssueUtil VALUBusy MemUnitBusy MeanOccupancyPerActiveCU"
G3="TCP_REQ TCP_REQ_MISS GL2C_HIT_sum GL2C_MISS_sum FETCH_SIZE SQ_INST_CYCLES_VMEM SQ_INST_CYCLES_VALU TA_BUSY_avr"
for pt in $points; do
    op=$(echo "$pt" | cut -d: -f1) tp=$(echo "$pt" | cut -d: -f2) M=$(echo "$pt" | cut -d: -f3)
    g=0
    for grp in "$G1" "$G2" "$G3"; do
        g=$((g + 1))
        d=$out/$op-tp$tp-M$M-g$g
        docker exec -w /work -e LD_LIBRARY_PATH="${PMC_LIBDW:-/work/build-gpu/dl-libdw/root/usr/lib/x86_64-linux-gnu}" \
            "$EXL3_CONTAINER" timeout "${PMC_TIMEOUT:-240}" rocprofv3 --pmc $grp \
            --kernel-include-regex "${PMC_KERNELS:-exl3r4d_gemm_kernel|moe_gemm_q_kernel}" --output-format csv -d "$d" -o run -- \
            build-gpu/kernels/exl3r4d/tests-gpu/perfmoe --exl3 "$so" --r4d "$r4d" --op "$op" --tp "$tp" --M "$M" \
            --layers 8 --iters 1 --reps 1 --warm-ms 0 > /dev/null 2>&1 || echo "FAILED $d"
        echo "done $d"
    done
done
