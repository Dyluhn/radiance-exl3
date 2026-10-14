#!/bin/sh
# residency_check.sh -- the fused gate_up_down row (interface v5) with its stream limited to part of the card
# (perfmoe --cu-pct): its persistent grid is sized to the whole card, so not every workgroup can be resident, as in the
# engine when other kernels hold slots. A launch that needs co-residency stalls to fu_wait's give-up and its output
# fails the byte check against the serial chain. Inside a run_gpu.sh session:
#   tests/gpu/run_gpu.sh <pci> <lock> <logdir> -- tests/gpu/residency_check.sh "<variants>" "<M...>" "<cu pcts>"
set -u
vars=$1 ms=$2 pcts=$3
for v in $vars; do
    for pct in $pcts; do
        for M in $ms; do
            t0=$(date +%s.%N)
            out=$(timeout 300 docker exec -w /work "$EXL3_CONTAINER" build-gpu/kernels/exl3r4d/tests-gpu/perfmoe \
                --exl3 "build-var/$v.so" --tp 2 --M "$M" --chain --h-rot 1 --gd --cu-pct "$pct" \
                --layers 2 --iters 1 --reps 1 --warm-ms 0 2>&1)
            rc=$?
            t1=$(date +%s.%N)
            printf 'RESIDENCY %s cu=%s%% M=%s rc=%s wall=%.1fs | %s\n' "$v" "$pct" "$M" "$rc" "$(echo "$t1 - $t0" | bc)" \
                "$(echo "$out" | grep -E 'GD-CHECK|WARM|limited|failed|FAILED' | tr '\n' ' ' | cut -c1-220)"
        done
    done
done
