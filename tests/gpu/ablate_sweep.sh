#!/bin/sh
# ablate_sweep.sh -- a reading or a lever trial inside a run_gpu.sh session: every build-var/<name>.so given, on the
# same (op, tp, M) points, PASSES times with the variants interleaved, exl3r4d alone (perfmoe without --r4d). A
# variant whose launch is refused (a shape that does not fit) prints its perfmoe error and the sweep goes on.
#   tests/gpu/run_gpu.sh <pci> <lock> <logdir> -- tests/gpu/ablate_sweep.sh "<variants>" "<op:tp:M ...>" [passes]
set -u
vars=$1 points=$2 passes=${3:-2}
for pass in $(seq 1 "$passes"); do
    for pt in $points; do
        op=$(echo "$pt" | cut -d: -f1) tp=$(echo "$pt" | cut -d: -f2) M=$(echo "$pt" | cut -d: -f3)
        for v in $vars; do
            printf 'VARIANT %s pass %s\n' "$v" "$pass"
            docker exec -w /work "$EXL3_CONTAINER" build-gpu/kernels/exl3r4d/tests-gpu/perfmoe \
                --exl3 "build-var/$v.so" --op "$op" --tp "$tp" --M "$M" ${PERF_ARGS:-} || echo "REFUSED $v $pt"
        done
    done
done
