#!/bin/sh
# run_all.sh -- every REQUIREMENTS gate (tools/gates/R*.sh), in order, one summary line each, then the tally.
#
#   tools/gates/run_all.sh [R1 R4 ...]                    host gates (each SKIPS by name what it cannot reach)
#   tests/gpu/run_gpu.sh <pci> <lock> <log dir> -- tools/gates/run_all.sh R5 R7 R12 R13 R8 R11 R3
#                                                         the same, with the device parts in one GPU session
# Inputs are each gate's (tools/gates/lib.sh lists the common ones; each R*.sh's header its own). Full output of
# each gate: $GATES_WORK/logs/<R>.out; the summary: $GATES_WORK/logs/run_all.txt.
# EXIT: 0 every gate PASS; 1 any FAIL; 77 none failed but some SKIP or PARTIAL (not met).
here=$(cd "$(dirname "$0")" && pwd)
: "${GATES_WORK:=$(cd "$here/../.." && pwd)/build-gates}"
mkdir -p "$GATES_WORK/logs"
gates=${*:-$(ls "$here"/R*.sh | sed 's|.*/||; s|\.sh$||' | sort -V)}
summary=$GATES_WORK/logs/run_all.txt
: > "$summary"
fail=0 open=0
for g in $gates; do
    sh "$here/$g.sh" > "$GATES_WORK/logs/$g.out" 2>&1
    rc=$?
    line=$(grep -E "^$g (PASS|FAIL|SKIP|PARTIAL): " "$GATES_WORK/logs/$g.out" | tail -1)
    [ -n "$line" ] || line="$g FAIL: no summary line (exit $rc; see $GATES_WORK/logs/$g.out)"
    echo "$line" | tee -a "$summary"
    case $rc in 0) ;; 77) open=1 ;; *) fail=1 ;; esac
done
echo "run_all: $(grep -c ' PASS: ' "$summary") PASS, $(grep -c ' PARTIAL: ' "$summary") PARTIAL, $(grep -c ' SKIP: ' "$summary") SKIP, $(grep -c ' FAIL: ' "$summary") FAIL" | tee -a "$summary"
[ $fail = 1 ] && exit 1
[ $open = 1 ] && exit 77
exit 0
