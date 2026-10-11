#!/bin/sh
# run_cases.sh -- the GPU phase of tests/gpu/prefill/test_prefill.py: P2's harness on every planned case, inside the
# session container of tests/gpu/run_gpu.sh (which holds the GPU lock and does the hygiene). Run as its host command:
#
#   tests/gpu/run_gpu.sh <pci> <lock> <logdir> -- tests/gpu/prefill/run_cases.sh <work> <harness> <rowpick.so> <lib.so>
#
# <work> holds runs.txt (one case directory a line, host paths under this repository); the other three are paths
# inside the repository too (the container sees it at /work). Each case gets harness.out / harness.err / harness.rc
# beside its case.txt. EXL3_ROWPICK_ROW (default _pf) picks the rows under test. Without $EXL3_CONTAINER the harness
# runs right here (inside a container already, the card pinned and the lock held by the caller: the ctest path).
set -eu
work=$1 harness=$2 shim=$3 lib=$4
repo=$(cd "$(dirname "$0")/../../.." && pwd)
c() { case "$1" in "$repo"/*) echo "/work${1#$repo}" ;; /*) echo "$1" ;; *) echo "/work/$1" ;; esac; }
row=${EXL3_ROWPICK_ROW:-_pf}
n=0 bad=0
while read -r d; do
    [ -n "$d" ] || continue
    rc=0
    if [ -n "${EXL3_CONTAINER:-}" ]; then
        docker exec -w /work -e EXL3_ROWPICK_LIB="$(c "$lib")" -e EXL3_ROWPICK_ROW="$row" "$EXL3_CONTAINER" \
            "$(c "$harness")" "$(c "$shim")" "$(c "$d")/case.txt" --domain device > "$d/harness.out" 2> "$d/harness.err" || rc=$?
    else
        EXL3_ROWPICK_LIB=$lib EXL3_ROWPICK_ROW=$row "$harness" "$shim" "$d/case.txt" --domain device \
            > "$d/harness.out" 2> "$d/harness.err" || rc=$?
    fi
    echo "$rc" > "$d/harness.rc"
    n=$((n + 1))
    [ "$rc" -eq 0 ] && grep -q '^rc 0$' "$d/harness.out" || { bad=$((bad + 1)); echo "not rc 0: $d (exit $rc) $(grep '^rc' "$d/harness.out")"; }
done < "$work/runs.txt"
echo "ran $n cases, $bad not launched with rc 0 (the check phase judges; some refusals are expected nowhere here)"
