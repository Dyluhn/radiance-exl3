#!/bin/sh
# R7 -- checkable: the engine-style harness (tests/ref, P2's; tests/gpu, P4's on top of it) reports 0 failed and
# 0 skipped for exl3ref (host rows: suites `cases` and `band-tops`) and for exl3r4d (device rows: suite `device`).
# A suite that checked nothing is SKIPPED, never passed. rad-kbench is NOT evidence here: at radiance 1.3.0 it
# zero-fills every weight-table operand (tools/kfixture.cpp:312; PLAN D15), so its rows for moe_exl3_* are NOT
# CHECKABLE and this gate never counts them.
#
#   RADIANCE_PREFIX=<host install> tools/gates/R7.sh                             host rows
#   tests/gpu/run_gpu.sh <pci> <lock> <log dir> -- tools/gates/R7.sh            and the device rows
GATE=R7
. "$(dirname "$0")/lib.sh"
why=$(need_prefix) || { part SKIP "pinned install" "$why"; finish "nothing checked"; }
host_build || { part FAIL "host build" "see $GATES_WORK/logs/host-build.log"; finish "no build"; }
summary=""
for suite in cases band-tops; do
    out=$GATES_WORK/logs/r7-exl3ref-$suite.out
    ref_host "$suite" "$out"
    if clean_tally "$out"; then part PASS "exl3ref $suite" "$(tally "$out")"
    else part FAIL "exl3ref $suite" "$(tally "$out" || echo 'no tally'): $(grep -E '^(FAIL|SKIP) ' "$out" | head -3 | tr '\n' ' ')"; fi
    summary="$summary exl3ref $(tally "$out");"
done
if why=$(device_session); then
    out=$GATES_WORK/logs/r7-exl3r4d-device.out
    device_suite device "$out"
    if clean_tally "$out"; then part PASS "exl3r4d device" "$(tally "$out")"
    else part FAIL "exl3r4d device" "$(tally "$out" || echo 'no tally'): $(grep -E '^(FAIL|SKIP) ' "$out" | head -3 | tr '\n' ' ')"; fi
    summary="$summary exl3r4d $(tally "$out");"
else
    part SKIP "exl3r4d device" "$why"
fi
echo "  note kbench: moe_exl3_gate_up / moe_exl3_down rows NOT CHECKABLE by rad-kbench at $PIN_VERSION (WTAB operands zero-filled); not counted"
finish "${summary# }"
