#!/bin/sh
# R12 -- nothing unimplemented returns success: each thing the plugins do not do is refused BY NAME, with a test.
# The required list and the test that checks each:
#   mcg, 3inst on the device         exl3r4d's layout hook -> RAD_E_DTYPE      ctest exl3r4d_layout (host-run C test
#   K != 4 on the device                                                        of the hook) + P4's device suite
#   half-integer K (exllamav3 1.5.2) the quantiser -> RAD_E_UNSUPPORTED          ctest exl3_quant_test
#   a bias                           the quantiser -> RAD_E_UNSUPPORTED          ctest exl3_quant_test
# Listed beside them, from the same runs: exl3ref's launch refusals (P2's harness), the architecture's refusals
# (arch_static), the quantiser's other refusals, and on the card the device rows' launch refusals.
#
#   RADIANCE_PREFIX=<host install> tools/gates/R12.sh                      host-run tests
#   tests/gpu/run_gpu.sh <pci> <lock> <log dir> -- tools/gates/R12.sh     and the device suite's refusals
GATE=R12
. "$(dirname "$0")/lib.sh"
why=$(need_prefix) || { part SKIP "pinned install" "$why"; finish "nothing checked"; }
host_build || { part FAIL "host build" "see $GATES_WORK/logs/host-build.log"; finish "no build"; }
L=$GATES_WORK/logs
ctest_v() { ctest --test-dir "$GATES_BUILD" -R "^$1\$" -V > "$L/r12-$1.out" 2>&1; }
list() { sed 's/^[0-9]*: //' | sed 's/^/        /'; }

# the quantiser: half-integer K and a bias, by their tests' names, and its other refusals
if ctest_v exl3_quant_test; then
    for t in refuses_half_integer_k refuses_a_bias; do
        if grep -qE "ok +$t\$" "$L/r12-exl3_quant_test.out"; then
            part PASS "quantiser: $t" "$(grep -E "half-integer K|has a .bias" "$L/r12-exl3_quant_test.out" | grep -m1 "$( [ $t = refuses_a_bias ] && echo bias || echo half-integer)" | sed 's/^[0-9]*: radiance: exl3: //' | cut -c1-140)"
        else part FAIL "quantiser: $t" "not reported ok (see $L/r12-exl3_quant_test.out)"; fi
    done
    echo "      the quantiser's refusals (exl3_quant_test):"; grep -E 'ok +refuses_' "$L/r12-exl3_quant_test.out" | list
else part FAIL "quantiser" "exl3_quant_test failed or skipped (see $L/r12-exl3_quant_test.out)"; fi

# exl3r4d's layout hook, run on the host: K 2,3,5,6,7,8, mcg, 3inst -> RAD_E_DTYPE
if [ -f "$REPO/tests/gpu/layout_test.c" ]; then
    if ctest_v exl3r4d_layout && grep -qE 'layout_test: [0-9]+ passed, 0 failed' "$L/r12-exl3r4d_layout.out"; then
        named=$(grep -cE '"K=%d refused RAD_E_DTYPE|"codebook %s refused RAD_E_DTYPE|stacked \(gate_up\) encoding on down: RAD_E_DTYPE' "$REPO/tests/gpu/layout_test.c")
        part PASS "exl3r4d layout hook: K!=4, mcg, 3inst" "$(grep -oE 'layout_test: [0-9]+ passed, 0 failed' "$L/r12-exl3r4d_layout.out"); its $named named refusal checks: K=2,3,5,6,7,8 mul1 -> RAD_E_DTYPE, codebook 3inst and mcg -> RAD_E_DTYPE, a stacked encoding on down -> RAD_E_DTYPE"
    else part FAIL "exl3r4d layout hook" "exl3r4d_layout did not pass (see $L/r12-exl3r4d_layout.out)"; fi
else part SKIP "exl3r4d layout hook" "kernels/exl3r4d and tests/gpu are not in this checkout"; fi

# exl3ref's launch refusals (P2's harness) and the architecture's (arch_static), listed
out=$L/r12-ref-cases.out
ref_host cases "$out"
n=$(grep -cE '^PASS refuse ' "$out"); f=$(grep -cE '^FAIL refuse ' "$out")
if [ "$n" -gt 0 ] && [ "$f" -eq 0 ]; then part PASS "exl3ref launch refusals" "$n named, each with its code:"; grep -E '^PASS refuse ' "$out" | sed 's/^PASS refuse //' | list
else part FAIL "exl3ref launch refusals" "$n passed, $f failed (see $out)"; fi
if ctest_v arch_static; then
    part PASS "architecture refusals (arch_static)" "$(grep -cE 'PASS .*(refused|refuses)' "$L/r12-arch_static.out") named:"
    grep -E 'PASS .*(refused|refuses)' "$L/r12-arch_static.out" | sed 's/.*PASS //' | list
else part FAIL "architecture refusals" "arch_static failed or skipped (see $L/r12-arch_static.out)"; fi

# on the card: the device rows' refusals through the loaded library (P4's device suite)
if why=$(device_session); then
    out=$L/r12-device.out
    device_suite device "$out"
    n=$(grep -cE '^PASS refuse ' "$out"); f=$(grep -cE '^(FAIL|SKIP) refuse ' "$out")
    if [ "$n" -gt 0 ] && [ "$f" -eq 0 ]; then part PASS "exl3r4d refusals on the card" "$n named:"; grep -E '^PASS refuse ' "$out" | sed 's/^PASS refuse //' | list
    else part FAIL "exl3r4d refusals on the card" "$n passed, $f failed/skipped (see $out)"; fi
else
    part SKIP "exl3r4d refusals on the card" "$why"
fi
finish "every R12 item refused by name with a test (mcg, 3inst, K!=4 on the device; half-integer K; bias)"
