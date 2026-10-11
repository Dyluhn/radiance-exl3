#!/bin/sh
# R5 -- an absent optional operand is null data, not a shorter operand list: the direct-call cases with the odd
# parity class (w_odd and its scale tables) absent, both ops, through P2's engine-style harness (tests/ref), which
# hands the row a zeroed RadTensor (data NULL) at the operand's own position and keeps n_t. Each case must launch,
# keep its margins and inputs, match the transcription, and its opd_shape must describe the operands absent.
# Controls in the same run: an odd class only PARTLY absent is refused (RAD_E_INVAL), and N_odd with no w_odd is
# refused (RAD_E_SHAPE) -- the row reads absence, it does not guess it.
# At 1.3.0 an architecture cannot issue an absent optional table (rad_builder.cpp:697-731), so in the engine tp1
# passes both parity tables; this is the direct-call contract only.
#
#   RADIANCE_PREFIX=<host install> tools/gates/R5.sh        host rows (exl3ref)
#   under tests/gpu/run_gpu.sh, with EXL3_REAL_EXTRACT, as well: the device rows (exl3r4d), below
GATE=R5
. "$(dirname "$0")/lib.sh"
why=$(need_prefix) || { part SKIP "pinned install" "$why"; finish "nothing checked"; }
host_build || { part FAIL "host build" "see $GATES_WORK/logs/host-build.log"; finish "no build"; }
out=$GATES_WORK/logs/r5-ref-cases.out
ref_host cases "$out"
check() {   # check <domain> <case name> <harness output>
    n=$(grep -cF "PASS $2" "$3"); bad=$(grep -E "^(FAIL|SKIP) " "$3" | grep -cF "$2")
    if [ "$n" -ge 6 ] && [ "$bad" -eq 0 ]; then part PASS "$1: $2" "$n checks passed: $(grep -F "PASS $2" "$3" | sed "s/^PASS $2: //; s/  .*//" | tr '\n' ',' | sed 's/,$//')"
    else part FAIL "$1: $2" "$n passed, $bad failed or skipped (see $3)"; fi
}
check exl3ref "gate_up w_odd absent (tp1)" "$out"
check exl3ref "down w_odd absent (tp1)" "$out"
for c in "refuse odd class partly absent" "refuse N_odd without w_odd"; do
    l=$(grep -F "$c " "$out")
    case $l in PASS*) part PASS "control: $c" "${l#PASS $c  }" ;; *) part FAIL "control: $c" "${l:-not run}" ;; esac
done
# Device rows: P4's `real` suite launches every case with the odd class absent (two=False): a real turboderp
# tensor at M = 1, 5, 7, 16, both ops, against decode-and-multiply.
if why=$(device_session); then
    if [ -n "${EXL3_REAL_EXTRACT:-}" ]; then
        dout=$GATES_WORK/logs/r5-device-real.out
        device_suite real "$dout"
        n=$(grep -cE '^PASS real (gate_up|down) M=[0-9]+: ' "$dout"); bad=$(grep -cE '^(FAIL|SKIP) real' "$dout")
        if [ "$n" -ge 16 ] && [ "$bad" -eq 0 ]; then part PASS "exl3r4d: odd class absent (real suite)" "$n launches+comparisons passed; $(tally "$dout")"
        else part FAIL "exl3r4d: odd class absent (real suite)" "$n passed, $bad failed/skipped (see $dout)"; fi
    else
        part SKIP "exl3r4d: odd class absent" "EXL3_REAL_EXTRACT not set (P4's real suite is the device case with the odd class absent)"
    fi
else
    part SKIP "exl3r4d: odd class absent" "$why"
fi
finish "direct call with the odd class absent: $(tally "$out")"
