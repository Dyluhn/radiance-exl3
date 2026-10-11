#!/bin/sh
# R13 -- the oracle. Host: common/exl3_decode.c bit-exact against exl3_golden.npz (vllm-exl3's golden: K=3,4 x
# mcg, mul1), the C oracle against its NumPy twin, and against REAL turboderp tensors (W_hat bit-exact, plane
# round trip exact); with EXL3_TRELLIS_PY, the twin against exl3-moe's trellis.py itself. Device (GPU session):
# exl3r4d vs dense decode-and-multiply on a real turboderp layer, rel L2 <= 4e-3 (P4's `real` suite), and
# RADIANCE_OP_ORACLE clean on a real step -- BOTH (REQUIREMENTS R13, 2026-10-11): (a) the fixture's engine steps on the
# card with the SHIPPED band split, decode rows and the M > 16 `_pf` rows (tests/gpu/prefill/oracle_pf.sh, P5's: every
# issue re-run on exl3ref and compared, OP_REPEAT byte-compares, and the `_pf` band must have been judged); (b) a
# real-model step, P6's: R13_REAL_ORACLE_LOG = the engine log of a real-container run with RADIANCE_OP_ORACLE set.
#
#   RADIANCE_PREFIX=<host install> EXL3_REAL_EXTRACT=<.safetensors of exllamav3 tensors> tools/gates/R13.sh
#   tests/gpu/run_gpu.sh <pci> <lock> <log dir> -- tools/gates/R13.sh     the device parts; TOKENIZER (the Qwen3.8
#   tokenizer files) for the fixture
GATE=R13
. "$(dirname "$0")/lib.sh"
why=$(need_prefix) || { part SKIP "pinned install" "$why"; finish "nothing checked"; }
host_build || { part FAIL "host build" "see $GATES_WORK/logs/host-build.log"; finish "no build"; }
L=$GATES_WORK/logs
ct() {   # ct <part> <regex of ctest names> -- PASS only when every named test ran and passed
    ctest --test-dir "$GATES_BUILD" -R "$2" -V > "$L/r13-ct.out" 2>&1; rc=$?
    n=$(grep -cE 'Test +#[0-9]+: .* Passed' "$L/r13-ct.out"); s=$(grep -cE '\*\*\*Skipped|Not Run' "$L/r13-ct.out")
    detail=$(grep -hE 'SKIPPED|mismatch|bit-exact|exact|worst|passed, [0-9]+ failed|ok |W_hat' "$L/r13-ct.out" | grep -vE '^[0-9]+: Test command' | sed 's/^[0-9]*: //' | tail -3 | tr '\n' ' ' | cut -c1-240)
    if [ $rc -ne 0 ]; then part FAIL "$1" "ctest failed: $detail (see $L/r13-ct.out)"
    elif [ "$s" -gt 0 ] || [ "$n" -eq 0 ]; then part SKIP "$1" "$detail"
    else part PASS "$1" "$n test(s): $detail"; fi
    cp "$L/r13-ct.out" "$L/r13-$(echo "$1" | tr -c 'a-z0-9' '_').out"
}
ct "golden: exl3_decode bit-exact vs exl3_golden.npz" '^(exl3_decode_test|exl3_golden_header|exl3_c_vs_ref)$'
ct "real extract: C oracle on turboderp tensors" '^exl3_real_extract$'
ct "twin vs exl3-moe trellis.py" '^exl3_ref_vs_trellis_py$'

if why=$(device_session); then
    if [ -n "${EXL3_REAL_EXTRACT:-}" ]; then
        out=$L/r13-device-real.out
        device_suite real "$out"
        worst=$(grep -E 'device vs decode-and-multiply' "$out" | grep -oE 'rel_l2=[0-9.e+-]+' | sed 's/rel_l2=//' | sort -g | tail -1)
        ctl=$(grep -E 'control, a wrong reference is caught' "$out" | grep -oE 'rel_l2=[0-9.e+-]+')
        if clean_tally "$out"; then part PASS "exl3r4d vs decode-and-multiply, real layer" "$(tally "$out"); worst rel_l2 $worst (tol 4e-3); control (a wrong reference) caught at $ctl"
        else part FAIL "exl3r4d vs decode-and-multiply, real layer" "$(tally "$out"): $(grep -E '^FAIL' "$out" | head -2 | tr '\n' ' ')"; fi
    else part SKIP "exl3r4d vs decode-and-multiply, real layer" "EXL3_REAL_EXTRACT not set"; fi
    if [ -f "$REPO/tests/gpu/prefill/oracle_pf.sh" ] && [ -f "${TOKENIZER:-}/tokenizer.json" ]; then
        W=$GATES_WORK/r13-oracle
        rel=${W#"$REPO"/}
        [ -f "$W/fx/fixture.json" ] || python3 "$REPO/tests/arch/fixture_variant.py" --suh-scale 10 --lm-scale 8 -- \
            --out "$W/fx" --tokenizer "$TOKENIZER" --layers 2 --experts 8 --topk 2 --hidden 256 --width 640 --seed 1 \
            > "$L/r13-fixture.log" 2>&1 || { part FAIL "OP_ORACLE" "fixture: $(tail -1 "$L/r13-fixture.log")"; finish "oracle"; }
        docker exec -w /work -e EXL3_HOME="/work/$GATES_DEVICE_BUILD/radiance_home" "$EXL3_CONTAINER" \
            sh tests/gpu/prefill/oracle_pf.sh "/work/$rel/fx/bf16" "/work/$rel/fx/exl3" "/work/$rel/run" > "$L/r13-oracle.out" 2>&1
        rc=$?
        o=$W/run/c2-oracle.json.log r=$W/run/c2-repeat.json.log
        checked=$(grep -oE 'op oracle: [0-9]+ op\(s\) checked, [0-9]+ skipped' "$o" 2>/dev/null | tail -1)
        rep=$(grep -oE 'op REPEAT: [0-9]+ issue\(s\) run twice, [0-9]+ of them ANSWERED DIFFERENTLY; [0-9]+ more were NOT JUDGED' "$r" 2>/dev/null | tail -1)
        pfline=$(grep -E '^PASS oracle:' "$L/r13-oracle.out" | sed 's/^PASS //')
        if [ $rc -eq 0 ] && ! grep -q '^FAIL' "$L/r13-oracle.out" && echo "$checked" | grep -qE ' [1-9][0-9]* op\(s\) checked, 0 skipped'; then
            part PASS "RADIANCE_OP_ORACLE on fixture engine steps (shipped bands)" "$checked; $pfline; $rep; $(grep -c '^PASS' "$L/r13-oracle.out") oracle_pf checks passed"
        else part FAIL "RADIANCE_OP_ORACLE on fixture engine steps (shipped bands)" "rc=$rc; ${checked:-no summary}; $(grep '^FAIL' "$L/r13-oracle.out" | head -2 | tr '\n' ' ') (see $L/r13-oracle.out)"; fi
    else
        part SKIP "RADIANCE_OP_ORACLE on fixture engine steps (shipped bands)" "needs tests/gpu/prefill/oracle_pf.sh and TOKENIZER"
    fi
else
    part SKIP "exl3r4d vs decode-and-multiply, real layer" "$why"
    part SKIP "RADIANCE_OP_ORACLE on fixture engine steps (shipped bands)" "$why"
fi
# (b) the real-model step: P6 runs the real container with RADIANCE_OP_ORACLE=moe_exl3_gate_up,moe_exl3_down and hands
# its engine log here; the same rule as the fixture's (something checked, nothing skipped, every issue agrees).
if [ -f "${R13_REAL_ORACLE_LOG:-}" ]; then
    o=$R13_REAL_ORACLE_LOG
    checked=$(grep -oE 'op oracle: [0-9]+ op\(s\) checked, [0-9]+ skipped' "$o" | tail -1)
    agree=$(grep -cE "oracle #[0-9]+ op 'moe_exl3_[a-z_]+' agrees" "$o")
    bad=$(grep -E "oracle #[0-9]+ op 'moe_exl3_[a-z_]+'" "$o" | grep -vc agrees)
    if [ "$agree" -gt 0 ] && [ "$bad" -eq 0 ] && echo "$checked" | grep -qE ' [1-9][0-9]* op\(s\) checked, 0 skipped'; then
        part PASS "RADIANCE_OP_ORACLE on a real-model step" "$checked; $agree agree, 0 not"
    else part FAIL "RADIANCE_OP_ORACLE on a real-model step" "${checked:-no summary}; $agree agree, $bad not agreeing ($o)"; fi
else
    part SKIP "RADIANCE_OP_ORACLE on a real-model step" "the real container is P6's: R13_REAL_ORACLE_LOG=<its engine log with RADIANCE_OP_ORACLE set>"
fi
finish "decoder bit-exact vs the golden and on real tensors; device parts in a GPU session"
