#!/bin/sh
# R13 -- the oracle. Host: common/exl3_decode.c bit-exact against exl3_golden.npz (vllm-exl3's golden: K=3,4 x
# mcg, mul1), the C oracle against its NumPy twin, and against REAL turboderp tensors (W_hat bit-exact, plane
# round trip exact); with EXL3_TRELLIS_PY, the twin against exl3-moe's trellis.py itself. Device (GPU session):
# exl3r4d vs dense decode-and-multiply on a real turboderp layer, rel L2 <= 4e-3 (P4's `real` suite), and
# RADIANCE_OP_ORACLE clean on real engine steps (tests/gpu/op_oracle.sh: every issue of our two ops re-run on the
# host reference and compared).
#
#   RADIANCE_PREFIX=<host install> EXL3_REAL_EXTRACT=<.safetensors of exllamav3 tensors> tools/gates/R13.sh
#   tests/gpu/run_gpu.sh <pci> <lock> <log dir> -- tools/gates/R13.sh, adding for the oracle part:
#     R13_ORACLE_HOME  a $RADIANCE_HOME relative to the repository holding this build's plugins with a TEST build of
#                      exl3r4d at -DEXL3R4D_M_MAX=64 (op_oracle.sh says why; never the shipped library)
#     TOKENIZER        the Qwen3.8 tokenizer files (the fixture needs the real vocabulary)
# The OP_ORACLE part runs on P3's fixture, not on the real model: a real-model step is P6's.
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
    if [ -n "${R13_ORACLE_HOME:-}" ] && [ -f "$REPO/$R13_ORACLE_HOME/kernels/exl3r4d.so" ] && [ -f "${TOKENIZER:-}/tokenizer.json" ]; then
        W=$GATES_WORK/r13-oracle
        rel=${W#"$REPO"/}
        [ -f "$W/fx/fixture.json" ] || python3 "$REPO/tests/arch/fixture_variant.py" --suh-scale 10 --lm-scale 8 -- \
            --out "$W/fx" --tokenizer "$TOKENIZER" --layers 2 --experts 8 --topk 2 --hidden 256 --width 640 --seed 1 \
            > "$L/r13-fixture.log" 2>&1 || { part FAIL "OP_ORACLE" "fixture: $(tail -1 "$L/r13-fixture.log")"; finish "oracle"; }
        docker exec -w /work -e EXL3_HOME="/work/$R13_ORACLE_HOME" "$EXL3_CONTAINER" \
            sh tests/gpu/op_oracle.sh "/work/$rel/fx/bf16" "/work/$rel/fx/exl3" "/work/$rel/run" > "$L/r13-oracle.out" 2>&1
        rc=$?
        o=$W/run/c2-oracle.json.log r=$W/run/c2-repeat.json.log
        checked=$(grep -oE 'op oracle: [0-9]+ op\(s\) checked, [0-9]+ skipped' "$o" 2>/dev/null | tail -1)
        agree=$(grep -cE "^W oracle #[0-9]+ op 'moe_exl3_[a-z_]+' agrees" "$o" 2>/dev/null)
        bad=$(grep -cE "^W oracle #[0-9]+ op 'moe_exl3_[a-z_]+'.*(MISMATCH|NOT CHECKED)" "$o" 2>/dev/null)
        rep=$(grep -oE 'op REPEAT: [0-9]+ issue\(s\) run twice, [0-9]+ of them ANSWERED DIFFERENTLY; [0-9]+ more were NOT JUDGED' "$r" 2>/dev/null | tail -1)
        if [ $rc -eq 0 ] && [ "${agree:-0}" -gt 0 ] && [ "${bad:-1}" -eq 0 ] && echo "$checked" | grep -qE ' [1-9][0-9]* op\(s\) checked, 0 skipped' &&
           echo "$rep" | grep -qE ' [1-9][0-9]* issue\(s\) run twice, 0 of them .* 0 more'; then
            part PASS "RADIANCE_OP_ORACLE on fixture engine steps" "$checked; $agree agree, 0 mismatch / not checked; $rep"
        else part FAIL "RADIANCE_OP_ORACLE on fixture engine steps" "rc=$rc; ${checked:-no summary}; $agree agree, $bad mismatch/not checked; ${rep:-no repeat summary} (see $L/r13-oracle.out)"; fi
    else
        part SKIP "RADIANCE_OP_ORACLE on fixture engine steps" "needs R13_ORACLE_HOME (a home with an -DEXL3R4D_M_MAX=64 test build of exl3r4d) and TOKENIZER"
    fi
else
    part SKIP "exl3r4d vs decode-and-multiply, real layer" "$why"
    part SKIP "RADIANCE_OP_ORACLE on fixture engine steps" "$why"
fi
finish "decoder bit-exact vs the golden and on real tensors; device parts in a GPU session"
