#!/bin/sh
# R2 -- every declared op resolves to OUR libraries at every band: `radiance --debug-graph` WITHOUT
# --debug-accept-reference-kernels on the fixture container and on the real one; no moe_exl3_* row may fall to a
# reference library (exl3ref is one: rad_plugin_reference() = 1) and the engine must not refuse to serve.
# Run where the card(s) and the model are visible, with this repository's device home:
#   docker exec -w /work "$EXL3_CONTAINER" env GATE_DEVICE=1 RADIANCE_PREFIX=/stage/opt/radiance \
#       EXL3_HOME=/work/build-gpu/radiance_home R2_MODELS="<fixture.rad> [<real.rad>]" sh tools/gates/R2.sh
# Each model is one part; the real container is P6's. Until exl3r4d declares rows above M = 16 (P5), the prefill
# bands fall to exl3ref and this gate FAILS when run -- which is what it is for.
GATE=R2
. "$(dirname "$0")/lib.sh"
[ "${GATE_DEVICE:-0}" = 1 ] || { part SKIP "device declare" "not in a device session: GATE_DEVICE=1 inside a container with the card visible (header); exl3r4d serves M <= 16 only until P5's prefill rows"; finish "nothing checked"; }
why=$(need_prefix) || { part SKIP "pinned install" "$why"; finish "nothing checked"; }
[ -f "${EXL3_HOME:-}/kernels/exl3r4d.so" ] || { part SKIP "device declare" "EXL3_HOME has no kernels/exl3r4d.so"; finish "nothing checked"; }
[ -n "${R2_MODELS:-}" ] || { part SKIP "device declare" "R2_MODELS names no container"; finish "nothing checked"; }
for model in $R2_MODELS; do
    log=$GATES_WORK/logs/r2-$(basename "$model").log
    RADIANCE_HOME=$EXL3_HOME:$RADIANCE_PREFIX/share/radiance timeout 900 "$RADIANCE_PREFIX/bin/radiance" --model "$model" \
        --kernels exl3r4d,exl3ref,libr4d,libavx,libref --debug-graph > "$log" 2>&1
    rc=$?
    ref=$(grep -E '^ +moe_exl3_(gate_up|down) .*\(exl3ref, ' "$log" | sed 's/  */ /g' | head -3)
    ours=$(grep -cE '^ +moe_exl3_(gate_up|down) .*\(exl3r4d, ' "$log")
    if [ -n "$ref" ]; then part FAIL "$(basename "$model")" "a band falls to the reference: $ref"
    elif [ $rc -ne 0 ]; then part FAIL "$(basename "$model")" "rc=$rc: $(grep -E '^E' "$log" | head -2 | tr '\n' ' ' | cut -c1-200)"
    elif [ "$ours" -eq 0 ]; then part FAIL "$(basename "$model")" "no moe_exl3 selection line names exl3r4d (see $log)"
    else part PASS "$(basename "$model")" "$ours moe_exl3 band/domain rows, every one exl3r4d"; fi
done
finish "moe_exl3_* resolves to exl3r4d at every band, no reference fallback"
