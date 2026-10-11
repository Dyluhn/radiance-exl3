#!/bin/sh
# R8 -- deterministic. (1) scripts/ident.sh's six hashes identical across >= 3 restarts of the EXL3 container: the
# ident step run_pass.sh runs first on every fresh server, read by summarize.py (EXL3's hashes will differ from the
# baseline's; they must match each other). (2) replay bit-identical: rad-kbench cannot replay our ops (WTAB operands
# zero-filled at 1.3.0), so the replay is P4's device suite's -- the same case in three processes, byte-identical --
# in a GPU session.
#   SPEED_EXL3_PASSES="<run_pass.sh dirs>" tools/gates/R8.sh   [under tests/gpu/run_gpu.sh for (2)]
GATE=R8
. "$(dirname "$0")/lib.sh"
if [ -n "${SPEED_EXL3_PASSES:-}" ]; then
    mkdir -p "$GATES_WORK/speed"
    # shellcheck disable=SC2086
    python3 "$REPO/tools/gates/speed/summarize.py" $SPEED_EXL3_PASSES --json "$GATES_WORK/speed/r8.json" > "$GATES_WORK/speed/r8.md" 2>&1
    r=$(python3 - "$GATES_WORK/speed/r8.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
counted = [p for p in s["passes"] if p["label"] in s["counted"]]
with6 = [p for p in counted if len(p["ident"]) == 6]
same = len({tuple(p["ident"]) for p in with6}) == 1
ok = len(with6) >= 3 and len(with6) == len(counted) and same
h = " ".join(x[:8] for x in with6[0]["ident"]) if with6 else "--"
print(("PASS" if ok else "FAIL" if len(with6) >= 3 else "SKIP") +
      f"|{len(with6)} of {len(counted)} counted passes carry 6 hashes; identical: {same}; {h}")
PY
) || r="FAIL|summarize.py: $(tail -1 "$GATES_WORK/speed/r8.md")"
    part "${r%%|*}" "ident.sh across restarts" "${r#*|}"
else
    part SKIP "ident.sh across restarts" "no EXL3 passes: SPEED_EXL3_PASSES (run_pass.sh dirs of the EXL3 container, P6)"
fi
if why=$(device_session); then
    out=$GATES_WORK/logs/r8-device.out
    device_suite device "$out"
    n=$(grep -cE '^PASS .*replay' "$out"); f=$(grep -cE '^(FAIL|SKIP) .*replay' "$out")
    if [ "$n" -gt 0 ] && [ "$f" -eq 0 ]; then part PASS "replay bit-identical (exl3r4d)" "$(grep -E '^PASS .*replay' "$out" | cut -c6- | tr '\n' ';' | cut -c1-240)"
    else part FAIL "replay bit-identical (exl3r4d)" "$n passed, $f failed/skipped (see $out)"; fi
else part SKIP "replay bit-identical (exl3r4d)" "$why"; fi
echo "  note kbench replay: NOT CHECKABLE for moe_exl3_* at $PIN_VERSION (WTAB operands zero-filled); not counted"
finish "ident across EXL3 restarts; device replay"
