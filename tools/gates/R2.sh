#!/bin/sh
# R2 -- every declared op resolves to OUR libraries at every band: `radiance --debug-graph` WITHOUT
# --debug-accept-reference-kernels, with the card visible. Per container: the engine accepts the graph (exit 0: an op
# served only by a reference library would be refused), every DEVICE-domain selection line of moe_exl3_gate_up /
# moe_exl3_down names an exl3r4d row (none exl3ref, none "-- none"), and neither op is in the planner's "host-only ops"
# list. The HOST-domain lines always name exl3ref -- the host twin the planner may place an op on -- and are counted,
# not judged; a host placement of our ops shows up in the host-only list instead.
#
#   tests/gpu/run_gpu.sh <pci> <lock> <log dir> -- tools/gates/R2.sh       (runs radiance in the session container)
#   or inside a card container: GATE_DEVICE=1 RADIANCE_PREFIX=/stage/opt/radiance sh tools/gates/R2.sh
# R2_MODELS   containers of ours, inside the repository (default: the fixture R13's oracle run converts on the card,
#             $GATES_WORK/r13-oracle/run/ours.rad, and R4's host-converted fixture, $GATES_WORK/r4/fixture.rad --
#             whichever exist). The real container is P6's: R2_REAL=<its path inside the container> adds it.
# R2_FLAGS    serving flags (default: --max-model-len 4096 --max-num-batched-tokens 2048, the production budget)
GATE=R2
. "$(dirname "$0")/lib.sh"
FLAGS=${R2_FLAGS:---max-model-len 4096 --max-num-batched-tokens 2048}
K=exl3r4d,exl3ref,libr4d,libavx,libref
if [ "${GATE_DEVICE:-0}" = 1 ]; then
    run() { RADIANCE_HOME="$REPO/$GATES_DEVICE_BUILD/radiance_home:$RADIANCE_PREFIX/share/radiance" \
            "$RADIANCE_PREFIX/bin/radiance" --model "$1" --kernels $K $FLAGS --debug-graph; }
    where() { echo "$1"; }
elif why=$(device_session); then
    run() { docker exec -w /work -e RADIANCE_HOME="/work/$GATES_DEVICE_BUILD/radiance_home:/stage/opt/radiance/share/radiance" \
            "$EXL3_CONTAINER" /stage/opt/radiance/bin/radiance --model "$1" --kernels $K $FLAGS --debug-graph; }
    where() { echo "/work/${1#"$REPO"/}"; }
else
    part SKIP "device declare" "$why"; finish "nothing checked"
fi
models=${R2_MODELS:-$(ls "$GATES_WORK/r13-oracle/run/ours.rad" "$GATES_WORK/r4/fixture.rad" 2>/dev/null)}
[ -n "$models" ] || { part SKIP "fixture" "no container: run R13 (device fixture) or R4 (host fixture) first, or set R2_MODELS"; finish "nothing checked"; }
for model in $models ${R2_REAL:+REAL:$R2_REAL}; do
    case $model in REAL:*) path=${model#REAL:} name="real: $(basename "$path")" ;;
                   *) path=$(where "$model") name="$(basename "$(dirname "$model")")/$(basename "$model")" ;; esac
    log=$GATES_WORK/logs/r2-$(echo "$name" | tr -c 'A-Za-z0-9.-' '_').log
    run "$path" > "$log" 2>&1
    rc=$?
    dev=$(grep -E '^ +moe_exl3_(gate_up|down) +device ' "$log")
    ours=$(printf '%s\n' "$dev" | grep -c '(exl3r4d, ')
    other=$(printf '%s\n' "$dev" | grep -v '(exl3r4d, ' | grep -c .)
    host=$(grep -cE '^ +moe_exl3_(gate_up|down) +host ' "$log")
    hostonly=$(grep -E '^I host-only ops:' "$log" | grep -oE 'moe_exl3_[a-z_]+' | tr '\n' ' ')
    bands=$(printf '%s\n' "$dev" | grep -oE 'M=[0-9]+' | sort -u | tr '\n' ' ')
    rows=$(printf '%s\n' "$dev" | grep -oE 'exl3r4d_moe_[a-z_0-9]+' | sort -u | tr '\n' ' ')
    if [ $rc -ne 0 ]; then part FAIL "$name" "the engine refused the graph without --debug-accept-reference-kernels: $(grep -E '^E' "$log" | head -2 | tr '\n' ' ' | cut -c1-220)"
    elif [ "$other" -gt 0 ]; then part FAIL "$name" "$other device line(s) not on exl3r4d: $(printf '%s\n' "$dev" | grep -v '(exl3r4d, ' | head -2 | sed 's/  */ /g' | tr '\n' ' ' | cut -c1-220)"
    elif [ -n "$hostonly" ]; then part FAIL "$name" "the planner placed $hostonly on the host (a reference library)"
    elif [ "$ours" -eq 0 ]; then part FAIL "$name" "no device selection line for moe_exl3_* (see $log)"
    else part PASS "$name" "accepted without --debug-accept-reference-kernels; $ours device band lines all exl3r4d (bands ${bands% }; rows ${rows% }); $host host-twin lines on exl3ref (not served)"; fi
done
[ -n "${R2_REAL:-}" ] || part SKIP "real container" "the EXL3 container is P6's: R2_REAL=<path inside the session container>"
finish "moe_exl3_* on exl3r4d at every device band, no reference fallback"
