#!/bin/sh
# R4 -- one stored layout for all bands: the engine declares a container of ours at --tp 1 and --tp 2 with no
# "wanted in two stored forms" (the layout hook is asked at every band and domain; the first answer binds).
#
#   host:   RADIANCE_PREFIX=<host install> [TOKENIZER=<dir>] tools/gates/R4.sh
#   device: inside a container of the pinned image with the card(s) visible and this repository at its path:
#           GATE_DEVICE=1 RADIANCE_PREFIX=/stage/opt/radiance EXL3_HOME=<home with exl3r4d.so ...> \
#           R4_MODEL=<the container host mode made> sh tools/gates/R4.sh       (tp2 needs both cards)
#
# The container: tools/exl3_fixture.py (2 layers, 8 experts, hidden 256, width 640) converted through this
# repository's architecture, quantiser and exl3ref, unless R4_MODEL names one. Declared by
# `radiance --debug-graph` (declares every rank and exits). Host tp2 runs two host devices (RADIANCE_HOST_DEVICES).
#   R4_TP       the worlds to declare (default "1 2")
#   TOKENIZER   a directory with the Qwen3.8 tokenizer files (optional: the fixture's own small vocabulary without)
# A declare refused ONLY because a host-only radiance has no row for an op the in-tree block asks for at that world
# (1.3.0: ar_hc_write, the tp2 gather/all-reduce of the hyper-connection write) is SKIPPED by name, not passed.
GATE=R4
. "$(dirname "$0")/lib.sh"
why=$(need_prefix) || { part SKIP "pinned install" "$why"; finish "nothing checked"; }
W=$GATES_WORK/r4
mkdir -p "$W"
INSTALL_HOME=$RADIANCE_PREFIX/share/radiance
if [ "${GATE_DEVICE:-0}" = 1 ]; then
    [ -f "${EXL3_HOME:-}/kernels/exl3r4d.so" ] || { part SKIP "device declare" "EXL3_HOME has no kernels/exl3r4d.so"; finish "nothing checked"; }
    HOMES=$EXL3_HOME:$INSTALL_HOME KERNELS=exl3r4d,exl3ref,libr4d,libavx,libref MODE=device
else
    host_build || { part FAIL "host build" "see $GATES_WORK/logs/host-build.log"; finish "no build"; }
    HOMES=$GATES_BUILD/tests/arch/test_home:$GATES_BUILD/radiance_home:$INSTALL_HOME
    KERNELS=exl3test_router,exl3ref,libavx,libref MODE=host
fi

MODEL=${R4_MODEL:-$W/fixture.rad}
if [ ! -f "$MODEL" ]; then
    [ "$MODE" = host ] || { part SKIP "fixture container" "device mode needs R4_MODEL (host mode makes it)"; finish "nothing checked"; }
    python3 -c 'import numpy' 2> /dev/null || { part SKIP "fixture container" "python3 has no numpy"; finish "nothing checked"; }
    rm -rf "$W/fx"
    python3 "$REPO/tools/exl3_fixture.py" --out "$W/fx" ${TOKENIZER:+--tokenizer "$TOKENIZER"} --layers 2 --experts 8 \
        --topk 2 --hidden 256 --width 640 --seed 1 > "$W/fixture.log" 2>&1 ||
        { part FAIL "fixture" "$(tail -1 "$W/fixture.log")"; finish "no fixture"; }
    EXL3_SRC=$W/fx/exl3 RADIANCE_HOME=$HOMES "$RADIANCE_PREFIX/bin/rad-convert" "$W/fx/bf16" \
        --recipe "$REPO/tests/arch/data/fixture-exl3.recipe" --kernels "$KERNELS" --set indexer_n_heads=0 --max-spec 0 \
        -o "$MODEL.tmp" > "$W/convert.log" 2>&1 && mv "$MODEL.tmp" "$MODEL" ||
        { part FAIL "convert" "$(grep -E '^E' "$W/convert.log" | head -2 | tr '\n' ' ')"; finish "no container"; }
fi

for tp in ${R4_TP:-1 2}; do
    log=$W/declare-$MODE-tp$tp.log
    hd=""; [ "$MODE" = host ] && hd="RADIANCE_HOST_DEVICES=$tp"
    env $hd RADIANCE_HOME="$HOMES" timeout 600 "$RADIANCE_PREFIX/bin/radiance" --model "$MODEL" --tp "$tp" \
        --kernels "$KERNELS" --num-speculative-tokens 0 --max-model-len 4096 --debug-graph \
        --debug-accept-reference-kernels > "$log" 2>&1
    rc=$?
    ops=$(grep -cE '^ *moe_exl3_(gate_up|down) ' "$log")
    missing=$(grep -oE "declares op '[a-z_0-9]+'" "$log" | sort -u | sed "s/declares op //" | tr '\n' ' ')
    if grep -q 'wanted in two stored forms' "$log"; then
        part FAIL "$MODE tp$tp" "$(grep -m1 'wanted in two stored forms' "$log" | cut -c1-220)"
    elif [ $rc -eq 0 ] && [ "$ops" -gt 0 ]; then
        part PASS "$MODE tp$tp" "declared, $ops moe_exl3 op lines, no second stored form (log $log)"
    elif [ "$MODE" = host ] && [ "$missing" = "'ar_hc_write' " ]; then
        part SKIP "$MODE tp$tp" "BLOCKED: radiance $PIN_VERSION host-only has no row for ar_hc_write (the in-tree block's tp2 hc gather); run in device mode with both cards"
    else
        part FAIL "$MODE tp$tp" "rc=$rc: $(grep -E '^E' "$log" | sort -u | head -2 | tr '\n' ' ' | cut -c1-220)"
    fi
done
finish "$MODE declare of $(basename "$MODEL") at tp ${R4_TP:-1 2}"
