#!/bin/sh
# R11 -- a machine the device library does not serve is DECLINED BY NAME, not failed. radiance's loader reads the
# code objects exl3r4d's dlopen registered; with none that runs on the card the ranks will use -- or no card at all
# -- it leaves the library's device kernels out of selection and says so: "kernel plugin 'exl3r4d' has no device
# code for device N: <why>. Its K device kernel(s) are left out of selection here" (core/plugin/loader.cpp).
# exl3r4d exports no rad_plugin_open, as abi/rad_abi.h allows (the loader declines a card for a kernel library).
# THE INSTRUMENT IS THE ENGINE: `radiance --model <container> --debug-graph`. rad-info only selects, and on the host
# backend it keeps every library's device rows (libr4d's too), so it cannot show this (measured, notes/gates.md).
#   1. no card visible (BUILD_IMAGE, no devices: the host backend)  -> declined by name; libr4d alike (control)
#   2. the GPU session's card when it is NOT gfx12                  -> declined by name
#   3. the GPU session's card when it IS gfx12                      -> no such line (positive control)
#   R11_MODEL=<a container of ours, inside the repository; default $GATES_WORK/r4/fixture.rad, which R4 makes>
#   tools/gates/R11.sh                                                    part 1
#   tests/gpu/run_gpu.sh <pci of a card> <lock> <log dir> -- tools/gates/R11.sh    part 2 or 3 as well
GATE=R11
. "$(dirname "$0")/lib.sh"
SO=$REPO/$GATES_DEVICE_BUILD/radiance_home/kernels/exl3r4d.so
MODEL=${R11_MODEL:-$GATES_WORK/r4/fixture.rad}
[ -f "$SO" ] || { part SKIP "exl3r4d" "no device build at $GATES_DEVICE_BUILD/radiance_home/kernels/exl3r4d.so"; finish "nothing checked"; }
[ -f "$MODEL" ] || { part SKIP "a container to load" "no $MODEL: run tools/gates/R4.sh first or set R11_MODEL"; finish "nothing checked"; }
case $MODEL in "$REPO"/*) REL=${MODEL#"$REPO"/} ;; *) part SKIP "a container to load" "R11_MODEL must be inside the repository (it is /work/... in a session)"; finish "nothing checked" ;; esac
ARGS="--kernels exl3r4d,exl3ref,libr4d,libavx,libref --num-speculative-tokens 0 --max-model-len 4096 --debug-graph"

judge() {   # judge <part> <engine log>
    dev=$(grep -m1 -oE '(AQL|host) backend, [0-9]+ device\(s\) \[[^]]*\]' "$2")
    warn=$(grep -m1 -E "kernel plugin 'exl3r4d' has no device code" "$2")
    r4d=$(grep -m1 -E "kernel plugin 'libr4d' has no device code" "$2")
    loaded=$(grep -cE 'exl3r4d' "$2")
    case $dev in
        "") part FAIL "$1" "the engine named no device: $(grep -E '^E' "$2" | head -2 | tr '\n' ' ')" ;;
        *gfx12*) if [ -z "$warn" ] && [ "$loaded" -gt 0 ]; then part PASS "$1" "$dev: exl3r4d served, no decline (positive control)"
                 else part FAIL "$1" "$dev: a gfx12 card and exl3r4d is ${warn:+declined: $warn}${warn:-not seen}"; fi ;;
        *) if [ -n "$warn" ]; then part PASS "$1" "$dev: declined by name: $(echo "$warn" | sed 's/^W //' | cut -c1-230)"
           else part FAIL "$1" "$dev: not gfx12 and exl3r4d is NOT declined"; fi
           [ -n "$r4d" ] && echo "      control: libr4d alike: $(echo "$r4d" | sed 's/^W //' | cut -c1-120)" ;;
    esac
}
if command -v docker > /dev/null && docker image inspect "$BUILD_IMAGE" > /dev/null 2>&1; then
    # shellcheck disable=SC2086
    in_image -v "$REPO":/work:ro -w /work -e RADIANCE_HOME="/work/$GATES_DEVICE_BUILD/radiance_home:/stage/opt/radiance/share/radiance" \
        "$BUILD_IMAGE" /stage/opt/radiance/bin/radiance --model "/work/$REL" $ARGS > "$GATES_WORK/logs/r11-nocard.log" 2>&1
    judge "no card visible (host backend)" "$GATES_WORK/logs/r11-nocard.log"
else part SKIP "no card visible" "no docker or no image $BUILD_IMAGE"; fi
if why=$(device_session); then
    # shellcheck disable=SC2086
    docker exec -w /work -e RADIANCE_HOME="/work/$GATES_DEVICE_BUILD/radiance_home:/stage/opt/radiance/share/radiance" \
        "$EXL3_CONTAINER" /stage/opt/radiance/bin/radiance --model "/work/$REL" $ARGS > "$GATES_WORK/logs/r11-session.log" 2>&1
    judge "the session's card" "$GATES_WORK/logs/r11-session.log"
else part SKIP "a card" "$why"; fi
finish "exl3r4d declined by name where its code cannot run, served where it can"
