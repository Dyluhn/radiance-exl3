#!/bin/sh
# R20 -- portability, over exactly what is committed (`git archive HEAD`):
#   1. no machine in the source: no home path, no PCI address, no host name, no card name, no storage label. A hit
#      passes only when it is listed in JUSTIFIED below with its reason (each is printed).
#   2. the machine-constant grep `\b(64|32768|65536|98304)\b` over the SHIPPED sources (kernels/, common/*.[ch],
#      include/, quantizers/exl3/*.c*, architectures/): every hit is prose in a comment, a line of the vendored
#      architecture byte-identical to the pinned release's (upstream's number; tests/upstream_drift.sh pins it), or
#      listed in CONSTANTS below with what fact it is; anything else FAILS. Tests and tools are counted, not judged.
#   3. the archive builds in the Docker build stage (BUILD_IMAGE, device rows on: RAD_WITH_HIP=AUTO), offline.
#   RADIANCE_SRC (for 2's vendored check) and docker with BUILD_IMAGE (for 3); each part SKIPS by name without.
GATE=R20
. "$(dirname "$0")/lib.sh"
W=$GATES_WORK/r20
archive_tree "$W/src" || { part FAIL "git archive" "of HEAD into $W/src"; finish "no archive"; }

# ---- 1. the machine. The pattern is spelled with brackets so that this file does not match itself.
MACHINE='/var/hom[e]|/home/dyla[n]|13:00[.]0|000[0]:|blackbo[x]|R970[0]|qwen-storag[e]'
JUSTIFIED='tests/p6/test_recipe.py:assert not re.search(r"/(var|home|mnt|tmp)/|qwen-|the recipe test ASSERTS that no machine path is in the recipe'
hits=$(cd "$W/src" && grep -rnE "$MACHINE" . | sed 's|^\./||')
bad=""
printf '%s\n' "$hits" | while IFS= read -r h; do [ -n "$h" ] && echo "$h"; done > "$W/machine.hits"
while IFS= read -r h; do
    f=${h%%:*}; text=${h#*:*:}
    ok=$(printf '%s\n' "$JUSTIFIED" | while IFS='|' read -r jf jt why; do
        [ "$f" = "${jf%%:*}" ] && case $text in *"${jf#*:}"*) echo "$why";; esac; done | head -1)
    if [ -n "$ok" ]; then echo "      justified  ${h%%:*}:$(echo "$h" | cut -d: -f2): $ok"; else bad="$bad
$h"; fi
done < "$W/machine.hits"
if [ "$(echo "a /var/""home/x" | grep -cE "$MACHINE")" != 1 ]; then part FAIL "machine grep" "the pattern misses a planted path (control)"
elif [ -n "$bad" ]; then part FAIL "machine grep" "unjustified:$(printf '%s' "$bad" | head -5 | tr '\n' ' ')"
else part PASS "machine grep" "$(grep -c . "$W/machine.hits") hit(s), each justified above; control (a planted path) caught"; fi

# ---- 2. machine constants in the shipped sources
NUM='\b(64|32768|65536|98304)\b'
CONSTANTS='kernels/exl3r4d/exl3r4d_dev.h:#define EXL3R4D_LDS_MAX (64 * 1024 - 256)|LDS a workgroup may hold on gfx12 (64 KiB), less static __shared__; the library is gfx1201 code only (build_target) and the comment above it cites the measured queue abort at the full 64 KiB (FACTS-exl3 trap 13)
quantizers/exl3/exl3_src.cpp:if (depth > 64) return fail("nested deeper than 64");|the JSON reader'"'"'s recursion bound on a safetensors index (exl3_src.h says so); a parser limit, not a machine fact'
shipped=$(cd "$W/src" && ls kernels/*/*.c kernels/*/*.h kernels/*/*.hip common/*.c common/*.h include/*.h quantizers/exl3/*.c* quantizers/exl3/*.h architectures/*/*.cpp architectures/*/*.h 2>/dev/null)
n_comment=0 n_vendor=0 n_listed=0 unexplained=""
for f in $shipped; do
    grep -nE "$NUM" "$W/src/$f" > "$W/num.tmp" || continue
    while IFS= read -r l; do
        text=${l#*:}; ln=${l%%:*}
        code=$(printf '%s' "$text" | sed 's|//.*||; s|/\*.*\*/||; s|^ *\*.*||; s|^ */\*.*||')
        if ! printf '%s' "$code" | grep -qE "$NUM"; then n_comment=$((n_comment + 1)); continue; fi
        case $f in architectures/*)
            if [ -n "${RADIANCE_SRC:-}" ] && grep -qxF -- "$text" "$RADIANCE_SRC/arch/qwen4exp_fp8/qwen4exp_fp8.cpp" "$RADIANCE_SRC/arch/common/rad_block_moe_fp8.h" 2>/dev/null; then
                n_vendor=$((n_vendor + 1)); continue; fi ;;
        esac
        why=$(printf '%s\n' "$CONSTANTS" | while IFS='|' read -r cf why; do
            [ "$f" = "${cf%%:*}" ] && case $text in *"${cf#*:}"*) echo "$why";; esac; done | head -1)
        if [ -n "$why" ]; then n_listed=$((n_listed + 1)); echo "      constant   $f:$ln: $why"
        else unexplained="$unexplained
$f:$ln:$text"; fi
    done < "$W/num.tmp"
done
counted=$(cd "$W/src" && grep -rlE "$NUM" tests tools 2>/dev/null | while read -r f; do echo "$f=$(grep -cE "$NUM" "$f")"; done | tr '\n' ' ')
if [ -n "$unexplained" ]; then
    part FAIL "machine constants in shipped sources" "unexplained:$(printf '%s' "$unexplained" | head -6 | tr '\n' ' ' | cut -c1-400)"
elif [ -z "${RADIANCE_SRC:-}" ]; then
    part SKIP "machine constants in shipped sources" "RADIANCE_SRC not set: the vendored lines cannot be told from ours"
else
    part PASS "machine constants in shipped sources" "$n_comment in comments, $n_vendor vendored verbatim from radiance $PIN_VERSION, $n_listed listed constants, 0 unexplained"
fi
echo "      not judged (tests, tools): $counted"

# ---- 3. the archive builds in the build stage, device rows on, no network
if ! command -v docker > /dev/null || ! docker image inspect "$BUILD_IMAGE" > /dev/null 2>&1; then
    part SKIP "git archive build in $BUILD_IMAGE" "no docker or no image $BUILD_IMAGE"
else
    mkdir -p "$W/out"
    in_image -v "$W/src":/exl3:ro -v "$W/out":/out "$BUILD_IMAGE" sh -c \
        'cmake -S /exl3 -B /tmp/b -G Ninja -DCMAKE_PREFIX_PATH=/stage/opt/radiance -DRADIANCE_SRC=/src > /out/configure.log 2>&1 &&
         cmake --build /tmp/b > /out/build.log 2>&1; rc=$?; ls /tmp/b/radiance_home/*/*.so > /out/plugins.txt 2>&1;
         grep -c "warning:" /out/build.log > /out/warnings.txt; exit $rc'
    rc=$?
    digest=$(docker image inspect --format '{{.Id}}' "$BUILD_IMAGE" | cut -c1-19)
    if [ $rc -eq 0 ] && grep -q 'kernels/exl3r4d.so' "$W/out/plugins.txt"; then
        part PASS "git archive build in $BUILD_IMAGE" "$digest: $(sed 's|.*/radiance_home/||' "$W/out/plugins.txt" | tr '\n' ' ')$(grep -m1 'device rows' "$W/out/configure.log" | sed 's/^-- radiance-exl3: /; /; s/, installs.*//'); $(cat "$W/out/warnings.txt") compiler warnings"
    else
        part FAIL "git archive build in $BUILD_IMAGE" "rc=$rc $(grep -m3 -E 'error|Error' "$W/out/"*.log | tr '\n' ' ' | cut -c1-300)"
    fi
fi
finish "committed tree: machine grep, machine constants, offline build in the build stage"
