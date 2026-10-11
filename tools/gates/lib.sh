# lib.sh -- sourced by tools/gates/R*.sh. Not run on its own.
#
# A gate prints one line per PART it checked ("  PASS|FAIL|SKIP <part>: <number or why>") and ends with one
# summary line, "R<n> PASS|FAIL|SKIP|PARTIAL: ...". Exit 0 PASS (every part passed), 1 FAIL (any part failed),
# 77 SKIP (nothing could be checked) or PARTIAL (some parts passed, the rest SKIPPED) -- a requirement with a part
# unchecked is not met, so PARTIAL is never exit 0. A part that checked nothing is SKIP, never PASS.
#
# Inputs every gate may read (machine facts come from here, never from the scripts):
#   RADIANCE_PREFIX     an install of the radiance release RADIANCE_VERSION pins, host build (bin/, include/,
#                       lib*/cmake/radiance, share/radiance). Default: the prefix of `radiance` on PATH.
#   RADIANCE_SRC        the radiance source tree at that release (scripts/, tests/oot_plugins.sh, docs)
#   GATES_WORK          scratch (builds, fixtures, logs); default <repo>/build-gates (git-ignored)
#   GATES_BUILD         a host-only build of this repository against RADIANCE_PREFIX; default $GATES_WORK/host,
#                       configured and built on first use
#   BUILD_IMAGE         the radiance Docker `build` stage at the pin; default radiance-build:<pinned version>
#   GATES_LOCK          a lock file every `docker run` of a gate is taken under (flock), so a gate's CPU-only
#                       container never lands inside another job's GPU session; unset: no lock
#   GATES_JOBS          parallel build jobs (default 4)
#   EXL3_CONTAINER      a running container with the card visible and this repository at /work
#                       (tests/gpu/run_gpu.sh sets it for its session): device parts run there, else SKIP
#   GATES_DEVICE_BUILD  a device build of this repository made in BUILD_IMAGE, relative to the repository (it is
#                       /work/<that> inside EXL3_CONTAINER); default build-gpu (tests/gpu/run_gpu.sh's)
set -u

REPO=$(cd "$(dirname "$0")/../.." && pwd)
PIN_VERSION=$(cut -d' ' -f1 "$REPO/RADIANCE_VERSION" 2>/dev/null)
: "${GATES_WORK:=$REPO/build-gates}"
: "${GATES_JOBS:=4}"
: "${BUILD_IMAGE:=radiance-build:$PIN_VERSION}"
: "${GATES_DEVICE_BUILD:=build-gpu}"
if [ -z "${RADIANCE_PREFIX:-}" ] && command -v radiance > /dev/null 2>&1; then
    RADIANCE_PREFIX=$(cd "$(dirname "$(command -v radiance)")/.." && pwd)
fi
: "${RADIANCE_PREFIX:=}"
mkdir -p "$GATES_WORK/logs"
_n_pass=0 _n_fail=0 _n_skip=0

part() {   # part PASS|FAIL|SKIP <name> <detail>
    case $1 in PASS) _n_pass=$((_n_pass + 1)) ;; FAIL) _n_fail=$((_n_fail + 1)) ;; *) _n_skip=$((_n_skip + 1)) ;; esac
    printf '  %-4s %s: %s\n' "$1" "$2" "$3"
}

finish() {   # finish <one-line summary of the numbers>
    local state=SKIP rc=77
    if [ $_n_fail -gt 0 ]; then state=FAIL rc=1
    elif [ $_n_pass -gt 0 ] && [ $_n_skip -eq 0 ]; then state=PASS rc=0
    elif [ $_n_pass -gt 0 ]; then state=PARTIAL
    fi
    printf '%s %s: %s [%d passed, %d failed, %d skipped parts; %s %s%s]\n' "$GATE" "$state" "$1" $_n_pass $_n_fail \
        $_n_skip "$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)" "$(date -u +%FT%TZ)" \
        "$(git -C "$REPO" rev-parse --git-dir > /dev/null 2>&1 && ! git -C "$REPO" diff --quiet HEAD && echo ' +dirty')"
    exit $rc
}

need_prefix() {   # true when RADIANCE_PREFIX is an install of the pinned release; else says why
    [ -n "$RADIANCE_PREFIX" ] && [ -x "$RADIANCE_PREFIX/bin/radiance" ] || {
        echo "RADIANCE_PREFIX has no bin/radiance ('$RADIANCE_PREFIX'): set it to an install of radiance $PIN_VERSION"
        return 1; }
}

host_build() {   # host_build -> the host-only build of this repository (GATES_BUILD), built incrementally
    : "${GATES_BUILD:=$GATES_WORK/host}"
    local log=$GATES_WORK/logs/host-build.log
    if [ ! -f "$GATES_BUILD/CMakeCache.txt" ]; then
        cmake -S "$REPO" -B "$GATES_BUILD" -DCMAKE_PREFIX_PATH="$RADIANCE_PREFIX" -DRAD_WITH_HIP=OFF \
            ${RADIANCE_SRC:+-DRADIANCE_SRC="$RADIANCE_SRC"} > "$log" 2>&1 || { tail -5 "$log"; return 1; }
    fi
    nice -n 19 cmake --build "$GATES_BUILD" -j "$GATES_JOBS" >> "$log" 2>&1 || { grep -m5 -E 'error' "$log"; return 1; }
}

in_image() {   # in_image <docker run args...> -- the gate's CPU-only container (no card, no network)
    ${GATES_LOCK:+flock "$GATES_LOCK"} docker run --rm --network none --security-opt label=disable \
        --cpus "$GATES_JOBS" --memory "${GATES_MEMORY:-8g}" "$@"
}

archive_tree() {   # archive_tree <dir> -- `git archive HEAD` of this repository into <dir> (fresh)
    rm -rf "$1" && mkdir -p "$1" && git -C "$REPO" archive HEAD | tar -x -C "$1"
}

device_session() {   # true when the device parts can run: exl3r4d built and a GPU session container up
    [ -f "$REPO/$GATES_DEVICE_BUILD/radiance_home/kernels/exl3r4d.so" ] || {
        echo "no device build of exl3r4d at $GATES_DEVICE_BUILD/radiance_home/kernels/exl3r4d.so"; return 1; }
    [ -n "${EXL3_CONTAINER:-}" ] || {
        echo "no GPU session: run under tests/gpu/run_gpu.sh <pci> <lock> <log dir> -- $0 (it sets EXL3_CONTAINER)"
        return 1; }
    case $GATES_WORK in "$REPO"/*) ;; *) echo "GATES_WORK must be inside the repository (it is /work in the session)"; return 1 ;; esac
}

ref_host() {   # ref_host <suite> <out file> -- P2's harness (tests/ref) on exl3ref's host rows; exit status of it
    python3 "$REPO/tests/ref/test_exl3ref.py" --harness "$GATES_BUILD/kernels/exl3ref/exl3ref_run" \
        --plugin "$GATES_BUILD/radiance_home/kernels/exl3ref.so" --golden "$REPO/common/tests/data/exl3_golden.npz" \
        --work "$GATES_WORK/ref-host" --probes "$GATES_BUILD/radiance_home/test/layout_probe.so" \
        "$GATES_BUILD/radiance_home/test/layout_probe_badinv.so" --suite "$1" > "$2" 2>&1
}

device_suite() {   # device_suite <device|real> <out file> -- P4's driver (tests/gpu) on exl3r4d in the GPU session.
    # One run per suite and session container: R5, R7, R12 and R13 read the same output.
    local cache="$GATES_WORK/device/$1.$EXL3_CONTAINER.out"
    mkdir -p "$GATES_WORK/device"
    if [ ! -s "$cache" ]; then
        python3 "$REPO/tests/gpu/test_exl3r4d.py" --container "$EXL3_CONTAINER" \
            --harness "$REPO/$GATES_DEVICE_BUILD/kernels/exl3ref/exl3ref_run" \
            --plugin "$REPO/$GATES_DEVICE_BUILD/radiance_home/kernels/exl3r4d.so" --work "$GATES_WORK/device" \
            --suite "$1" ${EXL3_REAL_EXTRACT:+--extract "$EXL3_REAL_EXTRACT"} > "$cache.tmp" 2>&1
        echo "exit $?" >> "$cache.tmp"
        mv "$cache.tmp" "$cache"
    fi
    cp "$cache" "$2"
}

tally() {   # tally <harness output> -> its "<suite>: N passed, F failed, S skipped" line
    grep -E '^[a-z-]+: [0-9]+ passed, [0-9]+ failed, [0-9]+ skipped' "$1" | tail -1
}

clean_tally() {   # clean_tally <harness output> -> true when it checked something and nothing failed or skipped
    tally "$1" | grep -qE ': [1-9][0-9]* passed, 0 failed, 0 skipped'
}

speed_summaries() {   # -> $BASE_JSON, $EXL3_JSON: summarize.py --json of each arm's run_pass.sh dirs, or given ones.
    # SPEED_BASE_PASSES / SPEED_EXL3_PASSES: the pass dirs (space-separated), or SPEED_BASE_SUMMARY /
    # SPEED_EXL3_SUMMARY: summarize.py --json files already made. False, saying why, when an arm has neither.
    mkdir -p "$GATES_WORK/speed"
    for arm in EXL3 BASE; do   # EXL3 first: when both are missing, its absence is the one to name
        eval "dirs=\${SPEED_${arm}_PASSES:-} js=\${SPEED_${arm}_SUMMARY:-}"
        out=$GATES_WORK/speed/$(echo $arm | tr A-Z a-z).json
        if [ -n "$js" ]; then cp "$js" "$out"
        elif [ -n "$dirs" ]; then
            # shellcheck disable=SC2086
            python3 "$REPO/tools/gates/speed/summarize.py" $dirs --json "$out" > "$out.md" 2>&1 || { echo "summarize.py failed: $(tail -1 "$out.md")"; return 1; }
        else echo "no ${arm} passes: set SPEED_${arm}_PASSES (run_pass.sh dirs) or SPEED_${arm}_SUMMARY"; return 1; fi
        eval "${arm}_JSON=\$out"
    done
}

speed_part() {   # speed_part <metric> lower|higher [--beside ...] -- one part from speed_verdict.py
    r=$(python3 "$REPO/tools/gates/speed_verdict.py" "$BASE_JSON" "$EXL3_JSON" "$@" ${SPEED_MIN_PASSES:+--min-passes "$SPEED_MIN_PASSES"}) ||
        r="FAIL|$1|speed_verdict.py failed"
    part "${r%%|*}" "$(echo "$r" | cut -d'|' -f2)" "$(echo "$r" | cut -d'|' -f3-)"
}
