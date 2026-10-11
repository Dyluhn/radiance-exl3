#!/bin/sh
# R10 -- host-only: `-DRAD_WITH_HIP=OFF` build and `ctest -LE 'gpu|oot'` green, from `git archive HEAD`, inside the
# radiance Docker image at the pin, offline.
#
# WHICH IMAGE. radiance's Dockerfile `test` stage is `FROM build` plus one RUN (`ctest` of radiance's own tree and a
# marker file), so its filesystem is the `build` stage's: the same compiler, CMake, Python and installed radiance
# (/stage/opt/radiance, /src = the release's source). BUILD_IMAGE (default radiance-build:<pin>) IS that build stage;
# no separately tagged test-stage image is needed for an out-of-tree project, and this gate says which it used.
# The image's python3 has no numpy and no pytest, so the Python cross-checks SKIP there by name (listed).
#   R10_NUMPY=1   a second run in a throwaway container of the same image with the distribution's python3-numpy and
#                 python3-pytest installed first (network for apt only): the Python tests on another numpy/python
#                 than the development box's. With R10_TOKENIZER=<dir> the end-to-end test (label e2e) runs too;
#                 this container is run with seccomp=unconfined because libavx reads the n-gram table through
#                 io_uring, which Docker's default profile refuses (radiance docs/DOCKER.md) -- without it the e2e
#                 test FAILS in its reference recording (measured 2026-10-11; notes/gates.md).
GATE=R10
. "$(dirname "$0")/lib.sh"
command -v docker > /dev/null && docker image inspect "$BUILD_IMAGE" > /dev/null 2>&1 ||
    { part SKIP "build image" "no docker or no image $BUILD_IMAGE"; finish "nothing checked"; }
W=$GATES_WORK/r10
archive_tree "$W/src" || { part FAIL "git archive" "of HEAD into $W/src"; finish "no archive"; }
digest=$(docker image inspect --format '{{.Id}}' "$BUILD_IMAGE" | cut -c1-19)
BUILD='cmake -S /exl3 -B /tmp/b -G Ninja -DCMAKE_PREFIX_PATH=/stage/opt/radiance -DRAD_WITH_HIP=OFF -DRADIANCE_SRC=/src $EXTRA > /out/configure.log 2>&1 &&
       cmake --build /tmp/b > /out/build.log 2>&1 || exit 3
       ctest --test-dir /tmp/b -LE "gpu|oot" --output-on-failure > /out/ctest.log 2>&1'

verdict() {   # verdict <part> <out dir> <docker rc>
    log=$2/ctest.log
    sum=$(grep -E 'tests passed|tests failed' "$log" | tail -1)
    skipped=$(grep -E '\*\*\*Skipped' "$log" | sed -E 's/.*Test +#[0-9]+: ([^ ]+).*/\1/' | tr '\n' ' ')
    if [ "$3" -eq 3 ]; then part FAIL "$1" "build failed: $(grep -m3 -E 'error|Error' "$2"/*.log | tr '\n' ' ' | cut -c1-300)"
    elif [ "$3" -eq 0 ]; then part PASS "$1" "$digest: $sum; skipped: ${skipped:-none}"
    else part FAIL "$1" "$sum: $(grep -E '\(Failed\)|\(Timeout\)' "$log" | sed 's/^ *//' | tr '\n' ' ' | cut -c1-300)"; fi
    grep -E '\*\*\*Skipped' "$log" | sed 's/^/      /'
}

mkdir -p "$W/out"
in_image -v "$W/src":/exl3:ro -v "$W/out":/out -e EXTRA= "$BUILD_IMAGE" sh -c "$BUILD"
verdict "host-only build + ctest -LE 'gpu|oot' in $BUILD_IMAGE (offline, source read-only)" "$W/out" $?

if [ "${R10_NUMPY:-0}" = 1 ]; then
    mkdir -p "$W/out-numpy"
    tok=""; extra=""
    if [ -f "${R10_TOKENIZER:-}/tokenizer.json" ]; then tok="-v $R10_TOKENIZER:/tok:ro"; extra="-DQWEN4EXP_EXL3_TOKENIZER=/tok"; fi
    # shellcheck disable=SC2086
    ${GATES_LOCK:+flock "$GATES_LOCK"} docker run --rm --security-opt label=disable --security-opt seccomp=unconfined --cpus "$GATES_JOBS" \
        --memory "${GATES_MEMORY:-8g}" -v "$W/src":/exl3:ro -v "$W/out-numpy":/out $tok -e EXTRA="$extra" "$BUILD_IMAGE" sh -c \
        "apt-get update > /out/apt.log 2>&1 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-numpy python3-pytest >> /out/apt.log 2>&1 || exit 4
         python3 -c 'import numpy, pytest, sys; print(sys.version.split()[0], numpy.__version__, pytest.__version__)' > /out/python.txt; $BUILD"
    rc=$?
    if [ $rc -eq 4 ]; then part SKIP "with the distribution's numpy and pytest" "apt-get failed (network?): $W/out-numpy/apt.log"
    else verdict "with the distribution's numpy and pytest (python, numpy, pytest = $(cat "$W/out-numpy/python.txt" 2>/dev/null))" "$W/out-numpy" $rc; fi
fi
finish "host-only build and host suite from git archive in $BUILD_IMAGE"
