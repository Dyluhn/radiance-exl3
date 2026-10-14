#!/bin/sh
# build_variant.sh -- one reading build of exl3r4d (an ablation or a lever arm), in the radiance build image, into
# build-var/<name>/ and build-var/<name>.so. Never installed, never shipped: the defines are the lab notes' toggles.
#   tests/gpu/build_variant.sh <name> "<HIP defines>" "<C defines>" ["<extra cmake -D arguments>"]
# EXL3_PREFIX=<host dir>: build against that radiance install (mounted read-only at /stage-prefix) instead of the
# image's own /stage/opt/radiance -- e.g. a newer release's prefix, so the variant's ABI matches that engine.
set -eu
name=$1 hip=${2:-} c=${3:-} extra=${4:-}
repo=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$repo/build-var"
prefix=/stage/opt/radiance mnt=
if [ -n "${EXL3_PREFIX:-}" ]; then prefix=/stage-prefix mnt="-v ${EXL3_PREFIX}:/stage-prefix:ro"; fi
# shellcheck disable=SC2086  # $mnt is empty or one -v pair
docker run --rm --security-opt label=disable -e EXL3_JOBS="${EXL3_JOBS:-4}" -v "$repo":/work $mnt -w /work "${EXL3_IMAGE:-radiance-build:1.3.0}" sh -euc "
    cmake -S . -B build-var/$name -DCMAKE_PREFIX_PATH=$prefix -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_HIP_FLAGS='$hip' -DCMAKE_C_FLAGS='$c' $extra > build-var/$name.configure.log 2>&1
    nice -n 19 cmake --build build-var/$name --target exl3r4d -j ${EXL3_JOBS:-4} > build-var/$name.build.log 2>&1 || { tail -20 build-var/$name.build.log; exit 1; }
    cp build-var/$name/radiance_home/kernels/exl3r4d.so build-var/$name.so
    sha256sum build-var/$name.so"
