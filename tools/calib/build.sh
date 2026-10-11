#!/bin/sh
# build.sh -- build the calibration instruments inside the radiance Docker build stage (hipcc lives only
# there). Output: tools/calib/build/. No GPU is touched.
#   CALIB_IMAGE   build image (default radiance-build:1.3.0)
#   CALIB_ARCH    offload arch (default gfx1201; the WMMA builtins in isarate.hip are gfx12-only)
set -eu
here=$(cd "$(dirname "$0")" && pwd)
image=${CALIB_IMAGE:-radiance-build:1.3.0}
arch=${CALIB_ARCH:-gfx1201}
mkdir -p "$here/build"
docker run --rm --security-opt label=disable -v "$here:/calib" -w /calib "$image" sh -euc "
    hipcc --version | head -1
    for t in devinfo stream sweep isarate graphrate; do
        hipcc -O3 -std=c++17 --offload-arch=$arch \$t.hip -o build/\$t
    done
    # The VALU reference arm compiles to v_dual_mul_f32 (two multiplies an instruction). This build keeps
    # them single so the 'slots' unit can be read both ways (the author's compiler may not have paired).
    hipcc -O3 -std=c++17 --offload-arch=$arch -mllvm -amdgpu-enable-vopd=0 isarate.hip -o build/isarate_novopd
    sha256sum build/devinfo build/stream build/sweep build/isarate build/isarate_novopd build/graphrate"
