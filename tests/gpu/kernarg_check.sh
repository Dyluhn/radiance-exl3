#!/bin/sh
# kernarg_check.sh <exl3r4d.so> [limit] -- every kernel in the library's gfx code object has an argument segment
# (explicit arguments + HIP's hidden ones) of at most `limit` bytes (default 768: radiance 1.3.0's kernarg slot --
# "cannot dispatch a kernel: ... its kernel argument segment is 912 bytes and the backend's slot holds 768", the
# fused gate_up_down launch, runs/p4b-s25b). hipLaunchKernel has no such limit, so perfmoe and the harness cannot see
# it; the engine refuses at dispatch. Needs the ROCm llvm tools (the build image). Exit 0 / 1, 77 without the tools.
set -u
so=$1 limit=${2:-768}
B=${ROCM_LLVM:-/opt/rocm/llvm/bin}
[ -x "$B/llvm-readelf" ] && [ -x "$B/clang-offload-bundler" ] || { echo "SKIP: no $B/llvm-readelf"; exit 77; }
t=$(mktemp -d) || exit 1
trap 'rm -rf "$t"' EXIT
objcopy -O binary --only-section=.hip_fatbin "$so" "$t/fb.bin" || { echo "FAIL: no .hip_fatbin in $so"; exit 1; }
"$B/clang-offload-bundler" --list --type=o --input="$t/fb.bin" | grep amdgcn | while read -r tgt; do
    "$B/clang-offload-bundler" --unbundle --type=o --input="$t/fb.bin" --targets="$tgt" --output="$t/k.co" &&
    "$B/llvm-readelf" --notes "$t/k.co"
done > "$t/notes.txt"
awk -v lim="$limit" '/\.kernarg_segment_size:/ {sz=$2} /^ *\.name: *_Z/ {n++; if (sz > lim) {bad++; print "FAIL kernarg " sz " > " lim ": " $2} else if (sz > max) max=sz}
     END {if (!n) {print "FAIL: no kernels found"; exit 1} if (bad) exit 1; print "kernarg_check: " n " kernels, largest " max " <= " lim}' "$t/notes.txt"
