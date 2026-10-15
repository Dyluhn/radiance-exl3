#!/bin/sh
# kernarg_check.sh <exl3r4d.so> [limit] -- every kernel in the library's gfx code object has an argument segment
# (explicit arguments + HIP's hidden ones) of at most `limit` bytes (default 768: radiance 1.3.0's kernarg slot --
# "cannot dispatch a kernel: ... its kernel argument segment is 912 bytes and the backend's slot holds 768", the
# fused gate_up_down launch, runs/p4b-s25b). hipLaunchKernel has no such limit, so perfmoe and the harness cannot see
# it; the engine refuses at dispatch. Needs the ROCm llvm tools (the build image). Exit 0 / 1, 77 without the tools.
# AND (interface v7) the four-class fused gate_up_down kernels (exl3r4d_gd_mod4.hip, argument type Exl3r4dFused4) take
# NO hidden argument: their 704 explicit bytes fit the slot only because nothing in them reads blockDim / gridDim or any
# other implicit argument -- the compiler lists every hidden argument a kernel reads in its metadata, so one listed
# there is a read from the hidden segment and fails this check.
set -u
so=$1 limit=${2:-768}
B=${ROCM_LLVM:-/opt/rocm/llvm/bin}
[ -x "$B/llvm-readelf" ] && [ -x "$B/clang-offload-bundler" ] || { echo "SKIP: no $B/llvm-readelf"; exit 77; }
t=$(mktemp -d) || exit 1
trap 'rm -rf "$t"' EXIT
objcopy -O binary --only-section=.hip_fatbin "$so" "$t/fb.bin" || { echo "FAIL: no .hip_fatbin in $so"; exit 1; }
# ONE BUNDLE A HIP TRANSLATION UNIT, back to back in the section (exl3r4d has five: the two kernel files at two class
# counts, and the fused section at four); the bundler reads the first it is handed, so each is cut out at its magic.
grep -obUa "__CLANG_OFFLOAD_BUNDLE__" "$t/fb.bin" | cut -d: -f1 > "$t/at.txt"
[ -s "$t/at.txt" ] || { echo "FAIL: no offload bundle in $so"; exit 1; }
size=$(wc -c < "$t/fb.bin")
i=0
while read -r at; do
    i=$((i + 1))
    end=$(sed -n "$((i + 1))p" "$t/at.txt"); end=${end:-$size}
    tail -c +"$((at + 1))" "$t/fb.bin" | head -c "$((end - at))" > "$t/b$i.bin"
    "$B/clang-offload-bundler" --list --type=o --input="$t/b$i.bin" | grep amdgcn | while read -r tgt; do
        "$B/clang-offload-bundler" --unbundle --type=o --input="$t/b$i.bin" --targets="$tgt" --output="$t/k.co" &&
        "$B/llvm-readelf" --notes "$t/k.co"
    done
done < "$t/at.txt" > "$t/notes.txt"
echo "kernarg_check: $(wc -l < "$t/at.txt") bundle(s)"
awk -v lim="$limit" '/^ *- \.args:/ {hid=0} /\.value_kind: *hidden_/ {hid++} /\.kernarg_segment_size:/ {sz=$2}
     /^ *\.name: *_Z/ {n++; if (sz > lim) {bad++; print "FAIL kernarg " sz " > " lim ": " $2} else if (sz > max) max=sz
                       if ($2 ~ /13Exl3r4dFused4/) {gd4++; if (hid) {bad++; print "FAIL " hid " hidden argument(s): " $2}}}
     END {if (!n) {print "FAIL: no kernels found"; exit 1} if (!gd4) {print "FAIL: no four-class fused kernel"; exit 1}
          if (bad) exit 1; print "kernarg_check: " n " kernels, largest " max " <= " lim "; " gd4 " four-class fused kernels, no hidden argument"}' "$t/notes.txt"
