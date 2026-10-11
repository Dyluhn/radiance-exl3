#!/bin/sh
# isa.sh -- static reading of one exl3r4d kernel (playbook D.0 R1): pull the gfx1201 code object out of a plugin .so,
# disassemble the kernel whose MANGLED name contains PAT, and classify its working loop with libr4d's isa/loop.sh
# (the smallest backward branch holding every ANCHOR instruction; counts by class and by opcode).
#   tests/gpu/isa.sh <plugin.so> <mangled-name substring> [anchor (v_wmma)]     host; runs in radiance-build:1.3.0
# e.g. the shipped gate_up decode GEMM:  tests/gpu/isa.sh build-var/base.so ILi4ELi2ELi2ELb0ELb0ELi0E
set -eu
docker run --rm --security-opt label=disable -v "$PWD":/work -w /work "${EXL3_IMAGE:-radiance-build:1.3.0}" sh -c '
B=/opt/rocm/llvm/bin O=/tmp/isa; mkdir -p $O
objcopy -O binary --only-section=.hip_fatbin "$1" $O/fb.bin
$B/clang-offload-bundler --unbundle --type=o --input=$O/fb.bin --targets=hipv4-amdgcn-amd-amdhsa--gfx1201 --output=$O/k.co
$B/llvm-objdump -d --symbolize-operands --no-show-raw-insn --mcpu=gfx1201 $O/k.co > $O/all.s
awk -v pat="$2" "/^[0-9a-f]+ <.*>:\$/ { if (\$2 !~ /^<L[0-9]+>:\$/) ink = (index(\$0, pat) > 0) } ink { print }" \
    $O/all.s > $O/one.s
[ -s $O/one.s ] || { echo "isa.sh: no kernel matches $2" >&2; exit 1; }
ISA_OUT=$O sh /src/libr4d/isa/loop.sh "$3"
grep -nE "s_wait_loadcnt|s_wait_dscnt|s_wait_kmcnt" $O/loop.s | sed -E "s/^([0-9]+):\s+/  wait at +\1: /"' \
    sh "$1" "$2" "${3:-v_wmma}"
