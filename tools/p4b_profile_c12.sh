#!/bin/sh
# p4b_profile_c12.sh <outdir> -- the per-op engine profile at C=1 and C=2 (tools/p6_profile.sh): base vs the v14 EXL3
# home, radiance 1.4.0, dpass.sh's requirement config (tp2 production flags + spec 0 + headroom 3072, EXL3 pool 11793).
#   flock <PF>/gpu.lock sh tools/p4b_profile_c12.sh <outdir>
set -eu
out=$1
here=$(cd "$(dirname "$0")" && pwd)
export PROFILE_IMAGE=stilldeadcode/radiance:1.4.0 PROFILE_LEVELS="1 2" PROFILE_MIN_MEM_GIB=24 PROFILE_MEM_ABORT_GIB=6
F="--tp 2 --tp-wire wht6 --max-num-seqs 8 --max-model-len 200000 --placement expert_tiered --host-pool-mib 12288
   --gpu-headroom-mib 96 --kv-cache-dtype fp8 --prefix-cache-host-mib 4096 --prefix-cache-disk-mib 131072
   --num-speculative-tokens 3 --max-num-batched-tokens 2048 --num-speculative-tokens 0 --gpu-headroom-mib 3072"
PROFILE_FLAGS="$F" bash "$here/p6_profile.sh" /var/mnt/qwen-storage/models/text/rad/qwen3.8-next-flash-fp8-iq4r-moe.rad \
    base-c12 "$out"
PROFILE_FLAGS="$F --host-pool-mib 11793" bash "$here/p6_profile.sh" \
    /var/home/dylan/models-boot/radiance-exl3/qwen3.8-flash-next-exl3.rad v14-c12 "$out" \
    /var/home/dylan/projects/inference/radiance-exl3-p6-real/build/v14-home
python3 "$here/p6_profile_diff.py" "$out/base-c12" "$out/v14-c12" > "$out/profile-base-v14-c12.md"
python3 "$here/p6_profile_diff.py" "$out/base-c12" "$out/v14-c12" --robust > "$out/profile-base-v14-c12-robust.md"
echo "done: $out/profile-base-v14-c12*.md"
