#!/bin/sh
# p4b_profile_job.sh <outdir> -- the orchestrator's per-op engine profile of three homes on radiance 1.4.0, decode at
# C=1, 4, 8 (tools/p6_profile.sh): base (the fp8 / w4 container, no plugin home), v9 (EXL3, GD off) and v12 (EXL3, GD
# from 6 tokens). The flags are job19's series flags (speed/base19-m1, v12-m1b provenance); the EXL3 arms take
# --host-pool-mib 11793 as the series does (equal VRAM). Run under the GPU lock, both cards free:
#   flock <PF>/gpu.lock sh tools/p4b_profile_job.sh <outdir>
# then: python3 tools/p6_profile_diff.py <outdir>/base-prof <outdir>/v9-prof <outdir>/v12-prof [--robust]
set -eu
out=$1
P6=/var/home/dylan/projects/inference/radiance-exl3-p6-real
here=$(cd "$(dirname "$0")" && pwd)
export PROFILE_IMAGE=stilldeadcode/radiance:1.4.0 PROFILE_LEVELS="1 4 8" PROFILE_MIN_MEM_GIB=24 PROFILE_MEM_ABORT_GIB=6
F="--tp 2 --tp-wire wht6 --max-num-seqs 8 --max-model-len 200000 --placement expert_tiered --host-pool-mib 12288
   --gpu-headroom-mib 96 --kv-cache-dtype fp8 --prefix-cache-host-mib 4096 --prefix-cache-disk-mib 131072
   --num-speculative-tokens 3 --max-num-batched-tokens 2048 --num-speculative-tokens 0 --gpu-headroom-mib 2048"
PROFILE_FLAGS="$F" bash "$here/p6_profile.sh" /var/mnt/qwen-storage/models/text/rad/qwen3.8-next-flash-fp8-iq4r-moe.rad \
    base-prof "$out"
PROFILE_FLAGS="$F --host-pool-mib 11793" bash "$here/p6_profile.sh" \
    /var/home/dylan/models-boot/radiance-exl3/qwen3.8-flash-next-exl3.rad v9-prof "$out" "$P6/build/v9-home"
PROFILE_FLAGS="$F --host-pool-mib 11793" bash "$here/p6_profile.sh" \
    /var/home/dylan/models-boot/radiance-exl3/qwen3.8-flash-next-exl3.rad v12-prof "$out" "$P6/build/v12-home"
python3 "$here/p6_profile_diff.py" "$out/base-prof" "$out/v9-prof" "$out/v12-prof" > "$out/profile-3way.md"
python3 "$here/p6_profile_diff.py" "$out/v9-prof" "$out/v12-prof" > "$out/profile-v9-v12.md"
python3 "$here/p6_profile_diff.py" "$out/v9-prof" "$out/v12-prof" --robust > "$out/profile-v9-v12-robust.md"
echo "done: $out/profile-*.md"
