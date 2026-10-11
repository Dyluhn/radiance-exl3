#!/bin/bash
# kld_score.sh -- score one container against a KL reference record: radiance --kld-ref in the
# runtime image, foreground, with a quiet-host preflight before and the amdgpu kernel-log watchdog
# after. Run it under the project's GPU lock:  flock <gpu.lock> tests/kld/kld_score.sh ...
#
# USAGE: tests/kld/kld_score.sh REF_DIR OUT.json [extra engine flags...]
#   writes OUT.json (the report), OUT.json.rows ([positions, 4] f32: kl, nll_cand, nll_ref, top1),
#   OUT.json.log (the engine's whole log) and OUT.json.cmd (provenance: UTC, image digest, model
#   size, the preflight readings, the full command, the watchdog verdict). Refuses to overwrite any.
#   Extra flags go after the defaults; radiance takes the last value of a repeated flag.
#
# Env (every machine fact is here, none in the file):
#   KLD_MODEL     required: host path of the .rad container (its directory is mounted read-only)
#   KLD_IMAGE     default stilldeadcode/radiance:<tag in the repo's RADIANCE_VERSION>
#   KLD_FLAGS     default below
#   KLD_VRAM_MIB  default 1024: a discrete card holding more VRAM than this fails the preflight
#                 (one card may drive a display: a few hundred MiB is normal there)
#
# KLD_FLAGS' default is the author's production command at radiance 1.3.0
# (deploy/compose/flashnext.yaml) with what the KL mode or 1.3.0 refuses taken out:
#   --num-speculative-tokens 3   refused in KL mode (engine_bringup.cpp:900-905: the mode scores the
#                                trunk's rows and drafts nothing); auto -> 0 there
#   --expert-vs-cache-ratio 0.82 in the HF README's command, removed in radiance 1.1.0 (unknown flag)
#   --prefix-cache-*             forced off in KL mode (engine_bringup.cpp:907)
#   --host/--port/--api-key      the KL mode does not serve
set -euo pipefail

die() { printf 'kld_score: FAIL: %s\n' "$*" >&2; exit 1; }

[ "$#" -ge 2 ] || die "usage: tests/kld/kld_score.sh REF_DIR OUT.json [extra engine flags...]"
ref=$(realpath -e -- "$1") || die "no such reference dir: $1"
out=$(realpath -m -- "$2")
shift 2
repo=$(cd "$(dirname "$0")/../.." && pwd)
: "${KLD_MODEL:?set KLD_MODEL to the host path of the .rad container}"
: "${KLD_IMAGE:=stilldeadcode/radiance:$(cut -d' ' -f1 "$repo/RADIANCE_VERSION")}"
: "${KLD_FLAGS:=--tp 2 --tp-wire wht6 --max-num-seqs 8 --max-model-len 200000 --placement expert_tiered --host-pool-mib 12288 --gpu-headroom-mib 96 --kv-cache-dtype fp8 --max-num-batched-tokens 2048}"
: "${KLD_VRAM_MIB:=1024}"
[ -f "$KLD_MODEL" ] || die "KLD_MODEL is not a file: $KLD_MODEL"
[ -f "$ref/kld.json" ] || die "not a finished reference (no kld.json): $ref"
for f in "$out" "$out.rows" "$out.log" "$out.cmd"; do [ ! -e "$f" ] || die "refusing to overwrite $f"; done
mkdir -p "$(dirname "$out")"
digest=$(docker image inspect --format '{{.Id}}' "$KLD_IMAGE") || die "no image $KLD_IMAGE"

# ---- preflight: no other GPU client, no inference container, cards near idle, clean kernel log
kfd=$(ls /sys/class/kfd/kfd/proc/ 2>/dev/null | tr '\n' ' ')
[ -z "$kfd" ] || die "KFD clients present (pids: $kfd) -- another process holds a GPU"
running=$(docker ps --format '{{.Names}} {{.Image}}')
[ -z "$running" ] || die "containers running: $running"
cards=""
for dev in /sys/class/drm/card*/device; do
    [ -r "$dev/mem_info_vram_total" ] || continue
    [ "$(cat "$dev/mem_info_vram_total")" -gt $((8 << 30)) ] || continue      # discrete cards only
    used=$(( $(cat "$dev/mem_info_vram_used") >> 20 ))
    cards="$cards $(basename "$(readlink -f "$dev")")=${used}MiB"
    [ "$used" -le "$KLD_VRAM_MIB" ] || die "$(basename "$(readlink -f "$dev")") holds ${used} MiB of VRAM (> KLD_VRAM_MIB $KLD_VRAM_MIB)"
done
klog_pat='amdgpu.*(MES|SMU|ring|timeout|reset|page fault)'
if journalctl -k --since -30min --no-pager | grep -iE "$klog_pat" >/dev/null; then
    die "amdgpu errors in the kernel log in the last 30 min: $(journalctl -k --since -30min --no-pager | grep -iE "$klog_pat" | tail -3)"
fi

start=$(date '+%Y-%m-%d %H:%M:%S')
# shellcheck disable=SC2086  # KLD_FLAGS is one flag or value per word by construction
cmd=(docker run --rm --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined
     --security-opt label=disable --init --ulimit memlock=-1
     -e RADIANCE_HOME=/opt/radiance/share/radiance
     -v "$(dirname "$(realpath "$KLD_MODEL")")":/models:ro -v "$ref":/data/ref:ro -v "$(dirname "$out")":/data/out
     "$KLD_IMAGE" --model "/models/$(basename "$KLD_MODEL")" $KLD_FLAGS "$@"
     --kld-ref /data/ref --kld-out "/data/out/$(basename "$out")")
{
    printf '# kld_score %s UTC (host local %s)\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$start"
    printf '# image %s %s\n# model %s (%s B)\n# ref %s\n' "$KLD_IMAGE" "$digest" "$KLD_MODEL" "$(stat -c %s "$KLD_MODEL")" "$ref"
    printf '# preflight: no KFD clients, no containers, VRAM%s, kernel log clean (30 min)\n' "$cards"
    printf '%q ' "${cmd[@]}"; printf '\n'
} > "$out.cmd"

t0=$(date +%s)
status=0
"${cmd[@]}" > "$out.log" 2>&1 || status=$?
printf '# exit %s after %s s\n' "$status" "$(( $(date +%s) - t0 ))" >> "$out.cmd"

hits=$(journalctl -k --since "$start" --no-pager | grep -iE "$klog_pat" || true)
if [ -n "$hits" ]; then
    printf '# WATCHDOG: amdgpu kernel errors during the run -- numbers INVALID:\n%s\n' "$hits" >> "$out.cmd"
    die "amdgpu kernel errors during the run; numbers invalid: $hits"
fi
printf '# watchdog: kernel log clean since %s\n' "$start" >> "$out.cmd"
[ "$status" -eq 0 ] || { tail -20 "$out.log" >&2; die "the engine exited $status (log: $out.log)"; }
[ -f "$out" ] && [ -f "$out.rows" ] || die "the engine exited 0 but wrote no report (log: $out.log)"
printf '# rows sha256 %s\n' "$(sha256sum "$out.rows" | cut -d' ' -f1)" >> "$out.cmd"
grep -E 'kld: (KLD mean|top-1)' "$out.log" || true
printf 'kld_score: OK %s (rows sha256 %s)\n' "$out" "$(sha256sum "$out.rows" | cut -c1-16)"
