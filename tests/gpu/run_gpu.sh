#!/bin/bash
# run_gpu.sh -- one GPU session of exl3r4d's checks or measurements, on ONE named card, with the hygiene every
# sample needs. Nothing about the machine is in here: the card, the lock and the image are arguments.
#
#   tests/gpu/run_gpu.sh <pci address> <lock file> <log dir> -- <command...>
#
# <command> runs ON THE HOST (cwd: this repository) while one container of the build image (EXL3_IMAGE, default
# radiance-build:1.3.0) is up for the session with this repository mounted at /work and the card made its device 0
# (HIP_VISIBLE_DEVICES=<its HIP index>); its name is in $EXL3_CONTAINER. Run device work in it with
# `docker exec -w /work "$EXL3_CONTAINER" ...` (tests/gpu/test_exl3r4d.py --container does, for the harness), so
# host-side tools (numpy) stay on the host. Before it, under the lock
# (flock <lock file>, held for the whole session and nothing else):
#   - precheck: no KFD client (/sys/class/kfd/kfd/proc empty), no running container, the card's VRAM in use under
#     EXL3_VRAM_IDLE_MB (256) -- a failure stops the session (exit 3) before anything runs;
#   - the card's HIP index looked up and PROVEN from inside the container: build/devcheck must list exactly one
#     device, at <pci address> (exit 4 otherwise);
#   - the clock sampler (tools/calib/clocks.py: the LACT socket, else sysfs DPM levels) for the session.
# After it: the amdgpu kernel-log watchdog since the session started -- any MES/SMU/ring/timeout/reset/page-fault
# line invalidates everything the session measured (exit 3) -- and a clock summary (median sclk/mclk) in the log.
# EXL3_DEVCHECK names the devcheck binary inside /work (default build-gpu/kernels/exl3r4d/tests-gpu/exl3r4d_devcheck).
set -euo pipefail

[ "$#" -ge 5 ] && [ "$4" = "--" ] || { echo "usage: $0 <pci> <lock> <log dir> -- <command...>" >&2; exit 2; }
pci=$(echo "$1" | tr 'A-F' 'a-f'); lock=$2; out=$3; shift 4
repo=$(cd "$(dirname "$0")/../.." && pwd)
image=${EXL3_IMAGE:-radiance-build:1.3.0}
idle_mb=${EXL3_VRAM_IDLE_MB:-256}
devcheck=${EXL3_DEVCHECK:-build-gpu/kernels/exl3r4d/tests-gpu/exl3r4d_devcheck}
dev=/sys/bus/pci/devices/$pci
[ -r "$dev/mem_info_vram_used" ] || { echo "no amdgpu card at $pci" >&2; exit 2; }
mkdir -p "$out"
log=$out/session.log

if [ -z "${EXL3_IN_LOCK:-}" ]; then
    EXL3_IN_LOCK=1 exec flock "$lock" "$0" "$pci" "$lock" "$out" -- "$@"
fi

since=$(date '+%Y-%m-%d %H:%M:%S')
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$log"; }
watchdog() {
    local hits
    hits=$(journalctl -k --since "$since" --no-pager 2>/dev/null | grep -iE 'amdgpu.*(MES|SMU|ring|timeout|reset|page fault)' || true)
    if [ -n "$hits" ]; then
        say "FAIL watchdog: amdgpu kernel-log lines since $since -- every number of this session is invalid:"
        echo "$hits" | tee -a "$log"
        return 3
    fi
    say "watchdog clean since $since"
}
dock() {   # dock <name> <env...> -- <command...>
    local name=$1; shift
    local envs=()
    while [ "$1" != "--" ]; do envs+=(-e "$1"); shift; done; shift
    timeout "${EXL3_TIMEOUT:-1800}" docker run --rm --name "$name" --device /dev/kfd --device /dev/dri \
        --security-opt seccomp=unconfined --security-opt label=disable "${envs[@]}" \
        -v "$repo":/work -w /work "$image" "$@"
}

say "session start: pci=$pci image=$image ($(docker image inspect --format '{{.Id}}' "$image")) head=$(git -C "$repo" rev-parse --short HEAD)$(git -C "$repo" diff --quiet || echo '+dirty')"
watchdog || exit 3
kfd=$(ls /sys/class/kfd/kfd/proc/ | wc -l); ctr=$(docker ps -q | wc -l)
vram=$(( $(cat "$dev/mem_info_vram_used") / 1048576 )); gtt=$(( $(cat "$dev/mem_info_gtt_used") / 1048576 ))
say "precheck kfd_clients=$kfd containers=$ctr vram_used_mb=$vram gtt_used_mb=$gtt loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
if [ "$kfd" -ne 0 ] || [ "$ctr" -ne 0 ] || [ "$vram" -gt "$idle_mb" ]; then
    say "FAIL precheck: card not idle (kfd: $(ls /sys/class/kfd/kfd/proc/ | tr '\n' ' ') containers: $(docker ps --format '{{.Names}}' | tr '\n' ' '))"
    exit 3
fi

# The HIP index: devcheck with every card visible, the line naming <pci>; then proven with only that one visible.
all=$(dock "exl3-dev-$$" -- "$devcheck") || { say "FAIL devcheck: $all"; exit 4; }
idx=$(echo "$all" | sed -n "s/^device \([0-9]*\) pci $pci .*/\1/p" | head -1)
[ -n "$idx" ] || { say "FAIL: HIP lists no device at $pci: $(echo "$all" | tr '\n' ' ')"; exit 4; }
one=$(dock "exl3-dev1-$$" "HIP_VISIBLE_DEVICES=$idx" -- "$devcheck")
echo "$one" | grep -q "^visible 1$" && echo "$one" | grep -q "^device 0 pci $pci " ||
    { say "FAIL: HIP_VISIBLE_DEVICES=$idx does not show exactly $pci: $(echo "$one" | tr '\n' ' ')"; exit 4; }
say "card $pci is HIP index $idx; with HIP_VISIBLE_DEVICES=$idx: $(echo "$one" | tr '\n' ' ')"

clocks=$repo/tools/calib/clocks.py
cpid=
if [ -x "$(command -v python3)" ] && [ -f "$clocks" ]; then
    python3 "$clocks" "$pci" "$out/clocks.jsonl" 0.25 2>>"$log" &
    cpid=$!
fi
export EXL3_CONTAINER="exl3-gpu-$$"
docker run -d --rm --name "$EXL3_CONTAINER" --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
    --security-opt label=disable -e "HIP_VISIBLE_DEVICES=$idx" -v "$repo":/work -w /work "$image" sleep infinity >/dev/null
trap 'docker rm -f "$EXL3_CONTAINER" >/dev/null 2>&1 || true; [ -n "$cpid" ] && kill "$cpid" 2>/dev/null || true' EXIT
inside=$(docker exec "$EXL3_CONTAINER" "$devcheck")
echo "$inside" | grep -q "^visible 1$" && echo "$inside" | grep -q "^device 0 pci $pci " ||
    { say "FAIL: the session container does not show exactly $pci: $(echo "$inside" | tr '\n' ' ')"; exit 4; }
rc=0
say "run (host, container $EXL3_CONTAINER): $*"
(cd "$repo" && timeout "${EXL3_TIMEOUT:-1800}" "$@") > "$out/job.log" 2>&1 || rc=$?
docker rm -f "$EXL3_CONTAINER" >/dev/null 2>&1 || true
[ -n "$cpid" ] && kill "$cpid" 2>/dev/null || true
say "job exit $rc (output: $out/job.log)"
if [ -f "$out/clocks.jsonl" ]; then
    python3 - "$out/clocks.jsonl" <<'EOF' | tee -a "$log"
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
busy = [r for r in rows if r.get("busy", 100) > 10] or rows
med = lambda k: statistics.median([r[k] for r in busy if r.get(k) is not None]) if any(r.get(k) is not None for r in busy) else None
print(f"clocks ({rows[0]['src'] if rows else 'none'}, {len(busy)} busy samples of {len(rows)}): "
      f"sclk median {med('sclk')} MHz, mclk median {med('mclk')} MHz, mclk max {max((r.get('mclk') or 0) for r in rows) if rows else None} MHz")
EOF
fi
watchdog || exit 3
exit "$rc"
