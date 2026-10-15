#!/bin/bash
# run_gpu.sh -- one GPU session of exl3r4d's checks or measurements, on ONE named card or a LIST of them (tp2: a
# comma-separated list, e.g. two cards), with the hygiene every sample needs. Nothing about the machine is in here:
# the cards, the lock and the image are arguments.
#
#   tests/gpu/run_gpu.sh <pci address>[,<pci address>...] <lock file> <log dir> -- <command...>
#
# <command> runs ON THE HOST (cwd: this repository) while one container of the build image (EXL3_IMAGE, default
# radiance-build:1.3.0) is up for the session with this repository mounted at /work and the card made its device 0
# (HIP_VISIBLE_DEVICES=<its HIP index>; with a list, every listed card, device i = the i-th listed); its name is in
# $EXL3_CONTAINER. Run device work in it with
# `docker exec -w /work "$EXL3_CONTAINER" ...` (tests/gpu/test_exl3r4d.py --container does, for the harness), so
# host-side tools (numpy) stay on the host. Before it, under the lock
# (flock <lock file>, held for the whole session and nothing else):
#   - precheck: no KFD client (/sys/class/kfd/kfd/proc empty), no running container that can reach a GPU (/dev/kfd or
#     a /dev/dri node mapped in, or privileged; a CPU-only container -- another stream's build -- is logged, not
#     refused), every listed card's VRAM in use under its idle allowance -- EXL3_VRAM_IDLE_MB (256), or for a card
#     driving a display (a connected DRM connector: the desktop holds 0.5-1.3 GB there) EXL3_VRAM_DISPLAY_MB (1536),
#     its baseline logged -- a failure stops the session (exit 3) before anything runs. Any process actually using
#     a GPU is a KFD client whatever started it;
#   - each card's HIP index looked up and PROVEN from inside the container: devcheck must list exactly the listed
#     cards, device i at the i-th address (exit 4 otherwise);
#   - one clock sampler a card (tools/calib/clocks.py: the LACT socket, else sysfs DPM levels) for the session:
#     clocks.jsonl for the first card, clocks-<pci>.jsonl for the others.
# After it: the amdgpu kernel-log watchdog since the session started -- any SMU/ring/timeout/reset/page-fault line
# invalidates everything the session measured (exit 3); MES lines are logged with their time and card, not fatal -- and a clock summary (median sclk/mclk) in the log.
# EXL3_DEVCHECK names the devcheck binary inside /work (default build-gpu/kernels/exl3r4d/tests-gpu/exl3r4d_devcheck).
set -euo pipefail

[ "$#" -ge 5 ] && [ "$4" = "--" ] || { echo "usage: $0 <pci>[,<pci>...] <lock> <log dir> -- <command...>" >&2; exit 2; }
pcis_arg=$(echo "$1" | tr 'A-F' 'a-f'); lock=$2; out=$3; shift 4
IFS=, read -r -a pcis <<< "$pcis_arg"
pci=${pcis[0]}
repo=$(cd "$(dirname "$0")/../.." && pwd)
image=${EXL3_IMAGE:-radiance-build:1.3.0}
idle_mb=${EXL3_VRAM_IDLE_MB:-256}
display_mb=${EXL3_VRAM_DISPLAY_MB:-1536}
devcheck=${EXL3_DEVCHECK:-build-gpu/kernels/exl3r4d/tests-gpu/exl3r4d_devcheck}
for p in "${pcis[@]}"; do
    [ -r "/sys/bus/pci/devices/$p/mem_info_vram_used" ] || { echo "no amdgpu card at $p" >&2; exit 2; }
done
mkdir -p "$out"
log=$out/session.log

if [ -z "${EXL3_IN_LOCK:-}" ]; then
    EXL3_IN_LOCK=1 exec flock "$lock" "$0" "$pcis_arg" "$lock" "$out" -- "$@"
fi

since=$(date '+%Y-%m-%d %H:%M:%S')
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$log"; }
# MES lines (the scheduler firmware's errors) are RECOVERABLE (Dylan, 2026-10-11): logged with their time and
# card, never fatal. Every other match -- SMU, a ring timeout, a reset, a page fault -- invalidates the session; a
# recovery that resets the GPU logs its own reset line, and that one still fails it.
watchdog() {
    local all mes hits
    all=$(journalctl -k --since "$since" --no-pager -o short-iso 2>/dev/null |
          grep -iE 'amdgpu.*(MES|SMU|ring|timeout|reset|page fault)' || true)
    mes=$(printf '%s\n' "$all" | grep -iE 'MES' || true)
    hits=$(printf '%s\n' "$all" | grep -viE 'MES' | grep -v '^$' || true)
    if [ -n "$mes" ]; then
        say "watchdog: amdgpu MES lines since $since (logged, not fatal):"
        printf '%s\n' "$mes" | tee -a "$log"
    fi
    if [ -n "$hits" ]; then
        say "FAIL watchdog: amdgpu kernel-log lines since $since -- every number of this session is invalid:"
        printf '%s\n' "$hits" | tee -a "$log"
        return 3
    fi
    say "watchdog clean since $since$([ -n "$mes" ] && echo ' (MES lines logged above)')"
}
dock() {   # dock <name> <env...> -- <command...>
    local name=$1; shift
    local envs=()
    while [ "$1" != "--" ]; do envs+=(-e "$1"); shift; done; shift
    timeout "${EXL3_TIMEOUT:-1800}" docker run --rm --name "$name" --device /dev/kfd --device /dev/dri \
        --security-opt seccomp=unconfined --security-opt label=disable "${envs[@]}" \
        -v "$repo":/work -w /work "$image" "$@"
}

say "session start: pci=$pcis_arg image=$image ($(docker image inspect --format '{{.Id}}' "$image")) head=$(git -C "$repo" rev-parse --short HEAD)$(git -C "$repo" diff --quiet || echo '+dirty')"
watchdog || exit 3
# The previous lock holder's process can still be tearing down for a few seconds after it releases the lock (an
# engine pass's server: refused at every handoff of a series, 2026-10-12): wait up to EXL3_HANDOFF_S (90) seconds
# for KFD clients to leave, as p6's tools do at handoff. The precheck below still judges.
# EXL3_CARD_SCOPED=1 (2026-10-15, P10): two single-card jobs may run at once on DIFFERENT cards, each under its own
# card's lock (0000:13:00.0: the plan folder's gpu-13.lock; 0000:03:00.0 and any two-card job: gpu.lock, two-card
# engine passes take both). Then a KFD client counts only if it holds VRAM on a listed card
# (/sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>), and containers with GPU devices are logged, not refused -- a job on
# the other card maps /dev/kfd too. Unset: the whole-machine check, as before.
kfd_on_listed() {
    local pr pc g loc n bus dev fn
    for pr in /sys/class/kfd/kfd/proc/*; do
        [ -d "$pr" ] || continue
        if [ -z "${EXL3_CARD_SCOPED:-}" ]; then basename "$pr"; continue; fi
        for pc in "${pcis[@]}"; do
            bus=$((16#$(echo "$pc" | cut -d: -f2))); dev=$((16#$(echo "$pc" | cut -d: -f3 | cut -d. -f1))); fn=$(echo "$pc" | cut -d. -f2)
            loc=$(( (bus << 8) | (dev << 3) | fn )); g=""
            for n in /sys/class/kfd/kfd/topology/nodes/*; do
                if grep -qx "location_id $loc" "$n/properties" 2>/dev/null; then g=$(cat "$n/gpu_id"); fi
            done
            if [ -n "$g" ] && [ "$(cat "$pr/vram_$g" 2>/dev/null || echo 0)" -gt 0 ]; then basename "$pr"; break; fi
        done
    done
    return 0   # set -e / pipefail: a false test as the last command must not end the session
}
for _ in $(seq 1 $(( ${EXL3_HANDOFF_S:-90} / 5 ))); do
    [ -z "$(kfd_on_listed)" ] && break
    sleep 5
done
kfd=$(kfd_on_listed | wc -l); ctr=$(docker ps -q | wc -l)
gpu_ctr=$(for id in $(docker ps -q); do
              docker inspect --format '{{.Name}} {{.HostConfig.Privileged}} {{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' "$id"
          done | awk '$2 == "true" || / \/dev\/(kfd|dri)/ { sub("^/", "", $1); print $1 }' | tr '\n' ' ')
if [ -n "${EXL3_CARD_SCOPED:-}" ] && [ -n "$gpu_ctr" ]; then say "card-scoped: containers with GPU devices logged, not refused: ${gpu_ctr% }"; gpu_ctr=""; fi
# A card with a connected display connector holds the desktop's buffers: its own allowance, and its baseline logged.
drives_display() { cat "/sys/bus/pci/devices/$1"/drm/card*/card*-*/status 2>/dev/null | grep -qx connected; }
busy=0
for p in "${pcis[@]}"; do
    d=/sys/bus/pci/devices/$p
    vram=$(( $(cat "$d/mem_info_vram_used") / 1048576 )); gtt=$(( $(cat "$d/mem_info_gtt_used") / 1048576 ))
    lim=$idle_mb; role=""
    if drives_display "$p"; then lim=$display_mb; role=" display(allowance ${display_mb} MB)"; fi
    if [ "${#pcis[@]}" -eq 1 ] && [ -z "$role" ]; then
        say "precheck kfd_clients=$kfd gpu_containers=[${gpu_ctr% }] containers=$ctr vram_used_mb=$vram gtt_used_mb=$gtt loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
    else
        say "precheck $p$role kfd_clients=$kfd gpu_containers=[${gpu_ctr% }] containers=$ctr vram_used_mb=$vram gtt_used_mb=$gtt loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
    fi
    if [ "$vram" -gt "$lim" ]; then say "FAIL precheck: $p has $vram MB of VRAM in use (allowance $lim MB)"; busy=1; fi
done
if [ "$kfd" -ne 0 ] || [ -n "$gpu_ctr" ] || [ "$busy" -ne 0 ]; then
    say "FAIL precheck: card not idle (kfd: $(kfd_on_listed | tr '\n' ' ') containers with a GPU: $gpu_ctr)"
    exit 3
fi

# The HIP indices: devcheck with every card visible, the line naming each listed card; then proven with only the
# listed ones visible: device i at the i-th listed address, and nothing else.
all=$(dock "exl3-dev-$$" -- "$devcheck") || { say "FAIL devcheck: $all"; exit 4; }
idx=""
for p in "${pcis[@]}"; do
    i=$(echo "$all" | sed -n "s/^device \([0-9]*\) pci $p .*/\1/p" | head -1)
    [ -n "$i" ] || { say "FAIL: HIP lists no device at $p: $(echo "$all" | tr '\n' ' ')"; exit 4; }
    idx=${idx:+$idx,}$i
done
proven() {   # proven <devcheck output>: exactly the listed cards, in the listed order
    echo "$1" | grep -q "^visible ${#pcis[@]}$" || return 1
    local k=0 q
    for q in "${pcis[@]}"; do echo "$1" | grep -q "^device $k pci $q " || return 1; k=$((k + 1)); done
}
one=$(dock "exl3-dev1-$$" "HIP_VISIBLE_DEVICES=$idx" -- "$devcheck")
proven "$one" || { say "FAIL: HIP_VISIBLE_DEVICES=$idx does not show exactly $pcis_arg: $(echo "$one" | tr '\n' ' ')"; exit 4; }
if [ "${#pcis[@]}" -eq 1 ]; then
    say "card $pci is HIP index $idx; with HIP_VISIBLE_DEVICES=$idx: $(echo "$one" | tr '\n' ' ')"
else
    say "cards $pcis_arg are HIP indices $idx; with HIP_VISIBLE_DEVICES=$idx: $(echo "$one" | tr '\n' ' ')"
fi

clocks=$repo/tools/calib/clocks.py
cpid=
clk_file() { if [ "$1" = "$pci" ]; then echo "$out/clocks.jsonl"; else echo "$out/clocks-$1.jsonl"; fi; }
if [ -x "$(command -v python3)" ] && [ -f "$clocks" ]; then
    for p in "${pcis[@]}"; do
        python3 "$clocks" "$p" "$(clk_file "$p")" 0.25 2>>"$log" &
        cpid="$cpid $!"
    done
fi
export EXL3_CONTAINER="exl3-gpu-$$"
docker run -d --rm --name "$EXL3_CONTAINER" --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
    --security-opt label=disable -e "HIP_VISIBLE_DEVICES=$idx" -v "$repo":/work -w /work "$image" sleep infinity >/dev/null
trap 'docker rm -f "$EXL3_CONTAINER" >/dev/null 2>&1 || true; [ -n "$cpid" ] && kill $cpid 2>/dev/null || true' EXIT
inside=$(docker exec "$EXL3_CONTAINER" "$devcheck")
proven "$inside" || { say "FAIL: the session container does not show exactly $pcis_arg: $(echo "$inside" | tr '\n' ' ')"; exit 4; }
rc=0
say "run (host, container $EXL3_CONTAINER): $*"
(cd "$repo" && timeout "${EXL3_TIMEOUT:-1800}" "$@") > "$out/job.log" 2>&1 || rc=$?
docker rm -f "$EXL3_CONTAINER" >/dev/null 2>&1 || true
[ -n "$cpid" ] && kill $cpid 2>/dev/null || true
say "job exit $rc (output: $out/job.log)"
for p in "${pcis[@]}"; do
  cf=$(clk_file "$p")
  [ -f "$cf" ] || continue
  [ "${#pcis[@]}" -gt 1 ] && printf '%s ' "$p" | tee -a "$log"
  python3 - "$cf" <<'EOF' | tee -a "$log"
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
busy = [r for r in rows if r.get("busy", 100) > 50] or rows
med = lambda k: statistics.median([r[k] for r in busy if r.get(k) is not None]) if any(r.get(k) is not None for r in busy) else None
print(f"clocks ({rows[0]['src'] if rows else 'none'}, {len(busy)} busy samples of {len(rows)}): "
      f"sclk median {med('sclk')} MHz, mclk median {med('mclk')} MHz, mclk max {max((r.get('mclk') or 0) for r in rows) if rows else None} MHz")
EOF
done
watchdog || exit 3
exit "$rc"
