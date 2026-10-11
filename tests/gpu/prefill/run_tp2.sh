#!/bin/bash
# run_tp2.sh -- one GPU session on TWO named cards (a --tp 2 check), with tests/gpu/run_gpu.sh's hygiene for both.
# Nothing about the machine is in here: the cards, the lock and the image are arguments.
#
#   tests/gpu/prefill/run_tp2.sh <pci A> <pci B> <lock file> <log dir> -- <command...>
#
# <command> runs ON THE HOST (cwd: this repository) while one container of the build image (EXL3_IMAGE, default
# radiance-build:1.3.0) is up with this repository at /work and BOTH cards visible (HIP_VISIBLE_DEVICES = their two HIP
# indices, A first); its name is in $EXL3_CONTAINER. Under the lock (flock <lock file>, for the whole session):
#   - precheck: no KFD client, no running container with a GPU, card A's VRAM in use under EXL3_VRAM_IDLE_MB (256) and
#     card B's under EXL3_VRAM_IDLE_MB_B (default 1024: a card driving a display holds a few hundred MiB) -- exit 3;
#   - both HIP indices looked up and PROVEN from inside the session container (devcheck: exactly the two cards) -- exit 4;
#   - the clock sampler for both cards (tools/calib/clocks.py), when present.
# After it: the amdgpu kernel-log watchdog since the session started (any MES/SMU/ring/timeout/reset/page-fault line
# invalidates the session, exit 3) and a clock summary a card.
set -euo pipefail

[ "$#" -ge 6 ] && [ "$5" = "--" ] || { echo "usage: $0 <pci A> <pci B> <lock> <log dir> -- <command...>" >&2; exit 2; }
pa=$(echo "$1" | tr 'A-F' 'a-f'); pb=$(echo "$2" | tr 'A-F' 'a-f'); lock=$3; out=$4; shift 5
repo=$(cd "$(dirname "$0")/../../.." && pwd)
image=${EXL3_IMAGE:-radiance-build:1.3.0}
devcheck=${EXL3_DEVCHECK:-build-gpu/kernels/exl3r4d/tests-gpu/exl3r4d_devcheck}
for p in "$pa" "$pb"; do
    [ -r "/sys/bus/pci/devices/$p/mem_info_vram_used" ] || { echo "no amdgpu card at $p" >&2; exit 2; }
done
mkdir -p "$out"
log=$out/session.log
if [ -z "${EXL3_IN_LOCK:-}" ]; then
    EXL3_IN_LOCK=1 exec flock "$lock" "$0" "$pa" "$pb" "$lock" "$out" -- "$@"
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
mb() { echo $(( $(cat "/sys/bus/pci/devices/$1/mem_info_vram_used") / 1048576 )); }

say "session start: pci=$pa,$pb image=$image ($(docker image inspect --format '{{.Id}}' "$image")) head=$(git -C "$repo" rev-parse --short HEAD)$(git -C "$repo" diff --quiet || echo '+dirty')"
watchdog || exit 3
kfd=$(ls /sys/class/kfd/kfd/proc/ | wc -l)
gpu_ctr=$(for id in $(docker ps -q); do
              docker inspect --format '{{.Name}} {{.HostConfig.Privileged}} {{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' "$id"
          done | awk '$2 == "true" || / \/dev\/(kfd|dri)/ { sub("^/", "", $1); print $1 }' | tr '\n' ' ')
va=$(mb "$pa"); vb=$(mb "$pb")
say "precheck kfd_clients=$kfd gpu_containers=[${gpu_ctr% }] vram_used_mb $pa=$va $pb=$vb loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
if [ "$kfd" -ne 0 ] || [ -n "$gpu_ctr" ] || [ "$va" -gt "${EXL3_VRAM_IDLE_MB:-256}" ] || [ "$vb" -gt "${EXL3_VRAM_IDLE_MB_B:-1024}" ]; then
    say "FAIL precheck: cards not idle"
    exit 3
fi

all=$(timeout 300 docker run --rm --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
      --security-opt label=disable -v "$repo":/work -w /work "$image" "$devcheck") || { say "FAIL devcheck: $all"; exit 4; }
ia=$(echo "$all" | sed -n "s/^device \([0-9]*\) pci $pa .*/\1/p" | head -1)
ib=$(echo "$all" | sed -n "s/^device \([0-9]*\) pci $pb .*/\1/p" | head -1)
[ -n "$ia" ] && [ -n "$ib" ] || { say "FAIL: HIP does not list both $pa and $pb: $(echo "$all" | tr '\n' ' ')"; exit 4; }

clocks=$repo/tools/calib/clocks.py
cpids=()
if [ -x "$(command -v python3)" ] && [ -f "$clocks" ]; then
    for p in "$pa" "$pb"; do python3 "$clocks" "$p" "$out/clocks-$p.jsonl" 0.25 2>>"$log" & cpids+=($!); done
fi
export EXL3_CONTAINER="exl3-tp2-$$"
docker run -d --rm --name "$EXL3_CONTAINER" --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
    --security-opt label=disable -e "HIP_VISIBLE_DEVICES=$ia,$ib" -v "$repo":/work -w /work "$image" sleep infinity >/dev/null
trap 'docker rm -f "$EXL3_CONTAINER" >/dev/null 2>&1 || true; for c in "${cpids[@]}"; do kill "$c" 2>/dev/null || true; done' EXIT
inside=$(docker exec "$EXL3_CONTAINER" "$devcheck")
echo "$inside" | grep -q "^visible 2$" && echo "$inside" | grep -q "^device 0 pci $pa " && echo "$inside" | grep -q "^device 1 pci $pb " ||
    { say "FAIL: the session container does not show exactly $pa, $pb: $(echo "$inside" | tr '\n' ' ')"; exit 4; }
say "cards: $pa is HIP index $ia, $pb is $ib; in the session container: $(echo "$inside" | tr '\n' ' ')"
rc=0
say "run (host, container $EXL3_CONTAINER): $*"
(cd "$repo" && timeout "${EXL3_TIMEOUT:-1800}" "$@") > "$out/job.log" 2>&1 || rc=$?
docker rm -f "$EXL3_CONTAINER" >/dev/null 2>&1 || true
for c in "${cpids[@]}"; do kill "$c" 2>/dev/null || true; done
say "job exit $rc (output: $out/job.log)"
for f in "$out"/clocks-*.jsonl; do
    [ -f "$f" ] || continue
    python3 - "$f" <<'EOF' | tee -a "$log"
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
busy = [r for r in rows if r.get("busy", 100) > 50] or rows
med = lambda k: statistics.median([r[k] for r in busy if r.get(k) is not None]) if any(r.get(k) is not None for r in busy) else None
print(f"clocks {sys.argv[1].rsplit('/', 1)[-1]} ({len(busy)} busy of {len(rows)}): sclk median {med('sclk')} MHz, mclk median {med('mclk')} MHz")
EOF
done
watchdog || exit 3
exit "$rc"
