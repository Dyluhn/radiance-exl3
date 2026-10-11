#!/bin/bash
# run.sh -- one calibration GPU session on ONE card: every instrument, ROUNDS times, interleaved.
# radiance-exl3 S0.11. Apache-2.0.
#
#   tools/calib/run.sh <pci address> <out dir> [rounds (5)] [plan (full)] [-- <instrument> <args...>]
#
# Take the project's GPU lock around it yourself (flock <lockfile> tools/calib/run.sh ...); this script
# does not know where the lock lives. Build first (build.sh). Plans: `full` (the S0.11 set), `smoke`
# (one short pass of each instrument, to check the harness), `custom` (the command after --, e.g.
# `custom -- ./sweep 5 1024 2048`, for a probe with the same hygiene).
#
# Hygiene, per sample, logged to <out>/session.log:
#   before: no KFD client (/sys/class/kfd/kfd/proc empty), no running container, the card's VRAM used
#           under CALIB_VRAM_IDLE_MB (256);
#   after:  the amdgpu kernel-log watchdog since the session started; a hit stops the session (exit 3)
#           and invalidates every number taken after it;
#   always: the device line the instrument prints must name the card asked for (exit 4 otherwise).
# The HIP index of the card is looked up at the start (devinfo, no HIP_VISIBLE_DEVICES) and recorded.
# Clocks are sampled through the session by clocks.py (LACT socket, else sysfs DPM levels).
#   CALIB_IMAGE  image with the ROCm runtime (default radiance-build:1.3.0)
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
[ "$#" -ge 2 ] || { echo "usage: $0 <pci address> <out dir> [rounds] [full|smoke|custom -- <cmd...>]" >&2; exit 2; }
pci=$(echo "$1" | tr 'A-F' 'a-f'); out=$2; rounds=${3:-5}; plan=${4:-full}
custom=()
if [ "$#" -ge 5 ]; then [ "$5" = "--" ] || { echo "expected -- before the custom command" >&2; exit 2; }; shift 5; custom=("$@"); fi
[ "$plan" != custom ] || [ "${#custom[@]}" -gt 0 ] || { echo "plan custom needs -- <instrument> <args...>" >&2; exit 2; }
image=${CALIB_IMAGE:-radiance-build:1.3.0}
vram_idle_mb=${CALIB_VRAM_IDLE_MB:-256}
dev=/sys/bus/pci/devices/$pci
[ -r "$dev/mem_info_vram_used" ] || { echo "no amdgpu card at $pci ($dev/mem_info_vram_used)" >&2; exit 2; }
for t in devinfo stream sweep isarate isarate_novopd graphrate; do
    [ -x "$here/build/$t" ] || { echo "missing $here/build/$t: run build.sh first" >&2; exit 2; }
done
mkdir -p "$out"
log=$out/session.log
since=$(date '+%Y-%m-%d %H:%M:%S')
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$log"; }

docker_gpu() {   # docker_gpu <name> [env...] -- <command...>
    local name=$1; shift
    local envs=()
    while [ "$1" != "--" ]; do envs+=(-e "$1"); shift; done; shift
    timeout 600 docker run --rm --name "$name" --device /dev/kfd --device /dev/dri \
        --security-opt seccomp=unconfined --security-opt label=disable "${envs[@]}" \
        -v "$here/build:/calib:ro" -w /calib "$image" "$@" || { local rc=$?; docker kill "$name" >/dev/null 2>&1 || true; return $rc; }
}

precheck() {
    local kfd containers vram gtt
    kfd=$(ls /sys/class/kfd/kfd/proc/ | wc -l)
    containers=$(docker ps -q | wc -l)
    vram=$(( $(cat "$dev/mem_info_vram_used") / 1048576 ))
    gtt=$(( $(cat "$dev/mem_info_gtt_used") / 1048576 ))
    say "precheck kfd_clients=$kfd containers=$containers vram_used_mb=$vram gtt_used_mb=$gtt loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
    if [ "$kfd" -ne 0 ] || [ "$containers" -ne 0 ] || [ "$vram" -gt "$vram_idle_mb" ]; then
        say "FAIL precheck: card not idle (kfd: $(ls /sys/class/kfd/kfd/proc/ | tr '\n' ' ') containers: $(docker ps --format '{{.Names}}' | tr '\n' ' '))"
        exit 3
    fi
}

watchdog() {
    local hits
    hits=$(journalctl -k --since "$since" --no-pager | grep -iE 'amdgpu.*(MES|SMU|ring|timeout|reset|page fault)' || true)
    if [ -n "$hits" ]; then
        say "FAIL watchdog: amdgpu kernel-log lines since $since -- every number after them is invalid:"
        echo "$hits" | tee -a "$log"
        exit 3
    fi
    say "watchdog clean since $since"
}

sample() {   # sample <label> <command...>
    local label=$1; shift
    local f=$out/$label.log
    precheck
    say "run $label: $*"
    local t0; t0=$(date +%s.%N)
    docker_gpu "calib-$$-$label" "HIP_VISIBLE_DEVICES=$idx" -- "$@" > "$f" 2>&1 || { say "FAIL $label exited $? (see $f)"; tail -5 "$f" | tee -a "$log"; exit 1; }
    local got
    got=$(sed -n 's/^device 0 pci=\([^ ]*\).*/\1/p' "$f" | head -1)
    [ "$got" = "$pci" ] || { say "FAIL $label ran on '$got', not $pci"; exit 4; }
    echo "$label $t0 $(date +%s.%N)" >> "$out/windows.txt"   # for attributing clock samples
    watchdog
}

say "session start: pci=$pci rounds=$rounds plan=$plan image=$image ($(docker image inspect --format '{{.Id}}' "$image"))"
say "host $(uname -r); binaries:"; (cd "$here/build" && sha256sum devinfo stream sweep isarate isarate_novopd graphrate) | tee -a "$log"
watchdog
precheck
docker_gpu "calib-$$-devinfo" -- ./devinfo > "$out/devinfo.log" 2>&1
cat "$out/devinfo.log" >> "$log"
idx=$(sed -n "s/^device \([0-9]*\) pci=$pci .*/\1/p" "$out/devinfo.log" | head -1)
[ -n "$idx" ] || { say "FAIL: HIP lists no device at $pci"; exit 4; }
say "HIP index of $pci is $idx (HIP_VISIBLE_DEVICES=$idx)"

python3 "$here/clocks.py" "$pci" "$out/clocks.jsonl" 0.2 2>>"$log" &
clocks_pid=$!
# On every exit, a failed sample's included: stop the sampler and run the watchdog once more.
trap 'kill $clocks_pid 2>/dev/null || true; (watchdog) || true' EXIT

# isarate occupancy arms, 256-thread blocks, LDS a block -> waves a SIMD on gfx12 (64 KiB a WGP):
#   0 -> 16 (the author's first arm), 16000 -> 8, 24656 -> 4, 36864 -> 2 (the author's second arm).
# 8 is where the EXL3 16x128 gate/up decode tile runs (512 threads, 24.6 KiB: two blocks a WGP).
ldss="0 16000 24656 36864"
for r in $(seq 1 "$rounds"); do
    if [ "$plan" = custom ]; then
        sample "r$r-$(basename "${custom[0]}")" "${custom[@]}"
    elif [ "$plan" = smoke ]; then
        sample "r$r-stream"    ./stream 2
        sample "r$r-isarate"   ./isarate 2000 256 256 $ldss
        sample "r$r-sweep"     ./sweep 1
        sample "r$r-graphrate" ./graphrate 45 30 10
    else
        sample "r$r-stream"         ./stream 10
        sample "r$r-isarate"        ./isarate 20000 1024 256 $ldss
        sample "r$r-isarate_novopd" ./isarate_novopd 20000 1024 256 $ldss
        sample "r$r-sweep"          ./sweep 5
        # The same fit on sizes clear of the two dispatch-shape bumps found 2026-10-11 (254-256 and
        # ~1920-2560 workgroups of 8 KiB; notes/calibration.md). Post-hoc, and reported as such.
        sample "r$r-sweepob"        ./sweep 5 1032 1536 2056 3072 4104 6144 8200 10240 12288 14336 24576 32776 49152 65544
        sample "r$r-graphrate"      ./graphrate 45 300 10
    fi
done
say "session end"
