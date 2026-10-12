#!/bin/bash
# p6_profile.sh -- radiance's per-op device profile (--profile-ops) of ONE container on the real model, decode only,
# at each concurrency of PROFILE_LEVELS (default "1 4 8") in turn on one server: the method of radiance 1.3.0
# scripts/dprofd.sh (the counters are cumulative and dumped every RADIANCE_PROFILE_EVERY steps; two dumps that both
# fall inside a steady decode phase, differenced, give that phase's per-step cost). Read the result with
# tools/p6_profile_diff.py.
#
#   flock <gpu lock> tools/p6_profile.sh <container.rad> <label> <outdir> [plugin home]
#
# Writes <outdir>/<label>/: server.log (docker logs -t: every line timestamped), phases.txt (UTC epoch edges of each
# level's decode phase, c<C>_start / c<C>_end), stats-*.json (/stats at the edges: the mover's copies are not ops,
# its link counters
# are), provenance.txt. A profiled step is several times slower than an unprofiled one (a synchronisation around every
# launch) and radiance says so in every dump: these are device times per op, NOT a step-time measurement.
# Env: PROFILE_IMAGE (stilldeadcode/radiance:<pin>), PROFILE_FLAGS (the full flag set, required; its --max-num-seqs
# must admit the largest level), PROFILE_LEVELS ("1 4 8": the concurrencies, in order), PROFILE_EVERY (25),
# PROFILE_MT (600 tokens a request, ignore_eos), PROFILE_PORT (8101), PROFILE_KV_PARENT (<outdir>),
# PROFILE_MIN_MEM_GIB / PROFILE_MEM_ABORT_GIB (unset = off: the same memory gate and in-pass watchdog as run_pass.sh).
set -euo pipefail
die() { printf 'p6_profile: %s\n' "$*" >&2; exit "${2:-2}"; }
[ "$#" -ge 3 ] && [ "$#" -le 4 ] || die "usage: p6_profile.sh <container.rad> <label> <outdir> [plugin home]"
repo=$(cd "$(dirname "$0")/.." && pwd)
model=$(realpath -e -- "$1") || die "no such container: $1"
pd=$(realpath -m -- "$3")/$2
[ ! -e "$pd" ] || die "refusing to overwrite $pd"
plugins=""; [ "$#" -eq 4 ] && { plugins=$(realpath -e -- "$4") || die "no plugin home $4"; }
: "${PROFILE_IMAGE:=stilldeadcode/radiance:$(cut -d' ' -f1 "$repo/RADIANCE_VERSION")}"
: "${PROFILE_FLAGS:?set PROFILE_FLAGS to the full flag set of the series}"
: "${PROFILE_LEVELS:=1 4 8}" "${PROFILE_EVERY:=25}" "${PROFILE_MT:=600}" "${PROFILE_PORT:=8101}"
: "${PROFILE_MIN_MEM_GIB:=}" "${PROFILE_MEM_ABORT_GIB:=}"
: "${PROFILE_KV_PARENT:=$(dirname "$pd")}"
for t in docker curl jq python3; do command -v "$t" > /dev/null || die "$t is not on PATH"; done
mkdir -p "$pd"
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$pd/profile.log"; }
mem() { awk '/^MemAvailable:/ {printf "%.1f", $2 / 1048576}' /proc/meminfo; }
gt() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a > b) }'; }
[ -z "$(ls /sys/class/kfd/kfd/proc/ 2>/dev/null)" ] || die "KFD clients present: run under the GPU lock" 3
if [ -n "$PROFILE_MIN_MEM_GIB" ]; then
    w=0; while gt "$PROFILE_MIN_MEM_GIB" "$(mem)"; do [ $w -lt 1800 ] || die "MemAvailable $(mem) GiB < $PROFILE_MIN_MEM_GIB for 30 min" 3; sleep 15; w=$((w + 15)); done
fi
name=p6-profile-$2-$$
kv=$(mktemp -d "$PROFILE_KV_PARENT/kvcache.XXXXXX")
wd=""
cleanup() { [ -z "$wd" ] || kill "$wd" 2> /dev/null || true
            docker logs -t "$name" > "$pd/server.log" 2>&1 || true; docker rm -f "$name" > /dev/null 2>&1 || true; rm -rf -- "$kv"; }
trap cleanup EXIT
home=(-e RADIANCE_HOME=/opt/radiance/share/radiance)
[ -z "$plugins" ] || home=(-v "$plugins:/plugins:ro" -e RADIANCE_HOME=/plugins:/opt/radiance/share/radiance)
# shellcheck disable=SC2086  # PROFILE_FLAGS is one flag or value per word
cmd=(docker run -d --name "$name" --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined
     --security-opt label=disable --init --ulimit memlock=-1 -p "127.0.0.1:$PROFILE_PORT:$PROFILE_PORT"
     -e RADIANCE_PROFILE_EVERY="$PROFILE_EVERY" -v "$(dirname "$model"):/models:ro" -v "$kv:/kvcache" "${home[@]}"
     "$PROFILE_IMAGE" --model "/models/$(basename "$model")" $PROFILE_FLAGS --profile-ops
     --prefix-cache-dir /kvcache --host 0.0.0.0 --port "$PROFILE_PORT")
{ printf 'label %s\nutc %s\nimage %s %s\nmodel %s\nplugins %s\nrepo %s\nevery %s mt %s\n' "$2" "$(date -u +%FT%TZ)" \
      "$PROFILE_IMAGE" "$(docker image inspect --format '{{.Id}}' "$PROFILE_IMAGE")" "$model" "${plugins:-none}" \
      "$(git -C "$repo" rev-parse --short HEAD)" "$PROFILE_EVERY" "$PROFILE_MT"; printf 'levels %s\ncommand ' "$PROFILE_LEVELS"
  printf '%q ' "${cmd[@]}"; echo; } > "$pd/provenance.txt"
say "starting $name (MemAvailable $(mem) GiB)"
"${cmd[@]}" > /dev/null
if [ -n "$PROFILE_MEM_ABORT_GIB" ]; then
    ( while sleep 1; do gt "$PROFILE_MEM_ABORT_GIB" "$(mem)" && { echo "$(date -u +%FT%TZ) MemAvailable $(mem) GiB" > "$pd/memory-abort"; docker stop -t 5 "$name" > /dev/null 2>&1; exit 0; }; done ) &
    wd=$!
fi
alive() { [ "$(docker container inspect -f '{{.State.Running}}' "$name" 2> /dev/null)" = true ]; }
w=0
until curl -s -m 5 -o /dev/null "http://127.0.0.1:$PROFILE_PORT/health"; do
    alive || die "the server exited before /health (see $pd/server.log)" 1
    [ $w -lt 1800 ] || die "no /health in 1800 s" 1; sleep 5; w=$((w + 5))
done
say "ready after ${w}s"
url=http://127.0.0.1:$PROFILE_PORT
body=$(jq -nc --argjson mt "$PROFILE_MT" '{model: "m", prompt: "Write a long story about a lighthouse keeper and the ships he watched.", max_tokens: $mt, temperature: 0, ignore_eos: true}')
phase() {   # phase <C>: C concurrent requests of PROFILE_MT tokens; edges into phases.txt
    local c=$1 i pids=""
    curl -s -m 30 "$url/stats" > "$pd/stats-c$c-start.json"
    printf 'c%s_start %s\n' "$c" "$(date +%s.%N)" >> "$pd/phases.txt"
    for i in $(seq 1 "$c"); do curl -s -m 3600 "$url/v1/completions" -H 'Content-Type: application/json' -d "$body" -o "$pd/c$c-$i.json" & pids="$pids $!"; done
    # shellcheck disable=SC2086
    wait $pids || true
    printf 'c%s_end %s\n' "$c" "$(date +%s.%N)" >> "$pd/phases.txt"
    curl -s -m 30 "$url/stats" > "$pd/stats-c$c-end.json"
    alive || die "the server exited during the C=$c phase$( [ -f "$pd/memory-abort" ] && echo ": $(cat "$pd/memory-abort")")" 5
    say "C=$c: $(jq -r '.usage.completion_tokens' "$pd/c$c-1.json") tokens a request"
}
first=1
for c in $PROFILE_LEVELS; do
    [ "$first" = 1 ] || sleep 2
    first=0
    phase "$c"
done
say "done; stopping $name"
