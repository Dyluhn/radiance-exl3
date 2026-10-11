#!/bin/bash
# run_pass.sh -- ONE speed pass of a .rad container: a fresh server in the radiance runtime image with
# the production flags, the measurement set run against it, everything that explains the numbers
# captured beside them, and a clean stop. radiance-exl3 P0 measures the baseline's noise floors with
# it; P6 runs the EXL3 container through it unchanged (with its plugin home). Apache-2.0.
#
#   tools/gates/speed/run_pass.sh <container.rad> <label> <outdir> [plugin home]
#
# Writes <outdir>/<label>/ and refuses to overwrite it. tools/gates/speed/summarize.py reads passes.
#
# ONE PASS, in this order, against ONE server started for it (a pass is a boot: the engine reads the
# card's "already held" VRAM once at startup and sizes the expert slab from it, and that reading moves
# by ~29 MiB from boot to boot -- RidgeFill notes/stagee.md -- so the spread between passes is the
# spread a different container will be judged against, and it must include that):
#   ident     radiance scripts/ident.sh: 6 hashes (3 temperatures x 2 questions). FIRST, so every pass
#             asks it of a fresh server -- across passes that is R8 (identical over restarts)
#   cstep2    scripts/cstep2.sh LEVELS="1 2 4 8" MT=<SPEED_CSTEP2_MT>, ONE invocation as the author runs
#             it: steady-state decode ms/step, tokens/step, MTP accept%.
#             STALLS. On this deployment the engine sometimes idles BOTH cards together for 0.5-14 s
#             inside a step with the host quiet (2026-10-11: 7 of 11 baseline passes); one inside a 6 s
#             window reads as a 5-14% slower level. summarize.py recovers each level's time window from
#             the server's per-request log lines and flags a level STALLED when either card's clock
#             sampler reads under 95% busy in [level start + 3 s, + 12 s] (the stretch cstep2 samples),
#             and SPLIT when the level's requests were admitted in more than one scheduler step.
#             REJECTED (2026-10-11, base-p6..p8): one cstep2 invocation per level. It gave each level a
#             window directly but split the admission of a level's requests in 5 of 9 level runs at
#             C >= 2 (0 of 24 as one invocation) and moved C=4/C=8 by up to 4.8%.
#             MT is raised from the script's 900 because with the production
#             MTP depth a 900-token sequence ENDS INSIDE the sampling window at C=1, 2 and 4 (measured
#             2026-10-11: ~2.6 tokens a step, done in ~7 s, and cstep2 flags "count moved to 0"). MT only
#             moves the END of each sequence; the window cstep2 samples (3 s after all C run, 6 s long)
#             is the same stretch of generation either way.
#   fnpf-8k   scripts/fnpf.sh N=8000, fnpf-24k N=24000: prefill tok/s at the 2048 chunk (warm + 3 reps)
#   fnconc    scripts/fnconc.sh C=4 DEC=300 PROMPT=24000: a 24K prefill beside 4 decoders, both halves
#   ttft-c1   ttft.py: TTFT of SPEED_TTFT_TOKENS-token prompts, one at a time: 6 warm-up requests left
#             out (the server settles on the new text over the first ~4), then 9 measured
#   ttft-c8   ttft.py: 8 at once, 1 warm-up burst, then 4 measured bursts (32 requests)
# After each step: /stats (per-layer expert residency, hit rate, draft acceptance) and /metrics.
# Throughout: both cards' clocks (tools/calib/clocks.py: the LACT daemon, else sysfs DPM levels) and
# the host load (hostload.py: load average, and the CPU the server used vs everything else).
#
# THE FLAGS are the author's production command at radiance 1.3.0, deploy/compose/flashnext.yaml
# (scripts/fnserve.sh records why each is what it is):
#   --tp 2 --tp-wire wht6 --max-num-seqs 8 --max-model-len 200000 --placement expert_tiered
#   --host-pool-mib 12288 --gpu-headroom-mib 96 --kv-cache-dtype fp8 --prefix-cache-host-mib 4096
#   --prefix-cache-disk-mib 131072 --num-speculative-tokens 3 --max-num-batched-tokens 2048
# and the harness adds --prefix-cache-dir /kvcache (a FRESH empty directory per pass: the disk tier
# is persistent, and a server that found the last pass's sessions there would answer from them),
# --host 0.0.0.0 --port <SPEED_PORT>. Left out of the compose line: --api-key (loopback only) and
# --host :: (the port is published on 127.0.0.1). The HF model card's line also carries
# --expert-vs-cache-ratio 0.82, which radiance 1.1.0 removed (1.3.0 refuses it).
#
# HYGIENE. Before the server starts: the host must be quiet (1-min load <= SPEED_MAX_LOAD, waited
# for), no process holds /dev/kfd (idle containers with the devices mapped are recorded), every discrete card is
# under SPEED_VRAM_MIB of VRAM, nothing answers on the port, and the kernel log of the last 30 min
# has no amdgpu MES/SMU/ring/timeout/reset line. With SPEED_LOCK set, the GPU lock file is taken
# after the host is quiet and held for this pass only.
# Verdict (pass.json, and the exit code):
#   valid         0
#   failed        1  the server or a step failed: no numbers
#   contaminated  4  1-min load > SPEED_MAX_LOAD at the start, or at the end while the CPU used by
#                    everything but the server averaged > SPEED_MAX_LOAD cores over the measurement
#                    (the server's own threads are in the load average; the cgroup split says whose),
#                    or any process outside the server held the GPU during the measurement
#   invalid       5  an amdgpu MES/SMU/ring/timeout/reset/page-fault line in the kernel log between
#                    the server's start and the end of the measurement
# and exit 6 whatever the verdict when a MES/SMU/ring/timeout/reset line appears while the server
# stops: the machine needs attention before any further GPU work. Page faults while it stops are
# recorded, not fatal: radiance's release profile faults the card as a server stops after long
# prompts, stock included (RidgeFill notes/rebase-master.md).
# Usage errors exit 2; a refusal before anything started (busy GPU, host never quiet) exits 3.
#
# Environment (no machine fact is in this file):
#   RADIANCE_SRC        radiance source tree at the pinned release (default <repo>/data/radiance-src):
#                       the measurement scripts and fnpf's corpus come from it; its HEAD must be the
#                       commit in <repo>/RADIANCE_VERSION when it is a git checkout
#   SPEED_IMAGE         default stilldeadcode/radiance:<tag in RADIANCE_VERSION>
#   SPEED_FLAGS         default the production flags above (a different set is a different benchmark)
#   SPEED_EXTRA_FLAGS   appended after SPEED_FLAGS (e.g. --kernels for a plugin home); default none
#   SPEED_PORT          8100
#   SPEED_LOCK          a lock file to hold for the pass (flock); default none (the caller locks)
#   SPEED_MAX_LOAD      3
#   SPEED_QUIET_WAIT_S  3600: how long to wait for a quiet host before refusing
#   SPEED_VRAM_MIB      1024: per-card VRAM ceiling before the start (a display card holds a few hundred)
#   SPEED_SERVE_TIMEOUT 1800 s for /health
#   SPEED_KV_PARENT     where the per-pass disk-tier directory is made (default <outdir>); deleted at the
#                       pass's end. The engine sizes its store files to the cap at once (ftruncate: 128 GiB
#                       APPARENT, sparse); what is allocated is what was written, a few GiB a pass
#   SPEED_TTFT_TOKENS   256 (8 x 256 = one 2048-token step for the C=8 burst)
#   SPEED_CSTEP2_MT     3000: tokens each cstep2 sequence generates (above)
# Needs: bash, docker, curl, jq, python3, flock, journalctl, awk.
set -euo pipefail

die() { printf 'run_pass: %s\n' "$*" >&2; exit "${2:-2}"; }
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)

[ "$#" -ge 3 ] && [ "$#" -le 4 ] || die "usage: run_pass.sh <container.rad> <label> <outdir> [plugin home]"
model=$(realpath -e -- "$1" 2>/dev/null) || die "no such container: $1"
[ -f "$model" ] || die "not a file: $1"
label=$2
[[ "$label" =~ ^[A-Za-z0-9._-]+$ ]] || die "label must be [A-Za-z0-9._-]+: $label"
outdir=$(realpath -m -- "$3")
plugins=""
if [ "$#" -eq 4 ]; then plugins=$(realpath -e -- "$4" 2>/dev/null) && [ -d "$plugins" ] || die "plugin home is not a directory: $4"; fi
pd=$outdir/$label
[ ! -e "$pd" ] || die "refusing to overwrite $pd"
for t in docker curl jq python3 flock journalctl awk; do command -v "$t" >/dev/null || die "$t is not on PATH"; done

[ -r "$repo/RADIANCE_VERSION" ] || die "no RADIANCE_VERSION at the repo root $repo (the pinned release: '<tag> <commit>')"
pin=$(cut -d' ' -f1 "$repo/RADIANCE_VERSION"); pin_commit=$(cut -d' ' -f2 "$repo/RADIANCE_VERSION")
: "${RADIANCE_SRC:=$repo/data/radiance-src}"
: "${SPEED_IMAGE:=stilldeadcode/radiance:$pin}"
: "${SPEED_FLAGS:=--tp 2 --tp-wire wht6 --max-num-seqs 8 --max-model-len 200000 --placement expert_tiered --host-pool-mib 12288 --gpu-headroom-mib 96 --kv-cache-dtype fp8 --prefix-cache-host-mib 4096 --prefix-cache-disk-mib 131072 --num-speculative-tokens 3 --max-num-batched-tokens 2048}"
: "${SPEED_EXTRA_FLAGS:=}"
: "${SPEED_PORT:=8100}"
: "${SPEED_LOCK:=}"
: "${SPEED_MAX_LOAD:=3}"
: "${SPEED_QUIET_WAIT_S:=3600}"
: "${SPEED_VRAM_MIB:=1024}"
: "${SPEED_SERVE_TIMEOUT:=1800}"
SPEED_KV_PARENT=$(realpath -m -- "${SPEED_KV_PARENT:-$outdir}")
: "${SPEED_TTFT_TOKENS:=256}"
: "${SPEED_CSTEP2_MT:=3000}"
S=$RADIANCE_SRC/scripts
for s in ident.sh cstep2.sh fnpf.sh fnconc.sh; do [ -r "$S/$s" ] || die "RADIANCE_SRC has no scripts/$s: $RADIANCE_SRC"; done
src_head=$(git -C "$RADIANCE_SRC" rev-parse HEAD 2>/dev/null || echo "not a git checkout")
[ "$src_head" = "not a git checkout" ] || [ "$src_head" = "$pin_commit" ] \
    || die "RADIANCE_SRC is at $src_head, not the pinned $pin $pin_commit"
digest=$(docker image inspect --format '{{.Id}}' "$SPEED_IMAGE" 2>/dev/null) || die "no image $SPEED_IMAGE (pull it)"

mkdir -p "$pd/steps"
log=$pd/pass.log
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$log"; }
load1() { cut -d' ' -f1 /proc/loadavg; }
gt() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a > b) }'; }
klog_pat='amdgpu.*(MES|SMU|ring|timeout|reset|page fault)'
klog_fatal='amdgpu.*(MES|SMU|ring|timeout|reset)'
discrete_cards() {   # PCI addresses of AMD cards with more than 8 GiB of VRAM
    for dev in /sys/class/drm/card*/device; do
        [ -r "$dev/mem_info_vram_total" ] && [ "$(cat "$dev/vendor")" = 0x1002 ] || continue
        [ "$(cat "$dev/mem_info_vram_total")" -gt $((8 << 30)) ] || continue
        basename "$(readlink -f "$dev")"
    done | sort -u
}

# ---- quiet host, then the lock (held for this pass only), then the GPU checks
waited=0
while :; do
    until ! gt "$(load1)" "$SPEED_MAX_LOAD"; do
        [ "$waited" -lt "$SPEED_QUIET_WAIT_S" ] || die "host not quiet: 1-min load $(load1) > $SPEED_MAX_LOAD for ${SPEED_QUIET_WAIT_S}s" 3
        [ $((waited % 300)) -ne 0 ] || say "waiting for a quiet host: load $(cut -d' ' -f1-3 /proc/loadavg)"
        sleep 15; waited=$((waited + 15))
    done
    if [ -n "$SPEED_LOCK" ]; then
        exec 9>"$SPEED_LOCK"; say "waiting for the lock $SPEED_LOCK"; flock 9; say "lock taken"
    fi
    gt "$(load1)" "$SPEED_MAX_LOAD" || break
    [ -z "$SPEED_LOCK" ] || { flock -u 9; say "load rose to $(load1) while waiting for the lock: released, waiting again"; }
done
say "quiet: load $(cut -d' ' -f1-3 /proc/loadavg) after ${waited}s"

kfd=$(ls /sys/class/kfd/kfd/proc/ 2>/dev/null | tr '\n' ' ')
[ -z "$kfd" ] || die "KFD clients present (pids: $kfd): another process holds a GPU" 3
gpu_containers=""   # recorded, not refused: an idle container with the devices mapped holds no GPU (the
                    # KFD check above and the sampler during the pass see any process that does)
for c in $(docker ps -q); do
    docker inspect -f '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' "$c" | grep -q '/dev/kfd' \
        && gpu_containers="$gpu_containers $(docker ps --filter id="$c" --format '{{.Names}}({{.Image}})')"
done
vram=""
for pci in $(discrete_cards); do
    used=$(( $(cat "/sys/bus/pci/devices/$pci/mem_info_vram_used") >> 20 ))
    vram="$vram $pci=${used}MiB"
    [ "$used" -le "$SPEED_VRAM_MIB" ] || die "$pci holds $used MiB of VRAM (> SPEED_VRAM_MIB $SPEED_VRAM_MIB)" 3
done
[ -n "$vram" ] || die "no discrete AMD card under /sys/class/drm" 3
! curl -s -m 2 -o /dev/null "http://127.0.0.1:$SPEED_PORT/health" || die "something already answers on port $SPEED_PORT" 3
pre_klog=$(journalctl -k --since -30min --no-pager | grep -iE "$klog_pat" || true)
! printf '%s\n' "$pre_klog" | grep -qiE "$klog_fatal" || die "amdgpu errors in the last 30 min: $(printf '%s\n' "$pre_klog" | grep -iE "$klog_fatal" | tail -2)" 3
pre_faults=$(printf '%s\n' "$pre_klog" | grep -ciE 'page fault' || true)
say "preflight OK: no KFD client, VRAM$vram, kernel log clean (30 min; $pre_faults earlier page-fault line(s)); idle containers with GPU devices:${gpu_containers:- none}"

# ---- a cleanup that always runs: stop the server, the samplers, remove the disk tier's directory
name=radiance-exl3-speed-$label-$$
kv=""
samplers=()
stopped=0
stop_server() {
    [ "$stopped" = 0 ] || return 0
    stopped=1
    if docker container inspect "$name" >/dev/null 2>&1; then
        say "stopping $name (SIGTERM, 60 s grace)"
        docker stop -t 60 "$name" >/dev/null || true
        docker logs "$name" > "$pd/server.log" 2>&1 || true
        docker rm "$name" >/dev/null || true
    fi
    for p in "${samplers[@]}"; do kill "$p" 2>/dev/null || true; done
    local w=0 hi
    while [ "$w" -lt 300 ]; do
        hi=0
        for pci in $(discrete_cards); do
            [ $(( $(cat "/sys/bus/pci/devices/$pci/mem_info_vram_used") >> 20 )) -le "$SPEED_VRAM_MIB" ] || hi=1
        done
        [ "$hi" = 1 ] || break
        sleep 2; w=$((w + 2))
    done
    say "VRAM back under $SPEED_VRAM_MIB MiB after ${w}s"
    kv_bytes=0
    case "$kv" in "$SPEED_KV_PARENT"/kvcache.??????)
        kv_bytes=$(du -sB1 "$kv" | cut -f1); rm -rf -- "$kv"; say "disk tier held $kv_bytes B on disk; removed $kv" ;;
    esac
}
trap 'stop_server' EXIT
trap 'exit 130' INT TERM
mkdir -p "$SPEED_KV_PARENT"
kv=$(mktemp -d "$SPEED_KV_PARENT/kvcache.XXXXXX")

# ---- the command and its provenance
home_args=(-e RADIANCE_HOME=/opt/radiance/share/radiance)
[ -z "$plugins" ] || home_args=(-v "$plugins:/plugins:ro" -e RADIANCE_HOME=/plugins:/opt/radiance/share/radiance)
# shellcheck disable=SC2206  # the flag strings are one flag or value per word by construction
cmd=(docker run -d --name "$name" --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined
     --security-opt label=disable --init --ulimit memlock=-1 -p "127.0.0.1:$SPEED_PORT:$SPEED_PORT"
     -v "$(dirname "$model"):/models:ro" -v "$kv:/kvcache" "${home_args[@]}"
     "$SPEED_IMAGE" --model "/models/$(basename "$model")" $SPEED_FLAGS $SPEED_EXTRA_FLAGS
     --prefix-cache-dir /kvcache --host 0.0.0.0 --port "$SPEED_PORT")
harness=$(git -C "$repo" rev-parse --short HEAD 2>/dev/null || echo unknown)
[ -z "$(git -C "$repo" status --porcelain -- tools/gates/speed 2>/dev/null)" ] || harness="$harness+dirty"
{
    printf 'label: %s\nutc_start: %s\nimage: %s %s\n' "$label" "$(date -u +%FT%TZ)" "$SPEED_IMAGE" "$digest"
    printf 'model: %s (%s B)\nplugin_home: %s\nharness: %s\n' "$model" "$(stat -c %s "$model")" "${plugins:-none}" "$harness"
    printf 'radiance_src: %s at %s\n' "$RADIANCE_SRC" "$src_head"
    (cd "$S" && sha256sum ident.sh cstep2.sh fnpf.sh fnconc.sh) | sed 's/^/script_sha256: /'
    printf 'host: %s, %s cpus, %s\n' "$(uname -r)" "$(nproc)" "$(grep -E 'MemAvailable|SwapFree' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"
    printf 'preflight: no KFD client; VRAM%s; kernel log clean; %s earlier page-fault line(s) in 30 min; idle containers with GPU devices:%s\n' "$vram" "$pre_faults" "${gpu_containers:- none}"
    printf 'load_start: %s\n' "$(cut -d' ' -f1-3 /proc/loadavg)"
    printf 'command: '; printf '%q ' "${cmd[@]}"; printf '\n'
} > "$pd/provenance.txt"

for pci in $(discrete_cards); do
    python3 "$repo/tools/calib/clocks.py" "$pci" "$pd/clocks-$pci.jsonl" 0.5 2>>"$log" & samplers+=($!)
done
t_start=$(date '+%Y-%m-%d %H:%M:%S'); load_start=$(load1)
say "starting $name"
verdict=valid; reasons=()
if "${cmd[@]}" > "$pd/container.id" 2>>"$log"; then
    cpid=$(docker inspect -f '{{.State.Pid}}' "$name")
    python3 "$here/hostload.py" "$pd/hostload.jsonl" 2 "/sys/fs/cgroup$(sed -n 's/^0:://p' "/proc/$cpid/cgroup")" 2>>"$log" & samplers+=($!)
else
    verdict=failed; reasons+=("docker run failed (pass.log)")
fi
waited=0
until [ "$verdict" = failed ] || [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$SPEED_PORT/health" || true)" = 200 ]; do
    [ "$(docker container inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" = true ] \
        || { verdict=failed; reasons+=("the server exited before /health answered"); break; }
    [ "$waited" -lt "$SPEED_SERVE_TIMEOUT" ] || { verdict=failed; reasons+=("no /health in ${SPEED_SERVE_TIMEOUT}s"); break; }
    sleep 5; waited=$((waited + 5))
    [ $((waited % 60)) -ne 0 ] || say "  ...loading, ${waited}s"
done

snap() {
    curl -s -m 10 "http://127.0.0.1:$SPEED_PORT/stats" > "$pd/stats-$1.json" || true
    curl -s -m 10 "http://127.0.0.1:$SPEED_PORT/metrics" > "$pd/metrics-$1.txt" || true
}
step() {   # step <name> <timeout s> <command...>
    local n=$1 to=$2; shift 2
    [ "$verdict" != failed ] || return 0
    local t0 rc=0; t0=$(date +%s.%N)
    say "step $n: $*"
    timeout "$to" "$@" > "$pd/steps/$n.txt" 2>&1 || rc=$?
    printf '%s %s %s\n' "$n" "$t0" "$(date +%s.%N)" >> "$pd/windows.txt"
    snap "$n"
    if [ "$rc" -ne 0 ]; then verdict=failed; reasons+=("step $n exited $rc"); tail -5 "$pd/steps/$n.txt" | tee -a "$log"; fi
}

if [ "$verdict" != failed ]; then
    say "ready after ${waited}s"
    snap ready
    P=$SPEED_PORT
    step ident    900  env P="$P" sh "$S/ident.sh"
    step cstep2   2400 env P="$P" LEVELS="1 2 4 8" MT="$SPEED_CSTEP2_MT" sh "$S/cstep2.sh"
    step fnpf-8k  1800 env P="$P" N=8000 CORPUS="$pd/fnpf-corpus.txt" sh "$S/fnpf.sh"
    step fnpf-24k 1800 env P="$P" N=24000 CORPUS="$pd/fnpf-corpus.txt" sh "$S/fnpf.sh"
    step fnconc   1800 env P="$P" C=4 DEC=300 PROMPT=24000 sh "$S/fnconc.sh"
    step ttft-c1  900  python3 "$here/ttft.py" --url "http://127.0.0.1:$P" --corpus "$pd/fnpf-corpus.txt" \
                       --tokens "$SPEED_TTFT_TOKENS" --conc 1 --warmup 6 --reps 9 --out "$pd/steps/ttft-c1.json"
    step ttft-c8  900  python3 "$here/ttft.py" --url "http://127.0.0.1:$P" --corpus "$pd/fnpf-corpus.txt" \
                       --tokens "$SPEED_TTFT_TOKENS" --conc 8 --warmup 1 --reps 4 --out "$pd/steps/ttft-c8.json"
fi
t_end=$(date '+%Y-%m-%d %H:%M:%S'); load_end=$(load1)
sha256sum "$pd/fnpf-corpus.txt" /dev/shm/fnconc-corpus.txt 2>/dev/null | sed 's/^/corpus_sha256: /' >> "$pd/provenance.txt" || true

# over the measurement: the mean CPU everything but the server used (cores), and the most processes
# outside the server that held the GPU in any 2 s sample
read -r foreign kfd_foreign < <(python3 -c '
import json, sys
t0, t1 = (float(x) for x in sys.argv[2:4])
r = [r for r in map(json.loads, open(sys.argv[1])) if "foreign_cores" in r and t0 <= r["t"] <= t1]
print(round(sum(x["foreign_cores"] for x in r) / len(r), 2) if r else "nan", max((x["kfd_foreign"] for x in r), default="nan"))
' "$pd/hostload.jsonl" "$(date -d "$t_start" +%s)" "$(date -d "$t_end" +%s)" 2>/dev/null || echo nan nan)

hits=$(journalctl -k --since "$t_start" --until "$t_end" --no-pager | grep -iE "$klog_pat" || true)
if [ "$verdict" != failed ] && [ -n "$hits" ]; then
    verdict=invalid; reasons+=("amdgpu kernel-log lines during the measurement: $(printf '%s' "$hits" | head -3 | tr '\n' '|')")
fi
if [ "$verdict" = valid ]; then
    [ "$kfd_foreign" = 0 ] || { verdict=contaminated; reasons+=("another GPU client during the measurement (up to $kfd_foreign process(es) outside the server)"); }
    gt "$load_start" "$SPEED_MAX_LOAD" && { verdict=contaminated; reasons+=("load1 $load_start at the start"); }
    gt "$load_end" "$SPEED_MAX_LOAD" && { [ "$foreign" = nan ] || gt "$foreign" "$SPEED_MAX_LOAD"; } \
        && { verdict=contaminated; reasons+=("load1 $load_end at the end with non-server CPU $foreign cores"); }
fi

stop_server
stop_hits=$(journalctl -k --since "$t_end" --no-pager | grep -iE "$klog_pat" || true)
stop_faults=$(printf '%s\n' "$stop_hits" | grep -ciE 'page fault' || true)
stop_fatal=$(printf '%s\n' "$stop_hits" | grep -iE "$klog_fatal" || true)

jq -n --arg label "$label" --arg verdict "$verdict" --arg image "$SPEED_IMAGE" --arg digest "$digest" \
      --arg model "$model" --arg plugins "${plugins:-}" --arg flags "$SPEED_FLAGS $SPEED_EXTRA_FLAGS" \
      --arg harness "$harness" --arg t_start "$t_start" --arg t_end "$t_end" \
      --arg load_start "$load_start" --arg load_end "$load_end" --arg foreign "$foreign" --arg kfd "$kfd_foreign" \
      --arg max_load "$SPEED_MAX_LOAD" --arg kv_bytes "${kv_bytes:-0}" --arg ttft_tokens "$SPEED_TTFT_TOKENS" \
      --arg stop_faults "$stop_faults" --arg stop_fatal "$stop_fatal" \
      --args '{label: $label, verdict: $verdict, reasons: $ARGS.positional, image: $image, digest: $digest,
               model: $model, plugin_home: $plugins, flags: $flags, harness: $harness,
               t_start_local: $t_start, t_end_local: $t_end, load_start: ($load_start|tonumber),
               load_end: ($load_end|tonumber), foreign_cores_mean: $foreign, kfd_foreign_max: $kfd,
               max_load: ($max_load|tonumber),
               disk_tier_bytes: ($kv_bytes|tonumber), ttft_tokens: ($ttft_tokens|tonumber),
               page_faults_at_stop: ($stop_faults|tonumber), kernel_errors_at_stop: $stop_fatal}' \
      "${reasons[@]+"${reasons[@]}"}" > "$pd/pass.json"
say "verdict $verdict${reasons[*]+: ${reasons[*]}}; load $load_start -> $load_end, non-server CPU $foreign cores; $stop_faults page-fault line(s) at stop"
[ -z "$stop_fatal" ] || { say "amdgpu errors while the server stopped -- stop all GPU work: $stop_fatal"; exit 6; }
case $verdict in valid) exit 0 ;; failed) exit 1 ;; contaminated) exit 4 ;; invalid) exit 5 ;; esac
