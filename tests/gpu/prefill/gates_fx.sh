#!/bin/sh
# gates_fx.sh -- the serving gates P5 owns, at FIXTURE level (the real model's are P6's): one server of the fixture
# container, this build's plugins, the shipped band split, MTP depth 3, a 2048 batched-token budget and the tier flags
# radiance's tiercross.sh asks for; driven from the HOST with radiance's own scripts (they need curl and jq, which the
# build image lacks):
#   R19  scripts/tiercross.sh   two sessions in flight = each alone (its own refusals are failures of the instrument)
#   R15  scripts/cstep2.sh      LEVELS 1 8 32: decode ms/step and tokens/step at C = 1, 8, 32 (M = 4, 32, 128 with
#                               MTP depth 3: the decode band, the run form, the map form) -- a reading, no verdict
#   R16  scripts/fnconc.sh      a long prefill beside 4 decoders: both halves -- a reading
#   R6   RADIANCE_LOG_STEPS=1 + `docker logs -t`: the first decode step after warm <= the steady median + 1 ms
#
#   tests/gpu/prefill/gates_fx.sh <ours.rad dir> <out dir> [port 8131]
#
# Run as tests/gpu/run_gpu.sh's host command: the card is the session container's (its HIP_VISIBLE_DEVICES); the
# server is a second container of the same image with the port published on 127.0.0.1. RADIANCE_SRC names the
# radiance 1.3.0 source (scripts/); EXL3_HOME this build's plugin home inside /work (default build-gpu/radiance_home).
set -u
[ "$#" -ge 2 ] || { echo "usage: $0 <ours.rad dir> <out dir> [port]" >&2; exit 2; }
repo=$(cd "$(dirname "$0")/../../.." && pwd)
rad=$(cd "$1" && pwd); out=$2; P=${3:-8131}
mkdir -p "$out"; out=$(cd "$out" && pwd)
src=${RADIANCE_SRC:?RADIANCE_SRC names the radiance 1.3.0 source}
image=${EXL3_IMAGE:-radiance-build:1.3.0}
home=${EXL3_HOME:-build-gpu/radiance_home}
idx=$(docker exec "$EXL3_CONTAINER" printenv HIP_VISIBLE_DEVICES) || { echo "FAIL no session container"; exit 2; }
w() { echo "/work${1#$repo}"; }
name=exl3-fxsrv-$$
rm -rf "$rad/kvcache"; mkdir -p "$rad/kvcache"
fail=0
say() { echo "$1 $2"; [ "$1" = PASS ] || [ "$1" = READ ] || fail=1; }

docker run -d --name "$name" --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
    --security-opt label=disable -e "HIP_VISIBLE_DEVICES=$idx" -e RADIANCE_LOG_STEPS=1 \
    -e "RADIANCE_HOME=/work/$home:/stage/opt/radiance/share/radiance" -p "127.0.0.1:$P:$P" \
    -v "$repo":/work -w "$(w "$rad")" "$image" /stage/opt/radiance/bin/radiance --model ours.rad \
    --kernels exl3r4d,exl3ref,libr4d,libavx,libref --host 0.0.0.0 --port "$P" --max-model-len 65536 \
    --max-num-batched-tokens 2048 --num-speculative-tokens 3 --checkpoint-slots 2 --prefix-cache-host-mib 512 \
    --prefix-cache-disk-mib 16384 --prefix-cache-dir "$(w "$rad")/kvcache" > /dev/null
trap 'docker rm -f "$name" > /dev/null 2>&1' EXIT
i=0; until curl -sf "http://127.0.0.1:$P/health" > /dev/null; do
    i=$((i + 1)); [ $i -gt 300 ] && { say FAIL "server never healthy"; docker logs "$name" 2>&1 | tail -5; exit 1; }
    sleep 1
done
echo "server healthy after ${i} s"

H=127.0.0.1 P=$P sh "$src/scripts/tiercross.sh" > "$out/tiercross.out" 2>&1; rc=$?
oks=$(grep -cE '^  ok ' "$out/tiercross.out"); fails=$(grep -cE '^  FAIL' "$out/tiercross.out")
[ $rc -eq 0 ] && [ "$fails" -eq 0 ] && [ "$oks" -gt 0 ] && say PASS "R19 tiercross: $oks comparisons ok, 0 FAIL" ||
    say FAIL "R19 tiercross: rc=$rc, $oks ok, $fails FAIL: $(grep -E 'FAIL|tiercross:' "$out/tiercross.out" | head -3 | tr '\n' ' ')"

P=$P LEVELS="1 8 32" MT=${CSTEP2_MT:-400} sh "$src/scripts/cstep2.sh" > "$out/cstep2.out" 2>&1
say READ "cstep2 C=1,8,32: $(tr '\n' ';' < "$out/cstep2.out")"
P=$P C=4 DEC=${FNCONC_DEC:-200} PROMPT=${FNCONC_PROMPT:-12000} ROOT="$src" sh "$src/scripts/fnconc.sh" > "$out/fnconc.out" 2>&1
say READ "fnconc: $(tail -6 "$out/fnconc.out" | tr '\n' ';')"

docker stop -t 60 "$name" > /dev/null
docker logs -t "$name" > "$out/server.log" 2>&1
python3 - "$out/server.log" > "$out/r6.out" <<'EOF'
import re, statistics, sys
from datetime import datetime
# A step's time = the gap to the next step line (RADIANCE_LOG_STEPS=1, `docker logs -t`). The first decode step is
# compared with the median of the decode steps of the SAME shape (n_seq, n_tok): a step at another concurrency is not
# its steady state.
t = [(datetime.fromisoformat(m.group(1)[:26]), m.group(2), m.group(3)) for m in
     (re.match(r"^(\S+) I step \d+ (\w+) (n_seq=\d+ n_tok=\d+)", l) for l in open(sys.argv[1], errors="replace")) if m]
d = [((b[0] - a[0]).total_seconds() * 1e3, a[2]) for a, b in zip(t, t[1:]) if a[1] == "decode"]
if len(d) < 8:
    print(f"FAIL {len(d)} timed decode steps (RADIANCE_LOG_STEPS=1 and docker logs -t needed)"); sys.exit()
first, shape = d[0]
same = [x for x, s in d if s == shape]
med = statistics.median(same)
print(f"{'PASS' if first <= med + 1.0 else 'FAIL'} first decode step {first:.2f} ms ({shape}) vs the median of "
      f"{len(same)} steps of that shape {med:.2f} ms (+{first - med:.2f}; bound +1 ms)")
EOF
r=$(cat "$out/r6.out"); say "${r%% *}" "R6 first-launch spike: ${r#* }"
exit $fail
