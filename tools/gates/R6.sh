#!/bin/sh
# R6 -- no allocation, getenv or synchronisation on the launch path.
# Static: `grep -nE 'hipMalloc|getenv|hipDeviceSynchronize|hipStreamSynchronize'` over every source compiled into
# the kernel libraries -- kernels/exl3r4d/*, kernels/exl3ref/*, and common/*.c which both compile in -- is empty.
# Control: the same pattern on a planted line must hit.
# Runtime (pre-registered 2026-10-11): no first-launch spike = the first decode step after a warm request is no slower
# than that request's steady-state median decode step + 1 ms. RADIANCE_LOG_STEPS lines carry no time at 1.3.0
# (core/engine.cpp:644-653), so the log is the server's taken with per-line timestamps (`docker logs -t`): a step lasts
# from its line to the next line's. Requests are told apart by their prefill steps; the first request is the warm one
# and is left out; the next request's decode steps are judged (its last step, which has no next line of its own, is
# left out too).
#   R6_STEPLOG   that log, of the EXL3 container served with RADIANCE_LOG_STEPS=1 (P6; the speed harness's server)
GATE=R6
. "$(dirname "$0")/lib.sh"
PAT='hipMalloc|getenv|hipDeviceSynchronize|hipStreamSynchronize'
if [ "$(echo '    hipMalloc(&p, n);' | grep -cE "$PAT")" -ne 1 ]; then part FAIL "control" "the pattern misses a planted hipMalloc"; fi
for lib in exl3r4d exl3ref; do
    d=$REPO/kernels/$lib
    [ -d "$d" ] || { part SKIP "$lib static" "no kernels/$lib in this checkout"; continue; }
    files=$(cd "$REPO" && ls kernels/$lib/*.c kernels/$lib/*.cc kernels/$lib/*.cpp kernels/$lib/*.h kernels/$lib/*.hip common/*.c common/*.h 2>/dev/null)
    hits=$(cd "$REPO" && grep -nE "$PAT" $files)
    if [ -n "$hits" ]; then part FAIL "$lib static" "$(printf '%s' "$hits" | head -5 | tr '\n' ' ')"
    else part PASS "$lib static" "0 hits in $(echo $files | wc -w) files: $(echo $files | tr '\n' ' ')"; fi
done
if [ -f "${R6_STEPLOG:-}" ]; then
    r=$(python3 - "$R6_STEPLOG" <<'PY'
import re, statistics, sys
from datetime import datetime
steps = [(datetime.fromisoformat(m.group(1)[:26]), m.group(2)) for m in
         (re.match(r"^(\S+) I step \d+ (\w+) ", l) for l in open(sys.argv[1], errors="replace")) if m]
reqs, cur = [], None
for (t, phase), (t2, _) in zip(steps, steps[1:]):
    if phase == "prefill":
        cur = []
        reqs.append(cur)
    elif phase == "decode" and cur is not None:
        cur.append((t2 - t).total_seconds() * 1e3)
reqs = [r for r in reqs if r]
if len(reqs) < 2 or len(reqs[1]) < 9:
    print(f"SKIP|{len(reqs)} request(s) with decode steps; need a warm one and a judged one of >= 9 steps (RADIANCE_LOG_STEPS=1, docker logs -t)")
    sys.exit()
d = reqs[1][:-1]
first, med = d[0], statistics.median(d[1:])
print(("PASS" if first <= med + 1.0 else "FAIL") + f"|first decode step after the warm request {first:.2f} ms vs steady median {med:.2f} ms + 1 ms ({len(d)} steps)")
PY
) || r="FAIL|the step-log reader failed"
    part "${r%%|*}" "first-launch spike (RADIANCE_LOG_STEPS)" "${r#*|}"
else
    part SKIP "first-launch spike (RADIANCE_LOG_STEPS)" "needs R6_STEPLOG: \`docker logs -t\` of the EXL3 container's server run with RADIANCE_LOG_STEPS=1 (P6)"
fi
finish "pattern '$PAT' over the kernel libraries' sources"
