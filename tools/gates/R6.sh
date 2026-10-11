#!/bin/sh
# R6 -- no allocation, getenv or synchronisation on the launch path.
# Static: `grep -nE 'hipMalloc|getenv|hipDeviceSynchronize|hipStreamSynchronize'` over every source compiled into
# the kernel libraries -- kernels/exl3r4d/*, kernels/exl3ref/*, and common/*.c which both compile in -- is empty.
# Control: the same pattern on a planted line must hit.
# Runtime: a server run with RADIANCE_LOG_STEPS=1, its log taken with per-line timestamps (`docker logs -t`), shows
# no first-launch spike: the first decode step's excess over the median decode step is no larger for the EXL3
# container than for the baseline's, plus R6_TOL_MS (default 1 ms; PROPOSED, not pre-registered -- notes/gates.md).
#   R6_STEPLOG       that log for the EXL3 container      R6_STEPLOG_BASE   the same for the baseline
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
if [ -f "${R6_STEPLOG:-}" ] && [ -f "${R6_STEPLOG_BASE:-}" ]; then
    r=$(python3 - "$R6_STEPLOG" "$R6_STEPLOG_BASE" "${R6_TOL_MS:-1}" <<'PY'
import re, statistics, sys
from datetime import datetime
def excess(path):   # (first decode step ms, median decode step ms) from `docker logs -t` lines
    t = [(datetime.fromisoformat(m.group(1)[:26]), m.group(2)) for m in
         (re.match(r"^(\S+) I step \d+ (\w+) ", l) for l in open(path, errors="replace")) if m]
    d = [(b[0] - a[0]).total_seconds() * 1e3 for a, b in zip(t, t[1:]) if a[1] == "decode"]
    if len(d) < 8:
        sys.exit(f"{path}: {len(d)} timed decode steps (need RADIANCE_LOG_STEPS=1 and docker logs -t)")
    return d[0], statistics.median(d)
(f1, m1), (f0, m0), tol = excess(sys.argv[1]), excess(sys.argv[2]), float(sys.argv[3])
ok = f1 - m1 <= (f0 - m0) + tol
print(f"{'PASS' if ok else 'FAIL'} EXL3 first decode step {f1:.2f} ms vs median {m1:.2f} (excess {f1 - m1:.2f}); "
      f"baseline {f0:.2f} vs {m0:.2f} (excess {f0 - m0:.2f}); tolerance {tol} ms")
PY
) || r="FAIL $r"
    part "${r%% *}" "first-launch spike (RADIANCE_LOG_STEPS)" "${r#* }"
else
    part SKIP "first-launch spike (RADIANCE_LOG_STEPS)" "needs a served EXL3 container and the baseline: R6_STEPLOG and R6_STEPLOG_BASE (docker logs -t of servers run with RADIANCE_LOG_STEPS=1)"
fi
finish "pattern '$PAT' over the kernel libraries' sources"
