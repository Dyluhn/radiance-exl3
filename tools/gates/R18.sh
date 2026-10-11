#!/bin/sh
# R18 -- tensor parallel: at --tp 2 no rank sits idle in our kernels, and both parity classes are hit on both ranks.
# (1) rskew.sh (radiance-isa: per-kernel device time per agent from a trace.sh CSV) over a tp2 decode trace of the
#     EXL3 container: every exl3r4d kernel (exl3r4d_gemm_kernel, _prep_kernel, _epi_kernel) that ran has device time
#     on BOTH agents, and the gap is printed (the reading, not judged: D13 says the limp is tuned after parity).
# (2) both parity classes on both ranks: NO INSTRUMENT at radiance 1.3.0 -- /stats' `experts` carries routed totals,
#     not per rank and parity class, and our ops take both classes in one launch, so a kernel trace cannot tell
#     them apart either. SKIPPED by name until one is chosen (notes/gates.md, open question).
#   R18_TRACE=<trace.sh CSV, tp2 decode> RSKEW=<radiance-isa rskew.sh> tools/gates/R18.sh
GATE=R18
. "$(dirname "$0")/lib.sh"
if [ -f "${R18_TRACE:-}" ] && [ -f "${RSKEW:-}" ]; then
    sh "$RSKEW" "$R18_TRACE" > "$GATES_WORK/logs/r18-rskew.out" 2>&1
    rows=$(grep -E '^exl3r4d_' "$GATES_WORK/logs/r18-rskew.out")
    idle=$(printf '%s\n' "$rows" | awk 'NF >= 5 && ($2 == 0 || $3 == 0)')
    if [ -z "$rows" ]; then part FAIL "no rank idle in exl3r4d" "no exl3r4d kernel over rskew's 20 ms floor in the trace"
    elif [ -n "$idle" ]; then part FAIL "no rank idle in exl3r4d" "a rank with no time: $idle"
    else part PASS "no rank idle in exl3r4d" "$(printf '%s\n' "$rows" | awk '{printf "%s %s/%s ms (gap %s); ", $1, $2, $3, $5}')"; fi
else
    part SKIP "no rank idle in exl3r4d" "needs R18_TRACE (trace.sh CSV of a tp2 decode of the EXL3 container) and RSKEW (radiance-isa rskew.sh)"
fi
part SKIP "both parity classes on both ranks" "no instrument at radiance $PIN_VERSION (per-rank, per-class route counts are not exposed); open question in notes/gates.md"
finish "tp2: our kernels on both ranks"
