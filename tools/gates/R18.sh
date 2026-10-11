#!/bin/sh
# R18 -- tensor parallel: at --tp 2 no rank sits idle in our ops, and both parity classes are present on both ranks.
#
# THE INSTRUMENT (orchestrator 2026-10-11: rad_route_report if it gives per-expert counts per rank, else the per-rank
# operand shapes of the RADIANCE_OP_ORACLE log). rad_route_report does NOT: it is the architecture handing the core
# its routing for the residency mover (core/runtime/ctx.cpp:637-680); nothing per rank is printed or put in /stats
# (`experts` carries totals). So this gate reads the OP_ORACLE log of a tp2 run (rank-tagged lines "W[r] oracle #n
# op 'moe_exl3_…' agrees … {N=… K=… N_odd=… K_odd=…}"):
#   1. every rank issued both ops and every issue agrees with exl3ref (no rank idle in our ops, none wrong);
#   2. both parity classes on both ranks: each rank's gate_up carries both class widths (N, N_odd > 0) and down both
#      (K, K_odd > 0), and the ranks' widths are the parity split's -- rank 0's (N, N_odd) is rank 1's swapped
#      (384/256 at the real width: 768/512 and 512/768), likewise down's (K, K_odd).
# What it does NOT show: how many tokens each class received on each rank (no per-rank route count exists at 1.3.0).
# (optional) rskew.sh over a trace.sh CSV of the same run prints each exl3r4d kernel's time per agent (the reading).
#
#   R18_ORACLE_LOG=<engine log of a --tp 2 run with RADIANCE_OP_ORACLE=moe_exl3_gate_up,moe_exl3_down>
#   [R18_TRACE=<trace.sh CSV> RSKEW=<radiance-isa rskew.sh>] tools/gates/R18.sh
GATE=R18
. "$(dirname "$0")/lib.sh"
if [ -f "${R18_ORACLE_LOG:-}" ]; then
    r=$(python3 - "$R18_ORACLE_LOG" <<'PY'
import re, sys
pat = re.compile(r"^W\[(\d+)\] oracle #\d+ op 'moe_exl3_(gate_up|down)' (\w+) .*\{([^}]*)\}")
seen, bad = {}, 0
for line in open(sys.argv[1], errors="replace"):
    m = pat.match(line)
    if not m:
        continue
    rank, op, verdict, params = int(m.group(1)), m.group(2), m.group(3), dict(kv.split("=", 1) for kv in m.group(4).split())
    bad += verdict != "agrees"
    w = (int(params["N"]), int(params["N_odd"])) if op == "gate_up" else (int(params["K"]), int(params["K_odd"]))
    seen.setdefault(rank, {}).setdefault(op, set()).add(w)
ranks = sorted(seen)
idle = [r for r in ranks if set(seen[r]) != {"gate_up", "down"}]
print(("PASS" if len(ranks) >= 2 and not idle and bad == 0 else "FAIL") +
      f"|ranks {ranks}, each issuing both ops: {not idle}; issues not agreeing: {bad}")
ok, why = len(ranks) == 2, []
for op in ("gate_up", "down"):
    w0, w1 = seen.get(0, {}).get(op, set()), seen.get(1, {}).get(op, set())
    both = all(a > 0 and b > 0 for a, b in w0 | w1)
    swapped = len(w0) == 1 and len(w1) == 1 and next(iter(w0)) == tuple(reversed(next(iter(w1)))) and len(set(next(iter(w0)))) == 2
    ok &= both and swapped
    why.append(f"{op} rank0 {sorted(w0)} rank1 {sorted(w1)}")
print(("PASS" if ok else "FAIL") + "|" + "; ".join(why) + " (widths per rank, not routed counts)")
PY
) || r="FAIL|the oracle-log reader failed"
    l1=$(printf '%s\n' "$r" | head -1) l2=$(printf '%s\n' "$r" | tail -1)
    part "${l1%%|*}" "no rank idle in our ops (OP_ORACLE, tp2)" "${l1#*|}"
    part "${l2%%|*}" "both parity classes on both ranks (operand widths)" "${l2#*|}"
else
    part SKIP "no rank idle in our ops (OP_ORACLE, tp2)" "needs R18_ORACLE_LOG: a --tp 2 run on both cards with RADIANCE_OP_ORACLE set (the orchestrator's two-card window)"
    part SKIP "both parity classes on both ranks (operand widths)" "same log"
fi
if [ -f "${R18_TRACE:-}" ] && [ -f "${RSKEW:-}" ]; then
    echo "  reading (rskew.sh, not judged; D13: the limp is tuned after parity):"
    sh "$RSKEW" "$R18_TRACE" | grep -E '^(kernel|exl3r4d_)' | sed 's/^/    /'
fi
finish "tp2: our ops on both ranks, both parity classes on each"
