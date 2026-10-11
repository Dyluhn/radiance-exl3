#!/bin/sh
# R22 -- the kernel ceiling, a STOP criterion for P4/P5, NOT a ship gate: gate_up decode dispatch weight-stream
# GB/s >= the in-tree hot number on the same bytes (libr4d's moe_gemm_w4a8 in the same harness), or >= 92% of the
# calibrated stream ceiling of the card it ran on (S0.11: 679.5 GB/s on the measured card -> 625 GB/s; a machine
# fact, so an argument here, never a default).
#   R22_PERF=<tools/perfmoe output, both libraries> R22_CEILING_GBS=<this card's measured stream GB/s> tools/gates/R22.sh
# R22_PERF: perfmoe's output (tools/perfmoe.cpp, P4): each "perfmoe:" block (one routing, one pool size) judged on
# its own "RESULT <lib> gate_up tp1 M=1 ... <GB/s> GB/s" lines; every block must pass.
GATE=R22
. "$(dirname "$0")/lib.sh"
if [ ! -f "${R22_PERF:-}" ] || [ -z "${R22_CEILING_GBS:-}" ]; then
    part SKIP "gate_up decode GB/s" "needs R22_PERF (perfmoe output, exl3r4d and libr4d on the same routing) and R22_CEILING_GBS (the card's calibrated stream ceiling)"
    finish "STOP criterion, not a ship gate"
fi
python3 - "$R22_PERF" "$R22_CEILING_GBS" > "$GATES_WORK/logs/r22.txt" <<'PY'
import re, sys
blocks, cur = [], None
for line in open(sys.argv[1], errors="replace"):
    if line.startswith("perfmoe: device"):
        pools = re.search(r"\d+ layers? over \d+ pools?", line)
        cur = {"what": pools.group(0) if pools else line.strip()[:60]}
        blocks.append(cur)
    m = re.match(r"RESULT\s+(exl3r4d|libr4d)\S*\s+(?:\S+\s+)?gate_up tp1 M=1\s.*?([\d.]+) GB/s", line)
    if m and cur is not None:
        cur[m.group(1)] = float(m.group(2))
line = 0.92 * float(sys.argv[2])
judged = [b for b in blocks if "exl3r4d" in b and "libr4d" in b]
if not judged:
    print("FAIL|no perfmoe block with tp1 gate_up M=1 for both exl3r4d and libr4d")
for b in judged:
    ok = b["exl3r4d"] >= b["libr4d"] or b["exl3r4d"] >= line
    print(("PASS" if ok else "FAIL") + "|%s|exl3r4d %.0f GB/s vs libr4d %.0f on the same bytes; 92%% line %.0f" % (
        b["what"], b["exl3r4d"], b["libr4d"], line))
PY
while IFS='|' read -r state what detail; do part "$state" "gate_up tp1 M=1: $what" "$detail"; done < "$GATES_WORK/logs/r22.txt"
finish "STOP criterion, not a ship gate"
