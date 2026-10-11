#!/bin/sh
# R17 -- TTFT p50 at C = 1 and C = 8 (tools/gates/speed/ttft.py: 256-token prompts; C=1 6 warm + 9, C=8 1 warm burst
# + 4; the first request after start discarded), EXL3 median inside the baseline's [min, max] or better.
# Baseline (6 passes, base-p6..p11): C=1 108.6 ms (3.2%), C=8 837.0 ms (0.57%).
#   SPEED_BASE_PASSES="<dirs>" SPEED_EXL3_PASSES="<dirs>" tools/gates/R17.sh
GATE=R17
. "$(dirname "$0")/lib.sh"
speed_summaries > "$GATES_WORK/logs/$GATE-speed.why" 2>&1 || { part SKIP "passes" "$(cat "$GATES_WORK/logs/$GATE-speed.why")"; finish "nothing checked"; }
speed_part "TTFT p50 ms C=1" lower
speed_part "TTFT p50 ms C=8" lower
finish "TTFT p50, EXL3 vs baseline"
