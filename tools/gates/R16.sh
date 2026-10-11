#!/bin/sh
# R16 -- prefill, by the rule pre-registered 2026-10-11: prefill tok/s at 8K and 24K (chunk 2048; baseline 2,361 /
# 2,433, spread 1.00% / 0.81%) and fnconc's prefill tok/s beside 4 decoders (2,180, 0.46%), EXL3 median inside the
# baseline's [min, max] or better. fnconc's decoders' slowdown % is printed, not judged (36-point spread).
#   SPEED_BASE_PASSES="<dirs>" SPEED_EXL3_PASSES="<dirs>" tools/gates/R16.sh
GATE=R16
. "$(dirname "$0")/lib.sh"
speed_summaries > "$GATES_WORK/logs/$GATE-speed.why" 2>&1 || { part SKIP "passes" "$(cat "$GATES_WORK/logs/$GATE-speed.why")"; finish "nothing checked"; }
speed_part "prefill tok/s 8K" higher
speed_part "prefill tok/s 24K" higher
speed_part "fnconc prefill tok/s beside 4 decoders" higher --beside "fnconc decoders' slowdown %"
finish "prefill tok/s, EXL3 vs baseline"
