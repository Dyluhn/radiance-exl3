#!/bin/sh
# R15 -- decode: cstep2 ms/step at C = 1, 2, 4, 8, the EXL3 container against the baseline, by the rule
# pre-registered 2026-10-11: >= 5 run_pass.sh passes per arm, interleaved ABAB with fresh servers; each C judged on
# the levels summarize.py does NOT flag (STALLED / SPLIT); EXL3 unflagged median inside the baseline's unflagged
# [min, max] (or faster: tools/gates/speed_verdict.py); at C = 8 tokens/step and MTP acceptance printed beside it.
# Baseline floors (8 passes): C=1 15.35 [15.34, 15.39], C=2 18.48, C=4 22.96, C=8 30.77 [29.93, 31.17] ms/step.
#   SPEED_BASE_PASSES="<dirs>" SPEED_EXL3_PASSES="<dirs>" tools/gates/R15.sh     (or SPEED_*_SUMMARY json files)
# Interleaving is the caller's (run_pass.sh labels); this gate checks counts, verdicts and flags only.
GATE=R15
. "$(dirname "$0")/lib.sh"
speed_summaries > "$GATES_WORK/logs/$GATE-speed.why" 2>&1 || { part SKIP "passes" "$(cat "$GATES_WORK/logs/$GATE-speed.why")"; finish "nothing checked"; }
for c in 1 2 4; do speed_part "decode ms/step C=$c" lower; done
speed_part "decode ms/step C=8" lower --beside "tokens/step C=8" "MTP accept % C=8"
finish "cstep2 ms/step, EXL3 vs baseline, unflagged levels"
