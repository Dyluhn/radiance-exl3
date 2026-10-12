#!/bin/sh
# oracle_gd.sh -- interface v5's moe_exl3_gate_up_down checked on REAL engine steps, through radiance's own dispatch
# (its kernarg slot included: the harness and perfmoe launch with hipLaunchKernel and cannot see that limit). P3's
# fixture converted through this repository's libraries, served with a GD build of the architecture
# (-DQWEN4EXP_EXL3_GD=1, tests/arch's gd_home) and exl3r4d, short prompts (passes of 1..10 rows: the fused launch at
# >= 3 tokens, the separate kernels below), at --tp 1 or 2:
#   RADIANCE_OP_ORACLE  every issue of the four EXL3 ops copied before and after the device launch and re-run on exl3ref
#                       on the host (core/runtime/oracle.cpp); every one must agree, and gate_up_down must have been
#                       judged at least once at >= 3 tokens and once below;
#   RADIANCE_OP_REPEAT  every issue's device kernel run twice on the same bytes, byte-compared (R8).
#
#   tests/gpu/oracle_gd.sh <fixture bf16 dir> <fixture exl3 dir> <work dir> [tp (1)]
#
# Run INSIDE the radiance build/runtime container with the card(s) visible (tests/gpu/run_gpu.sh's session).
# ENV: RADIANCE_PREFIX (/stage/opt/radiance), EXL3_HOME (a $RADIANCE_HOME with the GD arch, exl3r4d, exl3ref and
# exl3_quant), ORACLE_MAX (500). Prints PASS/FAIL lines; exit 0 only when every check passed and checked something.
set -u
[ "$#" -ge 3 ] || { echo "usage: $0 <fixture bf16 dir> <fixture exl3 dir> <work dir> [tp]" >&2; exit 2; }
BF16=$(cd "$1" && pwd) SRC=$(cd "$2" && pwd)
mkdir -p "$3" && WORK=$(cd "$3" && pwd)
TP=${4:-1}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
PREFIX=${RADIANCE_PREFIX:-/stage/opt/radiance}
HOME_LIST=${EXL3_HOME:?a home with the GD arch}:$PREFIX/share/radiance
K=exl3r4d,exl3ref,libr4d,libavx,libref
B=$PREFIX/bin
OPS=moe_exl3_gate_up_down,moe_exl3_gate_up,moe_exl3_down,moe_exl3_prep_gu
cd "$WORK" || exit 2
fail=0
say() { echo "$1 $2"; [ "$1" = PASS ] || fail=1; }
cat > corpus.jsonl <<'EOT'
{"prompt": "The capital of France is", "source": "p0"}
{"prompt": "def fibonacci(n):", "source": "p1"}
{"prompt": "Hi", "source": "p2"}
{"prompt": "Hello, my name is", "source": "p3"}
{"prompt": "Water boils at", "source": "p6"}
EOT
if [ ! -f ours.rad ]; then
    EXL3_SRC=$SRC RADIANCE_HOME=$HOME_LIST "$B/rad-convert" "$BF16" --recipe "$REPO/tests/arch/data/fixture-exl3.recipe" \
        --kernels $K -o ours.rad.tmp > convert.log 2>&1 && mv ours.rad.tmp ours.rad ||
        { echo "FAIL convert: $(grep -E '^E' convert.log | head -2 | tr '\n' ' ')"; exit 1; }
fi
SERVE="--model ours.rad --kernels $K --tp $TP --max-model-len 4096 --num-speculative-tokens 0"
RADIANCE_HOME=$HOME_LIST "$B/radiance" $SERVE --debug-graph > graph.log 2>&1
gd=$(grep -E '^ *moe_exl3_gate_up_down' graph.log | head -3)
echo "debug-graph: $gd"
grep -qE 'moe_exl3_gate_up_down.*exl3r4d' graph.log && say PASS "debug-graph: moe_exl3_gate_up_down declared, on exl3r4d" ||
    say FAIL "debug-graph: no gate_up_down on exl3r4d (a GD build of the arch? graph.log)"
RADIANCE_OP_ORACLE=$OPS RADIANCE_OP_ORACLE_MAX=${ORACLE_MAX:-500} RADIANCE_HOME=$HOME_LIST \
    python3 "$REPO/tests/arch/complete2.py" c2-oracle.json corpus.jsonl -- "$B/radiance" $SERVE > oracle.out 2>&1 ||
    say FAIL "oracle run: $(tail -3 oracle.out | tr '\n' ' ')"
cat c2-oracle.json.log* 2>/dev/null > oracle.all
agree=$(grep -c "oracle #[0-9]* op 'moe_exl3_[a-z_]*' agrees" oracle.all)
nagree=$(grep -E "oracle #[0-9]+ op 'moe_exl3" oracle.all | grep -vc 'agrees')
# gate_up_down's issues by rows: opd0 is a [M, K] -- M >= 3 took the fused launch, below the separate kernels
awk "/oracle #[0-9]+ op 'moe_exl3_gate_up_down' agrees/{g=1; next} g && /^  opd0 /{split(\$3, d, /[][,]/); if (d[2] + 0 >= 3) f++; else s++; g=0}
     END{print f + 0, s + 0}" oracle.all > gd.counts
read gdf gds < gd.counts
[ "$nagree" -eq 0 ] && [ "$gdf" -gt 0 ] && [ "$gds" -gt 0 ] &&
    say PASS "oracle (tp $TP): $agree issues agree ($gdf gate_up_down at >= 3 tokens, $gds below), 0 disagree or skipped" ||
    say FAIL "oracle (tp $TP): $agree agree ($gdf gate_up_down at >= 3 tokens, $gds below), $nagree not agreeing"
grep -E "oracle #[0-9]+ op 'moe_exl3" oracle.all | sed "s/oracle #[0-9]* //" | sort | uniq -c | head -8
grep -E "oracle #[0-9]+ op 'moe_exl3" oracle.all | grep -v agrees | head -5
RADIANCE_OP_REPEAT=$OPS RADIANCE_OP_REPEAT_MAX=${ORACLE_MAX:-500} RADIANCE_OP_REPEAT_MAXBYTES=4294967296 RADIANCE_HOME=$HOME_LIST \
    python3 "$REPO/tests/arch/complete2.py" c2-repeat.json corpus.jsonl -- "$B/radiance" $SERVE > repeat.out 2>&1 ||
    say FAIL "repeat run: $(tail -3 repeat.out | tr '\n' ' ')"
rep=$(cat c2-repeat.json.log* 2>/dev/null | grep -E 'op REPEAT: [0-9]+ issue' | tail -1)
echo "$rep"
echo "$rep" | grep -qE ': [1-9][0-9]* issue\(s\) run twice, 0 of them ANSWERED DIFFERENTLY; 0 more were NOT JUDGED' &&
    say PASS "repeat (tp $TP): device kernels a function of their inputs" || say FAIL "repeat (tp $TP): $rep"
cmp -s c2-oracle.json c2-repeat.json && say PASS "completions identical across the two server starts" ||
    say FAIL "completions differ across the two server starts"
exit $fail
