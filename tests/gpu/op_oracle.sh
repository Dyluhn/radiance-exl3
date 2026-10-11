#!/bin/sh
# op_oracle.sh -- exl3r4d checked on REAL engine steps: a fixture converted through this repository's architecture,
# quantiser and kernel libraries, served on the card, its two ops checked by the engine's own instruments:
#   RADIANCE_OP_ORACLE  every issue of moe_exl3_gate_up / moe_exl3_down copied before and after the device launch,
#                       the stored codes put back through exl3r4d's unrelayout, the host reference (exl3ref) run on
#                       the copy, every operand compared (core/runtime/oracle.cpp);
#   RADIANCE_OP_REPEAT  the same device kernel run twice on the same bytes, byte-compared (determinism, R8).
# Plus --debug-graph: which row serves each op at each band. Serving needs no reference kernel here: every band of
# both ops resolves to exl3r4d.
#
#   tests/gpu/op_oracle.sh <fixture bf16 dir> <fixture exl3 dir> <work dir>
#
# Run INSIDE a radiance build/runtime container with the card visible (tests/gpu/run_gpu.sh's session container:
# `docker exec -w /work "$EXL3_CONTAINER" sh tests/gpu/op_oracle.sh ...`). The fixture: tests/arch/fixture_variant.py
# (P3's, real Qwen3.8 tokenizer). ENV: RADIANCE_PREFIX (default /stage/opt/radiance), EXL3_HOME (this build's
# $RADIANCE_HOME, default build-gpu/radiance_home), ORACLE_MAX (checks, default 200).
# Prints PASS/FAIL lines; exit 0 when the oracle and the repeat found nothing and checked something.
set -u
[ "$#" -eq 3 ] || { echo "usage: $0 <fixture bf16 dir> <fixture exl3 dir> <work dir>" >&2; exit 2; }
BF16=$(cd "$1" && pwd) SRC=$(cd "$2" && pwd)
mkdir -p "$3" && WORK=$(cd "$3" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
PREFIX=${RADIANCE_PREFIX:-/stage/opt/radiance}
HOME_LIST=${EXL3_HOME:-$REPO/build-gpu/radiance_home}:$PREFIX/share/radiance
K=exl3r4d,exl3ref,libr4d,libavx,libref   # --kernels loads ONLY what it names (STEP0 drift 1)
B=$PREFIX/bin
cd "$WORK" || exit 2
cat > corpus.jsonl <<'EOF'
{"prompt": "The capital of France is", "source": "p0"}
{"prompt": "def fibonacci(n):", "source": "p1"}
{"prompt": "Hello, my name is", "source": "p3"}
{"prompt": "Water boils at", "source": "p6"}
EOF
fail=0
if [ ! -f ours.rad ]; then
    EXL3_SRC=$SRC RADIANCE_HOME=$HOME_LIST "$B/rad-convert" "$BF16" --recipe "$REPO/tests/arch/data/fixture-exl3.recipe" \
        --kernels $K -o ours.rad.tmp > convert.log 2>&1 && mv ours.rad.tmp ours.rad ||
        { echo "FAIL convert: $(grep -E '^E' convert.log | head -2 | tr '\n' ' ')"; exit 1; }
fi
# Every band must be exl3r4d's: with a band only the reference serves (M > 16, P5's), the planner puts the op on the
# host and its weights on the mapped tier, which cannot hold exl3r4d's re-laid codes ("its stored form is not the
# container's"), and qwen4exp refuses a batched-token budget below 64 (the delta net's chunk) -- both measured,
# notes/p4.md. So EXL3_HOME must hold a TEST build of exl3r4d with -DEXL3R4D_M_MAX=64, never the shipped one.
SERVE="--model ours.rad --kernels $K --max-model-len 4096 --max-num-batched-tokens 64"
RADIANCE_HOME=$HOME_LIST "$B/radiance" $SERVE --debug-graph > graph.log 2>&1
echo "debug-graph, our ops:"; grep -E 'moe_exl3|-> exl3' graph.log | head -12

RADIANCE_OP_ORACLE=moe_exl3_gate_up,moe_exl3_down RADIANCE_OP_ORACLE_MAX=${ORACLE_MAX:-200} RADIANCE_HOME=$HOME_LIST \
    python3 "$REPO/tests/arch/complete2.py" c2-oracle.json corpus.jsonl -- "$B/radiance" $SERVE > oracle.out 2>&1 ||
    { echo "FAIL oracle run: $(cat oracle.out)"; fail=1; }
RADIANCE_OP_REPEAT=moe_exl3_gate_up,moe_exl3_down RADIANCE_OP_REPEAT_MAX=${ORACLE_MAX:-200} RADIANCE_HOME=$HOME_LIST \
    python3 "$REPO/tests/arch/complete2.py" c2-repeat.json corpus.jsonl -- "$B/radiance" $SERVE > repeat.out 2>&1 ||
    { echo "FAIL repeat run: $(cat repeat.out)"; fail=1; }
echo "oracle log lines (moe_exl3):"; grep -iE 'oracle|repeat' c2-oracle.json.log | grep -iE 'moe_exl3|summary|checked|mismatch|skip|NOT' | head -20
echo "repeat log lines (moe_exl3):"; grep -iE 'oracle|repeat' c2-repeat.json.log | grep -iE 'moe_exl3|summary|checked|differ|skip|NOT' | head -20
exit $fail
