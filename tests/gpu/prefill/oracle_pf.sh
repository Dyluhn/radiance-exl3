#!/bin/sh
# oracle_pf.sh -- exl3r4d's M > 16 rows on REAL engine steps, the SHIPPED band split (decode rows M <= 16, the `_pf`
# rows above, no test build): P3's fixture converted through this repository's architecture, quantiser and kernel
# libraries, served on the card at a production-sized batched-token budget, with a prompt long enough that prefill
# chunks run at the budget. Checked by the engine's own instruments (tests/gpu/op_oracle.sh's, P4's):
#   --debug-graph       every band of both ops resolves to an exl3r4d row on the device -- none to a reference library
#                       -- and the band above 16 to the `_pf` rows;
#   RADIANCE_OP_ORACLE  every issue copied before and after the device launch and checked against exl3ref on the host;
#   RADIANCE_OP_REPEAT  every issue's device kernel run twice on the same bytes, byte-compared (R8);
#   the oracle must have judged issues of the `_pf` rows (else the long prompt did not reach them: FAIL, not pass).
#
#   tests/gpu/prefill/oracle_pf.sh <fixture bf16 dir> <fixture exl3 dir> <work dir> [max batched tokens (2048)]
#
# Run INSIDE a radiance build/runtime container with the card visible (tests/gpu/run_gpu.sh's session container:
# `docker exec -w /work "$EXL3_CONTAINER" sh tests/gpu/prefill/oracle_pf.sh ...`). ENV: RADIANCE_PREFIX (default
# /stage/opt/radiance), EXL3_HOME (this build's $RADIANCE_HOME, default build-gpu/radiance_home), ORACLE_MAX (500).
# Prints PASS/FAIL lines; exit 0 only when every check passed and checked something.
set -u
[ "$#" -ge 3 ] || { echo "usage: $0 <fixture bf16 dir> <fixture exl3 dir> <work dir> [max batched tokens]" >&2; exit 2; }
BF16=$(cd "$1" && pwd) SRC=$(cd "$2" && pwd)
mkdir -p "$3" && WORK=$(cd "$3" && pwd)
MAXTOK=${4:-2048}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
PREFIX=${RADIANCE_PREFIX:-/stage/opt/radiance}
HOME_LIST=${EXL3_HOME:-$REPO/build-gpu/radiance_home}:$PREFIX/share/radiance
K=exl3r4d,exl3ref,libr4d,libavx,libref   # --kernels loads ONLY what it names (STEP0 drift 1)
B=$PREFIX/bin
cd "$WORK" || exit 2
fail=0
say() { echo "$1 $2"; [ "$1" = PASS ] || fail=1; }

# Four short prompts (decode) and one of ~3000 tokens (prefill chunks of MAXTOK, then the remainder).
python3 - <<'EOF'
import json
long = " ".join(f"Item {i}: the river {['rose', 'fell', 'turned', 'slowed'][i % 4]} near the {i % 97}th mill." for i in range(260))
rows = [{"prompt": "The capital of France is", "source": "p0"}, {"prompt": "def fibonacci(n):", "source": "p1"},
        {"prompt": long, "source": "long"}, {"prompt": "Hello, my name is", "source": "p3"},
        {"prompt": "Water boils at", "source": "p6"}]
open("corpus.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
EOF
if [ ! -f ours.rad ]; then
    EXL3_SRC=$SRC RADIANCE_HOME=$HOME_LIST "$B/rad-convert" "$BF16" --recipe "$REPO/tests/arch/data/fixture-exl3.recipe" \
        --kernels $K -o ours.rad.tmp > convert.log 2>&1 && mv ours.rad.tmp ours.rad ||
        { echo "FAIL convert: $(grep -E '^E' convert.log | head -2 | tr '\n' ' ')"; exit 1; }
fi
SERVE="--model ours.rad --kernels $K --max-model-len 4096 --max-num-batched-tokens $MAXTOK"

# --debug-graph: the bucket table of both ops -- every DEVICE band line names an exl3r4d row, one of them a `_pf` row.
RADIANCE_HOME=$HOME_LIST "$B/radiance" $SERVE --debug-graph > graph.log 2>&1
awk '/^moe_exl3_(gate_up|down) /{on=1; print; next} on && /host domain:/{on=0} on && /->/{print}' graph.log > bands.txt
cat bands.txt
bad=$(grep -- '->' bands.txt | grep -vc 'exl3r4d_moe_')
pf=$(grep -- '->' bands.txt | grep -c '_pf (exl3r4d')
[ -s bands.txt ] && [ "$bad" -eq 0 ] && [ "$pf" -gt 0 ] && say PASS "debug-graph: every device band of both ops on exl3r4d, $pf band line(s) on the _pf rows" ||
    say FAIL "debug-graph: $bad device band line(s) not on exl3r4d, $pf on the _pf rows (bands.txt)"
ref=$(grep -E '^ +moe_exl3_(gate_up|down) +device' graph.log | grep -vc 'exl3r4d')
[ "$ref" -eq 0 ] && say PASS "debug-graph: no device instance of moe_exl3_* on a reference library" ||
    say FAIL "debug-graph: $ref device instance(s) of moe_exl3_* not on exl3r4d"

RADIANCE_OP_ORACLE=moe_exl3_gate_up,moe_exl3_down RADIANCE_OP_ORACLE_MAX=${ORACLE_MAX:-500} RADIANCE_HOME=$HOME_LIST \
    python3 "$REPO/tests/arch/complete2.py" c2-oracle.json corpus.jsonl -- "$B/radiance" $SERVE > oracle.out 2>&1 ||
    say FAIL "oracle run: $(tail -3 oracle.out | tr '\n' ' ')"
agree=$(grep -c 'agrees \[device exl3r4d' c2-oracle.json.log)
agree_pf=$(grep -c 'agrees \[device exl3r4d_moe_[a-z_]*_pf' c2-oracle.json.log)
nagree=$(grep -E "oracle #[0-9]+ op 'moe_exl3" c2-oracle.json.log | grep -vc 'agrees')
[ "$agree_pf" -gt 0 ] && [ "$nagree" -eq 0 ] && say PASS "oracle: $agree issues agree ($agree_pf on the _pf rows), 0 disagree or skipped" ||
    say FAIL "oracle: $agree agree ($agree_pf on _pf rows), $nagree not agreeing (c2-oracle.json.log)"
grep -E "oracle #[0-9]+ op 'moe_exl3" c2-oracle.json.log | grep -v agrees | head -5

RADIANCE_OP_REPEAT=moe_exl3_gate_up,moe_exl3_down RADIANCE_OP_REPEAT_MAX=${ORACLE_MAX:-500} \
RADIANCE_OP_REPEAT_MAXBYTES=4294967296 RADIANCE_HOME=$HOME_LIST \
    python3 "$REPO/tests/arch/complete2.py" c2-repeat.json corpus.jsonl -- "$B/radiance" $SERVE > repeat.out 2>&1 ||
    say FAIL "repeat run: $(tail -3 repeat.out | tr '\n' ' ')"
rep=$(grep -E 'op REPEAT: [0-9]+ issue' c2-repeat.json.log | tail -1)
echo "$rep"
echo "$rep" | grep -qE ': [1-9][0-9]* issue\(s\) run twice, 0 of them ANSWERED DIFFERENTLY; 0 more were NOT JUDGED' &&
    say PASS "repeat: device kernels a function of their inputs" || say FAIL "repeat: $rep"
cmp -s c2-oracle.json c2-repeat.json && say PASS "completions identical across the two server starts" ||
    say FAIL "completions differ across the two server starts (c2-oracle.json vs c2-repeat.json)"
exit $fail
