#!/bin/sh
# tp2_pf.sh -- exl3r4d at --tp 2 on the fixture, the SHIPPED band split, a 2048 batched-token budget and a prompt long
# enough for full prefill chunks: the per-rank half of R2 / R18 / R13 at fixture level.
#   1 --debug-graph: every device band of both ops on exl3r4d, the band above 16 on the `_pf` rows, and each rank's
#     parity widths (D7's 384/256 split of the fixture's 640: rank 0 gate_up N=768 N_odd=512, rank 1 the reverse);
#   2 RADIANCE_OP_ORACLE on both ranks: every issue agrees with exl3ref; per rank, issues at M > 16 judged; per rank,
#     BOTH PARITY CLASSES EXERCISED: an issue whose expert_offset shows a non-empty run for an even expert (class 0)
#     and for an odd one (class 1) -- the oracle prints the table's first four entries, experts 0..2 -- and the class
#     tables' shapes it was handed (rank-specific rows);
#   3 RADIANCE_OP_REPEAT on both ranks: 0 differ;
#   4 the engine's per-op device timing (--profile-ops) of the same workload: both ops issued on the DEVICE site and
#     timed on BOTH ranks (no rank doing the work alone; rskew's reading -- rocprofv3 cannot trace this engine at tp2
#     here, see step 4).
#
#   tests/gpu/prefill/tp2_pf.sh <fixture bf16 dir> <fixture exl3 dir> <work dir> [budget (2048)]
#
# Run INSIDE the session container of tests/gpu/run_gpu.sh given both cards (a comma-separated PCI list). ENV as
# oracle_pf.sh.
set -u
[ "$#" -ge 3 ] || { echo "usage: $0 <fixture bf16 dir> <fixture exl3 dir> <work dir> [budget]" >&2; exit 2; }
BF16=$(cd "$1" && pwd) SRC=$(cd "$2" && pwd)
mkdir -p "$3" && WORK=$(cd "$3" && pwd)
MAXTOK=${4:-2048}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
PREFIX=${RADIANCE_PREFIX:-/stage/opt/radiance}
HOME_LIST=${EXL3_HOME:-$REPO/build-gpu/radiance_home}:$PREFIX/share/radiance
K=exl3r4d,exl3ref,libr4d,libavx,libref
B=$PREFIX/bin
cd "$WORK" || exit 2
fail=0
say() { echo "$1 $2"; [ "$1" = PASS ] || fail=1; }

python3 - <<'EOF'
import json
long = " ".join(f"Item {i}: the river {['rose', 'fell', 'turned', 'slowed'][i % 4]} near the {i % 97}th mill." for i in range(200))
rows = [{"prompt": "The capital of France is", "source": "p0"}, {"prompt": "def fibonacci(n):", "source": "p1"},
        {"prompt": long, "source": "long"}, {"prompt": "Hello, my name is", "source": "p3"}]
open("corpus.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
EOF
if [ ! -f ours.rad ]; then
    EXL3_SRC=$SRC RADIANCE_HOME=$HOME_LIST "$B/rad-convert" "$BF16" --recipe "$REPO/tests/arch/data/fixture-exl3.recipe" \
        --kernels $K -o ours.rad.tmp > convert.log 2>&1 && mv ours.rad.tmp ours.rad ||
        { echo "FAIL convert: $(grep -E '^E' convert.log | head -2 | tr '\n' ' ')"; exit 1; }
fi
SERVE="--model ours.rad --kernels $K --tp 2 --max-model-len 4096 --max-num-batched-tokens $MAXTOK"

# 1 -- the graph, rank 0's (the debug graph prints rank 0's declare) and its widths
RADIANCE_HOME=$HOME_LIST "$B/radiance" $SERVE --debug-graph > graph.log 2>&1
awk '/^moe_exl3_(gate_up|down) /{on=1; print; next} on && /host domain:/{on=0} on && /->/{print}' graph.log > bands.txt
cat bands.txt
bad=$(grep -- '->' bands.txt | grep -vc 'exl3r4d_moe_')
pf=$(grep -- '->' bands.txt | grep -cE '_pf +\(exl3r4d')
[ -s bands.txt ] && [ "$bad" -eq 0 ] && [ "$pf" -gt 0 ] && say PASS "tp2 debug-graph: every device band on exl3r4d, $pf band line(s) on the _pf rows" ||
    say FAIL "tp2 debug-graph: $bad device band line(s) not on exl3r4d, $pf on _pf (bands.txt)"
grep -qE '^moe_exl3_gate_up +N=768 .*N_odd=512' bands.txt && grep -qE '^moe_exl3_down +N=256 K=384 .*K_odd=256' bands.txt &&
    say PASS "tp2 rank 0 widths: gate_up N=768 N_odd=512, down K=384 K_odd=256 (D7's 384/256 split of 640)" ||
    say FAIL "tp2 rank 0 widths not the 384/256 split: $(grep '^moe_exl3' bands.txt | tr '\n' ' ')"

# 2 -- the oracle on both ranks
RADIANCE_OP_ORACLE=moe_exl3_gate_up,moe_exl3_down RADIANCE_OP_ORACLE_MAX=${ORACLE_MAX:-500} RADIANCE_HOME=$HOME_LIST \
    python3 "$REPO/tests/arch/complete2.py" c2-oracle.json corpus.jsonl -- "$B/radiance" $SERVE > oracle.out 2>&1 ||
    say FAIL "tp2 oracle run: $(tail -3 oracle.out | tr '\n' ' ')"
python3 - c2-oracle.json.log <<'EOF' > oracle-ranks.txt
import re, sys
# Each oracle report: a header line (op, verdict) and its operand lines. The log carries no rank tag on these lines,
# so a report's RANK is read off the class-0 codes table it was handed: rank 0 holds the wider slice of even experts
# (gate_up opd1 [entries, 768, 256], down opd1 [entries, 256, 384]), rank 1 the narrower (512 / 256). Per rank:
# verdicts, issues at M > 16, the class tables' shapes, and issues whose expert_offset shows rows for expert 0 (class
# 0) and expert 1 (class 1) -- the oracle prints the table's first four entries.
hdr = re.compile(r"^W(?:\[(\d)\])? oracle #\d+ op '(moe_exl3_\w+)' (\w+)")
opd = re.compile(r"^  opd(\d+)~? +(\w+) +\[([\d,]+)\]")
head = re.compile(r"\[(-?\d+) (-?\d+) (-?\d+) (-?\d+)\]")
ranks = {}
def close(c):
    if c is None or c["rank"] is None:
        return
    r = ranks.setdefault(c["rank"], dict(agree=0, other=0, big=0, both=0, shapes=set()))
    r["agree" if c["verdict"] == "agrees" else "other"] += 1
    r["big"] += c["big"]
    r["both"] += c["both"]
    r["shapes"] |= c["shapes"]
cur = None
for line in open(sys.argv[1], errors="replace"):
    m = hdr.match(line)
    if m:
        close(cur)
        cur = dict(rank=None, op=m.group(2), verdict=m.group(3), big=0, both=0, shapes=set())
        continue
    if cur is None:
        continue
    m = opd.match(line)
    if not m:
        continue
    k, dims = int(m.group(1)), [int(v) for v in m.group(3).split(",")]
    gu = cur["op"] == "moe_exl3_gate_up"
    if k == 1:
        cur["rank"] = ("0" if dims[1] == 768 else "1") if gu else ("0" if dims[2] == 384 else "1")
    if k == 0 and gu and dims[0] > 16:
        cur["big"] = 1
    if k in ((1, 7) if gu else (1, 6)):
        cur["shapes"].add(f"{cur['op'][9:]} opd{k} {dims}")
    if k == (6 if gu else 5):
        h = head.search(line)
        if h:
            o = [int(h.group(i)) for i in range(1, 5)]
            if o[1] > o[0] and o[2] > o[1]:
                cur["both"] = 1
close(cur)
for rk in sorted(ranks):
    r = ranks[rk]
    print(f"rank {rk}: agree {r['agree']} other {r['other']} gate_up_issues_M>16 {r['big']} "
          f"issues_with_both_classes {r['both']} class_tables {sorted(r['shapes'])}")
EOF
cat oracle-ranks.txt
n=$(grep -c '^rank' oracle-ranks.txt)
okr=$(awk '$4 > 0 && $6 == 0 && $8 > 0 && $10 > 0' oracle-ranks.txt | wc -l)
[ "$n" -eq 2 ] && [ "$okr" -eq 2 ] && say PASS "tp2 oracle: both ranks agree on every issue, judged M > 16 issues, and ran issues with runs in both parity classes" ||
    say FAIL "tp2 oracle per rank (oracle-ranks.txt): $(tr '\n' ' ' < oracle-ranks.txt)"

# 3 -- repeat on both ranks
RADIANCE_OP_REPEAT=moe_exl3_gate_up,moe_exl3_down RADIANCE_OP_REPEAT_MAX=${ORACLE_MAX:-500} \
RADIANCE_OP_REPEAT_MAXBYTES=4294967296 RADIANCE_HOME=$HOME_LIST \
    python3 "$REPO/tests/arch/complete2.py" c2-repeat.json corpus.jsonl -- "$B/radiance" $SERVE > repeat.out 2>&1 ||
    say FAIL "tp2 repeat run: $(tail -3 repeat.out | tr '\n' ' ')"
grep -E 'op REPEAT: [0-9]+ issue' c2-repeat.json.log
nrep=$(grep -E 'op REPEAT: [1-9][0-9]* issue\(s\) run twice, 0 of them ANSWERED DIFFERENTLY; 0 more were NOT JUDGED' c2-repeat.json.log | wc -l)
[ "$nrep" -ge 2 ] && say PASS "tp2 repeat: $nrep rank summaries, 0 differ" || say FAIL "tp2 repeat: $nrep clean rank summaries (want 2)"
cmp -s c2-oracle.json c2-repeat.json && say PASS "tp2 completions identical across two server starts" ||
    say FAIL "tp2 completions differ across two server starts"

# 4 -- device time per rank, from the engine's own per-op timing (--profile-ops: device event pairs around each
# issue, one table a rank at exit). rocprofv3 cannot trace this engine at tp2 here: under it the AQL backend's
# all-reduce init() fails ("device error", ar_oneshot_2rank_exact / ar_hc_write) whichever way the cards are made
# visible -- notes/p5.md. The reading rskew.sh would give: our ops on BOTH ranks, on the device site, and the gap.
RADIANCE_HOME=$HOME_LIST timeout 900 python3 "$REPO/tests/arch/complete2.py" c2-prof.json corpus.jsonl -- \
    "$B/radiance" $SERVE --profile-ops > prof.out 2>&1 || say FAIL "tp2 --profile-ops run: $(tail -3 prof.out | tr '\n' ' ')"
python3 - c2-prof.json.log <<'EOF' > prof-ranks.txt
import collections, re, sys
rank, tot = None, collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0, set()]))
for line in open(sys.argv[1], errors="replace"):
    m = re.search(r"per-op timing, rank (\d+)", line)
    if m:
        rank = m.group(1)
        continue
    f = line.split()
    if rank is None or len(f) < 6 or not f[0].isdigit() or not f[5].startswith("moe_exl3"):
        continue
    t = tot[f[5]][rank]
    t[0] += int(f[0]); t[1] += float(f[1]); t[2].add(f[3])
for op in sorted(tot):
    print(op, " | ".join(f"rank {r}: {v[0]} calls {v[1]:.3f} ms site {','.join(sorted(v[2]))}" for r, v in sorted(tot[op].items())))
EOF
cat prof-ranks.txt
good=$(grep -cE 'rank 0: [1-9][0-9]* calls [0-9.]+ ms site dev \| rank 1: [1-9][0-9]* calls [0-9.]+ ms site dev$' prof-ranks.txt)
[ "$good" -eq 2 ] && say PASS "tp2 per-rank device time: both ops issued and timed on the device site on both ranks" ||
    say FAIL "tp2 per-rank device time (prof-ranks.txt): $(tr '\n' ' ' < prof-ranks.txt)"
exit $fail
