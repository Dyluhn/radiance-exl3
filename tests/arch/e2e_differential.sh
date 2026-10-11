#!/bin/sh
# e2e_differential.sh -- this plugin end to end against the in-tree one, on a tiny fixture: the same
# trunk byte for byte, the routed experts EXL3 through qwen4exp_exl3 on one side and the bf16 rendering
# of the same decoded weights through the in-tree qwen4exp_fp8 on the other. Everything the two do
# differently is the routed block, so a wiring defect there shows and nothing else can.
#
# USAGE: tests/arch/e2e_differential.sh WORK      (WORK: an empty or reusable directory, ~2.5 GB)
#
# ENV (no defaults name a machine):
#   RADIANCE_PREFIX   the radiance install prefix (bin/radiance, bin/rad-convert, share/radiance)
#   EXL3_HOME         a $RADIANCE_HOME holding this repository's plugins: architectures/qwen4exp_exl3.so,
#                     kernels/exl3ref.so (and exl3r4d.so in gpu mode), quantizers/exl3_quant.so
#   TEST_BUILD        tests/arch's build directory: test_home/, mut1_home/, mut2_home/
#   TOKENIZER         a directory with the Qwen3.8 tokenizer files (tokenizer.json, ...); the corpus is
#                     text, so the fixture needs the real vocabulary
#   E2E_MODE          host (default) or gpu
#   E2E_TP            ranks (default 1); 2 serves and scores every arm at --tp 2
#   E2E_STAGE         all (default), fixtures (generate them and stop: needs numpy and TOKENIZER), or run
#                     (fixtures already in WORK: everything else, no numpy -- what runs in the radiance image,
#                     whose python3 has none; give WORK as a path both sides see, e.g. relative to the repo)
#   PYTHON            python3 (with numpy for the fixtures stage; default python3)
#
# MODES. host: a host-only radiance at 1.3.0 has no row for four fusions qwen4exp always asks for, so
#   tests/arch's test-only library (exl3test_router, libref's chain byte for byte) serves them; the
#   QSA indexer has no host form at all, so the fixture carries none (exact below 2051 tokens of
#   context, which every prompt here is); the MTP head is left out (--max-spec 0); our ops run on
#   exl3ref. gpu: libr4d serves the trunk on the card and exl3r4d our ops, the fixture keeps its
#   indexer and its MTP head with its own EXL3 experts, and the servers draft (MTP acceptance is
#   reported side by side; the transcript is the trunk's either way). Run gpu mode only under the
#   project's GPU lock, inside the radiance build/runtime image, with the card pinned.
#
# THE INSTRUMENT AND THE RULE (pre-registered in P3's labbook, 2026-10-11; numbers there):
#   1. noise floor: the in-tree plugin over the same decoded experts rendered to bf16 TOWARD ZERO
#      instead of to nearest -- one rounding step, the scale of the gap under test. KL mean F, max G.
#   2. ours: mean KL <= 4 F and max KL <= 4 G.
#   3. controls, each mean KL >= 10 x 4 F: another seed's experts; this plugin with its parity classes
#      swapped; with gate_up's suh/suh.1 swapped (QWEN4EXP_EXL3_TEST_MUTATION, never installed).
#   4. decode: a greedy 2-token completion of every prompt identical to the reference's for ours, and
#      not for every prompt for each control.
#   5. gpu mode, MTP drafting (pre-registered 2026-10-11 for the GPU run): ours' draft acceptance --
#      radiance:draft_accepted_total / draft_tokens_total over a session that also completes every
#      prompt to 32 greedy tokens -- within max(2 pp, |floor - reference|) of the reference's.
#   A refused io_uring (libavx reads the n-gram table through it; Docker's default seccomp refuses it)
#   is a SKIP by name: the engine cannot run the model there at all.
#   KL is per scored position, radiance's own --kld-ref against a --kld-record of the reference. The
#   fixture is scaled so the comparison can see experts at all (fixture_variant.py says how and why).
#
# EXIT: 0 PASS, 1 FAIL, 77 SKIP (a prerequisite missing -- named), 2 usage.
set -u
WORK=${1:-}
[ -n "$WORK" ] || { echo "usage: $0 WORK" >&2; exit 2; }
MODE=${E2E_MODE:-host}
TP=${E2E_TP:-1}
STAGE=${E2E_STAGE:-all}
PY=${PYTHON:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
skip() { echo "SKIP: $*"; exit 77; }
die()  { echo "FAIL: $*"; exit 1; }
# A failed engine run is a SKIP, by name, where the process may not make an io_uring ring.
failed() {   # failed <log> <what>
    if grep -q "read through io_uring, and this process cannot" "$1" 2>/dev/null; then
        skip "$2: libavx cannot read the n-gram table -- io_uring is refused here (Docker's default seccomp; run with --security-opt seccomp=unconfined): $(grep -m1 'io_uring' "$1")"
    fi
    die "$2: $(grep -E '^E' "$1" | grep -v placement | head -2 | tr '\n' ' ') (see $1)"
}
case $STAGE in all|fixtures|run) ;; *) echo "E2E_STAGE is all, fixtures or run, not '$STAGE'" >&2; exit 2 ;; esac
if [ "$STAGE" != run ]; then
    [ -f "${TOKENIZER:-}/tokenizer.json" ] || skip "TOKENIZER has no tokenizer.json ('${TOKENIZER:-}')"
    $PY -c "import numpy" 2>/dev/null || skip "$PY has no numpy"
fi
if [ "$STAGE" != fixtures ]; then
    [ -x "${RADIANCE_PREFIX:-}/bin/radiance" ] || skip "RADIANCE_PREFIX has no bin/radiance ('${RADIANCE_PREFIX:-}')"
    [ -f "${EXL3_HOME:-}/architectures/qwen4exp_exl3.so" ] || skip "EXL3_HOME has no architectures/qwen4exp_exl3.so"
    [ -d "${TEST_BUILD:-}/mut1_home" ] || skip "TEST_BUILD has no mut1_home (build tests/arch first)"
fi
BIN=${RADIANCE_PREFIX:-}/bin
INSTALL_HOME=${RADIANCE_PREFIX:-}/share/radiance
case $MODE in
    host) HOME_LIST=${TEST_BUILD:-}/test_home:${EXL3_HOME:-}:$INSTALL_HOME
          KERNELS=exl3test_router,exl3ref,libavx,libref
          SETS="--set indexer_n_heads=0"; SPEC_CONVERT="--max-spec 0"; SPEC_SERVE="--num-speculative-tokens 0" ;;
    gpu)  HOME_LIST=${EXL3_HOME:-}:$INSTALL_HOME
          KERNELS=exl3r4d,exl3ref,libr4d,libavx,libref
          [ "$STAGE" = fixtures ] || [ -f "${EXL3_HOME:-}/kernels/exl3r4d.so" ] || skip "gpu mode needs kernels/exl3r4d.so in EXL3_HOME"
          SETS=""; SPEC_CONVERT=""; SPEC_SERVE="" ;;
    *)    echo "E2E_MODE is host or gpu, not '$MODE'" >&2; exit 2 ;;
esac
mkdir -p "$WORK" && cd "$WORK" || die "cannot use $WORK"
echo "e2e_differential: mode $MODE, tp $TP, stage $STAGE, kernels $KERNELS, work $WORK"

# ---- the fixtures: one model (seed 1), its toward-zero twin, and another seed's experts
FX="--tokenizer ${TOKENIZER:-} --layers 2 --experts 8 --topk 2 --hidden 256 --width 640"
KNOBS="--suh-scale 10 --lm-scale 8"
gen() {
    [ -f "$1/fixture.json" ] && return 0
    [ "$STAGE" = run ] && die "no fixture $1 in $WORK: run E2E_STAGE=fixtures first"
    $PY "$HERE/fixture_variant.py" $KNOBS $2 -- --out "$1" $3 > "$1.gen.log" 2>&1 || die "fixture $1: $(tail -1 "$1.gen.log")"
}
gen fx "" "$FX --seed 1"
gen fxt "--experts-rounding toward-zero --check fx/bf16" "$FX --seed 1"
gen fxs2 "" "--layers 2 --experts 8 --topk 2 --hidden 256 --width 640 --seed 2"
# the in-tree side reads the skeleton as a plain bf16 checkpoint: the same files, no quantization_config
plain() {
    rm -rf "$1/bf16-plain" && mkdir "$1/bf16-plain" || die "mkdir $1/bf16-plain"
    for f in "$1"/bf16/*; do [ "$(basename "$f")" = config.json ] || ln -f "$f" "$1/bf16-plain/" || cp "$f" "$1/bf16-plain/"; done
    $PY -c "import json,sys; c=json.load(open(sys.argv[1])); c.pop('quantization_config', None); json.dump(c, open(sys.argv[2], 'w'), indent=1)" \
        "$1/bf16/config.json" "$1/bf16-plain/config.json" || die "config of $1"
}
plain fx; plain fxt
[ "$STAGE" = fixtures ] && { echo "fixtures ready in $WORK"; exit 0; }
cat > corpus.jsonl <<'EOF'
{"prompt": "The capital of France is", "source": "p0"}
{"prompt": "def fibonacci(n):", "source": "p1"}
{"prompt": "In 1969, the first humans landed on", "source": "p2"}
{"prompt": "Hello, my name is", "source": "p3"}
{"prompt": "The quick brown fox jumps over the lazy dog because", "source": "p4"}
{"prompt": "List three prime numbers:", "source": "p5"}
{"prompt": "Water boils at", "source": "p6"}
{"prompt": "Radiance serves mixture-of-experts models with routed experts split across ranks, and the routed experts of this fixture are random EXL3 trellises. Radiance serves mixture-of-experts models with routed experts split across ranks, and the routed experts of this fixture are random EXL3 trellises. Radiance serves mixture-of-experts models with routed experts split across ranks, and the routed experts of this fixture are random EXL3 trellises.", "source": "p7"}
EOF

# ---- the containers
conv() {
    out=$1; shift
    [ -f "$out" ] && return 0
    RADIANCE_HOME=$HOME_LIST "$BIN/rad-convert" "$@" --kernels "$KERNELS" $SETS $SPEC_CONVERT -o "$out.tmp" \
        > "$out.log" 2>&1 && mv "$out.tmp" "$out" || failed "$out.log" "convert $out"
}
RECIPE=$HERE/data/fixture-exl3.recipe
conv ref.rad fx/bf16-plain
conv floor.rad fxt/bf16-plain
EXL3_SRC=$PWD/fx/exl3   conv ours.rad fx/bf16 --recipe "$RECIPE"
EXL3_SRC=$PWD/fxs2/exl3 conv neg.rad  fx/bf16 --recipe "$RECIPE"

# ---- prefill: the reference recorded, everything else scored against it
serve_flags="--kernels $KERNELS $SPEC_SERVE --tp $TP --max-model-len 4096 --debug-accept-reference-kernels"
if [ ! -f ref-kld/kld.json ]; then
    rm -rf ref-kld
    RADIANCE_HOME=$HOME_LIST "$BIN/radiance" --model ref.rad $serve_flags --kld-record ref-kld \
        --kld-corpus corpus.jsonl > ref-kld.log 2>&1 || failed ref-kld.log "recording the reference"
fi
score() {   # name model home
    RADIANCE_HOME=$3 "$BIN/radiance" --model "$2" $serve_flags --kld-ref ref-kld --kld-out "kl-$1.json" \
        > "kl-$1.log" 2>&1 || failed "kl-$1.log" "scoring $1"
}
score floor floor.rad "$HOME_LIST"
score ours  ours.rad  "$HOME_LIST"
score neg   neg.rad   "$HOME_LIST"
score mut1  ours.rad  "$TEST_BUILD/mut1_home:$HOME_LIST"
score mut2  ours.rad  "$TEST_BUILD/mut2_home:$HOME_LIST"

# ---- decode: two greedy tokens a prompt (and in gpu mode 32 more, for the MTP head's acceptance)
LONG=""
[ "$MODE" = gpu ] && LONG="--long 32"
complete() {   # name model home
    RADIANCE_HOME=$3 $PY "$HERE/complete2.py" $LONG "c2-$1.json" corpus.jsonl -- "$BIN/radiance" --model "$2" \
        $serve_flags > "c2-$1.out" 2>&1 || failed "c2-$1.json.log" "completing with $1"
}
complete ref   ref.rad   "$HOME_LIST"
complete floor floor.rad "$HOME_LIST"
complete ours ours.rad "$HOME_LIST"
complete neg  neg.rad  "$HOME_LIST"
complete mut1 ours.rad "$TEST_BUILD/mut1_home:$HOME_LIST"
complete mut2 ours.rad "$TEST_BUILD/mut2_home:$HOME_LIST"

# ---- the verdict, by the registered rule
$PY - "$MODE" <<'EOF'
import json, re, sys
kl = {n: json.load(open(f"kl-{n}.json"))["all"] for n in ("floor", "ours", "neg", "mut1", "mut2")}
docs = {n: json.load(open(f"c2-{n}.json")) for n in ("ref", "floor", "ours", "neg", "mut1", "mut2")}
c2 = {n: d["prompts"] for n, d in docs.items()}
F, G = kl["floor"]["kld"]["mean"], kl["floor"]["kld"]["max"]
tol_mean, tol_max, far = 4 * F, 4 * G, 10 * 4 * F
def same(n): return sum(a["tokens"] == b["tokens"] for a, b in zip(c2["ref"], c2[n]))
def acc(n):   # MTP acceptance over the session, gpu mode (complete2.py --long)
    d = docs[n].get("draft") or {}
    return d.get("ratio")
def acc_s(n):
    a = acc(n)
    return "-" if a is None else f"{100 * a:.1f}%"
fail = []
print(f"{'run':6} {'mean KL':>10} {'max KL':>9} {'top-1':>7} {'decode=':>8} {'MTP acc':>8}")
print(f"{'ref':6} {'':>10} {'':>9} {'':>7} {'-':>8} {acc_s('ref'):>8}")
for n in ("floor", "ours", "neg", "mut1", "mut2"):
    k = kl[n]
    d = f"{same(n)}/{len(c2['ref'])}"
    print(f"{n:6} {k['kld']['mean']:10.3e} {k['kld']['max']:9.3e} {100*k['top1_agreement']:6.2f}% {d:>8} {acc_s(n):>8}")
if sys.argv[1] == "gpu":
    long_same = {n: sum(a == b for a, b in zip(docs["ref"].get("long", []), docs[n].get("long", [])))
                 for n in ("floor", "ours", "neg", "mut1", "mut2")}
    print("32-token greedy transcripts identical to the reference's:",
          ", ".join(f"{n} {v}/{len(docs['ref'].get('long', []))}" for n, v in long_same.items()))
    for n in ("ref", "floor", "ours"):
        print(f"  draft counters {n}: {docs[n].get('draft')}")
print(f"tolerance: mean <= {tol_mean:.3e}, max <= {tol_max:.3e}; controls mean >= {far:.3e}")
if not (kl["ours"]["kld"]["mean"] <= tol_mean and kl["ours"]["kld"]["max"] <= tol_max):
    fail.append("ours is outside the tolerance")
if same("ours") != len(c2["ref"]):
    fail.append("ours' 2-token completions differ from the reference's")
for n in ("neg", "mut1", "mut2"):
    if kl[n]["kld"]["mean"] < far:
        fail.append(f"control {n} is not far outside the tolerance -- the instrument cannot see it")
    if same(n) == len(c2["ref"]):
        fail.append(f"control {n} completes every prompt like the reference")
if sys.argv[1] == "gpu":
    a_ref, a_floor, a_ours = acc("ref"), acc("floor"), acc("ours")
    if None in (a_ref, a_floor, a_ours):
        fail.append("no MTP draft counters (did the servers draft?)")
    else:
        band = max(0.02, abs(a_floor - a_ref))
        print(f"MTP acceptance: ref {100*a_ref:.2f}%, floor {100*a_floor:.2f}%, ours {100*a_ours:.2f}%; "
              f"allowed |ours - ref| <= {100*band:.2f} pp")
        if abs(a_ours - a_ref) > band:
            fail.append("ours' MTP acceptance is outside the band")
print("RESULT:", "FAIL -- " + "; ".join(fail) if fail else "PASS")
sys.exit(1 if fail else 0)
EOF
