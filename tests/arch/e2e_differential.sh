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
#   PYTHON            python3 with numpy (default python3)
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
#   KL is per scored position, radiance's own --kld-ref against a --kld-record of the reference. The
#   fixture is scaled so the comparison can see experts at all (fixture_variant.py says how and why).
#
# EXIT: 0 PASS, 1 FAIL, 77 SKIP (a prerequisite missing -- named), 2 usage.
set -u
WORK=${1:-}
[ -n "$WORK" ] || { echo "usage: $0 WORK" >&2; exit 2; }
MODE=${E2E_MODE:-host}
PY=${PYTHON:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
skip() { echo "SKIP: $*"; exit 77; }
die()  { echo "FAIL: $*"; exit 1; }
[ -x "${RADIANCE_PREFIX:-}/bin/radiance" ] || skip "RADIANCE_PREFIX has no bin/radiance ('${RADIANCE_PREFIX:-}')"
[ -f "${EXL3_HOME:-}/architectures/qwen4exp_exl3.so" ] || skip "EXL3_HOME has no architectures/qwen4exp_exl3.so"
[ -f "${TOKENIZER:-}/tokenizer.json" ] || skip "TOKENIZER has no tokenizer.json ('${TOKENIZER:-}')"
[ -d "${TEST_BUILD:-}/mut1_home" ] || skip "TEST_BUILD has no mut1_home (build tests/arch first)"
$PY -c "import numpy" 2>/dev/null || skip "$PY has no numpy"
BIN=$RADIANCE_PREFIX/bin
INSTALL_HOME=$RADIANCE_PREFIX/share/radiance
case $MODE in
    host) HOME_LIST=$TEST_BUILD/test_home:$EXL3_HOME:$INSTALL_HOME
          KERNELS=exl3test_router,exl3ref,libavx,libref
          SETS="--set indexer_n_heads=0"; SPEC_CONVERT="--max-spec 0"; SPEC_SERVE="--num-speculative-tokens 0" ;;
    gpu)  HOME_LIST=$EXL3_HOME:$INSTALL_HOME
          KERNELS=exl3r4d,exl3ref,libr4d,libavx,libref
          [ -f "$EXL3_HOME/kernels/exl3r4d.so" ] || skip "gpu mode needs kernels/exl3r4d.so in EXL3_HOME"
          SETS=""; SPEC_CONVERT=""; SPEC_SERVE="" ;;
    *)    echo "E2E_MODE is host or gpu, not '$MODE'" >&2; exit 2 ;;
esac
mkdir -p "$WORK" && cd "$WORK" || die "cannot use $WORK"
echo "e2e_differential: mode $MODE, kernels $KERNELS, work $WORK"

# ---- the fixtures: one model (seed 1), its toward-zero twin, and another seed's experts
FX="--tokenizer $TOKENIZER --layers 2 --experts 8 --topk 2 --hidden 256 --width 640"
KNOBS="--suh-scale 10 --lm-scale 8"
gen() { [ -f "$1/fixture.json" ] || $PY "$HERE/fixture_variant.py" $KNOBS $2 -- --out "$1" $3 > "$1.gen.log" 2>&1 ||
        die "fixture $1: $(tail -1 "$1.gen.log")"; }
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
        > "$out.log" 2>&1 && mv "$out.tmp" "$out" || die "convert $out: $(grep -E '^E' "$out.log" | head -2 | tr '\n' ' ')"
}
RECIPE=$HERE/data/fixture-exl3.recipe
conv ref.rad fx/bf16-plain
conv floor.rad fxt/bf16-plain
EXL3_SRC=$PWD/fx/exl3   conv ours.rad fx/bf16 --recipe "$RECIPE"
EXL3_SRC=$PWD/fxs2/exl3 conv neg.rad  fx/bf16 --recipe "$RECIPE"

# ---- prefill: the reference recorded, everything else scored against it
serve_flags="--kernels $KERNELS $SPEC_SERVE --max-model-len 4096 --debug-accept-reference-kernels"
if [ ! -f ref-kld/kld.json ]; then
    rm -rf ref-kld
    RADIANCE_HOME=$HOME_LIST "$BIN/radiance" --model ref.rad $serve_flags --kld-record ref-kld \
        --kld-corpus corpus.jsonl > ref-kld.log 2>&1 || die "recording the reference: see ref-kld.log"
fi
score() {   # name model home
    RADIANCE_HOME=$3 "$BIN/radiance" --model "$2" $serve_flags --kld-ref ref-kld --kld-out "kl-$1.json" \
        > "kl-$1.log" 2>&1 || die "scoring $1: $(grep -E '^E' "kl-$1.log" | grep -v placement | head -2 | tr '\n' ' ')"
}
score floor floor.rad "$HOME_LIST"
score ours  ours.rad  "$HOME_LIST"
score neg   neg.rad   "$HOME_LIST"
score mut1  ours.rad  "$TEST_BUILD/mut1_home:$HOME_LIST"
score mut2  ours.rad  "$TEST_BUILD/mut2_home:$HOME_LIST"

# ---- decode: two greedy tokens a prompt
complete() {   # name model home
    RADIANCE_HOME=$3 $PY "$HERE/complete2.py" "c2-$1.json" corpus.jsonl -- "$BIN/radiance" --model "$2" \
        $serve_flags > "c2-$1.out" 2>&1 || die "completing with $1: $(cat "c2-$1.out")"
}
complete ref  ref.rad  "$HOME_LIST"
complete ours ours.rad "$HOME_LIST"
complete neg  neg.rad  "$HOME_LIST"
complete mut1 ours.rad "$TEST_BUILD/mut1_home:$HOME_LIST"
complete mut2 ours.rad "$TEST_BUILD/mut2_home:$HOME_LIST"

# ---- the verdict, by the registered rule
$PY - "$MODE" <<'EOF'
import json, re, sys
kl = {n: json.load(open(f"kl-{n}.json"))["all"] for n in ("floor", "ours", "neg", "mut1", "mut2")}
c2 = {n: json.load(open(f"c2-{n}.json"))["prompts"] for n in ("ref", "ours", "neg", "mut1", "mut2")}
F, G = kl["floor"]["kld"]["mean"], kl["floor"]["kld"]["max"]
tol_mean, tol_max, far = 4 * F, 4 * G, 10 * 4 * F
def same(n): return sum(a["tokens"] == b["tokens"] for a, b in zip(c2["ref"], c2[n]))
def accept(n):   # MTP acceptance, gpu mode: the server's own summary, whatever it prints last
    try:
        m = re.findall(r"accept[a-z ]*[=:]\s*([0-9.]+)%?", open(f"c2-{n}.json.log").read())
        return m[-1] if m else "-"
    except OSError:
        return "-"
fail = []
print(f"{'run':6} {'mean KL':>10} {'max KL':>9} {'top-1':>7} {'decode=':>8} {'MTP acc':>8}")
for n in ("floor", "ours", "neg", "mut1", "mut2"):
    k = kl[n]
    d = f"{same(n)}/{len(c2['ref'])}" if n != "floor" else "-"
    print(f"{n:6} {k['kld']['mean']:10.3e} {k['kld']['max']:9.3e} {100*k['top1_agreement']:6.2f}% {d:>8} "
          f"{accept(n) if n != 'floor' else '-':>8}")
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
print("RESULT:", "FAIL -- " + "; ".join(fail) if fail else "PASS")
sys.exit(1 if fail else 0)
EOF
