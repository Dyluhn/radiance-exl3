#!/bin/sh
# run_probes.sh -- Step 0 probes S0.2, S0.3, S0.4, S0.5, S0.10 (docs/STEP0.md), host only.
#
# USAGE: RADIANCE_PREFIX=<radiance install> PROBE_BUILD=<this repo's build dir, configured with
#        -DEXL3_STEP0_PROBES=ON> FIXTURE_CKPT=<a small qwen4exp safetensors checkpoint dir>
#        WORK=<empty scratch dir> tests/probe/run_probes.sh
#
# Every probe prints one line, PASS / FAIL / SKIP with the reason, and keeps its log in $WORK/logs.
# Each has a positive and a negative control; a probe whose negative control does not fail is a FAIL
# (the check could not have failed). Writes only under $WORK: four checkpoint copies (reflinked
# where the filesystem can) and a handful of containers of a few tens of MB each.
set -u

for v in RADIANCE_PREFIX PROBE_BUILD FIXTURE_CKPT WORK; do
    eval "x=\${$v:-}"
    [ -n "$x" ] || { echo "run_probes: set $v (see the header)" >&2; exit 2; }
done
HERE=$(cd "$(dirname "$0")" && pwd)
P=$RADIANCE_PREFIX
H=$PROBE_BUILD/tests/probe
CONVERT="$P/bin/rad-convert"
for f in "$CONVERT" "$P/bin/radiance" "$P/bin/rad-info" "$H/home/architectures/exl3probe_arch.so"; do
    [ -x "$f" ] || [ -f "$f" ] || { echo "run_probes: missing $f" >&2; exit 2; }
done
[ -f "$FIXTURE_CKPT/config.json" ] || { echo "run_probes: $FIXTURE_CKPT has no config.json" >&2; exit 2; }
mkdir -p "$WORK/logs" || exit 2
L=$WORK/logs
HOME_PROBE="$H/home:$P/share/radiance"
FX="python3 -I $HERE/fixture.py"
RT="python3 -I $HERE/radtool.py"
fails=0
say()  { printf '%-6s %-6s %s\n' "$1" "$2" "$3" | tee -a "$L/summary.txt"; }
pass() { say PASS "$1" "$2"; }
fail() { say FAIL "$1" "$2"; fails=$((fails + 1)); }
has()  { grep -qF -- "$2" "$1"; }

# ---------------------------------------------------------------- fixtures and recipes
if [ ! -d "$WORK/fx-plain" ]; then
    $FX copy "$FIXTURE_CKPT" "$WORK/fx-plain" || exit 2
    for v in cfg poison holes; do
        $FX copy "$FIXTURE_CKPT" "$WORK/fx-$v" && $FX exl3cfg "$WORK/fx-$v" || exit 2
    done
    $FX poison "$WORK/fx-poison" 'shared_expert_gate' 'shared_expert\.down_proj' 'ngram_embedding' \
        > "$L/fixture-poison.log" || exit 2
    $FX holes "$WORK/fx-holes" > "$L/fixture-holes.log" || exit 2
fi
NONEXP='blk.*.ffn_down_shexp.weight  rtn  codes=i8 group=128 scale=bf16 clamp=sym rule=search
blk.*.ple_ngram.weight       rtn  codes=fp8_e4m3 block=*x* scale=bf16 rule=fixed scale_value=1.99317932128906e-4'
printf '%s\n%s\n' 'blk.*.ffn_*_exps.*.weight    rtn  codes=i8 group=128 scale=bf16 clamp=sym rule=search' \
    "$NONEXP" > "$WORK/A.recipe"
printf '%s\n%s\n' 'blk.*.ffn_*_exps.*.weight    exl3probe' "$NONEXP" > "$WORK/B.recipe"

conv() {  # log fixture recipe out home [extra args...]
    log=$1 fx=$2 rec=$3 out=$4 home=$5; shift 5
    rm -f "$out"
    timeout 900 "$CONVERT" "$WORK/$fx" -o "$out" --recipe "$WORK/$rec" --home "$home" "$@" \
        > "$L/$log.log" 2>&1
    echo "rc=$?" >> "$L/$log.log"
}

# ---------------------------------------------------------------- S0.4 selection by quant_method
conv s04-config fx-cfg B.recipe "$WORK/never.rad" "$HOME_PROBE" --plan-only
conv s04-set fx-plain B.recipe "$WORK/never.rad" "$HOME_PROBE" --plan-only \
    --set quantization_config.quant_method=exl3
conv s04-neg-noquant fx-plain B.recipe "$WORK/never.rad" "$HOME_PROBE" --plan-only
conv s04-neg-noplugin fx-cfg B.recipe "$WORK/never.rad" "$P/share/radiance" --plan-only
stub="exl3probe_arch: declare arch 'qwen4exp' quant 'exl3'"
if has "$L/s04-config.log" "$stub" && has "$L/s04-config.log" "rc=0" &&
   has "$L/s04-set.log" "$stub" && has "$L/s04-set.log" "rc=0" &&
   ! has "$L/s04-config.log" "import" &&
   ! has "$L/s04-neg-noquant.log" "$stub" && has "$L/s04-neg-noquant.log" "qwen4exp_fp8" &&
   ! has "$L/s04-neg-noplugin.log" "rc=0"; then
    pass S0.4 "quant_method=exl3 (config.json or --set) selects (qwen4exp, exl3); no quant selects qwen4exp_fp8; no plugin refuses"
else
    fail S0.4 "see $L/s04-*.log"
fi

# ---------------------------------------------------------------- S0.2 custom scheme + decode
conv s02-A fx-cfg A.recipe "$WORK/A.rad" "$HOME_PROBE"
conv s02-B fx-cfg B.recipe "$WORK/B.rad" "$HOME_PROBE"
conv s02-neg-baddecode fx-cfg B.recipe "$WORK/Bbad.rad" "$H/homebad:$HOME_PROBE"
"$P/bin/rad-info" -v "$WORK/B.rad" > "$L/s02-B-info.log" 2>&1
spell='exl3:codes=u4[1x1],suh=f16[*x1],suh.1=f16[*x1],svh=f16[1x*]/mul1/h128'
if has "$L/s02-A.log" "rc=0" && has "$L/s02-B.log" "rc=0" && has "$L/s02-B-info.log" "$spell" &&
   has "$L/s02-B.log" "exl3probe decode:" && has "$L/s02-neg-baddecode.log" "rc=0"; then
    pass S0.2 "scheme exl3 + transform mul1/h128 + 4 planes written, listed as $spell; error report via the decode hook (compare s02-B vs s02-neg-baddecode)"
else
    fail S0.2 "see $L/s02-*.log"
fi

# ---------------------------------------------------------------- S0.10 one operand, one stored form
conv s010-neg-twotags fx-cfg B.recipe "$WORK/never.rad" "$H/home2tag:$HOME_PROBE" --plan-only \
    --kernels exl3probe_k2:libref:libavx
if has "$L/s02-B.log" "selected 4 of 4: codes" && has "$L/s02-B.log" "selected 3 of 3: codes" &&
   ! grep 'exl3probe layout' "$L/s02-B.log" | grep -vq 'tag exl3probe.joined' &&
   has "$L/s010-neg-twotags.log" "wanted in two stored forms"; then
    pass S0.10 "one operand selects all 4 (gate_up) / 3 (down) planes; every band asks one tag; two tags are refused"
else
    fail S0.10 "see $L/s02-B.log and $L/s010-neg-twotags.log"
fi

# ---------------------------------------------------------------- S0.3 TP2 slicing
# The per-rank plane cut (core/format/share.cpp weight_share -> encoding.cpp enc_plane_rect) runs
# at ENGINE LOAD, not at declare: rad-tune --dry-run never reaches it. So the host engine loads the
# container at --tp 2 on two fake host devices; the stub declares no logits buffer, so a load that
# cut every share ends at that named refusal, after the load lines.
serve_tp2() {  # log container
    RADIANCE_HOST_DEVICES=2 timeout 120 "$P/bin/radiance" --model "$WORK/$2.rad" \
        --radiance-home "$HOME_PROBE" --tp 2 --port 0 --host 127.0.0.1 > "$L/$1.log" 2>&1
    echo "rc=$?" >> "$L/$1.log"
}
conv s03-shift fx-cfg B.recipe "$WORK/Bshift.rad" "$HOME_PROBE" --set exl3probe.cut_shift=1
serve_tp2 s03-tp2 B
serve_tp2 s03-neg-shift Bshift
if has "$L/s03-tp2.log" "load: 546 weights" && ! has "$L/s03-tp2.log" "at --tp 2:" &&
   has "$L/s03-tp2.log" "declared no logits buffer" &&
   has "$L/s03-tp2.log" "selected 4 of 4: codes[768x256] suh[1x256] suh.1[1x256] svh[768x1]" &&
   has "$L/s03-tp2.log" "selected 4 of 4: codes[512x256] suh[1x256] suh.1[1x256] svh[512x1]" &&
   has "$L/s03-tp2.log" "selected 3 of 3: codes[256x384] suh[1x384] svh[256x1]" &&
   has "$L/s03-tp2.log" "selected 3 of 3: codes[256x256] suh[1x256] svh[256x1]" &&
   has "$L/s03-neg-shift.log" "do not fall on whole bytes"; then
    pass S0.3 "tp2 load cuts every share on the 384/256 parity split: codes and svh with N, suh with K, suh/suh.1 whole for gate_up, svh whole for down; a cut off a byte is refused"
else
    fail S0.3 "see $L/s03-*.log"
fi

# ---------------------------------------------------------------- S0.5 --reuse
conv s05-poison-reuse fx-poison B.recipe "$WORK/Bp.rad" "$HOME_PROBE" --reuse "$WORK/A.rad" -v
conv s05-holes-reuse fx-holes B.recipe "$WORK/Bh.rad" "$HOME_PROBE" --reuse "$WORK/A.rad" -v
conv s05-neg-poison fx-poison B.recipe "$WORK/Bpn.rad" "$HOME_PROBE"
conv s05-neg-holes fx-holes B.recipe "$WORK/Bhn.rad" "$HOME_PROBE"
$RT cmp "$WORK/A.rad" "$WORK/A.rad" > "$L/s05-cmp-A-A.log" 2>&1
for x in Bp Bh Bpn Bhn; do
    [ -f "$WORK/$x.rad" ] && $RT cmp "$WORK/A.rad" "$WORK/$x.rad" > "$L/s05-cmp-A-$x.log" 2>&1
done
nonexp_same() { grep '^non-expert' "$1" | grep -q 'differ     0' && ! grep '^non-expert' "$1" | grep -vq 'differ     0'; }
nonexp_diff() { [ ! -s "$1" ] || grep '^non-expert' "$1" | grep -vq 'differ     0'; }
if has "$L/s05-poison-reuse.log" "rc=0" && has "$L/s05-holes-reuse.log" "rc=0" &&
   nonexp_same "$L/s05-cmp-A-A.log" && nonexp_same "$L/s05-cmp-A-Bp.log" &&
   nonexp_same "$L/s05-cmp-A-Bh.log" &&
   nonexp_diff "$L/s05-cmp-A-Bpn.log" && nonexp_diff "$L/s05-cmp-A-Bhn.log"; then
    pass S0.5 "every non-expert weight (rule-less bf16, int8 rule, n-gram fp8) copied byte-identical from --reuse over a poisoned and a hole-punched checkpoint; without --reuse both differ"
else
    fail S0.5 "see $L/s05-*.log"
fi

echo "logs: $L"
[ $fails -eq 0 ]
