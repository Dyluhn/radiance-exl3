#!/bin/bash
# p6_convert.sh -- P6's one command (PLAN.md §4 P6; D1-D3, D8): the EXL3 container, which is the baseline with its
# routed experts (trunk and MTP head) replaced by turboderp's EXL3 experts and every other weight byte-copied.
# rad-convert reads an all-holes skeleton of Qwen's bf16 checkpoint (tools/skeleton_ckpt.py), the author's served
# recipe with only the routed-expert rule changed (recipes/qwen4exp-exl3.recipe), and --reuse <baseline>.
#
#   flock <the project's gpu.lock> env BASELINE=... OUT=... EXL3_SRC=... P6_GPU=... tools/p6_convert.sh
#   P6_PLAN_ONLY=1 ... tools/p6_convert.sh       # steps 1-5 only: writes nothing but logs
#
# STEPS (each refuses by name: a FAIL line on stderr and a non-zero exit):
#  1. build this checkout's plugins in the radiance build image (<repo>/build/p6-plugins; seconds once built)
#  2. the skeleton at $P6_WORK/Qwen/Qwen3.8-Flash-Next, made from $BF16_META when absent (~0 disk)
#  3. preflight: OUT does not exist; no KFD client and no container running (the card gets opened)
#  4. the plan: rad-convert --plan-only -v with the EXL3 source NOT mounted (the plan never reads it), then
#     tools/p6_predict.py over it and the baseline's entry table: the weights reused and written, and the
#     output's predicted bytes. Anything written by a rule other than exl3 -> FAIL.
#  5. P6_PLAN_ONLY=1 stops here, with the space verdict.
#  6. the EXL3 source is whole (every shard its index names, whole, not sparse); free space on OUT's
#     filesystem >= predicted bytes + 5%.
#  7. the convert; then rad-convert's own `--reuse: N weight(s) ... the other M were written` must equal the
#     prediction, and the amdgpu kernel-log watchdog must be clean.
# Logs and provenance: $P6_WORK/run-<UTC>/ (plugins.log, plan.log, baseline-listing.txt, predict.json,
# convert.log, provenance.txt).
#
# WHY A GPU. rad-convert declares the model and refuses an op that resolves to no kernel in any domain
# (radiance 1.3.0 tools/rad_convert.cpp:604-650). Several qwen4exp ops (the router fusions, qk_norm_rope_gate, the
# QSA indexer's) have rows only in libr4d, a device library that loads only where its card is visible. Nothing is
# launched; the card is held open for the run, hence the lock. Our two expert ops resolve on the host domain
# (exl3ref) until exl3r4d exists; exl3r4d is named already and loads when it is built (a name that matches no
# library is a warning, core/plugin/loader.cpp:645-649).
#
# THE RECORDED NAMES. A container is meant to leave the machine: rad-convert records the checkpoint as its last
# two path components (rad_convert.cpp:744-756), and the recipe text with every $NAME expanded
# (core/format/recipe.cpp). So rad-convert runs in /p6 inside the container, the skeleton mounted at
# Qwen/Qwen3.8-Flash-Next and the EXL3 source at turboderp/Qwen3.8-Flash-Next-exl3, with
# EXL3_SRC=turboderp/Qwen3.8-Flash-Next-exl3: the container says "from Qwen/Qwen3.8-Flash-Next" (as the baseline
# does) and "src=turboderp/Qwen3.8-Flash-Next-exl3", never a path of this machine. The author did the same with
# calib=calib/w4nl-calib. The skeleton's config is Qwen's; --set selects ("qwen4exp", "exl3") (docs/STEP0.md S0.4).
#
# Env (every machine fact is here, none in the file):
#   BASELINE      required: the baseline .rad (mounted read-only)
#   OUT           required: the .rad to write; must not exist (its directory is mounted read-write)
#   P6_GPU        required: HIP_VISIBLE_DEVICES, as seen inside the container, of the card rad-convert opens
#   EXL3_SRC      required unless P6_PLAN_ONLY=1: turboderp's EXL3 checkpoint dir (config, index, all shards)
#   BF16_META     needed only when the skeleton is absent: Qwen's config.json, index and tokenizer files
#   P6_WORK       default <repo>/build/p6
#   P6_IMAGE      default radiance-build:<the release in RADIANCE_VERSION>
#   P6_PLAN_ONLY  1 = stop after the plan
set -euo pipefail

die() { printf 'p6_convert: FAIL: %s\n' "$*" >&2; exit 1; }
say() { printf 'p6_convert: %s\n' "$*"; }

repo=$(cd "$(dirname "$0")/.." && pwd)
: "${BASELINE:?set BASELINE to the baseline .rad}"
: "${OUT:?set OUT to the .rad to write (it must not exist)}"
: "${P6_GPU:?set P6_GPU to the in-container HIP_VISIBLE_DEVICES index of the card rad-convert opens}"
: "${P6_WORK:=$repo/build/p6}"
: "${P6_IMAGE:=radiance-build:$(cut -d' ' -f1 "$repo/RADIANCE_VERSION")}"
: "${P6_PLAN_ONLY:=0}"
[ "$P6_PLAN_ONLY" = 1 ] || : "${EXL3_SRC:?set EXL3_SRC to the turboderp EXL3 checkpoint directory}"

CKPT=Qwen/Qwen3.8-Flash-Next                          # the checkpoint, as the container names it
CKPT_REVISION=de4b8e4d43b917e7706784d8bb445c9af86a3540  # its revision the skeleton's headers come from
SRC=turboderp/Qwen3.8-Flash-Next-exl3                 # the EXL3 source, as the recipe's src= records it
PREFIX=/stage/opt/radiance                            # the build image's install (radiance Dockerfile: build stage)
KERNELS=exl3r4d:exl3ref:libr4d:libavx:libref
RECIPE=qwen4exp-exl3.recipe

for tool in docker python3 df journalctl realpath; do
    command -v "$tool" >/dev/null || die "$tool is not on PATH"
done
baseline=$(realpath -e -- "$BASELINE") || die "BASELINE does not exist: $BASELINE"
[ -f "$baseline" ] || die "BASELINE is not a file: $baseline"
out=$(realpath -m -- "$OUT")
outdir=$(dirname "$out")
[ -d "$outdir" ] || die "OUT's directory does not exist: $outdir"
[ ! -e "$out" ] || die "refusing to overwrite $out"
mkdir -p "$P6_WORK"
work=$(realpath -e -- "$P6_WORK")
run="$work/run-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$run"
docker image inspect "$P6_IMAGE" >/dev/null 2>&1 || die "no image $P6_IMAGE (build the radiance Dockerfile's build stage)"
{
    printf 'p6_convert %s UTC\nrepo %s @ %s%s\nimage %s %s\n' "$(date -u +%FT%TZ)" "$repo" \
        "$(git -C "$repo" rev-parse HEAD 2>/dev/null || echo '?')" \
        "$(git -C "$repo" status --porcelain 2>/dev/null | grep -q . && echo ' (dirty)' || true)" \
        "$P6_IMAGE" "$(docker image inspect --format '{{.Id}}' "$P6_IMAGE")"
    printf 'baseline %s (%s B)\nout %s\nP6_GPU %s\n' "$baseline" "$(stat -c %s "$baseline")" "$out" "$P6_GPU"
} > "$run/provenance.txt"

# ---- 1. plugins
say "1/7 building the plugins in $P6_IMAGE (log $run/plugins.log)"
docker run --rm --security-opt label=disable --network none -v "$repo:/work" -w /work "$P6_IMAGE" sh -euc "
    cmake -S /work -B /work/build/p6-plugins -DCMAKE_PREFIX_PATH=$PREFIX -DCMAKE_BUILD_TYPE=Release
    cmake --build /work/build/p6-plugins -j\$(nproc)
    cmake --install /work/build/p6-plugins" > "$run/plugins.log" 2>&1 || die "the plugin build failed (log $run/plugins.log)"
home="$repo/build/p6-plugins/home"
(cd "$home" && find . -name '*.so' -exec sha256sum {} +) >> "$run/provenance.txt"

# ---- 2. skeleton
skel="$work/$CKPT"
if [ ! -f "$skel/skeleton.json" ]; then
    : "${BF16_META:?no skeleton at $skel: set BF16_META to a directory holding the Qwen config.json, index and tokenizer}"
    say "2/7 making the skeleton from $BF16_META (headers of absent shards by HTTP Range; log $run/skeleton.log)"
    python3 "$repo/tools/skeleton_ckpt.py" "$BF16_META" "$skel" --repo "$CKPT" --revision "$CKPT_REVISION" \
        > "$run/skeleton.log" 2>&1 || die "the skeleton failed (log $run/skeleton.log)"
fi
say "2/7 skeleton $skel: $(du -sb --apparent-size "$skel" | cut -f1) B apparent, $(du -sB1 "$skel" | cut -f1) B on disk"

# ---- 3. preflight
kfd=$(ls /sys/class/kfd/kfd/proc/ 2>/dev/null | tr '\n' ' ')
[ -z "$kfd" ] || die "KFD clients present (pids: $kfd): another process holds a GPU; run under the GPU lock"
running=$(docker ps --format '{{.Names}} {{.Image}}')
[ -z "$running" ] || die "containers running: $running"
klog='amdgpu.*(MES|SMU|ring|timeout|reset|page fault)'
start=$(date '+%Y-%m-%d %H:%M:%S')
say "3/7 preflight: no KFD client, no container, $out does not exist"

dk=(docker run --rm --init --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined
    --security-opt label=disable --network none
    -e HIP_VISIBLE_DEVICES="$P6_GPU" -e RADIANCE_HOME="/plugins:$PREFIX/share/radiance" -e EXL3_SRC="$SRC"
    -v "$home:/plugins:ro" -v "$repo/recipes:/recipes:ro" -v "$skel:/p6/$CKPT:ro"
    -v "$(dirname "$baseline"):/baseline:ro" -w /p6)
convert=(--entrypoint "$PREFIX/bin/rad-convert" "$P6_IMAGE" "$CKPT" --tokenizer "$CKPT/tokenizer.json"
         --recipe "/recipes/$RECIPE" --reuse "/baseline/$(basename "$baseline")"
         --set quantization_config.quant_method=exl3 --kernels "$KERNELS" -o "/out/$(basename "$out")")

# ---- 4. plan and prediction
say "4/7 plan (--plan-only; the EXL3 source is not mounted; log $run/plan.log)"
"${dk[@]}" -v "$outdir:/out:ro" "${convert[@]}" --plan-only -v > "$run/plan.log" 2>&1 \
    || { grep -E '^[EW] ' "$run/plan.log" | tail -20 >&2; die "the plan was refused (log $run/plan.log)"; }
docker run --rm --security-opt label=disable --network none -v "$(dirname "$baseline"):/baseline:ro" \
    --entrypoint "$PREFIX/bin/rad-info" "$P6_IMAGE" -v "/baseline/$(basename "$baseline")" \
    > "$run/baseline-listing.txt" 2>&1 || die "rad-info -v on the baseline failed ($run/baseline-listing.txt)"
python3 "$repo/tools/p6_predict.py" "$run/plan.log" "$run/baseline-listing.txt" "$(stat -c %s "$baseline")" \
    --json "$run/predict.json" || die "the plan writes something other than the exl3 experts ($run/predict.json)"
read -r predicted reused written < <(python3 -c 'import json, sys
r = json.load(open(sys.argv[1])); print(r["predicted_bytes"], r["reused"], r["written"])' "$run/predict.json")
need=$(( predicted + predicted / 20 ))
avail=$(df --output=avail -B1 "$outdir" | tail -1 | tr -d ' ')
verdict=$([ "$avail" -ge "$need" ] && echo "enough" || echo "NOT enough")
say "4/7 space on $outdir: $avail B free; the container needs $predicted B, +5% = $need B: $verdict"

# ---- 5. plan only?
if [ "$P6_PLAN_ONLY" = 1 ]; then
    hits=$(journalctl -k --since "$start" --no-pager | grep -iE "$klog" || true)
    [ -z "$hits" ] || die "amdgpu kernel errors during the plan: $hits"
    say "5/7 P6_PLAN_ONLY=1: stopped after the plan ($run); kernel log clean"
    exit 0
fi

# ---- 6. the source, the space
src=$(realpath -e -- "$EXL3_SRC") || die "EXL3_SRC does not exist: $EXL3_SRC"
say "6/7 checking that $src is whole"
python3 "$repo/tools/skeleton_ckpt.py" "$src" --check-whole || die "the EXL3 source is not whole: $src"
[ "$avail" -ge "$need" ] || die "not enough space on $outdir: $avail B free, $need B needed (predicted $predicted B + 5%)"
printf 'exl3_src %s\npredicted %s B, reused %s, written %s\n' "$src" "$predicted" "$reused" "$written" >> "$run/provenance.txt"

# ---- 7. the convert
say "7/7 converting (log $run/convert.log)"
status=0
"${dk[@]}" -v "$src:/p6/$SRC:ro" -v "$outdir:/out" "${convert[@]}" 2>&1 | tee "$run/convert.log" || status=$?
hits=$(journalctl -k --since "$start" --no-pager | grep -iE "$klog" || true)
[ -z "$hits" ] || die "amdgpu kernel errors during the convert: $hits"
[ "$status" -eq 0 ] || die "rad-convert exited $status (log $run/convert.log); $out may be partial"
line=$(grep -E '^I --reuse: [0-9]+ weight\(s\)' "$run/convert.log" || true)
got_reused=$(sed -nE 's/^I --reuse: ([0-9]+) weight\(s\).*/\1/p' <<< "$line")
got_written=$(sed -nE 's/.*the other ([0-9]+) were written.*/\1/p' <<< "$line")
[ "$got_reused" = "$reused" ] && [ "$got_written" = "$written" ] \
    || die "rad-convert reused ${got_reused:-?} and wrote ${got_written:-?} weight(s); the plan predicted $reused and $written"

# ---- what --reuse does not carry: the vocab, the metadata and the recipe are re-derived; they must still match
radinfo() { docker run --rm --security-opt label=disable --network none -v "$(dirname "$baseline"):/baseline:ro" \
                -v "$outdir:/out:ro" --entrypoint "$PREFIX/bin/rad-info" "$P6_IMAGE" "$@"; }
b="/baseline/$(basename "$baseline")" o="/out/$(basename "$out")"
for what in vocab meta recipe; do
    radinfo "--$what" "$b" > "$run/$what.baseline" 2>&1 && radinfo "--$what" "$o" > "$run/$what.out" 2>&1 \
        || die "rad-info --$what failed ($run/$what.*)"
done
cmp -s "$run/vocab.baseline" "$run/vocab.out" || die "the vocab differs from the baseline's ($run/vocab.*)"
meta=$(diff "$run/meta.baseline" "$run/meta.out" | grep '^[<>]' | grep -vE '^[<>] (metadata \(|  quantization_config\.quant_method )' || true)
[ -z "$meta" ] || die "metadata differs from the baseline's beyond quantization_config.quant_method: $meta"
recipe=$(diff "$run/recipe.baseline" "$run/recipe.out" | grep '^[<>]' || true)
want="> blk.*.ffn_*_exps.*.weight exl3 src=$SRC bits=4 codebook=mul1"
[ "$(grep -c '^<' <<< "$recipe")" = 1 ] && [ "$(grep '^>' <<< "$recipe")" = "$want" ] \
    || die "the recorded recipe is not the baseline's with only the expert rule replaced: $recipe"
n_exl3=$(radinfo -v "$o" | grep -cE '^      blk\.[0-9]+\.ffn_(gate_up|down)_exps\.[0-9]+\.weight +exl3:' || true)
[ "$n_exl3" = "$written" ] || die "rad-info -v lists $n_exl3 exl3 expert entries; $written were written"

size=$(stat -c %s "$out")
printf 'wrote %s B (predicted %s B); reused %s, written %s; exl3 entries %s; vocab, metadata, recipe checked\n' \
    "$size" "$predicted" "$got_reused" "$got_written" "$n_exl3" >> "$run/provenance.txt"
say "OK $out: $size B (predicted $predicted B); $got_reused weight(s) reused, $got_written written, $n_exl3 exl3 expert entries; vocab and metadata as the baseline's; recipe records src=$SRC; kernel log clean"
