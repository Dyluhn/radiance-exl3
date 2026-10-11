# Step 0 — the plan's inferences, checked at radiance 1.3.0

Engine: radiance **1.3.0** (`89cee7ce414486c717aa2085c99e1f522e4c0ea6`, ABI 15), source `data/radiance-src` at tag
`v1.3.0`, a host install of it (no HIP, tests built). Checked 2026-10-11. Machine evidence (paths, PCI address, image
digests, raw logs) is in the plan folder's `notes/step0.md`; this file is what the repo needs to keep.
Scope (COMMON brief 2026-10-11): S0.1–S0.6, S0.9, S0.10, S0.12, S0.13. S0.7 (kld format) is P0's, S0.8 (routing
contract) is P2's, S0.11 (calibration numbers) is the calibration stream's.

**Re-run the host probes** (about 10 s, no GPU): configure this repo with `-DEXL3_STEP0_PROBES=ON` against the
1.3.0 install, build, then
`RADIANCE_PREFIX=<install> PROBE_BUILD=<build dir> FIXTURE_CKPT=<a small qwen4exp safetensors dir> WORK=<scratch> tests/probe/run_probes.sh`.
Each probe prints PASS/FAIL with a positive and a negative control; logs land in `$WORK/logs`. The probes
(`tests/probe/`) are throwaway: a quantiser writing a D4-shaped scheme, a kernel library that is only asked, and a
stub `("qwen4exp", "exl3")` architecture. None of them is the real plugin.

## Verdicts

| item | verdict | evidence | touches |
|---|---|---|---|
| S0.1 engine builds and tests here | **PASS** | host `ctest -LE 'gpu\|oot'`: 40/40 passed, 607 s. `radiance-build:1.3.0`'s `/src` is the v1.3.0 tree byte for byte (656/656 files). The runtime image's 73 installed files equal the build stage's after the Dockerfile's own strip. `rad-info --plugins` on the single-card GPU: libr4d 0.5.0, 201 kernels, built for gfx1201, device code runs. | D14 |
| S0.2 custom scheme, free transform, 4 planes, own decode | **PASS** | scheme `exl3`, transform `mul1/h128`, planes `codes u4[1x1]`, `suh f16[0x1]`, `suh.1 f16[0x1]`, `svh f16[1x0]` written by rad-convert. Listed as `exl3:codes=u4[1x1],suh=f16[*x1],suh.1=f16[*x1],svh=f16[1x*]/mul1/h128`. The error report goes through the plugin's decode: mean relative error 0.132; with a decode that ignores `suh.1` it is 0.573 (gate_up 1.01). | D4 |
| S0.3 TP2 slicing keeps tiles whole | **PASS** | a host engine load at `--tp 2` (two host devices) cut all 546 rank shares on the 384/256 parity split with no refusal. gate_up: `codes[2w x K]`, `svh[2w x 1]` cut with N, `suh` and `suh.1` whole. down: `codes[N x w]`, `suh[1 x w]` cut with K, `svh` whole. A cut moved one column off a byte is refused by name. | D4, D7 |
| S0.4 `quant_method="exl3"` selects `("qwen4exp","exl3")`, reader stays bf16 | **PASS** | config.json `quantization_config.quant_method="exl3"` selects the plugin, and so does `rad-convert --set quantization_config.quant_method=exl3` (new at 1.3.0). The checkpoint is read as bf16, and the key is kept unrenamed in the container. Controls: without the key, the in-tree `qwen4exp_fp8` is selected; without the plugin, the run is refused by name. | D8 |
| S0.5 `--reuse` copies every unchanged weight without reading it | **PASS** | over a checkpoint whose non-expert tensors are poisoned (0xFF), and over one whose whole data region is a hole (520 MB apparent, 56 KB on disk), all 17 non-expert weights were copied byte-identical from the `--reuse` container: rule-less bf16, the int8 rule, and the n-gram fp8 rule from 128 shards. Without `--reuse`, all 17 differ. | D3, §5 |
| S0.6 public container's recipe | **PASS** | its 26 rules equal `data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe` at 1.3.0, except that `calib=$CALIB` was expanded to `calib=calib/w4nl-calib`. The `.recipe` file beside the container is byte-identical to the 1.3.0 source's. | Q7 |
| S0.9 scratch semantics | **PASS (by reading; not run)** | scratch is one region per lane, not per instance. Each launch gets a fresh slice. A concurrent row's operand and scratch ranges are tracked as hazards. `rad_kernel_concurrent=1` is safe under the conditions in §S0.9. | Q4, P4 |
| S0.10 one operand selects all planes; layout once per tag | **PASS, with a correction** | one operand took all 4 (gate_up) / 3 (down) planes. Layout is asked **5 times per weight** at declare (3 at the M≤16 band top, 2 at M=8192), not once. The first answer fixes the stored form and every later one must match: a second tag is refused, "wanted in two stored forms". Relayout runs once per weight. | D6 |
| S0.12 assets and disk | **FAIL (disk)** | the EXL3 container is **120.53 GB** (112.25 GiB). Free: home 64.7 GB, the model-store disk 44.3 GB. It fits on neither. `--in-place` refuses new experts, and `--reuse` does not reflink, so a reflinked baseline cannot cut that. | §5, D3 |
| S0.13 FACTS citations at 1.3.0 | **done** | 325 citations: 310 CONFIRMED, 15 CHANGED, 0 GONE, 0 wrong at 1.0.8. | D6, D7, D8, D9 |

## S0.1 — build and tests

- Host install of 1.3.0 (built `-DRAD_WITH_HIP=OFF`, tests on): `ctest -LE 'gpu|oot' --output-on-failure -j 4` →
  `100% tests passed out of 40`, `Total Test time (real) = 607.14 sec`. Excluded by label: `mem_test` (gpu) and
  `oot_plugins` (oot). Slow: `kbench` 484.1 s, `kbench_moe` 76.5 s. No skips.
- `radiance-build:1.3.0` was not rebuilt. It has no OCI revision label: its labels are the ROCm base image's. Its history is
  the Dockerfile `build` stage: `cmake -S . -B /build … -DRAD_WITH_HIP=ON … -DRAD_GPU_TARGETS=gfx1201`, then
  `DESTDIR=/stage cmake --install`. Its `/src` is byte-identical to `git ls-files` of tag v1.3.0 (sha256 per file,
  656/656). `project(radiance VERSION 1.3.0)`, and `/build/bin/radiance` carries the string `1.3.0`. `rad-info` has no
  `--version` flag.
- `stilldeadcode/radiance:1.3.0` has labels `revision=89cee7ce4144`, `version=1.3.0` and `radiance.gpu-targets=gfx1201`. Its
  `/opt/radiance` (73 files) is byte-identical to the build stage's `/stage/opt/radiance` after the rootfs stage's
  `strip --strip-unneeded` of every ELF. So the runtime image is that build's output.
- GPU, a few-seconds touch under the GPU lock with the kernel-log watchdog: from inside the container,
  `hipDeviceGetPCIBusId` maps `HIP_VISIBLE_DEVICES=1` to the target card. `rad-info --plugins` in the runtime image
  with that pin: `AQL backend, 1 device(s) [0:gfx1201]` and `libr4d 0.5.0 kernels 201 kernel(s), 106 schema(s), built for
  gfx1201`, with no `, no device code for this card` suffix (tools/rad_info.cpp:390-393). No amdgpu
  MES/SMU/ring/timeout/reset/page-fault line in the kernel log.

**Finding (affects P3, R10):** a host-only `rad-convert --plan-only` of the in-tree qwen4exp fails at 1.3.0. libref
has no rows for three ops the block declares:
`no loaded kernel plugin declares op 'quant_act_i8g'` / `'router_topk_scatter'` / `'gated_had_quant_fp8'`.
Any host-only test that declares the vendored qwen4exp needs libr4d, which means a GPU-visible container, or host
rows of our own for those three ops.

## S0.2 — a custom scheme with its own decode

Recipe rule `blk.*.ffn_*_exps.*.weight exl3probe`. The probe quantiser answers `scheme "exl3"`,
`transform "mul1/h128"` and planes `codes u4 [1x1]`, `suh f16 [0x1]`, `suh.1 f16 [0x1]` (gate_up only), `svh f16 [1x0]`,
with `row_block 16`. It quantises to a 4-bit grid the decode inverts.
- `rad_enc_valid` accepts it: plane 0 is `codes`, TILED, and nothing past that is checked for a custom scheme. The
  writer re-checks it (core/format/radfile.cpp:850).
- `rad-info -v` lists every expert as `exl3:codes=u4[1x1],suh=f16[*x1],suh.1=f16[*x1],svh=f16[1x*]/mul1/h128`. The
  transform's own `/` makes that spelling ambiguous to read back, but nothing parses a spelling back: identity is
  `rad_enc_equal`, byte for byte. Cosmetic only.
- The error report goes through the plugin's decode, which logged `view 1280x256, rows [0, 1280), 4 planes,
  transform 'mul1/h128'`. Results: `B.recipe:1  256 weight(s), mean 1.317e-01`, the 4-bit grid's error. The negative control, a decode
  that reads `suh` for every row, gives `mean 5.727e-01` with every gate_up at about 1.01, as predicted. down is unchanged.
- **Contract for P1/P2:** a custom decoder sees only the view it is handed: `rows` is the caller's chunk or this
  rank's slice, the name is `""`, and there is no absolute row (core/format/encoding.cpp:156-176). "The up half uses
  `suh.1`" works for the gate_up rule only because an expert is decoded whole (its f32 rows are under rad-convert's 64
  MiB chunk) and a rank's gate_up slice keeps its two stacked parts.
- **`row_block 16` is legal beside `[0x1]` planes.** A plane with block rows 0 is held whole across calls and written
  once (tools/rad_convert.cpp:1090-1107). Every call must therefore write the same values into it.
- rad-convert hands the quantiser rows from the checkpoint and measures its error against them
  (rad_convert.cpp:1116-1180, summary at :1257). For the import quantiser (D2) over a hole skeleton, those rows are zeros, so its error report
  measures nothing. P1 should say so in its output.

## S0.3 — tensor-parallel slicing

A rank's share of a weight's planes is cut at **engine load**, not at declare: engine_bringup.cpp:2566-2577 →
core/format/share.cpp `weight_share` → core/format/encoding.cpp:421-462 `enc_plane_rect`. A declare-only check
(`rad-tune --dry-run`) never reaches it. The probe loads the container in the host engine at `--tp 2` on two
host devices (`RADIANCE_HOST_DEVICES=2`, core/device/host.cpp:277-283). The stub declares no logits buffer, so a load
that cut every share ends at that named refusal.
- Rules, read: a block dimension of 0 is carried whole by every slice. A non-zero block is cut only on its multiples. A
  packed plane's cut must land on a byte at both ends (share.cpp:150-155).
- Measured at tp2, layout hook input per rank, w = 384 or 256 by expert parity:
  `gate_up N=768: codes[768x256] suh[1x256] suh.1[1x256] svh[768x1]`, `N=512: codes[512x256] … svh[512x1]`,
  `down K=384: codes[256x384] suh[1x384] svh[256x1]`, `K=256: codes[256x256] suh[1x256] svh[256x1]`.
  `load: 546 weights, 38.55 MiB placed`, and no `at --tp 2:` refusal.
- Negative control: the same container with every cut moved one column right (`--set exl3probe.cut_shift=1`):
  `rank 0: 'blk.0.ffn_down_exps.0.weight' at --tp 2: columns [0, 385) of the u4 'codes' plane do not fall on whole bytes`.
- The split is the in-tree one: expert parity q's 128-column blocks are cut contiguously, and the rank's place is (rank − q)
  mod world (arch/qwen4exp_fp8/qwen4exp_fp8.cpp:600-626). At tp2, even experts go rank 0 [0,384), rank 1 [384,640); odd
  experts are the reverse.

## S0.4 — selecting `("qwen4exp", "exl3")`

- `quant_descriptor()` (core/format/checkpoint.cpp:272-282, called at :267) is still `quant_method[_fmt]`. The FP8
  block read (`weight_block_size`, :499-505) is inert without the key. The new import path (`detect_import`,
  :511-556) acts only on `auto-round`, `gptq` and `quark`. `exl3` is none of them, so nothing is imported or renamed.
- New at 1.3.0: `--set quantization_config.quant_method=exl3` re-derives the descriptor (tools/rad_convert.cpp:403-416).
  **The skeleton's config.json does not need editing**, and the key is written into the container either way.
- Measured: `I arch qwen4exp, quant exl3` / `exl3probe_arch: declare arch 'qwen4exp' quant 'exl3'` for both routes. The
  container shows `source quant exl3` and `quantization_config.quant_method exl3`, not `imported_…`.
- Controls: with no key, the in-tree `qwen4exp_fp8` is selected and refuses the encoding, naming its served forms
  (`E qwen4exp_fp8: rad_arch_declare returned dtype this kernel does not serve`). With no exl3 plugin installed:
  `E no architecture plugin claims 'qwen4exp' at quantisation 'exl3'. 'qwen4exp' is served at: (unquantised).`
- **Side effect for P3 (D8):** for an *imported* checkpoint (GPTQ/AutoRound/Quark), the loader picks the family's only
  quantised plugin (core/build/startup.cpp:191-216). Today qwen4exp has none, so such an import goes to `qwen4exp_fp8`. With
  `("qwen4exp","exl3")` installed, **our plugin becomes that pick**. It must refuse non-exl3 experts by name.

## S0.5 — `--reuse`

- Read: a weight found by name in the `--reuse` container with the same encoding, quantiser, options, rank,
  layer/expert/slot and shape is copied plane by plane from it (tools/rad_convert.cpp:829-866), with no checkpoint read
  (:986-1006). Rule-less weights are covered too: quantiser `""`, options `""`. The header check compares `arch_id` only, not
  the quant descriptor (:464-469), so an `exl3` run may reuse the `""` baseline.
- Measured on the mini model, both containers from the stub, recipe A → recipe B with only the expert rule changed:
  17 `reused` lines (8 rule-less bf16, 8 int8 `rtn`, 1 n-gram fp8 `rtn` from 128 shards). `radtool.py cmp` reports
  every non-expert entry the same bytes as in A, over the poisoned checkpoint and over the all-hole checkpoint. The experts
  were re-quantised from zeros without complaint.
- Controls: `cmp A A` gives all same. Without `--reuse`, all 17 non-expert entries differ for both fixtures, so the check
  can fail.
- **Consequence for §5: the skeleton checkpoint can be all holes.** It needs headers only, so its disk cost is about 0. No
  tensor has to be fetched for the trunk.
- New at 1.3.0: `--changed FILE` (with `--reuse`) converts every weight that reads a listed checkpoint tensor.
- For the real conversion, reuse matches `options` text exactly. The served recipe's only `$ENV` is in the routed
  expert rule (S0.6), which is the one rule D3 changes, so every other rule's options match the baseline's.

## S0.6 — the public container's recipe

`rad-info --recipe` on the baseline gives 26 rules. With comments and whitespace removed,
`data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe` at 1.3.0 differs in one line:
`calib=$CALIB` (file) vs `calib=calib/w4nl-calib` (container). The `.recipe` file beside the container is identical to
the 1.3.0 source's. The recipe did not change between v1.0.8 and v1.3.0. Header: `RAD2, 113.59 GiB`, `architecture qwen4exp`,
`source quant (none)`, `created by rad-convert Oct 2 2026 from Qwen/Qwen3.8-Flash-Next`, 51,489 entries.

## S0.9 — scratch, and whether `rad_kernel_concurrent` may return 1 (by reading)

- Sized at declare: each resolved row's `scratch(&a)` is called with the **band top's geometry and no tensors**
  (core/build/rad_builder.cpp:589-611). The program's scratch is the maximum over all rows. One region per lane: 2 when the
  architecture declares a second lane through `rad_buf_concurrent`, else 1 (:1540-1542). The arena is
  `regions × pitch` (core/runtime/arena.cpp:8-38).
- Per launch: `scratch_bytes` is the row's band-top value (core/runtime/ctx.cpp:381-394). The pointer is the next
  256-B-aligned slice of the lane's region, a ring that wraps to 0 (core/runtime/issue.cpp:931-968, 1107-1115). Two
  launches of one instance get different slices until the ring wraps.
- Concurrency: a row whose library answers `rad_kernel_concurrent(i) != 0` has its IN/OUT operand ranges, WTAB pointer
  arrays and scratch slice recorded as hazards. Any pair that meets is ordered, and the rest may overlap. A row that
  answers 0 is opaque and ordered behind everything (issue.cpp:1118-1160; abi/rad_abi.h:591-598).
- **Answer:** returning 1 is safe if and only if the kernel reads and writes nothing but its operands and the slice it is
  handed, keeps nothing between launches (no counters, globals or allocations), and expects nothing to persist in
  scratch from one launch to the next. Prep-to-scratch inside **one** launch (several dispatches from one `launch`
  call) is fine. Handing scratch from one op to another is not possible. The scratch hook must size from params alone
  (M band top, n_expert, top_k, K). It cannot see tensors.

## S0.10 — one operand, one stored form

- Read: a declared weight with `planes = null` selects every plane (core/build/rad_builder.cpp:186-190). A device row
  needs a layout hook to take more than one plane (:546-551, 1404-1411). The hook is asked in the candidate filter
  (:536-563) and again per band × domain in the stored-form pass (:1330-1430). The first claimant fixes the tag, and a
  different tag is refused (:1389-1400). `relayout` runs once per weight, with the claimant's geometry
  (core/engine_bringup.cpp:1910-1913).
- Measured: every expert weight `selected 4 of 4: codes suh suh.1 svh` (gate_up) / `3 of 3` (down), all
  `tag exl3probe.joined`. 1,365 layout calls for 273 weights at plan time, 5 per weight. Negative control (prefill row
  answers another tag): `weight 'blk.0.ffn_gate_up_exps.0.weight' is wanted in two stored forms: 'exl3probe.joined'
  by exl3probe_mm_m16 and 'exl3probe.other' by exl3probe_mm_any`.
- **Correction to the plan's wording:** the hook must be pure and return the same tag and bytes at every band. It is
  asked many times, and the stored form is decided once.

## S0.12 — assets and disk (measured)

Exact bytes from the baseline's directory (`tests/probe/radtool.py sizes`). An entry's footprint is the distance to the
next entry, so the 4 KiB unit padding is included. The categories sum exactly to the file size.

| item | bytes | GB | GiB |
|---|---:|---:|---:|
| baseline container, whole file | 121,969,901,568 | 121.97 | 113.59 |
| routed experts, gptq w4nl64: 25,078 experts = 48×512 + 512 MTP − 10 protected; 2,539,520 B each on disk | 63,686,082,560 | 63.69 | 59.31 |
| n-gram table (1 entry) | 51,200,249,856 | 51.20 | 47.68 |
| `rtn` weights (int8 trunk/shared/lm_head, fp8 hc/MTP fc, draft head), 426 entries | 4,526,051,328 | 4.53 | 4.22 |
| rule-less bf16, 886 entries | 2,438,242,304 | 2.44 | 2.27 |
| protected experts, `cast bf16`, 20 entries (declared expert = -1) | 98,304,000 | 0.10 | 0.09 |
| header and tables | 20,971,520 | 0.02 | 0.02 |
| **everything `--reuse` copies** (all but the routed experts) | **58,283,819,008** | **58.28** | **54.28** |
| EXL3 experts at K=4 in D4's planes: 2,476,800 B each, 2,482,176 on disk (gate_up 1,654,784 + down 827,392) | 62,248,009,728 | 62.25 | 57.97 |
| **EXL3 container (estimate)** | **120,531,828,736** | **120.53** | **112.25** |
| turboderp routed + MTP expert planes as stored (the importer's source) | 62,137,958,400 | 62.14 | 57.87 |
| turboderp `4.05bpw_h6_ng6`, 8 model shards without n-gram and vision (HF listing) | — | 67.7 | — |
| skeleton checkpoint (S0.5: all holes) | headers only | ~0 | ~0 |

PLAN §5 assumed "~113 GB = 61 GB experts + 52 GB reused". The reused part is 58.28 GB, because 7.08 GB of trunk sits beside the
51.20 GB n-gram table, and the output is 120.53 GB.

**Can a reflink copy of the baseline plus `rad-convert --in-place` cut the output's disk cost? No**, for three
independent reasons:
1. `--in-place` refuses any held weight that would change (tools/rad_convert.cpp:846-856), and refuses any **new expert**:
   `a container extended in place cannot take a new expert` (core/format/radfile.cpp:859-865). Experts can never be
   swapped in place.
2. `--reuse` copies planes with pread/pwrite (rad_convert.cpp:986-1003), so it shares no extents even on btrfs.
   `copy_file_range` appears only in the writer's spill splice (radfile.cpp:1167-1210).
3. The only baseline copy is on the ext4 volume, which has no reflink. Home is btrfs and holds no copy.

What could share bytes: a post-write dedupe (FIDEDUPERANGE) against a baseline copy **on the same btrfs**. The 58.28 GB
of reused entries start 4 KiB-aligned with the same plane layout, so they are block-identical. But the write still peaks
at the full 120.53 GB, and it needs that copy of the baseline (122 GB) on btrfs first.

**Result:** the convert needs 120.53 GB contiguous on one filesystem for the output, plus the importer's source
(62.14 GB of expert planes by range fetch, or 67.7 GB of whole shards) unless the importer streams it. Neither disk holds
the output alone. This blocks P6's conversion until space is freed or another volume is named. That is the owner's decision (PLAN §7 manual gate 1).

## Drift at 1.3.0 that touches a decision (from S0.13 and the probes)

1. **`--kernels` now loads only the libraries it names** (core/plugin/loader.cpp:593-620; docs/PLUGIN.md:480-488;
   seen live: `kernel libraries … that --kernels … does not name, and so not loaded: exl3probe_k`). Every command that
   passes a list must name exl3r4d, exl3ref, libr4d, libavx and libref. A reference library left off the list is left
   out (abi/rad_abi.h:605-606). Touches D6, D9, D11 and the gate scripts.
2. **D8 gets easier:** `--set quantization_config.quant_method=exl3` selects the plugin at convert (S0.4).
3. **D8 side effect:** imported qwen4exp checkpoints get routed to our plugin (S0.4). P3 must refuse them by name.
4. **D7 / P3 (6b394cc):** `geom_from` gained `arch_splits_ff`, and qwen4exp passes it, together with `attn_world_ok` from
   4596301 (qwen4exp_fp8.cpp:326-327). A vendored copy must pass both, or tp3 and other uneven worlds are refused.
   The tp2 384/256 split is byte-identical to 1.0.8's.
5. **D6/D7: tp4 uses four expert classes** (expert mod 4) through a new op `moe_gemm_q_mod4`, 16 weight runs
   (arch/common/rad_block_moe_fp8.h:1098-1149). Our block must refuse tp4 by name, or add mod-4 forms of both ops.
6. **D9: new fused router** `router_gemm_topk_scatter`: router GEMM, top-k, sort and shared gate in one launch at ≤64
   rows (rad_block_moe_fp8.h:81, 1410-1454; libr4d r4d_rows.cpp:4897; commit 0eeee5b). It sits ahead of the expert GEMMs
   and does not depend on the expert format, so a vendored block should keep issuing it.
7. **D9: tp4 exact-wire fused gather/all-reduce** (`ar_gather_hc_write_4rank`) for decode and verify steps. tp2 is
   unchanged.
8. **D6:** the core now decodes a partial selection that is a `scale`+`scale.N` chain (encoding.cpp:349-398). `suh`/`svh`
   are not in it, so one operand selecting all planes is still the only form the oracle can decode.
9. **D6/D9:** WTAB runs no longer have to divide evenly. The first tables take one more (rad_builder.cpp:711-729, 766).
10. **D7:** new `rad_weight_shard_span_parts` (abi/rad_builder.h:387-388). `rad_weight_shard_span` is unchanged.
11. **D2:** the core now imports GPTQ/AutoRound packed codes and keeps Quark MXFP4 itself (checkpoint.cpp:511-556). That
    is a precedent for core-side import, but EXL3 is not covered and the plugin route is unaffected. TOOLS.md:134-137 is
    stale on this.
12. **R4/R18 test pattern:** TP cuts are checked only at load. `RADIANCE_HOST_DEVICES=2` lets a host-only test load at
    `--tp 2` (S0.3).
13. **P3, R10:** a host-only declare of qwen4exp needs three ops libref lacks (S0.1).


## S0.13 — every FACTS-radiance citation in §1, §2, §3, §4, §9, re-checked at 1.3.0

Method: each `file:line` was opened at v1.0.8 (`git show v1.0.8:<file>`) and the same content found at v1.3.0 in `data/radiance-src`. CONFIRMED = same content (new line given, even when unchanged); CHANGED = the content moved in a way the claim depends on (what changed, and the decision it touches); GONE = no longer exists. Done by three read-only agents, one per section group; every CHANGED row and a random sample of CONFIRMED rows (1.30, 1.60, 2.20, 3.40, 4.50, 9.10) were re-opened and confirmed by hand. Paths are relative to the radiance root.

| section | rows | CONFIRMED | CHANGED | GONE | wrong at 1.0.8 |
|---|---:|---:|---:|---:|---:|
| §1 plugin ABI | 84 | 81 | 3 | 0 | 0 |
| §2 kernel rows, §3 quantisers | 137 | 134 | 3 | 0 | 0 |
| §4 qwen4exp, §9 gaps | 104 | 95 | 9 | 0 | 0 |
| **total** | **325** | **310** | **15** | **0** | **0** |

### §1 Plugin ABI

Files byte-identical 1.0.8→1.3.0: abi/rad_types.h, abi/rad_quant.h, arch/qwen35moe_fp8/qwen35moe_fp8.cpp, arch/llama_dense_fp8/llama_dense_fp8.cpp.
abi/rad_abi.h: one hunk only (+1 line at 605-606); `RAD_ABI_VERSION 15u` still at :22.

| # | claim (≤ 12 words) | 1.0.8 cite | 1.3.0 location | verdict | what changed / decision touched |
|---|---|---|---|---|---|
| 1.1 | Three kinds enum KERNEL=1/ARCH=2/QUANT=3 | abi/rad_abi.h:26 | abi/rad_abi.h:26 | CONFIRMED | |
| 1.2 | kind = directory loaded from (kernels/architectures/quantizers) | docs/PLUGIN.md:454 | docs/PLUGIN.md:455 | CONFIRMED | |
| 1.3 | rad_add_plugin kind must be kernels/architectures/quantizers | cmake/RadAddPlugin.cmake:70-75 | cmake/RadAddPlugin.cmake:70-75 | CONFIRMED | |
| 1.4 | Links nothing of radiance; abi/ only; in-tree same door | docs/PLUGIN.md:3-17 | docs/PLUGIN.md:3-17 | CONFIRMED | |
| 1.5 | Every plugin exports abi_version + info; RadPluginInfo fields | abi/rad_abi.h:28-42 | abi/rad_abi.h:28-42 | CONFIRMED | |
| 1.6 | ABI bumped on header shape change; optional export no bump | rad_abi.h:16-21 | abi/rad_abi.h:16-21 | CONFIRMED | still 15u (:22) though rad_builder.h grew trailing fields/exports (additive) |
| 1.7 | Return the macro, don't hardcode | docs/PLUGIN.md:46-48 | docs/PLUGIN.md:46-48 | CONFIRMED | |
| 1.8 | Missing symbol "not a radiance plugin"; mismatch refused by FILE | core/plugin/loader.cpp:130-146 | core/plugin/loader.cpp:130-146 | CONFIRMED | |
| 1.9 | find_package gives RADIANCE_ABI_VERSION | docs/PLUGIN.md:462-463 | docs/PLUGIN.md:463-464 | CONFIRMED | |
| 1.10 | build_target is prose, printed not trusted | docs/PLUGIN.md:50-53 | docs/PLUGIN.md:50-53 | CONFIRMED | |
| 1.11 | build_target field: "host", "", space/comma list | rad_abi.h:33-37 | abi/rad_abi.h:33-37 | CONFIRMED | |
| 1.12 | No fat binary fits card → device rows dropped, host stay | core/plugin/loader.cpp:357-378 | core/plugin/loader.cpp:357-382 | CONFIRMED | additive: also drops device rows when no card visible (`no_device_code_`, host backend) |
| 1.13 | Library with nothing for card left out, not failed | docs/PLUGIN.md:335-340 | docs/PLUGIN.md:337-342 | CONFIRMED | |
| 1.14 | Exact chip beats family; card mode beats either | docs/PLUGIN.md:328-333 | docs/PLUGIN.md:330-335 | CONFIRMED | |
| 1.15 | libr4d build_target "gfx1201" | r4d_rows.cpp:4820 | libr4d/r4d_rows.cpp:5571 | CONFIRMED | |
| 1.16 | libr4d builds only for ^gfx12, else skips | libr4d/CMakeLists.txt:22-27 | libr4d/CMakeLists.txt:22-27 | CONFIRMED | |
| 1.17 | rad_plugin_gpu_targets: empty result = skip, not error | cmake/RadAddPlugin.cmake:185-206 | cmake/RadAddPlugin.cmake:201-222 | CONFIRMED | moved +16 (new `-cuid` block in rad_add_plugin) |
| 1.18 | Four optional exports, exact names | docs/PLUGIN.md:55-62 | docs/PLUGIN.md:55-62 | CONFIRMED | |
| 1.19 | Optional-export typedefs open/close/concurrent/reference | rad_abi.h:576-611 | abi/rad_abi.h:576-612 | CONFIRMED | range +1 (reference comment grew, see 1.25) |
| 1.20 | open: UNSUPPORTED declines+unloads; other negative = error | loader.cpp:423-439 | core/plugin/loader.cpp:427-443 | CONFIRMED | |
| 1.21 | Misspelt symbol silently never called | rad_abi.h:579-581 | abi/rad_abi.h:579-581 | CONFIRMED | |
| 1.22 | libr4d exports neither open nor close | r4d_rows.cpp:4823-4864 | libr4d/r4d_rows.cpp:5574-5698 | CONFIRMED | no rad_plugin_open/close anywhere in libr4d at either tag; range now also holds `_host` twin rows |
| 1.23 | concurrent≠0: only operands+scratch, may overlap | rad_abi.h:591-598 | abi/rad_abi.h:591-598 | CONFIRMED | |
| 1.24 | libr4d concurrent=0 for ar, hc_, mtp_enter, w4a8/w8a8 tiled | r4d_rows.cpp:4841-4864 | libr4d/r4d_rows.cpp:5671-5698 | CONFIRMED | same kStateful list; comment adds router_gemm_topk_scatter_bf16, gemm_mxfp4a8_decode as concurrent |
| 1.25 | reference≠0: ranked below unnamed, refused for serving, kbench oracle | rad_abi.h:600-611 | abi/rad_abi.h:600-612 | CHANGED | adds "a --kernels list that does not name it leaves it out". D6 (reference lib must be named in any --kernels list) |
| 1.26 | Loader pins reference library below unnamed | loader.cpp:584-594 | core/plugin/loader.cpp:626-636 | CONFIRMED | wording "overrides this" → "places it where it is named"; context now preceded by list filter (see 1.84) |
| 1.27 | Kernel plugin exports schema_count/at, kernel_count/at | rad_abi.h:571-574 | abi/rad_abi.h:571-574 | CONFIRMED | |
| 1.28 | Hooks optional; only launch required | spec.md:179-180 | spec.md:180-181 | CONFIRMED | |
| 1.29 | RadInitFn signature | rad_abi.h:93-94 | abi/rad_abi.h:93-94 | CONFIRMED | |
| 1.30 | RadFiniFn signature | rad_abi.h:95 | abi/rad_abi.h:95 | CONFIRMED | |
| 1.31 | RadLaunchFn signature | rad_abi.h:99 | abi/rad_abi.h:99 | CONFIRMED | |
| 1.32 | RadDescribeFn signature | rad_abi.h:148-149 | abi/rad_abi.h:148-149 | CONFIRMED | |
| 1.33 | RadScratchFn signature | rad_abi.h:153 | abi/rad_abi.h:153 | CONFIRMED | |
| 1.34 | RadLayoutFn signature | rad_abi.h:193-195 | abi/rad_abi.h:193-195 | CONFIRMED | |
| 1.35 | RadRelayoutFn signature | rad_abi.h:200-202 | abi/rad_abi.h:200-202 | CONFIRMED | |
| 1.36 | RadUnrelayoutFn signature | rad_abi.h:215-217 | abi/rad_abi.h:215-217 | CONFIRMED | |
| 1.37 | RadShapeFn signature | rad_abi.h:296 | abi/rad_abi.h:296 | CONFIRMED | |
| 1.38 | RadTuneValidFn signature | rad_abi.h:340 | abi/rad_abi.h:340 | CONFIRMED | |
| 1.39 | RadFuseFn signature | rad_abi.h:511-512 | abi/rad_abi.h:511-512 | CONFIRMED | |
| 1.40 | init once per resolved instance; negative aborts startup | rad_abi.h:90-92 | abi/rad_abi.h:90-92 | CONFIRMED | |
| 1.41 | launch must not allocate/synchronise; negative fails requests | rad_abi.h:97-98 | abi/rad_abi.h:97-98 | CONFIRMED | |
| 1.42 | Own device memory only via scratch hook | docs/PLUGIN.md:215-216 | docs/PLUGIN.md:217-218 | CONFIRMED | |
| 1.43 | describe: no alloc/sync/device; RAD_E_FULL; same dispatch as launch | rad_abi.h:118-140 | abi/rad_abi.h:118-140 | CONFIRMED | |
| 1.44 | libr4d one body emits into stream-or-descriptor sink | libr4d/r4d_emit.h:37-115 | libr4d/r4d_emit.h:37-130 | CONFIRMED | +15 inside range: `R4dParams` compile-time param count/bytes check (additive) |
| 1.45 | RAD_MAX_KPARAMS 24, RAD_KPARAM_BLOB 256 | abi/rad_types.h:346 | abi/rad_types.h:346 | CONFIRMED | |
| 1.46 | scratch called at declare with band upper bound | rad_abi.h:151-152 | abi/rad_abi.h:151-152 | CONFIRMED | |
| 1.47 | opd_shape pure host; UNSUPPORTED → skip+count | rad_abi.h:289-295 | abi/rad_abi.h:289-295 | CONFIRMED | |
| 1.48 | tune_valid 1 legal / 0 skip | rad_abi.h:337-340 | abi/rad_abi.h:337-340 | CONFIRMED | |
| 1.49 | replaces chain exact or no hook | rad_abi.h:488-490 | abi/rad_abi.h:488-490 | CONFIRMED | |
| 1.50 | RadKernelInfo positional aggregate; describe last, append only | rad_abi.h:557-561 | abi/rad_abi.h:557-561 | CONFIRMED | |
| 1.51 | Model-changing setting is ABI field, not env var | abi/rad_builder.h:94-97 | abi/rad_builder.h:98-101 | CONFIRMED | |
| 1.52 | Bisect knob ≠ deployment knob | docs/ARCHITECTURES.md:372-377 | docs/ARCHITECTURES.md:386-391 | CONFIRMED | |
| 1.53 | Env vars are instruments, not features | docs/GUIDE.md:1472-1476 | docs/GUIDE.md:1837-1841 | CONFIRMED | |
| 1.54 | libr4d reads R4D_* pins (R4D_GEMM_MB1) | libr4d/r4d_gemm_args.h:210 | libr4d/r4d_gemm_args.h:210 | CONFIRMED | |
| 1.55 | libr4d reads R4D_* pins (R4D_GEMMCFG) | libr4d/r4d_gemm_args.h:287 | libr4d/r4d_gemm_args.h:287 | CONFIRMED | |
| 1.56 | Launch environment: allowed/forbidden HIP calls, code-object limits | docs/PLUGIN.md:188-216 | docs/PLUGIN.md:190-218 | CONFIRMED | |
| 1.57 | RadStream is engine-owned AQL queue | core/device/aql.cpp:1-14 | core/device/aql.cpp:1-14 | CONFIRMED | |
| 1.58 | RadStream not a hipStream_t | rad_types.h:338-344 | abi/rad_types.h:338-344 | CONFIRMED | |
| 1.59 | --no-offload-compress | cmake/RadAddPlugin.cmake:180-181 | cmake/RadAddPlugin.cmake:196-197 | CONFIRMED | |
| 1.60 | -DNDEBUG outside Debug | cmake/RadAddPlugin.cmake:161-172 | cmake/RadAddPlugin.cmake:177-188 | CONFIRMED | |
| 1.61 | Architecture exports list (id, quant, declare, step, probe, chat_format) | docs/PLUGIN.md:377-383 | docs/PLUGIN.md:378-384 | CONFIRMED | |
| 1.62 | Architecture exports declared in rad_builder.h | abi/rad_builder.h:537-685 | abi/rad_builder.h:591-751 | CONFIRMED | additive: RadArchProbe +enc_merge, +audio_frames (trailing); same six symbols |
| 1.63 | (id, quant) matched EXACTLY, no fallback | rad_builder.h:539-552 | abi/rad_builder.h:593-606 | CONFIRMED | |
| 1.64 | Same (id, quant) pair twice → RAD_E_DUPLICATE | loader.cpp:227-240 | core/plugin/loader.cpp:227-240 | CONFIRMED | |
| 1.65 | RAD_ARCH_PLUGIN(NS, ID, QUANT, VERSION, DESC) macro | arch/common/rad_arch.h:1003-1010 | arch/common/rad_arch.h:1206-1213 | CONFIRMED | |
| 1.66 | RAD_ARCH_NO_EXPORTS guard documented | rad_arch.h:997 | arch/common/rad_arch.h:1200 | CONFIRMED | |
| 1.67 | qwen4exp_fp8 exports behind guard | qwen4exp_fp8.cpp:1463 | arch/qwen4exp_fp8/qwen4exp_fp8.cpp:1567 | CONFIRMED | still `("qwen4exp", "")` at :1569; still no other #define hooks |
| 1.68 | Sibling precedent: #defines + #include (qwen35moe) | arch/qwen35moe_fp8/qwen35moe_fp8.cpp:20-24 | same file:20-24 | CONFIRMED | |
| 1.69 | Sibling precedent: #defines + #include (llama_dense) | arch/llama_dense_fp8/llama_dense_fp8.cpp:15-19 | same file:15-19 | CONFIRMED | |
| 1.70 | Quantiser exports rad_quant_count/at | abi/rad_quant.h:109-110 | abi/rad_quant.h:109-110 | CONFIRMED | |
| 1.71 | Quantiser name clash or missing name/encoding/quantize refused | loader.cpp:388-415 | core/plugin/loader.cpp:392-419 | CONFIRMED | |
| 1.72 | RAD_E_UNSUPPORTED = -18 | rad_types.h:40 | abi/rad_types.h:40 | CONFIRMED | |
| 1.73 | open UNSUPPORTED declines machine; unload; continue | rad_abi.h:583-587 | abi/rad_abi.h:583-587 | CONFIRMED | |
| 1.74 | open UNSUPPORTED handled as skip in loader | loader.cpp:427-434 | core/plugin/loader.cpp:431-438 | CONFIRMED | |
| 1.75 | layout UNSUPPORTED = stored as is; only these use file/mapped tiers | rad_abi.h:186-188 | abi/rad_abi.h:186-188 | CONFIRMED | |
| 1.76 | opd_shape UNSUPPORTED → caller skips and counts | rad_abi.h:289-293 | abi/rad_abi.h:289-293 | CONFIRMED | |
| 1.77 | replaces UNSUPPORTED = no unfused equivalent | rad_abi.h:484-486 | abi/rad_abi.h:484-486 | CONFIRMED | |
| 1.78 | quantiser encoding UNSUPPORTED → next recipe rule | rad_quant.h:60-62 | abi/rad_quant.h:60-62 | CONFIRMED | |
| 1.79 | quantiser decode UNSUPPORTED → next decoder | rad_quant.h:84-89 | abi/rad_quant.h:84-89 | CONFIRMED | |
| 1.80 | layout RAD_E_DTYPE = does not read this encoding | rad_abi.h:189-191 | abi/rad_abi.h:189-191 | CONFIRMED | |
| 1.81 | RAD_E_DTYPE → row not a candidate | core/build/rad_builder.cpp:510-543 | core/build/rad_builder.cpp:531-564 | CONFIRMED | (unchanged: hookless device row with >1 plane also rejected — relevant to D6's one-operand-all-planes) |
| 1.82 | $RADIANCE_HOME colon list, first wins, same file name shadows | docs/PLUGIN.md:468-477 | docs/PLUGIN.md:469-478 | CONFIRMED | D8 shadowing fallback still documented |
| 1.83 | --kernels comma-ordered list | docs/PLUGIN.md:479-483 | docs/PLUGIN.md:480-488 | CHANGED | was "unnamed loads after everything named"; now a kernel lib the list does not name is NOT loaded (name libavx/libref too); no list = all load, reference last; entry matches name or file (`libr4d`,`r4d`,`libr4d.so`, path). D6, D9/D11 deployment |
| 1.84 | Unnamed after named, reference last | loader.cpp:579-610 | core/plugin/loader.cpp:583-655 | CHANGED | new filter :593-620 unstages unnamed KERNEL libs when a list is given (INFO "not loaded"); arch/quant plugins exempt. Claim holds only with no list. D6, D9/D11; D8 not touched |

### §2 Kernel library rows and §3 Quantiser plugins

Files byte-identical between tags (line numbers carry over): abi/rad_types.h, abi/rad_quant.h, abi/rad_encoding.h, abi/rad_format.h, abi/rad_plugin.h, core/plugin/bands.cpp, core/plugin/select.cpp, core/format/radfile.cpp, core/plugin/testplugin/testquant.cc, libquant/* (lq_registry, lq_quant.h, lq_hess.h, lq_autoround, lq_awq, lq_paro, lq_ggml), libr4d/r4d_enc.h, libr4d/isa/README.md.

| # | claim (≤ 12 words) | 1.0.8 cite | 1.3.0 location | verdict | what changed / decision touched |
|---|---|---|---|---|---|
| 2.1 | RadKernelInfo row fields, describe last | abi/rad_abi.h:515-562 | abi/rad_abi.h:515-562 | CONFIRMED | — |
| 2.2 | example kernel row with constraints | docs/PLUGIN.md:68-85 | docs/PLUGIN.md:68-85 | CONFIRMED | — |
| 2.3 | RAD_C_EQ/LE/GE/DIV/IN and RAD_C* macros | abi/rad_types.h:297-324 | abi/rad_types.h:297-324 | CONFIRMED | — |
| 2.4 | constraints necessary, not sufficient | rad_types.h:294-296 | rad_types.h:294-296 | CONFIRMED | — |
| 2.5 | constraint on absent key does not hold | rad_types.h:311-312 | rad_types.h:311-312 | CONFIRMED | said of device facts; general rule also rad_abi.h:347-349 |
| 2.6 | floats never match | rad_types.h:280-284 | rad_types.h:280-284 | CONFIRMED | — |
| 2.7 | RAD_DERIVED skipped, winner's EQ written back | rad_abi.h:342-358 | rad_abi.h:342-358 | CONFIRMED | — |
| 2.8 | device.p2p the only machine fact | rad_types.h:306-317 | rad_types.h:306-317 | CONFIRMED | — |
| 2.9 | same, docs | docs/PLUGIN.md:100-108 | docs/PLUGIN.md:100-110 | CONFIRMED | +2 lines: collectives now have `_host` twins needing p2p==0; p2p still the only fact |
| 2.10 | at most one ranged param per op | rad_types.h:269-271 | rad_types.h:269-271 | CONFIRMED | — |
| 2.11 | same, docs | docs/OPS.md:47-48 | docs/OPS.md:47-48 | CONFIRMED | — |
| 2.12 | band edges are the constraint values | core/plugin/bands.cpp:9-12 | bands.cpp:9-12 | CONFIRMED | — |
| 2.13 | union of LE/GE/EQ, all plugins, both domains | bands.cpp:52-60 | bands.cpp:52-60 | CONFIRMED | — |
| 2.14 | same, docs; DIV/IN cut nothing | PLUGIN.md:87-91 | PLUGIN.md:87-91 | CONFIRMED | — |
| 2.15 | each band resolved at upper bound | core/build/rad_builder.cpp:505-509 | rad_builder.cpp:526-530 | CONFIRMED | — |
| 2.16 | over-claimed band shows only at concurrency | PLUGIN.md:93-98 | PLUGIN.md:93-98 | CONFIRMED | — |
| 2.17 | dtype string names operand combination | OPS.md:39-46 | OPS.md:39-46 | CONFIRMED | — |
| 2.18 | RAD_E_DTYPE / hookless multi-plane → not a candidate | rad_builder.cpp:510-543 | rad_builder.cpp:531-564 | CONFIRMED | "reads '%s' as stored, and it is %d planes" intact |
| 2.19 | same, spec | spec.md:666-670 | spec.md:667-671 | CONFIRMED | — |
| 2.20 | priority within plugin; hierarchy absolute | core/plugin/select.cpp:113-133 | select.cpp:113-133 | CONFIRMED | file unchanged; but which libs LOAD changed (see drift 5) |
| 2.21 | PRIO_ONLY 50, FP8_M16 60, PREFILL 40 | libr4d/r4d_plugin.h:50-73 | r4d_plugin.h:51-74 | CONFIRMED | values same; `R4D_PRIO_GEMM_MXDEC = 55` added above |
| 2.22 | schema per op; first fixes; conflict refuses plugin | OPS.md:134-148 | OPS.md:139-153 | CONFIRMED | — |
| 2.23 | same, header | rad_abi.h:565-570 | rad_abi.h:565-570 | CONFIRMED | — |
| 2.24 | union rule | OPS.md:62-82 | OPS.md:62-82 | CONFIRMED | rule 2 reworded at 68-69 (`hops` example dropped), same meaning |
| 2.25 | roles IN/OUT/INOUT/WEIGHT/WTAB | rad_abi.h:388-396 | rad_abi.h:388-396 | CONFIRMED | — |
| 2.26 | new op = schema + libref impl | OPS.md:1403-1408 | OPS.md:1590-1595 | CONFIRMED | — |
| 2.27 | layout hook contract | rad_abi.h:155-217 | rad_abi.h:155-217 | CONFIRMED | — |
| 2.28 | relayout lossless; numbers are quantiser's | rad_abi.h:155-166 | rad_abi.h:155-166 | CONFIRMED | "moves bits and derives nothing" |
| 2.29 | RadLayout fields | rad_abi.h:167-175 | rad_abi.h:167-175 | CONFIRMED | — |
| 2.30 | plugin-private dtypes ≥ RAD_DT_PLUGIN_BASE | rad_types.h:96-98 | rad_types.h:96-98 | CONFIRMED | the "=256" is at :146 |
| 2.31 | layout gets enc, sel, rank slice, null data | rad_abi.h:177-183 | rad_abi.h:177-183 | CONFIRMED | — |
| 2.32 | returns OK/UNSUPPORTED/DTYPE/SHAPE | rad_abi.h:185-192 | rad_abi.h:185-192 | CONFIRMED | — |
| 2.33 | relayout = host code on loader pool | rad_abi.h:197-199 | rad_abi.h:197-199 | CONFIRMED | — |
| 2.34 | relayout must keep pace with disk | spec.md:672-675 | spec.md:673-676 | CONFIRMED | — |
| 2.35 | unrelayout inverse; absent skipped by name | rad_abi.h:204-214 | rad_abi.h:204-214 | CONFIRMED | — |
| 2.36 | weight stored once; "wanted in two stored forms" | rad_builder.cpp:1217-1272 | rad_builder.cpp:1345-1400 | CONFIRMED | D6 holds; plane-join rule "only a kernel's layout can join planes" 1.0.8:1276-1283 → 1404-1411, unchanged |
| 2.37 | host domain exempt | rad_builder.cpp:1233-1238 | rad_builder.cpp:1361-1366 | CONFIRMED | — |
| 2.38 | hook's bytes wins | rad_builder.cpp:1304-1319 | rad_builder.cpp:1432-1447 | CONFIRMED | — |
| 2.39 | non-identity layout not in file/mapped tier | spec.md:677-681 | spec.md:678-682 | CONFIRMED | — |
| 2.40 | RadArgs fields | rad_abi.h:51-61 | rad_abi.h:51-61 | CONFIRMED | — |
| 2.41 | rad_args_geti / rad_args_gets | rad_abi.h:66-82 | rad_abi.h:66-82 | CONFIRMED | — |
| 2.42 | param/operand helpers | abi/rad_plugin.h:51-152 | rad_plugin.h:51-152 | CONFIRMED | — |
| 2.43 | narrowed extents, cu_seqlens length, null optional | PLUGIN.md:118-134 | PLUGIN.md:120-136 | CONFIRMED | — |
| 2.44 | narrowing conversions: f32 acc, RNE, no FMA | PLUGIN.md:220-229 | PLUGIN.md:222-231 | CONFIRMED | — |
| 2.45 | RadTunable fields | rad_abi.h:317-327 | rad_abi.h:317-327 | CONFIRMED | — |
| 2.46 | cross product; choices arrive as params | rad_abi.h:299-316 | rad_abi.h:299-316 | CONFIRMED | — |
| 2.47 | deflt not in values → plugin refused | core/plugin/loader.cpp:327-345 | loader.cpp:327-345 | CONFIRMED | — |
| 2.48 | tGemmSkinny axes + r4d_tune_valid_gemm | libr4d/r4d_rows.cpp:2839-2858 | r4d_rows.cpp:3217-3236 | CONFIRMED | — |
| 2.49 | NOTUNE | r4d_rows.cpp:3047 | r4d_rows.cpp:3445 | CONFIRMED | — |
| 2.50 | tune cache `$RADIANCE_HOME/tune/<machine>.tune` | docs/TOOLS.md:394-396 | TOOLS.md:405-407 | CONFIRMED | — |
| 2.51 | RADIANCE_TUNE pin | docs/GUIDE.md:1466 | GUIDE.md:1831 | CONFIRMED | — |
| 2.52 | --debug-graph contents | spec.md:1615-1619 | spec.md:1630-1634 | CONFIRMED | — |
| 2.53 | --debug-graph prints then exits | GUIDE.md:392-396 | GUIDE.md:452-456 | CONFIRMED | — |
| 2.54 | debug-graph table form | spec.md:259-266 | spec.md:260-267 | CONFIRMED | — |
| 2.55 | --debug-selection table | spec.md:1620-1624 | spec.md:1635-1639 | CONFIRMED | — |
| 2.56 | miss text "no loaded kernel plugin implements op" | select.cpp:156-164 | select.cpp:156-164 | CONFIRMED | — |
| 2.57 | /graph marks issued declarations | PLUGIN.md:309-310 | PLUGIN.md:311-312 | CONFIRMED | — |
| 2.58 | rad_note annotations | abi/rad_builder.h:533-534 | rad_builder.h:587-588 | CONFIRMED | — |
| 2.59 | libref ROW: host, prio 0, no hooks, fn##_shape | libref/ref_registry.cpp:931-939 | ref_registry.cpp:1069-1077 | CONFIRMED | — |
| 2.60 | ref_moe_gemm_q row | ref_registry.cpp:1049-1050 | ref_registry.cpp:1206-1207 | CONFIRMED | — |
| 2.61 | read kernel header before proposing | libr4d/isa/README.md:132-134 | README.md:132-134 | CONFIRMED | — |
| 2.62 | prefill tile study; 1750/2200/2706/2620 tok/s | libr4d/r4d_moe.hip:716-777 | r4d_moe.hip:807-868 | CONFIRMED | — |
| 2.63 | "do not go deeper", with reason | r4d_moe.hip:1072-1082 | r4d_moe.hip:1400-1410 | CONFIRMED | — |
| 3.1 | libquant plugin info, host-only | libquant/lq_registry.cpp:258-267 | lq_registry.cpp:258-267 | CONFIRMED | — |
| 3.2 | rtn, gptq, cast | lq_registry.cpp:227-242 | lq_registry.cpp:227-242 | CONFIRMED | — |
| 3.3 | awq, autoround, paroquant, ggml | lq_registry.cpp:245-255 | lq_registry.cpp:245-255 | CONFIRMED | — |
| 3.4 | rtn/gptq/cast decode = nullptr | lq_registry.cpp:232,237,241 | lq_registry.cpp:232,237,241 | CONFIRMED | — |
| 3.5 | autoround decode = nullptr | lq_autoround.cpp:291 | lq_autoround.cpp:291 | CONFIRMED | — |
| 3.6 | awq decode = nullptr | lq_awq.cpp:295 | lq_awq.cpp:295 | CONFIRMED | — |
| 3.7 | paroquant decode = nullptr | lq_paro.cpp:427 | lq_paro.cpp:427 | CONFIRMED | — |
| 3.8 | ggml decode = nullptr | lq_ggml.cpp:775 | lq_ggml.cpp:775 | CONFIRMED | — |
| 3.9 | custom-scheme quantiser with decode hook (test) | core/plugin/testplugin/testquant.cc:1-60 | testquant.cc:1-60 | CONFIRMED | — |
| 3.10 | grid options list; named tables | libquant/lq_quant.h:14-37 | lq_quant.h:14-37 | CONFIRMED | — |
| 3.11 | gptq adds calib damp act_order cd | lq_registry.cpp:54-64 | lq_registry.cpp:54-64 | CONFIRMED | — |
| 3.12 | ggml = llama.cpp types as affine planes | TOOLS.md:162 | TOOLS.md:163 | CONFIRMED | — |
| 3.13 | RadQuantWeight struct | abi/rad_quant.h:35-47 | rad_quant.h:35-47 | CONFIRMED | — |
| 3.14 | RadQuantOption struct | rad_quant.h:53-58 | rad_quant.h:53-58 | CONFIRMED | — |
| 3.15 | RadQuantEncodingFn signature | rad_quant.h:63-64 | rad_quant.h:63-64 | CONFIRMED | — |
| 3.16 | RadQuantRowBlockFn signature | rad_quant.h:70 | rad_quant.h:70 | CONFIRMED | — |
| 3.17 | RadQuantizeFn signature (`float* src`) | rad_quant.h:77-78 | rad_quant.h:77-78 | CONFIRMED | — |
| 3.18 | RadQuantDecodeFn signature | rad_quant.h:90-92 | rad_quant.h:90-92 | CONFIRMED | — |
| 3.19 | RadQuantizerInfo struct | rad_quant.h:94-103 | rad_quant.h:94-103 | CONFIRMED | — |
| 3.20 | encoding depends on name+shape; refused if moves | rad_quant.h:40-42 | rad_quant.h:40-42 | CONFIRMED | — |
| 3.21 | row_block 0 = whole; multiple of TILED block rows | rad_quant.h:66-69 | rad_quant.h:66-69 | CONFIRMED | — |
| 3.22 | quantize: planes[i] at covering row, zeroed | rad_quant.h:72-76 | rad_quant.h:72-76 | CONFIRMED | — |
| 3.23 | decode: load order, first non-UNSUPPORTED wins | rad_quant.h:80-92 | rad_quant.h:80-92 | CONFIRMED | — |
| 3.24 | core decoder registry for custom schemes | core/format/encoding.h:44-48 | encoding.h:44-48 | CONFIRMED | — |
| 3.25 | options typed; unknown option refused | rad_quant.h:49-52 | rad_quant.h:49-52 | CONFIRMED | — |
| 3.26 | spec shows old `const float* src` / decode (drift) | spec.md:753-766 | spec.md:754-767 | CONFIRMED | doc drift still present |
| 3.27 | RadEncPlane / RadEncoding structs | abi/rad_encoding.h:91-109 | rad_encoding.h:91-109 | CONFIRMED | D4 limits (8 planes, 24-char strings) hold |
| 3.28 | TILED blocks; 0 = whole extent | rad_encoding.h:24-29 | rad_encoding.h:24-29 | CONFIRMED | D4 [0x1]/[1x0] still legal |
| 3.29 | TABLE carried whole by every slice | rad_encoding.h:30-31 | rad_encoding.h:30-31 | CONFIRMED | — |
| 3.30 | sub-byte LSB-first, rows byte-aligned | rad_encoding.h:33-37 | rad_encoding.h:33-37 | CONFIRMED | — |
| 3.31 | affine chain | rad_encoding.h:39-57 | rad_encoding.h:39-57 | CONFIRMED | — |
| 3.32 | transforms per row; kernel rotates activation | rad_encoding.h:59-76 | rad_encoding.h:59-76 | CONFIRMED | — |
| 3.33 | rad_enc_valid: codes first, TILED, unique roles, core dtypes | rad_encoding.h:264-282 | rad_encoding.h:264-282 | CONFIRMED | code for codes/TILED/block[0]==1 is at :283-284 |
| 3.34 | custom scheme checked no further | rad_encoding.h:283-287 | rad_encoding.h:283-287 | CONFIRMED | `return e->plane[0].block[1] >= 1`; D4 holds |
| 3.35 | identity = byte-for-byte rad_enc_equal | rad_encoding.h:216-220 | rad_encoding.h:216-220 | CONFIRMED | — |
| 3.36 | canonical spelling `<scheme>:<role>=...` | rad_encoding.h:337-415 | rad_encoding.h:337-415 | CONFIRMED | — |
| 3.37 | header quant, recipe, encoding table | abi/rad_format.h:38-60 | rad_format.h:38-60 | CONFIRMED | — |
| 3.38 | RadFileEntry fields | rad_format.h:80-98 | rad_format.h:80-98 | CONFIRMED | — |
| 3.39 | RadFilePlane {offset 256 B, bytes} | rad_format.h:100-105 | rad_format.h:100-105 | CONFIRMED | — |
| 3.40 | one format version only | rad_format.h:16-17 | rad_format.h:16-17 | CONFIRMED | — |
| 3.41 | container KV metadata | rad_format.h:63-72 | rad_format.h:63-72 | CONFIRMED | — |
| 3.42 | `rad-convert --set K=V` | TOOLS.md:96 | TOOLS.md:97 | CONFIRMED | — |
| 3.43 | reader refuses experts without fixed stride | core/format/radfile.cpp:408-414 | radfile.cpp:408-414 | CONFIRMED | — |
| 3.44 | writer refuses experts differing slot-for-slot | radfile.cpp:795-840 | radfile.cpp:795-840 | CONFIRMED | — |
| 3.45 | movement unit, base + id*stride | spec.md:634-637 | spec.md:635-638 | CONFIRMED | — |
| 3.46 | RadWeightDecl source + planes view, null = all | abi/rad_builder.h:200-217 | rad_builder.h:213-230 | CONFIRMED | D6 one-operand-all-planes still expressible |
| 3.47 | moew4_enc_layout accepts its affine/table forms | libr4d/r4d_layout.cpp:674-718 | r4d_layout.cpp:697-741 | CONFIRMED | now also reached through new `_mod4` hooks |
| 3.48 | r4d_enc::affine / affine_table / affine_table2 | libr4d/r4d_enc.h:15-72 | r4d_enc.h:15-72 | CONFIRMED | — |
| 3.49 | tags r4d.moew4nh64.w/.s, moew5nh64.w | r4d_layout.cpp:698-716 | r4d_layout.cpp:721-739 | CONFIRMED | — |
| 3.50 | private dtypes R4D_DT_MOEW4/MOEW5 | libr4d/r4d_args.h:259-278 | r4d_args.h:259-278 | CONFIRMED | — |
| 3.51 | table values not in encoding; relayout compares | docs/MOE-W4.md:127-132 | MOE-W4.md:126-131 | CONFIRMED | — |
| 3.52 | table-vs-kernel compare functions | r4d_layout.cpp:720-734 | r4d_layout.cpp:743-757 | CONFIRMED | — |
| 3.53 | relayout: 2^-13 / table mismatch → RAD_E_DTYPE | r4d_layout.cpp:840-855 | r4d_layout.cpp:866-881 | CONFIRMED | — |
| 3.54 | rad-convert flow: declare, layout asked, measure | docs/TOOLS.md:80-130 | TOOLS.md:80-131 | CONFIRMED | `--kernels` line reworded (+1 line) |
| 3.55 | rad_convert header: runs declare, asks layout | tools/rad_convert.cpp:1-19 | rad_convert.cpp:1-19 | CONFIRMED | — |
| 3.56 | rad_weight_encoding answers the recipe's | rad_builder.h:221-233 | rad_builder.h:234-246 | CONFIRMED | — |
| 3.57 | unservable recipe refused before writing | spec.md:731-736 | spec.md:732-737 | CONFIRMED | — |
| 3.58 | checkpoint rows read as f32 | rad_convert.cpp:873-889 | rad_convert.cpp:965-981 | CONFIRMED | — |
| 3.59 | error measured via core decoder | TOOLS.md:127-128 | TOOLS.md:128-129 | CONFIRMED | — |
| 3.60 | recipe syntax, first match, $ENV, --quant first | TOOLS.md:131-150 | TOOLS.md:132-151 | CONFIRMED | — |
| 3.61 | unmatched weights must be bf16/f16/f32/block-FP8 | TOOLS.md:133-136 | TOOLS.md:134-137 | CHANGED | text unchanged but stale: code now also keeps Quark MXFP4 (trivial) and imports GPTQ/AutoRound packed codes (core/format/checkpoint.h:13-22; rad_convert.cpp:564,575). D2: in-core import path exists, EXL3 not in it |
| 3.62 | same, code comment | rad_convert.cpp:7-9 | rad_convert.cpp:7-9 | CHANGED | same as 3.61 |
| 3.63 | flags --list-quantizers --arch --reuse ... --report | TOOLS.md:83-110 | TOOLS.md:83-111 | CONFIRMED | all present; new `--changed FILE` (with --reuse) only in rad_convert.cpp:238-240 usage; D3 --reuse match rule unchanged |
| 3.64 | header quant = quant_method[_fmt]; only --arch overrides | core/format/checkpoint.cpp:258-264 | checkpoint.cpp:267 (call), :272-282 `quant_descriptor()` | CHANGED | derivation same, moved; "only --arch overrides" now false: `--set quantization_config.*` re-derives it (rad_convert.cpp:407-416); gptq/auto-round/quark imports rename keys to `imported_quantization_config.*` (checkpoint.cpp:547-553) and record the serving plugin's descriptor (rad_convert.cpp:486-491). D8 OK: "exl3" not an import method; exact (arch, quant) match unchanged (core/build/startup.cpp:183) |
| 3.65 | RADIANCE_CALIB_DIR / CALIB_TOKENS 65536 | GUIDE.md:1467-1468 | GUIDE.md:1832-1833 | CONFIRMED | — |
| 3.66 | RADGRAM2 files; per-expert down Grams | MOE-W4.md:287-310 | MOE-W4.md:285-308 | CONFIRMED | — |
| 3.67 | quantiser reads calib=DIR itself | rad_quant.h:13-16 | rad_quant.h:13-16 | CONFIRMED | — |
| 3.68 | expert's own Gram if ≥ 4·K rows | libquant/lq_hess.h:38-64 | lq_hess.h:38-64 | CONFIRMED | — |
| 3.69 | gram_accum op | OPS.md:224-240 | OPS.md:293-309 | CONFIRMED | — |
| 3.70 | imatrix via RadQuantWeight::importance | rad_quant.h:44-46 | rad_quant.h:44-46 | CONFIRMED | — |
| 3.71 | ≤ 8 planes, role ≤ 23 chars, core dtypes | rad_encoding.h:91-101 | rad_encoding.h:91-101 | CONFIRMED | — |
| 3.72 | TILED planes cut on block boundaries under TP | core/format/encoding.cpp:371-412 | encoding.cpp:421-462 | CONFIRMED | D4: block 0 → whole plane on every slice (:436-440); column cut must land on a byte (:453-459) |
| 3.73 | TABLE planes copied whole to every rank | encoding.cpp:375-377 | encoding.cpp:425-427 | CONFIRMED | — |
| 3.74 | `options` is recipe text | rad_format.h:83 | rad_format.h:83 | CONFIRMED | — |

### §4 Architecture plugins (qwen4exp) and §9 Gaps

Paths are relative to the radiance root. "moe.h" = arch/common/rad_block_moe_fp8.h, "q4x.cpp" = arch/qwen4exp_fp8/qwen4exp_fp8.cpp.
The bare cites in §4 (":683" ... ":1543-1559", ":1550-1573", ":423-478" ... ":767-771") all point to **moe.h**. They cannot be q4x.cpp, which has only 1469 lines at 1.0.8. At 1.0.8, line 1543 of moe.h is `MoeFP8::step` and line 1575 is the `rad_route_report` call.

#### 6b394cc — "qwen4exp: the plugin owns the expert width split; geom_from does not refuse it first"
- **What moved:** `geom_from` (rad_arch.h) gained the parameter `bool arch_splits_ff = false`. When it is set and the world does not divide `n_ff_all`, `g.n_ff` is left whole and the refusal is skipped (`divide_or_fail` does not run). Widths the world divides behave as before, and every other plugin keeps the refusal.
- **Who opts in:** q4x.cpp `declare()` now passes `true`.
- **Why:** `geom_from` divided n_ff by the world before the plugin's own split ran. 640 does not divide by 3, so tp3 was refused even though the plugin's split by whole fp8 blocks and parity serves any world up to 5 blocks.
- **At 1.3.0:** the call also passes `attn_world_ok=true` (from 4596301), at q4x.cpp:326-327. A vendored copy (D8) must pass both, or tp3 and other uneven worlds are refused.

| # | claim (≤ 12 words) | 1.0.8 cite | 1.3.0 location | verdict | what changed / decision touched |
|---|---|---|---|---|---|
| 4.1 | RAD_ARCH_PLUGIN(qwen4exp_fp8,"qwen4exp","","0.1.0") | q4x.cpp:1465 | q4x.cpp:1569 | CONFIRMED | still inside `#ifndef RAD_ARCH_NO_EXPORTS` (:1567) |
| 4.2 | formats come from recipe, read per weight | q4x.cpp:44-62 | q4x.cpp:44-62 | CONFIRMED | |
| 4.3 | same, in docs | docs/ARCHITECTURES.md:273-279 | :274-280 | CONFIRMED | |
| 4.4 | Flash-Next geometry | ARCHITECTURES.md:254-262 | :255-263 | CONFIRMED | |
| 4.5 | 176.94 B total / ~6.04 B active | docs/qwen3-8-flash-next-gguf-quant-recipes.md:7-9 | same | CONFIRMED | file unchanged |
| 4.6 | active-parameter table | same doc:150-172 | same | CONFIRMED | file unchanged |
| 4.7 | block list (hc, PLE, QSA, GDN, gated attn, MoE, MTP) | q4x.cpp:12-26 | :12-26 | CONFIRMED | |
| 4.8 | block header includes | q4x.cpp:97-102 | :97-102 | CONFIRMED | PLE/QSA includes at :103-104 (both versions) |
| 4.9 | MTP head has its own 512 routed experts | q4x.cpp:1309 | q4x.cpp:1413 | CONFIRMED | |
| 4.10 | MTP experts quantised by the same rule | recipe:41-47 | same | CONFIRMED | recipe unchanged |
| 4.11 | 10 protected experts `cast dtype=bf16` | recipe:30-39 | same | CONFIRMED | |
| 4.12 | routed experts gptq w4nl g64 e4m3 fwht128 | recipe:47 | same | CONFIRMED | |
| 4.13 | trunk linears, shared expert, lm_head int8 | recipe:63-74 | same | CONFIRMED | |
| 4.14 | hc mixing / MTP fc E4M3 rows | recipe:53-58 | same | CONFIRMED | |
| 4.15 | n-gram table fp8 | recipe:77 | same | CONFIRMED | |
| 4.16 | expert names/shapes gate_up [1280,2560], down [2560,640] | moe.h:876-886 | moe.h:957-967 | CONFIRMED | |
| 4.17 | down codes view declaration | moe.h:948-951 | moe.h:1030-1033 | CONFIRMED | |
| 4.18 | router `gemm_nt` | moe.h:683 | moe.h:742 | CONFIRMED | at ≤64 rows it is now usually replaced by the fused op (see 4.19) |
| 4.19 | router_topk_scatter if M·top_k ≤1024, else topk+scatter | moe.h:688-729 | moe.h:747-788 | CHANGED | Cited lines are identical, but there is a third form. `router_gemm_topk_scatter` runs the router GEMM, top-k, sort and the shared-expert gate in one launch at ≤64 rows (`kMoeRouterFusedRows`). It is declared at moe.h:1410-1454 and issued at :1527-1536. libr4d serves it (r4d_rows.cpp:4897). **D9** |
| 4.20 | had_quant_act_fp8, skipped if upstream wrote rotated codes | moe.h:994-998 | moe.h:1076-1080 | CONFIRMED | `if (!hr_in)` at :1075 |
| 4.21 | moe_gemm_q gate_up a_order=token parts=2 | moe.h:1015-1026 | moe.h:1098-1120 | CHANGED | At tp4 (four classes) the op is `moe_gemm_q_mod4` with 4 tables and N_1..N_3/K_1..K_3 (:1101-1110). Other worlds keep `moe_gemm_q` verbatim (:1111-1118). **D6, D7** |
| 4.22 | gated quantiser gated_had_quant_fp8 | moe.h:1031-1032 | moe.h:1125-1126 | CONFIRMED | op name chosen at rad_fp8.h:632 |
| 4.23 | gated_had_quant_fp8 op row | docs/OPS.md:306 | OPS.md:375 | CONFIRMED | |
| 4.24 | moe_gemm_q down a_order=sorted | moe.h:1033-1044 | moe.h:1127-1149 | CHANGED | Same as 4.21: `moe_gemm_q_mod4` at tp4 (:1130-1139). **D6, D7** |
| 4.25 | protected bf16 moe_gemm pair on trailing offsets | moe.h:1045-1137 | moe.h:1150-1243 | CONFIRMED | comment only: extra 64-wide tiles go to the lowest ranks |
| 4.26 | moe_gather | moe.h:1234 | moe.h:1340 | CONFIRMED | |
| 4.27 | 2-rank fused ar_gather_hc_write kernel row | libr4d/r4d_rows.cpp:4701-4716 | r4d_rows.cpp:5431-5446 | CONFIRMED | new row `ar_gather_hc_write_4rank` at :5448 |
| 4.28 | GUIDE: all-reduce rides the residual write, gather in front | docs/GUIDE.md:676-680 | GUIDE.md:836-846 | CHANGED | Reworded. Now "at two ranks" either wire; at four ranks the exact wire has a fused form for decode and verify steps. **D9** |
| 4.29 | tp4 runs unfused all_reduce | moe.h:1304 | moe.h:1458 | CHANGED | The declaration line is identical. At tp4, decode/verify steps on the exact wire now skip it: `gather_taken` (rad_fp8.h:417) and `ar_wide_write_exact` (:367). It still runs for prefill and the lossy wire. **D9** |
| 4.30 | fused all-reduce forms are two-rank only | ARCHITECTURES.md:331-337 | ARCHITECTURES.md:336-351 | CHANGED | Now "two-rank, except the exact write": `ar_hc_write_4rank` and `ar_gather_hc_write_4rank` for decode/verify rows up to max_num_seqs × (1 + spec). Prefill and lossy messages stay unfused. **D9** |
| 4.31 | rad_route_report after the passes | moe.h:1550-1573 | moe.h:1742-1765 | CONFIRMED | block header, not q4x.cpp. The call itself is at 1.0.8:1575 / 1.3.0:1767 |
| 4.32 | recurrent blocks split decode/prefill by n_seq_decode | docs/PLUGIN.md:247-264 | PLUGIN.md:249-266 | CONFIRMED | |
| 4.33 | RadBatch n_seq_decode / n_tok_decode | abi/rad_runtime.h:108-129 | same | CONFIRMED | |
| 4.34 | MoE runs passes of c.rows, no decode split | moe.h:1543-1559 | moe.h:1735-1751 | CONFIRMED | block header. Still no `n_seq_decode` in moe.h |
| 4.35 | probe reads first non-bf16 expert's encoding | moe.h:496-530 | moe.h:550-584 | CONFIRMED | callers now fill the source through `moe_expert_sources()` (moe.h:516), stacked or per-expert |
| 4.36 | format_of → 0..7 | moe.h:423-478 | moe.h:452-507 | CONFIRMED | |
| 4.37 | kServed[8] dtype strings | moe.h:428-429 | moe.h:457-458 | CONFIRMED | |
| 4.38 | `if (!rad_enc_is(&e,"affine")) return RAD_E_DTYPE;` | moe.h:442 | moe.h:471 | CONFIRMED | |
| 4.39 | refusal message listing served forms | moe.h:518-527 | moe.h:572-581 | CONFIRMED | |
| 4.40 | one table = one encoding; only 5/6/7 vary | moe.h:579-641 | moe.h:636-698 | CONFIRMED | |
| 4.41 | experts declared as 8 runs | moe.h:763-766 | moe.h:822-825 | CHANGED | Comment unchanged, but the run loops now use `ncls` classes: 16 runs at four classes (moe.h:994-1056, `ncls` set at :845). **D6, D7** |
| 4.42 | codes view `codes,table` | moe.h:801-809 | moe.h:882-890 | CONFIRMED | |
| 4.43 | scale view `scale,scale.1` | moe.h:846-854 | moe.h:927-935 | CONFIRMED | |
| 4.44 | ROW/COL shard + rad_weight_shard_span parity split | moe.h:835-860 | moe.h:916-941 | CHANGED | The span lambda now uses `qlo/qhi` (`ff_lo` parity or `ff_lo4` four-class, :940). tp2 is identical. **D7** |
| 4.45 | rad_weight_shard_span ABI | abi/rad_builder.h:354-365 | rad_builder.h:367-378 | CONFIRMED | plus an additive `rad_weight_shard_span_parts` at :380-388 |
| 4.46 | expert count per rank must be even | moe.h:767-771 | moe.h:826-830 | CONFIRMED | `ne % 2` check kept. The quantised-count (`n_reg`) parity check is now skipped at four classes (:710) |
| 4.47 | author quote: new encoding needs kernel lib + arch | docs/PLUGIN.md:436-440 | PLUGIN.md:437-441 | CONFIRMED | |
| 4.48 | arch picks dtype string from format_of first | moe.h:843-845 | moe.h:924-926 | CONFIRMED | |
| 4.49 | "Adding a quant is a quantiser plugin…" | spec.md:803-804 | spec.md:804-805 | CONFIRMED | |
| 4.50 | second ("qwen4exp","") plugin refused | core/plugin/loader.cpp:232-240 | same | CONFIRMED | |
| 4.51 | rad-convert --arch ID | docs/TOOLS.md:90 | TOOLS.md:90 | CONFIRMED | |
| 4.52 | descriptor from quantization_config.quant_method (+_fmt) | core/format/checkpoint.cpp:261-264 | checkpoint.cpp:275-281 (`quant_descriptor`, called :267) | CONFIRMED | moved into a function, same rule. New: `rad-convert --set quantization_config.*` re-derives it (tools/rad_convert.cpp:407-415). **helps D8** |
| 4.53 | $RADIANCE_HOME search path, earlier home shadows | PLUGIN.md:468-470 | PLUGIN.md:469-471 | CONFIRMED | |
| 4.54 | block headers installed as `<arch/rad_arch.h>` etc. | PLUGIN.md:371-374 | PLUGIN.md:372-375 | CONFIRMED | |
| 9.1 | core validates only plane-0 codes for custom scheme | abi/rad_encoding.h:283-287 | same | CONFIRMED | file unchanged |
| 9.2 | decode hook used by oracle, kbench, convert | abi/rad_quant.h:80-89 | same | CONFIRMED | file unchanged |
| 9.3 | loader registers quantiser decoders | core/format/encoding.h:44-48 | same | CONFIRMED | |
| 9.4 | planes row-major, codes block[0]==1 | rad_encoding.h:21-37 | same | CONFIRMED | |
| 9.5 | fixed core dtype table | abi/rad_types.h:118-147 | same | CONFIRMED | file unchanged |
| 9.6 | plane dtype never plugin-private | rad_encoding.h:97 | same | CONFIRMED | |
| 9.7 | TP cuts refused only on declared block splits | core/format/encoding.cpp:371-412 | encoding.cpp:421-462 | CONFIRMED | |
| 9.8 | shard spans for tile integrity | rad_builder.h:354-365 | rad_builder.h:367-378 | CONFIRMED | |
| 9.9 | fixed per-slot expert size, refused otherwise | core/format/radfile.cpp:408-414 | same | CONFIRMED | file unchanged |
| 9.10 | every expert slot matches expert 0 | radfile.cpp:827-840 | same | CONFIRMED | |
| 9.11 | WTAB entries share dtype/shape | rad_runtime.h:274-276 | rad_runtime.h:280-282 | CONFIRMED | |
| 9.12 | per-layer/per-projection format choice | docs/MOE-W4.md:210-213 | MOE-W4.md:209-212 | CONFIRMED | |
| 9.13 | in-tree variation only within formats 5–7 | moe.h:609-626 | moe.h:666-683 | CONFIRMED | |
| 9.14 | protected experts bf16 path, rest even | MOE-W4.md:323-341 | MOE-W4.md:321-339 | CONFIRMED | doc unchanged; the code drops the parity need at tp4 (moe.h:710) |
| 9.15 | transforms input-side only (fwhtN, perm, givens) | rad_encoding.h:59-76 | same | CONFIRMED | |
| 9.16 | custom scheme valid if codes block[1]≥1 | rad_encoding.h:287 | same | CONFIRMED | |
| 9.17 | had_* FP8: 128 wide, unnormalised, factor in scale | OPS.md:262-270 | OPS.md:331-339 | CONFIRMED | |
| 9.18 | residual factor folded into stored scale | MOE-W4.md:102-107 | MOE-W4.md:101-106 | CONFIRMED | |
| 9.19 | rotation width must divide per-rank n | OPS.md:280-282 | OPS.md:349-351 | CONFIRMED | |
| 9.20 | TABLE planes replicated in every slice | encoding.cpp:375-377 | encoding.cpp:425-427 | CONFIRMED | |
| 9.21 | table values compared at relayout | libr4d/r4d_layout.cpp:720-734 | r4d_layout.cpp:743-757 | CONFIRMED | |
| 9.22 | table entries exact in E4M3 | libr4d/r4d_moe.hip:646-650 | r4d_moe.hip:676-680 | CONFIRMED | |
| 9.23 | moe_gemm_q activations must be E4M3 | r4d_moe.hip:2156-2157 | r4d_moe.hip:3179-3180 | CONFIRMED | now in a shape function shared by both class counts (parity and mod 4) |
| 9.24 | inter-GEMM gated quantiser writes E4M3 | moe.h:1031-1032 | moe.h:1125-1126 | CONFIRMED | |
| 9.25 | moe_gemm bf16: one table, no parity/parts | OPS.md:604-609 | OPS.md:730-735 | CONFIRMED | |
| 9.26 | bf16 experts whole-expert only | moe.h:862-873 | moe.h:943-954 | CONFIRMED | |
| 9.27 | isarate measures fp8 WMMA only | libr4d/isa/isarate.hip:82-85 | same | CONFIRMED | file unchanged |
| 9.28 | parity split: scale blocks + rotation on 128 cols | ARCHITECTURES.md:289-296 | ARCHITECTURES.md:290-297 | CONFIRMED | tp4 four-class text added at :311-322 |
| 9.29 | libref moe_gemm_q: codes, scales separate operands | libref/ref_moe.cpp:176-189 | ref_moe.cpp:176-190 | CONFIRMED | edited to add the four-class path (13 operands); codes and scales still separate |
| 9.30 | partial plane selection "decodes to nothing" | encoding.cpp:349-351 | encoding.cpp:399-401 | CHANGED | New exception before it (:349-398): a `scale` + `scale.N` chain now decodes, as the product of its levels. EXL3's `suh`/`svh` roles are not in that chain, so **D6** (one operand, all planes) still holds |
| 9.31 | only a layout hook can join planes | core/build/rad_builder.cpp:1276-1283 | rad_builder.cpp:1404-1411 | CONFIRMED | |
| 9.32 | quantiser input is the f32 logical weight | rad_quant.h:72-78 | same | CONFIRMED | |
| 9.33 | unmatched tensors must be bf16/f16/f32/block-FP8 | TOOLS.md:133-136 | TOOLS.md:134-137 | CONFIRMED | Doc text identical but stale. Since 1.2.1 the core imports GPTQ, AutoRound (GPTQ packing) and Quark MXFP4 (checkpoint.cpp:514-552). EXL3 is not imported |
| 9.34 | --plan-only | TOOLS.md:107 | TOOLS.md:108 | CONFIRMED | |
| 9.35 | stable sort, byte-level reproducibility | r4d_moe.hip:47-54 | same | CONFIRMED | |
| 9.36 | fusions must be byte-exact | abi/rad_abi.h:488-490 | same | CONFIRMED | |
| 9.37 | -ffp-contract=off in libr4d | libr4d/CMakeLists.txt:36-40 | same | CONFIRMED | |
| 9.38 | -ffp-contract=off in libquant | libquant/CMakeLists.txt:20-23 | same | CONFIRMED | file unchanged |
| 9.39 | converts give same bytes on any machine | spec.md:814-816 | spec.md:815-817 | CONFIRMED | |
| 9.40 | scratch via hook, sized at declare for band top | rad_abi.h:151-153 | same | CONFIRMED | |
| 9.41 | LDS ≤64 KiB, args ≤768 B, no printf | PLUGIN.md:205-213 | PLUGIN.md:207-215 | CONFIRMED | |
| 9.42 | code tables passed as kernel args | libr4d/r4d_args.h:281-313 | same | CONFIRMED | |
| 9.43 | one stored layout for all bands | rad_builder.cpp:1221-1272 | rad_builder.cpp:1349-1400 | CONFIRMED | |
| 9.44 | non-identity layouts not on file/mapped tiers | spec.md:677-681 | spec.md:678-682 | CONFIRMED | |
| 9.45 | residency is the decode lever | r4d_moe.hip:814-820 | r4d_moe.hip:905-911 | CONFIRMED | wording: `--expert-vs-cache-ratio` is gone; the KV pool lends the experts the blocks no request holds |
| 9.46 | +8% expert bytes → −19% prefill | MOE-W4.md:40-43 | same | CONFIRMED | |
| 9.47 | PLUGIN.md §12 quantiser plugins | PLUGIN.md:410-440 | PLUGIN.md:411-441 | CONFIRMED | |
| 9.48 | "quantiser plugin can invent one" | rad_encoding.h:12-17 | same | CONFIRMED | |
| 9.49 | "A container can hold only what some kernel reads" | TOOLS.md:184-191 | TOOLS.md:185-192 | CONFIRMED | |
| 9.50 | "Adding a quant is a quantiser plugin…" | spec.md:803-804 | spec.md:804-805 | CONFIRMED | |

Counts: 104 citations. 95 CONFIRMED, 9 CHANGED (4.19, 4.21, 4.24, 4.28, 4.29, 4.30, 4.41, 4.44, 9.30), 0 GONE, 0 WRONG-AT-1.0.8. Some 1.0.8 cites were slightly loose but not wrong: 4.19 ends partway through a declaration, 4.31 stops before the call, and 9.20 shows only the TABLE case.
