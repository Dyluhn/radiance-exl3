# Where this plugin's code comes from

`qwen4exp_exl3.cpp` and `rad_block_moe_exl3.h` are COPIES of two radiance files, changed only where
EXL3 experts need it. Radiance is not modified (PLAN D11): the copies are built out of tree against an
installed radiance of the pinned release (`RADIANCE_VERSION` at the repository root).

| this directory | radiance file it was copied from |
|---|---|
| `qwen4exp_exl3.cpp`   | `arch/qwen4exp_fp8/qwen4exp_fp8.cpp` |
| `rad_block_moe_exl3.h` | `arch/common/rad_block_moe_fp8.h` |

Release `v1.12.0`, commit `5a9ce4912bb4ae5315e4a177f9e7d5b10d035766` (the copies were made at `v1.4.0`,
`dc339b4cd239f7124700dd1f52213f73e8939a14`, and both files stayed byte-identical through `v1.6.0`), fetched from
`https://codeberg.org/StillDeadcode/radiance.git`. The first commit that added these files holds them
byte for byte as upstream ships them; every later change is in git history, so
`git log -p -- architectures/qwen4exp_exl3/` is the whole port.

## The pinned bytes

`tests/upstream_drift.sh` reads this block (`sha256  path`, the `sha256sum` format) and fails if the
release it fetches holds anything else at those paths:

```upstream-sha256
0e63d8f02ac85dde132ca753117d47389dcac423977e864287ac712517ca2908  arch/qwen4exp_fp8/qwen4exp_fp8.cpp
cb6680217d5a9c8f317431ebef6fe0dfb7ba6cdb98427988b114c31c978e92a4  arch/common/rad_block_moe_fp8.h
```

## Called, not copied

Every other block the model uses -- the gated residual, attention, delta net, PLE, QSA and the MTP
head (`arch/common/rad_block_mtp_hc.h`) -- is the INSTALLED header, compiled as it ships. The MTP head
names the fp8 MoE block as `MoeFP8`; in this plugin's translation unit that name is an alias for
`MoeEXL3` (see the includes in `qwen4exp_exl3.cpp`), so the head's routed experts are EXL3 with no
copy of the head. `CMakeLists.txt` refuses an installed radiance that is not the pinned release, so
a release bump is always deliberate: run the drift test, and read the release's changes to
`arch/common/` -- a head that asks the block for more than `MoeEXL3` has fails to compile here.

## Moving to a new radiance release

1. Put the new release in `RADIANCE_VERSION`.
2. `tests/upstream_drift.sh` names each copied file that changed and prints the `git diff` to port.
3. Port that diff into the copy here, then replace the hash above with the new release's.

## Ported releases
- **1.3.0 -> 1.4.0** (dc339b4): the next-layer expert prefetch -- `MoeFP8::xp_next` / `xp` / `xp_parts` /
  `xp_count_at`, `rad_expert_prefetch` at the router input, `rad_expert_prefetch_step` ahead of the routed
  GEMMs, `rad_expert_prefetch_mark` after down, the layer link in `declare`, and `step_rows` (the 1.4.0 MTP
  head's history pass calls it). EXL3 difference: the parts are not "the four handles" but the planes of
  this build that sit in EXACTLY ONE weight table (the core's rule, core/runtime/xprefetch.cpp `xp_bind`;
  at most 4) -- with the default split decode form (`QWEN4EXP_EXL3_PREP_SPLIT=1`) that is down's codes and
  svh only, with `=0` gate_up's codes and svh as well (`MoeEXL3::declare`, "THE LAYER AS THE PREFETCH SEES IT").
- **1.4.0 -> 1.5.0** (b93b531): nothing to port. Both copied files are byte-identical upstream (their 1.4.0 hashes
  still hold). The installed headers this plugin compiles in changed: `arch/common/rad_fp8.h` (ParoQuant's rotated
  linears, chosen by a weight's encoding: `linear_is_givens` / `_paro_mxfp4` / `_paro_w5`; for any other encoding
  `reads_x` and the `gemm_nt_q` parameters are as before), `rad_block_gdn_fp8.h` (the in-projection's per-tensor
  rotation spans) and `rad_block_mlp_fp8.h` (no gated fold for a paro w5 gate_up). The container's trunk is int8 /
  fp8, so none of it is taken. Kernels and quantiser build byte-identical to 1.4.0's.
- **1.5.0 -> 1.6.0** (b068190): nothing to port. Both copied files are byte-identical upstream (their 1.4.0 hashes
  still hold). `arch/common/` changed only `rad_block_vit.h` (vision; not compiled here). The plugin ABI is 16
  (`RAD_ABI_VERSION`): the op vocabulary gains the image generator's ops and `all_reduce` an optional `overlap`,
  which this plugin does not pass, so a rebuild is the whole port.
- **1.6.0 -> 1.7.0** (1e968f6): three-way merged both copies against b068190. Took the prefill span and
  pipeline (`max_mixed_tok`, frozen span choices in sizing declares, pass-sized connection reads and QSA score
  windows, bounds uploads, piecewise mixer/MoE issue and lane-1 all-reduces at `overlap=1`); the serial and MTP
  paths pass those bounds to the installed QSA block too. `MoeEXL3` now exposes `pass(..., own_ar)`, `issues_ar`,
  `issue_ar`, `ar_overlap`, `op_ar_ov` and `route_report` with the upstream meanings. Serial passes still select
  the ordinary collective; the pipeline owns each deferred collective and reports routing once after all passes.
  Also took the MTP gather's rope-table liveness edge, shared layer-schedule parser, common whole-block slice
  helper, optional router-schema guards, conditional protect parameters, thread-safe debug-once bookkeeping,
  updated diagnostics/comments and removal of the unused `n_run` member.
  EXL3 decisions: retain the per-plane tables, f16 intermediate and internal rotations; derive bf16 mode from
  the trunk and feed its int8 codes directly; retain imported-format refusals, per-layer trellis/codebook checks,
  protected experts/calibration basis, all split/GD/mod4 forms and thresholds, EXTRA_RANK arm, prefill-stage
  control, single-reader prefetch parts and the shared-expert lane (joined before the gather, including inside
  pipelined passes). The new slice helper preserves EXTRA_RANK's chosen leading rank. The installed MTP head
  still uses the `MoeFP8 = MoeEXL3` alias and `step_rows`; no extra fp8-only config flags or activation planes
  are needed. ABI 17 removes `RadKernelInfo::describe`, so the test rows omit that slot; the recording builder
  implements `rad_op_declared` as a loaded-schema lookup, like core. No upstream functional change was omitted;
  fp8-only text and code retain their existing EXL3 equivalents. No existing numeric reference or behavior test
  was relaxed; the four-class mutation now targets the equivalent slice-helper call. See `NOTES.md` for the
  conflict audit, checks and numerical limits.
- **1.7.0 -> 1.12.0** (5a9ce49): three-way merged both copies against 1e968f6 (`git merge-file --diff3`, ours = the
  1.7.0 port, base = upstream 1.7.0, theirs = upstream 1.12.0). The architecture had one conflict: `Layer` keeps
  `MoeEXL3 mlp` and takes upstream's `HcProject proj` (the control vector's projection after every block's write,
  inert without `--control-vector`); the cleanly merged hunks take the `--kv-cache-dtype fp8_v4 / k6_v4` layouts and
  `--draft-coupling`. The MoE block had 16 conflicts. Taken from upstream, format-independent: the GLM routing fields
  (`sigmoid_router`, `router_fuse`, `route_scale`, `shared_ungated`, `swiglu_limit`, `swiglu_quant`,
  `Src::router_bias`, the installed `GlmRouter`), the fused sigmoid scatter (`op_sigmoid_scatter` and its shared-gate
  fill, also in the three EXL3 issue orders of `pass`), the sort's slot table for the gather (`Wire::inv`,
  `moe_scatter_inv` / `moe_gather_inv`, used only where a caller allocates `inv`), the prefetch guard for the sigmoid
  router, and the shared arm's clamp parameter. Kept as EXL3 has them: every fp8 stored-format piece (`format_of`,
  `kServed`, the format flags and the scale views); upstream's new formats there (OCP MXFP4, block fp8 at f32 scales)
  are fp8-block formats and are not taken, nor the clean hunk refusing MXFP4 at four classes. New refusal: a routed
  `swiglu_limit` (gate_up's epilogue has no clamp). Qwen3.8-Flash-Next sets none of the GLM fields, so its graph is
  the 1.7.0 port's. The installed MTP head asks `MoeEXL3` for nothing new. ABI 17 -> 27 (abi/ diff 1e968f6..5a9ce49):
  every change a plugin of this kind compiles against is an addition (RadBuildCtx / RadArchProbe fields,
  `RAD_LAYOUT_ANY`, the optional `rad_kernel_chain` and `rad_kernel_reference` exports, the aux-vocab header slots,
  new KV kinds and name-map kinds); the kernels and the quantiser compile unchanged. Checks: the toolchain first
  rebuilt v1.3.0 against 1.7.0 byte-identical to the published plugin home (all four `.so`); this tree builds
  warning-free; `ctest -L host` 35/35; `tests/upstream_drift.sh` PASS at v1.12.0 and FAIL at v1.7.0. On 2x R9700, tp2,
  the production flags: greedy output identical to v1.3.0 on 1.7.0 for 8 of 8 prompts (MTP on). KL record over 6,129
  positions (11 documents: code, English and Turkish prose, math, a tool-call transcript): this port on 1.12.0 against
  v1.3.0 on 1.7.0 mean 0.0570, top-1 90.8%, perplexity ratio 0.998; an iq4r container on the in-tree `qwen4exp_fp8`
  (no plugin), 1.7.0 against 1.12.0, mean 0.0627, top-1 90.9%, ratio 1.013 -- the engine moves both by the same
  amount, most on long documents cut from files and least on short hand-written ones, which the chat-format greedy
  prompts do not reach. Speed, one stream with no other load: 206 / 172 / 199 tok/s (code / Turkish / English prose)
  against 1.7.0's 211 / 173 / 198, a 37K-token prefill 5,131 against 5,104 tok/s.
