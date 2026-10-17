# Where this plugin's code comes from

`qwen4exp_exl3.cpp` and `rad_block_moe_exl3.h` are COPIES of two radiance files, changed only where
EXL3 experts need it. Radiance is not modified (PLAN D11): the copies are built out of tree against an
installed radiance of the pinned release (`RADIANCE_VERSION` at the repository root).

| this directory | radiance file it was copied from |
|---|---|
| `qwen4exp_exl3.cpp`   | `arch/qwen4exp_fp8/qwen4exp_fp8.cpp` |
| `rad_block_moe_exl3.h` | `arch/common/rad_block_moe_fp8.h` |

Release `v1.7.0`, commit `1e968f6e1d86e2544455b00e9e57c09690d9ea10` (the copies were made at `v1.4.0`,
`dc339b4cd239f7124700dd1f52213f73e8939a14`, and both files stayed byte-identical through `v1.6.0`), fetched from
`https://codeberg.org/StillDeadcode/radiance.git`. The first commit that added these files holds them
byte for byte as upstream ships them; every later change is in git history, so
`git log -p -- architectures/qwen4exp_exl3/` is the whole port.

## The pinned bytes

`tests/upstream_drift.sh` reads this block (`sha256  path`, the `sha256sum` format) and fails if the
release it fetches holds anything else at those paths:

```upstream-sha256
04f707dd93a7e71716f55204091a65562ca16e7b36e4bc0a3a91c8a9f111ad15  arch/qwen4exp_fp8/qwen4exp_fp8.cpp
cd59f7d882b1220e9380870ae3483801d5cdb8455f332f6418a626502a10646b  arch/common/rad_block_moe_fp8.h
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
