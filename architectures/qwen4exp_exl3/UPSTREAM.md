# Where this plugin's code comes from

`qwen4exp_exl3.cpp` and `rad_block_moe_exl3.h` are COPIES of two radiance files, changed only where
EXL3 experts need it. Radiance is not modified (PLAN D11): the copies are built out of tree against an
installed radiance of the pinned release (`RADIANCE_VERSION` at the repository root).

| this directory | radiance file it was copied from |
|---|---|
| `qwen4exp_exl3.cpp`   | `arch/qwen4exp_fp8/qwen4exp_fp8.cpp` |
| `rad_block_moe_exl3.h` | `arch/common/rad_block_moe_fp8.h` |

Release `v1.4.0`, commit `dc339b4cd239f7124700dd1f52213f73e8939a14`, fetched from
`https://codeberg.org/StillDeadcode/radiance.git`. The first commit that added these files holds them
byte for byte as upstream ships them; every later change is in git history, so
`git log -p -- architectures/qwen4exp_exl3/` is the whole port.

## The pinned bytes

`tests/upstream_drift.sh` reads this block (`sha256  path`, the `sha256sum` format) and fails if the
release it fetches holds anything else at those paths:

```upstream-sha256
0f867356e756d55bbedc77023881810beeb13abdeff0eaf373c113ca9ae6d6be  arch/qwen4exp_fp8/qwen4exp_fp8.cpp
ef60e0ddd4c06e8686c72c050ec3c88e5596384ecc6834e24241c2f84110111b  arch/common/rad_block_moe_fp8.h
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
