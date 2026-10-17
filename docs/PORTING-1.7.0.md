# Kernel and quantiser port to radiance 1.7.0

Audit on 2026-10-17: upstream `b068190` (1.6.0) to `1e968f6` (1.7.0), ABI 16 to 17.
The release and ABI pins were already updated in `e1cefa9`. Architecture changes are handled separately.

| Upstream area | Effect here |
|---|---|
| `rad_abi.h`, `rad_types.h` | Remove `RadDescribeFn`, `RadKernelInfo.describe`, `RadLaunchDesc` and its two size constants. No fields were added or reordered before the removed trailing field. EXL3's rows and test kernels use named initializers and never implemented `describe`; its HIP files use none of the removed declarations. No launch adapter or argument block needs changing. |
| `rad_abi.h` operand descriptions | Add sampler parameter fill and prefix/history index flags. EXL3's routing operands still use `RAD_FILL_INDEX` and `RAD_OPD_F_IDX_CU`; the new flags describe other ops. |
| `rad_plugin.h` | Fix E5M2 rounding/NaNs and integer stores (round to nearest even, saturate, handle NaNs, support U16). EXL3 uses neither changed helper; its f16/bf16 conversions and arithmetic stay unchanged. Rebuilding uses the installed header's definitions. |
| `rad_device.h`, `rad_encoding.h`, `rad_quant.h`, `rad_format.h` | Comments clarified; declarations, encoding validation, canonical bytes, quantiser hooks and file format stay unchanged. EXL3's custom decoder and whole-weight row block already satisfy the contract. |
| New `rad_planes.h` | Offers a shared joined form for an operand selecting several planes, with matching layout/relayout/unrelayout helpers. EXL3 selects one plane per operand and keeps its existing stored trellis record, so this helper does not apply. |
| `docs/PLUGIN.md` | Removes `describe`, retains the existing optional hooks, and documents the engine's `hipMemset`/`hipStreamSynchronize` interposition. EXL3 already launches via the supported HIP entry points; no new export is required. |

Compared the matching upstream implementations: `libr4d/r4d_rows.cpp`, `r4d_plugin.h`, `r4d_emit.h`,
`r4d_layout.cpp`; `libref/ref_registry.cpp`, `ref_common.h`, new `ref_planes.h`; and
`libquant/lq_registry.cpp`. Upstream removes its describe adapters and uses the joined-plane helpers
where operands select several planes. Neither change requires an EXL3 arithmetic or layout change.
New upstream ops, sampler fills, broader reference dtype support, reference transcendental changes,
OpenMP work and quantisation optimisations are not copied: EXL3 defines its own ops and imports already
quantised planes. In particular, its reference SiLU keeps its existing `expf` calculation.

The standalone reference harness now respects `RAD_WITH_HIP=OFF` when the installed prefix advertises
HIP; otherwise a host without ROCm could not configure it. The speed source-pin refusal test now
declares its existing Docker dependency, as the neighbouring refusal tests do. It previously failed
on this host at the earlier missing-Docker check instead of reaching source validation.

Current README requirements, image and prefix examples name 1.7.0/ABI 17 and direct 1.6.0 users to
v1.2.0. Measurements and numbered release history retain their original versions. There was no
Unreleased entry in this checkout, so one was added without renaming a numbered release.
Generic kernel-check/calibration build-image defaults now use 1.7.0. The P4b profile launchers
still name 1.4.0 because they reproduce measurements with specific historical plugin homes.

No EXL3 numerical change was introduced. Device execution, engine output identity, quality and speed
still need the coordinator's GPU runs; successful compilation does not establish those results.

## Validation

GCC 16.2.1, Release, installed radiance 1.7.0 prefix; logs remain in the ignored build directories.
Set `RADIANCE_PREFIX` to that prefix before using these commands.

- Configured with `cmake -S . -B build-host -DCMAKE_PREFIX_PATH="$RADIANCE_PREFIX"
  -DCMAKE_BUILD_TYPE=Release -DRAD_WITH_HIP=OFF`. Built the targets outside `tests/arch`:
  `exl3_common exl3_common_shared exl3_decode_test exl3_layout_test exl3_link_test exl3_quant
  exl3_quant_test exl3ref exl3ref_run layout_probe layout_probe_badinv exl3r4d_layout_test
  exl3r4d_rowpick`. Build exit 0, zero compiler or CMake warnings (`build-host/build-owned-final.log`).
- `ctest --test-dir build-host -L host -E '^arch_' --output-on-failure`: exit 0,
  **17 passed, 3 skipped, 0 failed**, 98.08 s (`build-host/ctest-owned-final.log`). Skips:
  `exl3_ref_vs_trellis_py`, `exl3_real_extract`, `exl3ref_ops_doc`, because their optional
  external reference, tensor extract or source-tree configuration was not provided.
- `EXL3_STEP0_PROBES=ON`: built all five probe plugin targets, exit 0, zero warnings
  (`build-host/build-probes.log`). Both production host plugins return ABI 17 via their exports.
- Standalone builds from `git archive` against the installed prefix: quantiser and reference
  configure/build/install exit 0 with zero warnings; reference schema test 1/1 passes
  (`build-host/clean-final-quant.log`, `build-host/clean-final-ref.log`). The GPU-library
  standalone host build correctly leaves device code out and builds its host layout checks.
- Shell syntax checks pass for the five scripts whose default build image changed; `git diff --check` passes.
- Full bridge requested with `.hip-build-request`: **result exit 1**, raw build exit 2 and container
  host-test exit 8, **zero warnings** (`.hip-build.log`). The GPU library, its row-table check,
  kernel-argument check and twin-kernel check pass. Outside `tests/arch`, container host tests have
  **8 passed, 15 skipped, 0 failed**; Python dependencies are unavailable there. Overall container
  totals: 10 passed, 15 skipped, 8 failed, 1 not run.

The bridge's build failures are confined to `tests/arch`: its static harness lacks a stub for the
new `rad_op_declared` call, and `router_rows.cpp` still initializes the removed trailing
`RadKernelInfo` field. They were not edited. The local host run excludes all eleven architecture
tests: `arch_stage`, `arch_static`, `arch_static_split`, `arch_static_gd`, `arch_static_lane`,
`arch_static_gd6`, `arch_static_gd_o0`, `arch_static_nostage`, `arch_static_default`,
`arch_router_rows`, `arch_duplicate_key`. A full bridge exit 0 needs the architecture worker's fixes.

No merge conflicts occurred. The two host-only compatibility failures found during validation
were reproduced before their fixes and passed or skipped with a named dependency reason afterwards.
No new machine paths were introduced; existing P4b launchers and hardware measurement comments
retain their historical machine provenance. No second-machine or GPU execution was available here.
