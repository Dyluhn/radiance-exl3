# 1.7.0 architecture port audit

Upstream: b068190326fbf1ff60b2316c43d144bdc0350552 ->
1e968f6e1d86e2544455b00e9e57c09690d9ea10. EXL3 starting point: e1cefa9.
Implementation: d694a7f; comment corrections: d95fcad. All checks below use Release builds.
Default-build coverage: ad05b18; mutation cleanup: 81caa94.

## Upstream changes accounted for

Reviewed both complete upstream diffs, then compared their changes without comments/whitespace
against the EXL3 port. All 16 architecture code hunks and all 9 MoE code hunks are accounted for,
including the architecture export's EXL3 identity/description in place of upstream's fp8 identity.
The generic comment rewrites landed where they describe this plugin; fp8-specific descriptions
retain their EXL3 equivalents. No upstream functional change was omitted.

| Area | Result |
|---|---|
| Prefill spans | `max_mixed_tok` selects spans; sizing declares inherit the real declaration's choices. Connection reads and QSA scores use pass-sized windows, with per-step device bounds. |
| Pipeline | Embedding, mixer and MoE collectives use lane 1. Blocks can hand pieces to the caller and defer their own reduction. Every residual write follows its piece's reduction; PLE remains before the following connection reads. |
| MoE | `own_ar`, `issues_ar`, `issue_ar`, overlap declarations and the separate routing handoff have upstream semantics. Serial passes use ordinary all-reduce rows. The existing shared-expert lane and its gather join remain inside each pass. |
| MTP | The installed head still aliases `MoeFP8` to `MoeEXL3` and uses `step_rows`. Its new QSA window pointer is supplied. Its first gather keeps the shared rope table live beside the carry. |
| Metadata and splitting | Shared schedule parsing replaces the local parser. `block_slice` replaces repeated arithmetic while preserving the EXL3 EXTRA_RANK override and the four-class split. Head-split messages use `rad_note`; dead `n_run` is removed. |
| Optional ops | Router fusions are requested only when their schemas are loaded. Scatter/topk protect parameters are one array cut to length. The installed attention and gated-quant helpers supply their new unfused fallbacks. |
| Debug | Routing uses installed `dbg_first`, whose mutex/set avoids shared counters and layer-slot collisions. Diagnostics follow upstream. |
| Prefetch | Read the core prefetch changes too: recording now uses `tape_host`; the single-table-per-part restriction is unchanged. EXL3's part selection, predict/step/mark order and layer links remain. |

Read the installed `arch/common/` changes, especially the attention, GDN, hyperconnection, MTP,
QSA and fp8 helpers. The new MoE methods are called with the same row/step distinction as upstream:
an output write decides on total T, a following norm on the pass's rows. No fp8-only stored-format
flags or activation planes were added to the EXL3 block.

ABI review: 17 removes `RadDescribeFn`, `RadLaunchDesc` and the final `RadKernelInfo::describe`
member; only the test kernel rows needed an initializer change here. `max_mixed_tok` is a trailing,
zero-defaulted build-context field. The optional audio-format export does not apply to this model.
New operand-fill flags, packed-plane helpers and the C YaRN implementation need no architecture
declarations. Installed float-to-integer stores now round-even/saturate, and E5M2 conversion fixes
ties/NaNs; this architecture performs no such conversion itself. `docs/PLUGIN.md` also updates
media processors and allows comma/colon kernel lists; no new export is required of this plugin.

## Three-way conflicts

Numbers are in file order from `git merge-file -p ours base theirs`, with ours at e1cefa9,
base the 1.6.0 upstream file and theirs the 1.7.0 upstream file. Both sides of all 56 conflicts
were read. Cleanly merged functional hunks were reviewed separately as described above.

| Architecture conflict | Resolution |
|---|---|
| 1 | Keep the EXL3 identity/change list; take the new upstream introduction. |
| 2 | Keep `MoeEXL3::Config`; take the trunk comment. |
| 3 | Keep the single f16 intermediate and omit fp8-only gate/input activation planes. |
| 4 | Take the shared-place explanation with the EXL3 type name. |
| 5 | Keep imported-checkpoint refusal before the new sizing-declare comment. |
| 6 | Take the shared-split explanation with the EXL3 type name. |
| 7 | Take the calibration explanation with the EXL3 type name. |
| 8 | Keep EXL3 encoding probing; the upstream fp8 format choices do not describe it. |
| 9 | Keep bf16 mode and connection quantization derived later from the trunk. |
| 10 | Take the updated parity/four-class explanation with EXL3 type names. |
| 11 | Keep whole-expert parallelism for calibration; all-bf16 experts remain refused. |
| 12 | Use upstream `block_slice`, preserving EXTRA_RANK's leading rank and four-class guard. |
| 13 | Keep direct trunk int8-code production; EXL3 reads the bf16 input. |
| 14 | Keep the pass-sized f16 intermediate; gating/rotation stays inside EXL3 kernels. |
| 15 | Remove `n_run` and obsolete bisect prose; retain stage-off reader before the final vision tower. |
| 16 | Keep the EXL3 plugin namespace, key and description. |

| MoE conflict | Resolution |
|---|---|
| 1 | Keep EXL3 file identity. |
| 2 | Keep EXL3 contract differences; correct stale D6 whole-weight prose to the existing D15 plane tables. |
| 3 | Keep EXL3 plane/class runs and fused-decode description. |
| 4 | Keep EXL3 schemas, switches, encoding parser and file-tier detection. |
| 5 | Take the protection-limit comment with the EXL3 type name. |
| 6 | Keep the Hadamard-sized slice unit. |
| 7 | Keep EXL3 mod4 support and its contract explanation. |
| 8 | Keep the always-zero external-rotation flag and EXL3 encoding state. |
| 9 | Keep the model-basis calibration tap and its f16-to-bf16 copy. |
| 10 | Keep the inert MTP-compatible `hr` member. |
| 11 | Keep `Wire::eff` as one f16 buffer. |
| 12 | Keep per-plane table arrays and EXL3 class/slot helpers. |
| 13 | Keep split/GD forms, EXL3 single-reader prefetch parts and per-projection encodings. |
| 14 | Keep the down tap's copy/cast and model-basis domain. |
| 15 | Keep EXL3 probe/refusal rules and omit fp8-only `format_of`/`kServed`. |
| 16 | Keep stage-off/decode-form members; take the new `step_rows` explanation. |
| 17 | Keep the source-probe helper; omit the fp8-only format decoder. |
| 18 | Take the shorter upstream probe explanation; retain EXL3's all-bf16 refusal logic. |
| 19 | Keep complete EXL3 encoding equality and stacked-gate checks. |
| 20 | Keep the EXL3 run layout explanation. |
| 21 | Keep external-rotation refusal and EXL3 expert-slot mapping. |
| 22 | Keep per-plane vector allocation and class-width bookkeeping. |
| 23 | Keep whole-expert parallelism; omit fp8 dtype/plane selection. |
| 24 | Keep EXL3 plane extents/sharding/run declarations; omit fp8-only rotations, scale grids and bf16-model branch. |
| 25 | Keep EXL3 op-specific row order and plane-table construction. |
| 26 | Keep EXL3 class shape parameters and f16 gated output. |
| 27 | Keep fused gate_up_down declaration and bands; omit the fp8 expert quantizer. |
| 28 | Take the shorter calibration-placement explanation, without the fp8-only outer branch's closing brace. |
| 29 | Keep the existing EXL3 whole-group validation; omit redundant fp8-only check. |
| 30 | Keep model-basis domain and bf16 input tap. |
| 31 | Keep f16 intermediate cast before the down tap. |
| 32 | Keep the cast buffer as down-tap input and the per-expert tap. |
| 33 | Take concurrent-buffer explanation, naming EXL3's `eff` instead of fp8's `egu`. |
| 34 | Keep single-reader plane selection, capped at four prefetch parts. |
| 35 | Keep stage-off declaration; take the shorter shared-arm explanation and remove duplicated pass prose. |
| 36 | Take the new `own_ar` pass signature and explanation, renamed to `MoeEXL3`. |
| 37 | Keep EXL3 split/GD/mod4 issue order, table slots and prefetch placement. |
| 38 | Keep conditional shared down ordering and EXL3 calibration explanation. |
| 39 | Keep the shared-arm join before the gather. |
| 40 | Take the per-expert tap explanation, renamed to `MoeEXL3`. |

The clean merge introduced `MoeFP8::route_report`; it was renamed to `MoeEXL3::route_report`.
The clean upstream pipeline was checked against EXL3's lane use: the shared arm has concurrent
buffers, joins before the gather, and returns lane 0 before the caller issues its deferred reduction.

## Numerical scope

No EXL3 arithmetic, encoding, rotation, band default, plane ordering or shard ownership changed.
Without a span the old serial issue path remains, apart from upstream fixes/fallbacks. Upstream's
span path changes operation grouping (piece-sized connection reads, recurrent chunks and collective
messages); it is intended to reproduce the narrower chunk's bytes, but GPU agreement is not claimed
here. The MTP rope-table liveness fix can change drafts/acceptance and tie-level accepted text where
1.6.0 previously let the rope refill overwrite carry storage. That is an intentional upstream fix.
No golden output or existing behavior assertion was weakened. The moved four-class mutation targets
the same ownership arithmetic through `block_slice`.

## Validation

Host: GCC 16.2.1, installed radiance 1.7.0, `RAD_WITH_HIP=OFF`. Bridge: GCC 14.2.0,
ROCm Clang 22.0.0, gfx1201, installed radiance 1.7.0. No GPU execution was available here.
Logs live in ignored build directories.

| Check | Command / output |
|---|---|
| Host build | Brief's `cmake -S . -B build-host -DCMAKE_PREFIX_PATH="$PREFIX" -DCMAKE_BUILD_TYPE=Release -DRAD_WITH_HIP=OFF`, then `cmake --build build-host -j 8`: exit 0, zero warnings. |
| Standalone plugin | `cmake -S architectures/qwen4exp_exl3 -B build-host/arch-oot -DCMAKE_PREFIX_PATH="$PREFIX" -DCMAKE_BUILD_TYPE=Release && cmake --build build-host/arch-oot -j 8`: exit 0, zero warnings. Loaded with lazy symbol resolution: ABI 17 and key `(qwen4exp, exl3)`. |
| Fresh archive | Extracted `git archive d694a7f` under `build-host/port-clean`, then ran the same standalone configure/build: exit 0, zero warnings. |
| Missing/wrong dependency | Configuring with radiance disabled fails at the REQUIRED package lookup; configuring against the 1.6.0 prefix fails immediately with `built for radiance ABI 17 ... is ABI 16`. |
| Pins | Both SHA-256 values match 1e968f6. `RADIANCE_GIT_URL` set to the supplied local history clone and `TMPDIR` under `build-host`: `ctest --test-dir build-host -R '^upstream_drift' --output-on-failure`: 2/2 pass, including the old-release negative control. |
| Full mutation run | `python3 tests/arch/mutation_check.py build-host/tests/arch`: exit 0, `RESULT: 72 of 72 caught`; `git status --short` immediately afterwards was empty. Seven controls exercise the new port contracts; the other 65 retain their original meanings. |
| Native architecture suite | `ctest --test-dir build-host -R '^arch_' --output-on-failure`: 10 passed, 3 skipped (duplicate-key loader needs ROCm on the host; both e2e modes lack configured fixtures). The default binary's full run reports `32 passed, 0 failed, 0 skipped`; ad05b18 makes CTest run all of it too. |
| HIP bridge, refreshed after ad05b18/81caa94 | `.hip-build-request` -> `.hip-build-result`: `exit 0`; log: `build exit 0, host tests exit 0`, zero `warning:` lines. CTest: 19 passed, 15 skipped, 0 failed out of 34 (the HIP build adds host-labelled kernel checks; the container lacks Python test dependencies). All 11 architecture host tests pass there, including duplicate-key loading and the full default case set. |

The mutation run exposed a pre-existing cleanup omission: it restored sources but never rebuilt
`arch_static_test_lane`. With a clean tree after the 72 controls, that binary still failed the
lane-open assertion from the last deliberate missing-join mutation. The cleanup now rebuilds LANE
too. Imported the runner, selected its two existing LANE entries and called `main` with the same
build directory: `RESULT: 2 of 2 caught`. The restored binary then passes
`decode_issues_gate_up_then_down_over_both_parity_tables`. No mutation source remains.

The full native host suite has an unrelated failure outside this worker's files:
`tests/speed/test_run_pass_refuses.py::test_radiance_src_at_another_commit` lacks the Docker skip
that adjacent cases have. `run_pass.sh` checks tools before the source pin, so without Docker its
stderr is `run_pass: docker is not on PATH`, while the test expects `not the pinned`.
The run was **26 passed, 4 skipped, 1 failed out of 31**. The four CTest skips were
`exl3_ref_vs_trellis_py`, `exl3_real_extract`, `exl3ref_ops_doc` and `arch_duplicate_key`.
Fixed at integration by the other worker's commit (the test now declares its Docker dependency like its
neighbours), so a host without Docker skips it.

Portability review found no new machine paths or device assumptions in changed source. Existing
hits are the staging test's synthetic PCI description and the end-to-end client's private loopback
server. The installed prefix was always a CMake argument. A second physical machine, constrained
CPU/RAM run and GPU numerical/concurrency/TP/ident checks remain with the coordinator.
