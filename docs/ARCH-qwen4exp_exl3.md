# qwen4exp_exl3 -- build switches and their limits

The `("qwen4exp", "exl3")` architecture plugin (`architectures/qwen4exp_exl3/`). Its MoE block is
`rad_block_moe_exl3.h`; the ops it issues are `include/exl3_abi.h`'s (interface v4, `docs/OPS-exl3.md`).

## Compile-time switches

| define | default | what it changes |
|---|---|---|
| `QWEN4EXP_EXL3_H_ROT` | 1 | EPI 1: gate_up writes down's input already rotated (`h_rot = 1` on both ops). 0 is the header's version-2 math on the same operands, for an A/B build only. Same bits either way. |
| `QWEN4EXP_EXL3_PREP_SPLIT` | 1 | The decode-band issue order. At 1, a pass of at most 16 rows issues `moe_exl3_prep_gu` (gate_up's input rotation as an op), then the shared expert's gate, gate_up and quantiser, then `gate_up(a_rot = 1)` over its output; a larger pass issues the one `gate_up(a_rot = 0)` as at 0. Same bits either way. 1 since 2026-10-12 on engine step time (real model, MTP off: -0.215 to -0.31 ms a step at 1-8 concurrent, every interval excluding 0); 0 stays selectable and has none of the limits below. |
| `QWEN4EXP_EXL3_TEST_MUTATION` | 0 | Test-only wiring defects (tests/arch builds 1..3, never installed). |

Any build of `tests/arch` also makes `order0_home/` and `order1_home/`: the plugin under its real file
name at `PREP_SPLIT` 0 and 1, for an engine A/B (put first in `RADIANCE_HOME`).

## What the split build (`PREP_SPLIT = 1`) requires of the engine

radiance 1.3.0 keys a routed layer's expert movement on the layer's **first declared op that reads
its experts** (`core/place/stager.cpp`, `first_use_op`):

- the **prefill stager** copies the next layer's pooled experts, and orders the layer's ops behind its
  own copies, at that op, in passes above the arena's smaller level;
- the **file stager** (`--weights-disk-tier`) reads a layer's file-tier experts at that op, on every
  pass, decode included.

The split build declares two forms of the routed gate_up, one per band. The one `gate_up(a_rot = 0)`
is declared first, so every prefill pass issues the op the prefill stager keys on (the form declared
first before 92d140d staged one layer a pass and raced its copies). No declaration order can make
both forms "first", so in the decode band the file stager would never read those layers' experts:

**The split build must not be served with `--weights-disk-tier`.** It refuses that combination by
name at declare where it can see it -- the process's command line (`/proc/self/cmdline`), since the
plugin ABI hands an architecture no placement. Where the engine is embedded without a radiance
command line, or there is no `/proc`, the check cannot see the flag: do not combine them.

The `PREP_SPLIT = 0` build (`-DQWEN4EXP_EXL3_PREP_SPLIT=0`) has one gate_up form and none of these limits: build
it where `--weights-disk-tier` is needed.

## Tests

`tests/arch`: `arch_static` / `arch_static_split` (the declared graph and issues at both orders,
incl. "the one gate_up is every layer's first declared expert reader" and the disk-tier refusal),
`mutation_check.py`, and `e2e_differential.sh` (host and gpu; gpu rule 7 serves a fixture with half
its expert plane pooled and checks ours stages as many routed layers as the in-tree reference).
