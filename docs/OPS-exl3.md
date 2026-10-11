# The EXL3 routed-expert ops

Two ops that radiance's vocabulary (`docs/OPS.md` in the engine) has no name for: the gate/up and the
down projection of a routed mixture of experts whose weights are EXL3 trellis codes (exllamav3's
format, imported as the canonical `exl3` encoding of `include/exl3_encoding.h`). They are what the
`("qwen4exp", "exl3")` architecture issues in place of the two `moe_gemm_q` of the in-tree block.

The schemas live in `include/exl3_abi.h` (`exl3_schema_at`), and every library implementing the
ops exports exactly those, so two libraries cannot mean different things by one name. **This
document is checked against them**: the table below is held to what `rad-schemas` dumps from the
built libraries by radiance's own `tools/ops_doc_check.sh`, operand for operand and parameter for
parameter (ctest `exl3ref_ops_doc`).

The host reference is `kernels/exl3ref` (`rad_plugin_reference() = 1`): every device row is checked
against it, it is never served without `--debug-accept-reference-kernels`, and it is written to be
read rather than to be fast.

## The ops

| op | params | operands |
|---|---|---|
| `moe_exl3_gate_up` | `M`(range) `N` `K` `n_expert` `top_k` `dtype` `codebook` `N_odd`/opt `K_odd`/opt | `a`, `codes`wt, `suh`wt, `suh1`wt, `svh`wt, `sorted_tok`, `expert_offset`, `codes_odd`wt?, `suh_odd`wt?, `suh1_odd`wt?, `svh_odd`wt? → `h` |
| `moe_exl3_down` | `M`(range) `N` `K` `n_expert` `top_k` `dtype` `codebook` `N_odd`/opt `K_odd`/opt | `h`, `codes`wt, `suh`wt, `svh`wt, `sorted_tok`, `expert_offset`, `codes_odd`wt?, `suh_odd`wt?, `svh_odd`wt? → `y` |

`wt` marks a weight TABLE (`RAD_OPD_WTAB`): the issue names a run of declared weights and the kernel
receives a pointer per expert, because an expert is a movement unit and may sit in VRAM, in pinned
host memory or on disk (engine `docs/OPS.md`, "Weight operands, and the weight TABLE"). **One table
per plane per parity class** (interface version 2, PLAN.md D15): the even experts' codes, suh,
[suh.1,] svh, then after the routing operands the odd experts' — `moe_gemm_q`'s pattern of codes and
scale tables, one more plane each.

### Parameters

| key | meaning |
|---|---|
| M | the TOKEN count of the pass, the op's ranged parameter. The row count `T` is read off the output; the in-tree block issues `T = M * top_k`. |
| N, K | class 0's weight `[rows, cols]` = `[out features, in features]` of this rank's slice. gate_up: `N = 2 * width` (gate rows, then up rows), `K = hidden`. down: `N = hidden`, `K = width`. |
| n_expert | the length of `expert_offset` minus one: the SORT's expert count, not the router's |
| top_k | slots per token |
| dtype | `exl3k2` .. `exl3k8`: the codes' K, bits per weight |
| codebook | `mul1`, `mcg` or `3inst`: the trellis codebook. The encoding's transform names it; the architecture passes it so the geometry says what is decoded. A device row that decodes one codebook constrains on it (`RAD_CIN("codebook", "mul1")`) and is refused by name at selection for the others. |
| N_odd, K_odd | class 1's `[rows, cols]`; optional, default `N`, `K` |

Extents the format cannot hold are refused (`RAD_E_SHAPE`): every `K` a multiple of 128 (the input
rotation), every gate_up `N` a multiple of 256 and every down `N` of 128 (the output rotation, per
half). They are the per-rank extents, so a tensor-parallel width split must cut on 128 — which the
384/256 parity split of a 640-wide expert does.

### Operands

| operand | dtype | extent | |
|---|---|---|---|
| gate_up `a` | f16 or bf16 | `[n_tok, >= max(K, K_odd)]` | one row per TOKEN; sorted row `i` reads row `sorted_tok[i] / top_k` |
| down `h` | f16 | `[T, >= max(K, K_odd)]` | one row per (token, slot) in the scatter's order; row `i` is row `i` |
| codes, suh, suh1, svh | table entries: one plane each (below) | `entries` ≥ the class's experts | the even experts, expert `e` at entry `e / 2`; every expert when the odd tables are absent. suh1: gate_up only |
| codes_odd, suh_odd, suh1_odd, svh_odd | as above | | optional, all together or none: the odd experts, expert `e` at entry `e / 2` |
| sorted_tok | i32 | `[>= T]` | `token * top_k + slot`, grouped by expert, `-1` past the live count |
| expert_offset | i32 | `[n_expert + 1]` | expert `e` owns rows `[off[e], off[e+1])`; `off[n_expert]` is the live row count |
| gate_up `h` | f16 | `[T, max(N, N_odd) / 2]` | in SORTED order |
| down `y` | bf16 | `[T, max(N, N_odd)]` | in SORTED order; a per-rank PARTIAL under a width split |

Each activation and output row may carry a pitch wider than its width; the last axis is
contiguous. `sorted_tok` and `expert_offset` are contiguous.

## What they compute

Per output row `i`:

- **The row is ZERO** — every column of it — when `i >= expert_offset[n_expert]` or
  `sorted_tok[i] < 0` (or `i < expert_offset[0]`, a row no expert owns). Both are dropped slots or rows another GEMM writes after this one (the
  protected experts of a `protect` sort). This is libr4d's rule for `moe_gemm_q`
  (`libr4d/r4d_moe.hip:1834-1841` at radiance 1.3.0); libref zeroes only on `sorted_tok < 0`. The
  output is never left holding stale arena bytes.
- Otherwise the row belongs to expert `e`, the one with `off[e] <= i < off[e+1]`, of class
  `c = e % 2` (0 when the odd tables are absent) at table entry `e / 2` (`e`). Its weight is
  `[R, C]` = `[N, K]` for class 0 and `[N_odd, K_odd]` for class 1, and its planes are entry `e / 2`
  of class `c`'s four (down: three) tables.

`What` is the expert's trellis-decoded weight in the rotated basis (`common/exl3_decode.h`
`exl3_decode_what_rows`: exact fp16 values, held as f32), `[R, C]`. `had` is the normalised
128-block Hadamard (`exl3_had128`): the natural-order Sylvester butterfly in f32 over each block of
128, then one f32 multiply by `(float)(1/sqrt(128))`. f16 and bf16 narrowing is round-to-nearest-even
(`rad_plugin.h`); arithmetic is f32, no fused multiply-add, sums in ascending index order.

### `moe_exl3_gate_up`

With `W = R / 2`, token `t = sorted_tok[i] / top_k`, and the stacked weight's halves — rows
`[0, W)` are the gate, read with `suh`; rows `[W, 2W)` the up projection, read with `suh.1`:

1. for each half `p` (gate with `suh`, up with `suh.1`), over `k < C`:
   `x_p[k] = f16(a[t][k] * s_p[k])`, then `x_p = f16(had(x_p))`;
2. `acc_p[n] = sum over k of x_p[k] * What[p*W + n][k]`, `n < W`;
3. `o_p = had(acc_p) * svh[p*W + n]` (f32, not rounded);
4. `h[i][n] = f16(silu(o_0[n]) * o_1[n])` for `n < W`, with `silu(g) = g * (1 / (1 + exp(-g)))`;
   and `h[i][n] = 0` for `W <= n < max(N, N_odd) / 2` — a narrower class fills the leading columns.

That is exllamav3's rounding order (Dylan's `exl3_moe_wmma.hip` `rotate128` and EPI 1): the scaled
input rounds to fp16 before the rotation, the rotated input rounds to fp16 before the GEMM, the
activation product rounds to fp16 once.

### `moe_exl3_down`

1. `x[k] = f16(h[i][k] * suh[k])`, `k < C`, then `x = f16(had(x))`;
2. `acc[n] = sum over k of x[k] * What[n][k]`, `n < R`;
3. `y[i][n] = bf16(had(acc)[n] * svh[n])` for `n < R`, and `0` for `R <= n < max(N, N_odd)`.

Under the intra-expert width split each rank holds a K-slice of every expert, so `y` is that rank's
partial: the output rotation and `svh` are linear, the block's all-reduce sums the partials
(PLAN.md D7).

### What they do not do

- **`expert_w` is not applied.** The gather (`moe_gather`, or `ar_gather_hc_write` at tp2) weights
  and sums a token's slots, in ascending slot order, exactly as it does for `moe_gemm_q`.
- **No `a_order` parameter.** The activation's row order is fixed per op and named by it — gate_up
  gathers by token, down reads the sorted rows — because gathering the down input a second time is
  the fluent-and-wrong failure engine `docs/OPS.md` "`a_order`" describes.
- **No down-input rotation fused into gate_up** (Dylan's EPI 1 does it): `h` is the plain fp16
  activation so the two ops stay separable and each is checkable on its own. A device library may
  fuse internally only if its `h` is the one written here.
- **Two classes only.** At four ranks radiance 1.3.0 splits a five-block expert over four classes
  (`moe_gemm_q_mod4`); these ops have no four-class form. A tp4 EXL3 deployment needs one.

## The two weight classes

The even experts at `[N, K]` and the odd ones at `[N_odd, K_odd]`, expert `e` at entry `e / 2` of
class `e % 2`, as `moe_gemm_q`'s two classes: a rank that holds three of an expert's five
128-column blocks for one parity and two for the other has two geometries, and every entry of one
table shares one (PLAN.md D7).

**The odd tables absent** (each a null `RadTensor.data`, never a shorter operand count; all four, or
three for down, together — a partly absent set is refused) means one class: expert `e` at entry `e`
of the even tables, all at `[N, K]`; `N_odd`/`K_odd`, if given, must equal `N`/`K`. An architecture
cannot issue it at radiance 1.3.0 — a declaration divides its weight list evenly over every table
position, optional or not (`core/build/rad_builder.cpp:697-731`), and an issue naming another run is
refused (`core/runtime/issue.cpp:249-254`) — so at tp1 the engine passes both classes with
`N_odd = N`, `K_odd = K`, as the in-tree block does. A shape description (`opd_shape`) follows the
parameters: the odd tables are described ABSENT exactly when neither `N_odd` nor `K_odd` is given.

**The declaration.** The weight list divides evenly over the table positions in schema order, the
remainder going to the first (`rad_builder.cpp:703-731`), so an architecture lists the even class's
codes run, its suh run, [its suh.1 run,] its svh run, then the odd class's in the same order — each
run the class's experts in entry order, each weight a one-plane view (`RadWeightDecl::planes` =
`"codes"`, `"suh"`, `"suh.1"`, `"svh"`) of the expert's one `exl3` encoding.

## The planes

A table's entry is one plane of one expert exactly as the container holds it (this rank's slice),
with the extents the core gives a plane of the canonical `exl3` encoding (`include/exl3_encoding.h`;
`rad_enc_plane_dims`):

| table | dtype | entry | content |
|---|---|---|---|
| codes | uK | `[R, C]` | element `(i, j)` is the K-bit trellis symbol of `What[i][j]`; rows LSB-first, each starting on a byte |
| suh | f16 | `[1, C]` | the input scales (the gate's, for gate_up) — TILED `[0x1]` |
| suh1 | f16 | `[1, C]` | gate_up only: the up half's input scales (plane `suh.1`) |
| svh | f16 | `[R, 1]` | the output scales (gate rows then up rows, for gate_up) — TILED `[1x0]` |

A reader takes the codes as `[R, C]` exactly (its rows are bit streams) and a scale vector by element
count, in whatever shape the declaration gave it. A table operand is the engine's pointer table
(`stride[0] = 0`) or the STACKED form — one tensor `[entries, rows, cols]` with a real leading
stride — which a tool uses and which the per-op oracle hands a host row. Nothing is reordered and
nothing is derived; a device library stores its own form of `codes` through its layout hook.

## What they refuse

| return | when |
|---|---|
| RAD_E_INVAL | a required operand absent; the odd tables partly absent |
| RAD_E_SCHEMA | a required parameter absent |
| RAD_E_DTYPE | `dtype` not `exl3k2`..`exl3k8`, an unknown `codebook`, or an operand of the wrong dtype (codes not the uK `dtype` names, a scale not f16) |
| RAD_E_SHAPE | extents the format cannot hold (above); an output not exactly the op's width; too few rows in an input; a table with too few entries, a codes entry not `[R, C]`, a scale entry of the wrong element count; `expert_offset` not non-decreasing from >= 0 to at most `T`; a sorted row naming a token past `a`'s rows |
| RAD_E_STRIDE | a last axis, an index operand or a table entry that is not contiguous |
| RAD_E_ALIGN | a scale entry not 2-byte aligned |

## Checking

- **The reference against an independent transcription.** `tests/ref/test_exl3ref.py` builds cases
  (random symbols — any sequence is a valid trellis — routed through a port of libref's
  `moe_scatter`), runs a library through a harness (`tests/ref/exl3ref_run.cpp`: the library is a
  path argument, `--domain host|device`, every buffer between poisoned margins, every table through
  the row's own layout hooks with an unrelayout round trip) and compares with
  `tests/ref/exl3_numpy.py`, whose trellis decode is checked bit for bit against the vllm-exl3
  golden. Cases cover both table forms (pointer table and stacked), the odd tables absent, two
  classes of different width, dropped slots and protected rows, every codebook, K 2..8, a bf16 `a`,
  the real model's tp1/tp2 geometry, band tops M = 16 and M = 4096, P1's fixture against its own
  dense weights, the refusals above, and the shape hook against what each case built. A device
  library is the same command with `--plugin <its .so> --domain device`, compared within `--tol`.
- **`rad-kbench` cannot check weight-TABLE ops (`tools/kfixture.cpp:312`)** at radiance 1.3.0: it
  fills a table operand with zeros (`tools/rad_kbench.cpp:2292-2400` handle only `RAD_OPD_WEIGHT`),
  so every case compares zeros with zeros; all 40 `moe_gemm_q` cases in the shipped
  `kernels_moe.rkb` record an all-zero reference output. A kbench row for these ops is reported as
  "kbench cannot check WTAB ops (kfixture.cpp:312)", never as passed (PLAN.md D15).
- **`RADIANCE_OP_ORACLE`** — the check on a real step. From the source at 1.3.0: a table operand is
  snapshotted entry by entry into one STACKED host buffer (`core/runtime/oracle.cpp:328-347,
  405-418`); a plane the device row stores as it is reaches the host row unchanged; a plane it
  re-lays goes back through the device row's `unrelayout` and, being a one-plane selection, is
  passed through element for element into the dtype and extents `exl3ref`'s `opd_shape` names
  (`oracle.cpp:456-548`, `core/format/encoding.cpp:306-321`) — uK `[entries, R, C]` for codes. So the
  oracle hands `exl3ref` the canonical planes for these ops. Not yet run live: that needs the
  architecture (P3) and a device library (P4).
