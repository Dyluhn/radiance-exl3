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
| `moe_exl3_gate_up` | `M`(range) `N` `K` `n_expert` `top_k` `dtype` `codebook` `N_odd`/opt `K_odd`/opt `h_rot` `a_rot` | `a`, `codes`wt, `suh`wt, `suh1`wt, `svh`wt, `dn_suh`wt?, `sorted_tok`, `expert_offset`, `codes_odd`wt?, `suh_odd`wt?, `suh1_odd`wt?, `svh_odd`wt?, `dn_suh_odd`wt? → `h` |
| `moe_exl3_down` | `M`(range) `N` `K` `n_expert` `top_k` `dtype` `codebook` `N_odd`/opt `K_odd`/opt `h_rot` | `h`, `codes`wt, `suh`wt, `svh`wt, `sorted_tok`, `expert_offset`, `codes_odd`wt?, `suh_odd`wt?, `svh_odd`wt? → `y` |
| `moe_exl3_prep_gu` | `M`(range) `K` `n_expert` `top_k` `dtype` `K_odd`/opt | `a`, `suh`wt, `suh1`wt, `sorted_tok`, `expert_offset`, `suh_odd`wt?, `suh1_odd`wt? → `xr` |
| `moe_exl3_gate_up_down` | `M`(range) `N` `K` `n_expert` `top_k` `dtype` `codebook` `N_odd`/opt `K_odd`/opt | `a`, `codes`wt, `suh`wt, `suh1`wt, `svh`wt, `dn_suh`wt, `dn_codes`wt, `dn_svh`wt, `sorted_tok`, `expert_offset`, `codes_odd`wt?, `suh_odd`wt?, `suh1_odd`wt?, `svh_odd`wt?, `dn_suh_odd`wt?, `dn_codes_odd`wt?, `dn_svh_odd`wt? → `y` |

`wt` marks a weight TABLE (`RAD_OPD_WTAB`): the issue names a run of declared weights and the kernel
receives a pointer per expert, because an expert is a movement unit and may sit in VRAM, in pinned
host memory or on disk (engine `docs/OPS.md`, "Weight operands, and the weight TABLE"). **One table
per plane per parity class** (interface version 2, PLAN.md D15): the even experts' codes, suh,
[suh.1,] svh, [gate_up: dn_suh,] then after the routing operands the odd experts' — `moe_gemm_q`'s
pattern of codes and scale tables, one more plane each. Interface version 3 adds the `h_rot` flag and
gate_up's `dn_suh` tables ("EPI 1", below); version 4 adds `moe_exl3_prep_gu`, gate_up's input
rotation as an op of its own, and gate_up's `a_rot` flag that reads what it wrote (below); version 5
adds `moe_exl3_gate_up_down`, the decode chain as one op (below).

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
| a_rot | gate_up only, 0 or 1, REQUIRED (for the reason `h_rot` is): 1 when gate_up's `a` is the `xr` `moe_exl3_prep_gu` wrote — sorted rows, already rotated — instead of the token rows. A row that does not implement it constrains `RAD_CEQ("a_rot", 0)`. Any other value is `RAD_E_INVAL`. |
| h_rot | gate_up and down (gate_up_down has no flags: it is defined at `a_rot = 0`, `h_rot = 1`), 0 or 1, REQUIRED: 1 when `h` crosses between the two ops already rotated ("EPI 1", below). Required rather than optional because a constraint on an absent key never holds (`core/util/rad_util.cpp:264-265`): a row that does not implement `h_rot = 1` constrains `RAD_CEQ("h_rot", 0)` and is refused by name at selection. Any other value is `RAD_E_INVAL`. |

Extents the format cannot hold are refused (`RAD_E_SHAPE`): every `K` a multiple of 128 (the input
rotation), every gate_up `N` a multiple of 256 and every down `N` of 128 (the output rotation, per
half). They are the per-rank extents, so a tensor-parallel width split must cut on 128 — which the
384/256 parity split of a 640-wide expert does.

### Operands

| operand | dtype | extent | |
|---|---|---|---|
| gate_up `a`, `a_rot = 0`; prep_gu `a` | f16 or bf16 | `[n_tok, >= max(K, K_odd)]` | one row per TOKEN; sorted row `i` reads row `sorted_tok[i] / top_k` |
| gate_up `a`, `a_rot = 1` | f16 | `[T, >= 2 * max(K, K_odd)]` | prep_gu's `xr`: one row per SORTED row, row `i` is row `i` |
| down `h` | f16 | `[T, >= max(K, K_odd)]` | one row per (token, slot) in the scatter's order; row `i` is row `i` |
| codes, suh, suh1, svh | table entries: one plane each (below) | `entries` ≥ the class's experts | the even experts, expert `e` at entry `e / 2`; every expert when the odd tables are absent. suh1: gate_up only |
| codes_odd, suh_odd, suh1_odd, svh_odd | as above | | optional, all together or none: the odd experts, expert `e` at entry `e / 2` |
| dn_suh, dn_suh_odd | f16 table entries | entry `[1, R / 2]` | gate_up only, optional: DOWN's suh of the same experts, per class. Read only with `h_rot = 1`, where both classes' are required |
| sorted_tok | i32 | `[>= T]` | `token * top_k + slot`, grouped by expert, `-1` past the live count |
| expert_offset | i32 | `[n_expert + 1]` | expert `e` owns rows `[off[e], off[e+1])`; `off[n_expert]` is the live row count |
| gate_up `h` | f16 | `[T, max(N, N_odd) / 2]` | in SORTED order |
| prep_gu `xr` | f16 | `[T, 2 * max(K, K_odd)]` | in SORTED order; rows outside the runs not written (below) |
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

With `h_rot = 1`, one more step per live row, after step 4 and over the row's `W` columns:

5. `h[i] = f16(had(f16(h[i] * dn_suh_e)))`, `dn_suh_e` entry `e / 2` of the class's `dn_suh` table —
   exactly down's step 1 below, with down's suh of the same expert; the tail past `W` stays zero.

With `a_rot = 1`, step 1 is not computed: `x_0 = xr[i][0 : C]` and `x_1 = xr[i][C : 2C]` as given
(`moe_exl3_prep_gu` computed them, below), and `suh` / `suh.1` are not read.

### `moe_exl3_prep_gu`

gate_up's step 1, as an op, for every row of every run:

1. for row `i` in `[off[e], off[e+1])` of expert `e` (class `c`, input width `C`), token `t`:
   `xr[i][0 : C] = f16(had(f16(a[t] * suh_e)))` and `xr[i][C : 2C] = f16(had(f16(a[t] * suh.1_e)))`
   — exactly gate_up's step 1 — or zeros across `2C` if `sorted_tok[i] < 0`.
2. **Not written**, and left as they were: every row outside the runs `[off[0], off[n_expert])`
   (dropped slots, protected experts' rows) and the columns past `2C` of a narrower class's row.
   gate_up reads none of them — its output rows there are zero whatever `xr` holds. "Not written" is
   exact: the per-op oracle runs the host row on the pre-launch copy of the output
   (`core/runtime/oracle.cpp:405-423`), so a library that writes them is the one disagreeing.

`xr` is the layout exl3r4d's prep kernels write in both bands today (`xr[row][m * C + k]`, row pitch
`2 * max(K, K_odd)`), agreed with P4 (decode) and P5 (prefill).

### `moe_exl3_down`

1. `x[k] = f16(h[i][k] * suh[k])`, `k < C`, then `x = f16(had(x))` — skipped with `h_rot = 1`,
   where `x = h[i]` as given (gate_up already did it) and `suh` is not read;
2. `acc[n] = sum over k of x[k] * What[n][k]`, `n < R`;
3. `y[i][n] = bf16(had(acc)[n] * svh[n])` for `n < R`, and `0` for `R <= n < max(N, N_odd)`.

Under the intra-expert width split each rank holds a K-slice of every expert, so `y` is that rank's
partial: the output rotation and `svh` are linear, the block's all-reduce sums the partials
(PLAN.md D7).

### EPI 1: the flag, and why the pair's bits do not depend on it

`h_rot = 1` moves down's input rotation into gate_up's epilogue (Dylan's original design; requested
by P4, whose gate_up workgroup owns whole 128-column blocks of `h` for its rows, so the rotation fits
there and down needs no rotation dispatch at any M). Step 5 is down's step 1, the same arithmetic on
the same bytes, so **the pair gate_up → down issued with both flags 1 gives the same bits as the
pair issued with both flags 0**, and `h_rot = 0` is version 2's behaviour exactly. The two flags go
together: down with `h_rot = 1` reading an unrotated `h`, or the reverse, computes something else
and nothing in either op can see it — the architecture issues the pair consistently.

### prep_gu and a_rot: one choice per pass

**prep_gu → gate_up(`a_rot = 1`) gives the bits gate_up(`a_rot = 0`) gives**, at either `h_rot`: the
same prologue on the same bytes, computed by another op. The point is the issue order: an
architecture can issue independent work (the shared expert's stages) between the rotation and the
routed GEMM. `a_rot` is fixed per declared op, like every parameter but `M`, so an architecture that
wants the split at decode and gate_up's own rotation above it declares gate_up TWICE — `a_rot = 1`
beside `moe_exl3_prep_gu`, both ranged over the decode band, and `a_rot = 0` — and chooses at issue
by the pass's row count, the way the in-tree block chooses its router form
(`arch/common/rad_block_moe_fp8.h:1438-1447, 1527-1551`). Both gate_up declarations read the same
weights, so their rows must ask the same stored form of `codes`.

### `moe_exl3_gate_up_down` (version 5)

**Defined as the pair: `y = down(h_rot = 1)(gate_up(a_rot = 0, h_rot = 1)(a))`**, with gate_up's
geometry (`N`, `K`, `N_odd`, `K_odd` are gate_up's; down's are `[K, N / 2]` a class), one `dtype` and
one `codebook` for both halves, and no flags. Its tables are gate_up's five a class (`dn_suh` is
down's suh, required here) then down's `codes` and `svh` (`dn_codes` entry `[K, N / 2]` in down's
encoding, `dn_svh` `[K, 1]`); `a` is gate_up's (token rows, f16 or bf16), `y` is down's. `h` is not
an operand: an implementation keeps it in its scratch. **Its bits are the pair's bits** — exactly
what gate_up then down write on the same operands — so an architecture may issue it in place of the
pair wherever both halves carry one encoding, and keep the pair elsewhere (a calibration tap that
reads `h`, a layer whose halves differ in K). `include/exl3_abi.h` reads it as its two halves
(`exl3_gd_params`, `exl3_gd_operands`) and every library implements it through them, so it refuses
what either half refuses. The point is the dispatch: the gfx1201 row runs it as ONE persistent launch
(the input rotation in its prologue, then gate_up's and down's units ordered by counters, down of an
expert after its gate_up units), where the pair is two or three dispatches with a boundary between.

### What they do not do

- **`expert_w` is not applied.** The gather (`moe_gather`, or `ar_gather_hc_write` at tp2) weights
  and sums a token's slots, in ascending slot order, exactly as it does for `moe_gemm_q`.
- **No `a_order` parameter.** The activation's row order is fixed per op and named by it — gate_up
  gathers by token, down reads the sorted rows — because gathering the down input a second time is
  the fluent-and-wrong failure engine `docs/OPS.md` "`a_order`" describes.
- **No down-input rotation fused into gate_up unless asked** (`h_rot = 1`, above): with `h_rot = 0`
  `h` is the plain fp16 activation, and every library writes exactly the `h` this document defines
  for the flag it was given.
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
`"codes"`, `"suh"`, `"suh.1"`, `"svh"`) of the expert's one `exl3` encoding. gate_up's list carries
the `dn_suh` run after each class's `svh` run. **Every table position takes its share whatever the
flag**, so an architecture binds `dn_suh` (and `dn_suh_odd`) at `h_rot = 0` too — the natural run is
the one down's own `suh` table names: down's suh weights, read by two ops.

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
| dn_suh | f16 | `[1, R / 2]` | gate_up only: DOWN's input scales of the same expert (down's `suh`, down's `[hidden, R / 2]` weight) |

A reader takes the codes as `[R, C]` exactly (its rows are bit streams) and a scale vector by element
count, in whatever shape the declaration gave it. A table operand is the engine's pointer table
(`stride[0] = 0`) or the STACKED form — one tensor `[entries, rows, cols]` with a real leading
stride — which a tool uses and which the per-op oracle hands a host row. Nothing is reordered and
nothing is derived; a device library stores its own form of `codes` through its layout hook.

## What they refuse

| return | when |
|---|---|
| RAD_E_INVAL | a required operand absent; the odd tables partly absent; `h_rot` or `a_rot` neither 0 nor 1; gate_up `h_rot = 1` without a class's `dn_suh` table (gate_up_down: always) |
| RAD_E_SCHEMA | a required parameter absent (`h_rot`, gate_up's `a_rot` included) |
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
  dense weights, the refusals above, and the shape hook against what each case built. For `h_rot`,
  CHAINS: gate_up then down at both flags on the fixture, a two-class case with dropped and protected
  rows, the real tp2 geometry, K = 3 mcg and both band tops — gate_up's `h_rot = 1` output equal to
  down's step 1 of its `h_rot = 0` output, and the pair's output equal across the flags, to the bit,
  with controls (the flag moves gate_up's output; another expert's `dn_suh` is caught; down told
  `h_rot = 1` about an unrotated `h` is caught; `dn_suh` filled with NaN at `h_rot = 0` changes nothing).
  For `a_rot`, PREP CHAINS on the same cases: `moe_exl3_prep_gu` against the transcription with every
  row outside the runs and every column past `2C` left as the harness prefilled it, then gate_up
  `a_rot = 1` reading that `xr` equal to gate_up `a_rot = 0`, to the bit, at both `h_rot` (control: an
  `xr` built with the halves' scales swapped is caught); prep_gu's own controls (swapped scales,
  swapped classes, one flipped scale sign moving only its expert's rows, the neutral variants). For
  `moe_exl3_gate_up_down`, on the fixture, the two-class case, the real tp2 and tp1 geometry, K = 3 mcg
  and both band tops: against the transcription, and against gate_up(`a_rot = 0`, `h_rot = 1`) then
  down(`h_rot = 1`) run through the SAME library, to the bit, with controls (another expert's down
  weight is caught; one flipped down symbol moves only its expert's rows; the stacked form changes
  nothing) and its refusals. A device library is the same command with `--plugin <its .so> --domain
  device`, compared within `--tol` against the transcription and to the bit against its own pair.
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
