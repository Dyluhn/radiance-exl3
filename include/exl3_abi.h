/* exl3_abi.h -- the interface between the EXL3 plugins: the two routed-expert GEMM ops, the routing
 * contract they read, the weight planes they read, and one parse of their geometry.
 * Header-only C; owner R (PLAN.md §2). The encoding itself is include/exl3_encoding.h (owner Q).
 *
 * Every library that implements `moe_exl3_gate_up` / `moe_exl3_down` exports exl3_schema_at() as its
 * schemas -- the reference (kernels/exl3ref) and the device library (kernels/exl3r4d) -- so the two
 * cannot declare different schemas (the loader refuses the second plugin if they do: docs/OPS.md
 * :139-150 at radiance 1.3.0) and `rad-schemas exl3ref.so exl3r4d.so` prints no difference.
 * docs/OPS-exl3.md is the prose form of the schemas below and is checked against them.
 *
 * Both sides check EXL3_IFACE_VERSION. Bump it on ANY change below: a schema, an operand position,
 * the routing contract, the planes' forms or the geometry rule.
 *
 * VERSION 2 (2026-10-11, PLAN.md D15): ONE WEIGHT-TABLE OPERAND PER PLANE PER PARITY CLASS. Version 1
 * carried all of an expert's planes in one table operand (D6), and at radiance 1.3.0 that weight
 * could be served by no host row without a layout hook (core/build/rad_builder.cpp:1385, 1453-1460)
 * and handed to the per-op oracle only decoded to dense W (core/runtime/oracle.cpp:425-560), which
 * a reference computing in exllamav3's rounding order cannot read. One plane an operand is stored as
 * it is wherever no kernel re-lays it, and the oracle passes a single-plane selection through
 * unchanged (core/format/encoding.cpp:306-321). See "THE PLANES" below.
 *
 * VERSION 3 (2026-10-11, requested by P4, approved by the orchestrator): "EPI 1", Dylan's original
 * design. Both ops take a REQUIRED flag `h_rot`. With h_rot = 1, gate_up also reads down's input
 * scales (`dn_suh`, `dn_suh_odd`) and writes h ALREADY ROTATED for down -- exactly down's step 1,
 * so the pair is bit-identical to the h_rot = 0 pair -- and down reads it as rotated and skips that
 * step. Down's operands are unchanged; gate_up's operands after `svh` move by one (and by two after
 * `svh_odd`). With h_rot = 0 both ops compute what version 2 computed, to the bit.
 *
 * VERSION 4 (2026-10-11, requested by P4, approved by the orchestrator): gate_up's input rotation as an
 * op of its own, so an architecture can issue independent work between the rotation and the routed
 * GEMM. A third op, `moe_exl3_prep_gu`, writes xr -- every sorted row's input rotated for gate and for
 * up, exactly the prologue gate_up runs -- and gate_up takes a REQUIRED flag `a_rot`: with a_rot = 1
 * its `a` IS that xr and it does not rotate (nor read suh / suh.1). prep_gu -> gate_up(a_rot = 1) gives
 * the bits gate_up(a_rot = 0) gives. With a_rot = 0 gate_up is version 3's, to the bit. Operand lists
 * of gate_up and down are unchanged; prep_gu is schema 2, after them.
 *
 * VERSION 5 (2026-10-12, P4b, ownership extended by the orchestrator): a FOURTH op,
 * `moe_exl3_gate_up_down`, the whole routed-expert chain of a decode pass as one op -- prep_gu's rotation,
 * gate_up with h_rot = 1, down with h_rot = 1 -- so a library can run it as ONE launch (exl3r4d's fused
 * kernel: no dispatch boundaries, down's work fills gate_up's tail). Its output is exactly down's y from
 * gate_up(a_rot = 0, h_rot = 1) -> down(h_rot = 1) on the same operands, to the bit; h and xr are the
 * op's own scratch. Operands: gate_up's planes, then down's codes and svh (down's suh is gate_up's
 * dn_suh, the same weight), per class (THE OPS below). The three earlier ops are unchanged; the planes
 * enumeration gains EXL3_P_DN_CODES and EXL3_P_DN_SVH (-1 for the earlier ops). Schema 3, after them.
 *
 * VERSION 6 (2026-10-14, P8): FOUR-CLASS FORMS of gate_up, down and prep_gu -- `moe_exl3_gate_up_mod4`,
 * `moe_exl3_down_mod4`, `moe_exl3_prep_gu_mod4` -- the experts as four classes by number mod 4, radiance's
 * `moe_gemm_q_mod4` beside `moe_gemm_q` (its docs/OPS.md, "moe_gemm_q_mod4"): four ranks splitting a five-block
 * expert give the extra block four places to go instead of two. Separate ops, because the declaration divides a
 * table op's weights over its table operands by count alone, so the operand list IS the class count. Each is its
 * two-class op with three more class table sets and N_q / K_q for N_odd / K_odd (THE OPS below). The four earlier
 * ops are unchanged, to the schema; the new ones are schemas 4, 5, 6, after them. No four-class gate_up_down: its
 * fused launch's two views do not fit the engine's kernel-argument slot at four classes (docs/OPS-exl3.md).
 */
#ifndef EXL3_ABI_H
#define EXL3_ABI_H

#include "rad_abi.h"
#include "rad_plugin.h"
#include "exl3_encoding.h"

#ifdef __cplusplus
extern "C" {
#endif

#define EXL3_IFACE_VERSION 6

/* THE FEWEST TOKENS a pass issues moe_exl3_gate_up_down for, and exl3r4d fuses it from -- one build-time value for
 * the architecture's issue band (QWEN4EXP_EXL3_GD) and the library's fused launch (EXL3R4D_GD_MIN_TOK's default), so
 * both move together. Not an interface fact: below it the same bits come from the other forms. 4 since 2026-10-14
 * (it was 6): the 4-row stall that kept it at 6 was the fused launch's co-residency bug, fixed in f30ee0c; engine,
 * radiance 1.5.0, tp2 production MTP flags, 3 interleaved rounds: C=1 MTP step 16.34 / 16.34 / 16.34 ms from 4 against
 * 16.59 / 16.61 / 16.62 from 6 (-0.26 ms), C=2 identical, ident unchanged; perfmoe tp2 M=4 MTP-verify routing 69.8 us
 * a layer fused vs 77.4 split (libr4d 72.2). From 3 the fused form also wins at M=3 in perfmoe (75.4 vs 78.7) and is
 * not measured in the engine. */
#ifndef EXL3_GD_MIN_ROWS
#define EXL3_GD_MIN_ROWS 4
#endif

/* ===================================================================== THE ROUTING CONTRACT (S0.8)
 *
 * What the in-tree router hands a routed-expert GEMM, read at radiance 1.3.0 (89cee7c). Both ops
 * below consume it exactly as `moe_gemm_q` does, so the in-tree router, scatter and gather serve
 * them unchanged. Every claim cites the line that makes it.
 *
 * WHO WRITES IT. `router_topk` (softmax over ALL n_expert logits, top_k, optional renormalise)
 * then `moe_scatter`, or the two fused as `router_topk_scatter`, or with the router GEMM in front
 * as `router_gemm_topk_scatter` (docs/OPS.md:567-570, 601-641; libref/ref_registry.cpp:876-886).
 * The block reports the five buffers to the heat engine with rad_route_report (abi/rad_runtime.h
 * :352-369; the struct is RadRouting, :356-364); nothing is read back on the compute stream.
 *
 *   expert_ids    i32 [n_tok, top_k]   device. The router's choice for each (token, slot);
 *                                      RadRouting::expert_ids (rad_runtime.h:357).
 *   expert_w      f32 [n_tok, top_k]   device. The routing weight of each (token, slot)
 *                                      (rad_runtime.h:358). NOT READ BY EITHER OP HERE: the
 *                                      gather (`moe_gather`, or `ar_gather_hc_write` at tp2) multiplies
 *                                      it in as it sums a token's slots in ascending slot order
 *                                      (docs/OPS.md:572; libref/ref_registry.cpp:907-909).
 *   sorted_tok    i32 [n_tok * top_k]  device. The FLATTENED (token, slot) index token*top_k + slot,
 *                                      grouped by expert by a STABLE counting sort -- not the bare
 *                                      token id (libref/ref_moe.cpp:3-10; ref_registry.cpp:880-881).
 *                                      Row i of every expert GEMM output belongs to pair sorted_tok[i].
 *   expert_offset i32 [n_expert + 1]   device. Expert e's rows are [off[e], off[e+1]);
 *                                      off[n_expert] is the LIVE row count (rad_runtime.h:361;
 *                                      libref/ref_moe.cpp:135-140).
 *   expert_count  i32 [n_expert]       device. off[e+1] - off[e]; the heat engine's histogram,
 *                                      written in place via rad_route_counts (rad_runtime.h:359,
 *                                      371-377). Not read by either op here.
 *
 * DROPPED SLOTS. A slot whose expert is outside the range the sort owns -- [expert_base,
 * expert_base + n_expert), which is how expert parallelism is spelt, or a negative id for a padded
 * batch row -- is not counted, and sorted_tok[off[n_expert] .. n_tok*top_k) is -1
 * (libref/ref_moe.cpp:11-13, 122-126, 144; ref_registry.cpp:364-371, 880-886).
 *
 * ROWS PAST off[n_expert] ARE ZEROED BY THE GEMM, WHATEVER sorted_tok HOLDS THERE. Two reasons
 * a row can be past the live count: a dropped slot (-1), or a PROTECTED expert's row. A layer
 * that keeps a few experts at bf16 sorts them last (`protect`, docs/OPS.md:646-651) and hands the
 * quantised GEMMs n_expert = the regular count, so off[n_expert] stops short of the protected
 * rows, which hold valid flat indices; the protected pair writes those rows AFTER the quantised
 * down GEMM zeroed them (arch/common/rad_block_moe_fp8.h:1611-1622). libr4d zeroes every row past
 * the live count (libr4d/r4d_moe.hip:83-86, 1834-1841, 2221-2222); libref zeroes a row only when
 * sorted_tok is negative (libref/ref_moe.cpp:285-288), which agrees on dropped slots and not on
 * protected rows. These ops follow libr4d: row i is zero when i >= off[n_expert] OR
 * sorted_tok[i] < 0 (or i < off[0], a row no expert owns), across the whole output width.
 *
 * WHICH EXPERT OWNS ROW i: the e with off[e] <= i < off[e+1] (experts with no rows have
 * off[e] == off[e+1] and own nothing). Under `protect` the offsets are in SORT order, not id
 * order -- the regular experts closed up, then the protected ones (docs/OPS.md:646-651) -- and the
 * weight tables are declared in the same order (rad_block_moe_fp8.h:831-834), so "expert e" below
 * always means offsets index e.
 *
 * TWO EXPERT COUNTS. The router's n_expert is the model's; the sort's (`n_local`, or the
 * scatter's n_expert) is this rank's slice and sizes expert_offset (docs/OPS.md:609-615). The
 * `n_expert` parameter of both ops below is the SORT's: the length of expert_offset minus one.
 * Under the intra-expert width split every rank holds a slice of EVERY expert, so the two are
 * equal there; under expert parallelism they are not.
 *
 * THE ROW ORDER OF THE ACTIVATION. The in-tree grouped GEMM says it with `a_order`
 * (docs/OPS.md:700-721). Here it is FIXED per op and not a parameter: gate_up reads one row per
 * TOKEN, row sorted_tok[i] / top_k (a_order "token"); down reads one row per (token, slot) in the
 * scatter's order, row i itself (a_order "sorted"). Gathering the down input a second time hands
 * every slot of a token the first slot's intermediate and stays fluent -- the failure the
 * parameter exists to prevent (docs/OPS.md:713-721) -- so the order is part of each op's name.
 *
 * `M` IS THE TOKEN COUNT OF A PASS, not the sorted row count (rad_block_moe_fp8.h:1052-1054): the
 * kernels read the row count T off the output and M is what the scratch hook and the operand
 * descriptions size from. T = M * top_k at every issue the in-tree block makes.
 */

/* ===================================================================== THE TWO OPS
 *
 * Exact semantics, rounding points and refusals: docs/OPS-exl3.md (the checked prose form). In
 * one line each, per live row i of expert e, token t = sorted_tok[i] / top_k:
 *
 *   moe_exl3_gate_up   h[i] = f16( silu(had(f16(had(f16(a[t] * suh)))  @ What_gate) * svh_gate)
 *                                        * (had(f16(had(f16(a[t] * suh.1))) @ What_up) * svh_up) )
 *   moe_exl3_down      y[i] = bf16( had(f16(had(f16(h[i] * suh))) @ What) * svh )
 *
 * moe_exl3_prep_gu   xr[i] = [ f16(had(f16(a[t] * suh_e))) | f16(had(f16(a[t] * suh.1_e))) ]
 *                    -- gate_up's step 1 for its two halves, side by side (below, "XR"); gate_up with
 *                    a_rot = 1 reads xr[i] where it would have computed it.
 *
 * moe_exl3_gate_up_down  y[i] = moe_exl3_down(h_rot = 1)( moe_exl3_gate_up(a_rot = 0, h_rot = 1)(a) )[i]
 *                    -- the decode chain as ONE op (version 5): the same rotations, roundings and zero rules, h and
 *                    xr in the op's scratch, nothing but y written.
 *
 * h_rot = 1 ("EPI 1"): gate_up writes h'[i] = f16(had(f16(h[i] * dn_suh_e))) per 128-block --
 * down's own first step, with down's suh of the same expert -- and zeros past the class's width as
 * before; down then reads its input as that rotated activation, y[i] = bf16(had(h'[i] @ What) * svh),
 * and does not read its suh. Issued as a PAIR (both flags 1, or both 0) the pair's output is the same
 * bits either way; mixing the flags is a caller error nothing here can see.
 *
 * had = the normalised 128-block Hadamard (common/exl3_decode.h exl3_had128). `expert_w` is NOT
 * applied by either: the gather does it, exactly as for `moe_gemm_q`.
 *
 * THE EXPERTS ARE TWO CLASSES BY PARITY, as `moe_gemm_q`'s are (docs/OPS.md:677-688): the even
 * experts at [N, K], the odd ones at [N_odd, K_odd] (each defaulting to N, K), expert e at entry
 * e / 2 of class e % 2 -- because a rank holding 3 of an expert's five 128-column blocks for one
 * parity and 2 for the other (PLAN.md D7) has two geometries. Each class is one table operand per
 * plane. The output is the WIDER class's width and a narrower class writes zeros past its own.
 *
 * THE ODD CLASS ABSENT (every one of its tables a null RadTensor.data) means ONE class: expert e at
 * entry e of the even tables, every expert at [N, K]; N_odd / K_odd, if given, must equal N / K. A
 * partly absent odd set is refused. NOTE: at 1.3.0 an architecture cannot issue an absent table:
 * the declaration divides its weight list EVENLY over every table position, optional or not
 * (core/build/rad_builder.cpp:697-731), and an issue naming another run is refused
 * (core/runtime/issue.cpp:249-254). So the engine always passes both classes (at tp1 with
 * N_odd == N, K_odd == K, as the in-tree block does); absence is for a caller holding operands
 * directly -- a harness, a tool -- and is what a shape description without N_odd and K_odd means.
 *
 * THE _mod4 OPS (version 6) TAKE FOUR CLASSES BY NUMBER MOD 4, as `moe_gemm_q_mod4`'s are: class q at [N_q, K_q]
 * (N_0, K_0 being N, K; each N_q, K_q defaulting to N, K), expert e at entry e / 4 of class e % 4, so class q holds
 * ceil((n_expert - q) / 4) entries. Every class's tables are REQUIRED (no absent form: the architecture issues all
 * four, and a caller with one geometry has the two-class ops). Everything else -- the routing, the zero rules, the
 * output the widest class's width with a narrower class zero past its own, h_rot, a_rot, xr -- is the two-class
 * op's, word for word, with "the odd class" read as "classes 1, 2 and 3".
 *
 * PARAMETERS (both ops):
 *   M         the token count of the pass (the op's ranged parameter)
 *   N, K      class 0's weight [rows, cols] = [out features, in features] of this rank's slice.
 *             gate_up: N = 2 * width (gate rows then up rows, stacked), K = hidden;
 *             down:    N = hidden, K = width
 *   n_expert  the length of expert_offset minus one (the sort's expert count)
 *   top_k     slots per token
 *   dtype     "exl3k2" .. "exl3k8": the codes' K, bits per weight (exl3_gemm_dtype)
 *   codebook  "mul1" | "mcg" | "3inst": the trellis codebook (the encoding's transform names it;
 *             the architecture passes it so the geometry says what is decoded -- a host row has no
 *             layout hook through which to see the encoding)
 *   N_odd, K_odd   class 1's [rows, cols]; optional, default N, K
 *   h_rot     0 or 1, REQUIRED: whether h crosses between the two ops rotated (above). Required and
 *             not optional because a constraint on an absent key never holds (core/util/rad_util.cpp
 *             :264-265): a row that does not implement h_rot = 1 says RAD_CEQ("h_rot", 0) and is
 *             refused BY NAME at selection, which it could not if callers might omit the key.
 *             Any other value is RAD_E_INVAL.
 *   a_rot     gate_up only, 0 or 1, REQUIRED (for the reason h_rot is): 1 when gate_up's `a` is the
 *             xr moe_exl3_prep_gu wrote -- sorted rows, already rotated -- rather than the token rows.
 *             A row that does not implement it says RAD_CEQ("a_rot", 0). Any other value RAD_E_INVAL.
 *
 * moe_exl3_gate_up_down takes gate_up's parameters without the flags (h_rot = 1 and a_rot = 0 are what it is):
 * M, N, K, n_expert, top_k, dtype, codebook, N_odd?, K_odd?. Down's geometry follows: [K, N / 2] (and
 * [K_odd, N_odd / 2]); both matrices are the same K and codebook (an architecture whose two projections differ
 * issues the separate ops).
 *
 * moe_exl3_prep_gu takes M, K, n_expert, top_k, dtype (the routed GEMMs' "exl3kK": the family it
 * belongs to, so a library serving one K constrains all three ops alike) and K_odd?: the classes'
 * input widths are all it needs of the weight.
 *
 * The _mod4 ops take the same parameters with N_1?, K_1?, N_2?, K_2?, N_3?, K_3? in place of N_odd?, K_odd?
 * (prep_gu_mod4: K_1?, K_2?, K_3?), `moe_gemm_q_mod4`'s names.
 *
 * OPERANDS, in schema order (moe_gemm_q's pattern: the even tables, the routing, the odd tables,
 * the output):
 *   gate_up  a, codes, suh, suh1, svh, dn_suh?, sorted_tok, expert_offset,
 *            codes_odd?, suh_odd?, suh1_odd?, svh_odd?, dn_suh_odd? -> h
 *   down     h, codes, suh, svh, sorted_tok, expert_offset, codes_odd?, suh_odd?, svh_odd? -> y
 *   prep_gu  a, suh, suh1, sorted_tok, expert_offset, suh_odd?, suh1_odd? -> xr
 *   the _mod4 ops: class 0's tables, the routing, then class 1's, 2's and 3's tables (suffix _1, _2, _3; every one
 *            required but dn_suh_q, as dn_suh), then the output -- `moe_gemm_q_mod4`'s pattern
 *   gate_up_down  a, codes, suh, suh1, svh, dn_suh, dn_codes, dn_svh, sorted_tok, expert_offset,
 *            codes_odd?, suh_odd?, suh1_odd?, svh_odd?, dn_suh_odd?, dn_codes_odd?, dn_svh_odd? -> y
 *            (dn_suh is down's suh; dn_codes / dn_svh are down's codes and svh, [K, N / 2] and [K, 1])
 * dn_suh / dn_suh_odd are DOWN's suh tables of the same experts (entry [1, rows/2]); read only with
 * h_rot = 1, where they are required (and dn_suh_odd with the odd class).
 *
 * THE DECLARATION: the weight list is divided evenly over the table positions in this order
 * (rad_builder.cpp:703-731), so an architecture lists the even class's codes run, its suh run,
 * [its suh.1 run,] its svh run, [gate_up: its dn_suh run,] then the odd class's in the same order --
 * each run the class's experts in entry order, each weight a one-plane view
 * (`RadWeightDecl::planes` = "codes", "suh", "suh.1", "svh") of the expert's one `exl3` encoding.
 * An odd expert count gives the even tables one more each, which is the rule's own remainder
 * (:711-729). EVERY TABLE POSITION TAKES ITS SHARE, OPTIONAL OR NOT, so an architecture issuing
 * gate_up binds dn_suh (and dn_suh_odd) whatever h_rot is -- the natural run is the one down's own
 * suh table already names (down's suh weights, read by two ops); with h_rot = 0 it is not read.
 * prep_gu's suh / suh1 runs are gate_up's (the same weights, read by both); gate_up binds them at
 * a_rot = 1 too and does not read them.
 *
 * XR, moe_exl3_prep_gu's output and gate_up's `a` at a_rot = 1: f16 [T, 2 * max(K, K_odd)], one row
 * per SORTED row (row i is slot sorted_tok[i], like the outputs). Row i of expert e (class c, input
 * width K_c) holds columns [0, K_c) the gate half's rotated input (suh) and [K_c, 2 * K_c) the up
 * half's (suh.1) -- a narrower class uses the first 2 * K_c of the row. It is the layout exl3r4d's
 * prep kernels write today in both bands (xr[row][m * K_c + k], row pitch 2 * x_cols;
 * kernels/exl3r4d/exl3r4d.hip, exl3r4d_prefill.hip), agreed with P4 (decode) and P5 (prefill).
 * WHAT IS NOT WRITTEN, and is left as it was: every row outside the experts' runs [off[0],
 * off[n_expert]) -- dropped slots, protected experts' rows -- and the columns past 2 * K_c. gate_up
 * reads none of them (its output rows there are zero whatever xr holds). A row inside a run whose
 * sorted_tok is negative (no scatter writes one) is zero. "Not written" is exact, so the per-op
 * oracle, which runs the host row on the pre-launch copy (core/runtime/oracle.cpp:405-423), compares
 * those bytes equal; a library that wrote them would be the one disagreeing.
 *
 * ONE a_rot PER DECLARED OP, A CHOICE PER PASS. a_rot is a fixed parameter like every other but M, so
 * an architecture that wants prep_gu + gate_up(a_rot = 1) at decode and gate_up(a_rot = 0) above it
 * declares gate_up twice -- a_rot = 1 with M ranged over the decode band, beside prep_gu over the same
 * band, and a_rot = 0 -- and picks at issue by the pass's row count. That is how the in-tree block
 * already chooses its router form: `router_gemm_topk_scatter` declared for M in [1,
 * router_fused_rows] and issued only when rows <= router_fused_rows, else the separate pair or the
 * split router (arch/common/rad_block_moe_fp8.h:1438-1447, 1527-1551). Both gate_up declarations
 * read the same weights, so their rows must ask the same stored form of `codes`.
 */

/* The ops, as exl3_positions() and exl3_geom_parse() take them. 0 and 1 are what version 3's
 * `gate_up` flag passed, so a caller written for it is unchanged. A four-class form (v6) is its op | EXL3_MOD4. */
enum { EXL3_DOWN = 0, EXL3_GATE_UP = 1, EXL3_PREP_GU = 2, EXL3_GU_DN = 3, EXL3_MOD4 = 4,
       EXL3_DOWN4 = EXL3_MOD4 | EXL3_DOWN, EXL3_GATE_UP4 = EXL3_MOD4 | EXL3_GATE_UP,
       EXL3_PREP_GU4 = EXL3_MOD4 | EXL3_PREP_GU };
/* The two-class op a form is of (what it computes); EXL3_MOD4 says only how its experts are tabled. */
static inline int exl3_op_base(int op) { return op & ~EXL3_MOD4; }

/* Where each operand of an op sits. `plane[c][p]` is class c's table of plane p (EXL3_P_*), -1 for a
 * plane the op does not have (down has no suh.1 and no dn_suh; prep_gu has only suh and suh.1) and for
 * every plane of a class past the op's `n_cls` (2, or 4 for a _mod4 op).
 * EXL3_P_DN_SUH is gate_up's view of DOWN's suh plane (h_rot = 1); the expert's own weight is planes
 * EXL3_P_CODES..EXL3_P_SVH. */
enum { EXL3_P_CODES = 0, EXL3_P_SUH = 1, EXL3_P_SUH1 = 2, EXL3_P_SVH = 3, EXL3_P_DN_SUH = 4,
       EXL3_P_DN_CODES = 5, EXL3_P_DN_SVH = 6, EXL3_N_P = 7 };
enum { EXL3_MAX_CLS = 4 };

typedef struct Exl3Positions {
    int x, sorted, offset, y, n_opd, n_cls;
    int plane[EXL3_MAX_CLS][EXL3_N_P];
} Exl3Positions;

static inline const Exl3Positions* exl3_positions(int op) {
#define EXL3_NO_CLASS { -1, -1, -1, -1, -1, -1, -1 }
    static const Exl3Positions gu = { 0, 6, 7, 13, 14, 2, { { 1, 2, 3, 4, 5, -1, -1 }, { 8, 9, 10, 11, 12, -1, -1 },
                                                            EXL3_NO_CLASS, EXL3_NO_CLASS } };
    static const Exl3Positions dn = { 0, 4, 5, 9, 10, 2, { { 1, 2, -1, 3, -1, -1, -1 }, { 6, 7, -1, 8, -1, -1, -1 },
                                                           EXL3_NO_CLASS, EXL3_NO_CLASS } };
    static const Exl3Positions pg = { 0, 3, 4, 7, 8, 2, { { -1, 1, 2, -1, -1, -1, -1 }, { -1, 5, 6, -1, -1, -1, -1 },
                                                          EXL3_NO_CLASS, EXL3_NO_CLASS } };
    static const Exl3Positions gd = { 0, 8, 9, 17, 18, 2, { { 1, 2, 3, 4, 5, 6, 7 }, { 10, 11, 12, 13, 14, 15, 16 },
                                                            EXL3_NO_CLASS, EXL3_NO_CLASS } };
    static const Exl3Positions gu4 = { 0, 6, 7, 23, 24, 4, { { 1, 2, 3, 4, 5, -1, -1 }, { 8, 9, 10, 11, 12, -1, -1 },
                                                             { 13, 14, 15, 16, 17, -1, -1 },
                                                             { 18, 19, 20, 21, 22, -1, -1 } } };
    static const Exl3Positions dn4 = { 0, 4, 5, 15, 16, 4, { { 1, 2, -1, 3, -1, -1, -1 }, { 6, 7, -1, 8, -1, -1, -1 },
                                                             { 9, 10, -1, 11, -1, -1, -1 },
                                                             { 12, 13, -1, 14, -1, -1, -1 } } };
    static const Exl3Positions pg4 = { 0, 3, 4, 11, 12, 4, { { -1, 1, 2, -1, -1, -1, -1 }, { -1, 5, 6, -1, -1, -1, -1 },
                                                             { -1, 7, 8, -1, -1, -1, -1 },
                                                             { -1, 9, 10, -1, -1, -1, -1 } } };
#undef EXL3_NO_CLASS
    switch (op) {
    case EXL3_GU_DN:    return &gd;
    case EXL3_PREP_GU:  return &pg;
    case EXL3_GATE_UP:  return &gu;
    case EXL3_DOWN4:    return &dn4;
    case EXL3_GATE_UP4: return &gu4;
    case EXL3_PREP_GU4: return &pg4;
    default:            return &dn;
    }
}

#define EXL3_OP_GATE_UP "moe_exl3_gate_up"
#define EXL3_OP_DOWN    "moe_exl3_down"
#define EXL3_OP_PREP_GU "moe_exl3_prep_gu"
#define EXL3_OP_GU_DN   "moe_exl3_gate_up_down"
#define EXL3_OP_GATE_UP4 "moe_exl3_gate_up_mod4"
#define EXL3_OP_DOWN4    "moe_exl3_down_mod4"
#define EXL3_OP_PREP_GU4 "moe_exl3_prep_gu_mod4"

/* An op's id (EXL3_*) from its name, or -1. */
static inline int exl3_op_of(const char* name) {
    static const char* const names[] = { EXL3_OP_DOWN, EXL3_OP_GATE_UP, EXL3_OP_PREP_GU, EXL3_OP_GU_DN,
                                         EXL3_OP_DOWN4, EXL3_OP_GATE_UP4, EXL3_OP_PREP_GU4 };
    for (int i = 0; name && i < (int)(sizeof names / sizeof names[0]); ++i)
        if (!strcmp(name, names[i])) return i;
    return -1;
}

/* The schemas, in this order: 0 gate_up, 1 down, 2 prep_gu, 3 gate_up_down, 4 gate_up_mod4, 5 down_mod4,
 * 6 prep_gu_mod4. A plugin's rad_kernel_schema_count / rad_kernel_schema_at answer with these. Static locals in a
 * static inline function, so a translation unit that includes this header and never asks pays nothing and warns
 * about nothing. */
static inline int exl3_schema_count(void) { return 7; }

static inline const RadOpSchema* exl3_schema_at(int i) {
#define EXL3_GEMM_PARAMS                                         \
        { "M",        RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE }, \
        { "N",        RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE }, \
        { "K",        RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE }, \
        { "n_expert", RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE }, \
        { "top_k",    RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE }, \
        { "dtype",    RAD_P_STR, RAD_REQUIRED, RAD_PROLE_NONE }, \
        { "codebook", RAD_P_STR, RAD_REQUIRED, RAD_PROLE_NONE }, \
        { "N_odd",    RAD_P_INT, RAD_OPTIONAL, RAD_PROLE_NONE }, \
        { "K_odd",    RAD_P_INT, RAD_OPTIONAL, RAD_PROLE_NONE }, \
        { "h_rot",    RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE }
    static const RadParamSpec p_gu[] = {
        EXL3_GEMM_PARAMS,
        { "a_rot",    RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
    };
    static const RadParamSpec p_dn[] = { EXL3_GEMM_PARAMS };
#undef EXL3_GEMM_PARAMS
    static const RadParamSpec p_pg[] = {
        { "M",        RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
        { "K",        RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
        { "n_expert", RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
        { "top_k",    RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
        { "dtype",    RAD_P_STR, RAD_REQUIRED, RAD_PROLE_NONE },
        { "K_odd",    RAD_P_INT, RAD_OPTIONAL, RAD_PROLE_NONE },
    };
    static const RadOperandSpec gate_up[14] = {
        { "a",             RAD_OPD_IN,   0 },
        { "codes",         RAD_OPD_WTAB, 0 },
        { "suh",           RAD_OPD_WTAB, 0 },
        { "suh1",          RAD_OPD_WTAB, 0 },
        { "svh",           RAD_OPD_WTAB, 0 },
        { "dn_suh",        RAD_OPD_WTAB, 1 },
        { "sorted_tok",    RAD_OPD_IN,   0 },
        { "expert_offset", RAD_OPD_IN,   0 },
        { "codes_odd",     RAD_OPD_WTAB, 1 },
        { "suh_odd",       RAD_OPD_WTAB, 1 },
        { "suh1_odd",      RAD_OPD_WTAB, 1 },
        { "svh_odd",       RAD_OPD_WTAB, 1 },
        { "dn_suh_odd",    RAD_OPD_WTAB, 1 },
        { "h",             RAD_OPD_OUT,  0 },
    };
    static const RadOperandSpec down[10] = {
        { "h",             RAD_OPD_IN,   0 },
        { "codes",         RAD_OPD_WTAB, 0 },
        { "suh",           RAD_OPD_WTAB, 0 },
        { "svh",           RAD_OPD_WTAB, 0 },
        { "sorted_tok",    RAD_OPD_IN,   0 },
        { "expert_offset", RAD_OPD_IN,   0 },
        { "codes_odd",     RAD_OPD_WTAB, 1 },
        { "suh_odd",       RAD_OPD_WTAB, 1 },
        { "svh_odd",       RAD_OPD_WTAB, 1 },
        { "y",             RAD_OPD_OUT,  0 },
    };
    static const RadParamSpec p_gd[] = {
        { "M",        RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
        { "N",        RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
        { "K",        RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
        { "n_expert", RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
        { "top_k",    RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
        { "dtype",    RAD_P_STR, RAD_REQUIRED, RAD_PROLE_NONE },
        { "codebook", RAD_P_STR, RAD_REQUIRED, RAD_PROLE_NONE },
        { "N_odd",    RAD_P_INT, RAD_OPTIONAL, RAD_PROLE_NONE },
        { "K_odd",    RAD_P_INT, RAD_OPTIONAL, RAD_PROLE_NONE },
    };
    static const RadOperandSpec gate_up_down[18] = {
        { "a",             RAD_OPD_IN,   0 },
        { "codes",         RAD_OPD_WTAB, 0 },
        { "suh",           RAD_OPD_WTAB, 0 },
        { "suh1",          RAD_OPD_WTAB, 0 },
        { "svh",           RAD_OPD_WTAB, 0 },
        { "dn_suh",        RAD_OPD_WTAB, 0 },
        { "dn_codes",      RAD_OPD_WTAB, 0 },
        { "dn_svh",        RAD_OPD_WTAB, 0 },
        { "sorted_tok",    RAD_OPD_IN,   0 },
        { "expert_offset", RAD_OPD_IN,   0 },
        { "codes_odd",     RAD_OPD_WTAB, 1 },
        { "suh_odd",       RAD_OPD_WTAB, 1 },
        { "suh1_odd",      RAD_OPD_WTAB, 1 },
        { "svh_odd",       RAD_OPD_WTAB, 1 },
        { "dn_suh_odd",    RAD_OPD_WTAB, 1 },
        { "dn_codes_odd",  RAD_OPD_WTAB, 1 },
        { "dn_svh_odd",    RAD_OPD_WTAB, 1 },
        { "y",             RAD_OPD_OUT,  0 },
    };
    static const RadOperandSpec prep_gu[8] = {
        { "a",             RAD_OPD_IN,   0 },
        { "suh",           RAD_OPD_WTAB, 0 },
        { "suh1",          RAD_OPD_WTAB, 0 },
        { "sorted_tok",    RAD_OPD_IN,   0 },
        { "expert_offset", RAD_OPD_IN,   0 },
        { "suh_odd",       RAD_OPD_WTAB, 1 },
        { "suh1_odd",      RAD_OPD_WTAB, 1 },
        { "xr",            RAD_OPD_OUT,  0 },
    };
    /* v6, the four-class forms: the two-class op's parameters with N_q / K_q for N_odd / K_odd, and its operands with
     * classes 1..3's tables after the routing (every one required but dn_suh_q, as dn_suh). */
#define EXL3_MOD4_SHAPE                                          \
        { "M",        RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE }, \
        { "N",        RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE }, \
        { "K",        RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE }, \
        { "n_expert", RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE }, \
        { "top_k",    RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE }, \
        { "dtype",    RAD_P_STR, RAD_REQUIRED, RAD_PROLE_NONE }, \
        { "codebook", RAD_P_STR, RAD_REQUIRED, RAD_PROLE_NONE }, \
        { "N_1",      RAD_P_INT, RAD_OPTIONAL, RAD_PROLE_NONE }, \
        { "K_1",      RAD_P_INT, RAD_OPTIONAL, RAD_PROLE_NONE }, \
        { "N_2",      RAD_P_INT, RAD_OPTIONAL, RAD_PROLE_NONE }, \
        { "K_2",      RAD_P_INT, RAD_OPTIONAL, RAD_PROLE_NONE }, \
        { "N_3",      RAD_P_INT, RAD_OPTIONAL, RAD_PROLE_NONE }, \
        { "K_3",      RAD_P_INT, RAD_OPTIONAL, RAD_PROLE_NONE }, \
        { "h_rot",    RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE }
    static const RadParamSpec p_gu4[] = {
        EXL3_MOD4_SHAPE,
        { "a_rot",    RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
    };
    static const RadParamSpec p_dn4[] = { EXL3_MOD4_SHAPE };
#undef EXL3_MOD4_SHAPE
    static const RadParamSpec p_pg4[] = {
        { "M",        RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
        { "K",        RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
        { "n_expert", RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
        { "top_k",    RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
        { "dtype",    RAD_P_STR, RAD_REQUIRED, RAD_PROLE_NONE },
        { "K_1",      RAD_P_INT, RAD_OPTIONAL, RAD_PROLE_NONE },
        { "K_2",      RAD_P_INT, RAD_OPTIONAL, RAD_PROLE_NONE },
        { "K_3",      RAD_P_INT, RAD_OPTIONAL, RAD_PROLE_NONE },
    };
#define EXL3_GU_CLASS(q)                                  \
        { "codes_" #q,  RAD_OPD_WTAB, 0 },                \
        { "suh_" #q,    RAD_OPD_WTAB, 0 },                \
        { "suh1_" #q,   RAD_OPD_WTAB, 0 },                \
        { "svh_" #q,    RAD_OPD_WTAB, 0 },                \
        { "dn_suh_" #q, RAD_OPD_WTAB, 1 }
    static const RadOperandSpec gate_up4[24] = {
        { "a",             RAD_OPD_IN,   0 },
        { "codes",         RAD_OPD_WTAB, 0 },
        { "suh",           RAD_OPD_WTAB, 0 },
        { "suh1",          RAD_OPD_WTAB, 0 },
        { "svh",           RAD_OPD_WTAB, 0 },
        { "dn_suh",        RAD_OPD_WTAB, 1 },
        { "sorted_tok",    RAD_OPD_IN,   0 },
        { "expert_offset", RAD_OPD_IN,   0 },
        EXL3_GU_CLASS(1), EXL3_GU_CLASS(2), EXL3_GU_CLASS(3),
        { "h",             RAD_OPD_OUT,  0 },
    };
#undef EXL3_GU_CLASS
#define EXL3_DN_CLASS(q)                                  \
        { "codes_" #q,  RAD_OPD_WTAB, 0 },                \
        { "suh_" #q,    RAD_OPD_WTAB, 0 },                \
        { "svh_" #q,    RAD_OPD_WTAB, 0 }
    static const RadOperandSpec down4[16] = {
        { "h",             RAD_OPD_IN,   0 },
        { "codes",         RAD_OPD_WTAB, 0 },
        { "suh",           RAD_OPD_WTAB, 0 },
        { "svh",           RAD_OPD_WTAB, 0 },
        { "sorted_tok",    RAD_OPD_IN,   0 },
        { "expert_offset", RAD_OPD_IN,   0 },
        EXL3_DN_CLASS(1), EXL3_DN_CLASS(2), EXL3_DN_CLASS(3),
        { "y",             RAD_OPD_OUT,  0 },
    };
#undef EXL3_DN_CLASS
#define EXL3_PG_CLASS(q)                                  \
        { "suh_" #q,    RAD_OPD_WTAB, 0 },                \
        { "suh1_" #q,   RAD_OPD_WTAB, 0 }
    static const RadOperandSpec prep_gu4[12] = {
        { "a",             RAD_OPD_IN,   0 },
        { "suh",           RAD_OPD_WTAB, 0 },
        { "suh1",          RAD_OPD_WTAB, 0 },
        { "sorted_tok",    RAD_OPD_IN,   0 },
        { "expert_offset", RAD_OPD_IN,   0 },
        EXL3_PG_CLASS(1), EXL3_PG_CLASS(2), EXL3_PG_CLASS(3),
        { "xr",            RAD_OPD_OUT,  0 },
    };
#undef EXL3_PG_CLASS
    static const RadOpSchema s[7] = {
        { EXL3_OP_GATE_UP, p_gu, (int)(sizeof p_gu / sizeof p_gu[0]), gate_up, 14,
          "EXL3 routed experts, gate and up: per sorted row, a[sorted_tok/top_k] scaled by suh "
          "(suh.1 for up) and rotated, times W_hat, rotated back and scaled by svh, then "
          "f16(silu(gate) * up); with h_rot = 1 that then scaled by down's suh (dn_suh) and "
          "rotated, which is down's first step; with a_rot = 1 `a` is moe_exl3_prep_gu's xr and "
          "the input rotation is not redone. One weight table a plane a parity class, expert e "
          "at entry e/2; rows past expert_offset[n_expert] are zero. expert_w is not applied. "
          "docs/OPS-exl3.md." },
        { EXL3_OP_DOWN, p_dn, (int)(sizeof p_dn / sizeof p_dn[0]), down, 10,
          "EXL3 routed experts, down: per sorted row, h[i] scaled by suh and rotated (with h_rot = "
          "1: h[i] as gate_up already rotated it), times W_hat, rotated back and scaled by svh, "
          "bf16. One weight table a plane a parity class, expert e at entry e/2; rows past "
          "expert_offset[n_expert] are zero. expert_w is not applied. docs/OPS-exl3.md." },
        { EXL3_OP_PREP_GU, p_pg, (int)(sizeof p_pg / sizeof p_pg[0]), prep_gu, 8,
          "EXL3 routed experts, gate_up's input rotation as an op: per sorted row, "
          "a[sorted_tok/top_k] scaled by suh and rotated, then by suh.1 and rotated, side by side "
          "(xr, f16 [T, 2*max(K, K_odd)]) -- what moe_exl3_gate_up with a_rot = 1 reads. Rows "
          "past expert_offset[n_expert] are zero. docs/OPS-exl3.md." },
        { EXL3_OP_GU_DN, p_gd, (int)(sizeof p_gd / sizeof p_gd[0]), gate_up_down, 18,
          "EXL3 routed experts, the decode chain as one op: y = moe_exl3_down(h_rot = 1) of "
          "moe_exl3_gate_up(a_rot = 0, h_rot = 1) of a, on the same routing; dn_suh / dn_codes / dn_svh "
          "are down's planes ([K, N/2]); h and the rotated input are scratch. Rows past "
          "expert_offset[n_expert] are zero. docs/OPS-exl3.md." },
        { EXL3_OP_GATE_UP4, p_gu4, (int)(sizeof p_gu4 / sizeof p_gu4[0]), gate_up4, 24,
          "moe_exl3_gate_up over FOUR classes of experts by number mod 4 (moe_gemm_q_mod4's): class q at "
          "[N_q, K_q], expert e at entry e/4 of class e%4; otherwise moe_exl3_gate_up. docs/OPS-exl3.md." },
        { EXL3_OP_DOWN4, p_dn4, (int)(sizeof p_dn4 / sizeof p_dn4[0]), down4, 16,
          "moe_exl3_down over FOUR classes of experts by number mod 4 (moe_gemm_q_mod4's): class q at "
          "[N_q, K_q], expert e at entry e/4 of class e%4; otherwise moe_exl3_down. docs/OPS-exl3.md." },
        { EXL3_OP_PREP_GU4, p_pg4, (int)(sizeof p_pg4 / sizeof p_pg4[0]), prep_gu4, 12,
          "moe_exl3_prep_gu over FOUR classes of experts by number mod 4: class q's input width K_q, "
          "expert e at entry e/4 of class e%4; xr f16 [T, 2*max(K_q)]; otherwise moe_exl3_prep_gu. "
          "docs/OPS-exl3.md." },
    };
    return i >= 0 && i < 7 ? &s[i] : 0;
}

/* ===================================================================== THE PLANES
 *
 * A table's entry e is one plane of one expert, exactly as the container holds it (the canonical
 * `exl3` encoding, include/exl3_encoding.h) -- this rank's slice of it, [rows, cols] in the plane's
 * own elements, the extents the core gives a plane (rad_enc_plane_dims, abi/rad_encoding.h:236-245):
 *
 *   codes   uK  [R, C]   element (i, j) = the K-bit trellis symbol of W_hat[i][j], LSB-first rows
 *   suh     f16 [1, C]   input scales (the gate's, for gate_up)                 TILED [0x1]
 *   suh1    f16 [1, C]   gate_up only: the up half's input scales (plane suh.1) TILED [0x1]
 *   svh     f16 [R, 1]   output scales (gate rows then up rows, for gate_up)    TILED [1x0]
 *   dn_suh  f16 [1, R/2] gate_up only: DOWN's suh of the same expert (down's [hidden, R/2] weight)
 *
 * with [R, C] the class's [rows, cols]. A reader takes the scales by element count (any rank whose
 * extents multiply to C or R), because the architecture's declared shape is what an issue carries.
 * A table operand is the engine's pointer table (stride[0] == 0) or the STACKED form a tool uses --
 * one rank-3 tensor [entries, rows, cols] with a real leading stride, which is also the form the
 * per-op oracle hands a host row (core/runtime/oracle.cpp:328-347, 405-418).
 *
 * What a DEVICE row stores is its own business (its layout hook; codes alone is what a trellis
 * kernel re-lays). The host reference has no layout hook and reads the planes as they are, which a
 * one-plane weight nobody re-lays is (rad_builder.cpp:1385, 1404-1411) -- so the host-only engine
 * serves it -- and which the per-op oracle restores for a plane a device row did re-lay: the device
 * row's unrelayout gives back the canonical plane, and a one-plane selection reaches the host row
 * as those bytes in the dtype its shape hook names (oracle.cpp:456-548, encoding.cpp:306-321). */

/* The extents of plane p of class c's [rows, cols] weight (dn_suh: down's suh, [1, rows / 2]; dn_codes / dn_svh,
 * moe_exl3_gate_up_down's down planes: [cols, rows / 2] and [cols, 1] -- [rows, cols] being gate_up's). */
static inline void exl3_plane_extents(int64_t rows, int64_t cols, int p, int64_t* r, int64_t* c) {
    if (p == EXL3_P_DN_CODES || p == EXL3_P_DN_SVH) {
        *r = cols;
        *c = p == EXL3_P_DN_SVH ? 1 : rows / 2;
        return;
    }
    *r = p == EXL3_P_SUH || p == EXL3_P_SUH1 || p == EXL3_P_DN_SUH ? 1 : rows;
    *c = p == EXL3_P_SVH ? 1 : p == EXL3_P_DN_SUH ? rows / 2 : cols;
}

/* ===================================================================== THE GEOMETRY, PARSED ONCE
 *
 * Every implementation reads the parameters through this, so a device row can never accept a
 * geometry its reference refuses. `two_classes` is whether the odd tables are present -- an operand
 * fact the parameters do not carry; a shape hook, which has no operands, passes
 * exl3_geom_two_classes(). A _mod4 op ignores it: its four classes are always present. */
typedef struct Exl3Geom {
    int     op;             /* the two-class op: EXL3_DOWN, EXL3_GATE_UP, EXL3_PREP_GU, EXL3_GU_DN (a _mod4 op's base) */
    int     gate_up;        /* 1 for gate_up AND prep_gu (stacked: suh and suh.1), 0 for down */
    int64_t M;              /* tokens (a shape hook reads a range's high bound) */
    int64_t n_expert, top_k;
    int     bits, codebook; /* K and EXL3_CB_* */
    int     n_class;        /* 2 with the odd tables, 1 without; 4 for a _mod4 op */
    int     h_rot;          /* 1: h crosses rotated (gate_up writes, down reads, down's step 1) */
    int     a_rot;          /* gate_up: 1 when `a` is prep_gu's xr */
    int64_t rows[EXL3_MAX_CLS], cols[EXL3_MAX_CLS];   /* class c's weight [rows, cols], c < n_class (prep_gu: rows 0,
                                                       * it has no N); past n_class, class 0's */
    int64_t x_cols;         /* max(cols): the token input's width (gate_up a_rot = 1: xr is 2 * x_cols) */
    int64_t y_cols;         /* the output's width: max(rows)/2 gate_up, max(rows) down, 2 * x_cols prep_gu */
} Exl3Geom;

/* Does this geometry name an odd class? -- N_odd or K_odd given. What a shape description of the
 * odd tables follows (present, or RAD_OPD_F_ABSENT). */
static inline int exl3_geom_two_classes(const RadParam* p, int n_p) {
    return rad_param_find(p, n_p, "N_odd") != 0 || rad_param_find(p, n_p, "K_odd") != 0;
}

/* RAD_OK, or: RAD_E_SCHEMA a required parameter absent; RAD_E_DTYPE a dtype or codebook this
 * interface has no name for; RAD_E_INVAL h_rot or a_rot neither 0 nor 1; RAD_E_SHAPE extents the
 * format cannot hold -- K % 128 (the input rotation), N % 128 for down and N % 256 for gate_up
 * (each half's output rotation, and with it the h_rot rotation of N / 2), a non-positive count, or
 * N_odd/K_odd differing from N/K without the odd class. `op` is EXL3_DOWN, EXL3_GATE_UP or
 * EXL3_PREP_GU (version 3 callers passed gate_up's 0/1, which are the first two), EXL3_GU_DN, or a _mod4
 * form (EXL3_DOWN4, EXL3_GATE_UP4, EXL3_PREP_GU4: N_q / K_q, every class checked as class 0 is). */
static inline int exl3_geom_parse(const RadParam* p, int n_p, int op_in, int two_classes, Exl3Geom* g) {
    static const char* const common[] = { "M", "K", "n_expert", "top_k", "dtype" };
    static const char* const gemm[] = { "N", "codebook", "h_rot" };
    static const char* const nq[EXL3_MAX_CLS] = { "N", "N_1", "N_2", "N_3" };
    static const char* const kq[EXL3_MAX_CLS] = { "K", "K_1", "K_2", "K_3" };
    const int mod4 = (op_in & EXL3_MOD4) != 0, op = exl3_op_base(op_in);
    const int prep = op == EXL3_PREP_GU, gd = op == EXL3_GU_DN, gate_up = op == EXL3_GATE_UP || prep || gd;
    const char* dt = rad_param_gets(p, n_p, "dtype", 0);
    const char* cb = rad_param_gets(p, n_p, "codebook", 0);
    const int64_t N = rad_param_geti(p, n_p, "N", 0), K = rad_param_geti(p, n_p, "K", 0);
    const int64_t No = rad_param_geti(p, n_p, "N_odd", N), Ko = rad_param_geti(p, n_p, "K_odd", K);
    const int64_t rows_mult = gate_up ? 256 : 128;
    int c;
    if (mod4 && gd) return RAD_E_SCHEMA;   /* there is no four-class gate_up_down (version 6) */
    memset(g, 0, sizeof *g);
    for (c = 0; c < (int)(sizeof common / sizeof common[0]); ++c)
        if (!rad_param_find(p, n_p, common[c])) return RAD_E_SCHEMA;
    for (c = 0; !prep && c < (int)(sizeof gemm / sizeof gemm[0]) - (gd ? 1 : 0); ++c)   /* gd: no h_rot (last) */
        if (!rad_param_find(p, n_p, gemm[c])) return RAD_E_SCHEMA;
    if (op == EXL3_GATE_UP && !rad_param_find(p, n_p, "a_rot")) return RAD_E_SCHEMA;
    if (!dt || (!prep && !cb)) return RAD_E_SCHEMA;   /* present, but not as strings */
    g->op       = op;
    g->gate_up  = gate_up;
    g->M        = rad_param_getdim(p, n_p, "M", 0);
    g->n_expert = rad_param_geti(p, n_p, "n_expert", 0);
    g->top_k    = rad_param_geti(p, n_p, "top_k", 0);
    g->h_rot    = prep ? 0 : gd ? 1 : (int)rad_param_geti(p, n_p, "h_rot", -1);
    g->a_rot    = op == EXL3_GATE_UP ? (int)rad_param_geti(p, n_p, "a_rot", -1) : 0;
    g->bits     = exl3_gemm_bits(dt);
    g->codebook = prep ? -1 : exl3_codebook_from_name(cb);
    if (g->bits < 0 || (!prep && g->codebook < 0)) return RAD_E_DTYPE;
    if ((g->h_rot != 0 && g->h_rot != 1) || (g->a_rot != 0 && g->a_rot != 1)) return RAD_E_INVAL;
    if (g->M < 1 || g->n_expert < 1 || g->top_k < 1) return RAD_E_SHAPE;
    if (!mod4 && !two_classes && ((!prep && No != N) || Ko != K)) return RAD_E_SHAPE;
    g->n_class = mod4 ? 4 : two_classes ? 2 : 1;
    for (c = 0; c < EXL3_MAX_CLS; ++c) {   /* two-class ops: class 1 is the odd class, 2 and 3 class 0 again */
        const int has = mod4 || c == 1;
        g->rows[c] = prep ? 0 : !has ? N : mod4 ? rad_param_geti(p, n_p, nq[c], N) : No;
        g->cols[c] = !has ? K : mod4 ? rad_param_geti(p, n_p, kq[c], K) : Ko;
    }
    g->x_cols = g->y_cols = 0;
    for (c = 0; c < EXL3_MAX_CLS; ++c) {
        if (g->cols[c] <= 0 || g->cols[c] % EXL3_HAD ||
            (!prep && (g->rows[c] <= 0 || g->rows[c] % rows_mult)))
            return RAD_E_SHAPE;
        if (g->cols[c] > g->x_cols) g->x_cols = g->cols[c];
        if (g->rows[c] > g->y_cols) g->y_cols = g->rows[c];
    }
    g->y_cols = prep ? 2 * g->x_cols : gd ? g->x_cols : g->y_cols / (gate_up ? 2 : 1);
    return RAD_OK;
}

/* ===================================================================== moe_exl3_gate_up_down AS ITS TWO HALVES
 *
 * The fused op read as the pair it is defined by, so every implementation reuses its gate_up and down: the
 * gate_up view (a_rot = 0, h_rot = 1, gate_up's operands, output h) and the down view (h_rot = 1, down's
 * geometry [K, N / 2], operands h, dn_codes, dn_suh, dn_svh, routing, output y). `h` is the caller's buffer (a
 * [T, max(N, N_odd) / 2] f16 tensor). The parameter arrays are the caller's: gate_up at most 11, down 10. */
static inline void exl3_gd_params(const RadParam* p, int n_p, RadParam* pg, int* n_pg, RadParam* pd, int* n_pd) {
    static const RadParam h1 = { "h_rot", RAD_P_INT, 1, 0, 0, 0.0 }, a0 = { "a_rot", RAD_P_INT, 0, 0, 0, 0.0 };
    const int64_t N = rad_param_geti(p, n_p, "N", 0), K = rad_param_geti(p, n_p, "K", 0);
    const int two = exl3_geom_two_classes(p, n_p);
    const int64_t No = rad_param_geti(p, n_p, "N_odd", N), Ko = rad_param_geti(p, n_p, "K_odd", K);
    int ng = 0, nd = 0;
    for (int i = 0; i < n_p; ++i) {
        const char* k = p[i].key;
        if (!strcmp(k, "h_rot") || !strcmp(k, "a_rot")) continue;
        if (ng < 9) pg[ng++] = p[i];
        if (!strcmp(k, "N") || !strcmp(k, "K") || !strcmp(k, "N_odd") || !strcmp(k, "K_odd")) continue;
        if (nd < 6) pd[nd++] = p[i];   /* M, n_expert, top_k, dtype, codebook */
    }
    pg[ng++] = h1;
    pg[ng++] = a0;
    pd[nd] = p[0]; pd[nd].key = "N"; pd[nd].kind = RAD_P_INT; pd[nd].ival = K; pd[nd].ihi = 0; ++nd;
    pd[nd] = pd[nd - 1]; pd[nd].key = "K"; pd[nd].ival = N / 2; ++nd;
    if (two) {
        pd[nd] = pd[nd - 1]; pd[nd].key = "N_odd"; pd[nd].ival = Ko; ++nd;
        pd[nd] = pd[nd - 1]; pd[nd].key = "K_odd"; pd[nd].ival = No / 2; ++nd;
    }
    pd[nd++] = h1;
    *n_pg = ng;
    *n_pd = nd;
}

/* The operand views over the fused op's operands `t` (18): gate_up's 14 (output h) and down's 10 (input h, output
 * t[17]). */
static inline void exl3_gd_operands(const RadTensor* t, const RadTensor* h, RadTensor* tg, RadTensor* td) {
    const Exl3Positions* gd = exl3_positions(EXL3_GU_DN);
    const Exl3Positions* gu = exl3_positions(EXL3_GATE_UP);
    const Exl3Positions* dn = exl3_positions(EXL3_DOWN);
    for (int c = 0; c < 2; ++c) {
        for (int q = EXL3_P_CODES; q <= EXL3_P_DN_SUH; ++q) tg[gu->plane[c][q]] = t[gd->plane[c][q]];
        td[dn->plane[c][EXL3_P_CODES]] = t[gd->plane[c][EXL3_P_DN_CODES]];
        td[dn->plane[c][EXL3_P_SUH]] = t[gd->plane[c][EXL3_P_DN_SUH]];
        td[dn->plane[c][EXL3_P_SVH]] = t[gd->plane[c][EXL3_P_DN_SVH]];
    }
    tg[gu->x] = t[gd->x];
    tg[gu->sorted] = td[dn->sorted] = t[gd->sorted];
    tg[gu->offset] = td[dn->offset] = t[gd->offset];
    tg[gu->y] = td[dn->x] = *h;
    td[dn->y] = t[gd->y];
}

/* Expert e's class and its entry in that class's tables. */
static inline int     exl3_class_of(const Exl3Geom* g, int64_t e) { return (int)(e % g->n_class); }
static inline int64_t exl3_entry_of(const Exl3Geom* g, int64_t e) { return e / g->n_class; }
/* How many entries class c's tables must hold: ceil((n_expert - c) / n_class). */
static inline int64_t exl3_class_entries(const Exl3Geom* g, int c) {
    return (g->n_expert - c + g->n_class - 1) / g->n_class;
}

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* EXL3_ABI_H */
