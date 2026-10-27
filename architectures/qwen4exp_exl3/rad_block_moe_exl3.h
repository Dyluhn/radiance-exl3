/* rad_block_moe_exl3.h -- the ROUTED feed-forward block with EXL3 experts.
 *
 *   rmsnorm -> quant_act_fp8 -> router gemm -> router_topk -> moe_scatter
 *                            -> moe_exl3_gate_up (bf16 in, silu(gate)*up out in f16)
 *                            -> moe_exl3_down (f16 in, bf16 out)
 *                            -> moe_gather
 *                            -> [shared expert: gate_up -> silu_mul + quant -> down -> gate]
 *                            -> add -> [all_reduce] -> residual add
 *
 * A COPY OF radiance's arch/common/rad_block_moe_fp8.h (UPSTREAM.md: release, commit, sha256),
 * changed only where an EXL3 expert differs from a block-fp8 or four-bit one. What changed, and why:
 *
 *   - THE EXPERT TABLES SELECT ONE PLANE EACH. EXL3 stores trellis `codes` and sign vectors
 *     (`suh`, `suh.1` for gate_up's up half, `svh`). Each plane is a view at its own dtype and
 *     extents, split into contiguous runs by parity or number mod 4. This lets the oracle hand
 *     the same plane bytes to the reference and device rows (D15, superseding D6's whole-weight
 *     operand). The EXL3 contract below lists each op's tables, including down's suh at gate_up.
 *   - THE BLOCK QUANTISES NO ACTIVATION. The ops read the bf16 normed input and their own f16
 *     intermediate, and apply the input sign vector, the 128-wide Hadamard and the output one
 *     inside the kernels (PLAN D5: EXL3's trellis values are not E4M3-exact, and E4M3 activations
 *     cost EXL3 x1.24 KLD). So `had_quant_act_fp8`, the rotated input pair `hr` and the gated
 *     quantiser between the GEMMs are gone, and gate_up writes silu(gate) * up itself.
 *   - ONE ENCODING A TABLE, NAMED BY IT. A projection's op takes its dtype string from its
 *     experts' codes dtype (`u4` -> `exl3k4`); the fp8 block's `format_of` table is gone.
 *
 * Everything else is the fp8 block's, unchanged: the router and its fused forms, the protected
 * bf16 experts and their grouped GEMM pair, the refusal to mix expert parallelism with a width
 * split, the calibration tap, rad_route_report, the gather that the write behind the block takes
 * at two ranks (ar_gather_hc_write) and the all-reduce elsewhere, and the parity width split with
 * its 384/256 spans -- whose 128-column unit is EXL3's Hadamard width as well as the fp8 block.
 *
 * IT IS rad_block_mlp_fp8.h WITH THE FEED-FORWARD ROUTED, and everything that block's header
 * argues -- why the gate/up fusion survives block scaling, why the norm and the residual add and
 * the collective are one op, why the activations between ops stay bf16 -- is argued there and not
 * repeated. What follows is only what routing changes.
 *
 * ============================== THE SHARED EXPERT ==============================
 *
 * Qwen3.5-MoE runs a dense expert beside the routed ones on every token, gated by a one-output
 * projection: `y = routed(x) + sigmoid(shared_gate(x)) * shared(x)` (`mlp.shared_expert.*` and
 * `mlp.shared_expert_gate.weight`). `moe_gather` applies the gate to the scalar and multiplies as
 * it sums the routed slots (docs/OPS.md's `scale_rows` says why `mul` cannot). A family without
 * one passes `Src::shared` null and the arm, buffers included, is not declared.
 *
 * ============================== ROW SLICING ==============================
 *
 * The expert intermediates are `top_k` times as tall as the step: at an 8192-token chunk and
 * top_k 8 the down projection's input is 65536 rows, 256 MiB at n_embd 2048. So the block runs in
 * passes of at most `Config::rows` tokens and declares those buffers at that height. Routing is
 * per pass (`expert_offset` and `sorted_tok` describe one pass), which is exact because a token's
 * routing depends only on that token.
 *
 * The default is 2048, the prefill chunk: core/sched/geometry.cpp clamps the chunk to the
 * linear-state checkpoint interval (2048 on this family), so the shipped configuration is one
 * pass. Slicing serves deployments with a larger chunk.
 *
 * ============================== EXPERT WEIGHT RUNS ==============================
 *
 * ============================== THE EXPERT WEIGHTS ARE FOUR RUNS ==============================
 *
 * The two ops take their weights as TABLES (RAD_OPD_WTAB), and a table is a run of CONSECUTIVELY
 * DECLARED handles -- `RAD_WTAB(first, n)` names the first and the count. Each takes a
 * projection's experts as TWO tables by parity, the even experts and the odd ones, because every
 * entry of a table has one geometry and a rank holding an uneven slice of each expert has two (see
 * Config::ff_lo). So the arrays are declared in separate loops -- gate_up, then down, each for the
 * even experts and then the odd ones -- and NOT as a group per expert. Interleaving them would
 * give runs of stride two, which the operand cannot express; the failure would be a gate_up table
 * whose second entry is a down projection. At four ranks the classes are FOUR, by the expert's
 * number mod 4 (Config::ff_lo4), as the fp8 block's `moe_gemm_q_mod4` -- the `_mod4` ops (interface
 * v6) over four runs a plane -- and the decode band issues their split forms below kGdMinRows rows and
 * `moe_exl3_gate_up_down_mod4` (interface v7) from there, as two ranks do with the two-class op.
 *
 * Each run's entry is the WHOLE stored weight, every plane of it, so an expert holds two logical
 * weights, gate_up and down, and is its own movement unit (`grp_expert_slot`, slots 0 and 1) and
 * RAD_ACCESS_CONDITIONAL, which is what lets the placement planner tier them and the heat engine
 * promote the hot ones.
 */
#ifndef RAD_BLOCK_MOE_EXL3_H
#define RAD_BLOCK_MOE_EXL3_H

#include <arch/rad_fp8.h>
#include <arch/rad_block_glm_router.h>
#include <rad_device.h>

#include "../../include/exl3_abi.h"
static_assert(EXL3_IFACE_VERSION == 7,
              "this block issues interface v7: one table a plane a parity class, gate_up also binding "
              "down's suh tables (dn_suh), both GEMMs carrying h_rot, gate_up a_rot, prep_gu, "
              "gate_up_down, and the _mod4 forms at four classes, gate_up_down's too");

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace rad {
namespace arch {

/* Most routing slots the sort serves in one workgroup, bounding `router_topk_scatter` (libr4d's
 * `kScatterSmall`). Fixed here so every rank decides identically at declare whether the op
 * exists. */
static const int64_t kMoeSortRows = 1024;

/* The skinny bf16 GEMM's decode band, within which the fused router op reproduces that GEMM's
 * arithmetic; above it the router uses the tiled kernel and the unfused pair. */
static const int64_t kMoeRouterFusedRows = 64;

/* ============================== THE EXL3 CONTRACT ==============================
 *
 * The encoding is include/exl3_encoding.h's (owner Q, PLAN D4) and the two ops' schemas, operand
 * positions and semantics include/exl3_abi.h's (owner R, interface v4: D15, EPI 1, prep_gu); this block reads
 * both through this namespace and nowhere else.
 *
 * ONE TABLE OPERAND A PLANE A PARITY CLASS (D15, superseding D6's one operand over every plane): a
 * host row with no layout hook then reads each plane as it is stored, and RADIANCE_OP_ORACLE hands
 * a reference row a single-plane selection through unchanged, where a multi-plane one is decoded to
 * a dense weight (P2 notes, findings C1-C3). gate_up writes silu(gate) * up in f16; down reads it
 * and writes bf16; both zero every row past expert_offset[n_expert]. N and K are the WEIGHT's:
 * gate_up's N is gate and up together, so its output is half as wide. */
namespace exl3 {

constexpr const char* kOpGateUp = EXL3_OP_GATE_UP;
constexpr const char* kOpDown   = EXL3_OP_DOWN;
constexpr const char* kOpPrepGu = EXL3_OP_PREP_GU;
constexpr const char* kOpGateUpDown = EXL3_OP_GU_DN;
/* The four-class forms (interface v6), at four ranks (Config::ff_lo4). */
constexpr const char* kOpGateUp4 = EXL3_OP_GATE_UP4;
constexpr const char* kOpDown4   = EXL3_OP_DOWN4;
constexpr const char* kOpPrepGu4 = EXL3_OP_PREP_GU4;
constexpr const char* kOpGateUpDown4 = EXL3_OP_GU_DN4;   /* interface v7 */

/* The planes each op takes, one table a plane a parity class, in the schema's order (exl3_abi.h
 * "THE DECLARATION"), with the operand slot each sits in (EXL3_P_*). The roles are the encoding's
 * (exl3_encoding.h): `codes` the trellis symbols, `suh` (and gate_up's `suh.1`, the up half's) one
 * f16 a column, `svh` one f16 a row. */
constexpr int kPlanesGateUp = 4, kPlanesDown = 3;
constexpr const char* kRoleGateUp[kPlanesGateUp] = { "codes", "suh", "suh.1", "svh" };
constexpr const char* kRoleDown[kPlanesDown]     = { "codes", "suh", "svh" };
/* gate_up's TABLES are its four planes and then DOWN's suh run (`dn_suh`, interface v3): the same
 * weights down's own suh table names, read by two ops. Every table position takes its share of the
 * declaration whatever h_rot is (exl3_abi.h "THE DECLARATION"). */
constexpr int kTablesGateUp = kPlanesGateUp + 1, kDownSuh = 1;   /* kRoleDown[kDownSuh] is "suh" */
constexpr int kSlotGateUp[kTablesGateUp] = { EXL3_P_CODES, EXL3_P_SUH, EXL3_P_SUH1, EXL3_P_SVH,
                                             EXL3_P_DN_SUH };
constexpr int kSlotDown[kPlanesDown]     = { EXL3_P_CODES, EXL3_P_SUH, EXL3_P_SVH };
/* prep_gu's tables are gate_up's suh and suh.1 runs, the same weights read by both (v4). */
constexpr int kTablesPrep = 2;
constexpr int kSlotPrep[kTablesPrep] = { EXL3_P_SUH, EXL3_P_SUH1 };
/* gate_up_down's tables (v5) are gate_up's five and then down's codes and svh runs: every run the
 * two ops read, once each (down's suh is the dn_suh table). */
constexpr int kTablesGd = kTablesGateUp + 2, kDownCodes = 0, kDownSvh = 2;
constexpr int kSlotGd[kTablesGd] = { EXL3_P_CODES, EXL3_P_SUH, EXL3_P_SUH1, EXL3_P_SVH, EXL3_P_DN_SUH,
                                     EXL3_P_DN_CODES, EXL3_P_DN_SVH };

/* EPI 1 (interface v3): with h_rot = 1 gate_up writes h already rotated for down -- down's own first
 * step, with down's suh of the same expert -- and down skips that step; the pair's output is the
 * same bits as with h_rot = 0, which is version 2's math on the same operands. 1 in every real
 * build; 0 only for an A/B build (-DQWEN4EXP_EXL3_H_ROT=0). Both ops always carry the same value:
 * a mixed pair is a caller error no kernel can see. */
#ifndef QWEN4EXP_EXL3_H_ROT
#define QWEN4EXP_EXL3_H_ROT 1
#endif
static_assert(QWEN4EXP_EXL3_H_ROT == 0 || QWEN4EXP_EXL3_H_ROT == 1, "h_rot is 0 or 1");
constexpr int kHRot = QWEN4EXP_EXL3_H_ROT;

/* THE ISSUE ORDER (interface v4, P4's request): gate_up's input rotation as an op of its own,
 * `moe_exl3_prep_gu`, writing xr, so that independent work runs in the gap before the routed GEMM
 * instead of beside it. With QWEN4EXP_EXL3_PREP_SPLIT = 1 a pass of at most kPrepRows rows (the
 * decode band) issues prep_gu, then the shared arm's gate, gate_up and quantiser, then
 * gate_up(a_rot = 1) over xr; a larger pass -- where the prefill rows rotate in the GEMM and must
 * not take xr's round trip -- and every pass at 0 issues the one gate_up(a_rot = 0) with the shared
 * arm behind it, as before. Same bits either way (exl3_abi.h v4). Both builds are tested
 * (tests/arch). THE DEFAULT IS 1, picked by engine step time (P6's harness, 2026-10-12, real model,
 * MTP off, two full rotations at equal VRAM): order 1 - order 0 = -0.258 / -0.31 / -0.273 / -0.215
 * ms a step at 1 / 2 / 4 / 8 concurrent, every interval excluding 0; prefill KL byte-identical; the
 * prefill stager stages every layer in both. 0 stays selectable (-DQWEN4EXP_EXL3_PREP_SPLIT=0): the
 * one gate_up, and none of the split build's limits -- docs/ARCH-qwen4exp_exl3.md. */
#ifndef QWEN4EXP_EXL3_PREP_SPLIT
#define QWEN4EXP_EXL3_PREP_SPLIT 1
#endif
static_assert(QWEN4EXP_EXL3_PREP_SPLIT == 0 || QWEN4EXP_EXL3_PREP_SPLIT == 1, "prep split is 0 or 1");
constexpr int kPrepSplit = QWEN4EXP_EXL3_PREP_SPLIT;
constexpr int64_t kPrepRows = 16;

/* THE DECODE CHAIN AS ONE OP (interface v5, P4b): with QWEN4EXP_EXL3_GD = 1 a pass of kGdMinRows to
 * kPrepRows rows issues the shared arm's gate, gate_up and quantiser and then `moe_exl3_gate_up_down`
 * -- down(h_rot 1) of gate_up(a_rot 0, h_rot 1), h in the op's scratch -- in place of the decode
 * band's gate_up form and down; the same bits as the pair (exl3_abi.h v5). Declared only where the
 * layer's gate_up and down carry ONE encoding (the op has one dtype and codebook) and no calibration
 * tap reads h, and between the layer's gate_up ops and its down (the prefill stager's keys, declare);
 * fewer rows, and every larger pass, take the build's other forms as before. 1 by default (engine
 * v12-v14, radiance 1.4.0: C=8 -0.25 ms a step from kGdMinRows 6); 0 is the split band alone. */
#ifndef QWEN4EXP_EXL3_GD
#define QWEN4EXP_EXL3_GD 1
#endif
static_assert(QWEN4EXP_EXL3_GD == 0 || QWEN4EXP_EXL3_GD == 1, "gate_up_down is 0 or 1");
constexpr int kGd = QWEN4EXP_EXL3_GD;
/* The fewest rows gate_up_down is issued for: below, the band's own form (prep_gu, the shared arm in the gap, gate_up
 * over xr, down) -- the fused launch loses there (P4b runs/p4b-s23: tp2 M=1 40.7 vs 39.7 us, M=2 63.4 vs 60.3) and
 * the split order's overlap is worth 0.26-0.31 ms a step at 1-2 concurrent (P6). */
/* THE BAND STARTS AT 3 since P10-K2 (2026-10-15; exl3_abi.h EXL3_GD_MIN_ROWS says why); at 4 before that because of
 * THE C=1 MTP VERIFY PASS (4 consecutive tokens of ONE sequence, which share experts -- few live experts, several
 * rows each): the split form loses bandwidth there (perfmoe tp2 M=4 with ~26 live
 * experts: chain 77.4-79.8 us a layer vs libr4d 72.2-73.0) and the fused op, which deals its units across the
 * experts, wins (69.0-69.8). P4b's engine passes from 4 stalled MTP C=4 steps; that was the co-residency bug below
 * (f30ee0c), not the band. */
constexpr int64_t kGdMinRows = EXL3_GD_MIN_ROWS;   /* include/exl3_abi.h: shared with exl3r4d */
/* THE DECLARED BAND'S BOTTOM (QWEN4EXP_EXL3_GD_DECL_MIN, a reading arm until measured, P10-K2): the M range gate_up_down
 * is declared over, which may start below the rows it is issued from (pass(): rows >= kGdMinRows). */
#ifndef QWEN4EXP_EXL3_GD_DECL_MIN
#define QWEN4EXP_EXL3_GD_DECL_MIN EXL3_GD_MIN_ROWS
#endif
static_assert(QWEN4EXP_EXL3_GD_DECL_MIN >= 1 && QWEN4EXP_EXL3_GD_DECL_MIN <= EXL3_GD_MIN_ROWS,
              "gate_up_down is declared from at most the rows it is issued from");
constexpr int64_t kGdDeclMin = QWEN4EXP_EXL3_GD_DECL_MIN;
/* THE FUSED BAND'S TOP (QWEN4EXP_EXL3_GD_MAX_ROWS): passes of more rows take the split forms. 16, the decode band's
 * top. v1.1.0 capped it at 8 because a 16-row pass (C=4 with MTP depth 3) stalled rank 0 inside the persistent
 * gate_up_down launch while rank 1 waited in the all-reduce. CAUSE (found 2026-10-14, fixed f30ee0c in exl3r4d): the
 * launch's prologue dealt prep rows to workgroups by index and every workgroup waited for all rows; with other kernels
 * holding slots, rows landed on workgroups that could not be resident. perfmoe --cu-pct 50 reproduces it without the
 * engine (old library 11 s a run at M=8-16, fixed 1.9 s, 0 bytes differ); with the fix, C=4-only MTP probes on
 * radiance 1.5.0 ran clean 2/2 at top 16 (v1.0.0 froze on that probe at the same token every time), ident unchanged. */
#ifndef QWEN4EXP_EXL3_GD_MAX_ROWS
#define QWEN4EXP_EXL3_GD_MAX_ROWS 16
#endif
static_assert(QWEN4EXP_EXL3_GD_MAX_ROWS >= EXL3_GD_MIN_ROWS && QWEN4EXP_EXL3_GD_MAX_ROWS <= 16,
              "the fused band's top is between its bottom and the decode band's 16 rows");
constexpr int64_t kGdMaxRows = QWEN4EXP_EXL3_GD_MAX_ROWS;
/* THE GD PASS'S ISSUE ORDER (QWEN4EXP_EXL3_GD_ORDER). 0: the shared arm's gate / gate_up and quantiser, then
 * gate_up_down, then the shared down -- which reads the quantiser's output, so a recorded pass orders it behind the
 * whole persistent launch (core/runtime/ctx.cpp Ctx::haz_mark). 1: the shared gate_up right behind the router (it
 * reads nothing the router writes, so the two run at once), the quantiser and the shared down next, then
 * gate_up_down unordered behind the down -- one shared stage serial where 0 leaves two (P5b, notes/p5.md). The
 * same bits either way. SHIPPED 1 (P5d): engine C=8 14.02-14.06 vs order 0's 14.12-14.16 ms/step, C=1/2/4 unchanged. */
#ifndef QWEN4EXP_EXL3_GD_ORDER
#define QWEN4EXP_EXL3_GD_ORDER 1
#endif
static_assert(QWEN4EXP_EXL3_GD_ORDER == 0 || QWEN4EXP_EXL3_GD_ORDER == 1, "the GD pass's issue order is 0 or 1");
constexpr int kGdOrder = QWEN4EXP_EXL3_GD_ORDER;
/* THE SHARED ARM ON THE SECOND LANE ON A gate_up_down PASS (QWEN4EXP_EXL3_SHARED_LANE, P10-K2 reading arm, 0 = off):
 * on one queue the arm's quantiser and down carry barrier bits, so the arm has to be issued AHEAD of the persistent
 * launch (order 1 above), which then starts only after the arm's gate_up and quantiser -- on the step's critical path.
 * On lane 1 (rad_lane, gemma4's dense-arm pattern) the arm runs beside the launch from the normed input on, and lane 0
 * joins it back before the gather. Every buffer the arm writes is already out of the sharing pool (declare). 2 also
 * takes the split form's passes (M <= 2 at the shipped band).
 * MEASURED (P10-K2, radiance 1.6.0, distinct prompts; notes/p10-k2.md): the same output (MTP idents equal, mini and
 * tp2). Link-free mini (8 layers, tp1 all_vram): MTP-off C=4 4.18/4.19 vs 4.48/4.48, C=8 5.34/5.33 vs 5.62/5.63, MTP
 * C=1 5.94 vs 6.21 ms a step -- ~35 us a layer. tp2 production flags (expert_tiered): CATASTROPHIC where the persistent
 * launch has few units -- MTP C=1 30.08 vs 16.57, MTP-off C=3 19.86 vs 12.77 ms -- and +0.15..+0.25 at C=4/8
 * independent. Not slot starvation alone: with exl3r4d's GD_WGS 75 (a quarter of the slots left free) MTP C=1 is 29.78
 * and MTP-off C=3 23.68 ms; MTP C=8, whose verify pass is the M > 16 band (no lane), is +15 ms with the lane, so the
 * MTP head's draft passes alone (3 lane joins a step) carry ms-scale stalls.
 * FACTORS (P10-K2 LANE, mini unless said): tp2 all_vram -- no stall and no gain (MTP C=1 4.17/4.17); tp1 with the
 * experts forced into the host pool -- stalled once (MTP-off C=3 4.16 -> 5.45) beside ~11 cores of other host work and
 * not on a quiet host (4.17 -> 3.96); that config with --deterministic (no moves) -- no stall, the lane faster. At tp2
 * production the stall is a MODE, not a per-step cost: in one MTP-off pass only C=3 doubled (C=1/2/4/8 equal), with
 * GD_WGS 75 it held from C=3 on; sclk/mclk unchanged and both cards' power down in the slow windows (idle inside the
 * step); the heat engine's and mover's work identical with or without the lane (MTP-off 17111/17147 promotions,
 * every histogram credited), so not worse placement. And OUTSIDE the stall the lane is equal or slower at tp2
 * (MTP-off C=4 13.99 vs 13.74, C=8 18.38 vs 18.23): nothing to win at production. PARKED, off; the slow mode's cause
 * (host-bound issue vs a device-side wait beside the mover's copy kernels -- three compute queues a card) stays open. */
#ifndef QWEN4EXP_EXL3_SHARED_LANE
#define QWEN4EXP_EXL3_SHARED_LANE 0   /* DEFAULT off (0); 3 the prefill band's passes (opt-in), 1 / 2 the parked arms */
#endif
/* OFF ON RADIANCE 1.7.0 (port170, 2026-10-17): rc-1.4 shipped 3, and at tp2 production flags (distinct prompts, one
 * pass each, the same session) 0 beats it on every reading: prefill 8K / 24K 1734 / 1743 vs 1720 / 1635 tok/s, prefill
 * beside 4 decoders 1614 vs 1458, TTFT 110 vs 132 ms (one request) and 1115 vs 1286 (eight), MTP-off decode C=4 / C=8
 * 14.16 / 18.23 vs 15.96 / 19.86 ms. Likely why: 1.7.0 issues its own collectives on lane 1. 3 stays an opt-in arm.
 * 3 (P10-K5 TTFT ledger, lever L-T1): the PREFILL band's passes only (neither gate_up_down nor the split decode form).
 * A recorded prefill pass lets only an op's first dispatch run unordered beside the ops in flight, so the baseline's
 * shared expert runs beside its one routed GEMM kernel -- ~1.2 of its 1.55 ms a 2048-token mini step hidden -- while ours
 * can only sit beside the last of our routed op's launches (K5: recorded passes off cost base 1.2 ms a chunk, ours ~0).
 * On lane 1 the arm overlaps the whole routed op. MEASURED (K5, mini tp1 all_vram, MTP off, one request, two rounds,
 * ident unchanged): TTFT 2048 tokens 70.80 / 70.87 ms vs rc-1.3 74.29 / 74.19 (baseline 70.66 / 70.85); 8192 269.92 /
 * 270.46 vs 283.64 / 282.44 (baseline 271.59 / 272.19); 512 29.78 / 30.08 vs 30.86 / 30.26 -- more than the arm's own
 * 1.43 ms a 2048 chunk: on lane 0 its quantiser and down also drained our routed launches. The M > 16 band also takes
 * MTP verify at C >= 8 (32+ rows): see notes/p10-k5.md for the decode and tp2 readings. rc-1.4 makes 3 the default
 * (P10-RC14; the tp2 production A/B in notes/p10-k4.md). */
static_assert(QWEN4EXP_EXL3_SHARED_LANE >= 0 && QWEN4EXP_EXL3_SHARED_LANE <= 3, "the shared lane is 0, 1, 2 or 3");
constexpr int kSharedLane = QWEN4EXP_EXL3_SHARED_LANE;

/* WHO TAKES AN UNEVEN SPLIT'S EXTRA BLOCKS (QWEN4EXP_EXL3_EXTRA_RANK). -1: the rank taking them alternates with the
 * expert's parity (qwen4exp_exl3.cpp; equal routed work a rank). r >= 0: rank r's place comes first in BOTH parities,
 * so it takes the extra blocks of every expert (at tp2 of 5 blocks: 3 / 3 for it, 2 / 2 for the other) -- the A/B for
 * a box whose ranks are not alike (one card on a narrow host path). A test arm, off. */
#ifndef QWEN4EXP_EXL3_EXTRA_RANK
#define QWEN4EXP_EXL3_EXTRA_RANK -1
#endif
static_assert(QWEN4EXP_EXL3_EXTRA_RANK >= -1, "the extra blocks' rank is -1 (by parity) or a rank");
constexpr int kExtraRank = QWEN4EXP_EXL3_EXTRA_RANK;

/* PREFILL STAGING, DECIDED PER SETUP (exl3_stage.h: from a pooled expert's zero-copy reads a step; EXL3_PREFILL_STAGE overrides).
 * radiance 1.4.0's prefill stager copies a routed layer's pooled experts onto the card ahead of the layer
 * (core/place/stager.h) and gives an architecture no switch; it turns itself off, by name ("routed layers %d and %d
 * are read by interleaved ops ...; staging is off"), when two routed layers' spans of declared expert readers
 * overlap. So when staging is off this build APPENDS one op after the language model's last (no real op's index or
 * order moves): a moe_exl3_prep_gu over layer 0's suh / suh.1 tables, ranged M 1..1, with no buffer read or written
 * and never issued -- a declared reader, which counts for the spans whether or not it is ever issued
 * (core/build/rad_builder.cpp finish_weight_uses). Layer 0's span then reaches past every other layer's and the
 * stager stages nothing: pooled experts are read zero-copy by the kernels' miss path, as in decode. Its one other
 * effect: layer 0's experts' last declared use is that op, which only widens their interval (later release of a
 * staged slot -- the builder's "safe direction"). QWEN4EXP_EXL3_PREFILL_STAGE: -1 automatic (every shipped build),
 * 0 / 1 forced (test builds and A/Bs). */
#ifndef QWEN4EXP_EXL3_PREFILL_STAGE
#define QWEN4EXP_EXL3_PREFILL_STAGE -1
#endif
static_assert(QWEN4EXP_EXL3_PREFILL_STAGE >= -1 && QWEN4EXP_EXL3_PREFILL_STAGE <= 1,
              "prefill staging is -1 (decided at declare from EXL3_PREFILL_STAGE), 0 or 1");
constexpr int kPrefillStageForced = QWEN4EXP_EXL3_PREFILL_STAGE;

/* TEST-ONLY WIRING MUTATIONS, 0 in every real build: tests/arch/CMakeLists.txt builds two plugins
 * with 1 and 2, never installed, as the end-to-end differential's controls -- defects that test
 * must see. They live in the DECLARATION, consistently with the issue, because the engine refuses
 * an issue naming another table run than the one declared, by name, before anything is computed
 * (measured: "a table's run is fixed by the declaration") -- so a defect there is caught already,
 * and a numerical control has to be one the engine cannot see. 1: the parity classes' tables
 * swapped (every expert reads its neighbour's weights). 2: gate_up's `suh` view selects plane
 * `suh.1` and the `suh.1` view `suh` (gate and up read each other's input scales). 3: gate_up's
 * dn_suh tables are the OTHER class's (every expert rotates h with its neighbour's down scales) --
 * visible only where h_rot = 1 is honoured, and the only control that sees dn_suh at all. */
#ifndef QWEN4EXP_EXL3_TEST_MUTATION
#define QWEN4EXP_EXL3_TEST_MUTATION 0
#endif
constexpr int kTestMutation = QWEN4EXP_EXL3_TEST_MUTATION;

/* THE CUT UNIT. Every width this block slices -- a rank's share of an expert, the parity classes'
 * 384 and 256 -- must be whole 128-column groups: the Hadamard is 128 wide and must not be cut, and
 * 128 is also eight of the trellis's 16x16 tiles. It is the fp8 block's unit too (RAD_FP8_BLOCK),
 * which is why the architecture's parity split serves both unchanged; the assert keeps that true. */
constexpr int64_t kCut = 128;
static_assert(kCut == RAD_FP8_BLOCK, "the parity split's unit is the fp8 block and the Hadamard width");
/* A PLAIN (protected, bf16) expert's split unit: it has no scale blocks, so a rank's share of it is
 * whole output tiles of the bf16 grouped GEMM that serves it, 64 wide (declare, the protected pair). */
constexpr int64_t kPlainTile = 64;

/* What a stored expert is, as this block asks it: EXL3 at a K some op dtype names (and with it the
 * op's dtype and codebook strings), a plain bf16 expert a quantised layer keeps (MoeEXL3::prot),
 * or something it does not serve. exl3_encoding_read refuses an "exl3" encoding that is not
 * exactly what the quantiser writes, so a malformed one is unserved, not misread. */
enum Kind { kUnserved = 0, kExl3 = 1, kPlainBf16 = 2 };
struct Enc {
    int         bits = 0, codebook = 0, stacked = 0;
    const char* dtype = nullptr;      /* "exl3k4" ... */
    const char* cb    = nullptr;      /* "mul1" ...   */
};
inline Kind kind_of(const RadEncoding& e, Enc* out = nullptr) {
    if (rad_enc_is(&e, "plain") && e.n_planes == 1 && e.plane[0].dtype == RAD_BF16) return kPlainBf16;
    Enc x;
    if (exl3_encoding_read(&e, &x.bits, &x.codebook, &x.stacked) != RAD_OK) return kUnserved;
    x.dtype = exl3_gemm_dtype(x.bits);
    x.cb    = exl3_codebook_name(x.codebook);
    if (!x.dtype || !x.cb) return kUnserved;
    if (out) *out = x;
    return kExl3;
}

/* Whether this process was started with `flag` among its arguments. The ABI hands a block no
 * placement (RadBuildCtx), so the one way it can see a serve option is the process's own command
 * line: /proc/self/cmdline, NUL-separated. False where there is none (not Linux, or a host that
 * embeds the engine without a radiance command line) -- a check that can miss, never one that
 * refuses wrongly. */
inline bool process_has_flag(const char* flag) {
    FILE* f = std::fopen("/proc/self/cmdline", "rb");
    if (!f) return false;
    std::string arg;
    bool hit = false;
    for (int ch; (ch = std::fgetc(f)) != EOF;) {
        if (ch) { arg.push_back((char)ch); continue; }
        hit = hit || arg == flag;
        arg.clear();
    }
    std::fclose(f);
    return hit || arg == flag;
}

}  /* namespace exl3 */



/* ---- RADIANCE_DEBUG_ROUTING: the routing WEIGHTS of token 0, which nothing else reports.
 *
 * The core's histogram (core/runtime/ctx.cpp) shows which experts were chosen and dbg_resid shows
 * each arm's magnitude; neither shows the weights, so a top-k that sums to 0.4 instead of 1 would
 * go unnoticed. Synchronises the stream. Off unless set: a layer number, or negative for all.
 * `f32`: the weights are f32 rather than bf16 (the sigmoid router). */
inline void moe_debug_weights(RadCtx* c_, int layer, int64_t top_k, rad_buf ids, rad_buf ew,
                              bool f32 = false) {
    static const char* dbg = std::getenv("RADIANCE_DEBUG_ROUTING");
    if (!dbg || !ids || !ew) return;
    static const int want = (int)std::strtol(dbg, nullptr, 10);
    if ((want >= 0 && layer != want) || !dbg_first(layer, "moe")) return;

    void* pi = rad_buf_ptr(c_, ids);
    void* pw = rad_buf_ptr(c_, ew);
    if (!pi || !pw) return;
    RadStream s = rad_stream(c_);
    std::vector<int32_t>  vi((size_t)top_k);
    std::vector<uint16_t> vw((size_t)top_k);
    std::vector<float> vf((size_t)top_k);
    rad_memcpy_async(vi.data(), pi, top_k * 4, s);
    rad_memcpy_async(f32?(void*)vf.data():(void*)vw.data(), pw, top_k * (f32?4:2), s);
    rad_stream_sync(s);

    std::string line;
    double sum = 0.0;
    for (int64_t r = 0; r < top_k; ++r) {
        const uint32_t u = (uint32_t)vw[(size_t)r] << 16;
        float f;
        std::memcpy(&f, &u, 4);
        if (f32) f = vf[(size_t)r];
        sum += f;
        char b[48];
        std::snprintf(b, sizeof b, "%s%d:%.4f", r ? " " : "", vi[(size_t)r], f);
        line += b;
    }
    std::fprintf(stderr, "D moe layer %d token 0 top_k = %s  (sum %.4f)\n",
                 layer, line.c_str(), sum);
}

/* Most experts a layer protects on one rank (MoeEXL3::prot): libr4d's sort takes at most eight in
 * its `protect` list. */
constexpr int64_t kMaxProtected = 8;

struct MoeEXL3 {
    struct Config {
        /* Routed experts this rank owns. Equal to `n_expert_all` unless experts are split across
         * ranks, which only a calibration run does (a Hessian spans an expert's whole K); otherwise
         * each rank holds a slice of every expert (see ff_lo). */
        int64_t n_expert     = 0;
        /* The model's expert count: the router's width and what top_k picks from. 0 means
         * n_expert.
         *
         * The router stays replicated under expert parallelism: every rank scores every expert from
         * the same residual stream and picks the same top_k, so ranks agree on routing without a
         * collective. Each sorts only the slots in its own range and the end-of-block all-reduce
         * sums the partials. */
        int64_t n_expert_all = 0;
        /* This rank's first expert id in the model's numbering. moe_scatter subtracts it so the
         * offsets, counts and weight table are local. */
        int64_t expert_base  = 0;
        int64_t top_k        = 0;
        int64_t n_ff_exp     = 0;   /* one expert's intermediate width, PER RANK */
        /* THE ROUTED EXPERTS' SLICE ON THIS RANK WHEN THEIR WIDTH DOES NOT HALVE INTO WHOLE
         * 128-COLUMN GROUPS (exl3::kCut): expert e's intermediate columns [ff_lo[e % 2], ff_hi[e % 2]) in the model's
         * numbering, the slice depending on the expert's PARITY. hi 0 is the even split (or no
         * split). `n_ff_exp` is then the wider slice, which is what the buffers between the two
         * grouped GEMMs are.
         *
         * This family's expert is 640 columns (five 128-blocks), so two ranks split each expert
         * three blocks and two. Alternating by parity balances the ranks on average; the imbalance
         * is a fifth of an expert per unit of even/odd difference, versus whole experts under
         * expert parallelism, which the all-reduce turns into waiting (about 1 ms a step at one
         * sequence).
         *
         * Each rank computes its gate/up rows and its K slice of down for every routed expert; the
         * routing weight scales each partial and the all-reduce sums them. Rounding differs (two
         * partial sums), so this is checked by agreement, not bytes. */
        int64_t ff_lo[2] = { 0, 0 }, ff_hi[2] = { 0, 0 };
        /* Or by number mod 4 when four ranks split each expert: expert e's columns
         * [ff_lo4[e % 4], ff_hi4[e % 4]), served by `moe_gemm_q_mod4`, taking precedence over the
         * parity slices. hi 0 offers none.
         *
         * Parity gives the extra block of a five-block expert two places to go, so at four ranks
         * two of them carry three blocks of every even/odd pair and two carry two -- and the
         * block's all-reduce has the narrow ranks wait for the wide ones. Four classes give it
         * four: each rank takes two blocks of one class and one of each other, five of every four
         * experts. Any count of quantised experts serves: a layer that keeps two plain bf16 has
         * 510, and its first two classes hold one expert more than the last two (the declaration
         * gives a table op's first tables the remainder). `n_ff_exp` is the widest slice. "e" is
         * the expert's place in the sort, as for the parity classes.
         *
         * Served by the `_mod4` ops (interface v6), which the architecture's split asks for at
         * four ranks exactly as the in-tree one does. */
        int64_t ff_lo4[4] = { 0, 0, 0, 0 }, ff_hi4[4] = { 0, 0, 0, 0 };
        int64_t n_ff_shared  = 0;   /* the shared expert's, per rank; 0 = no shared expert */
        /* The shared arm's slice on this rank when its width does not halve into whole fp8 scale
         * blocks: columns [shared_lo, shared_hi) in model numbering, `n_ff_shared` its width. hi 0
         * is the even split (or none at world 1); an empty slice declares no arm on this rank.
         *
         * The 640-column shared expert is split three blocks and two rather than run on one rank,
         * which would leave the other waiting at the all-reduce (about 20 us a layer at one
         * sequence). Each rank computes its gate/up columns and K slice of down, scales the partial
         * by the replicated per-token gate, and the all-reduce sums them:
         * `sgate * (down_0 + down_1)`. Checked by agreement, not bytes.
         *
         * The arm runs behind the routed stages on the same queue (see pass()), filling compute
         * units a routed GEMM's tail leaves idle. */
        int64_t shared_lo = 0, shared_hi = 0;
        int     norm_topk    = 1;   /* renormalise the kept routing weights to sum to one */
        /* GLM routing (GlmRouter): sigmoid scores with a correction bias, top-k, `route_scale` on
         * the kept weights, which are then f32 in `ew`. */
        int     sigmoid_router = 0;
        /* With `sigmoid_router`: the selection, the scatter and an ungated shared arm's gate fill in
         * one launch (`router_sigmoid_topk_scatter`) wherever the pass fits the sort's one
         * workgroup; the separate ops otherwise. Byte-identical either way. */
        int     router_fuse = 0;
        /* Block-fp8 experts whose scale plane is f32 (format 9, dtype fp8a8f32). */
        int     expert_f32_scale = 0;
        double  route_scale = 1.0;
        /* Clamp of the SwiGLU product (swiglu_clamped); 0 is none. */
        double  swiglu_limit = 0.0;
        /* With `swiglu_limit`: the clamped product and its quantiser as one launch
         * (GateQuantFP8::fuse_clamped), for the routed and the shared arm. */
        int     swiglu_quant = 0;
        /* A shared arm without its sigmoid gate: the gate plane is filled with 1. */
        int     shared_ungated = 0;
        /* THE fp8 BLOCK'S STORED-FORMAT FLAGS ARE GONE: every routed expert this block serves is
         * EXL3 (or a protected plain bf16 one), which probe() checks and declare() checks again
         * expert by expert -- one table of experts takes one encoding.
         *
         * `expert_rot` STAYS, ALWAYS 0, because rad_block_mtp_hc.h reads it off the MTP head's
         * Src to decide whether the connection in front of the block writes a rotated E4M3 copy
         * of its input. EXL3's rotations are inside the kernels and read the bf16 input, so the
         * answer is no; declare() refuses a caller that says otherwise. */
        int     expert_rot   = 0;
        /* The layer and the GLOBAL expert whose gate_up probe() read the format from and therefore
         * mapped; declare() maps every other. -1 for the layer: none. */
        int64_t probed_layer = -1;
        int64_t probed_expert = 0;
        /* Experts from `probed_expert` on that probe() mapped: it skips experts a quantised layer
         * keeps bf16 (see `prot`) to reach the first in the layer's format. */
        int64_t probed_count = 1;
        /* The GPTQ calibration tap (rad_arch.h). On, the block declares an f32 [K, K] PERSIST
         * accumulator and a `gram_accum` for each expert projection, runs until
         * RADIANCE_CALIB_TOKENS tokens, writes them to RADIANCE_CALIB_DIR and switches off.
         *
         * A separate run, not a serving mode: the accumulators are 1.34 GB over 48 layers at this
         * model's widths and the dump synchronises the stream. rad-convert reads the files.
         *
         * IT TAPS THE EXPERT OPS' INPUTS AS THEY ARE HANDED THEM, which for EXL3 is the
         * unrotated bf16 normed input and the f16 gated intermediate -- the model's own basis.
         * Each expert's `suh` and Hadamard are applied inside the kernels and differ by expert,
         * so a Hessian in a rotated basis would describe no one expert; the unrotated Gram is
         * what PLAN P7.4 means to hand an EXL3 quantiser. The down tap reads a bf16 copy of the f16 intermediate,
         * because `gram_accum` reads `bf16` or `fp8a8` and nothing else (docs/OPS.md): the
         * Hessian is of that rounding, three mantissa bits short of what the GEMM contracts. */
        int     calib        = 0;
        int64_t rows         = 0;   /* tokens per pass; 0 means the whole step */
    };

    /* Only the four planes with one row per (token, slot) are a pass tall; everything else has one
     * row per token, is sized at max_tok and sliced by the pass offset. This is required, not an
     * optimisation: LinearFP8 and GateQuantFP8 slice input and output with the same `r0`, so a
     * pass-tall shared-expert output would be overrun on the second pass. */
    struct Wire {
        rad_buf x       = 0;   /* [max_tok, n_embd]  the residual stream */
        ActFP8  h{};           /* [max_tok, n_embd]  normed input, then the block's delta */
        /* NEVER DECLARED AND NEVER READ: the fp8 block's rotated E4M3 copy of the input, which
         * EXL3 experts do not take. It stays a member because rad_block_mtp_hc.h names it, under
         * `Config::expert_rot`, which is always 0 here. */
        ActFP8  hr{};
        /* the routing -- one row per TOKEN, so the step's height */
        rad_buf logits  = 0;   /* [max_tok, n_expert_all] bf16 -- the MODEL's expert count */
        rad_buf ids     = 0;   /* [max_tok, top_k]      i32 */
        rad_buf ew      = 0;   /* [max_tok, top_k]      F32 for sigmoid_router, BF16 otherwise */
        rad_buf eoff    = 0;   /* [n_expert + 1]        i32 */
        rad_buf ecnt    = 0;   /* [n_expert]            i32 */
        /* the expert planes, one row per (token, slot), a pass tall */
        rad_buf sorted  = 0;   /* [rows * top_k]               i32, pass-relative slot indices */
        /* Each slot's row in `sorted`, -1 when dropped; with it the sort writes the gather's slot
         * table (moe_scatter_inv, moe_gather_inv). Optional. */
        rad_buf inv     = 0;   /* [rows * top_k]               i32 */
        /* silu(gate) * up, rounded to f16 by gate_up itself (exllamav3's rounding) -- and down's
         * input as it stands. There is no [rows * top_k, 2 * n_ff_exp] gate/up plane: the gating
         * is in gate_up's epilogue. */
        rad_buf eff     = 0;   /* [rows * top_k, n_ff_exp]     f16 */
        rad_buf edn     = 0;   /* [rows * top_k, n_embd]       bf16, in SORTED order */
        /* back in token order, and the shared expert beside it: the step's height again */
        rad_buf moe     = 0;   /* [max_tok, n_embd]           bf16 */
        rad_buf sgu     = 0;   /* [max_tok, 2 * n_ff_shared]  bf16 */
        ActFP8  sff{};         /* [max_tok, n_ff_shared] */
        rad_buf sout    = 0;   /* [max_tok, n_embd]           bf16 */
        rad_buf sgate   = 0;   /* [max_tok, 1]                bf16 */
    };

    /* Checkpoint names. `experts` is a prefix ("....mlp.experts") from which per-expert names are
     * built; `shared` likewise for the dense expert, null when the family has none (then
     * `shared_gate` must be null too).
     *
     * The config travels here, not as a declare() argument, so declare() has the same signature
     * as MlpFP8's and MtpBlockFP8 can be a template over both. */
    struct Src {
        Config      cfg{};
        const char* norm        = nullptr;
        const char* router      = nullptr;   /* mlp.gate.weight, [n_expert, n_embd] bf16 */
        const char* router_bias = nullptr;   /* GLM noaux_tc correction bias */
        const char* experts     = nullptr;   /* prefix; ".<e>.{gate,up,down}_proj.weight" */
        const char* shared      = nullptr;   /* prefix; ".{gate,up,down}_proj.weight" */
        /* Stacked checkpoints (recent releases) store all experts in two tensors:
         * `mlp.experts.gate_up_proj` [n_expert, 2*n_ff, n_embd] and `mlp.experts.down_proj`
         * [n_expert, n_embd, n_ff]. Set these instead of `experts`; each expert maps to a dim-0
         * slice (rad_arch.h: map_slice; core/format/checkpoint.cpp takes it as a view, with a
         * block-fp8 checkpoint's stacked `_scale_inv` at the same index). The stacked gate_up is
         * already fused gate rows first, as this block wants. */
        const char* experts_gate_up = nullptr;
        const char* experts_down    = nullptr;
        const char* shared_gate = nullptr;   /* mlp.shared_expert_gate.weight, [1, n_embd] */
    };

    Geom   g{};
    Config c{};
    Wire   w{};
    int    layer = 0;

    rad_weight w_norm = 0, w_router = 0, w_sgate = 0;
    GlmRouter glm_router{};
    /* The input was normed and quantised by the caller; this block declares no norm. */
    bool       ext_in = false;
    /* The runs: one a plane, each split by parity, indexed [plane][place in the sort]. Only the
     * FIRST handle of a class and the count reach an issue (RAD_WTAB); the rest are held so that a
     * declaration failure can be reported against the expert it happened on. */
    std::vector<rad_weight> w_gu[exl3::kPlanesGateUp], w_dn[exl3::kPlanesDown];
    /* The classes the runs are split into: 2 by parity, or 4 by number mod 4 (Config::ff_lo4), the
     * `_mod4` ops' -- expert place s in class s % ncls. */
    int ncls = 2;
    /* An op's form at this layer's class count: the two-class op, or its `_mod4` form. */
    int form(int op) const { return ncls == 4 ? op | EXL3_MOD4 : op; }
    /* The entries of class q's tables: ceil((n_reg - q) / ncls), the declaration's own remainder rule. */
    int64_t class_n(int q) const { return (n_reg - q + ncls - 1) / ncls; }
    /* Table t of an op (EXL3_GATE_UP, EXL3_DOWN, EXL3_PREP_GU, EXL3_GU_DN), in its schema order:
     * gate_up's planes, then down's suh run (dn_suh); down's planes; prep_gu's suh and suh.1 --
     * gate_up's; gate_up_down's: gate_up's five, then down's codes and svh. */
    const std::vector<rad_weight>& table_run(int op, int t) const {
        if (op == EXL3_GU_DN && t >= exl3::kTablesGateUp)
            return w_dn[t == exl3::kTablesGateUp ? exl3::kDownCodes : exl3::kDownSvh];
        return op == EXL3_GATE_UP || op == EXL3_GU_DN
                   ? (t < exl3::kPlanesGateUp ? w_gu[t] : w_dn[exl3::kDownSuh])
             : op == EXL3_PREP_GU ? w_gu[1 + t] : w_dn[t];
    }
    static int n_tables(int op) {
        return op == EXL3_GATE_UP ? exl3::kTablesGateUp
             : op == EXL3_GU_DN   ? exl3::kTablesGd
             : op == EXL3_PREP_GU ? exl3::kTablesPrep : exl3::kPlanesDown;
    }
    static const int* table_slots(int op) {
        return op == EXL3_GATE_UP ? exl3::kSlotGateUp
             : op == EXL3_GU_DN   ? exl3::kSlotGd
             : op == EXL3_PREP_GU ? exl3::kSlotPrep : exl3::kSlotDown;
    }
    /* Test mutation 3 (above): gate_up's dn_suh table taken from the other parity class (at four
     * classes, the class q ^ 1) -- in gate_up_down's too, which reads the same run in the same role. */
    static bool mut3(int op, int t) {
        return exl3::kTestMutation == 3 && (op == EXL3_GATE_UP || op == EXL3_GU_DN) &&
               t == exl3::kPlanesGateUp;
    }

    NormQuantFP8 nq_h{};
    GateQuantFP8 gq_sh{};
    LinearFP8    sh_gate_up{}, sh_down{};
    rad_op op_router = 0, op_topk = 0, op_scatter = 0;
    /* `op_scatter` is moe_scatter_inv and `op_gather` moe_gather_inv: the sort writes `w.inv` and
     * the gather reads it (Wire::inv). */
    bool   scatter_inv = false, gather_inv = false;
    /* `router_topk_scatter`, if it resolves: topk and scatter in one launch. The pair is declared
     * either way, since the fused form is bounded by the sort's single workgroup and a prefill
     * chunk exceeds it. */
    rad_op  op_topk_scatter = 0;
    int64_t topk_scatter_rows = 0;
    /* `router_gemm_topk_scatter`, if it resolves: the router GEMM fused in front, up to
     * `router_fused_rows` tokens, plus the shared expert's gate (`op_sgate`) when
     * `router_fused_gate`. Otherwise, or for larger passes, the separate ops run. */
    rad_op  op_router_fused = 0;
    int64_t router_fused_rows = 0;
    bool    router_fused_gate = false;
    /* `router_sigmoid_topk_scatter` (Config::router_fuse), up to `sigmoid_scatter_rows` tokens,
     * writing the shared gate's fill too when `sigmoid_scatter_gate`. */
    rad_op  op_sigmoid_scatter = 0;
    int64_t sigmoid_scatter_rows = 0;
    bool    sigmoid_scatter_gate = false;
    /* Set by rad_block_mtp_hc.h from the connection in front of the block: whether that read
     * writes the rotated E4M3 pair. Always false for EXL3 experts (Config::expert_rot), and
     * declare() refuses true rather than ignore a pair it would never read. */
    bool   hr_in = false;
    rad_op op_gu = 0, op_dn = 0, op_gather = 0;
    /* The decode-band form (kPrepSplit): prep_gu and gate_up(a_rot = 1) over xr, for passes of at
     * most prep_rows rows; 0 where the build issues the one gate_up only. */
    rad_op  op_pg = 0, op_gu_ar = 0;
    int64_t prep_rows = 0;
    rad_buf b_xr = 0;
    /* The decode chain as one op (exl3::kGd), for passes of at most gd_rows rows; 0 where it is
     * not declared or did not resolve, and the forms above are issued. */
    rad_op  op_gd = 0;
    int64_t gd_rows = 0;
    /* THE PROTECTED EXPERTS (declare): local experts a quantised layer keeps plain bf16 -- the few
     * whose output dwarfs the rest's (docs/MOE-W4.md). The scatter sorts them last (`protect`,
     * `prot_list`), the quantised GEMMs above serve the other `n_reg`, and a bf16 grouped GEMM
     * pair serves these over the trailing run of the offsets: gate_up into `b_pgu`, the gate into
     * `pff`, and down into the same `edn` rows the quantised down left zero. */
    std::vector<int64_t>    prot;
    const char*             prot_list = nullptr;
    int64_t                 n_reg = 0;
    /* THE NEXT-LAYER EXPERT PREFETCH (radiance 1.4.0, rad_expert_prefetch; the fp8 block's members of
     * the same names). `xp_next` is the next routed layer, which this one predicts for at its router
     * input -- the architecture links them once every layer is declared. `xp` is what this layer is to
     * the one before it: its router, and each of the model's experts' unit as the weights the core may
     * point at a copy, and its place in the routing counts. THE PARTS ARE THE PLANES IN EXACTLY ONE
     * TABLE: the core takes an expert only where each listed weight is in one weight table (tables are
     * per declared op and operand, core/runtime/xprefetch.cpp xp_bind), and at most kXpParts = 4 of
     * them; a plane two of this layer's ops read (gate_up's planes with the decode-band pair or
     * gate_up_down declared, down's suh -- h_rot) stays out and is read from its pool slot as before.
     * set in declare (xp_describe); a layer with no part is predicted for by nobody.
     * STILL ROOM TO IMPROVE AT TP1 (Dylan accepted tp1 as it stands, 2026-10-13): the shipped layout
     * (split + gate_up_down) lists no part, so the prefetch makes no copies; order 0 (PREP_SPLIT 0, GD 0)
     * lists all four and its prefetch works (tp1: ~100k copies, 47 % used). Measured at tp1 (single
     * passes): equal prefill and decode, and on the Gen4 x4 card TTFT C=8 2,717 ms for order 0 against
     * 3,350 shipped and 2,954 base (noisy). A possible automatic choice when radiance's prefetcher is
     * active, not made. */
    const MoeEXL3*          xp_next = nullptr;
    RadPrefetchNext         xp{};
    std::vector<rad_weight> xp_parts;
    std::vector<int32_t>    xp_count_at;
    /* THIS LAYER'S gate_up AND down ENCODINGS -- K, so the op dtype ("exl3k4"...), and the
     * codebook ("mul1"...) -- resolved by declare() from the layer's own experts: each projection
     * of each layer may carry its own, and every expert of one table must carry the same. */
    exl3::Enc               eenc[2]{};
    std::vector<rad_weight> w_pgu, w_pdn;
    rad_buf                 b_pgu = 0;
    ActFP8                  pff{};
    GateQuantFP8            gq_prot{};
    rad_op                  op_pgu = 0, op_pdn = 0;
    /* The gather folds in the shared arm when there is one, reading `sout` and `sgate` and writing
     * the block's delta straight into `h.x`; `w.moe` is then unused. */
    bool   gfold = false;
    /* The following write performs the gather up to this many tokens, per wire (HyperConn's
     * gather_rows / gather_rows6, set by the architecture after that write is declared; 0 where
     * not). Where gather_taken() agrees, the block leaves its routed rows unsummed. */
    int64_t gather_out_rows = 0;
    int64_t gather_out_rows6 = 0;
    /* The calibration tap's accumulators and ops; zero unless `Config::calib`. `calib_rows` and
     * `calib_done` are mutable because `pass` is const. The count is per block, so every block
     * reaches its target on the same pass. */
    rad_buf b_gram_gu = 0, b_gram_dn = 0;
    rad_op  op_gram_gu = 0, op_gram_dn = 0;
    /* The down tap's bf16 copy of `w.eff` and the cast that writes it (Config::calib says why). */
    rad_buf b_calib_eff = 0;
    rad_op  op_calib_cast = 0;
    /* The domain the tap sums in, which its files say (calib_write): always the model's own (0)
     * here -- see Config::calib. */
    uint32_t gram_domain = 0;
    /* Files are named after the weight they will pack, minus the expert index. That is the whole
     * protocol with the GPTQ quantiser: the recipe gives the directory (`calib=`), rad-convert the
     * weight's logical name, and libquant strips `.<e>.weight` to find the file. */
    const char* gram_gu_stem = nullptr;
    const char* gram_dn_stem = nullptr;
    mutable long long calib_rows = 0;
    mutable bool      calib_done = false;
    /* Per-expert down-projection Hessians (rad_arch.h: CalibExperts), beside the pooled one: a
     * layer-sized scratch of `ne` [n_ff_exp, n_ff_exp] squares, a `gram_accum` into each over that
     * expert's rows, summed on the host. */
    rad_buf b_gram_dne = 0;
    rad_op  op_gram_dne = 0;
    mutable CalibExperts calib_dne;
    mutable bool         calib_dne_failed = false;
    void calib_experts_pass(RadCtx* c_, int64_t slots) const;
    rad_op op_sgate = 0;
    rad_op op_ar = 0, op_add = 0;
    bool   ar_out = false;
    /* Which wires the consumer of this block's all-reduce takes. Set before declare; see
     * AttnGatedFP8::ar_out_take. */
    int    ar_out_take = kArOutNorm;

    /* THAT THE ROUTED EXPERTS ARE EXL3, read off the encoding of layer `l`'s first expert's
     * gate_up -- its name map declared here, which declare() then leaves out. Asked before any
     * layer is declared, as the fp8 block's probe is, so a container whose experts are something
     * else is refused before anything is sized. Unlike the fp8 block, NOTHING KNOWN about the
     * weight (a builder with no model behind it) is refused too: the ops' dtype is the experts'
     * trellis width, and there is no default width to guess. */
    static int probe(RadBuilder* b, Names& nm, int l, const Src& src, Config* cfg);

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l,
                 const Wire& wire, const Src& src, bool fold = false, bool add_out = true,
                 int ar_in = kArNone, bool ar_out_ = false);
    void step(RadCtx* c, const RadBatch* batch) const;
    /* staging off: the never-issued reader of this block's experts the architecture declares after the next
     * routed block (above, exl3::kPrefillStageForced; exl3_stage.h). */
    rad_op op_stage_off = 0;
    /* This run's decode-band forms: the build's (exl3::kPrepSplit / kGd), off where the file tier is in use
     * (declare says why). */
    bool   band_split = false, band_gd = false;
    int    declare_stage_off(RadBuilder* b);
    /* The block over the first `T` rows of its buffers, for passes that need fewer rows than the
     * batch (the MTP head's history pass reads its output only at drafting rows). */
    void step_rows(RadCtx* c, int64_t T) const;
    /* One pass of at most `c.rows` tokens. `own_ar` false: the caller issues the pass's
     * all-reduce itself (issue_ar), as GdnFP8::step describes. */
    void pass(RadCtx* c, int64_t T, int64_t r0, int64_t rows, bool own_ar = true) const;
    /* The heat engine's handoff, after a step's last pass (step() says what it reports). */
    void route_report(RadCtx* c) const;
    bool issues_ar(int64_t T, int64_t rows) const {
        return op_ar && !ar_taken(g, ar_out_take != kArOutNorm ? T : rows, ar_out, ar_out_take);
    }
    /* `beside` false: the collective runs on its own (a pass that owns it), so it takes the
     * size-chosen block count and not the overlap cap, which only pays beside other work. */
    void issue_ar(RadCtx* c, int64_t r0, int64_t rows, bool beside = true) const {
        RAD_ISSUE_N(c, beside && op_ar_ov ? op_ar_ov : op_ar, rows * g.n_embd,
                    brow_slice(w.h.x, r0, rows, g.n_embd), RAD_NONE);
    }
    /* Set before declare: the `overlap` issue_ar's collective is declared at, 0 for none. */
    int    ar_overlap = 0;
    rad_op op_ar_ov = 0;
    void shared_up(RadCtx* c, int64_t T, int64_t r0, int64_t rows, bool gate_done = false) const;
};

/* WHETHER A CHECKPOINT SHIPS ONE LAYER'S EXPERTS STACKED OR ONE BY ONE, as the Src it fills.
 * Recent releases stack them -- `<prefix>.gate_up_proj` and `<prefix>.down_proj` -- and a
 * quantised export (AutoRound's) writes each expert on its own, `<prefix>.<e>.{gate,up,down}_proj`.
 * Each form is asked of the checkpoint through a probe name mapped to its first expert, which the
 * checkpoint answers or does not. Stacked is the answer when neither is found: a container is
 * read by declared name, the same under both maps, so it asks nothing here. `tag` keeps one
 * caller's probe name apart from another's for the same layer. */
inline int moe_expert_sources(RadBuilder* b, Names& nm, const char* tag, const char* prefix,
                              MoeEXL3::Src* src) {
    const char* gu = nm.ckpt("%s.gate_up_proj", prefix);
    const char* probe = nm.f("probe.%s.%s.0", tag, prefix);
    RadEncoding e{};
    RAD_ARCH_TRY(map_concat(b, probe, { nm.ckpt("%s.0.gate_proj.weight", prefix),
                                        nm.ckpt("%s.0.up_proj.weight", prefix) }));
    if (weight_enc(b, probe, &e)) {
        src->experts = nm.ckpt("%s", prefix);
        src->experts_gate_up = src->experts_down = nullptr;
    } else {
        src->experts = nullptr;
        src->experts_gate_up = gu;
        src->experts_down    = nm.ckpt("%s.down_proj", prefix);
    }
    return RAD_OK;
}

/* The gate_up and down name maps of one expert: a dim-0 slice of the stacked tensors, or the
 * checkpoint's per-expert gate and up joined along rows. `gate_up` false maps down alone. */
inline int moe_map_expert(RadBuilder* b, Names& nm, const MoeEXL3::Src& src, const char* guw,
                          const char* dnw, long long ge, bool gate_up) {
    if (src.experts_gate_up) {
        /* The slice index is global: the checkpoint holds every expert. */
        if (gate_up) RAD_ARCH_TRY(map_slice(b, guw, src.experts_gate_up, (int)ge));
        return map_slice(b, dnw, src.experts_down, (int)ge);
    }
    const char* gate = nm.ckpt("%s.%lld.gate_proj.weight", src.experts, ge);
    const char* up   = nm.ckpt("%s.%lld.up_proj.weight",   src.experts, ge);
    if (gate_up) RAD_ARCH_TRY(map_concat(b, guw, { gate, up }));
    return map_copy(b, dnw, nm.ckpt("%s.%lld.down_proj.weight", src.experts, ge));
}

inline int MoeEXL3::probe(RadBuilder* b, Names& nm, int l, const Src& src, Config* cfg) {
    if (!cfg || (!src.experts && !src.experts_gate_up)) return RAD_E_INVAL;
    /* The caller names the expert in model numbering (the first this rank owns under expert
     * parallelism); declare() skips the ones mapped here.
     *
     * A bf16 expert does not settle the format: a quantised layer keeps up to kMaxProtected
     * experts bf16, so the probe reads on to the first expert in another format, and only a longer
     * run of bf16 ones means the model is bf16. */
    const long long g0 = (long long)cfg->probed_expert;
    const long long all = cfg->n_expert_all > 0 ? (long long)cfg->n_expert_all
                                                : (long long)cfg->n_expert;
    const long long gn = all > g0 ? all : g0 + 1;
    cfg->probed_layer = l;
    const char* guw = nullptr;
    RadEncoding e{};
    bool known = false;
    exl3::Kind k = exl3::kUnserved;
    for (long long ge = g0; ge < gn && ge <= g0 + kMaxProtected; ++ge) {
        guw = nm.f("blk.%d.ffn_gate_up_exps.%lld.weight", l, ge);
        if (src.experts_gate_up) RAD_ARCH_TRY(map_slice(b, guw, src.experts_gate_up, (int)ge));
        else RAD_ARCH_TRY(map_concat(b, guw, { nm.ckpt("%s.%lld.gate_proj.weight", src.experts, ge),
                                               nm.ckpt("%s.%lld.up_proj.weight", src.experts, ge) }));
        cfg->probed_count = ge - g0 + 1;
        e = RadEncoding{};
        known = weight_enc(b, guw, &e);
        k = known ? exl3::kind_of(e) : exl3::kUnserved;
        if (k != exl3::kPlainBf16) break;
    }
    cfg->expert_rot = 0;
    if (k == exl3::kExl3) return RAD_OK;
    if (!known) {
        fprintf(stderr, "radiance: nothing is known about '%s' -- a builder with no model behind it, "
                        "or a name the model does not hold -- and the EXL3 experts' ops are declared at "
                        "their trellis width, which only the model can say\n", guw ? guw : "?");
        return RAD_E_NOTFOUND;
    }
    char en[256];
    rad_enc_format(&e, en, sizeof en);
    if (k == exl3::kPlainBf16)
        fprintf(stderr, "radiance: layer %d's routed experts from %lld on are plain bf16, more of them "
                        "than any protected set: that is a bf16 model, which the in-tree "
                        "qwen4exp_fp8 serves (quantisation \"\"), not this EXL3 plugin: serve it from a "
                        "RADIANCE_HOME without the qwen4exp_exl3 plugin (docs/TROUBLESHOOTING.md)\n", l, g0);
    else
        fprintf(stderr, "radiance: '%s' is %s, and this plugin serves the routed experts as EXL3 -- "
                        "scheme `%s` as the exl3 quantiser writes it, K 2..8 -- beside plain bf16 "
                        "protected ones. This container is not an EXL3 one: serve it from a "
                        "RADIANCE_HOME without the qwen4exp_exl3 plugin, or convert it with the exl3 "
                        "quantiser (recipes/qwen4exp-exl3.recipe; docs/TROUBLESHOOTING.md).\n", guw, en,
                        EXL3_SCHEME);
    return RAD_E_DTYPE;
}

inline int MoeEXL3::declare(RadBuilder* b, Names& nm, const Geom& geom, int l,
                           const Wire& wire, const Src& src, bool fold, bool add_out,
                           int ar_in, bool ar_out_) {
    g = geom; c = src.cfg; w = wire; layer = l;
    ar_out = ar_out_;
    /* 0 means not expert-parallel: this rank owns every expert. */
    if (c.n_expert_all <= 0) c.n_expert_all = c.n_expert;
    if (c.n_expert <= 0 || c.top_k <= 0 || c.top_k > c.n_expert_all || c.n_ff_exp <= 0)
        return RAD_E_INVAL;
    /* The expert slice must lie inside the model, or tokens routed past it are silently dropped. */
    if (c.expert_base < 0 || c.expert_base + c.n_expert > c.n_expert_all) return RAD_E_INVAL;
    /* THE SPLIT AND FUSED DECODE FORMS AND THE FILE TIER DO NOT MIX, SO THE FILE TIER TAKES THE ONE gate_up.
     * --weights-disk-tier's file stager reads a layer's file-tier experts at the layer's FIRST DECLARED expert op,
     * on every pass (core/place/stager.cpp FileStager), and with the decode-band forms declared that op -- the one
     * gate_up, declared first so the prefill stager keys on it -- is not issued in the decode band, where prep_gu
     * and gate_up(a_rot = 1), or gate_up_down, are: a decode pass would issue those layers' GEMMs over experts
     * that were never read. No declaration order makes two per-band forms over the same experts both "first".
     * So where the command line asks for the file tier (process_has_flag) this run declares neither form: every
     * pass issues the one gate_up and down -- a machine whose experts fit in neither the cards nor the host pool
     * is served (decode a few tenths of a ms a step slower than the split form, the same bits), not refused. */
    {
        const bool file_tier = (exl3::kPrepSplit || exl3::kGd) && exl3::process_has_flag("--weights-disk-tier");
        band_split = exl3::kPrepSplit && !file_tier;
        band_gd    = exl3::kGd && !file_tier;
        static bool told = false;
        if (file_tier && !told) {
            told = true;
            fprintf(stderr, "I qwen4exp_exl3: --weights-disk-tier: the decode band issues the one gate_up and down "
                            "(the %s form keys the file stager on ops a decode pass does not issue); decode a few "
                            "tenths of a ms a step slower, the same bits\n",
                    exl3::kGd ? "gate_up_down" : "prep_gu + gate_up(a_rot = 1)");
        }
    }
    if (c.rows <= 0) c.rows = g.max_tok;
    /* A rank whose slice of the shared arm is empty (see Config) declares no arm, and the
     * all-reduce carries the others' share. */
    const bool shared = c.n_ff_shared > 0 && src.shared;
    if (shared && c.shared_ungated && src.shared_gate) return RAD_E_INVAL;
    /* THE ROUTED SwiGLU IS gate_up's OWN EPILOGUE (silu(gate) * up, rounded to f16), with no clamp:
     * a model that clamps it (Config::swiglu_limit, upstream's swiglu_clamped) is refused rather
     * than served unclamped. The shared arm's gated quantiser is the installed one and clamps. */
    if (c.swiglu_limit > 0) {
        fprintf(stderr, "radiance: layer %d clamps its SwiGLU (swiglu_limit %g); the EXL3 gate_up "
                        "computes silu(gate) * up in its epilogue and has no clamp\n", l, c.swiglu_limit);
        return RAD_E_UNSUPPORTED;
    }
    /* Whether `w.sgate` is excluded from buffer sharing; set with the shared arm and required by
     * the fused router's gate write (see there). */
    bool sgate_concurrent = false;

    /* `src.norm == nullptr`: the architecture prepares the input (rad_block_hc.h's `hc_read`), so
     * `wire.h` is already normed and quantised and no norm weight is declared. `ar_in` is then
     * refused, since the fused all-reduce lives inside the norm. See rad_block_attn_gated_fp8.h. */
    ext_in = (src.norm == nullptr);
    if (ext_in) {
        if (!w.h.x || (!g.w_bf16 && (!w.h.q || !w.h.s)) || ar_in != kArNone) return RAD_E_INVAL;
    } else {
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_norm.weight", l), src.norm));
        w_norm = decl_w(b, nm.f("blk.%d.ffn_norm.weight", l), RAD_F32, {g.n_embd},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    }

    /* ---- The experts' name maps (one per logical weight, gate_up and down; scale views have
     * none; the probed experts' gate_up is already mapped), and from their encodings the experts
     * this layer protects. One table takes one encoding, but a quantised layer may keep a few
     * whole experts bf16: those whose output dwarfs the rest, where the same relative rounding
     * error is large in absolute terms (docs/MOE-W4.md). They are sorted last and served by their
     * own bf16 GEMM pair, so they must be known before the sort is declared. */
    const int64_t ne = c.n_expert;
    for (int64_t e = 0; e < ne; ++e) {
        const long long ge = (long long)(c.expert_base + e);   /* the checkpoint's numbering */
        const bool probed = c.probed_layer == l && ge >= (long long)c.probed_expert &&
                            ge < (long long)(c.probed_expert + c.probed_count);
        RAD_ARCH_TRY(moe_map_expert(b, nm, src, nm.f("blk.%d.ffn_gate_up_exps.%lld.weight", l, ge),
                                    nm.f("blk.%d.ffn_down_exps.%lld.weight", l, ge), ge, !probed));
    }
    prot.clear();
    eenc[0] = eenc[1] = exl3::Enc{};
    {
        /* EVERY EXPERT IS EXL3, OR A WHOLE EXPERT KEPT PLAIN bf16, and every EXL3 expert of one
         * projection carries the SAME encoding: one table takes one geometry and one decode. The
         * layer's first EXL3 expert decides each projection's encoding, and so its op's dtype and
         * codebook -- a layer, and either projection of it, may carry its own trellis width, as the
         * fp8 block's codebook forms may differ by layer and projection -- and every other must
         * equal it byte for byte. gate_up is the STACKED form (its up half's `suh.1`), down is not.
         * An expert the model says nothing about is taken to be the layer's, as the fp8 block
         * takes it: a container that lacks it is refused by name when it loads. */
        RadEncoding first[2]{};
        bool have[2] = { false, false };
        for (int64_t e = 0; e < ne; ++e) {
            const long long ge = (long long)(c.expert_base + e);
            const char* names[2] = { nm.f("blk.%d.ffn_gate_up_exps.%lld.weight", l, ge),
                                     nm.f("blk.%d.ffn_down_exps.%lld.weight", l, ge) };
            RadEncoding enc[2]{};
            bool known[2];
            exl3::Kind k[2];
            exl3::Enc x[2];
            for (int i = 0; i < 2; ++i) {
                known[i] = weight_enc(b, names[i], &enc[i]);
                k[i] = known[i] ? exl3::kind_of(enc[i], &x[i]) : exl3::kExl3;
            }
            if (k[0] == exl3::kPlainBf16 && k[1] == exl3::kPlainBf16) { prot.push_back(e); continue; }
            for (int i = 0; i < 2; ++i) {
                if (!known[i]) continue;
                if (k[i] == exl3::kExl3 && !have[i] && x[i].stacked == (i == 0)) {
                    first[i] = enc[i];
                    eenc[i]  = x[i];
                    have[i]  = true;
                    continue;
                }
                if (have[i] && rad_enc_equal(&enc[i], &first[i])) continue;
                char en[256], want[256];
                rad_enc_format(&enc[i], en, sizeof en);
                if (have[i]) rad_enc_format(&first[i], want, sizeof want);
                else snprintf(want, sizeof want, "EXL3 (scheme `%s`, %s)", EXL3_SCHEME,
                              i == 0 ? "stacked: suh.1 for the up half" : "not stacked");
                fprintf(stderr, "radiance: '%s' is %s, and this layer's experts are served as %s -- "
                                "one table of experts takes one encoding, and only a whole expert "
                                "kept plain bf16 is served beside it.\n", names[i], en, want);
                return RAD_E_DTYPE;
            }
        }
        if (!have[0] || !have[1]) {
            fprintf(stderr, "radiance: layer %d has no routed expert whose %s the model knows as EXL3, "
                            "so its %s op has no dtype to be declared at\n", l,
                    have[0] ? "down" : "gate_up", have[0] ? "down" : "gate_up");
            return RAD_E_DTYPE;
        }
    }
    n_reg = ne - (int64_t)prot.size();
    if (!prot.empty()) {
        if ((int64_t)prot.size() > kMaxProtected) {
            fprintf(stderr, "radiance: layer %d keeps %zu experts plain bf16 on this rank; the sort "
                            "places at most %lld last\n", l, prot.size(), (long long)kMaxProtected);
            return RAD_E_INVAL;
        }
        /* The quantised GEMM takes two equal tables by parity; with four classes
         * (Config::ff_lo4), four tables with the first n_reg % 4 one longer, so any count works. */
        if (!(c.ff_hi[0] > 0 && c.ff_hi4[0] > 0) && n_reg % 2) {
            fprintf(stderr, "radiance: layer %d keeps %zu experts plain bf16 on this rank, which "
                            "leaves %lld quantised; they come in pairs on a rank so the rest split "
                            "into two equal tables by parity\n", l, prot.size(), (long long)n_reg);
            return RAD_E_INVAL;
        }
        /* The tap's per-expert files are named by expert, but this sort numbers by place. */
        if (c.calib) {
            fprintf(stderr, "radiance: calibrate the bf16 container: layer %d of this one keeps "
                            "experts plain bf16 and sorts them out of order\n", l);
            return RAD_E_INVAL;
        }
        std::string lst;
        for (int64_t e : prot) lst += (lst.empty() ? "" : ",") + std::to_string(e);
        prot_list = nm.ckpt("%s", lst.c_str());
    }

    /* The router is replicated (see Config::n_expert_all): every rank computes the same routing
     * from the full residual stream, so `sorted_tok` and `expert_offset` agree across ranks
     * without a collective. The sigmoid router (GlmRouter) declares its own projection. */
    if (!c.sigmoid_router) {
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_gate_inp.weight", l), src.router));
    w_router = decl_w(b, nm.f("blk.%d.ffn_gate_inp.weight", l), RAD_BF16,
                      {c.n_expert_all, g.n_embd},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    }

    if (!ext_in)
        RAD_ARCH_TRY(nq_h.declare(b, g, w_norm, fold ? w.h.x : w.x, w.h, g.n_embd,
                                  fold ? w.x : 0, 0, ar_in));

    if (c.sigmoid_router) {
        if (!src.router_bias) return RAD_E_INVAL;
        GlmRouter::Config rc;
        rc.experts = c.n_expert_all;
        rc.top_k = c.top_k;
        rc.rows = c.rows;
        rc.norm = c.norm_topk;
        rc.route_scale = c.route_scale;
        const GlmRouter::Wire wire{ w.h.x, w.logits, w.ids, w.ew };
        const GlmRouter::Src rs{ src.router, src.router_bias };
        RAD_ARCH_TRY(glm_router.declare(b, nm, g, l, nm.f("blk.%d.glm_router", l), rc, wire, rs));
        w_router = glm_router.projection;
        op_router = glm_router.op_project;
        op_topk = glm_router.op_select;
    } else {
    op_router = rw(b, RAD_OP(b, "gemm_nt",
                      RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("N", c.n_expert_all),
                                 RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(w_router)),
                  {w.h.x}, {w.logits});
    op_topk = rw(b, RAD_OP(b, "router_topk",
                    RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n_expert", c.n_expert_all),
                               RAD_INT("top_k", c.top_k), RAD_INT("norm", c.norm_topk),
                               RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.logits}, {w.ids, w.ew});
    }
    /* Protected experts are sorted last (`protect`); layers without them declare no protect
     * list, so it is the last parameter and cut off. */
    const int n_prot = prot_list ? 1 : 0;
    {
        const RadParam ps[] = { RAD_RANGE("M", 1, c.rows), RAD_INT("n_expert", c.n_expert),
                                RAD_INT("top_k", c.top_k), RAD_INT("expert_base", c.expert_base),
                                RAD_STR("protect", prot_list) };
        /* With `w.inv` the sort also writes the gather's slot table. It writes everything
         * moe_scatter does, so a moe_gather_inv that does not resolve leaves moe_gather to serve. */
        if (w.inv && rad_op_declared(b, "moe_scatter_inv") && rad_op_declared(b, "moe_gather_inv"))
            op_scatter = rw(b, rad_decl_op(b, "moe_scatter_inv", ps, 4 + n_prot, RAD_NOWEIGHTS),
                            {w.ids}, {w.sorted, w.eoff, w.ecnt, w.inv});
        scatter_inv = op_scatter != 0;
        if (!op_scatter)
            op_scatter = rw(b, rad_decl_op(b, "moe_scatter", ps, 4 + n_prot, RAD_NOWEIGHTS),
                            {w.ids}, {w.sorted, w.eoff, w.ecnt});
    }

    /* topk and scatter in one launch. At decode each is a single 256-thread workgroup ranking
     * about a kilobyte, so each is a dispatch with almost no work, and the sort reads only what
     * the selection writes.
     *
     * Bounded by the sort's single-workgroup form: `moe_scatter` is one launch at
     * M*top_k <= 1024 and three above (for a stable sort without atomics), and only the first is
     * fusible. That covers decode (eight sequences at depth 3 and top_k 10 is 320); prefill
     * chunks exceed it. The pair is declared either way, so the graph and buffer plan do not
     * depend on it. The sigmoid router selects in its own op and takes no fused pair. */
    topk_scatter_rows = (int64_t)c.rows * c.top_k <= kMoeSortRows ? c.rows
                                                                   : kMoeSortRows / c.top_k;
    if (c.sigmoid_router || scatter_inv) topk_scatter_rows = 0;
    /* Asked for only where a loaded library declares it: a CPU-only engine would refuse the
     * declare, and the two separate ops serve without it. */
    if (topk_scatter_rows > 0 && rad_op_declared(b, "router_topk_scatter")) {
        const RadParam ps[] = { RAD_RANGE("M", 1, topk_scatter_rows),
                                RAD_INT("n_expert", c.n_expert_all), RAD_INT("n_local", c.n_expert),
                                RAD_INT("top_k", c.top_k), RAD_INT("norm", c.norm_topk),
                                RAD_STR("dtype", g.dtype), RAD_INT("expert_base", c.expert_base),
                                RAD_STR("protect", prot_list) };
        op_topk_scatter = rw(b, rad_decl_op(b, "router_topk_scatter", ps, 7 + n_prot,
                                            RAD_NOWEIGHTS),
                             {w.logits}, {w.ids, w.ew, w.sorted, w.eoff, w.ecnt});
    }

    /* Exactly one of a prefix or the two stacked tensors. */
    const bool stacked = src.experts_gate_up != nullptr;
    if (stacked != (src.experts_down != nullptr) || (stacked == (src.experts != nullptr))) {
        fprintf(stderr, "radiance: a routed feed-forward takes EITHER an `experts` prefix or "
                        "both stacked tensors; got prefix %s, gate_up %s, down %s\n",
                src.experts ? "yes" : "no", src.experts_gate_up ? "yes" : "no",
                src.experts_down ? "yes" : "no");
        return RAD_E_INVAL;
    }

    /* ---- the expert weights, in FOUR RUNS: gate_up and down, each for the even experts and then
     * the odd ones. See the header: a weight table is a run of consecutive handles, and both ops
     * take each projection's experts as two tables by parity -- entry e / 2 of class e % 2 -- so
     * these loops must not be merged. */
    if (ne % 2) {
        fprintf(stderr, "radiance: %lld routed experts on this rank; the grouped GEMM takes them as "
                        "two equal tables by parity, so the count must be even\n", (long long)ne);
        return RAD_E_INVAL;
    }
    /* A ROTATED E4M3 COPY OF THE INPUT IS NOTHING THESE EXPERTS READ, so a connection told to write
     * one (rad_block_mtp_hc.h, HyperConn::Config::rotate) was told wrong, and is refused rather
     * than left writing a pair into buffers this block never declared. */
    if (hr_in || c.expert_rot) {
        fprintf(stderr, "radiance: layer %d: the input was to arrive with a rotated E4M3 copy "
                        "(expert_rot), which EXL3 experts do not read -- their sign vectors and "
                        "Hadamards are applied inside the kernels, to the bf16 input\n", l);
        return RAD_E_INVAL;
    }
    /* The quantised experts by their PLACE in the sort -- the protected ones sorted last are not
     * among them -- which is also the expert slot each is declared in, so the routing counts the
     * sort writes in place order credit the right unit. */
    std::vector<int64_t> reg;
    reg.reserve((size_t)n_reg);
    for (int64_t e = 0; e < ne; ++e)
        if (std::find(prot.begin(), prot.end(), e) == prot.end()) reg.push_back(e);
    for (auto& v : w_gu) v.assign((size_t)ne, 0);
    for (auto& v : w_dn) v.assign((size_t)ne, 0);
    /* Each class's width on this rank: the Config's slice when the width is split unevenly, and
     * `n_ff_exp` for every class otherwise. The buffer between the two ops is the widest one's. FOUR
     * CLASSES where the Config offers them (Config::ff_lo4) -- the `_mod4` ops -- the parity classes
     * otherwise. */
    const bool split = c.ff_hi[0] > 0 || c.ff_hi4[0] > 0;
    ncls = c.ff_hi4[0] > 0 ? 4 : 2;
    if (ncls == 4 && n_reg < 4) {
        fprintf(stderr, "radiance: layer %d serves %lld quantised experts on this rank; four "
                        "classes need at least one each\n", l, (long long)n_reg);
        return RAD_E_INVAL;
    }
    int64_t qlo[4] = { 0, 0, 0, 0 }, qhi[4] = { 0, 0, 0, 0 };
    int64_t wq[4] = { c.n_ff_exp, c.n_ff_exp, c.n_ff_exp, c.n_ff_exp };
    if (split)
        for (int q = 0; q < ncls; ++q) {
            qlo[q] = ncls == 4 ? c.ff_lo4[q] : c.ff_lo[q];
            qhi[q] = ncls == 4 ? c.ff_hi4[q] : c.ff_hi[q];
            wq[q]  = qhi[q] - qlo[q];
        }
    if (split) {
        int64_t widest = 0;
        bool    bad = false;
        for (int q = 0; q < ncls; ++q) {
            bad = bad || wq[q] <= 0;
            widest = wq[q] > widest ? wq[q] : widest;
        }
        if (bad || c.n_ff_exp != widest) {
            fprintf(stderr, "radiance: the routed experts' %d class slices, the first [%lld, %lld) "
                            "and [%lld, %lld), do not describe a width of %lld\n", ncls,
                    (long long)qlo[0], (long long)qhi[0], (long long)qlo[1], (long long)qhi[1],
                    (long long)c.n_ff_exp);
            return RAD_E_INVAL;
        }
    }
    /* EVERY CUT ON A WHOLE 128-COLUMN GROUP (exl3::kCut): the Hadamard is 128 wide on both sides of
     * both projections, and 128 is eight of the trellis's 16x16 tiles, so a cut there keeps every
     * tile and every rotation whole -- gate_up's rows and down's columns are sliced by it. */
    for (int q = 0; q < ncls; ++q)
        if (wq[q] % exl3::kCut || g.n_embd % exl3::kCut || (split && qlo[q] % exl3::kCut)) {
            fprintf(stderr, "radiance: an expert is %lldx%lld per rank, which is not a whole number "
                            "of %lld-wide Hadamard groups each way\n",
                    (long long)wq[q], (long long)g.n_embd, (long long)exl3::kCut);
            return RAD_E_INVAL;
        }
    /* UNDER EXPERT PARALLELISM AN EXPERT IS WHOLE, so its weights are not sharded at all: this
     * rank holds all of the columns of the experts it owns and none of the others. Declaring them
     * RAD_SHARD_ROW would have the loader slice each one again -- it refuses by name, because the
     * declared row parts sum to the full width while the rank's share is half of it. */
    const bool ep = c.n_expert != c.n_expert_all;
    if (ep && split) {
        fprintf(stderr, "radiance: a rank either owns whole experts or a slice of every one; this "
                        "configuration asks for both\n");
        return RAD_E_INVAL;
    }
    const int  sh_row = ep ? RAD_SHARD_NONE : RAD_SHARD_ROW;
    const int  sh_col = ep ? RAD_SHARD_NONE : RAD_SHARD_COL;
    const int  n_parts = ep ? 0 : 2;
    /* ---- the runs: ONE VIEW A PLANE (D15), each a run of its own -- plane by plane, the even
     * experts then the odd ones (at four classes, class 0's then 1's, 2's and 3's) -- so every
     * (plane, class) is one contiguous table.
     *
     * A view of one plane is declared at that plane's own dtype and extents, this rank's share of
     * them (rad_builder.h, RadWeightDecl::source), and it is cut along the dimension its block
     * follows and no other: the codes everywhere; `svh`, one value a ROW, with gate_up's rows (and
     * its gate and up parts, like the codes, so a rank holds [its gate rows; its up rows] and the
     * up half's rows are the ones `suh.1` scales); `suh` and `suh.1`, one value a COLUMN, with
     * down's columns. A vector that the split does not cross -- gate_up's `suh` pair, down's `svh`
     * -- is whole on every rank. Every view of an expert's projection takes that projection's
     * movement slot, as the fp8 block's codes and scale views do. */
    const auto declare_run = [&](int proj, int p, int q) -> int {
        const char* role = proj == 0 ? exl3::kRoleGateUp[p] : exl3::kRoleDown[p];
        const char* stem = proj == 0 ? "ffn_gate_up_exps" : "ffn_down_exps";
        /* the plane the view selects: its role, except under test mutation 2 */
        const char* plane = exl3::kTestMutation == 2 && proj == 0 && (p == 1 || p == 2)
                                ? exl3::kRoleGateUp[3 - p] : role;
        const bool  codes = p == 0;
        const bool  per_row = std::strcmp(role, "svh") == 0;      /* [1x0] */
        const bool  per_col = !codes && !per_row;                  /* [0x1]: suh, suh.1 */
        const int64_t w = wq[q];
        int64_t rows, cols;
        bool cut;
        if (proj == 0) {       /* gate_up [2w, K], cut by rows */
            rows = per_col ? 1 : 2 * w;
            cols = per_row ? 1 : g.n_embd;
            cut  = !per_col;
        } else {               /* down [N, w], cut by columns */
            rows = per_col ? 1 : g.n_embd;
            cols = per_row ? 1 : w;
            cut  = !per_row;
        }
        const int shard = !cut ? RAD_SHARD_NONE : proj == 0 ? sh_row : sh_col;
        const int64_t parts[2] = { w, w };
        std::vector<rad_weight>& run = proj == 0 ? w_gu[p] : w_dn[p];
        for (int64_t s = q; s < n_reg; s += ncls) {
            const long long ge = (long long)(c.expert_base + reg[(size_t)s]);
            const char* weight = nm.f("blk.%d.%s.%lld.weight", l, stem, ge);
            rad_weight& h = run[(size_t)s];
            h = decl_view(b, codes ? weight : nm.f("blk.%d.%s.%lld.%s", l, stem, ge, role),
                          codes ? nullptr : weight, plane,
                          codes ? exl3_codes_dtype(eenc[proj].bits) : (uint32_t)RAD_F16,
                          {rows, cols}, RAD_ACCESS_CONDITIONAL, shard,
                          grp_expert_slot(l, (int)s, proj),
                          proj == 0 && shard != RAD_SHARD_NONE ? parts : nullptr,
                          proj == 0 && shard != RAD_SHARD_NONE ? n_parts : 0);
            if (!h) return RAD_E_INVAL;
            if (split && cut) RAD_ARCH_TRY(rad_weight_shard_span(b, h, qlo[q], qhi[q]));
        }
        return RAD_OK;
    };
    for (int p = 0; p < exl3::kPlanesGateUp; ++p)
        for (int q = 0; q < ncls; ++q) RAD_ARCH_TRY(declare_run(0, p, q));
    for (int p = 0; p < exl3::kPlanesDown; ++p)
        for (int q = 0; q < ncls; ++q) RAD_ARCH_TRY(declare_run(1, p, q));

    /* ---- the two ops. `M` is the TOKEN count of a pass, not the sorted row count: the kernels
     * read the row count off their output and the parameter is what the scratch hook and the
     * operand descriptions size from.
     *
     * THE TWO READ DIFFERENT ROWS, and here the op says which rather than a parameter: gate_up
     * reads the block's normed INPUT, one row per TOKEN, so a sorted row gathers
     * `a[sorted_tok[i] / top_k]`; down reads gate_up's output, which is already one row per
     * (token, slot) in the scatter's own order, so a sorted row is row i. The fp8 block's
     * `a_order` exists because one op serves both; these are two ops, and each schema fixes its
     * own (docs/OPS.md on `a_order` says what a gathering down projection does: every slot of a
     * token reads the first slot's intermediate, and the model answers fluently and wrongly).
     *
     * A table op's weights in its table operands' order (exl3_abi.h "THE DECLARATION"): the even
     * experts' tables, then the odd experts' -- gate_up's ending with dn_suh -- or at four classes
     * class 0's, 1's, 2's and 3's (the first n_reg % 4 one expert longer: the division's remainder).
     * N and K are the WEIGHT's, so gate_up's N is gate and up together. N_odd and K_odd are always
     * given, at one rank too, where they equal N and K: both tables are always issued (P2 notes,
     * finding 5); at four classes N_q and K_q, the `_mod4` ops' names. */
    const auto tables = [&](int op) {
        const int n_tab = n_tables(op);
        std::vector<rad_weight> tab;
        tab.reserve((size_t)(n_reg * n_tab));
        for (int q = 0; q < ncls; ++q) {
            const int sq = exl3::kTestMutation == 1 ? q ^ 1 : q;
            for (int t = 0; t < n_tab; ++t)
                for (int64_t s = sq; s < n_reg; s += ncls)
                    tab.push_back(table_run(op, t)[(size_t)(mut3(op, t) ? s ^ 1 : s)]);
        }
        return tab;
    };
    /* The classes' shape parameters after N and K: N_odd / K_odd, or N_1 .. K_3 (`rows` / `cols` the
     * class's [N, K]); prep_gu's K alone. */
    std::vector<RadParam> cls_p;
    const auto class_params = [&](const int64_t* rows, const int64_t* cols) {
        static const char* const nq[4] = { "N_odd", "N_1", "N_2", "N_3" };
        static const char* const kq[4] = { "K_odd", "K_1", "K_2", "K_3" };
        cls_p.clear();
        for (int q = 1; q < ncls; ++q) {
            if (rows) cls_p.push_back(RAD_INT(nq[ncls == 4 ? q : 0], rows[q]));
            cls_p.push_back(RAD_INT(kq[ncls == 4 ? q : 0], cols[q]));
        }
    };
    const int64_t kq_embd[4] = { g.n_embd, g.n_embd, g.n_embd, g.n_embd };
    int64_t nq_gu[4], kq_dn[4], nq_dn[4];
    for (int q = 0; q < 4; ++q) {
        nq_gu[q] = 2 * wq[q];
        kq_dn[q] = wq[q];
        nq_dn[q] = g.n_embd;
    }
    /* gate_up WRITES silu(gate) * up, ITS WIDER CLASS'S WIDTH: a row of the narrower class fills
     * its leading columns and zeros the rest, which its down projection does not read. */
    {
        const std::vector<rad_weight> tab = tables(EXL3_GATE_UP);
        const auto decl_gu = [&](int64_t hi, int a_rot, rad_buf in) {
            class_params(nq_gu, kq_embd);
            std::vector<RadParam> gp = { RAD_RANGE("M", 1, hi), RAD_INT("N", 2 * wq[0]),
                                         RAD_INT("K", g.n_embd), RAD_INT("n_expert", n_reg),
                                         RAD_INT("top_k", c.top_k) };
            gp.insert(gp.end(), cls_p.begin(), cls_p.end());
            gp.push_back(RAD_STR("dtype", eenc[0].dtype));
            gp.push_back(RAD_STR("codebook", eenc[0].cb));
            gp.push_back(RAD_INT("h_rot", exl3::kHRot));
            gp.push_back(RAD_INT("a_rot", a_rot));
            return rw(b, rad_decl_op(b, ncls == 4 ? exl3::kOpGateUp4 : exl3::kOpGateUp, gp.data(), (int)gp.size(),
                                     tab.data(), (int)tab.size()),
                      {in, w.sorted, w.eoff}, {w.eff});
        };
        /* THE ONE gate_up (a_rot = 0, every M) IS DECLARED FIRST, the decode-band pair after it. The
         * engine keys a routed layer's staging on the FIRST DECLARED op that reads its experts
         * (core/place/stager.cpp:23-31, first_use_op): the prefill stager copies the next layer's
         * pooled experts at that op and orders the layer's ops behind its own copies there, and the
         * file stager reads a layer's file-tier experts there. Declared first, the decode-band pair
         * would be that op -- never issued in a prefill pass, so the stager staged only the pass's
         * first layer and never ordered behind it (measured: P6's order-1 prefill, 2.6 GB staged
         * against ~200 GB). The file stager keys every pass on the same op, decode included, so
         * the split form must not be served with --weights-disk-tier: no declaration order makes
         * two per-band forms over the same experts both "first" (P3 notes). xr's life in the
         * declared order is still prep_gu -> gate_up(a_rot = 1), nothing between. xr: one f16 row a
         * sorted row, the gate half's rotated input then the up half's (exl3_abi.h "XR"); M is the
         * pass's TOKENS. */
        const auto declare_decode_band = [&]() -> int {
            if (!band_split) return RAD_OK;
            prep_rows = std::min<int64_t>(exl3::kPrepRows, c.rows);
            b_xr = decl_b(b, nm.f("blk.%d.moe_exl3_xr", l), RAD_F16,
                          {prep_rows * c.top_k, 2 * g.n_embd});
            const std::vector<rad_weight> pt = tables(EXL3_PREP_GU);
            class_params(nullptr, kq_embd);
            std::vector<RadParam> pp = { RAD_RANGE("M", 1, prep_rows), RAD_INT("K", g.n_embd),
                                         RAD_INT("n_expert", n_reg), RAD_INT("top_k", c.top_k),
                                         RAD_STR("dtype", eenc[0].dtype) };
            pp.insert(pp.end(), cls_p.begin(), cls_p.end());
            const rad_op pg_h = rad_decl_op(b, ncls == 4 ? exl3::kOpPrepGu4 : exl3::kOpPrepGu, pp.data(),
                                            (int)pp.size(), pt.data(), (int)pt.size());
            /* at four classes prep_gu also reads the router's ids (topk_ids: each row's expert without a search;
             * exl3r4d launch.c, P10-TP4S (a')) */
            op_pg = ncls == 4 ? rw(b, pg_h, {w.h.x, w.sorted, w.eoff, w.ids}, {b_xr})
                              : rw(b, pg_h, {w.h.x, w.sorted, w.eoff}, {b_xr});
            op_gu_ar = decl_gu(prep_rows, 1, b_xr);
            return b_xr && op_pg && op_gu_ar ? RAD_OK : RAD_E_INVAL;
        };
        op_gu = decl_gu(c.rows, 0, w.h.x);
        RAD_ARCH_TRY(declare_decode_band());
    }
    /* ---- THE DECODE CHAIN AS ONE OP (exl3::kGd), declared BETWEEN the layer's gate_up ops and its
     * down: the engine keys a routed layer's prefill staging on the FIRST and LAST declared ops that
     * read its experts (core/place/stager.cpp: the copies go out at the first, the buffer is let go
     * after the last), and both must be issued in every pass the stager arms for -- the one gate_up
     * and down are; gate_up_down is a decode-band op. Declared after down (58905d9) it was every
     * layer's last reader, never issued in prefill, so the buffers were never let go and staging
     * stopped after the pass's first layers (the engine's stats: 32 layers staged against v6's 768).
     * One dtype and codebook for both halves, so only where the layer's gate_up and down carry the
     * same encoding; not with the calibration tap, which reads h (w.eff), never written here. Ranged
     * from kGdMinRows: below it the band's own form is faster (exl3::kGdMinRows). Optional: a library
     * without the row leaves it 0 and the forms above are issued. At four classes its _mod4 form
     * (interface v7) with N_q / K_q, the same rows. */
    if (band_gd && (ncls == 2 || ncls == 4) && !c.calib && eenc[0].dtype && eenc[1].dtype && eenc[0].cb &&
        eenc[1].cb && !std::strcmp(eenc[0].dtype, eenc[1].dtype) && !std::strcmp(eenc[0].cb, eenc[1].cb) &&
        c.rows >= exl3::kGdDeclMin) {
        gd_rows = std::min<int64_t>(exl3::kGdMaxRows, c.rows);
        const std::vector<rad_weight> tab = tables(EXL3_GU_DN);
        if (ncls == 4) {
            class_params(nq_gu, kq_embd);
            std::vector<RadParam> gdp = { RAD_RANGE("M", exl3::kGdDeclMin, gd_rows), RAD_INT("N", 2 * wq[0]),
                                          RAD_INT("K", g.n_embd), RAD_INT("n_expert", n_reg),
                                          RAD_INT("top_k", c.top_k) };
            gdp.insert(gdp.end(), cls_p.begin(), cls_p.end());
            gdp.push_back(RAD_STR("dtype", eenc[0].dtype));
            gdp.push_back(RAD_STR("codebook", eenc[0].cb));
            op_gd = rad_decl_op(b, exl3::kOpGateUpDown4, gdp.data(), (int)gdp.size(), tab.data(), (int)tab.size());
        } else {
            op_gd = rad_decl_op(b, exl3::kOpGateUpDown,
                                RAD_PARAMS(RAD_RANGE("M", exl3::kGdDeclMin, gd_rows), RAD_INT("N", 2 * wq[0]),
                                           RAD_INT("K", g.n_embd), RAD_INT("n_expert", n_reg),
                                           RAD_INT("top_k", c.top_k), RAD_INT("N_odd", 2 * wq[1]),
                                           RAD_INT("K_odd", g.n_embd), RAD_STR("dtype", eenc[0].dtype),
                                           RAD_STR("codebook", eenc[0].cb)),
                                tab.data(), (int)tab.size());
        }
        if (op_gd) {
            const rad_buf rd[3] = { w.h.x, w.sorted, w.eoff };
            const rad_buf wr[1] = { w.edn };
            rad_op_reads (b, op_gd, rd, 3);
            rad_op_writes(b, op_gd, wr, 1);
        }
    }
    /* down ZEROES every sorted row past the regular experts' offsets -- dropped slots, and the
     * protected experts' rows, which their own pair below then writes. */
    {
        const std::vector<rad_weight> tab = tables(EXL3_DOWN);
        class_params(nq_dn, kq_dn);
        std::vector<RadParam> dp = { RAD_RANGE("M", 1, c.rows), RAD_INT("N", g.n_embd), RAD_INT("K", wq[0]),
                                     RAD_INT("n_expert", n_reg), RAD_INT("top_k", c.top_k) };
        dp.insert(dp.end(), cls_p.begin(), cls_p.end());
        dp.push_back(RAD_STR("dtype", eenc[1].dtype));
        dp.push_back(RAD_STR("codebook", eenc[1].cb));
        dp.push_back(RAD_INT("h_rot", exl3::kHRot));
        op_dn = rw(b, rad_decl_op(b, ncls == 4 ? exl3::kOpDown4 : exl3::kOpDown, dp.data(), (int)dp.size(),
                                  tab.data(), (int)tab.size()),
                   {w.eff, w.sorted, w.eoff}, {w.edn});
    }
    /* ---- The protected experts' own GEMM pair. Their weights belong to the layer group, not to
     * expert slots: each container expert unit has a fixed size and these are four times larger, so
     * they are static and resident with the router. One run per projection, a table of
     * `prot.size()`; the GEMMs take the trailing offsets, the rows the quantised down GEMM left
     * zero.
     *
     * When every rank holds a slice of every expert, it holds a slice of these too. A bf16 expert
     * has no scale blocks, so the slice is even where possible (rank r takes intermediate units
     * [r*w, (r+1)*w)), and the all-reduce sums the partials as for the other experts. */
    if (!prot.empty()) {
        const int64_t np = (int64_t)prot.size();
        int64_t wp = c.n_ff_exp;
        int     psh_row = RAD_SHARD_NONE, psh_col = RAD_SHARD_NONE, pparts = 0;
        /* This rank's range within one plain expert, in model numbering; `phi` 0 is the even split,
         * which the loader divides itself. */
        int64_t plo = 0, phi = 0;
        if (split) {
            /* The expert's whole width: `n_ff_all` (the max of `ff_hi` is the whole width only at
             * two ranks). A plain expert has no scale blocks, so the unit is the GEMM's 64-wide
             * output tile. */
            const int64_t full = g.n_ff_all > 0 ? g.n_ff_all
                                                : (c.ff_hi[0] > c.ff_hi[1] ? c.ff_hi[0]
                                                                          : c.ff_hi[1]);
            /* One slice per rank, possibly uneven in whole 64-wide tiles: 640 columns are ten
             * tiles, which four ranks take as 3, 3, 2, 2 (lower ranks take the extra). Each tile
             * has one owner and the all-reduce sums the partials. An evenly divisible width uses
             * the plain even split with no span. */
            psh_row = RAD_SHARD_ROW;
            psh_col = RAD_SHARD_COL;
            pparts  = 2;
            const int64_t tile = exl3::kPlainTile;
            if (full % g.world == 0 && (full / g.world) % tile == 0) {
                wp = full / g.world;
            } else if (full % tile || full / tile < g.world) {
                fprintf(stderr, "radiance: layer %d's plain experts are %lld wide, which %d ranks "
                                "cannot share in %lld-wide tiles\n", l, (long long)full, g.world,
                        (long long)tile);
                return RAD_E_INVAL;
            } else {
                const int64_t nt = full / tile, base = nt / g.world, extra = nt % g.world;
                const int64_t t0 = (int64_t)g.rank * base + std::min<int64_t>(g.rank, extra);
                const int64_t t1 = t0 + base + ((int64_t)g.rank < extra ? 1 : 0);
                plo = t0 * tile;
                phi = t1 * tile;
                wp  = phi - plo;
            }
        }
        w_pgu.assign((size_t)np, 0);
        w_pdn.assign((size_t)np, 0);
        for (int64_t j = 0; j < np; ++j) {
            const int64_t parts[2] = { wp, wp };
            w_pgu[(size_t)j] = decl_w_n(b, nm.f("blk.%d.ffn_gate_up_exps.%lld.weight", l,
                                                (long long)(c.expert_base + prot[(size_t)j])),
                                        RAD_BF16, {2 * wp, g.n_embd}, RAD_ACCESS_PER_TOKEN,
                                        psh_row, grp_layer(l), 0, parts, pparts);
            if (!w_pgu[(size_t)j]) return RAD_E_INVAL;
            if (phi > 0) RAD_ARCH_TRY(rad_weight_shard_span(b, w_pgu[(size_t)j], plo, phi));
        }
        for (int64_t j = 0; j < np; ++j) {
            w_pdn[(size_t)j] = decl_w(b, nm.f("blk.%d.ffn_down_exps.%lld.weight", l,
                                              (long long)(c.expert_base + prot[(size_t)j])),
                                      RAD_BF16, {g.n_embd, wp}, RAD_ACCESS_PER_TOKEN, psh_col,
                                      grp_layer(l));
            if (!w_pdn[(size_t)j]) return RAD_E_INVAL;
            if (phi > 0) RAD_ARCH_TRY(rad_weight_shard_span(b, w_pdn[(size_t)j], plo, phi));
        }
        /* The gate_up and gate outputs, bf16, one sorted row each; regular experts' rows are
         * unused. */
        b_pgu = decl_b(b, nm.f("blk.%d.moe_prot_gu", l), RAD_BF16, {c.rows * c.top_k, 2 * wp});
        pff.x = decl_b(b, nm.f("blk.%d.moe_prot_ff", l), RAD_BF16, {c.rows * c.top_k, wp});
        if (!b_pgu || !pff.x) return RAD_E_INVAL;
        op_pgu = rw(b, rad_decl_op(b, "moe_gemm",
                           RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("N", 2 * wp),
                                      RAD_INT("K", g.n_embd), RAD_INT("n_expert", np),
                                      RAD_INT("top_k", c.top_k), RAD_STR("a_order", "token"),
                                      RAD_STR("dtype", "bf16")),
                           w_pgu.data(), (int)np),
                    {w.h.x, w.sorted, w.eoff}, {b_pgu});
        Geom gx = g;
        gx.w_bf16 = true;
        RAD_ARCH_TRY(gq_prot.declare(b, gx, b_pgu, pff, wp, "silu", c.rows * c.top_k, true,
                                     false, c.swiglu_limit));
        op_pdn = rw(b, rad_decl_op(b, "moe_gemm",
                           RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("N", g.n_embd),
                                      RAD_INT("K", wp), RAD_INT("n_expert", np),
                                      RAD_INT("top_k", c.top_k), RAD_STR("a_order", "sorted"),
                                      RAD_STR("dtype", "bf16")),
                           w_pdn.data(), (int)np),
                    {pff.x, w.sorted, w.eoff}, {w.edn});
        if (!op_pgu || !op_pdn) return RAD_E_INVAL;
    }
    /* ---- the calibration tap, declared after the routed arm. Its PERSIST buffers would not be
     * aliased anyway, but declaration order is the buffer plan, so keep it last. */
    if (c.calib) {
        /* A rank holding a slice of each expert sees a slice of K, and the packer needs the full
         * Hessian, so calibration runs without expert slicing. */
        if (split) {
            fprintf(stderr, "radiance: calibration runs at one rank; this rank holds a slice of "
                            "every expert's width, and its Hessians would be of the slice\n");
            return RAD_E_INVAL;
        }
        if (!calib_dir()) {
            fprintf(stderr, "radiance: calibration is configured but RADIANCE_CALIB_DIR is not "
                            "set; there is nowhere to write the Hessians\n");
            return RAD_E_INVAL;
        }
        /* Both K are whole 128-groups already (the cut check above); gram_accum needs only a
         * multiple of four. */
        gram_gu_stem = nm.f("blk.%d.ffn_gate_up_exps", l);
        gram_dn_stem = nm.f("blk.%d.ffn_down_exps", l);
        b_gram_gu = decl_b(b, nm.f("blk.%d.gram_gu", l), RAD_F32, {g.n_embd, g.n_embd},
                           RAD_BUF_PERSIST);
        b_gram_dn = decl_b(b, nm.f("blk.%d.gram_dn", l), RAD_F32, {c.n_ff_exp, c.n_ff_exp},
                           RAD_BUF_PERSIST);
        /* The model's own basis (Config::calib), so the files say no rotation. */
        gram_domain = 0;
        /* gate_up's operand IS the bf16 input, read as it stands. */
        op_gram_gu = rw(b, RAD_OP(b, "gram_accum",
                           RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n", g.n_embd),
                                      RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "bf16")),
                           RAD_NOWEIGHTS),
                       {w.h.x, b_gram_gu}, {b_gram_gu});
        /* down's is the f16 intermediate, which gram_accum does not read: a bf16 copy of it, one
         * `cast` a pass. The down tap's M is the SORTED height, `rows * top_k`, because `w.eff`
         * is one row per (token, slot). gate_up's is the token height -- its op gathers
         * internally, and every token is routed to exactly top_k experts, so the gathered
         * distribution is the token distribution times top_k and H is used only up to a
         * positive scale. */
        b_calib_eff = decl_b(b, nm.f("blk.%d.moe_calib_eff", l), RAD_BF16,
                             {c.rows * c.top_k, c.n_ff_exp});
        op_calib_cast = rw(b, RAD_OP(b, "cast",
                              RAD_PARAMS(RAD_RANGE("M", 1, c.rows * c.top_k),
                                         RAD_INT("n", c.n_ff_exp), RAD_STR("from", "f16"),
                                         RAD_STR("to", "bf16")),
                              RAD_NOWEIGHTS),
                          {w.eff}, {b_calib_eff});
        op_gram_dn = rw(b, RAD_OP(b, "gram_accum",
                           RAD_PARAMS(RAD_RANGE("M", 1, c.rows * c.top_k),
                                      RAD_INT("n", c.n_ff_exp),
                                      RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "bf16")),
                           RAD_NOWEIGHTS),
                       {b_calib_eff, b_gram_dn}, {b_gram_dn});
        /* One expert's Gram a launch: the same op over that expert's run of the sorted rows,
         * into that expert's square of the scratch. TRANSIENT, because nothing on the card reads
         * it after the pass -- it goes back to the host before the next op is issued. */
        b_gram_dne = decl_b(b, nm.f("blk.%d.gram_dn_experts", l), RAD_F32,
                            {c.n_expert * c.n_ff_exp, c.n_ff_exp});
        op_gram_dne = rw(b, RAD_OP(b, "gram_accum",
                            RAD_PARAMS(RAD_RANGE("M", 1, c.rows * c.top_k),
                                       RAD_INT("n", c.n_ff_exp),
                                       RAD_INT("group", RAD_FP8_BLOCK),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                         {b_calib_eff, b_gram_dne}, {b_gram_dne});
        if (!b_calib_eff || !op_calib_cast || !b_gram_dne || !op_gram_dne) return RAD_E_INVAL;
    }

    /* The gather folds the shared arm in, so `w.moe` is never written; `scale_rows` plus a
     * separate gather would be two dispatches per layer and a 160 KiB round trip. */
    gfold = shared;
    for (int form = scatter_inv ? 0 : 1; form < 2 && !op_gather; ++form) {
        rad_buf grd[5] = { w.edn, w.ew, form ? w.sorted : w.inv, w.sout, w.sgate };
        rad_buf gwr[1] = { w.h.x };
        op_gather = RAD_OP(b, form ? "moe_gather" : "moe_gather_inv",
                           RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n", g.n_embd),
                                      RAD_INT("top_k", c.top_k), RAD_STR("dtype", g.dtype),
                                      RAD_STR("act", gfold && !c.shared_ungated ? "sigmoid" : "none")),
                           RAD_NOWEIGHTS);
        if (!op_gather) continue;
        gather_inv = !form;
        rad_op_reads (b, op_gather, grd, gfold ? 5 : 3);
        rad_op_writes(b, op_gather, gwr, 1);
    }
    if (!op_gather) return RAD_E_INVAL;

    /* ---- the shared expert. Ordinary dense linears over the same normed input. */
    if (shared) {
        /* The shared arm runs concurrently with the routed GEMMs, so its buffers are excluded from
         * sharing (rad_buf_concurrent). Otherwise the linear order would allow `sgu` over `eff`,
         * two streams writing the same bytes. Costs a few MiB of arena. */
        RAD_ARCH_TRY(rad_buf_concurrent(b, w.sgu));
        /* Probe the down projection first: what it reads (codes, or the bf16 plane for a plain
         * weight) decides what the gate writes. */
        const char* down_base = nm.f("blk.%d.ffn_down_shexp", l);
        RAD_ARCH_TRY(sh_down.probe(b, nm, g, down_base,
                                   { nm.ckpt("%s.down_proj.weight", src.shared) }));
        /* The code pair the gate writes, E4M3 or int8 (ActFP8::q8_fed); an int8 down projection
         * that quantises for itself also writes the int8 pair on this lane. */
        if (w.sff.cq()) {
            RAD_ARCH_TRY(rad_buf_concurrent(b, w.sff.cq()));
            RAD_ARCH_TRY(rad_buf_concurrent(b, w.sff.cs()));
        }
        if (w.sff.q8 && !w.sff.q8_fed && sh_down.probed_i8) {
            RAD_ARCH_TRY(rad_buf_concurrent(b, w.sff.q8));
            RAD_ARCH_TRY(rad_buf_concurrent(b, w.sff.s8));
        }
        if (!w.sff.q || c.swiglu_limit>0 || (sh_down.reads_x && !(w.sff.q8_fed && sh_down.probed_i8)))
            RAD_ARCH_TRY(rad_buf_concurrent(b, w.sff.x));
        RAD_ARCH_TRY(rad_buf_concurrent(b, w.sout));
        RAD_ARCH_TRY(rad_buf_concurrent(b, w.sgate));
        sgate_concurrent = true;
        /* gate/up rows and down columns of this rank's slice (even, or Config's uneven split); the
         * gate/up row parts stay, as a property of the stacked checkpoint tensor. */
        RAD_ARCH_TRY(sh_gate_up.declare(b, nm, g, nm.f("blk.%d.ffn_gate_up_shexp", l),
                                        2 * c.n_ff_shared, g.n_embd, RAD_SHARD_ROW, grp_layer(l),
                                        { nm.ckpt("%s.gate_proj.weight", src.shared),
                                          nm.ckpt("%s.up_proj.weight",   src.shared) },
                                        w.h, w.sgu, c.rows,
                                        { c.n_ff_shared, c.n_ff_shared },
                                        c.shared_lo, c.shared_hi));
        /* An int8 down projection fed the int8 pair reads no bf16 plane, so none is written. */
        gq_sh.fuse_clamped = c.swiglu_quant != 0;
        RAD_ARCH_TRY(gq_sh.declare(b, g, w.sgu, w.sff, c.n_ff_shared, "silu", c.rows,
                                   sh_down.reads_x && !(w.sff.q8_fed && sh_down.probed_i8), false,
                                   c.swiglu_limit));
        /* THE DECODE-BAND FORM ISSUES THIS ARM BETWEEN prep_gu AND THE ROUTED GEMM, which is safe
         * only because every buffer it writes is out of the sharing pool (above). The one write the
         * helper can add is an int8 gate_up's own quantiser (LinearFP8::op_q) into the INPUT's int8
         * pair, which is not -- it could land on xr. This plugin always feeds that pair from the
         * producer (a_x.q8_fed with an int8 trunk), so op_q never exists; if it ever does, refuse. */
        if (op_pg && sh_gate_up.op_q) {
            fprintf(stderr, "radiance: layer %d: the shared expert's gate_up quantises its own int8 "
                            "input, a write in the sharing pool that the decode-band issue order "
                            "would put between moe_exl3_prep_gu and the routed GEMM (over xr) -- "
                            "refusing; feed the int8 pair from the producer (ActFP8::q8_fed)\n", l);
            return RAD_E_UNSUPPORTED;
        }
        RAD_ARCH_TRY(sh_down.declare(b, nm, g, down_base,
                                     g.n_embd, c.n_ff_shared, RAD_SHARD_COL, grp_layer(l),
                                     { nm.ckpt("%s.down_proj.weight", src.shared) },
                                     w.sff, w.sout, c.rows, {},
                                     c.shared_lo, c.shared_hi));
        /* The shared gate, replicated like the router so every rank computes the same gate. One
         * output column: the skinny GEMM kernel clamps its N tile rather than masking, so N = 1 is
         * legal. An ungated arm (shared_ungated) fills the gate with 1 instead. */
        if (c.shared_ungated) {
            op_sgate = rw(b, RAD_OP(b, "fill",
                             RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n", 1),
                                        RAD_STR("dtype", g.dtype), RAD_F64("value", 1.0)),
                             RAD_NOWEIGHTS),
                         {}, {w.sgate});
        } else {
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_gate_shexp.weight", l), src.shared_gate));
        w_sgate = decl_w(b, nm.f("blk.%d.ffn_gate_shexp.weight", l), RAD_BF16, {1, g.n_embd},
                         RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
        op_sgate = rw(b, RAD_OP(b, "gemm_nt",
                         RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("N", 1),
                                    RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(w_sgate)),
                     {w.h.x}, {w.sgate});
        }
    }

    /* ---- The router GEMM and `router_topk_scatter` as one op, plus the shared gate when the
     * layer has one; up to the pair's row bound and the skinny GEMM's decode band. Unresolved, the
     * separate ops run. Declared here, after the gate weight, because it names both weights;
     * declaring late only lengthens its buffers' planned lifetimes, since each is also declared by
     * the op it replaces.
     *
     * Exception: it writes `w.sgate` before `op_sgate`'s declared index, so the planner could lend
     * those bytes to a buffer live in between. That is safe only because the shared arm excluded
     * `w.sgate` from sharing (rad_buf_concurrent); otherwise the fused gate is refused.
     *
     * The parameter list, weights and write set are conditional, so each is an array cut to
     * length. */
    if (op_topk_scatter && !c.shared_ungated && rad_op_declared(b, "router_gemm_topk_scatter")) {
        router_fused_rows = topk_scatter_rows < kMoeRouterFusedRows ? topk_scatter_rows
                                                                    : kMoeRouterFusedRows;
        const bool gate = op_sgate != 0;
        if (gate && !sgate_concurrent) {
            fprintf(stderr, "radiance: layer %d: the fused router writes the shared gate ahead of "
                            "its declared writer, which needs `sgate` out of the sharing pool, "
                            "and it is not\n", l);
            return RAD_E_STATE;
        }
        const RadParam fp[] = {
            RAD_RANGE("M", 1, router_fused_rows), RAD_INT("K", g.n_embd),
            RAD_INT("n_expert", c.n_expert_all), RAD_INT("n_local", c.n_expert),
            RAD_INT("top_k", c.top_k), RAD_INT("norm", c.norm_topk),
            RAD_STR("dtype", g.dtype), RAD_INT("expert_base", c.expert_base),
            RAD_STR("protect", prot_list),
        };
        const int n_fp = (int)(sizeof(fp) / sizeof(fp[0])) - (prot_list ? 0 : 1);
        const rad_weight fw[2] = { w_router, w_sgate };
        op_router_fused = rad_decl_op(b, "router_gemm_topk_scatter", fp, n_fp, fw, gate ? 2 : 1);
        if (op_router_fused) {
            const rad_buf rd[1] = { w.h.x };
            const rad_buf wr[7] = { w.logits, w.ids, w.ew, w.sorted, w.eoff, w.ecnt, w.sgate };
            rad_op_reads (b, op_router_fused, rd, 1);
            rad_op_writes(b, op_router_fused, wr, gate ? 7 : 6);
        }
        router_fused_gate = op_router_fused && gate;
    }

    /* GLM's selection and the sort as one op, plus the ungated shared arm's gate fill. Declared
     * here for the reason the fused router above is: it writes `w.sgate` ahead of `op_sgate`'s
     * declared index, which is safe only with `sgate` out of the sharing pool. Its other outputs
     * and its input are also declared by the ops it replaces, so their planned lifetimes cover it.
     * Undeclared or unresolved, the separate ops serve. */
    if (c.sigmoid_router && c.router_fuse) {
        if (!rad_op_declared(b, "router_sigmoid_topk_scatter")) {
            static bool told = false;
            if (!told) rad_note(b, "no loaded library declares router_sigmoid_topk_scatter; the "
                                   "selection and the sort stay separate launches");
            told = true;
        } else {
            sigmoid_scatter_rows = (int64_t)c.rows * c.top_k <= kMoeSortRows
                                       ? c.rows : kMoeSortRows / c.top_k;
            const bool gate = op_sgate != 0 && c.shared_ungated;
            if (gate && !sgate_concurrent) {
                fprintf(stderr, "radiance: layer %d: the fused selection writes the shared gate "
                                "ahead of its declared writer, which needs `sgate` out of the "
                                "sharing pool, and it is not\n", l);
                return RAD_E_STATE;
            }
            const RadParam sp[] = {
                RAD_RANGE("M", 1, sigmoid_scatter_rows), RAD_INT("n_expert", c.n_expert_all),
                RAD_INT("n_local", c.n_expert), RAD_INT("top_k", c.top_k),
                RAD_INT("norm", c.norm_topk), RAD_F64("route_scale", c.route_scale),
                RAD_STR("dtype", "f32"), RAD_INT("expert_base", c.expert_base),
                RAD_STR("protect", prot_list),
            };
            const int n_sp = (int)(sizeof(sp) / sizeof(sp[0])) - (prot_list ? 0 : 1);
            op_sigmoid_scatter = rad_decl_op(b, "router_sigmoid_topk_scatter", sp, n_sp,
                                             &glm_router.correction, 1);
            if (op_sigmoid_scatter) {
                const rad_buf rd[1] = { w.logits };
                rad_buf wr[7] = { w.ids, w.ew, w.sorted, w.eoff, w.ecnt };
                int n_wr = 5;
                if (gate) wr[n_wr++] = w.sgate;
                if (scatter_inv) wr[n_wr++] = w.inv;
                rad_op_reads (b, op_sigmoid_scatter, rd, 1);
                rad_op_writes(b, op_sigmoid_scatter, wr, n_wr);
            }
            sigmoid_scatter_gate = op_sigmoid_scatter && gate;
        }
    }

    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                       RAD_PARAMS(RAD_INT("world_size", g.world),
                                  RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                  RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                  RAD_INT("min_bytes", g.wire_min_bytes)),
                       RAD_NOWEIGHTS),
                   {w.h.x}, {w.h.x});
    /* The same collective for a caller that runs it beside other work (issue_ar, a pipelined
     * prefill): at `overlap` 1 it gets the rows that hold fewer CUs while waiting on the link. */
    if (g.world > 1 && ar_overlap > 0)
        op_ar_ov = rw(b, RAD_OP(b, "all_reduce",
                          RAD_PARAMS(RAD_INT("world_size", g.world),
                                     RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                     RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                     RAD_INT("min_bytes", g.wire_min_bytes),
                                     RAD_INT("overlap", ar_overlap)),
                          RAD_NOWEIGHTS),
                      {w.h.x}, {w.h.x});

    if (add_out)
        op_add = rw(b, RAD_OP(b, "add",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.x, w.h.x}, {w.x});

    /* THE LAYER AS THE PREFETCH SEES IT, by the model's expert number -- what the router ranks. A plane
     * is a part when exactly one of the ops this layer declared tables it (see xp_next): gate_up's in
     * the one gate_up (+ the decode-band pair, + gate_up_down), prep_gu's suh and suh.1, down's in down
     * (+ gate_up_down; its suh also in every gate_up, h_rot). Codes first, then svh, then the suh
     * vectors: the bytes in that order. */
    if (!c.sigmoid_router && c.n_expert_all > 0 && n_reg > 0) {
        const int gu_ops = (op_gu ? 1 : 0) + (op_gu_ar ? 1 : 0) + (op_gd ? 1 : 0);
        int n_in[2][exl3::kPlanesGateUp] = {};   /* [proj][plane]: tables holding it */
        for (int p = 0; p < exl3::kPlanesGateUp; ++p)
            n_in[0][p] = gu_ops + (op_pg && (p == 1 || p == 2) ? 1 : 0);
        n_in[1][exl3::kDownCodes] = (op_dn ? 1 : 0) + (op_gd ? 1 : 0);
        n_in[1][exl3::kDownSvh]   = (op_dn ? 1 : 0) + (op_gd ? 1 : 0);
        n_in[1][exl3::kDownSuh]   = (op_dn ? 1 : 0) + (exl3::kHRot ? gu_ops : 0);
        static const int order[][2] = { {0, 0}, {1, exl3::kDownCodes}, {0, 3}, {1, exl3::kDownSvh},
                                        {0, 1}, {0, 2}, {1, exl3::kDownSuh} };
        int pick[4][2], np = 0;
        for (const auto& o : order)
            if (np < 4 && n_in[o[0]][o[1]] == 1) { pick[np][0] = o[0]; pick[np][1] = o[1]; ++np; }
        if (np > 0) {
            const int64_t na = c.n_expert_all;
            xp_parts.assign((size_t)(na * 4), 0);
            xp_count_at.assign((size_t)na, -1);
            for (int64_t s = 0; s < n_reg; ++s) {
                const int64_t e = c.expert_base + reg[(size_t)s];
                if (e < 0 || e >= na) continue;
                for (int k = 0; k < np; ++k)
                    xp_parts[(size_t)(e * 4 + k)] =
                        (pick[k][0] == 0 ? w_gu[pick[k][1]] : w_dn[pick[k][1]])[(size_t)s];
                xp_count_at[(size_t)e] = (int32_t)s;
            }
            xp.layer = layer;
            xp.router = w_router;
            xp.n_expert = na;
            xp.top_k = c.top_k;
            xp.n_parts = 4;
            xp.parts = xp_parts.data();
            xp.count_at = xp_count_at.data();
        }
    }
    return RAD_OK;
}

/* staging off (exl3_stage.h): a moe_exl3_prep_gu over this block's suh / suh.1 tables, ranged M 1..1, no buffer edges,
 * never issued -- appended after the language model's last op so this layer's span of declared expert readers
 * overlaps every other's and the engine's prefill stager turns itself off (exl3::kPrefillStageForced says why). */
inline int MoeEXL3::declare_stage_off(RadBuilder* b) {
    static const char* const kq[4] = { "K_odd", "K_1", "K_2", "K_3" };
    std::vector<rad_weight> tab;
    for (int q = 0; q < ncls; ++q)
        for (int t = 0; t < n_tables(EXL3_PREP_GU); ++t)
            for (int64_t s = q; s < n_reg; s += ncls) tab.push_back(table_run(EXL3_PREP_GU, t)[(size_t)s]);
    std::vector<RadParam> pp = { RAD_RANGE("M", 1, 1), RAD_INT("K", g.n_embd), RAD_INT("n_expert", n_reg),
                                 RAD_INT("top_k", c.top_k), RAD_STR("dtype", eenc[0].dtype) };
    for (int q = 1; q < ncls; ++q) pp.push_back(RAD_INT(kq[ncls == 4 ? q : 0], g.n_embd));
    op_stage_off = rad_decl_op(b, ncls == 4 ? exl3::kOpPrepGu4 : exl3::kOpPrepGu, pp.data(), (int)pp.size(),
                               tab.data(), (int)tab.size());
    return RAD_OK;   /* resolved or not, it is a declared reader (rad_builder.cpp finish_weight_uses) */
}

/* The shared arm's first stage: its one-column gate and its gate_up over the normed input. Its
 * buffers are step-tall, so the pass offset applies as in an MlpFP8 slice. */
inline void MoeEXL3::shared_up(RadCtx* c_, int64_t T, int64_t r0, int64_t rows,
                              bool gate_done) const {
    /* `gate_done`: the fused router already wrote this pass's gate. */
    if (!gate_done && c.shared_ungated)
        RAD_ISSUE_N(c_, op_sgate, rows, brow_slice(w.sgate, r0, rows, 1));
    else if (!gate_done)
        RAD_ISSUE_N(c_, op_sgate, rows, brow_slice(w.h.x, r0, rows, g.n_embd), RAD_W(w_sgate),
                    brow_slice(w.sgate, r0, rows, 1));
    sh_gate_up.step(c_, w.h, w.sgu, T, r0, rows);
}

/* One pass of at most `c.rows` tokens. Every op is per row (norm and quantiser within a row,
 * router and top-k per row, per-sorted-row GEMM accumulation, the gather per token), so a pass
 * computes exactly its rows.
 *
 * The expert buffers are addressed from zero, not `r0`: they are a pass tall and `sorted_tok` is
 * pass-relative. Only step-tall operands (residual stream, normed input) carry the offset. */
inline void MoeEXL3::pass(RadCtx* c_, int64_t T, int64_t r0, int64_t rows, bool own_ar) const {
    /* `ne` is this rank's expert count, `ne_all` the model's: logits and top-k are model-wide,
     * everything after the scatter is this rank's slice. */
    const int64_t ne = c.n_expert, ne_all = c.n_expert_all, k = c.top_k;
    const bool shared = op_sgate != 0;

    if (!ext_in) nq_h.step(c_, T, r0, rows);
#if QWEN4EXP_EXL3_SHARED_LANE
    /* exl3::kSharedLane: on a gate_up_down pass the shared arm goes to lane 1 here, behind the normed input */
    /* 1: gate_up_down passes; 2: every decode-band pass (the split form's too) */
    const bool sh_lane = shared && (QWEN4EXP_EXL3_SHARED_LANE == 3
                                        ? !(op_gd && rows >= exl3::kGdMinRows && rows <= gd_rows) && !(op_pg && rows <= prep_rows)
                                        : ((op_gd && rows >= exl3::kGdMinRows && rows <= gd_rows) ||
                                           (QWEN4EXP_EXL3_SHARED_LANE == 2 && op_pg && rows <= prep_rows)));
    if (sh_lane) {
        (void)rad_lane_join(c_, 0, 1);
        (void)rad_lane(c_, 1);
        shared_up(c_, T, r0, rows, op_router_fused && rows <= router_fused_rows && router_fused_gate);
        gq_sh.step(c_, w.sgu, T, r0, rows);
        sh_down.step(c_, w.sff, w.sout, T, r0, rows);
        (void)rad_lane(c_, 0);
    }
    const bool sh_l0 = shared && !sh_lane;   /* the arm on lane 0, as issued below */
#define QX_SH_L0 sh_l0
#else
#define QX_SH_L0 shared   /* the code object of a build without the arm is the one before it, byte for byte */
#endif

    /* The shared arm is interleaved with the routed arm on the same queue. It reads the same
     * input and writes nothing the routed arm touches before the gather, so each of its stages
     * (gate + gate_up, quantiser, down) is issued right behind the matching routed stage. A
     * recorded pass leaves each such pair unordered (Ctx::haz_mark), so they overlap without a
     * second queue's fork/join cost. The arm's buffers are excluded from sharing (see declare), so
     * issuing them ahead of their declared position is safe. */

    /* The routed arm. */
    /* Predict the next routed layer's experts from this router's input, on the compute stream
     * (rad_expert_prefetch). A failure only loses the prediction. */
    if (xp_next && xp_next->xp.n_parts > 0)
        (void)rad_expert_prefetch(c_, &xp_next->xp, w.h.x, r0, rows, g.n_embd);
    /* Write the routing counts straight into the core's row for this layer when it keeps one, so
     * the report in step() needs no copy (otherwise a dependent blit per routed layer). */
    int32_t* const counts = rad_route_counts(c_, layer, ne);
    const RadOperand ecnt = counts ? praw(counts, RAD_I32, ne) : brows(w.ecnt, ne);
    /* The fused router when it resolved and the pass fits: router, topk + scatter and (with
     * `router_fused_gate`) the shared gate in one launch. Otherwise the separate ops. */
    const bool rfused = op_router_fused && rows <= router_fused_rows;
    const bool sfused = op_sigmoid_scatter && rows <= sigmoid_scatter_rows;
    if (sfused) {
        glm_router.project(c_, r0, rows, r0);
        RAD_ISSUE_N(c_, op_sigmoid_scatter, rows, brow_slice(w.logits, r0, rows, ne_all),
                    RAD_W(glm_router.correction), brow_slice(w.ids, r0, rows, k),
                    brow_slice(w.ew, r0, rows, k), brows(w.sorted, rows * k),
                    brows(w.eoff, ne + 1), ecnt,
                    sigmoid_scatter_gate ? brow_slice(w.sgate, r0, rows, 1) : RAD_NONE,
                    scatter_inv ? brows(w.inv, rows * k) : RAD_NONE);
    } else if (c.sigmoid_router) {
        glm_router.step(c_, r0, rows, r0);
    } else if (rfused)
        RAD_ISSUE_N(c_, op_router_fused, rows, brow_slice(w.h.x, r0, rows, g.n_embd),
                    RAD_W(w_router), router_fused_gate ? RAD_W(w_sgate) : RAD_NONE,
                    brow_slice(w.logits, r0, rows, ne_all),
                    brow_slice(w.ids, r0, rows, k), brow_slice(w.ew, r0, rows, k),
                    brows(w.sorted, rows * k), brows(w.eoff, ne + 1), ecnt,
                    router_fused_gate ? brow_slice(w.sgate, r0, rows, 1) : RAD_NONE);
    else
        RAD_ISSUE_N(c_, op_router, rows, brow_slice(w.h.x, r0, rows, g.n_embd), RAD_W(w_router),
                    brow_slice(w.logits, r0, rows, ne_all));
    /* topk + scatter fused, or the pair. In the fused op `w.ids` is an output. */
    if (rfused || sfused) {
        /* the fused router wrote all five */
    } else if (op_topk_scatter && rows <= topk_scatter_rows) {
        RAD_ISSUE_N(c_, op_topk_scatter, rows, brow_slice(w.logits, r0, rows, ne_all),
                    brow_slice(w.ids, r0, rows, k), brow_slice(w.ew, r0, rows, k),
                    brows(w.sorted, rows * k), brows(w.eoff, ne + 1), ecnt);
    } else {
        if (!c.sigmoid_router)
        RAD_ISSUE_N(c_, op_topk, rows, brow_slice(w.logits, r0, rows, ne_all),
                    brow_slice(w.ids, r0, rows, k), brow_slice(w.ew, r0, rows, k));
        if (scatter_inv)
            RAD_ISSUE_N(c_, op_scatter, rows, brow_slice(w.ids, r0, rows, k),
                        brows(w.sorted, rows * k), brows(w.eoff, ne + 1), ecnt,
                        brows(w.inv, rows * k));
        else
            RAD_ISSUE_N(c_, op_scatter, rows, brow_slice(w.ids, r0, rows, k),
                        brows(w.sorted, rows * k), brows(w.eoff, ne + 1), ecnt);
    }

    /* exl3::kGdOrder 1: on a gate_up_down pass the shared gate_up goes right behind the router. */
    const bool gd = op_gd && rows >= exl3::kGdMinRows && rows <= gd_rows;
    if (exl3::kGdOrder == 1 && gd && QX_SH_L0) shared_up(c_, T, r0, rows, rfused && router_fused_gate);

    /* The two ops, and the weight tables: the FIRST handle of each run and its length. The runs
     * are contiguous by construction -- see declare -- so `w_gu[p][q] + i` is plane p of expert
     * 2 * i + q (at four classes 4 * i + q, class q holding class_n(q)). Each operand goes where
     * exl3_abi.h's exl3_positions() puts it: the input, the even class's tables, the routing, the
     * odd class's tables (classes 1..3's), the output -- gate_up's tables ending with down's suh
     * run (dn_suh, dn_suh_odd). */
    const auto issue = [&](rad_op op_h, int op, RadOperand in, RadOperand out) {
        const Exl3Positions& at = *exl3_positions(form(op));
        const int  n_tab = n_tables(op);
        const int* slot  = table_slots(op);
        RadOperand opd[32];
        static_assert(sizeof opd / sizeof opd[0] >= 32, "gate_up_down_mod4 has 32 operands");
        opd[at.x]      = in;
        opd[at.sorted] = brows(w.sorted, rows * k);
        opd[at.offset] = brows(w.eoff, n_reg + 1);
        opd[at.y]      = out;
        if (at.ids >= 0) opd[at.ids] = brow_slice(w.ids, r0, rows, k);   /* prep_gu_mod4's topk_ids: the sort's input */
        for (int q = 0; q < ncls; ++q)
            for (int t = 0; t < n_tab; ++t) {
                const int sq = exl3::kTestMutation == 1 ? q ^ 1 : q, tq = mut3(op, t) ? sq ^ 1 : sq;
                opd[at.plane[q][slot[t]]] = RAD_WTAB(table_run(op, t)[(size_t)tq], class_n(tq));
            }
        rad_issue(c_, op_h, opd, at.n_opd, rows);
    };
    /* THE DECODE-BAND FORM (kPrepSplit, see exl3::kPrepRows): prep_gu, then the shared arm's first
     * two stages in the gap before the routed GEMM -- they read the normed input and write only
     * buffers out of the sharing pool, prep_gu writes only xr, and the GEMM reads xr -- so a recorded
     * pass leaves them unordered behind prep_gu (Ctx::haz_mark) where the GEMM, today the second
     * dispatch of the one gate_up op, is always ordered. Otherwise the one gate_up, as before.
     * ROOM LEFT (P4b, 2026-10-13, engine measurements): prep_gu is one more DEPENDENT step a layer than
     * the fp8 baseline has -- its rotation needs the routing (each expert's suh), where libr4d's input
     * quant uses one fixed Hadamard and runs beside the router. A dependent dispatch costs ~6.7 us of a
     * decode step's critical path a layer (an empty one added: +0.31 ms at C=1..4), which is the whole
     * C=1 gap to the baseline with MTP off (~0.1 ms) and ~0.28 ms of a C=1 MTP step. The prep's own body
     * is not the cost (a prep without its run search, bit-identical, changed nothing). Taking the rotation
     * into gate_up's launch removes the step but has cost more every way tried (in-workgroup prep, P1c
     * global scratch, LDS-fed A, one launch with tickets, the fused op at 1-2 rows): the rotation phase's
     * loads stall inside the GEMM workgroup (~+14 us at M=1) where the same loads in this separate kernel
     * are cheap -- not yet explained. */
    /* THE DECODE CHAIN AS ONE OP (exl3::kGd), from kGdMinRows rows: gate_up and down in one op over
     * the same operands, the shared arm's first two stages ahead of it. */
    /* The layer before predicted for this one: its copies go into the tables here, where they landed. */
    (void)rad_expert_prefetch_step(c_, layer);
    if (gd) {
        /* the shared arm's first two stages FIRST: they read the normed input and write only buffers
         * out of the sharing pool, so nothing orders gate_up_down behind them and they can run beside
         * it -- issued after, they would wait for the persistent launch to leave the card */
        if (QX_SH_L0 && exl3::kGdOrder == 0) shared_up(c_, T, r0, rows, (rfused && router_fused_gate) || (sfused && sigmoid_scatter_gate));
        if (QX_SH_L0) gq_sh.step(c_, w.sgu, T, r0, rows);
        if (QX_SH_L0 && exl3::kGdOrder == 1) sh_down.step(c_, w.sff, w.sout, T, r0, rows);   /* beside the launch */
        issue(op_gd, EXL3_GU_DN, brow_slice(w.h.x, r0, rows, g.n_embd), brows(w.edn, rows * k));
    } else if (op_pg && rows <= prep_rows) {
        issue(op_pg, EXL3_PREP_GU, brow_slice(w.h.x, r0, rows, g.n_embd), brows(b_xr, rows * k));
        if (QX_SH_L0) shared_up(c_, T, r0, rows, (rfused && router_fused_gate) || (sfused && sigmoid_scatter_gate));
        if (QX_SH_L0) gq_sh.step(c_, w.sgu, T, r0, rows);
        issue(op_gu_ar, EXL3_GATE_UP, brows(b_xr, rows * k), brows(w.eff, rows * k));
    } else {
        issue(op_gu, EXL3_GATE_UP, brow_slice(w.h.x, r0, rows, g.n_embd), brows(w.eff, rows * k));
        if (QX_SH_L0) shared_up(c_, T, r0, rows, (rfused && router_fused_gate) || (sfused && sigmoid_scatter_gate));
        if (QX_SH_L0) gq_sh.step(c_, w.sgu, T, r0, rows);
    }
    if (!gd) issue(op_dn, EXL3_DOWN, brows(w.eff, rows * k), brows(w.edn, rows * k));
    /* This layer's streamed experts are read: the next one's copies cross the link from here. */
    (void)rad_expert_prefetch_mark(c_);
    /* THE PROTECTED EXPERTS, over the trailing run of the offsets: their sorted rows are the
     * ones the EXL3 down op above zeroed as past its live count, and the down GEMM here writes
     * them -- after it, which the shared `edn` orders. */
    if (op_pgu) {
        const int64_t np = (int64_t)prot.size();
        const RadOperand poff = brow_slice(w.eoff, n_reg, np + 1, 1);
        RAD_ISSUE_N(c_, op_pgu, rows, brow_slice(w.h.x, r0, rows, g.n_embd),
                    RAD_WTAB(w_pgu[0], np), brows(w.sorted, rows * k), poff,
                    brows(b_pgu, rows * k));
        gq_prot.step(c_, b_pgu, rows * k, 0, rows * k);
        RAD_ISSUE_N(c_, op_pdn, rows, brows(pff.x, rows * k), RAD_WTAB(w_pdn[0], np),
                    brows(w.sorted, rows * k), poff, brows(w.edn, rows * k));
    }
    if (QX_SH_L0 && !(gd && exl3::kGdOrder == 1)) sh_down.step(c_, w.sff, w.sout, T, r0, rows);
    /* THE CALIBRATION TAP, issued between the two GEMMs' inputs being written and the pass
     * ending. It reads exactly the operands the GEMMs above read and writes nothing they
     * touch, so removing it changes no number the model produces -- which is the property
     * that lets a calibration run be compared against a serving one. The down tap's bf16 copy
     * is the tap's own buffer, written by its own cast. */
    if (op_gram_gu && !calib_done) {
        if (calib_rows == 0) {
            /* The accumulators are arena memory, so zero them first. */
            rad_memset_async(rad_buf_ptr(c_, b_gram_gu), 0,
                             g.n_embd * g.n_embd * 4, rad_stream(c_));
            rad_memset_async(rad_buf_ptr(c_, b_gram_dn), 0,
                             c.n_ff_exp * c.n_ff_exp * 4, rad_stream(c_));
        }
        RAD_ISSUE_N(c_, op_gram_gu, rows, brow_slice(w.h.x, r0, rows, g.n_embd), RAD_NONE,
                    brows(b_gram_gu, g.n_embd));
        RAD_ISSUE_N(c_, op_calib_cast, rows * k, brows(w.eff, rows * k),
                    brows(b_calib_eff, rows * k));
        RAD_ISSUE_N(c_, op_gram_dn, rows * k, brows(b_calib_eff, rows * k), RAD_NONE,
                    brows(b_gram_dn, c.n_ff_exp));
        if (op_gram_dne) calib_experts_pass(c_, rows * k);
        calib_rows += rows;
        if (calib_rows >= calib_target()) {
            const bool a = calib_write(c_, b_gram_gu, g.n_embd, calib_rows, gram_gu_stem,
                                       gram_domain);
            const bool bb = calib_write(c_, b_gram_dn, c.n_ff_exp, calib_rows * k,
                                        gram_dn_stem, gram_domain);
            const bool ce = !op_gram_dne ||
                            (!calib_dne_failed &&
                             calib_dne.write(gram_dn_stem, c.expert_base, gram_domain,
                                             rad_rank(c_)));
            calib_done = true;
            if (layer == 0)
                std::fprintf(stderr, "radiance: calibration wrote layer Hessians after %lld "
                                     "tokens to %s%s\n", calib_rows, calib_dir(),
                             (a && bb && ce) ? "" : ", with errors");
        }
    }

#if QWEN4EXP_EXL3_SHARED_LANE
    if (sh_lane) (void)rad_lane_join(c_, 1, 0);   /* the arm on lane 1 joins back before the gather reads it */
#endif
#undef QX_SH_L0
    /* THE GATHER LAST, because with the fold it reads the arm. Its operands are the routed slots
     * plus `sout` and `sgate`, and it writes the block's delta straight into `h.x`. */
    if (!gather_taken(g, T, gather_out_rows, gather_out_rows6))
        RAD_ISSUE_N(c_, op_gather, rows, brows(w.edn, rows * k), brow_slice(w.ew, r0, rows, k),
                    brows(gather_inv ? w.inv : w.sorted, rows * k),
                    gfold ? brow_slice(w.sout, r0, rows, g.n_embd) : RAD_NONE,
                    gfold ? brow_slice(w.sgate, r0, rows, 1) : RAD_NONE,
                    brow_slice(w.h.x, r0, rows, g.n_embd));

    /* A write that takes the all-reduce does it once for the whole step, deciding on T, so this
     * pass must ask with T, not its own rows. */
    if (own_ar && issues_ar(T, rows)) issue_ar(c_, r0, rows, false);
    if (op_add)
        RAD_ISSUE_N(c_, op_add, rows, brow_slice(w.x, r0, rows, g.n_embd),
                    brow_slice(w.h.x, r0, rows, g.n_embd),
                    brow_slice(w.x, r0, rows, g.n_embd));
}

/* One pass of the per-expert tap. The scatter's expert offsets are copied back first (they decide
 * the launches); then the scratch is zeroed, each expert's run of the `slots` sorted rows is summed
 * into its square, and the scratch is copied back and added on the host. Both syncs are on the
 * block's stream with nothing issued after the tap, so the transient scratch never overlaps a live
 * buffer's use. */
inline void MoeEXL3::calib_experts_pass(RadCtx* c_, int64_t slots) const {
    if (calib_dne_failed) return;
    const int64_t ne = c.n_expert, K = c.n_ff_exp;
    const RadStream s = rad_stream(c_);
    if (calib_dne.ne == 0) calib_dne.init(ne, K);

    std::vector<int32_t> off((size_t)(ne + 1));
    rad_memcpy_async(off.data(), rad_buf_ptr(c_, w.eoff), (ne + 1) * 4, s);
    rad_stream_sync(s);
    if (off[0] != 0 || off[(size_t)ne] < 0 || off[(size_t)ne] > slots) {
        std::fprintf(stderr, "radiance: layer %d's expert offsets end at %d of %lld sorted rows; "
                             "the per-expert calibration stops\n",
                     layer, off[(size_t)ne], (long long)slots);
        calib_dne_failed = true;
        return;
    }

    const int64_t bytes = ne * K * K * 4;
    float* staging = calib_staging(c_, bytes);
    if (!staging) { calib_dne_failed = true; return; }
    void* scratch = rad_buf_ptr(c_, b_gram_dne);
    rad_memset_async(scratch, 0, bytes, s);
    for (int64_t e = 0; e < ne; ++e) {
        const int64_t r0 = off[(size_t)e], n = off[(size_t)e + 1] - r0;
        if (n <= 0) continue;
        RAD_ISSUE_N(c_, op_gram_dne, n, brow_slice(b_calib_eff, r0, n, K), RAD_NONE,
                    brow_slice(b_gram_dne, e * K, K, K));
        calib_dne.rows[(size_t)e] += n;
    }
    rad_memcpy_async(staging, scratch, bytes, s);
    rad_stream_sync(s);
    calib_dne.add(staging);
}

inline void MoeEXL3::step(RadCtx* c_, const RadBatch* batch) const {
    step_rows(c_, batch->n_tok);
}

inline void MoeEXL3::step_rows(RadCtx* c_, int64_t T) const {
    for (int64_t r0 = 0; r0 < T; r0 += c.rows) {
        const int64_t n = (T - r0 < c.rows) ? (T - r0) : c.rows;
        pass(c_, T, r0, n);
    }
    route_report(c_);
}

inline void MoeEXL3::route_report(RadCtx* c_) const {
    /* ---- Hand the routing to the heat engine. `rad_route_report` copies the histogram back
     * asynchronously and the engine consumes it next step, so the compute stream is not blocked
     * (spec §5.5). Without it, expert-tiered placement has nothing to promote on.
     *
     * It reports the last pass, which is the whole step by default (`c.rows` is the prefill
     * chunk). With more passes it is a sample, adequate for a popularity estimate.
     *
     * `expert_w` stays null: RadRouting declares it `const float*` but these weights are bf16
     * (f32 under the sigmoid router). Nothing in the core reads it (the heat engine uses
     * `expert_count`). */
    moe_debug_weights(c_, layer, c.top_k, w.ids, w.ew, c.sigmoid_router != 0);
    RadRouting r{};
    r.expert_ids    = (const int32_t*)rad_buf_ptr(c_, w.ids);
    r.expert_w      = nullptr;
    const int32_t* const counts = rad_route_counts(c_, layer, c.n_expert);
    r.expert_count  = counts ? counts : (const int32_t*)rad_buf_ptr(c_, w.ecnt);
    r.sorted_tok    = (const int32_t*)rad_buf_ptr(c_, w.sorted);
    r.expert_offset = (const int32_t*)rad_buf_ptr(c_, w.eoff);
    r.top_k         = c.top_k;
    r.n_expert      = c.n_expert;
    rad_route_report(c_, layer, &r);
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_MOE_EXL3_H */
