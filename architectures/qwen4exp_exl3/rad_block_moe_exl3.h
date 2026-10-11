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
 *   - THE EXPERTS ARE ONE OPERAND EACH. An EXL3 weight is a trellis `codes` plane and its sign
 *     vectors (`suh`, `suh.1` for gate_up's up half, `svh`), and the two ops take each expert as
 *     ONE weight selecting every plane -- a declaration with no `planes`, at the codes' dtype, the
 *     way this tree declares its lm_head. A reference library can decode a whole-plane operand
 *     through the quantiser's decode hook; it cannot decode a partial selection of a scheme the
 *     core does not define (PLAN D6). So the eight runs become four: gate_up and down, each for
 *     the even experts and then the odd ones.
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
 * ============================== THE SHARED EXPERT IS NOT OPTIONAL HERE ==============================
 *
 * Qwen3.5-MoE runs a dense expert BESIDE the routed ones on every token, gated by a one-output
 * projection: `y = routed(x) + sigmoid(shared_gate(x)) * shared(x)`. It is `mlp.shared_expert.*`
 * and `mlp.shared_expert_gate.weight` in the checkpoint, and a block that dropped it would produce
 * a plausible distribution and the wrong model. `moe_gather` does the multiply as it sums the
 * routed slots -- see docs/OPS.md's `scale_rows` for why `mul` cannot do it, and note that the
 * gate is applied to the SCALAR before the multiply, not to the product.
 *
 * A family without one passes `Src::shared` null and the whole arm disappears, buffers included.
 *
 * ============================== WHY THE BLOCK SLICES ITS ROWS ==============================
 *
 * The expert path's intermediates are `top_k` times as tall as the step: at an 8192-token chunk and
 * top_k 8 the down-projection's input alone is 65536 rows, which at n_embd 2048 is 256 MiB of
 * transient for ONE buffer. So the block runs in passes of at most `Config::rows` tokens and its
 * buffers are declared at that height. Routing is per pass and has to be -- `expert_offset` and
 * `sorted_tok` describe the rows of one pass -- which is not an approximation: a token's routing
 * depends on that token alone, so a pass computes exactly the rows it names.
 *
 * THE DEFAULT IS 2048 AND THAT IS THE CHUNK, not a tuning choice: core/sched/geometry.cpp clamps
 * the prefill chunk to the linear-state checkpoint interval, which on this family is 2048, so the
 * shipped configuration is one pass and the slicing is inert. It exists so that a deployment with a
 * larger chunk does not need this file edited.
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
 * whose second entry is a down projection. The fp8 block's four-class form at four ranks
 * (`moe_gemm_q_mod4`) has no EXL3 op, and is refused by name (declare).
 *
 * Each run's entry is the WHOLE stored weight, every plane of it, so an expert holds two logical
 * weights, gate_up and down, and is its own movement unit (`grp_expert_slot`, slots 0 and 1) and
 * RAD_ACCESS_CONDITIONAL, which is what lets the placement planner tier them and the heat engine
 * promote the hot ones.
 */
#ifndef RAD_BLOCK_MOE_EXL3_H
#define RAD_BLOCK_MOE_EXL3_H

#include <arch/rad_fp8.h>
#include <rad_device.h>

#include "../../include/exl3_abi.h"
static_assert(EXL3_IFACE_VERSION == 2, "this block issues interface v2: one table a plane a parity class");

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace rad {
namespace arch {

/* THE MOST ROUTING SLOTS THE SORT SERVES IN ONE WORKGROUP, which is what bounds
 * `router_topk_scatter`. It is libr4d's `kScatterSmall` and it is stated here rather than asked
 * of the plugin for the reason every other such bound in this library is: the architecture has to
 * decide, at declare time and identically on both ranks, whether the op exists at all. */
static const int64_t kMoeSortRows = 1024;

/* The skinny bf16 GEMM's decode band, which is where the fused op's GEMM is that GEMM's own
 * arithmetic; past it the router runs the tiled kernel and the pair is issued as before. */
static const int64_t kMoeRouterFusedRows = 64;

/* ============================== THE EXL3 CONTRACT ==============================
 *
 * The encoding is include/exl3_encoding.h's (owner Q, PLAN D4) and the two ops' schemas, operand
 * positions and semantics include/exl3_abi.h's (owner R, interface v2, PLAN D15); this block reads
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

/* The planes each op takes, one table a plane a parity class, in the schema's order (exl3_abi.h
 * "THE DECLARATION"), with the operand slot each sits in (EXL3_P_*). The roles are the encoding's
 * (exl3_encoding.h): `codes` the trellis symbols, `suh` (and gate_up's `suh.1`, the up half's) one
 * f16 a column, `svh` one f16 a row. */
constexpr int kPlanesGateUp = 4, kPlanesDown = 3;
constexpr const char* kRoleGateUp[kPlanesGateUp] = { "codes", "suh", "suh.1", "svh" };
constexpr const char* kRoleDown[kPlanesDown]     = { "codes", "suh", "svh" };
constexpr int kSlotGateUp[kPlanesGateUp] = { EXL3_P_CODES, EXL3_P_SUH, EXL3_P_SUH1, EXL3_P_SVH };
constexpr int kSlotDown[kPlanesDown]     = { EXL3_P_CODES, EXL3_P_SUH, EXL3_P_SVH };

/* TEST-ONLY WIRING MUTATIONS, 0 in every real build: tests/arch/CMakeLists.txt builds two plugins
 * with 1 and 2, never installed, as the end-to-end differential's controls -- defects that test
 * must see. They live in the DECLARATION, consistently with the issue, because the engine refuses
 * an issue naming another table run than the one declared, by name, before anything is computed
 * (measured: "a table's run is fixed by the declaration") -- so a defect there is caught already,
 * and a numerical control has to be one the engine cannot see. 1: the parity classes' tables
 * swapped (every expert reads its neighbour's weights). 2: gate_up's `suh` view selects plane
 * `suh.1` and the `suh.1` view `suh` (gate and up read each other's input scales). */
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

}  /* namespace exl3 */



/* ---- RADIANCE_DEBUG_ROUTING: the routing WEIGHTS of token 0, which nothing else reports.
 *
 * The core's histogram (core/runtime/ctx.cpp) says which experts a step chose and rad_arch.h's
 * dbg_resid says how big each arm's contribution was; neither shows the weights themselves, and a
 * router whose top-k sums to 0.4 instead of 1 is a block that is right in every other respect and
 * scaled down by more than half. It SYNCHRONISES THE STREAM and is off unless the variable is set:
 * a layer number, or negative for every layer. */
inline void moe_debug_weights(RadCtx* c_, int layer, int64_t top_k, rad_buf ids, rad_buf ew) {
    static const char* dbg = std::getenv("RADIANCE_DEBUG_ROUTING");
    if (!dbg || !ids || !ew) return;
    const int want = (int)std::strtol(dbg, nullptr, 10);
    if (want >= 0 && layer != want) return;
    static int seen[256] = {0};
    if (seen[layer & 255]++) return;

    void* pi = rad_buf_ptr(c_, ids);
    void* pw = rad_buf_ptr(c_, ew);
    if (!pi || !pw) return;
    RadStream s = rad_stream(c_);
    std::vector<int32_t>  vi((size_t)top_k);
    std::vector<uint16_t> vw((size_t)top_k);
    rad_memcpy_async(vi.data(), pi, top_k * 4, s);
    rad_memcpy_async(vw.data(), pw, top_k * 2, s);
    rad_stream_sync(s);

    std::string line;
    double sum = 0.0;
    for (int64_t r = 0; r < top_k; ++r) {
        const uint32_t u = (uint32_t)vw[(size_t)r] << 16;
        float f;
        std::memcpy(&f, &u, 4);
        sum += f;
        char b[48];
        std::snprintf(b, sizeof b, "%s%d:%.4f", r ? " " : "", vi[(size_t)r], f);
        line += b;
    }
    std::fprintf(stderr, "D moe layer %d token 0 top_k = %s  (sum %.4f)\n",
                 layer, line.c_str(), sum);
}

/* THE MOST EXPERTS A LAYER PROTECTS ON ONE RANK (MoeEXL3::prot): libr4d's sort takes at most eight
 * in its `protect` list. */
constexpr int64_t kMaxProtected = 8;

struct MoeEXL3 {
    struct Config {
        /* ROUTED EXPERTS THIS RANK OWNS. Equal to `n_expert_all` in every deployment that does not
         * use expert parallelism -- every rank then holds a slice of every expert (see ff_lo). A
         * calibration run is the one that splits the experts themselves, because a Hessian is
         * over an expert's whole K. */
        int64_t n_expert     = 0;
        /* THE MODEL'S expert count -- what the ROUTER is wide in and what top_k picks from. 0 means
         * "the same as n_expert", the whole-model case.
         *
         * The router stays replicated under expert parallelism and that is the property the scheme
         * rests on: every rank scores every expert from the same residual stream and picks the
         * same top_k, so the ranks agree on the routing without a collective. Each then sorts only
         * the slots that landed in ITS range and the block's end-of-block all-reduce sums the
         * partials. Sharding the router instead would have two ranks disagree about where a token
         * went, and the all-reduce would sum two different models. */
        int64_t n_expert_all = 0;
        /* This rank's first expert id, in the model's numbering. Straight through to moe_scatter,
         * which subtracts it so the offsets, the counts and the weight table are all local. */
        int64_t expert_base  = 0;
        int64_t top_k        = 0;
        int64_t n_ff_exp     = 0;   /* one expert's intermediate width, PER RANK */
        /* THE ROUTED EXPERTS' SLICE ON THIS RANK WHEN THEIR WIDTH DOES NOT HALVE INTO WHOLE
         * 128-COLUMN GROUPS (exl3::kCut): expert e's intermediate columns [ff_lo[e % 2], ff_hi[e % 2]) in the model's
         * numbering, the slice depending on the expert's PARITY. hi 0 is the even split (or no
         * split). `n_ff_exp` is then the wider slice, which is what the buffers between the two
         * grouped GEMMs are.
         *
         * WHY BY PARITY. This family's expert is 640 columns, five 128-column blocks, so two ranks
         * split each expert three blocks and two. Giving every expert's three blocks to one rank
         * would make that rank read 3/5 of every routed expert; alternating which rank takes three
         * by the expert's parity makes the ranks equal on average, and unequal only by the
         * difference between how many even and odd experts a step routes -- a fifth of an expert
         * per unit of that difference, where splitting the EXPERTS between the ranks (expert
         * parallelism) is unequal by whole experts, the binomial spread of which rank the routed
         * experts land on. The block's all-reduce turns that spread into the faster rank waiting,
         * about a millisecond a step at one sequence.
         *
         * Every rank computes its slice of every routed expert: its gate/up rows and its K slice of
         * down, and the all-reduce sums the partials -- the routing weight multiplies each partial,
         * and the sum of the partials is the expert's output. Rounding moves (two partial sums
         * where there was one), so this is checked by agreement rather than byte for byte. */
        int64_t ff_lo[2] = { 0, 0 }, ff_hi[2] = { 0, 0 };
        /* OR BY THE EXPERT'S NUMBER MOD 4, where four ranks split it: expert e's columns
         * [ff_lo4[e % 4], ff_hi4[e % 4]), served by `moe_gemm_q_mod4`, and taking precedence over
         * the parity slices. hi 0 offers none.
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
         * KEPT AS A FIELD AND REFUSED IN declare(): the architecture sets it at four ranks exactly
         * as the in-tree one does, and the EXL3 ops have only the two parity tables. */
        int64_t ff_lo4[4] = { 0, 0, 0, 0 }, ff_hi4[4] = { 0, 0, 0, 0 };
        int64_t n_ff_shared  = 0;   /* the shared expert's, per rank; 0 = no shared expert */
        /* THE SHARED ARM'S SLICE ON THIS RANK, when its width does not halve into whole fp8 scale
         * blocks: [shared_lo, shared_hi) of its columns in the model's numbering, with
         * `n_ff_shared` then the slice's width. hi 0 is the even split (or no split at world 1);
         * an empty slice declares no arm on this rank.
         *
         * WHY A SLICE AND NOT ONE RANK. This family's shared expert is 640 columns, five
         * 128-column blocks, so two ranks cannot halve it -- and computing it on one rank alone
         * makes the ranks unequal at every routed layer, which the block's all-reduce turns into
         * the other rank waiting: about 20 us a layer at one sequence, which costs far more than
         * the arm's own bytes. Slicing it three blocks and two puts both ranks within one block of
         * each other: each computes its columns of the gate/up product, its own K slice of the
         * down projection, scales that partial by the replicated gate, and the all-reduce sums the
         * partials -- `sgate * (down_0 + down_1)` is the arm, since the gate is a per-token
         * scalar. Rounding moves (two partial sums where there was one), so this is checked
         * against the unsplit arm by agreement rather than byte for byte.
         *
         * The arm's stages run behind the routed ones on the same queue (see pass()): a routed
         * GEMM's tail leaves compute units idle, and the arm fills them. */
        int64_t shared_lo = 0, shared_hi = 0;
        int     norm_topk    = 1;   /* renormalise the kept routing weights to sum to one */
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
        /* How many experts from `probed_expert` on probe() mapped: it reads past experts a
         * quantised layer keeps plain bf16 (see `prot`) to the first one in the layer's format. */
        int64_t probed_count = 1;
        /* THE GPTQ CALIBRATION TAP. On, this block declares an f32 [K, K] PERSIST accumulator for
         * each of its two expert projections and a `gram_accum` into it, running until
         * RADIANCE_CALIB_TOKENS tokens have passed and then writing the pair to
         * RADIANCE_CALIB_DIR and switching itself off.
         *
         * IT IS A SEPARATE RUN AND NOT A SERVING MODE. The accumulators are 1.34 GB across 48
         * layers at this model's widths, the dump synchronises the stream, and the two extra
         * launches a layer are pure overhead once the corpus is in. Nothing reads the files at
         * serve time -- rad-convert does, in another process, to pack the next container.
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

    /* WHICH BUFFERS ARE A PASS TALL AND WHICH ARE THE STEP'S, and the division is the whole of what
     * the slicing costs in bookkeeping. Only the four planes that carry one row per (token, slot)
     * are a pass tall -- they are `top_k` times the step and they are what the slicing exists to
     * bound. Everything else is one row per TOKEN, is a few megabytes at any chunk, and is sized
     * at max_tok and sliced by the pass offset like every other activation in this tree.
     *
     * That is not a size optimisation, it is what keeps the pass offset expressible: LinearFP8 and
     * GateQuantFP8 slice their INPUT and their OUTPUT with the same `r0`, so a shared expert whose
     * output buffer were a pass tall would be written past its end on the second pass. The shipped
     * chunk is one pass, so that overrun would show no symptom there. */
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
        rad_buf ew      = 0;   /* [max_tok, top_k]      bf16 */
        rad_buf eoff    = 0;   /* [n_expert + 1]        i32 */
        rad_buf ecnt    = 0;   /* [n_expert]            i32 */
        /* the expert planes -- one row per (token, slot), so a PASS's height */
        rad_buf sorted  = 0;   /* [rows * top_k]               i32, pass-relative slot indices */
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

    /* The checkpoint side. `experts` is a PREFIX -- "....mlp.experts" -- and the per-expert names
     * are built from it, because a caller cannot pass 256 of them and a format string with the
     * expert in it is a convention this file would have to parse. `shared` is the same for the
     * dense expert beside them; null means the family has none, and `shared_gate` must then be
     * null too.
     *
     * THE CONFIG TRAVELS IN HERE rather than as an argument to declare, and that is not tidiness:
     * it makes this block's declare() arity IDENTICAL to MlpFP8's, which is what lets MtpBlockFP8
     * be a template over the two. A routed feed-forward and a dense one are the same thing in the
     * same position of the same graph, and the only component that should know the difference is
     * the plugin that picks one. */
    struct Src {
        Config      cfg{};
        const char* norm        = nullptr;
        const char* router      = nullptr;   /* mlp.gate.weight, [n_expert, n_embd] bf16 */
        const char* experts     = nullptr;   /* prefix; ".<e>.{gate,up,down}_proj.weight" */
        const char* shared      = nullptr;   /* prefix; ".{gate,up,down}_proj.weight" */
        /* A STACKED CHECKPOINT NAMES TWO TENSORS AND NOT A PREFIX, which is the shape every
         * recent release uses: `mlp.experts.gate_up_proj` [n_expert, 2*n_ff, n_embd] and
         * `mlp.experts.down_proj` [n_expert, n_embd, n_ff], one tensor holding every expert.
         * Set these INSTEAD of `experts` and each declared expert maps to a dim-0 slice
         * (rad_arch.h's map_slice; core/format/checkpoint.cpp takes the slice as a view, and a
         * block-fp8 checkpoint's stacked `_scale_inv` plane at the same index).
         * Note that a stacked gate_up is ALREADY FUSED in the order this block wants -- gate
         * rows first -- so there is no concatenation left to do. */
        const char* experts_gate_up = nullptr;
        const char* experts_down    = nullptr;
        const char* shared_gate = nullptr;   /* mlp.shared_expert_gate.weight, [1, n_embd] */
    };

    Geom   g{};
    Config c{};
    Wire   w{};
    int    layer = 0;

    rad_weight w_norm = 0, w_router = 0, w_sgate = 0;
    /* The input was normed and quantised by the caller; this block declares no norm. */
    bool       ext_in = false;
    /* The runs: one a plane, each split by parity, indexed [plane][place in the sort]. Only the
     * FIRST handle of a class and the count reach an issue (RAD_WTAB); the rest are held so that a
     * declaration failure can be reported against the expert it happened on. */
    std::vector<rad_weight> w_gu[exl3::kPlanesGateUp], w_dn[exl3::kPlanesDown];

    NormQuantFP8 nq_h{};
    GateQuantFP8 gq_sh{};
    LinearFP8    sh_gate_up{}, sh_down{};
    rad_op op_router = 0, op_topk = 0, op_scatter = 0;
    /* `router_topk_scatter`, if it resolves: the two above in one launch. 0 leaves the pair, which
     * is declared either way -- the fused form is bounded by the sort's single-workgroup shape and
     * a prefill chunk is past it. */
    rad_op  op_topk_scatter = 0;
    int64_t topk_scatter_rows = 0;
    /* `router_gemm_topk_scatter`, if it resolves: the router GEMM in front of the pair above, up
     * to `router_fused_rows` tokens, and with `router_fused_gate` the shared expert's gate
     * (`op_sgate`'s work) as well. 0 -- the op did not resolve -- issues the router and the pair
     * as always, and so does a pass of more rows. */
    rad_op  op_router_fused = 0;
    int64_t router_fused_rows = 0;
    bool    router_fused_gate = false;
    /* Set by rad_block_mtp_hc.h from the connection in front of the block: whether that read
     * writes the rotated E4M3 pair. Always false for EXL3 experts (Config::expert_rot), and
     * declare() refuses true rather than ignore a pair it would never read. */
    bool   hr_in = false;
    rad_op op_gu = 0, op_dn = 0, op_gather = 0;
    /* THE PROTECTED EXPERTS (declare): local experts a quantised layer keeps plain bf16 -- the few
     * whose output dwarfs the rest's (docs/MOE-W4.md). The scatter sorts them last (`protect`,
     * `prot_list`), the quantised GEMMs above serve the other `n_reg`, and a bf16 grouped GEMM
     * pair serves these over the trailing run of the offsets: gate_up into `b_pgu`, the gate into
     * `pff`, and down into the same `edn` rows the quantised down left zero. */
    std::vector<int64_t>    prot;
    const char*             prot_list = nullptr;
    int64_t                 n_reg = 0;
    /* THIS LAYER'S gate_up AND down ENCODINGS -- K, so the op dtype ("exl3k4"...), and the
     * codebook ("mul1"...) -- resolved by declare() from the layer's own experts: each projection
     * of each layer may carry its own, and every expert of one table must carry the same. */
    exl3::Enc               eenc[2]{};
    std::vector<rad_weight> w_pgu, w_pdn;
    rad_buf                 b_pgu = 0;
    ActFP8                  pff{};
    GateQuantFP8            gq_prot{};
    rad_op                  op_pgu = 0, op_pdn = 0;
    /* The gather FOLDS the shared arm in wherever there is one: it takes `sout` and `sgate` and
     * writes the block's delta straight into `h.x`, so `w.moe` is never written. */
    bool   gfold = false;
    /* THE WRITE BEHIND THIS BLOCK TAKES THE GATHER up to this many tokens -- HyperConn's
     * gather_rows and gather_rows6, one a wire, set by the architecture after that write declared
     * its fused forms, 0 where it did not. Where gather_taken() says so the block leaves its routed
     * rows unsummed. */
    int64_t gather_out_rows = 0;
    int64_t gather_out_rows6 = 0;
    /* The calibration tap: two accumulators and the two ops that fill them. Zero unless
     * `Config::calib` is set. `calib_rows` and `calib_done` are mutable because `pass` is const
     * and the tap's whole state is a token count -- per block, so a block reaches its target on
     * the same pass every other block does and nothing has to know which layer is last. */
    rad_buf b_gram_gu = 0, b_gram_dn = 0;
    rad_op  op_gram_gu = 0, op_gram_dn = 0;
    /* The down tap's bf16 copy of `w.eff` and the cast that writes it (Config::calib says why). */
    rad_buf b_calib_eff = 0;
    rad_op  op_calib_cast = 0;
    /* The domain the tap sums in, which its files say (calib_write): always the model's own (0)
     * here -- see Config::calib. */
    uint32_t gram_domain = 0;
    /* THE FILE IS NAMED AFTER THE WEIGHT IT WILL BE USED TO PACK, minus the expert index, and
     * that is the whole protocol between this and the GPTQ quantiser: the recipe hands it the
     * directory (`calib=`) and rad-convert the weight's logical name, and libquant strips
     * `.<e>.weight` off the name to get the file. A name of this tap's own -- "blk3.gate_up" --
     * would have made the quantiser carry a table mapping one to the other, for every
     * architecture. */
    const char* gram_gu_stem = nullptr;
    const char* gram_dn_stem = nullptr;
    mutable long long calib_rows = 0;
    mutable bool      calib_done = false;
    /* THE DOWN PROJECTION'S HESSIAN AN EXPERT (rad_arch.h, CalibExperts), beside the pooled one:
     * a layer-sized scratch, `ne` squares of [n_ff_exp, n_ff_exp], a `gram_accum` into each
     * expert's square over that expert's rows, and the sum kept on the host. */
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
    /* One pass of at most `c.rows` tokens. */
    void pass(RadCtx* c, int64_t T, int64_t r0, int64_t rows) const;
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
        /* THE SLICE INDEX IS GLOBAL: the checkpoint holds every expert, and this rank declares a
         * window into it. Slicing by the local index would give every rank experts 0..E/W. */
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
    /* The caller names the expert, in the model's numbering: the first one this rank will own if
     * the experts are split between the ranks, which declare() then skips -- and an expert every
     * rank declares a slice of otherwise, which every rank skips. */
    /* A PLAIN bf16 EXPERT DOES NOT SETTLE IT: a quantised layer keeps up to kMaxProtected of its
     * experts plain, so the probe reads on past them to the first expert in another format, and
     * only a run of plain ones longer than any protected set says the model is bf16 -- which is
     * refused here: a bf16 container is the in-tree plugin's. */
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
                        "qwen4exp_fp8 serves (quantisation \"\"), not this EXL3 plugin\n", l, g0);
    else
        fprintf(stderr, "radiance: '%s' is %s, and this plugin serves the routed experts as EXL3 -- "
                        "scheme `%s` as the exl3 quantiser writes it, K 2..8 -- beside plain bf16 "
                        "protected ones. Convert them with the exl3 quantiser.\n", guw, en,
                        EXL3_SCHEME);
    return RAD_E_DTYPE;
}

inline int MoeEXL3::declare(RadBuilder* b, Names& nm, const Geom& geom, int l,
                           const Wire& wire, const Src& src, bool fold, bool add_out,
                           int ar_in, bool ar_out_) {
    g = geom; c = src.cfg; w = wire; layer = l;
    ar_out = ar_out_;
    /* 0 means "not expert-parallel": this rank owns the model's whole expert set. Normalised once,
     * here, so nothing below has to ask which case it is in. */
    if (c.n_expert_all <= 0) c.n_expert_all = c.n_expert;
    if (c.n_expert <= 0 || c.top_k <= 0 || c.top_k > c.n_expert_all || c.n_ff_exp <= 0)
        return RAD_E_INVAL;
    /* THE SLICE HAS TO BE INSIDE THE MODEL. A base past the end silently routes nothing to this
     * rank -- a model that serves and drops a fraction of its own feed-forward. */
    if (c.expert_base < 0 || c.expert_base + c.n_expert > c.n_expert_all) return RAD_E_INVAL;
    if (c.rows <= 0) c.rows = g.max_tok;
    /* A rank whose slice of the shared arm is empty (see Config) declares no arm, and the
     * all-reduce carries the others' share. */
    const bool shared = c.n_ff_shared > 0 && src.shared;
    /* Whether `w.sgate` is out of the sharing pool -- set where the shared arm says so, and what
     * the fused router's gate write relies on (see there). */
    bool sgate_concurrent = false;

    /* THE INPUT MAY BE PREPARED BY THE ARCHITECTURE, and `src.norm == nullptr` is how it says so.
     *
     * A pre-norm model's block owns its own norm: one op that adds the previous block's delta into
     * the residual stream, normalises it and quantises it (rad_fp8.h's NormQuantFP8). A GATED
     * RESIDUAL model's does not -- arch/common/rad_block_hc.h's `hc_read` produces the block input
     * from a stream four times as wide, with a data-dependent mix the block has no way to compute,
     * and there is nothing left for a norm here to do. So this block takes `wire.h` as ALREADY
     * normed and quantised and declares no norm weight at all.
     *
     * `ar_in` is refused in that mode rather than ignored: the fused all-reduce lives INSIDE the
     * norm, so with no norm there is nowhere to fold it and a caller that asked would silently get
     * an unreduced input. It emits a standalone collective instead (`add_out`/`ar_out`). */
    ext_in = (src.norm == nullptr);
    if (ext_in) {
        if (!w.h.x || (!g.w_bf16 && (!w.h.q || !w.h.s)) || ar_in != kArNone) return RAD_E_INVAL;
    } else {
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_norm.weight", l), src.norm));
        w_norm = decl_w(b, nm.f("blk.%d.ffn_norm.weight", l), RAD_F32, {g.n_embd},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    }

    /* ---- THE EXPERTS' NAME MAPS, and from them which experts this layer PROTECTS. The maps are
     * how the model is asked what each expert is (rad_weight_encoding): one a LOGICAL weight,
     * gate_up and down, the scale views having none; the probed experts' gate_up is mapped
     * already. Every expert is the format the block was set up for -- one table of experts takes
     * one encoding -- except that a quantised layer may keep a few WHOLE experts plain bf16: the
     * ones whose output dwarfs the rest's, where the same relative rounding error is a large
     * absolute one (docs/MOE-W4.md). Those are sorted last and served by a bf16 GEMM pair of
     * their own; they are found here, before the sort is declared, because the sort takes them. */
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
        /* The quantised GEMM takes its experts as two equal tables by parity -- at four classes
         * (Config::ff_lo4) as four tables, the first n_reg % 4 one expert longer, which any count
         * gives. */
        if (!(c.ff_hi[0] > 0 && c.ff_hi4[0] > 0) && n_reg % 2) {
            fprintf(stderr, "radiance: layer %d keeps %zu experts plain bf16 on this rank, which "
                            "leaves %lld quantised; they come in pairs on a rank so the rest split "
                            "into two equal tables by parity\n", l, prot.size(), (long long)n_reg);
            return RAD_E_INVAL;
        }
        /* The tap's per-expert files are named by expert, and the sort here numbers by place. */
        if (c.calib) {
            fprintf(stderr, "radiance: calibrate the bf16 container: layer %d of this one keeps "
                            "experts plain bf16 and sorts them out of order\n", l);
            return RAD_E_INVAL;
        }
        std::string lst;
        for (int64_t e : prot) lst += (lst.empty() ? "" : ",") + std::to_string(e);
        prot_list = nm.ckpt("%s", lst.c_str());
    }

    /* THE ROUTER IS REPLICATED, NOT SHARDED, and that is the property the whole block rests on.
     * Every rank routes every token to the same experts -- the routing is computed from the full
     * residual stream, which every rank holds -- so `sorted_tok` and `expert_offset` are identical
     * across ranks and the expert GEMMs' row layout agrees without a collective. Sharding the
     * router would make two ranks disagree about which expert a token went to, and the all-reduce
     * at the end of the block would then sum two different models. */
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_gate_inp.weight", l), src.router));
    w_router = decl_w(b, nm.f("blk.%d.ffn_gate_inp.weight", l), RAD_BF16,
                      {c.n_expert_all, g.n_embd},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

    if (!ext_in)
        RAD_ARCH_TRY(nq_h.declare(b, g, w_norm, fold ? w.h.x : w.x, w.h, g.n_embd,
                                  fold ? w.x : 0, 0, ar_in));

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
    /* THE PROTECTED EXPERTS ARE SORTED LAST (`protect`), and only a layer that has them says so,
     * so every other layer's sort is declared without a protect list. */
    if (!prot_list)
        op_scatter = rw(b, RAD_OP(b, "moe_scatter",
                           RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n_expert", c.n_expert),
                                      RAD_INT("top_k", c.top_k),
                                      RAD_INT("expert_base", c.expert_base)),
                           RAD_NOWEIGHTS),
                       {w.ids}, {w.sorted, w.eoff, w.ecnt});
    else
        op_scatter = rw(b, RAD_OP(b, "moe_scatter",
                           RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n_expert", c.n_expert),
                                      RAD_INT("top_k", c.top_k),
                                      RAD_INT("expert_base", c.expert_base),
                                      RAD_STR("protect", prot_list)),
                           RAD_NOWEIGHTS),
                       {w.ids}, {w.sorted, w.eoff, w.ecnt});

    /* THE TWO ABOVE IN ONE LAUNCH, and it is worth asking for because NEITHER IS WORK. At a decode
     * shape each is a single 256-thread workgroup ranking about a kilobyte, so each costs a full
     * dispatch for no arithmetic, once per layer per step, twice over. The sort reads
     * `expert_ids` and nothing else and the selection writes it, so the launch between them buys a
     * barrier and nothing more.
     *
     * BOUNDED BY THE SORT'S SINGLE-WORKGROUP FORM. `moe_scatter` runs one launch at
     * M*top_k <= 1024 and three above it -- the three exist to keep the sort stable without an
     * atomic deciding the order -- and only the first shape is fusible. That is the DECODE band by
     * construction (eight sequences at depth 3 and top_k 10 is 320) and a prefill chunk is past
     * it, which is also where the pair is amortised over real work and the fusion would buy
     * nothing. Declared either way, so the graph and the buffer plan do not depend on it. */
    if (c.top_k > 0 && (int64_t)c.rows * c.top_k <= kMoeSortRows)
        topk_scatter_rows = c.rows;
    else
        topk_scatter_rows = kMoeSortRows / c.top_k;
    if (topk_scatter_rows > 0 && !prot_list)
        op_topk_scatter = rw(b, RAD_OP(b, "router_topk_scatter",
                                RAD_PARAMS(RAD_RANGE("M", 1, topk_scatter_rows),
                                           RAD_INT("n_expert", c.n_expert_all),
                                           RAD_INT("n_local", c.n_expert),
                                           RAD_INT("top_k", c.top_k),
                                           RAD_INT("norm", c.norm_topk),
                                           RAD_STR("dtype", g.dtype),
                                           RAD_INT("expert_base", c.expert_base)),
                                RAD_NOWEIGHTS),
                            {w.logits}, {w.ids, w.ew, w.sorted, w.eoff, w.ecnt});
    else if (topk_scatter_rows > 0)
        op_topk_scatter = rw(b, RAD_OP(b, "router_topk_scatter",
                                RAD_PARAMS(RAD_RANGE("M", 1, topk_scatter_rows),
                                           RAD_INT("n_expert", c.n_expert_all),
                                           RAD_INT("n_local", c.n_expert),
                                           RAD_INT("top_k", c.top_k),
                                           RAD_INT("norm", c.norm_topk),
                                           RAD_STR("dtype", g.dtype),
                                           RAD_INT("expert_base", c.expert_base),
                                           RAD_STR("protect", prot_list)),
                                RAD_NOWEIGHTS),
                            {w.logits}, {w.ids, w.ew, w.sorted, w.eoff, w.ecnt});

    /* EITHER a prefix or the two stacked tensors, never both and never neither. A checkpoint
     * cannot be read two ways at once and a caller that set both has not decided. */
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
    /* Each parity's width on this rank: the Config's slice when the width is split unevenly, and
     * `n_ff_exp` for both otherwise. The buffer between the two ops is the wider one's.
     *
     * THE FOUR-CLASS SPLIT IS REFUSED BY NAME. At four ranks the architecture rotates an expert's
     * extra 128-column group over four classes by number mod 4 (Config::ff_lo4) -- the fp8 block's
     * `moe_gemm_q_mod4` -- and the EXL3 ops take the two parity tables only. */
    if (c.ff_hi4[0] > 0) {
        fprintf(stderr, "radiance: layer %d: four ranks split each expert's width four ways by its "
                        "number mod 4, and %s / %s take two parity tables only; serve EXL3 experts "
                        "at one, two or three ranks\n", l, exl3::kOpGateUp, exl3::kOpDown);
        return RAD_E_UNSUPPORTED;
    }
    const bool split = c.ff_hi[0] > 0;
    int64_t qlo[2] = { 0, 0 }, qhi[2] = { 0, 0 };
    int64_t wq[2] = { c.n_ff_exp, c.n_ff_exp };
    if (split)
        for (int q = 0; q < 2; ++q) {
            qlo[q] = c.ff_lo[q];
            qhi[q] = c.ff_hi[q];
            wq[q]  = qhi[q] - qlo[q];
        }
    if (split && (wq[0] <= 0 || wq[1] <= 0 || c.n_ff_exp != (wq[0] > wq[1] ? wq[0] : wq[1]))) {
        fprintf(stderr, "radiance: the routed experts' slices [%lld, %lld) and [%lld, %lld) do not "
                        "describe a width of %lld\n", (long long)qlo[0], (long long)qhi[0],
                (long long)qlo[1], (long long)qhi[1], (long long)c.n_ff_exp);
        return RAD_E_INVAL;
    }
    /* EVERY CUT ON A WHOLE 128-COLUMN GROUP (exl3::kCut): the Hadamard is 128 wide on both sides of
     * both projections, and 128 is eight of the trellis's 16x16 tiles, so a cut there keeps every
     * tile and every rotation whole -- gate_up's rows and down's columns are sliced by it. */
    for (int q = 0; q < 2; ++q)
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
     * experts then the odd ones -- so every (plane, class) is one contiguous table.
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
        for (int64_t s = q; s < n_reg; s += 2) {
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
        for (int q = 0; q < 2; ++q) RAD_ARCH_TRY(declare_run(0, p, q));
    for (int p = 0; p < exl3::kPlanesDown; ++p)
        for (int q = 0; q < 2; ++q) RAD_ARCH_TRY(declare_run(1, p, q));

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
     * experts' planes, then the odd experts'. N and K are the WEIGHT's, so gate_up's N is
     * gate and up together. N_odd and K_odd are always given, at one rank too, where they equal N
     * and K: both tables are always issued (P2 notes, finding 5). */
    const auto tables = [&](const std::vector<rad_weight>* planes, int n_planes) {
        std::vector<rad_weight> tab;
        tab.reserve((size_t)(n_reg * n_planes));
        for (int q = 0; q < 2; ++q) {
            const int sq = exl3::kTestMutation == 1 ? 1 - q : q;
            for (int p = 0; p < n_planes; ++p)
                for (int64_t s = sq; s < n_reg; s += 2) tab.push_back(planes[p][(size_t)s]);
        }
        return tab;
    };
    /* gate_up WRITES silu(gate) * up, ITS WIDER CLASS'S WIDTH: a row of the narrower class fills
     * its leading columns and zeros the rest, which its down projection does not read. */
    {
        const std::vector<rad_weight> tab = tables(w_gu, exl3::kPlanesGateUp);
        op_gu = rw(b, rad_decl_op(b, exl3::kOpGateUp,
                          RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("N", 2 * wq[0]),
                                     RAD_INT("K", g.n_embd), RAD_INT("n_expert", n_reg),
                                     RAD_INT("top_k", c.top_k), RAD_INT("N_odd", 2 * wq[1]),
                                     RAD_INT("K_odd", g.n_embd), RAD_STR("dtype", eenc[0].dtype),
                                     RAD_STR("codebook", eenc[0].cb)),
                          tab.data(), (int)tab.size()),
                   {w.h.x, w.sorted, w.eoff}, {w.eff});
    }
    /* down ZEROES every sorted row past the regular experts' offsets -- dropped slots, and the
     * protected experts' rows, which their own pair below then writes. */
    {
        const std::vector<rad_weight> tab = tables(w_dn, exl3::kPlanesDown);
        op_dn = rw(b, rad_decl_op(b, exl3::kOpDown,
                          RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("N", g.n_embd),
                                     RAD_INT("K", wq[0]), RAD_INT("n_expert", n_reg),
                                     RAD_INT("top_k", c.top_k), RAD_INT("N_odd", g.n_embd),
                                     RAD_INT("K_odd", wq[1]), RAD_STR("dtype", eenc[1].dtype),
                                     RAD_STR("codebook", eenc[1].cb)),
                          tab.data(), (int)tab.size()),
                   {w.eff, w.sorted, w.eoff}, {w.edn});
    }
    /* ---- THE PROTECTED EXPERTS' OWN PAIR. Their weights are the layer's, not expert slots: a
     * container's experts in a layer are one fixed-size unit each, and these are four times the
     * bytes -- so they are static, always resident, and joined to the layer's unit with the
     * router. One run of handles a projection, a table of `prot.size()`; the GEMMs are handed the
     * trailing run of the offsets, the rows the quantised down GEMM left zero, and write them.
     *
     * WHERE EVERY RANK HOLDS A SLICE OF EVERY EXPERT, IT HOLDS ONE OF THESE TOO, and an EVEN one:
     * the parity split exists because a 640-wide expert is five 128-wide scale blocks, and a bf16
     * expert has no blocks -- so rank r takes intermediate units [r*w, (r+1)*w) of each, rows of
     * gate and of up and columns of down, and the block's all-reduce sums the partials as it does
     * every other expert's. */
    if (!prot.empty()) {
        const int64_t np = (int64_t)prot.size();
        int64_t wp = c.n_ff_exp;
        int     psh_row = RAD_SHARD_NONE, psh_col = RAD_SHARD_NONE, pparts = 0;
        /* This rank's tile range within one plain expert, in the model's numbering; `phi` 0 is the
         * even split, which needs no span because the loader divides the entry itself. */
        int64_t plo = 0, phi = 0;
        if (split) {
            /* THE EXPERT'S WHOLE WIDTH, not this rank's slice of it: `n_ff_all` is the model's, and
             * where the parity tables end says the same thing at two ranks and nothing else at
             * more -- reading it off `ff_hi` gave a per-rank answer, which happened to be right
             * only when there were two of them. A plain expert has no scale blocks to land on, so
             * the unit here is the GEMM's 64-wide output tile. */
            const int64_t full = g.n_ff_all > 0 ? g.n_ff_all
                                                : (c.ff_hi[0] > c.ff_hi[1] ? c.ff_hi[0]
                                                                          : c.ff_hi[1]);
            /* ONE SLICE A RANK, and it does not have to be the same size on every rank: an uneven
             * share of the 64-wide tiles is still whole tiles, and an uneven share is the only
             * kind available when the width does not divide -- this family's 640 columns are ten
             * tiles, which four ranks take as 3, 3, 2, 2, the first ranks by index taking the
             * extra ones. Every tile has exactly one owner, each rank writes its own columns of
             * gate/up and reads its own of down, and the block's all-reduce sums the partials
             * exactly as it does for the quantised experts and for the shared arm. The even case
             * keeps the even-split declaration verbatim -- no span at all -- so a rank count that
             * divides declares exactly as it did before. */
            psh_row = RAD_SHARD_ROW;
            psh_col = RAD_SHARD_COL;
            pparts  = 2;
            if (full % g.world == 0 && (full / g.world) % 64 == 0) {
                wp = full / g.world;
            } else if (full % 64 || full / 64 < g.world) {
                fprintf(stderr, "radiance: layer %d's plain experts are %lld wide, which %d ranks "
                                "cannot share in 64-wide tiles\n", l, (long long)full, g.world);
                return RAD_E_INVAL;
            } else {
                const int64_t nt = full / 64, base = nt / g.world, extra = nt % g.world;
                const int64_t t0 = (int64_t)g.rank * base + std::min<int64_t>(g.rank, extra);
                const int64_t t1 = t0 + base + ((int64_t)g.rank < extra ? 1 : 0);
                plo = t0 * 64;
                phi = t1 * 64;
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
        /* The gate_up output and the gate's, a sorted row each and in bf16: the rows of the
         * regular experts are never written and never read. */
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
                                     false));
        op_pdn = rw(b, rad_decl_op(b, "moe_gemm",
                           RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("N", g.n_embd),
                                      RAD_INT("K", wp), RAD_INT("n_expert", np),
                                      RAD_INT("top_k", c.top_k), RAD_STR("a_order", "sorted"),
                                      RAD_STR("dtype", "bf16")),
                           w_pdn.data(), (int)np),
                    {pff.x, w.sorted, w.eoff}, {w.edn});
        if (!op_pgu || !op_pdn) return RAD_E_INVAL;
    }
    /* ---- the calibration tap, declared LAST of the routed arm so its buffers cannot alias one
     * the arm is still using. They are PERSIST, so the planner would not alias them anyway; the
     * order is kept because declaration order IS the buffer plan and a reader should not have to
     * check which kind a name is to know that. */
    if (c.calib) {
        /* A rank holding a slice of each expert sees a slice of its K, and a Hessian of a slice is
         * not the Hessian the packer needs. Calibration runs at one rank. */
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

    /* THE GATHER FOLDS THE SHARED ARM IN, so `w.moe` is never written: `scale_rows` and a
     * separate gather would be two dispatches a layer, with a 160 KiB round trip through the
     * buffer between them on top. */
    gfold = shared;
    {
        rad_buf grd[5] = { w.edn, w.ew, w.sorted, w.sout, w.sgate };
        rad_buf gwr[1] = { w.h.x };
        op_gather = RAD_OP(b, "moe_gather",
                           RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n", g.n_embd),
                                      RAD_INT("top_k", c.top_k), RAD_STR("dtype", g.dtype),
                                      RAD_STR("act", gfold ? "sigmoid" : "none")),
                           RAD_NOWEIGHTS);
        if (!op_gather) return RAD_E_INVAL;
        rad_op_reads (b, op_gather, grd, gfold ? 5 : 3);
        rad_op_writes(b, op_gather, gwr, 1);
    }

    /* ---- the shared expert. Ordinary dense linears over the same normed input. */
    if (shared) {
        /* THE SECOND LANE NEEDS THESE OUT OF THE SHARING POOL, and rad_buf_concurrent is where
         * that is said. The arm runs beside the routed GEMMs, so a plan that laid `sgu` over
         * `eff` -- legal from the linear order, which puts one range entirely after the other --
         * would have two streams writing the same arena bytes. It costs a few MiB of arena. */
        RAD_ARCH_TRY(rad_buf_concurrent(b, w.sgu));
        /* WHAT THE DOWN PROJECTION READS decides what the gate writes, so it is asked first: the
         * codes, or the bf16 plane when the weight is plain -- a bf16 model, or a quantised one
         * that keeps the shared expert at the checkpoint's precision. */
        const char* down_base = nm.f("blk.%d.ffn_down_shexp", l);
        RAD_ARCH_TRY(sh_down.probe(b, nm, g, down_base,
                                   { nm.ckpt("%s.down_proj.weight", src.shared) }));
        /* The code pair the gate writes, E4M3 or the int8 one (ActFP8::q8_fed) -- and an int8
         * down projection that quantises for itself writes the int8 pair on this lane too. */
        if (w.sff.cq()) {
            RAD_ARCH_TRY(rad_buf_concurrent(b, w.sff.cq()));
            RAD_ARCH_TRY(rad_buf_concurrent(b, w.sff.cs()));
        }
        if (w.sff.q8 && !w.sff.q8_fed && sh_down.probed_i8) {
            RAD_ARCH_TRY(rad_buf_concurrent(b, w.sff.q8));
            RAD_ARCH_TRY(rad_buf_concurrent(b, w.sff.s8));
        }
        if (!w.sff.q || (sh_down.reads_x && !(w.sff.q8_fed && sh_down.probed_i8)))
            RAD_ARCH_TRY(rad_buf_concurrent(b, w.sff.x));
        RAD_ARCH_TRY(rad_buf_concurrent(b, w.sout));
        RAD_ARCH_TRY(rad_buf_concurrent(b, w.sgate));
        sgate_concurrent = true;
        /* The gate/up rows and the down columns of this rank's slice -- the even split, or the
         * uneven one Config states -- and the split of gate from up stays, because that is a
         * property of the stacked checkpoint tensor and not of the rank count. */
        RAD_ARCH_TRY(sh_gate_up.declare(b, nm, g, nm.f("blk.%d.ffn_gate_up_shexp", l),
                                        2 * c.n_ff_shared, g.n_embd, RAD_SHARD_ROW, grp_layer(l),
                                        { nm.ckpt("%s.gate_proj.weight", src.shared),
                                          nm.ckpt("%s.up_proj.weight",   src.shared) },
                                        w.h, w.sgu, c.rows,
                                        { c.n_ff_shared, c.n_ff_shared },
                                        c.shared_lo, c.shared_hi));
        /* An int8 down projection fed the int8 pair reads no bf16 plane, so none is written. */
        RAD_ARCH_TRY(gq_sh.declare(b, g, w.sgu, w.sff, c.n_ff_shared, "silu", c.rows,
                                   sh_down.reads_x && !(w.sff.q8_fed && sh_down.probed_i8)));
        RAD_ARCH_TRY(sh_down.declare(b, nm, g, down_base,
                                     g.n_embd, c.n_ff_shared, RAD_SHARD_COL, grp_layer(l),
                                     { nm.ckpt("%s.down_proj.weight", src.shared) },
                                     w.sff, w.sout, c.rows, {},
                                     c.shared_lo, c.shared_hi));
        /* The gate. REPLICATED like the router and for the same reason: it is computed from the
         * whole residual stream and a sharded copy would give two ranks two different gates for
         * one token. One output column, so the GEMM is skinny and the kernel clamps rather than
         * masks its N tile -- which is why N = 1 is legal here. */
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_gate_shexp.weight", l), src.shared_gate));
        w_sgate = decl_w(b, nm.f("blk.%d.ffn_gate_shexp.weight", l), RAD_BF16, {1, g.n_embd},
                         RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
        op_sgate = rw(b, RAD_OP(b, "gemm_nt",
                         RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("N", 1),
                                    RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(w_sgate)),
                     {w.h.x}, {w.sgate});
    }

    /* ---- THE ROUTER GEMM AND `router_topk_scatter` AS ONE OP, with the shared expert's gate in
     * the same launch wherever the layer has one. Asked for wherever the pair itself is declared,
     * up to the pair's own row bound and the skinny GEMM's decode band; a plugin that does not
     * serve it leaves the handle 0 and the router and the pair are issued as declared above.
     * Declared HERE, after the gate's weight, and not beside the router: the op names both
     * weights, and declaring it late only lengthens the planned lifetime of the buffers it
     * touches, never shortens one -- each of them is also declared by the op it stands in for, at
     * that op's place.
     *
     * EXCEPT `w.sgate`, WHICH THE FUSED OP WRITES AHEAD OF ITS DECLARED WRITER. The gate is issued
     * at the router's place, before `op_sgate`'s declared index, so a plan bounded by the linear
     * order could lend `w.sgate`'s bytes to a buffer that is live in between. It cannot here
     * because the shared arm took `w.sgate` out of the sharing pool (rad_buf_concurrent), and the
     * declaration refuses a fused gate on a layer where that has not happened.
     *
     * NOT rw()'s braced lists and not one RAD_OP per combination: the parameter list (a protect
     * list or not), the weights and the write set (the gate or not) are conditional, so each is
     * one array cut to its length. */
    if (op_topk_scatter) {
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

    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                       RAD_PARAMS(RAD_INT("world_size", g.world),
                                  RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                  RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                  RAD_INT("min_bytes", g.wire_min_bytes)),
                       RAD_NOWEIGHTS),
                   {w.h.x}, {w.h.x});

    if (add_out)
        op_add = rw(b, RAD_OP(b, "add",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.x, w.h.x}, {w.x});
    return RAD_OK;
}

/* ONE PASS OF AT MOST `c.rows` TOKENS. Every op here is per row -- the norm and the quantiser group
 * along K within a row, the router and the top-k are a row each, the grouped GEMMs give each sorted
 * row its own accumulation, and the gather closes each token's sum over its own slots -- so a pass
 * computes exactly the rows it names.
 *
 * THE EXPERT BUFFERS ARE ADDRESSED FROM ZERO, not from `r0`. They are one pass tall by declaration,
 * and `sorted_tok` indexes them relative to the pass; the only operands that carry the pass offset
 * are the ones that live at the step's height -- the residual stream and the normed input. */


/* The shared arm's first stage: its one-column gate and its gate_up, both over the normed input.
 * The shared expert is ordinary dense linears over that input, and every buffer it touches is the
 * step's height -- so the pass offset goes through all of them exactly as it does through an
 * MlpFP8 slice. */
inline void MoeEXL3::shared_up(RadCtx* c_, int64_t T, int64_t r0, int64_t rows,
                              bool gate_done) const {
    /* `gate_done`: the fused router already wrote this pass's gate. */
    if (!gate_done)
    RAD_ISSUE_N(c_, op_sgate, rows, brow_slice(w.h.x, r0, rows, g.n_embd), RAD_W(w_sgate),
                brow_slice(w.sgate, r0, rows, 1));
    sh_gate_up.step(c_, w.h, w.sgu, T, r0, rows);
}

inline void MoeEXL3::pass(RadCtx* c_, int64_t T, int64_t r0, int64_t rows) const {
    /* `ne` is THIS RANK'S expert count and `ne_all` the model's: the logits plane and the top-k
     * are the model's width, everything after the scatter is this rank's slice. */
    const int64_t ne = c.n_expert, ne_all = c.n_expert_all, k = c.top_k;
    const bool shared = op_sgate != 0;

    if (!ext_in) nq_h.step(c_, T, r0, rows);

    /* THE SHARED ARM BESIDE THE ROUTED ONE, ON THE SAME QUEUE. It reads the same normed input the
     * routed arm does and writes nothing the routed arm touches until the gather joins them, so
     * each of its three stages is independent of the routed stage it is issued behind: its gate
     * and gate_up beside the routed gate_up, its quantiser beside the routed one, its down beside
     * the routed down. A recorded pass plays each such pair with the second launch unordered
     * behind the first (Ctx::haz_mark), so the two run at once, and the next routed stage --
     * which waits for everything in front of it -- waits for the shorter shared stage too, which
     * has long finished. That is the overlap a second queue would buy, without the queue's fork and
     * join: a barrier packet on each side of the arm, and a join on another queue's signal that
     * the packet processor sees microseconds late. Every buffer the arm writes is out of the
     * sharing pool (see declare), so issuing it ahead of its declared position moves no byte. */

    /* THE ROUTED ARM. */
    /* THE COUNTS GO STRAIGHT INTO THE CORE'S ROW for this layer when it keeps one, so the
     * report at the end of step() hands them over with no copy -- otherwise a card-local blit
     * a routed layer, a whole dependent dispatch for a kilobyte. */
    int32_t* const counts = rad_route_counts(c_, layer, ne);
    const RadOperand ecnt = counts ? praw(counts, RAD_I32, ne) : brows(w.ecnt, ne);
    /* THE FUSED ROUTER where it resolved and the pass is within its rows: the router, the pair
     * below and -- with `router_fused_gate` -- the shared gate, in one launch with the operands
     * the three are handed. Otherwise the router and the pair, as declared. */
    const bool rfused = op_router_fused && rows <= router_fused_rows;
    if (rfused)
        RAD_ISSUE_N(c_, op_router_fused, rows, brow_slice(w.h.x, r0, rows, g.n_embd),
                    RAD_W(w_router), router_fused_gate ? RAD_W(w_sgate) : RAD_NONE,
                    brow_slice(w.logits, r0, rows, ne_all),
                    brow_slice(w.ids, r0, rows, k), brow_slice(w.ew, r0, rows, k),
                    brows(w.sorted, rows * k), brows(w.eoff, ne + 1), ecnt,
                    router_fused_gate ? brow_slice(w.sgate, r0, rows, 1) : RAD_NONE);
    else
    RAD_ISSUE_N(c_, op_router, rows, brow_slice(w.h.x, r0, rows, g.n_embd), RAD_W(w_router),
                brow_slice(w.logits, r0, rows, ne_all));
    /* One launch or two. `w.ids` is the fused op's OUTPUT where it was the sort's input, and
     * every other operand is the one the pair is handed. */
    if (rfused) {
        /* the fused router wrote all five */
    } else if (op_topk_scatter && rows <= topk_scatter_rows) {
        RAD_ISSUE_N(c_, op_topk_scatter, rows, brow_slice(w.logits, r0, rows, ne_all),
                    brow_slice(w.ids, r0, rows, k), brow_slice(w.ew, r0, rows, k),
                    brows(w.sorted, rows * k), brows(w.eoff, ne + 1), ecnt);
    } else {
        RAD_ISSUE_N(c_, op_topk, rows, brow_slice(w.logits, r0, rows, ne_all),
                    brow_slice(w.ids, r0, rows, k), brow_slice(w.ew, r0, rows, k));
        RAD_ISSUE_N(c_, op_scatter, rows, brow_slice(w.ids, r0, rows, k),
                    brows(w.sorted, rows * k), brows(w.eoff, ne + 1), ecnt);
    }

    /* The two ops, and the weight tables: the FIRST handle of each run and its length. The runs
     * are contiguous by construction -- see declare -- so `w_gu[p][q] + i` is plane p of expert
     * 2 * i + q. Each operand goes where exl3_abi.h's exl3_positions() puts it: the input, the
     * even class's planes, the routing, the odd class's planes, the output. */
    const int64_t nh = n_reg / 2;
    const auto issue = [&](rad_op op, int gate_up, RadOperand in, RadOperand out) {
        const Exl3Positions& at = *exl3_positions(gate_up);
        const std::vector<rad_weight>* planes = gate_up ? w_gu : w_dn;
        const int  n_planes = gate_up ? exl3::kPlanesGateUp : exl3::kPlanesDown;
        const int* slot     = gate_up ? exl3::kSlotGateUp : exl3::kSlotDown;
        RadOperand opd[16];
        static_assert(sizeof opd / sizeof opd[0] >= 12, "gate_up has 12 operands");
        opd[at.x]      = in;
        opd[at.sorted] = brows(w.sorted, rows * k);
        opd[at.offset] = brows(w.eoff, n_reg + 1);
        opd[at.y]      = out;
        for (int q = 0; q < 2; ++q)
            for (int p = 0; p < n_planes; ++p) {
                const int sq = exl3::kTestMutation == 1 ? 1 - q : q;
                opd[at.plane[q][slot[p]]] = RAD_WTAB(planes[p][(size_t)sq], nh);
            }
        rad_issue(c_, op, opd, at.n_opd, rows);
    };
    issue(op_gu, 1, brow_slice(w.h.x, r0, rows, g.n_embd), brows(w.eff, rows * k));
    if (shared) shared_up(c_, T, r0, rows, rfused && router_fused_gate);
    if (shared) gq_sh.step(c_, w.sgu, T, r0, rows);
    issue(op_dn, 0, brows(w.eff, rows * k), brows(w.edn, rows * k));
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
    if (shared) sh_down.step(c_, w.sff, w.sout, T, r0, rows);
    /* THE CALIBRATION TAP, issued between the two GEMMs' inputs being written and the pass
     * ending. It reads exactly the operands the GEMMs above read and writes nothing they
     * touch, so removing it changes no number the model produces -- which is the property
     * that lets a calibration run be compared against a serving one. The down tap's bf16 copy
     * is the tap's own buffer, written by its own cast. */
    if (op_gram_gu && !calib_done) {
        if (calib_rows == 0) {
            /* The accumulators are arena memory and arrive holding whatever was there. */
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
                             (a && bb && ce) ? "" : " (WITH ERRORS)");
        }
    }

    /* THE GATHER LAST, because with the fold it reads the arm. Its operands are the routed slots
     * plus `sout` and `sgate`, and it writes the block's delta straight into `h.x`. */
    if (!gather_taken(g, T, gather_out_rows, gather_out_rows6))
        RAD_ISSUE_N(c_, op_gather, rows, brows(w.edn, rows * k), brow_slice(w.ew, r0, rows, k),
                    brows(w.sorted, rows * k),
                    gfold ? brow_slice(w.sout, r0, rows, g.n_embd) : RAD_NONE,
                    gfold ? brow_slice(w.sgate, r0, rows, 1) : RAD_NONE,
                    brow_slice(w.h.x, r0, rows, g.n_embd));

    /* A write that takes the all-reduce does it ONCE over the whole step, so it answers on T and
     * this pass has to ask the same question of the same T, not of its own rows. */
    if (op_ar && !ar_taken(g, ar_out_take != kArOutNorm ? T : rows, ar_out, ar_out_take))
        RAD_ISSUE_N(c_, op_ar, rows * g.n_embd, brow_slice(w.h.x, r0, rows, g.n_embd), RAD_NONE);
    if (op_add)
        RAD_ISSUE_N(c_, op_add, rows, brow_slice(w.x, r0, rows, g.n_embd),
                    brow_slice(w.h.x, r0, rows, g.n_embd),
                    brow_slice(w.x, r0, rows, g.n_embd));
}

/* ONE PASS OF THE PER-EXPERT TAP. The expert offsets the scatter wrote decide how many launches
 * there are and where each reads, so they come back first; then the scratch is zeroed, each
 * routed expert's run of the `slots` sorted rows is summed into its square, and the scratch comes
 * back to be added on the host. Both syncs are on the block's own stream, with nothing issued
 * after the tap, so every op the scratch could share storage with is finished before it is
 * written and none is issued until it has been read. */
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
    const int64_t T = batch->n_tok;
    for (int64_t r0 = 0; r0 < T; r0 += c.rows) {
        const int64_t n = (T - r0 < c.rows) ? (T - r0) : c.rows;
        pass(c_, T, r0, n);
    }

    /* ---- THE HANDOFF TO THE HEAT ENGINE, and it is a handoff and not a computation: the buffers
     * below were written by ops this block already issued. `rad_route_report` copies the histogram
     * back ASYNCHRONOUSLY and the heat engine consumes it on the NEXT step, so nothing here blocks
     * the compute stream (spec §5.5). Without it the expert-tiered placement has no measurement to
     * promote on, and every expert stays wherever the warm start put it.
     *
     * IT REPORTS THE LAST PASS, which on the shipped configuration is the whole step: `c.rows`
     * defaults to the prefill chunk, so there is one pass. A deployment that raises the chunk past
     * it gets the last pass's histogram -- a sample of the step rather than the whole of it, which
     * is what the heat engine wants anyway (it is a popularity estimate consumed a step late).
     *
     * `expert_w` IS LEFT NULL rather than filled: RadRouting declares it `const float*` and this
     * block's routing weights are bf16, which is what the gather multiplies. Nothing in the core
     * reads the field -- the heat engine takes `expert_count` -- and handing it a bf16 plane under
     * a float pointer would be a lie that survives until something does. */
    moe_debug_weights(c_, layer, c.top_k, w.ids, w.ew);
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
