/* exl3_abi.h -- the interface between the EXL3 plugins: the two routed-expert GEMM ops, the routing
 * contract they read, the weight-table entry they read it from, and one parse of their geometry.
 * Header-only C; owner R (PLAN.md §2). The encoding itself is include/exl3_encoding.h (owner Q).
 *
 * Every library that implements `moe_exl3_gate_up` / `moe_exl3_down` exports exl3_schema_at() as its
 * schemas -- the reference (kernels/exl3ref) and the device library (kernels/exl3r4d) -- so the two
 * cannot declare different schemas (the loader refuses the second plugin if they do: docs/OPS.md
 * :139-150 at radiance 1.3.0) and `rad-schemas exl3ref.so exl3r4d.so` prints no difference.
 * docs/OPS-exl3.md is the prose form of the schemas below and is checked against them.
 *
 * Both sides check EXL3_IFACE_VERSION. Bump it on ANY change below: a schema, an operand position,
 * the routing contract, the weight-table entry or the geometry rule.
 */
#ifndef EXL3_ABI_H
#define EXL3_ABI_H

#include "rad_abi.h"
#include "rad_plugin.h"
#include "exl3_encoding.h"

#ifdef __cplusplus
extern "C" {
#endif

#define EXL3_IFACE_VERSION 1

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
 * had = the normalised 128-block Hadamard (common/exl3_decode.h exl3_had128). `expert_w` is NOT
 * applied by either: the gather does it, exactly as for `moe_gemm_q`.
 *
 * THE EXPERTS ARE TWO TABLES BY PARITY, as `moe_gemm_q`'s are (docs/OPS.md:677-688): `w` holds the
 * even experts at [N, K], `w_odd` the odd ones at [N_odd, K_odd] (each defaulting to N, K), expert e
 * at entry e / 2 of class e % 2 -- because a rank holding 3 of an expert's five 128-column blocks
 * for one parity and 2 for the other (PLAN.md D7) has two geometries. The output is the WIDER
 * class's width and a narrower class writes zeros past its own.
 *
 * `w_odd` ABSENT (a null RadTensor.data) means ONE class: expert e at entry e of `w`, every expert
 * at [N, K]; N_odd / K_odd, if given, must then equal N / K. NOTE: at 1.3.0 an architecture cannot
 * issue an absent table: the declaration divides its weight list EVENLY over every table position,
 * optional or not (core/build/rad_builder.cpp:697-731), and an issue naming another run is refused
 * (core/runtime/issue.cpp:249-254). So the engine always passes both tables (at tp1 with
 * N_odd == N, K_odd == K, as the in-tree block does); absence is for a caller holding operands
 * directly -- a harness, a tool -- and is what a shape description without N_odd and K_odd means.
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
 */

/* Operand positions, shared by both ops. */
enum {
    EXL3_OPD_X      = 0,  /* gate_up: a, f16|bf16 [n_tok, >= max K];  down: h, f16 [T, >= max K] */
    EXL3_OPD_W      = 1,  /* WTAB: the even experts (or every expert, w_odd absent)              */
    EXL3_OPD_W_ODD  = 2,  /* WTAB, optional: the odd experts                                     */
    EXL3_OPD_SORTED = 3,  /* i32 [>= T]                                                          */
    EXL3_OPD_OFFSET = 4,  /* i32 [n_expert + 1]                                                  */
    EXL3_OPD_Y      = 5,  /* gate_up: h, f16 [T, max(N, N_odd)/2];  down: y, bf16 [T, max(N, N_odd)] */
    EXL3_N_OPD      = 6
};

#define EXL3_OP_GATE_UP "moe_exl3_gate_up"
#define EXL3_OP_DOWN    "moe_exl3_down"

/* The schemas, in this order: 0 gate_up, 1 down. A plugin's rad_kernel_schema_count /
 * rad_kernel_schema_at answer with these. Static locals in a static inline function, so a
 * translation unit that includes this header and never asks pays nothing and warns about nothing. */
static inline int exl3_schema_count(void) { return 2; }

static inline const RadOpSchema* exl3_schema_at(int i) {
    static const RadParamSpec p[] = {
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
    static const RadOperandSpec gate_up[EXL3_N_OPD] = {
        { "a",             RAD_OPD_IN,   0 },
        { "w",             RAD_OPD_WTAB, 0 },
        { "w_odd",         RAD_OPD_WTAB, 1 },
        { "sorted_tok",    RAD_OPD_IN,   0 },
        { "expert_offset", RAD_OPD_IN,   0 },
        { "h",             RAD_OPD_OUT,  0 },
    };
    static const RadOperandSpec down[EXL3_N_OPD] = {
        { "h",             RAD_OPD_IN,   0 },
        { "w",             RAD_OPD_WTAB, 0 },
        { "w_odd",         RAD_OPD_WTAB, 1 },
        { "sorted_tok",    RAD_OPD_IN,   0 },
        { "expert_offset", RAD_OPD_IN,   0 },
        { "y",             RAD_OPD_OUT,  0 },
    };
    static const RadOpSchema s[2] = {
        { EXL3_OP_GATE_UP, p, (int)(sizeof p / sizeof p[0]), gate_up, EXL3_N_OPD,
          "EXL3 routed experts, gate and up: per sorted row, a[sorted_tok/top_k] scaled by suh "
          "(suh.1 for up) and rotated, times W_hat, rotated back and scaled by svh, then "
          "f16(silu(gate) * up). Two weight tables by parity, expert e at entry e/2; rows past "
          "expert_offset[n_expert] are zero. expert_w is not applied. docs/OPS-exl3.md." },
        { EXL3_OP_DOWN, p, (int)(sizeof p / sizeof p[0]), down, EXL3_N_OPD,
          "EXL3 routed experts, down: per sorted row, h[i] scaled by suh and rotated, times W_hat, "
          "rotated back and scaled by svh, bf16. Two weight tables by parity, expert e at entry "
          "e/2; rows past expert_offset[n_expert] are zero. expert_w is not applied. "
          "docs/OPS-exl3.md." },
    };
    return i >= 0 && i < 2 ? &s[i] : 0;
}

/* ===================================================================== THE WEIGHT-TABLE ENTRY
 *
 * PLAN.md D6: one table operand carries ALL of an expert's planes. What a host row reads at entry
 * e is that expert's PLANE RECORD: its canonical planes (include/exl3_encoding.h) back to back in
 * plane order -- codes, suh, svh, then suh.1 for a stacked gate_up -- each region starting a
 * multiple of EXL3_REC_ALIGN bytes from the record's start. Nothing is reordered and nothing is
 * derived: the record is the identity join of the encoding's planes, so a reader decodes it with
 * common/exl3_decode.h exactly as the container holds it.
 *
 *   codes   uK [rows][cols], row-major, rows ceil(cols*K/8) bytes apart (dense), LSB-first
 *   suh     f16 [cols]          input scales  (gate's, for a stacked gate_up)
 *   svh     f16 [rows]          output scales (gate rows then up rows, for a stacked gate_up)
 *   suh.1   f16 [cols]          stacked gate_up only: the up half's input scales
 *
 * A table entry's RadTensor is u8 [record bytes]; the STACKED form (the one a tool with nowhere to
 * put a pointer array uses, docs/OPS.md:120-125) is u8 [entries, record bytes].
 *
 * What a DEVICE row stores is its own business (its layout hook), and a reference row reading the
 * engine's stored bytes reads THAT form -- which is why serving on a reference is refused
 * (docs/PLUGIN.md:62; core/build/rad_builder.cpp:1361-1367, 1385). See notes/p2.md for what this means
 * for the host-only engine and for RADIANCE_OP_ORACLE at 1.3.0. */
enum { EXL3_REC_ALIGN = 16 };

typedef struct Exl3RecLayout {
    int64_t codes, suh, svh, suh1;  /* byte offsets from the record's start; suh1 = -1 unstacked */
    int64_t codes_row_bytes;        /* ceil(cols * K / 8) */
    int64_t bytes;                  /* the whole record */
} Exl3RecLayout;

/* The record of one expert's [rows, cols] weight at K bits. RAD_OK, or RAD_E_DTYPE for K outside
 * 1..8, RAD_E_SHAPE for extents that are not positive multiples of 16 (a tile). */
static inline int exl3_rec_layout(int64_t rows, int64_t cols, int bits, int stacked,
                                  Exl3RecLayout* out) {
    const uint32_t dt = exl3_codes_dtype(bits);
    int64_t at;
    if (dt == RAD_DT_INVALID) return RAD_E_DTYPE;
    if (rows <= 0 || cols <= 0 || rows % 16 || cols % 16) return RAD_E_SHAPE;
#define EXL3_REC_UP(x) (((x) + EXL3_REC_ALIGN - 1) / EXL3_REC_ALIGN * EXL3_REC_ALIGN)
    out->codes_row_bytes = rad_enc_row_bytes(dt, cols);
    out->codes = 0;
    at = EXL3_REC_UP(rows * out->codes_row_bytes);
    out->suh = at;
    at = EXL3_REC_UP(at + 2 * cols);
    out->svh = at;
    at = EXL3_REC_UP(at + 2 * rows);
    out->suh1 = stacked ? at : -1;
    if (stacked) at = EXL3_REC_UP(at + 2 * cols);
    out->bytes = at;
#undef EXL3_REC_UP
    return RAD_OK;
}

/* ===================================================================== THE GEOMETRY, PARSED ONCE
 *
 * Every implementation reads the parameters through this, so a device row can never accept a
 * geometry its reference refuses. `two_classes` is whether w_odd is present -- an operand fact the
 * parameters do not carry; a shape hook, which has no operands, passes exl3_geom_two_classes(). */
typedef struct Exl3Geom {
    int     gate_up;        /* 1 moe_exl3_gate_up, 0 moe_exl3_down */
    int64_t M;              /* tokens (a shape hook reads a range's high bound) */
    int64_t n_expert, top_k;
    int     bits, codebook; /* K and EXL3_CB_* */
    int     n_class;        /* 2 with w_odd, 1 without */
    int64_t rows[2], cols[2];   /* class c's weight [rows, cols] */
    int64_t x_cols;         /* columns the input operand must have: max(cols) */
    int64_t y_cols;         /* the output's width: max(rows)/2 for gate_up, max(rows) for down */
} Exl3Geom;

/* Does this geometry name an odd class? -- N_odd or K_odd given. What a shape description of
 * w_odd follows (present, or RAD_OPD_F_ABSENT). */
static inline int exl3_geom_two_classes(const RadParam* p, int n_p) {
    return rad_param_find(p, n_p, "N_odd") != 0 || rad_param_find(p, n_p, "K_odd") != 0;
}

/* RAD_OK, or: RAD_E_SCHEMA a required parameter absent; RAD_E_DTYPE a dtype or codebook this
 * interface has no name for; RAD_E_SHAPE extents the format cannot hold -- K % 128 (the input
 * rotation), N % 128 for down and N % 256 for gate_up (each half's output rotation), a
 * non-positive count, or N_odd/K_odd differing from N/K with w_odd absent. */
static inline int exl3_geom_parse(const RadParam* p, int n_p, int gate_up, int two_classes,
                                  Exl3Geom* g) {
    static const char* const required[] = { "M", "N", "K", "n_expert", "top_k", "dtype", "codebook" };
    const char* dt = rad_param_gets(p, n_p, "dtype", 0);
    const char* cb = rad_param_gets(p, n_p, "codebook", 0);
    const int64_t N = rad_param_geti(p, n_p, "N", 0), K = rad_param_geti(p, n_p, "K", 0);
    const int64_t No = rad_param_geti(p, n_p, "N_odd", N), Ko = rad_param_geti(p, n_p, "K_odd", K);
    const int64_t rows_mult = gate_up ? 256 : 128;
    int c;
    memset(g, 0, sizeof *g);
    for (c = 0; c < (int)(sizeof required / sizeof required[0]); ++c)
        if (!rad_param_find(p, n_p, required[c])) return RAD_E_SCHEMA;
    if (!dt || !cb) return RAD_E_SCHEMA;   /* present, but not as strings */
    g->gate_up  = gate_up;
    g->M        = rad_param_getdim(p, n_p, "M", 0);
    g->n_expert = rad_param_geti(p, n_p, "n_expert", 0);
    g->top_k    = rad_param_geti(p, n_p, "top_k", 0);
    g->bits     = exl3_gemm_bits(dt);
    g->codebook = exl3_codebook_from_name(cb);
    if (g->bits < 0 || g->codebook < 0) return RAD_E_DTYPE;
    if (g->M < 1 || g->n_expert < 1 || g->top_k < 1) return RAD_E_SHAPE;
    if (!two_classes && (No != N || Ko != K)) return RAD_E_SHAPE;
    g->n_class = two_classes ? 2 : 1;
    g->rows[0] = N;  g->cols[0] = K;
    g->rows[1] = No; g->cols[1] = Ko;
    for (c = 0; c < 2; ++c)
        if (g->rows[c] <= 0 || g->cols[c] <= 0 || g->rows[c] % rows_mult || g->cols[c] % EXL3_HAD)
            return RAD_E_SHAPE;
    g->x_cols = K > Ko ? K : Ko;
    g->y_cols = (N > No ? N : No) / (gate_up ? 2 : 1);
    return RAD_OK;
}

/* Expert e's class and its entry in that class's table. */
static inline int     exl3_class_of(const Exl3Geom* g, int64_t e) { return (int)(e % g->n_class); }
static inline int64_t exl3_entry_of(const Exl3Geom* g, int64_t e) { return e / g->n_class; }
/* How many entries class c's table must hold: ceil((n_expert - c) / n_class). */
static inline int64_t exl3_class_entries(const Exl3Geom* g, int c) {
    return (g->n_expert - c + g->n_class - 1) / g->n_class;
}

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* EXL3_ABI_H */
