/* rows.c -- exl3r4d, the gfx1201 EXL3 kernel library: its identity, its rows, the shape hook.
 *
 * DECODE BAND ONLY (M <= 16 tokens). The prefill band (M > 16) is P5's and reads the SAME stored codes record
 * (R4: one stored form for every band). Until it exists a geometry above 16 falls to the next library in the
 * hierarchy, which for these ops is the reference (and serving on a reference is refused by name).
 *
 * WHAT IS REFUSED, AND WHERE (R12, nothing unimplemented returns success):
 *   at selection, by constraint (--debug-selection names it): dtype other than exl3k2 .. exl3k6, codebook other than
 *     mul1, M > 16 (decode rows) or <= 16 (the M > 16 rows), K or N not a multiple of 128, n_expert > 4096;
 *   at declare, by the layout hook: an encoding that is not exl3 at K=4 mul1 (RAD_E_DTYPE), a selection or extent
 *     that is not the class's plane (RAD_E_SHAPE);
 *   at launch, by the adapter (launch.c): the same again plus every operand fact a constraint cannot express.
 * Half-integer K has no dtype string (exl3_encoding.h), so it never reaches a row; bias has no plane, likewise. */
#include "exl3r4d.h"
#include "exl3r4d_prefill.h"   /* P5: the M > 16 rows' hooks */

/* ------------------------------------------------------------------ the shape hook
 *
 * What a tool calling a row cold must build, at the band's high M -- described exactly as exl3ref describes it
 * (kernels/exl3ref/rows.c), because a weight is drawn ONCE as its canonical planes and each library lays it out
 * through its own hooks (tools/rad_kbench.cpp:842-877): each plane table is its plane's own dtype and extents,
 * stacked [entries, rows, cols]; the codes are any symbols; the routing operands are valid (flat indices below
 * M * top_k, expert_offset a cumulative run table). A geometry these rows do not decode (K != 4, not mul1) is not
 * described (RAD_E_UNSUPPORTED: a tool skips and counts it). NOTE (P2 finding 1, notes/p2.md): rad-kbench at
 * 1.3.0 zero-fills every table operand (tools/kfixture.cpp:312), so kbench checks nothing of these rows; tests/gpu
 * is the device-vs-reference check. */
static void desc(RadOpdDesc* d, uint32_t dt, uint32_t rank, const int64_t* shape, uint32_t fill) {
    memset(d, 0, sizeof *d);
    d->dtype = dt;
    d->rank = rank;
    for (uint32_t r = 0; r < rank; ++r) d->shape[r] = shape[r];
    d->fill = fill;
    d->idx_const = -1;
}

static int shape(const RadParam* p, int n_p, int op, int operand, RadOpdDesc* out) {
    Exl3Geom g;
    const int gate_up = exl3_op_base(op) != EXL3_DOWN, prep = exl3_op_base(op) == EXL3_PREP_GU;
    const Exl3Positions* at = exl3_positions(op);
    if (exl3_geom_parse(p, n_p, op, exl3_geom_two_classes(p, n_p), &g) != RAD_OK) return RAD_E_UNSUPPORTED;
    if (!exl3r4d_bits_ok(g.bits) || (!prep && g.codebook != EXL3_CB_MUL1)) return RAD_E_UNSUPPORTED;
    if (operand < 0 || operand >= at->n_opd || !out) return RAD_E_SHAPE;
    const int64_t T = g.M * g.top_k;
    RadOpdDesc v[24];   /* the most operands an op has (gate_up_mod4) */
    /* gate_up's a: token rows; at a_rot = 1 prep_gu's xr, sorted rows [T, 2 * x_cols] (v4) */
    desc(&v[at->x], RAD_F16, 2, (int64_t[]){ gate_up && !g.a_rot ? g.M : T, g.a_rot ? 2 * g.x_cols : g.x_cols },
         RAD_FILL_NORMAL);
    for (int c = 0; c < at->n_cls; ++c)
        for (int pl = 0; pl < EXL3_N_P; ++pl) {
            const int k = at->plane[c][pl];
            if (k < 0) continue;
            if (c >= g.n_class || (pl == EXL3_P_DN_SUH && !g.h_rot)) {   /* not read (dn_suh: h_rot = 1 only) */
                memset(&v[k], 0, sizeof v[k]);
                v[k].idx_const = -1;
                v[k].flags = RAD_OPD_F_ABSENT;
                continue;
            }
            int64_t r, cl;
            exl3_plane_extents(g.rows[c], g.cols[c], pl, &r, &cl);
            const int64_t ext[3] = { exl3_class_entries(&g, c), r, cl };
            if (pl == EXL3_P_CODES) {
                desc(&v[k], exl3_codes_dtype(g.bits), 3, ext, RAD_FILL_INDEX);
                v[k].idx_max = (int64_t)1 << g.bits;
            } else {
                desc(&v[k], RAD_F16, 3, ext, RAD_FILL_NORMAL);
            }
        }
    desc(&v[at->sorted], RAD_I32, 1, (int64_t[]){ T }, RAD_FILL_INDEX);
    v[at->sorted].idx_max = T;
    desc(&v[at->offset], RAD_I32, 1, (int64_t[]){ g.n_expert + 1 }, RAD_FILL_INDEX);
    v[at->offset].flags = RAD_OPD_F_IDX_CU;
    desc(&v[at->y], gate_up ? RAD_F16 : RAD_BF16, 2, (int64_t[]){ T, g.y_cols }, RAD_FILL_NORMAL);   /* prep: xr */
    if (at->ids >= 0) {   /* prep_gu_mod4's optional topk_ids: the router's expert of each (token, slot) */
        desc(&v[at->ids], RAD_I32, 2, (int64_t[]){ g.M, g.top_k }, RAD_FILL_INDEX);
        v[at->ids].idx_max = g.n_expert;
    }
    for (int k = 0; k < at->n_opd; ++k) {   /* an odd class of no experts: nothing a tool can build */
        if (v[k].flags & RAD_OPD_F_ABSENT) continue;
        for (uint32_t r = 0; r < v[k].rank; ++r)
            if (v[k].shape[r] <= 0) return RAD_E_UNSUPPORTED;
    }
    *out = v[operand];
    return RAD_OK;
}

static int gu_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return shape(p, n_p, 1, operand, out);
}
static int dn_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return shape(p, n_p, 0, operand, out);
}
/* v6: the four-class forms */
static int gu4_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return shape(p, n_p, EXL3_GATE_UP4, operand, out);
}
static int dn4_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return shape(p, n_p, EXL3_DOWN4, operand, out);
}
static int pg4_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return shape(p, n_p, EXL3_PREP_GU4, operand, out);
}
/* v5 moe_exl3_gate_up_down (and v7's _mod4 form): each operand described as the half that reads it describes it
 * (exl3_gd_params, exl3_gd4_params), as exl3ref's row does. */
static int gd_shape_of(int op, const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int mod4 = op == EXL3_GU_DN4, gop = mod4 ? EXL3_GATE_UP4 : EXL3_GATE_UP, dop = mod4 ? EXL3_DOWN4 : EXL3_DOWN;
    const Exl3Positions* gd = exl3_positions(op);
    const Exl3Positions* gu = exl3_positions(gop);
    const Exl3Positions* dn = exl3_positions(dop);
    RadParam pg[15], pd[14];
    int ng = 0, nd = 0;
    if (mod4) exl3_gd4_params(p, n_p, pg, &ng, pd, &nd);
    else exl3_gd_params(p, n_p, pg, &ng, pd, &nd);
    if (operand == gd->x) return shape(pg, ng, gop, gu->x, out);
    if (operand == gd->sorted) return shape(pg, ng, gop, gu->sorted, out);
    if (operand == gd->offset) return shape(pg, ng, gop, gu->offset, out);
    if (operand == gd->y) return shape(pd, nd, dop, dn->y, out);
    for (int c = 0; c < gd->n_cls; ++c) {
        for (int q = EXL3_P_CODES; q <= EXL3_P_DN_SUH; ++q)
            if (operand == gd->plane[c][q]) return shape(pg, ng, gop, gu->plane[c][q], out);
        if (operand == gd->plane[c][EXL3_P_DN_CODES]) return shape(pd, nd, dop, dn->plane[c][EXL3_P_CODES], out);
        if (operand == gd->plane[c][EXL3_P_DN_SVH]) return shape(pd, nd, dop, dn->plane[c][EXL3_P_SVH], out);
    }
    return RAD_E_SHAPE;
}
static int gd_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return gd_shape_of(EXL3_GU_DN, p, n_p, operand, out);
}
static int gd4_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return gd_shape_of(EXL3_GU_DN4, p, n_p, operand, out);
}

/* ------------------------------------------------------------------ hooks, one pair a row */

static int gu_launch(const RadArgs* a, RadStream s) { return exl3r4d_launch(a, s, 1); }
/* D18: the gate_up decode row's counter pool (launch.c), at instance init / fini -- never the launch path. */
int  exl3r4d_ctr_init(const RadParam* p, int n_p, void** out);
void exl3r4d_ctr_fini(void* inst);
static int gu_init(const RadParam* p, int n_p, int rank, int world_size, void** out) {
    (void)rank; (void)world_size;
    return exl3r4d_ctr_init(p, n_p, out);
}
static int dn_launch(const RadArgs* a, RadStream s) { return exl3r4d_launch(a, s, 0); }
/* PDN: the down decode row's ticket pool (launch.c), at instance init / fini -- never the launch path. */
int  exl3r4d_ctr_init_dn(const RadParam* p, int n_p, void** out);
/* P10-TP4S L1: the four-class decode gate_up's counter pool (its K slices), at instance init -- never the launch path. */
int  exl3r4d_ctr_init_gu4(const RadParam* p, int n_p, void** out);
static int gu4_init(const RadParam* p, int n_p, int rank, int world_size, void** out) {
    (void)rank; (void)world_size;
    return exl3r4d_ctr_init_gu4(p, n_p, out);
}
static int dn_init(const RadParam* p, int n_p, int rank, int world_size, void** out) {
    (void)rank; (void)world_size;
    return exl3r4d_ctr_init_dn(p, n_p, out);
}
static int gd_launch(const RadArgs* a, RadStream s) { return exl3r4d_launch_gd(a, s); }
static int gd_init(const RadParam* p, int n_p, int rank, int world_size, void** out) {
    (void)rank; (void)world_size;
    return exl3r4d_ctr_init_gd(p, n_p, out);   /* the fused launch's counter pool, at instance init (R6) */
}
static int64_t gd_scratch(const RadArgs* a) { return exl3r4d_scratch_gd(a); }
static int gd4_launch(const RadArgs* a, RadStream s) { return exl3r4d_launch_gd4(a, s); }
static int64_t gd4_scratch(const RadArgs* a) { return exl3r4d_scratch_gd4(a); }
static int64_t gu_scratch(const RadArgs* a) { return exl3r4d_scratch(a, 1); }
static int64_t dn_scratch(const RadArgs* a) { return exl3r4d_scratch(a, 0); }

static int gu_layout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,
                     const RadTensor* pl, int n, RadLayout* out) {
    return exl3r4d_layout(1, p, n_p, o, e, sel, pl, n, out);
}
static int dn_layout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,
                     const RadTensor* pl, int n, RadLayout* out) {
    return exl3r4d_layout(0, p, n_p, o, e, sel, pl, n, out);
}
/* v5: each plane operand laid out as the half that reads it lays it out (layout.c, gd_view) */
static int gd_layout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,
                     const RadTensor* pl, int n, RadLayout* out) {
    return exl3r4d_layout(EXL3_GU_DN, p, n_p, o, e, sel, pl, n, out);
}
static int gd_relayout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,
                       const RadTensor* pl, int n, void* dst, int64_t b) {
    return exl3r4d_relayout(EXL3_GU_DN, p, n_p, o, e, sel, pl, n, dst, b);
}
static int gd_unrelayout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,
                         const void* src, int64_t b, const RadTensor* pl, int n) {
    return exl3r4d_unrelayout(EXL3_GU_DN, p, n_p, o, e, sel, src, b, pl, n);
}
static int gd4_layout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,
                      const RadTensor* pl, int n, RadLayout* out) {
    return exl3r4d_layout(EXL3_GU_DN4, p, n_p, o, e, sel, pl, n, out);
}
static int gd4_relayout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,
                        const RadTensor* pl, int n, void* dst, int64_t b) {
    return exl3r4d_relayout(EXL3_GU_DN4, p, n_p, o, e, sel, pl, n, dst, b);
}
static int gd4_unrelayout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,
                          const void* src, int64_t b, const RadTensor* pl, int n) {
    return exl3r4d_unrelayout(EXL3_GU_DN4, p, n_p, o, e, sel, src, b, pl, n);
}
static int gu_relayout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,
                       const RadTensor* pl, int n, void* dst, int64_t b) {
    return exl3r4d_relayout(1, p, n_p, o, e, sel, pl, n, dst, b);
}
static int dn_relayout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,
                       const RadTensor* pl, int n, void* dst, int64_t b) {
    return exl3r4d_relayout(0, p, n_p, o, e, sel, pl, n, dst, b);
}
static int gu_unrelayout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,
                         const void* src, int64_t b, const RadTensor* pl, int n) {
    return exl3r4d_unrelayout(1, p, n_p, o, e, sel, src, b, pl, n);
}
static int dn_unrelayout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,
                         const void* src, int64_t b, const RadTensor* pl, int n) {
    return exl3r4d_unrelayout(0, p, n_p, o, e, sel, src, b, pl, n);
}
/* v6: the four-class forms' hooks -- the same stored forms (layout.c), their own operand positions */
#define EXL3R4D_HOOKS4(name, OP)                                                                                \
    static int name##_launch(const RadArgs* a, RadStream s) { return exl3r4d_launch(a, s, OP); }               \
    static int name##_pf_launch(const RadArgs* a, RadStream s) { return exl3r4d_pf_launch(a, s, OP); }         \
    static int64_t name##_scratch(const RadArgs* a) { return exl3r4d_scratch(a, OP); }                          \
    static int64_t name##_pf_scratch(const RadArgs* a) { return exl3r4d_pf_scratch(a, OP); }                    \
    static int name##_layout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,           \
                             const RadTensor* pl, int n, RadLayout* out) {                                     \
        return exl3r4d_layout(OP, p, n_p, o, e, sel, pl, n, out);                                             \
    }                                                                                                         \
    static int name##_relayout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,         \
                               const RadTensor* pl, int n, void* dst, int64_t b) {                             \
        return exl3r4d_relayout(OP, p, n_p, o, e, sel, pl, n, dst, b);                                        \
    }                                                                                                         \
    static int name##_unrelayout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,       \
                                 const void* src, int64_t b, const RadTensor* pl, int n) {                     \
        return exl3r4d_unrelayout(OP, p, n_p, o, e, sel, src, b, pl, n);                                      \
    }
EXL3R4D_HOOKS4(gu4, EXL3_GATE_UP4)
EXL3R4D_HOOKS4(dn4, EXL3_DOWN4)
#undef EXL3R4D_HOOKS4

/* ------------------------------------------------------------------ the rows */

/* The band's top. 16 is the decode band this library is tuned and declared for. A TEST build may raise it
 * (tests/gpu/op_oracle.sh builds 64): the engine refuses a batched-token budget below 64 for qwen4exp (the delta net's
 * chunk), and an op with a band no device row serves is placed on the host with its weights on the mapped tier,
 * which cannot hold the re-laid codes -- so the real-step oracle needs every band on this library. The kernels are
 * correct at any M (a run longer than a pass takes several: tests/gpu "one run of N rows"); untuned above 16. */
#ifndef EXL3R4D_M_MAX
#define EXL3R4D_M_MAX 16
#endif
/* R23: both bands and prep_gu at every K the device code is built for, codebook mul1 (the M > 16 rows' kernels are
 * templated on K: exl3r4d_prefill.hip). EXL3R4D_PREFILL_DTYPES "exl3k4" with launch.c's EXL3R4D_PF_ANY_K 0 puts the
 * M > 16 band back to K = 4 alone -- one switch each, so neither band is selected for a K its kernels refuse. */
#define EXL3R4D_DECODE_DTYPES "exl3k2 exl3k3 exl3k4 exl3k5 exl3k6"
#ifndef EXL3R4D_PREFILL_DTYPES
#define EXL3R4D_PREFILL_DTYPES EXL3R4D_DECODE_DTYPES
#endif
static const RadConstraint c_decode[] = {
    RAD_CLE("M", EXL3R4D_M_MAX),
    RAD_CIN("h_rot", "0 1"),   /* v3: both forms (EPI 1); matched on the decimal (core/util/rad_util.cpp:249-258) */
    RAD_CIN("dtype", EXL3R4D_DECODE_DTYPES),
    RAD_CIN("codebook", "mul1"),
    RAD_CDIV("K", 128),
    RAD_CDIV("N", 128),
    RAD_CLE("n_expert", 4096),
};
#define N_DECODE (int)(sizeof c_decode / sizeof c_decode[0])
/* gate_up's decode row: both v4 input forms -- a_rot 0 (token rows, the prep inside the op) and a_rot 1 (prep_gu's
 * xr). A list of its own: down carries no a_rot, and a constraint on a key the op does not carry never holds. */
static const RadConstraint c_decode_gu[] = {
    RAD_CLE("M", EXL3R4D_M_MAX),
    RAD_CIN("h_rot", "0 1"),
    RAD_CIN("a_rot", "0 1"),
    RAD_CIN("dtype", EXL3R4D_DECODE_DTYPES),
    RAD_CIN("codebook", "mul1"),
    RAD_CDIV("K", 128),
    RAD_CDIV("N", 128),
    RAD_CLE("n_expert", 4096),
};
#define N_DECODE_GU (int)(sizeof c_decode_gu / sizeof c_decode_gu[0])

/* v5 moe_exl3_gate_up_down, the decode band's: no h_rot / a_rot keys (the op is defined at a_rot 0, h_rot 1). */
static const RadConstraint c_decode_gd[] = {
    RAD_CLE("M", EXL3R4D_M_MAX),
    RAD_CIN("dtype", EXL3R4D_DECODE_DTYPES),
    RAD_CIN("codebook", "mul1"),
    RAD_CDIV("K", 128),
    RAD_CDIV("N", 256),
    RAD_CLE("n_expert", 4096),
};
#define N_DECODE_GD (int)(sizeof c_decode_gd / sizeof c_decode_gd[0])

/* The M > 16 band (P5). Starts where the decode band ends, so a test build that raises EXL3R4D_M_MAX moves both. */
static const RadConstraint c_prefill[] = {
    RAD_CGE("M", EXL3R4D_M_MAX + 1),
    RAD_CIN("h_rot", "0 1"),   /* v3: both forms (EPI 1), as the decode rows */
    RAD_CIN("dtype", EXL3R4D_PREFILL_DTYPES),
    RAD_CIN("codebook", "mul1"),
    RAD_CDIV("K", 128),
    RAD_CDIV("N", 128),
    RAD_CLE("n_expert", 4096),
};
#define N_PREFILL (int)(sizeof c_prefill / sizeof c_prefill[0])
/* gate_up's M > 16 row: v4's a_rot = 1 is a decode-band form (prep_gu is declared over M 1..16): refused by name.
 * A list of its own -- a constraint on a key the op does not carry never holds, and down has no a_rot. */
static const RadConstraint c_prefill_gu[] = {
    RAD_CGE("M", EXL3R4D_M_MAX + 1),
    RAD_CIN("h_rot", "0 1"),
    RAD_CEQ("a_rot", 0),
    RAD_CIN("dtype", EXL3R4D_PREFILL_DTYPES),
    RAD_CIN("codebook", "mul1"),
    RAD_CDIV("K", 128),
    RAD_CDIV("N", 128),
    RAD_CLE("n_expert", 4096),
};
#define N_PREFILL_GU (int)(sizeof c_prefill_gu / sizeof c_prefill_gu[0])
/* v4: moe_exl3_prep_gu, the decode band's (prep kernel as an op). */
static const RadConstraint c_prep[] = {
    RAD_CLE("M", EXL3R4D_M_MAX),
    RAD_CIN("dtype", EXL3R4D_DECODE_DTYPES),
    RAD_CDIV("K", 128),
    RAD_CLE("n_expert", 4096),
};
#define N_PREP (int)(sizeof c_prep / sizeof c_prep[0])
int exl3r4d_launch_prep_op(const RadArgs* a, RadStream s, int op);
static int pg_launch(const RadArgs* a, RadStream s) { return exl3r4d_launch_prep_op(a, s, EXL3_PREP_GU); }
static int pg4_launch(const RadArgs* a, RadStream s) { return exl3r4d_launch_prep_op(a, s, EXL3_PREP_GU4); }
#if EXL3R4D_SUHC
int  exl3r4d_suhc_init(const RadParam* p, int n_p, void** out);
void exl3r4d_suhc_fini(void* inst);
static int pg_init(const RadParam* p, int n_p, int rank, int world_size, void** out) {
    (void)rank; (void)world_size;
    return exl3r4d_suhc_init(p, n_p, out);   /* the prep's resident suh (launch.c, SUHC) */
}
#endif
/* prep_gu's tables are gate_up's suh / suh.1 runs, scale vectors stored as they are: the hook checks the encoding
 * and the extents (refused by name at declare) and lays nothing out. */
static int pg_layout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,
                     const RadTensor* pl, int n, RadLayout* out) {
    return exl3r4d_layout(EXL3_PREP_GU, p, n_p, o, e, sel, pl, n, out);
}
/* ...and since the hook claims a stored form ("exl3r4d.f16.vec", L-ALIGN), the row must produce it and give it back:
 * a copy each way, the same hooks gate_up's row registers. */
static int pg_relayout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,
                       const RadTensor* pl, int n, void* dst, int64_t b) {
    return exl3r4d_relayout(EXL3_PREP_GU, p, n_p, o, e, sel, pl, n, dst, b);
}
static int pg_unrelayout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,
                         const void* src, int64_t b, const RadTensor* pl, int n) {
    return exl3r4d_unrelayout(EXL3_PREP_GU, p, n_p, o, e, sel, src, b, pl, n);
}
static int pg4_layout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,
                      const RadTensor* pl, int n, RadLayout* out) {
    return exl3r4d_layout(EXL3_PREP_GU4, p, n_p, o, e, sel, pl, n, out);
}
static int pg4_relayout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,
                        const RadTensor* pl, int n, void* dst, int64_t b) {
    return exl3r4d_relayout(EXL3_PREP_GU4, p, n_p, o, e, sel, pl, n, dst, b);
}
static int pg4_unrelayout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,
                          const void* src, int64_t b, const RadTensor* pl, int n) {
    return exl3r4d_unrelayout(EXL3_PREP_GU4, p, n_p, o, e, sel, src, b, pl, n);
}
static int pg_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return shape(p, n_p, EXL3_PREP_GU, operand, out);
}
static int gu_pf_launch(const RadArgs* a, RadStream s) { return exl3r4d_pf_launch(a, s, 1); }
static int dn_pf_launch(const RadArgs* a, RadStream s) { return exl3r4d_pf_launch(a, s, 0); }
static int64_t gu_pf_scratch(const RadArgs* a) { return exl3r4d_pf_scratch(a, 1); }
static int64_t dn_pf_scratch(const RadArgs* a) { return exl3r4d_pf_scratch(a, 0); }

static const RadKernelInfo k_rows[] = {
    {
        .name = "exl3r4d_moe_gate_up_m16", .op = EXL3_OP_GATE_UP, .family = "moe",
        .computes = "EXL3 routed experts, gate and up, f16 h: input rotation (prep kernel into scratch; a_rot 1: "
                    "prep_gu's xr read as it is), trellis decode in registers, f16 WMMA, the output rotation, "
                    "SiLU(gate)*up (decode band)",
        .shape = "M <= 16; K % 128, N % 256 (each class); two parity classes in one grid",
        .dtypes = "exl3k2 .. exl3k6, mul1; a f16 or bf16; h f16",
        .domain = RAD_DOMAIN_DEVICE, .priority = 10,
        .constraints = c_decode_gu, .n_constraints = N_DECODE_GU,
        .init = gu_init, .fini = exl3r4d_ctr_fini,
        .launch = gu_launch, .scratch = gu_scratch,
        .layout = gu_layout, .relayout = gu_relayout, .opd_shape = gu_shape, .unrelayout = gu_unrelayout,
    },
    {
        .name = "exl3r4d_moe_down_m16", .op = EXL3_OP_DOWN, .family = "moe",
        .computes = "EXL3 routed experts, down, bf16 y: input rotation (prep kernel into scratch), trellis decode "
                    "in registers, f16 WMMA, output rotation (decode band)",
        .shape = "M <= 16; K % 128, N % 128 (each class); two parity classes in one grid",
        .dtypes = "exl3k2 .. exl3k6, mul1; h f16; y bf16",
        .domain = RAD_DOMAIN_DEVICE, .priority = 10,
        .constraints = c_decode, .n_constraints = N_DECODE,
        .init = EXL3R4D_PDN ? dn_init : NULL, .fini = EXL3R4D_PDN ? exl3r4d_ctr_fini : NULL,
        .launch = dn_launch, .scratch = dn_scratch,
        .layout = dn_layout, .relayout = dn_relayout, .opd_shape = dn_shape, .unrelayout = dn_unrelayout,
    },
    {
        .name = "exl3r4d_moe_prep_gu_m16", .op = EXL3_OP_PREP_GU, .family = "moe",
        .computes = "EXL3 routed experts, gate_up's input rotation as an op (v4): every sorted row of every run, "
                    "a[token] x suh and x suh.1, each 128-block Hadamard-rotated, side by side in xr (decode band)",
        .shape = "M <= 16; K % 128 (each class)",
        .dtypes = "exl3k2 .. exl3k6; a f16 or bf16; xr f16",
        .domain = RAD_DOMAIN_DEVICE, .priority = 10,
        .constraints = c_prep, .n_constraints = N_PREP,
#if EXL3R4D_SUHC
        .init = pg_init, .fini = exl3r4d_suhc_fini,
#endif
        .launch = pg_launch,
        .layout = pg_layout, .relayout = pg_relayout, .opd_shape = pg_shape, .unrelayout = pg_unrelayout,
    },
    {
        .name = "exl3r4d_moe_gate_up_down_m16", .op = EXL3_OP_GU_DN, .family = "moe",
        .computes = "EXL3 routed experts, the decode chain as one op (v5): gate_up (a_rot 0, h_rot 1) then down "
                    "(h_rot 1); at K = 4 ONE persistent launch (input rotation in its prologue, gate_up's and down's "
                    "units ordered by counters, down of an expert after its gate_up), else the separate kernels; the "
                    "separate ops' bits either way (decode band)",
        .shape = "M <= 16; K % 128, N % 256 (each class); two parity classes in one grid",
        .dtypes = "exl3k2 .. exl3k6, mul1; a f16 or bf16; y bf16",
        .domain = RAD_DOMAIN_DEVICE, .priority = 10,
        .constraints = c_decode_gd, .n_constraints = N_DECODE_GD,
        .init = gd_init, .fini = exl3r4d_ctr_fini,
        .launch = gd_launch, .scratch = gd_scratch,
        .layout = gd_layout, .relayout = gd_relayout, .opd_shape = gd_shape, .unrelayout = gd_unrelayout,
    },
    /* ---------------------------------------------------- the M > 16 band (P5: prefill.c, exl3r4d_prefill.hip)
     * The same layout and shape hooks as the decode rows -- one stored form for every band (R4) -- and the band's
     * own launch and scratch hooks, which choose the form by slots per expert (prefill.c). */
    {
        .name = "exl3r4d_moe_gate_up_pf", .op = EXL3_OP_GATE_UP, .family = "moe",
        .computes = "EXL3 routed experts, gate and up, f16 h, M > 16: decode kernels while runs average <= 1.5 rows "
                    "an expert, else the run-aligned MTX GEMM (map and prep kernel into scratch, trellis decode in "
                    "registers feeding MTX f16 WMMAs, output rotation, SiLU(gate)*up)",
        .shape = "M > 16; K % 128, N % 256 (each class); two parity classes in one grid",
        .dtypes = "exl3k2 .. exl3k6, mul1; a f16 or bf16; h f16",
        .domain = RAD_DOMAIN_DEVICE, .priority = 10,
        .constraints = c_prefill_gu, .n_constraints = N_PREFILL_GU,
        .launch = gu_pf_launch, .scratch = gu_pf_scratch,
        .layout = gu_layout, .relayout = gu_relayout, .opd_shape = gu_shape, .unrelayout = gu_unrelayout,
    },
    {
        .name = "exl3r4d_moe_down_pf", .op = EXL3_OP_DOWN, .family = "moe",
        .computes = "EXL3 routed experts, down, bf16 y, M > 16: decode kernels while runs average <= 1.5 rows an "
                    "expert, else the run-aligned MTX GEMM (map and prep kernel into scratch, trellis decode in "
                    "registers feeding MTX f16 WMMAs, output rotation)",
        .shape = "M > 16; K % 128, N % 128 (each class); two parity classes in one grid",
        .dtypes = "exl3k2 .. exl3k6, mul1; h f16; y bf16",
        .domain = RAD_DOMAIN_DEVICE, .priority = 10,
        .constraints = c_prefill, .n_constraints = N_PREFILL,
        .launch = dn_pf_launch, .scratch = dn_pf_scratch,
        .layout = dn_layout, .relayout = dn_relayout, .opd_shape = dn_shape, .unrelayout = dn_unrelayout,
    },
    /* ---------------------------------------------------- v6: the four-class forms (tp4's rotation by number mod 4)
     * The rows above over four class tables a plane: the same kernels compiled with four class records in the
     * argument block (exl3r4d_mod4.hip, exl3r4d_prefill_mod4.hip), the same constraints, stored forms and bands. No
     * counter pools: the shipped configuration's decode kernels take none (FUSE1, WGQ, PDN off). */
    {
        .name = "exl3r4d_moe_gate_up_mod4_m16", .op = EXL3_OP_GATE_UP4, .family = "moe",
        .computes = "EXL3 routed experts, gate and up, f16 h, four classes by number mod 4: the decode band's "
                    "gate_up kernels (decode band)",
        .shape = "M <= 16; K % 128, N % 256 (each class); four classes in one grid",
        .dtypes = "exl3k2 .. exl3k6, mul1; a f16 or bf16; h f16",
        .domain = RAD_DOMAIN_DEVICE, .priority = 10,
        .constraints = c_decode_gu, .n_constraints = N_DECODE_GU,
        .init = gu4_init, .fini = exl3r4d_ctr_fini,
        .launch = gu4_launch, .scratch = gu4_scratch,
        .layout = gu4_layout, .relayout = gu4_relayout, .opd_shape = gu4_shape, .unrelayout = gu4_unrelayout,
    },
    {
        .name = "exl3r4d_moe_down_mod4_m16", .op = EXL3_OP_DOWN4, .family = "moe",
        .computes = "EXL3 routed experts, down, bf16 y, four classes by number mod 4: the decode band's down "
                    "kernels (decode band)",
        .shape = "M <= 16; K % 128, N % 128 (each class); four classes in one grid",
        .dtypes = "exl3k2 .. exl3k6, mul1; h f16; y bf16",
        .domain = RAD_DOMAIN_DEVICE, .priority = 10,
        .constraints = c_decode, .n_constraints = N_DECODE,
        .launch = dn4_launch, .scratch = dn4_scratch,
        .layout = dn4_layout, .relayout = dn4_relayout, .opd_shape = dn4_shape, .unrelayout = dn4_unrelayout,
    },
    {
        .name = "exl3r4d_moe_prep_gu_mod4_m16", .op = EXL3_OP_PREP_GU4, .family = "moe",
        .computes = "EXL3 routed experts, gate_up's input rotation as an op, four classes by number mod 4 "
                    "(decode band)",
        .shape = "M <= 16; K % 128 (each class)",
        .dtypes = "exl3k2 .. exl3k6; a f16 or bf16; xr f16",
        .domain = RAD_DOMAIN_DEVICE, .priority = 10,
        .constraints = c_prep, .n_constraints = N_PREP,
        .launch = pg4_launch,
        .layout = pg4_layout, .relayout = pg4_relayout, .opd_shape = pg4_shape, .unrelayout = pg4_unrelayout,
    },
    {   /* v7: the four-class decode chain as one op -- the fused kernel compiled at four classes with no hidden
         * argument block (exl3r4d_gd_mod4.hip), its counter pool as the two-class row's */
        .name = "exl3r4d_moe_gate_up_down_mod4_m16", .op = EXL3_OP_GU_DN4, .family = "moe",
        .computes = "EXL3 routed experts, the decode chain as one op, four classes by number mod 4 (v7): "
                    "gate_up_mod4 (a_rot 0, h_rot 1) then down_mod4 (h_rot 1); at K = 4 ONE persistent launch, else "
                    "the separate kernels; the separate ops' bits either way (decode band)",
        .shape = "M <= 16; K % 128, N % 256 (each class); four classes in one grid",
        .dtypes = "exl3k2 .. exl3k6, mul1; a f16 or bf16; y bf16",
        .domain = RAD_DOMAIN_DEVICE, .priority = 10,
        .constraints = c_decode_gd, .n_constraints = N_DECODE_GD,
        .init = gd_init, .fini = exl3r4d_ctr_fini,
        .launch = gd4_launch, .scratch = gd4_scratch,
        .layout = gd4_layout, .relayout = gd4_relayout, .opd_shape = gd4_shape, .unrelayout = gd4_unrelayout,
    },
    {
        .name = "exl3r4d_moe_gate_up_mod4_pf", .op = EXL3_OP_GATE_UP4, .family = "moe",
        .computes = "EXL3 routed experts, gate and up, f16 h, M > 16, four classes by number mod 4: the M > 16 "
                    "band's gate_up kernels",
        .shape = "M > 16; K % 128, N % 256 (each class); four classes in one grid",
        .dtypes = "exl3k2 .. exl3k6, mul1; a f16 or bf16; h f16",
        .domain = RAD_DOMAIN_DEVICE, .priority = 10,
        .constraints = c_prefill_gu, .n_constraints = N_PREFILL_GU,
        .launch = gu4_pf_launch, .scratch = gu4_pf_scratch,
        .layout = gu4_layout, .relayout = gu4_relayout, .opd_shape = gu4_shape, .unrelayout = gu4_unrelayout,
    },
    {
        .name = "exl3r4d_moe_down_mod4_pf", .op = EXL3_OP_DOWN4, .family = "moe",
        .computes = "EXL3 routed experts, down, bf16 y, M > 16, four classes by number mod 4: the M > 16 band's "
                    "down kernels",
        .shape = "M > 16; K % 128, N % 128 (each class); four classes in one grid",
        .dtypes = "exl3k2 .. exl3k6, mul1; h f16; y bf16",
        .domain = RAD_DOMAIN_DEVICE, .priority = 10,
        .constraints = c_prefill, .n_constraints = N_PREFILL,
        .launch = dn4_pf_launch, .scratch = dn4_pf_scratch,
        .layout = dn4_layout, .relayout = dn4_relayout, .opd_shape = dn4_shape, .unrelayout = dn4_unrelayout,
    },
};

/* ------------------------------------------------------------------ the exports */

#ifndef EXL3R4D_BUILD_TARGET
#define EXL3R4D_BUILD_TARGET "gfx1201"
#endif

static const RadPluginInfo k_info = {
    RAD_PLUGIN_KERNEL, "exl3r4d", "0.1.0",
    "EXL3 routed-expert GEMMs (moe_exl3_gate_up, moe_exl3_down, moe_exl3_prep_gu, moe_exl3_gate_up_down, and their _mod4 forms) on gfx1201: trellis-in-register decode band",
    EXL3R4D_BUILD_TARGET,
};

uint32_t             rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
const RadPluginInfo* rad_plugin_info(void) { return &k_info; }

int                rad_kernel_schema_count(void) { return exl3_schema_count(); }
const RadOpSchema* rad_kernel_schema_at(int i) { return exl3_schema_at(i); }
int                rad_kernel_count(void) { return (int)(sizeof k_rows / sizeof k_rows[0]); }
const RadKernelInfo* rad_kernel_at(int i) {
    return i >= 0 && i < rad_kernel_count() ? &k_rows[i] : 0;
}

/* Both rows read and write their operands and the scratch slice they are handed and nothing else: no counter, no
 * global, nothing kept between launches; the prep kernel's rows live in the launch's own slice and its GEMM is the
 * same op's second dispatch, which the engine always orders (core/runtime/ctx.cpp:1100-1110). STEP0 S0.9. */
int rad_kernel_concurrent(int i) { return i >= 0 && i < rad_kernel_count(); }
