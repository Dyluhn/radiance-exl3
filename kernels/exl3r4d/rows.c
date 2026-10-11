/* rows.c -- exl3r4d, the gfx1201 EXL3 kernel library: its identity, its two rows, the shape hook.
 *
 * DECODE BAND ONLY (M <= 16 tokens). The prefill band (M > 16) is P5's and reads the SAME stored codes record
 * (R4: one stored form for every band). Until it exists a geometry above 16 falls to the next library in the
 * hierarchy, which for these ops is the reference (and serving on a reference is refused by name).
 *
 * WHAT IS REFUSED, AND WHERE (R12, nothing unimplemented returns success):
 *   at selection, by constraint (--debug-selection names it): dtype other than exl3k4, codebook other than mul1,
 *     M > 16, K or N not a multiple of 128, n_expert > 4096;
 *   at declare, by the layout hook: an encoding that is not exl3 at K=4 mul1 (RAD_E_DTYPE), a selection or extent
 *     that is not the class's plane (RAD_E_SHAPE);
 *   at launch, by the adapter (launch.c): the same again plus every operand fact a constraint cannot express.
 * Half-integer K has no dtype string (exl3_encoding.h), so it never reaches a row; bias has no plane, likewise. */
#include "exl3r4d.h"

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

static int shape(const RadParam* p, int n_p, int gate_up, int operand, RadOpdDesc* out) {
    Exl3Geom g;
    const Exl3Positions* at = exl3_positions(gate_up);
    if (exl3_geom_parse(p, n_p, gate_up, exl3_geom_two_classes(p, n_p), &g) != RAD_OK) return RAD_E_UNSUPPORTED;
    if (g.bits != EXL3R4D_BITS || g.codebook != EXL3_CB_MUL1) return RAD_E_UNSUPPORTED;
    if (operand < 0 || operand >= at->n_opd || !out) return RAD_E_SHAPE;
    const int64_t T = g.M * g.top_k;
    RadOpdDesc v[12];
    desc(&v[at->x], RAD_F16, 2, (int64_t[]){ gate_up ? g.M : T, g.x_cols }, RAD_FILL_NORMAL);
    for (int c = 0; c < 2; ++c)
        for (int pl = 0; pl < EXL3_N_P; ++pl) {
            const int k = at->plane[c][pl];
            if (k < 0) continue;
            if (c >= g.n_class) {
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
    desc(&v[at->y], gate_up ? RAD_F16 : RAD_BF16, 2, (int64_t[]){ T, g.y_cols }, RAD_FILL_NORMAL);
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

/* ------------------------------------------------------------------ hooks, one pair a row */

static int gu_launch(const RadArgs* a, RadStream s) { return exl3r4d_launch(a, s, 1); }
static int dn_launch(const RadArgs* a, RadStream s) { return exl3r4d_launch(a, s, 0); }
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

/* ------------------------------------------------------------------ the rows */

static const RadConstraint c_decode[] = {
    RAD_CLE("M", 16),
    RAD_CIN("dtype", "exl3k4"),
    RAD_CIN("codebook", "mul1"),
    RAD_CDIV("K", 128),
    RAD_CDIV("N", 128),
    RAD_CLE("n_expert", 4096),
};
#define N_DECODE (int)(sizeof c_decode / sizeof c_decode[0])

static const RadKernelInfo k_rows[] = {
    {
        .name = "exl3r4d_moe_gate_up_m16", .op = EXL3_OP_GATE_UP, .family = "moe",
        .computes = "EXL3 routed experts, gate and up, f16 h: trellis decode in registers, f16 WMMA, the 128-point "
                    "rotations on both sides, SiLU(gate)*up (decode band)",
        .shape = "M <= 16; K % 128, N % 256 (each class); two parity classes in one grid",
        .dtypes = "exl3k4, mul1; a f16 or bf16; h f16",
        .domain = RAD_DOMAIN_DEVICE, .priority = 10,
        .constraints = c_decode, .n_constraints = N_DECODE,
        .launch = gu_launch, .scratch = gu_scratch,
        .layout = gu_layout, .relayout = gu_relayout, .opd_shape = gu_shape, .unrelayout = gu_unrelayout,
    },
    {
        .name = "exl3r4d_moe_down_m16", .op = EXL3_OP_DOWN, .family = "moe",
        .computes = "EXL3 routed experts, down, bf16 y: input rotation fused (rows into LDS), trellis decode in "
                    "registers, f16 WMMA, output rotation (decode band)",
        .shape = "M <= 16; K % 128, N % 128 (each class); two parity classes in one grid",
        .dtypes = "exl3k4, mul1; h f16; y bf16",
        .domain = RAD_DOMAIN_DEVICE, .priority = 10,
        .constraints = c_decode, .n_constraints = N_DECODE,
        .launch = dn_launch, .scratch = dn_scratch,
        .layout = dn_layout, .relayout = dn_relayout, .opd_shape = dn_shape, .unrelayout = dn_unrelayout,
    },
};

/* ------------------------------------------------------------------ the exports */

#ifndef EXL3R4D_BUILD_TARGET
#define EXL3R4D_BUILD_TARGET "gfx1201"
#endif

static const RadPluginInfo k_info = {
    RAD_PLUGIN_KERNEL, "exl3r4d", "0.1.0",
    "EXL3 routed-expert GEMMs (moe_exl3_gate_up, moe_exl3_down) on gfx1201: trellis-in-register decode band",
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
