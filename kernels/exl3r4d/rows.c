/* rows.c -- exl3r4d, the gfx1201 EXL3 kernel library: its identity, its rows, the shape hook.
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
    const int gate_up = op != EXL3_DOWN, prep = op == EXL3_PREP_GU;
    const Exl3Positions* at = exl3_positions(op);
    if (exl3_geom_parse(p, n_p, op, exl3_geom_two_classes(p, n_p), &g) != RAD_OK) return RAD_E_UNSUPPORTED;
    if (g.bits != EXL3R4D_BITS || (!prep && g.codebook != EXL3_CB_MUL1)) return RAD_E_UNSUPPORTED;
    if (operand < 0 || operand >= at->n_opd || !out) return RAD_E_SHAPE;
    const int64_t T = g.M * g.top_k;
    RadOpdDesc v[14];
    /* gate_up's a: token rows; at a_rot = 1 prep_gu's xr, sorted rows [T, 2 * x_cols] (v4) */
    desc(&v[at->x], RAD_F16, 2, (int64_t[]){ gate_up && !g.a_rot ? g.M : T, g.a_rot ? 2 * g.x_cols : g.x_cols },
         RAD_FILL_NORMAL);
    for (int c = 0; c < 2; ++c)
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
/* D18: the gate_up decode row's counter pool (launch.c), at instance init / fini -- never the launch path. */
int  exl3r4d_ctr_init(const RadParam* p, int n_p, void** out);
void exl3r4d_ctr_fini(void* inst);
static int gu_init(const RadParam* p, int n_p, int rank, int world_size, void** out) {
    (void)rank; (void)world_size;
    return exl3r4d_ctr_init(p, n_p, out);
}
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

/* The band's top. 16 is the decode band this library is tuned and declared for. A TEST build may raise it
 * (tests/gpu/op_oracle.sh builds 64): the engine refuses a batched-token budget below 64 for qwen4exp (the delta net's
 * chunk), and an op with a band no device row serves is placed on the host with its weights on the mapped tier,
 * which cannot hold the re-laid codes -- so the real-step oracle needs every band on this library. The kernels are
 * correct at any M (a run longer than a pass takes several: tests/gpu "one run of N rows"); untuned above 16. */
#ifndef EXL3R4D_M_MAX
#define EXL3R4D_M_MAX 16
#endif
static const RadConstraint c_decode[] = {
    RAD_CLE("M", EXL3R4D_M_MAX),
    RAD_CIN("h_rot", "0 1"),   /* v3: both forms (EPI 1); matched on the decimal (core/util/rad_util.cpp:249-258) */
    RAD_CIN("dtype", "exl3k4"),
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
    RAD_CIN("dtype", "exl3k4"),
    RAD_CIN("codebook", "mul1"),
    RAD_CDIV("K", 128),
    RAD_CDIV("N", 128),
    RAD_CLE("n_expert", 4096),
};
#define N_DECODE_GU (int)(sizeof c_decode_gu / sizeof c_decode_gu[0])

/* The M > 16 band (P5). Starts where the decode band ends, so a test build that raises EXL3R4D_M_MAX moves both. */
static const RadConstraint c_prefill[] = {
    RAD_CGE("M", EXL3R4D_M_MAX + 1),
    RAD_CIN("h_rot", "0 1"),   /* v3: both forms (EPI 1), as the decode rows */
    RAD_CIN("dtype", "exl3k4"),
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
    RAD_CIN("dtype", "exl3k4"),
    RAD_CIN("codebook", "mul1"),
    RAD_CDIV("K", 128),
    RAD_CDIV("N", 128),
    RAD_CLE("n_expert", 4096),
};
#define N_PREFILL_GU (int)(sizeof c_prefill_gu / sizeof c_prefill_gu[0])
/* v4: moe_exl3_prep_gu, the decode band's (prep kernel as an op). */
static const RadConstraint c_prep[] = {
    RAD_CLE("M", EXL3R4D_M_MAX),
    RAD_CIN("dtype", "exl3k4"),
    RAD_CDIV("K", 128),
    RAD_CLE("n_expert", 4096),
};
#define N_PREP (int)(sizeof c_prep / sizeof c_prep[0])
int exl3r4d_launch_prep_op(const RadArgs* a, RadStream s);
static int pg_launch(const RadArgs* a, RadStream s) { return exl3r4d_launch_prep_op(a, s); }
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
        .dtypes = "exl3k4, mul1; a f16 or bf16; h f16",
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
        .dtypes = "exl3k4, mul1; h f16; y bf16",
        .domain = RAD_DOMAIN_DEVICE, .priority = 10,
        .constraints = c_decode, .n_constraints = N_DECODE,
        .launch = dn_launch, .scratch = dn_scratch,
        .layout = dn_layout, .relayout = dn_relayout, .opd_shape = dn_shape, .unrelayout = dn_unrelayout,
    },
    {
        .name = "exl3r4d_moe_prep_gu_m16", .op = EXL3_OP_PREP_GU, .family = "moe",
        .computes = "EXL3 routed experts, gate_up's input rotation as an op (v4): every sorted row of every run, "
                    "a[token] x suh and x suh.1, each 128-block Hadamard-rotated, side by side in xr (decode band)",
        .shape = "M <= 16; K % 128 (each class)",
        .dtypes = "exl3k4; a f16 or bf16; xr f16",
        .domain = RAD_DOMAIN_DEVICE, .priority = 10,
        .constraints = c_prep, .n_constraints = N_PREP,
        .launch = pg_launch,
        .layout = pg_layout, .relayout = pg_relayout, .opd_shape = pg_shape, .unrelayout = pg_unrelayout,
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
        .dtypes = "exl3k4, mul1; a f16 or bf16; h f16",
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
        .dtypes = "exl3k4, mul1; h f16; y bf16",
        .domain = RAD_DOMAIN_DEVICE, .priority = 10,
        .constraints = c_prefill, .n_constraints = N_PREFILL,
        .launch = dn_pf_launch, .scratch = dn_pf_scratch,
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
