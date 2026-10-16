/* rows.c -- exl3ref, the EXL3 host reference library: its identity, its two rows, its shape hook.
 *
 * A REFERENCE (rad_plugin_reference): ranked below every other library, never served without
 * --debug-accept-reference-kernels, what the device rows of kernels/exl3r4d are checked against.
 * The schemas are include/exl3_abi.h's, exported as they are, so `rad-schemas` finds the two
 * libraries identical by construction.
 *
 * The rows constrain only what names the format -- the dtype (K) and the codebook -- so a geometry
 * this library cannot decode is refused at selection with the constraint named; every extent is
 * read off the operands at launch and refused there (docs/OPS-exl3.md, "What they refuse"). No
 * layout hook: every plane is read as the container holds it, one weight table a plane
 * (include/exl3_abi.h "THE PLANES"), which a host-only engine stores as it is and the per-op oracle
 * hands over as it is. */
#include "exl3ref.h"

/* ------------------------------------------------------------------ the shape hook
 *
 * Every operand of every case a tool builds cold, described from the parameters: what `launch`
 * reads, at the band's high M. Each plane table is its plane's own dtype and extents, STACKED
 * ([entries, rows, cols], include/exl3_abi.h "THE PLANES"): that is what the per-op oracle sizes
 * the host copy of a table by and inverts a re-laid plane into (core/runtime/oracle.cpp:456-548),
 * so the description is what makes these rows checkable on a real step. The odd tables are ABSENT
 * when the geometry names no odd class (exl3_geom_two_classes), and gate_up's dn_suh tables ABSENT
 * unless h_rot = 1, the only geometry that reads them. The routing operands are described
 * so a tool draws VALID ones: flat indices below M * top_k, and expert_offset as a cumulative run
 * table (RAD_OPD_F_IDX_CU). Codes are any symbols (every sequence is a valid trellis). */
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
    const Exl3Positions* at = exl3_positions(op);
    const int two = exl3_geom_two_classes(p, n_p);
    if (exl3_geom_parse(p, n_p, op, two, &g) != RAD_OK) return RAD_E_UNSUPPORTED;
    if (operand < 0 || operand >= at->n_opd || !out) return RAD_E_SHAPE;
    const int64_t T = g.M * g.top_k;
    RadOpdDesc v[24];   /* the most operands an op has (gate_up_mod4) */
    /* The input: token rows (gate_up a_rot = 0, prep_gu), prep_gu's xr (gate_up a_rot = 1), or down's sorted h. */
    const int64_t x_rows = g.gate_up && !g.a_rot ? g.M : T, x_cols = g.a_rot ? 2 * g.x_cols : g.x_cols;
    desc(&v[at->x], RAD_F16, 2, (int64_t[]){ x_rows, x_cols }, RAD_FILL_NORMAL);
    for (int c = 0; c < at->n_cls; ++c)
        for (int pl = 0; pl < EXL3_N_P; ++pl) {
            const int k = at->plane[c][pl];
            if (k < 0) continue;
            if (c >= g.n_class || (pl == EXL3_P_DN_SUH && !g.h_rot)) {   /* not read */
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
    desc(&v[at->y], g.op == EXL3_DOWN ? RAD_BF16 : RAD_F16, 2, (int64_t[]){ T, g.y_cols }, RAD_FILL_NORMAL);
    if (at->ids >= 0) {   /* prep_gu_mod4's optional topk_ids: the router's expert of each (token, slot) */
        desc(&v[at->ids], RAD_I32, 2, (int64_t[]){ g.M, g.top_k }, RAD_FILL_INDEX);
        v[at->ids].idx_max = g.n_expert;
    }

    /* A present operand with an empty extent -- an odd class of no experts -- is a geometry this
     * description cannot build, and a tool skips and counts it rather than inventing one. */
    for (int k = 0; k < at->n_opd; ++k) {
        if (v[k].flags & RAD_OPD_F_ABSENT) continue;
        for (uint32_t r = 0; r < v[k].rank; ++r)
            if (v[k].shape[r] <= 0) return RAD_E_UNSUPPORTED;
    }
    *out = v[operand];
    return RAD_OK;
}

static int gate_up_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return shape(p, n_p, EXL3_GATE_UP, operand, out);
}
static int down_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return shape(p, n_p, EXL3_DOWN, operand, out);
}
static int prep_gu_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return shape(p, n_p, EXL3_PREP_GU, operand, out);
}
static int gate_up4_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return shape(p, n_p, EXL3_GATE_UP4, operand, out);
}
static int down4_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return shape(p, n_p, EXL3_DOWN4, operand, out);
}
static int prep_gu4_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return shape(p, n_p, EXL3_PREP_GU4, operand, out);
}
/* moe_exl3_gate_up_down (and v7's _mod4 form): each operand is described as the half that reads it describes it
 * (exl3_gd_params, exl3_gd4_params). */
static int gd_shape(const RadParam* p, int n_p, int op, int operand, RadOpdDesc* out) {
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
static int gu_dn_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return gd_shape(p, n_p, EXL3_GU_DN, operand, out);
}
static int gu_dn4_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return gd_shape(p, n_p, EXL3_GU_DN4, operand, out);
}

static int gate_up_launch(const RadArgs* a, RadStream s) { (void)s; return exl3ref_launch(a, EXL3_GATE_UP); }
static int gu_dn_launch(const RadArgs* a, RadStream s)   { (void)s; return exl3ref_launch(a, EXL3_GU_DN); }
static int down_launch(const RadArgs* a, RadStream s)    { (void)s; return exl3ref_launch(a, EXL3_DOWN); }
static int prep_gu_launch(const RadArgs* a, RadStream s) { (void)s; return exl3ref_launch(a, EXL3_PREP_GU); }
static int gate_up4_launch(const RadArgs* a, RadStream s) { (void)s; return exl3ref_launch(a, EXL3_GATE_UP4); }
static int down4_launch(const RadArgs* a, RadStream s)    { (void)s; return exl3ref_launch(a, EXL3_DOWN4); }
static int prep_gu4_launch(const RadArgs* a, RadStream s) { (void)s; return exl3ref_launch(a, EXL3_PREP_GU4); }
static int gu_dn4_launch(const RadArgs* a, RadStream s)   { (void)s; return exl3ref_launch(a, EXL3_GU_DN4); }

/* ------------------------------------------------------------------ the rows */

/* Every form, both flags (matched on the decimal, core/util/rad_util.cpp:249-258): the reference serves all. */
#define EXL3REF_DTYPES RAD_CIN("dtype", "exl3k2 exl3k3 exl3k4 exl3k5 exl3k6 exl3k7 exl3k8")
static const RadConstraint c_gate_up[] = {
    EXL3REF_DTYPES, RAD_CIN("codebook", "mul1 mcg 3inst"), RAD_CIN("h_rot", "0 1"), RAD_CIN("a_rot", "0 1"),
};
static const RadConstraint c_down[] = {
    EXL3REF_DTYPES, RAD_CIN("codebook", "mul1 mcg 3inst"), RAD_CIN("h_rot", "0 1"),
};
static const RadConstraint c_prep_gu[] = { EXL3REF_DTYPES };
static const RadConstraint c_gu_dn[] = { EXL3REF_DTYPES, RAD_CIN("codebook", "mul1 mcg 3inst") };
#define N_OF(a) (int)(sizeof(a) / sizeof((a)[0]))

static const RadKernelInfo k_rows[] = {
    {
        .name = "exl3ref_moe_gate_up", .op = EXL3_OP_GATE_UP, .family = "moe",
        .computes = "EXL3 routed experts, gate and up, f16 out: the reference (docs/OPS-exl3.md)",
        .shape = "any M; K % 128, N % 256 (each class)",
        .dtypes = "exl3k2..exl3k8; mul1, mcg, 3inst; a f16 or bf16 (a_rot = 1: prep_gu's xr, f16)",
        .domain = RAD_DOMAIN_HOST, .priority = 0,
        .constraints = c_gate_up, .n_constraints = N_OF(c_gate_up),
        .launch = gate_up_launch, .opd_shape = gate_up_shape,
    },
    {
        .name = "exl3ref_moe_down", .op = EXL3_OP_DOWN, .family = "moe",
        .computes = "EXL3 routed experts, down, bf16 out: the reference (docs/OPS-exl3.md)",
        .shape = "any M; K % 128, N % 128 (each class)",
        .dtypes = "exl3k2..exl3k8; mul1, mcg, 3inst; h f16",
        .domain = RAD_DOMAIN_HOST, .priority = 0,
        .constraints = c_down, .n_constraints = N_OF(c_down),
        .launch = down_launch, .opd_shape = down_shape,
    },
    {
        .name = "exl3ref_moe_prep_gu", .op = EXL3_OP_PREP_GU, .family = "moe",
        .computes = "EXL3 routed experts, gate_up's input rotation as an op (xr): the reference (docs/OPS-exl3.md)",
        .shape = "any M; K % 128 (each class)",
        .dtypes = "a f16 or bf16, suh/suh.1 f16, xr f16",
        .domain = RAD_DOMAIN_HOST, .priority = 0,
        .constraints = c_prep_gu, .n_constraints = N_OF(c_prep_gu),
        .launch = prep_gu_launch, .opd_shape = prep_gu_shape,
    },
    {
        .name = "exl3ref_moe_gate_up_down", .op = EXL3_OP_GU_DN, .family = "moe",
        .computes = "EXL3 routed experts, the decode chain as one op (gate_up a_rot 0 h_rot 1, then down h_rot 1): "
                    "the reference (docs/OPS-exl3.md)",
        .shape = "any M; K % 128, N % 256 (each class)",
        .dtypes = "exl3k2..exl3k8; mul1, mcg, 3inst; a f16 or bf16; y bf16",
        .domain = RAD_DOMAIN_HOST, .priority = 0,
        .constraints = c_gu_dn, .n_constraints = N_OF(c_gu_dn),
        .launch = gu_dn_launch, .opd_shape = gu_dn_shape,
    },
    /* v6: the four-class forms, the same arithmetic over four tables a plane */
    {
        .name = "exl3ref_moe_gate_up_mod4", .op = EXL3_OP_GATE_UP4, .family = "moe",
        .computes = "EXL3 routed experts, gate and up, f16 out, four classes by number mod 4: the reference "
                    "(docs/OPS-exl3.md)",
        .shape = "any M; K % 128, N % 256 (each class)",
        .dtypes = "exl3k2..exl3k8; mul1, mcg, 3inst; a f16 or bf16 (a_rot = 1: prep_gu's xr, f16)",
        .domain = RAD_DOMAIN_HOST, .priority = 0,
        .constraints = c_gate_up, .n_constraints = N_OF(c_gate_up),
        .launch = gate_up4_launch, .opd_shape = gate_up4_shape,
    },
    {
        .name = "exl3ref_moe_down_mod4", .op = EXL3_OP_DOWN4, .family = "moe",
        .computes = "EXL3 routed experts, down, bf16 out, four classes by number mod 4: the reference "
                    "(docs/OPS-exl3.md)",
        .shape = "any M; K % 128, N % 128 (each class)",
        .dtypes = "exl3k2..exl3k8; mul1, mcg, 3inst; h f16",
        .domain = RAD_DOMAIN_HOST, .priority = 0,
        .constraints = c_down, .n_constraints = N_OF(c_down),
        .launch = down4_launch, .opd_shape = down4_shape,
    },
    {
        .name = "exl3ref_moe_prep_gu_mod4", .op = EXL3_OP_PREP_GU4, .family = "moe",
        .computes = "EXL3 routed experts, gate_up's input rotation as an op (xr), four classes by number mod 4: the "
                    "reference (docs/OPS-exl3.md)",
        .shape = "any M; K % 128 (each class)",
        .dtypes = "a f16 or bf16, suh/suh.1 f16, xr f16",
        .domain = RAD_DOMAIN_HOST, .priority = 0,
        .constraints = c_prep_gu, .n_constraints = N_OF(c_prep_gu),
        .launch = prep_gu4_launch, .opd_shape = prep_gu4_shape,
    },
    /* v7: the four-class decode chain as one op */
    {
        .name = "exl3ref_moe_gate_up_down_mod4", .op = EXL3_OP_GU_DN4, .family = "moe",
        .computes = "EXL3 routed experts, the decode chain as one op, four classes by number mod 4 (gate_up_mod4 a_rot "
                    "0 h_rot 1, then down_mod4 h_rot 1): the reference (docs/OPS-exl3.md)",
        .shape = "any M; K % 128, N % 256 (each class)",
        .dtypes = "exl3k2..exl3k8; mul1, mcg, 3inst; a f16 or bf16; y bf16",
        .domain = RAD_DOMAIN_HOST, .priority = 0,
        .constraints = c_gu_dn, .n_constraints = N_OF(c_gu_dn),
        .launch = gu_dn4_launch, .opd_shape = gu_dn4_shape,
    },
};

/* ------------------------------------------------------------------ the exports */

static const RadPluginInfo k_info = {
    RAD_PLUGIN_KERNEL, "exl3ref", "0.1.0",
    "EXL3 routed experts (moe_exl3_gate_up, moe_exl3_down, moe_exl3_prep_gu, moe_exl3_gate_up_down, and their "
    "_mod4 forms): the host reference", "host",
};

uint32_t             rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
const RadPluginInfo* rad_plugin_info(void) { return &k_info; }
int                  rad_plugin_reference(void) { return 1; }

int                rad_kernel_schema_count(void) { return exl3_schema_count(); }
const RadOpSchema* rad_kernel_schema_at(int i) { return exl3_schema_at(i); }
int                rad_kernel_count(void) { return (int)(sizeof k_rows / sizeof k_rows[0]); }
const RadKernelInfo* rad_kernel_at(int i) {
    return i >= 0 && i < rad_kernel_count() ? &k_rows[i] : 0;
}
