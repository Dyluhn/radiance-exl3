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
 * layout hook: what a host row reads is the plane record (include/exl3_abi.h), and see
 * notes/p2.md for what that means for a host-only engine at radiance 1.3.0. */
#include "exl3ref.h"

/* ------------------------------------------------------------------ the shape hook
 *
 * Every operand of every case a tool builds cold, described from the parameters: what `launch`
 * reads, at the band's high M. w_odd is ABSENT when the geometry names no odd class
 * (exl3_geom_two_classes). The routing operands are described so a tool draws VALID ones: the flat
 * indices below M * top_k, and expert_offset as a cumulative run table (RAD_OPD_F_IDX_CU). A table
 * entry is any bytes -- every symbol sequence is a valid trellis -- which leaves the scale regions
 * free to draw non-finite f16; a tool comparing two libraries sees the same non-finite rows on
 * both. */
static void desc(RadOpdDesc* d, uint32_t dt, int64_t d0, int64_t d1, uint32_t fill) {
    memset(d, 0, sizeof *d);
    d->dtype = dt;
    d->rank = d1 > 0 ? 2 : 1;
    d->shape[0] = d0;
    d->shape[1] = d1 > 0 ? d1 : 0;
    d->fill = fill;
    d->idx_const = -1;
}

static int shape(const RadParam* p, int n_p, int gate_up, int operand, RadOpdDesc* out) {
    Exl3Geom g;
    const int two = exl3_geom_two_classes(p, n_p);
    if (exl3_geom_parse(p, n_p, gate_up, two, &g) != RAD_OK) return RAD_E_UNSUPPORTED;
    const int64_t T = g.M * g.top_k;
    RadOpdDesc v[EXL3_N_OPD];
    desc(&v[EXL3_OPD_X], RAD_F16, gate_up ? g.M : T, g.x_cols, RAD_FILL_NORMAL);
    for (int c = 0; c < 2; ++c) {
        RadOpdDesc* d = &v[c ? EXL3_OPD_W_ODD : EXL3_OPD_W];
        Exl3RecLayout L;
        if (c >= g.n_class) {
            desc(d, RAD_DT_INVALID, 0, 0, RAD_FILL_NORMAL);
            d->rank = 0;
            d->shape[0] = 0;
            d->flags = RAD_OPD_F_ABSENT;
            continue;
        }
        if (exl3_rec_layout(g.rows[c], g.cols[c], g.bits, gate_up, &L) != RAD_OK)
            return RAD_E_UNSUPPORTED;
        desc(d, RAD_U8, exl3_class_entries(&g, c), L.bytes, RAD_FILL_INDEX);
        d->idx_max = 256;
    }
    desc(&v[EXL3_OPD_SORTED], RAD_I32, T, 0, RAD_FILL_INDEX);
    v[EXL3_OPD_SORTED].idx_max = T;
    desc(&v[EXL3_OPD_OFFSET], RAD_I32, g.n_expert + 1, 0, RAD_FILL_INDEX);
    v[EXL3_OPD_OFFSET].flags = RAD_OPD_F_IDX_CU;
    desc(&v[EXL3_OPD_Y], gate_up ? RAD_F16 : RAD_BF16, T, g.y_cols, RAD_FILL_NORMAL);

    if (operand < 0 || operand >= EXL3_N_OPD || !out) return RAD_E_SHAPE;
    /* A present operand with an empty extent -- an odd class of no experts -- is a geometry this
     * description cannot build, and a tool skips and counts it rather than inventing one. */
    for (int k = 0; k < EXL3_N_OPD; ++k) {
        if (v[k].flags & RAD_OPD_F_ABSENT) continue;
        for (uint32_t r = 0; r < v[k].rank; ++r)
            if (v[k].shape[r] <= 0) return RAD_E_UNSUPPORTED;
    }
    *out = v[operand];
    return RAD_OK;
}

static int gate_up_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return shape(p, n_p, 1, operand, out);
}
static int down_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return shape(p, n_p, 0, operand, out);
}

static int gate_up_launch(const RadArgs* a, RadStream s) { (void)s; return exl3ref_launch(a, 1); }
static int down_launch(const RadArgs* a, RadStream s)    { (void)s; return exl3ref_launch(a, 0); }

/* ------------------------------------------------------------------ the rows */

static const RadConstraint c_format[] = {
    RAD_CIN("dtype", "exl3k2 exl3k3 exl3k4 exl3k5 exl3k6 exl3k7 exl3k8"),
    RAD_CIN("codebook", "mul1 mcg 3inst"),
};

static const RadKernelInfo k_rows[] = {
    {
        .name = "exl3ref_moe_gate_up", .op = EXL3_OP_GATE_UP, .family = "moe",
        .computes = "EXL3 routed experts, gate and up, f16 out: the reference (docs/OPS-exl3.md)",
        .shape = "any M; K % 128, N % 256 (each class)",
        .dtypes = "exl3k2..exl3k8; mul1, mcg, 3inst; a f16 or bf16",
        .domain = RAD_DOMAIN_HOST, .priority = 0,
        .constraints = c_format, .n_constraints = 2,
        .launch = gate_up_launch, .opd_shape = gate_up_shape,
    },
    {
        .name = "exl3ref_moe_down", .op = EXL3_OP_DOWN, .family = "moe",
        .computes = "EXL3 routed experts, down, bf16 out: the reference (docs/OPS-exl3.md)",
        .shape = "any M; K % 128, N % 128 (each class)",
        .dtypes = "exl3k2..exl3k8; mul1, mcg, 3inst; h f16",
        .domain = RAD_DOMAIN_HOST, .priority = 0,
        .constraints = c_format, .n_constraints = 2,
        .launch = down_launch, .opd_shape = down_shape,
    },
};

/* ------------------------------------------------------------------ the exports */

static const RadPluginInfo k_info = {
    RAD_PLUGIN_KERNEL, "exl3ref", "0.1.0",
    "EXL3 routed-expert GEMMs (moe_exl3_gate_up, moe_exl3_down): the host reference", "host",
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
