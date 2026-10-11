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
 * when the geometry names no odd class (exl3_geom_two_classes). The routing operands are described
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

static int shape(const RadParam* p, int n_p, int gate_up, int operand, RadOpdDesc* out) {
    Exl3Geom g;
    const Exl3Positions* at = exl3_positions(gate_up);
    const int two = exl3_geom_two_classes(p, n_p);
    if (exl3_geom_parse(p, n_p, gate_up, two, &g) != RAD_OK) return RAD_E_UNSUPPORTED;
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
