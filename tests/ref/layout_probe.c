/* layout_probe.c -- a TEST plugin: exl3ref's host rows behind a layout hook, so the harness's layout path
 * (tests/ref/exl3ref_run.cpp: layout, relayout, unrelayout, the stored-form description) is exercised
 * before a device library depends on it. Never installed.
 *
 * Every codes table is stored XOR 0xFF (tag "exl3probe.xor"); the scale tables answer
 * RAD_E_UNSUPPORTED and stay as they are. `launch` undoes the XOR into a host copy and calls the
 * reference, so a correct harness gets exl3ref's output to the bit. Built twice: as is, and with
 * -DLAYOUT_PROBE_BAD_INVERSE, whose unrelayout returns the stored bytes untouched -- the case the
 * harness's round-trip check has to catch (the negative control). */
#include "exl3ref.h"

#include <stdlib.h>
#include <string.h>

static int codes_table(int gate_up, int operand) {
    const Exl3Positions* at = exl3_positions(gate_up);
    return operand == at->plane[0][EXL3_P_CODES] || operand == at->plane[1][EXL3_P_CODES];
}

static int layout(int gate_up, int operand, const RadTensor* planes, int n, RadLayout* out) {
    if (!codes_table(gate_up, operand)) return RAD_E_UNSUPPORTED;
    if (n != 1 || planes[0].rank != 2) return RAD_E_SHAPE;
    memset(out, 0, sizeof *out);
    out->tag = "exl3probe.xor";
    out->dtype = planes[0].dtype;
    out->rank = 2;
    out->shape[0] = planes[0].shape[0];
    out->shape[1] = planes[0].shape[1];
    out->align = 16;
    out->bytes = planes[0].shape[0] * rad_enc_row_bytes(planes[0].dtype, planes[0].shape[1]);
    return RAD_OK;
}

static void xor_copy(void* dst, const void* src, int64_t bytes) {
    for (int64_t i = 0; i < bytes; ++i) ((uint8_t*)dst)[i] = ((const uint8_t*)src)[i] ^ 0xFF;
}

static int64_t plane_bytes(const RadTensor* p) { return p->shape[0] * rad_enc_row_bytes(p->dtype, p->shape[1]); }

static int gu_layout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,
                     const RadTensor* planes, int n, RadLayout* out) {
    (void)p; (void)n_p; (void)e; (void)sel;
    return layout(1, o, planes, n, out);
}
static int dn_layout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,
                     const RadTensor* planes, int n, RadLayout* out) {
    (void)p; (void)n_p; (void)e; (void)sel;
    return layout(0, o, planes, n, out);
}
static int relayout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,
                    const RadTensor* planes, int n, void* dst, int64_t dst_bytes) {
    (void)p; (void)n_p; (void)o; (void)e; (void)sel;
    if (n != 1 || dst_bytes != plane_bytes(&planes[0])) return RAD_E_SHAPE;
    xor_copy(dst, planes[0].data, dst_bytes);
    return RAD_OK;
}
static int unrelayout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,
                      const void* src, int64_t src_bytes, const RadTensor* planes, int n) {
    (void)p; (void)n_p; (void)o; (void)e; (void)sel;
    if (n != 1 || src_bytes != plane_bytes(&planes[0])) return RAD_E_SHAPE;
#ifdef LAYOUT_PROBE_BAD_INVERSE
    memcpy(planes[0].data, src, (size_t)src_bytes);
#else
    xor_copy(planes[0].data, src, src_bytes);
#endif
    return RAD_OK;
}

/* Undo the XOR of every codes table into a stacked host copy, then run the reference. */
static int launch(const RadArgs* a, int gate_up) {
    const Exl3Positions* at = exl3_positions(gate_up);
    if (a->n_t < at->n_opd || a->n_t > 12) return RAD_E_INVAL;
    RadTensor t[12];
    void* copy[2] = { 0, 0 };
    memcpy(t, a->t, (size_t)a->n_t * sizeof *t);
    int rc = RAD_OK;
    for (int c = 0; c < 2 && rc == RAD_OK; ++c) {
        RadTensor* x = &t[at->plane[c][EXL3_P_CODES]];
        if (!x->data) continue;
        if (x->rank != 3 || x->stride[0] != 0) { rc = RAD_E_SHAPE; break; }
        const int64_t ent = x->shape[1] * rad_enc_row_bytes(x->dtype, x->shape[2]);
        if (!(copy[c] = malloc((size_t)(x->shape[0] * ent + 1)))) { rc = RAD_E_NOMEM; break; }
        for (int64_t j = 0; j < x->shape[0]; ++j)
            xor_copy((uint8_t*)copy[c] + j * ent, ((void* const*)x->data)[j], ent);
        x->data = copy[c];
        x->stride[0] = x->shape[1] * x->shape[2];
    }
    RadArgs b = *a;
    b.t = t;
    if (rc == RAD_OK) rc = exl3ref_launch(&b, gate_up);
    free(copy[0]);
    free(copy[1]);
    return rc;
}
static int gu_launch(const RadArgs* a, RadStream s) { (void)s; return launch(a, 1); }
static int dn_launch(const RadArgs* a, RadStream s) { (void)s; return launch(a, 0); }

static const RadKernelInfo k_rows[] = {
    { .name = "layout_probe_gate_up", .op = EXL3_OP_GATE_UP, .family = "moe",
      .computes = "exl3ref behind an XOR codes layout (a test plugin)", .shape = "", .dtypes = "",
      .domain = RAD_DOMAIN_HOST, .launch = gu_launch, .layout = gu_layout, .relayout = relayout,
      .unrelayout = unrelayout },
    { .name = "layout_probe_down", .op = EXL3_OP_DOWN, .family = "moe",
      .computes = "exl3ref behind an XOR codes layout (a test plugin)", .shape = "", .dtypes = "",
      .domain = RAD_DOMAIN_HOST, .launch = dn_launch, .layout = dn_layout, .relayout = relayout,
      .unrelayout = unrelayout },
};

static const RadPluginInfo k_info = { RAD_PLUGIN_KERNEL, "layout_probe", "0", "a test plugin", "host" };

uint32_t             rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
const RadPluginInfo* rad_plugin_info(void) { return &k_info; }
int                  rad_kernel_schema_count(void) { return exl3_schema_count(); }
const RadOpSchema*   rad_kernel_schema_at(int i) { return exl3_schema_at(i); }
int                  rad_kernel_count(void) { return 2; }
const RadKernelInfo* rad_kernel_at(int i) { return i >= 0 && i < 2 ? &k_rows[i] : 0; }
