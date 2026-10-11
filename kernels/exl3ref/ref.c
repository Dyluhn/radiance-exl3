/* ref.c -- the host reference for moe_exl3_gate_up and moe_exl3_down: what the two ops compute,
 * written to be read. docs/OPS-exl3.md is the contract, step for step; include/exl3_abi.h holds the
 * schemas, the routing contract and the table entry (the plane record) read here.
 *
 * Slow on purpose: one row at a time, one expert's weight decoded once per launch by the oracle
 * decoder (common/exl3_decode.h), f32 sums in ascending index order. Built with -ffp-contract=off:
 * a product and a sum are two roundings here, as the contract says, never one.
 *
 * Host memory is allocated per launch (the decoded weight of the largest class), as libref's
 * grouped GEMM does; a reference is never on a served path (rad_plugin_reference). */
#include "exl3ref.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static float    h2f(uint16_t h) { return rad_f16_to_f32(h); }
static uint16_t f2h(float f)    { return rad_f32_to_f16(f); }

/* An f16 element of a plane record, read without assuming the region's alignment. */
static float rec_f16(const uint8_t* base, int64_t off, int64_t i) {
    uint16_t v;
    memcpy(&v, base + off + 2 * i, 2);
    return h2f(v);
}

/* ------------------------------------------------------------------ the arithmetic */

/* xr = f16(had(f16(x * s))) over `cols` values: the input side of every projection. */
static int prologue(const float* x, const uint8_t* rec, int64_t s_off, int64_t cols, float* xr) {
    for (int64_t k = 0; k < cols; ++k) xr[k] = h2f(f2h(x[k] * rec_f16(rec, s_off, k)));
    if (exl3_had128(xr, cols) != EXL3_OK) return RAD_E_SHAPE;
    for (int64_t k = 0; k < cols; ++k) xr[k] = h2f(f2h(xr[k]));
    return RAD_OK;
}

/* acc[n] = sum over k of xr[k] * what[n][k], n < rows, k ascending. */
static void gemv(const float* xr, const float* what, int64_t rows, int64_t cols, float* acc) {
    for (int64_t n = 0; n < rows; ++n) {
        const float* w = what + n * cols;
        float s = 0.0f;
        for (int64_t k = 0; k < cols; ++k) s = s + xr[k] * w[k];
        acc[n] = s;
    }
}

static float silu(float g) { return g * (1.0f / (1.0f + expf(-g))); }

/* ------------------------------------------------------------------ the operands */

typedef struct Io {
    const RadTensor *x, *w[2], *sorted, *off, *y;
    const int32_t   *tok, *eoff;
    int64_t          T, x_rows, x_pitch, y_pitch;
    Exl3RecLayout    rec[2];
} Io;

static const uint8_t* entry(const RadTensor* t, int64_t j) {
    if (t->stride[0] == 0) return ((const uint8_t* const*)t->data)[j];   /* the engine's table */
    return (const uint8_t*)t->data + j * t->stride[0];                     /* the stacked form */
}

/* Everything a launch refuses, before anything is written. */
static int check(const RadArgs* a, const Exl3Geom* g, Io* io) {
    io->x = rad_arg_in(a, EXL3_OPD_X);
    io->w[0] = rad_arg_in(a, EXL3_OPD_W);
    io->w[1] = rad_arg_in(a, EXL3_OPD_W_ODD);
    io->sorted = rad_arg_in(a, EXL3_OPD_SORTED);
    io->off = rad_arg_in(a, EXL3_OPD_OFFSET);
    io->y = rad_arg_in(a, EXL3_OPD_Y);

    const uint32_t xdt = io->x->dtype, ydt = io->y->dtype;
    if (g->gate_up ? (xdt != RAD_F16 && xdt != RAD_BF16) || ydt != RAD_F16
                   : xdt != RAD_F16 || ydt != RAD_BF16)
        return RAD_E_DTYPE;
    if (io->sorted->dtype != RAD_I32 || io->off->dtype != RAD_I32) return RAD_E_DTYPE;
    if (!rad_tensor_is_contiguous(io->sorted) || !rad_tensor_is_contiguous(io->off))
        return RAD_E_STRIDE;
    if (io->x->rank < 2 || io->y->rank < 2) return RAD_E_SHAPE;
    if (io->x->stride[io->x->rank - 1] != 1 || io->y->stride[io->y->rank - 1] != 1)
        return RAD_E_STRIDE;

    io->T = rad_tensor_rows(io->y);
    io->x_rows = rad_tensor_rows(io->x);
    io->x_pitch = rad_tensor_row_pitch(io->x);
    io->y_pitch = rad_tensor_row_pitch(io->y);
    if (rad_tensor_lastdim(io->y) != g->y_cols || rad_tensor_lastdim(io->x) < g->x_cols)
        return RAD_E_SHAPE;
    if (rad_tensor_numel(io->sorted) < io->T || rad_tensor_numel(io->off) < g->n_expert + 1)
        return RAD_E_SHAPE;
    if (!g->gate_up && io->x_rows < io->T) return RAD_E_SHAPE;
    io->tok = (const int32_t*)io->sorted->data;
    io->eoff = (const int32_t*)io->off->data;

    /* The tables: u8 entries of exactly one plane record, enough of them for the class. */
    for (int c = 0; c < g->n_class; ++c) {
        const RadTensor* t = io->w[c];
        const int rc = exl3_rec_layout(g->rows[c], g->cols[c], g->bits, g->gate_up, &io->rec[c]);
        if (rc != RAD_OK) return rc;
        if (t->dtype != RAD_U8) return RAD_E_DTYPE;
        if (t->rank != 2 || t->shape[0] < exl3_class_entries(g, c) ||
            t->shape[1] != io->rec[c].bytes)
            return RAD_E_SHAPE;
        if (t->stride[1] != 1) return RAD_E_STRIDE;
        for (int64_t j = 0; j < exl3_class_entries(g, c); ++j)
            if (!rad_ptr_aligned(entry(t, j), EXL3_REC_ALIGN)) return RAD_E_ALIGN;
    }

    /* The routing: a run table from >= 0, never past T; and gate_up's tokens inside `a`. */
    if (io->eoff[0] < 0 || io->eoff[g->n_expert] > io->T) return RAD_E_SHAPE;
    for (int64_t e = 0; e < g->n_expert; ++e)
        if (io->eoff[e + 1] < io->eoff[e]) return RAD_E_SHAPE;
    if (g->gate_up)
        for (int64_t i = io->eoff[0]; i < io->eoff[g->n_expert]; ++i)
            if (io->tok[i] >= 0 && io->tok[i] / g->top_k >= io->x_rows) return RAD_E_SHAPE;
    return RAD_OK;
}

/* Row r of the activation, widened to f32. */
static void load_row(const Io* io, int64_t r, int64_t cols, float* out) {
    const uint16_t* p = (const uint16_t*)io->x->data + r * io->x_pitch;
    for (int64_t k = 0; k < cols; ++k)
        out[k] = io->x->dtype == RAD_BF16 ? rad_bf16_to_f32(p[k]) : h2f(p[k]);
}

/* ------------------------------------------------------------------ one expert's rows */

typedef struct Work { float *what, *x, *xr, *acc; } Work;

static int expert_rows(const Exl3Geom* g, const Io* io, int64_t e, Work* wk) {
    const int c = exl3_class_of(g, e);
    const int64_t R = g->rows[c], C = g->cols[c];
    const Exl3RecLayout* L = &io->rec[c];
    const uint8_t* rec = entry(io->w[c], exl3_entry_of(g, e));
    const Exl3View v = {
        .rows = R, .cols = C, .bits = g->bits, .codebook = g->codebook,
        .codes = rec + L->codes, .codes_row_bytes = L->codes_row_bytes,
        .suh = (const uint16_t*)(rec + L->suh), .svh = (const uint16_t*)(rec + L->svh),
        .suh1 = g->gate_up ? (const uint16_t*)(rec + L->suh1) : 0,
    };
    if (exl3_decode_what_rows(&v, 0, R, wk->what) != EXL3_OK) return RAD_E_SHAPE;

    for (int64_t i = io->eoff[e]; i < io->eoff[e + 1]; ++i) {
        if (io->tok[i] < 0) continue;                       /* dropped: left zero */
        if (g->gate_up) {
            const int64_t W = R / 2;
            load_row(io, io->tok[i] / g->top_k, C, wk->x);
            for (int p = 0; p < 2; ++p) {                   /* gate with suh, up with suh.1 */
                int rc = prologue(wk->x, rec, p ? L->suh1 : L->suh, C, wk->xr);
                if (rc != RAD_OK) return rc;
                gemv(wk->xr, wk->what + p * W * C, W, C, wk->acc + p * W);
                if (exl3_had128(wk->acc + p * W, W) != EXL3_OK) return RAD_E_SHAPE;
                for (int64_t n = 0; n < W; ++n)
                    wk->acc[p * W + n] = wk->acc[p * W + n] * rec_f16(rec, L->svh, p * W + n);
            }
            uint16_t* h = (uint16_t*)io->y->data + i * io->y_pitch;
            for (int64_t n = 0; n < W; ++n) h[n] = f2h(silu(wk->acc[n]) * wk->acc[W + n]);
        } else {
            load_row(io, i, C, wk->x);
            const int rc = prologue(wk->x, rec, L->suh, C, wk->xr);
            if (rc != RAD_OK) return rc;
            gemv(wk->xr, wk->what, R, C, wk->acc);
            if (exl3_had128(wk->acc, R) != EXL3_OK) return RAD_E_SHAPE;
            uint16_t* y = (uint16_t*)io->y->data + i * io->y_pitch;
            for (int64_t n = 0; n < R; ++n) y[n] = rad_f32_to_bf16(wk->acc[n] * rec_f16(rec, L->svh, n));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ the launch */

int exl3ref_launch(const RadArgs* a, int gate_up) {
    for (int k = 0; k < EXL3_N_OPD; ++k)
        if (k != EXL3_OPD_W_ODD && !rad_arg_in(a, k)) return RAD_E_INVAL;
    Exl3Geom g;
    int rc = exl3_geom_parse(a->p, a->n_p, gate_up, rad_arg_in(a, EXL3_OPD_W_ODD) != 0, &g);
    if (rc != RAD_OK) return rc;
    Io io;
    if ((rc = check(a, &g, &io)) != RAD_OK) return rc;

    /* Every row zero first: dropped slots, rows past the live count, a narrower class's tail. */
    for (int64_t i = 0; i < io.T; ++i)
        memset((uint16_t*)io.y->data + i * io.y_pitch, 0, (size_t)g.y_cols * 2);

    int64_t most = 0, widest = 0;
    for (int c = 0; c < g.n_class; ++c) {
        if (g.rows[c] * g.cols[c] > most) most = g.rows[c] * g.cols[c];
        if (g.rows[c] > widest) widest = g.rows[c];
    }
    Work wk = { malloc((size_t)most * sizeof(float)), malloc((size_t)g.x_cols * sizeof(float)),
                malloc((size_t)g.x_cols * sizeof(float)), malloc((size_t)widest * sizeof(float)) };
    rc = wk.what && wk.x && wk.xr && wk.acc ? RAD_OK : RAD_E_NOMEM;
    for (int64_t e = 0; e < g.n_expert && rc == RAD_OK; ++e) {
        int live = 0;
        for (int64_t i = io.eoff[e]; i < io.eoff[e + 1] && !live; ++i) live = io.tok[i] >= 0;
        if (live) rc = expert_rows(&g, &io, e, &wk);
    }
    free(wk.what); free(wk.x); free(wk.xr); free(wk.acc);
    return rc;
}
