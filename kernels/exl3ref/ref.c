/* ref.c -- the host reference for moe_exl3_gate_up and moe_exl3_down: what the two ops compute,
 * written to be read. docs/OPS-exl3.md is the contract, step for step; include/exl3_abi.h holds the
 * schemas, the routing contract and the planes read here (one weight table per plane per class).
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

/* ------------------------------------------------------------------ the arithmetic */

/* xr = f16(had(f16(x * s))) over `cols` values: the input side of every projection. */
static int prologue(const float* x, const uint16_t* s, int64_t cols, float* xr) {
    for (int64_t k = 0; k < cols; ++k) xr[k] = h2f(f2h(x[k] * h2f(s[k])));
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
    const RadTensor *x, *sorted, *off, *y;
    const RadTensor *tab[2][EXL3_N_P];   /* [class][plane], null for a plane the op has not */
    const int32_t   *tok, *eoff;
    int64_t          T, x_rows, x_pitch, y_pitch;
} Io;

/* Entry j of a table: the engine's pointer table (stride[0] == 0) or the stacked form. */
static const void* entry(const RadTensor* t, int64_t j) {
    if (t->stride[0] == 0) return ((const void* const*)t->data)[j];
    return (const uint8_t*)t->data + j * t->stride[0] * rad_dtype_bits(t->dtype) / 8;
}

/* A table of plane p of a [rows, cols] weight: `need` entries of the plane's own dtype, each the
 * plane's elements densely -- codes as [rows, cols] exactly (its rows are bit streams), a scale
 * vector in whatever shape the declaration gave it, by element count. */
static int check_table(const RadTensor* t, int p, int bits, int64_t rows, int64_t cols,
                       int64_t need) {
    int64_t pr, pc, numel = 1, acc = 1;
    exl3_plane_extents(rows, cols, p, &pr, &pc);
    if (t->dtype != (p == EXL3_P_CODES ? exl3_codes_dtype(bits) : (uint32_t)RAD_F16))
        return RAD_E_DTYPE;
    if (t->rank < 2 || t->shape[0] < need) return RAD_E_SHAPE;
    for (uint32_t r = 1; r < t->rank; ++r) numel *= t->shape[r];
    if (numel != pr * pc) return RAD_E_SHAPE;
    if (p == EXL3_P_CODES && (t->rank != 3 || t->shape[1] != pr || t->shape[2] != pc))
        return RAD_E_SHAPE;
    for (int r = (int)t->rank - 1; r >= 1; --r) {           /* each entry dense */
        if (t->shape[r] != 1 && t->stride[r] != acc) return RAD_E_STRIDE;
        acc *= t->shape[r];
    }
    if (t->stride[0] != 0 && (t->stride[0] < numel || t->stride[0] * rad_dtype_bits(t->dtype) % 8))
        return RAD_E_STRIDE;
    if (p != EXL3_P_CODES)
        for (int64_t j = 0; j < need; ++j)
            if (!rad_ptr_aligned(entry(t, j), 2)) return RAD_E_ALIGN;
    return RAD_OK;
}

/* Everything a launch refuses, before anything is written. */
static int check(const RadArgs* a, const Exl3Geom* g, const Exl3Positions* at, Io* io) {
    io->x = rad_arg_in(a, at->x);
    io->sorted = rad_arg_in(a, at->sorted);
    io->off = rad_arg_in(a, at->offset);
    io->y = rad_arg_in(a, at->y);
    for (int c = 0; c < 2; ++c)
        for (int p = 0; p < EXL3_N_P; ++p)
            io->tab[c][p] = at->plane[c][p] >= 0 ? rad_arg_in(a, at->plane[c][p]) : 0;

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

    for (int c = 0; c < g->n_class; ++c)
        for (int p = 0; p < EXL3_N_P; ++p) {
            if (!io->tab[c][p]) continue;
            const int rc = check_table(io->tab[c][p], p, g->bits, g->rows[c], g->cols[c],
                                       exl3_class_entries(g, c));
            if (rc != RAD_OK) return rc;
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
    const int64_t R = g->rows[c], C = g->cols[c], j = exl3_entry_of(g, e);
    const uint16_t* suh = entry(io->tab[c][EXL3_P_SUH], j);
    const uint16_t* suh1 = g->gate_up ? entry(io->tab[c][EXL3_P_SUH1], j) : 0;
    const uint16_t* svh = entry(io->tab[c][EXL3_P_SVH], j);
    const Exl3View v = {
        .rows = R, .cols = C, .bits = g->bits, .codebook = g->codebook,
        .codes = entry(io->tab[c][EXL3_P_CODES], j),
        .codes_row_bytes = rad_enc_row_bytes(exl3_codes_dtype(g->bits), C),
        .suh = suh, .suh1 = suh1, .svh = svh,
    };
    if (exl3_decode_what_rows(&v, 0, R, wk->what) != EXL3_OK) return RAD_E_SHAPE;

    for (int64_t i = io->eoff[e]; i < io->eoff[e + 1]; ++i) {
        if (io->tok[i] < 0) continue;                       /* dropped: left zero */
        if (g->gate_up) {
            const int64_t W = R / 2;
            load_row(io, io->tok[i] / g->top_k, C, wk->x);
            for (int p = 0; p < 2; ++p) {                   /* gate with suh, up with suh.1 */
                const int rc = prologue(wk->x, p ? suh1 : suh, C, wk->xr);
                if (rc != RAD_OK) return rc;
                gemv(wk->xr, wk->what + p * W * C, W, C, wk->acc + p * W);
                if (exl3_had128(wk->acc + p * W, W) != EXL3_OK) return RAD_E_SHAPE;
                for (int64_t n = 0; n < W; ++n)
                    wk->acc[p * W + n] = wk->acc[p * W + n] * h2f(svh[p * W + n]);
            }
            uint16_t* h = (uint16_t*)io->y->data + i * io->y_pitch;
            for (int64_t n = 0; n < W; ++n) h[n] = f2h(silu(wk->acc[n]) * wk->acc[W + n]);
        } else {
            load_row(io, i, C, wk->x);
            const int rc = prologue(wk->x, suh, C, wk->xr);
            if (rc != RAD_OK) return rc;
            gemv(wk->xr, wk->what, R, C, wk->acc);
            if (exl3_had128(wk->acc, R) != EXL3_OK) return RAD_E_SHAPE;
            uint16_t* y = (uint16_t*)io->y->data + i * io->y_pitch;
            for (int64_t n = 0; n < R; ++n) y[n] = rad_f32_to_bf16(wk->acc[n] * h2f(svh[n]));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ the launch */

int exl3ref_launch(const RadArgs* a, int gate_up) {
    const Exl3Positions* at = exl3_positions(gate_up);
    if (a->n_t < at->n_opd) return RAD_E_INVAL;
    if (!rad_arg_in(a, at->x) || !rad_arg_in(a, at->sorted) || !rad_arg_in(a, at->offset) ||
        !rad_arg_in(a, at->y))
        return RAD_E_INVAL;
    /* The even tables are required; the odd ones come all together or not at all. */
    int odd = 0, odd_planes = 0;
    for (int p = 0; p < EXL3_N_P; ++p) {
        if (at->plane[0][p] < 0) continue;
        if (!rad_arg_in(a, at->plane[0][p])) return RAD_E_INVAL;
        ++odd_planes;
        odd += rad_arg_in(a, at->plane[1][p]) != 0;
    }
    if (odd != 0 && odd != odd_planes) return RAD_E_INVAL;

    Exl3Geom g;
    int rc = exl3_geom_parse(a->p, a->n_p, gate_up, odd != 0, &g);
    if (rc != RAD_OK) return rc;
    Io io;
    if ((rc = check(a, &g, at, &io)) != RAD_OK) return rc;

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
