/* ref.c -- the host reference for moe_exl3_gate_up and moe_exl3_down: what the two ops compute,
 * written to be read. docs/OPS-exl3.md is the contract, step for step; include/exl3_abi.h holds the
 * schemas, the routing contract and the planes read here (one weight table per plane per class),
 * the h_rot flag (version 3): gate_up writing h already rotated for down, down reading it so; and
 * (version 4) moe_exl3_prep_gu, gate_up's input rotation as its own op, with gate_up's a_rot flag
 * reading what it wrote -- the one prologue below computes it in every case. The _mod4 forms (versions 6, 7) are
 * the same code: a class is e % n_class and an entry e / n_class, with n_class 4.
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
    const RadTensor *tab[EXL3_MAX_CLS][EXL3_N_P];   /* [class][plane], null for a plane the op has not */
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
    for (int c = 0; c < EXL3_MAX_CLS; ++c)
        for (int p = 0; p < EXL3_N_P; ++p)
            io->tab[c][p] = at->plane[c][p] >= 0 ? rad_arg_in(a, at->plane[c][p]) : 0;

    /* The token rows (gate_up a_rot = 0, prep_gu) are f16 or bf16; a sorted-row input (down's h, gate_up's
     * xr at a_rot = 1) is f16; gate_up and prep_gu write f16, down bf16. */
    const uint32_t xdt = io->x->dtype, ydt = io->y->dtype;
    const int token_rows = g->gate_up && !g->a_rot;
    if (token_rows ? xdt != RAD_F16 && xdt != RAD_BF16 : xdt != RAD_F16) return RAD_E_DTYPE;
    if (ydt != (g->op == EXL3_DOWN ? (uint32_t)RAD_BF16 : (uint32_t)RAD_F16)) return RAD_E_DTYPE;
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
    if (rad_tensor_lastdim(io->y) != g->y_cols ||
        rad_tensor_lastdim(io->x) < (g->a_rot ? 2 * g->x_cols : g->x_cols))
        return RAD_E_SHAPE;
    if (rad_tensor_numel(io->sorted) < io->T || rad_tensor_numel(io->off) < g->n_expert + 1)
        return RAD_E_SHAPE;
    if (!token_rows && io->x_rows < io->T) return RAD_E_SHAPE;
    io->tok = (const int32_t*)io->sorted->data;
    io->eoff = (const int32_t*)io->off->data;

    for (int c = 0; c < g->n_class; ++c)
        for (int p = 0; p < EXL3_N_P; ++p) {
            if (!io->tab[c][p] || (p == EXL3_P_DN_SUH && !g->h_rot)) continue;   /* not read */
            const int rc = check_table(io->tab[c][p], p, g->bits, g->rows[c], g->cols[c],
                                       exl3_class_entries(g, c));
            if (rc != RAD_OK) return rc;
        }

    /* The routing: a run table from >= 0, never past T; and gate_up's tokens inside `a`. */
    if (io->eoff[0] < 0 || io->eoff[g->n_expert] > io->T) return RAD_E_SHAPE;
    for (int64_t e = 0; e < g->n_expert; ++e)
        if (io->eoff[e + 1] < io->eoff[e]) return RAD_E_SHAPE;
    if (token_rows)
        for (int64_t i = io->eoff[0]; i < io->eoff[g->n_expert]; ++i)
            if (io->tok[i] >= 0 && io->tok[i] / g->top_k >= io->x_rows) return RAD_E_SHAPE;
    return RAD_OK;
}

/* Columns [col0, col0 + cols) of row r of the input, widened to f32. */
static void load_row(const Io* io, int64_t r, int64_t col0, int64_t cols, float* out) {
    const uint16_t* p = (const uint16_t*)io->x->data + r * io->x_pitch + col0;
    for (int64_t k = 0; k < cols; ++k)
        out[k] = io->x->dtype == RAD_BF16 ? rad_bf16_to_f32(p[k]) : h2f(p[k]);
}

/* ------------------------------------------------------------------ one expert's rows */

typedef struct Work { float *what, *x, *xr, *acc; } Work;

/* moe_exl3_prep_gu: gate_up's prologue for both halves of each of expert e's rows, side by side (xr). Only the
 * rows of the run are written; one whose sorted_tok is negative is zero across its 2 * C. */
static int prep_rows(const Exl3Geom* g, const Io* io, int64_t e, Work* wk) {
    const int c = exl3_class_of(g, e);
    const int64_t C = g->cols[c], j = exl3_entry_of(g, e);
    const uint16_t* s[2] = { entry(io->tab[c][EXL3_P_SUH], j), entry(io->tab[c][EXL3_P_SUH1], j) };
    for (int64_t i = io->eoff[e]; i < io->eoff[e + 1]; ++i) {
        uint16_t* xr = (uint16_t*)io->y->data + i * io->y_pitch;
        if (io->tok[i] < 0) { memset(xr, 0, (size_t)(2 * C) * 2); continue; }
        load_row(io, io->tok[i] / g->top_k, 0, C, wk->x);
        for (int p = 0; p < 2; ++p) {                       /* gate with suh, up with suh.1 */
            const int rc = prologue(wk->x, s[p], C, wk->xr);
            if (rc != RAD_OK) return rc;
            for (int64_t k = 0; k < C; ++k) xr[p * C + k] = f2h(wk->xr[k]);
        }
    }
    return RAD_OK;
}

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
            if (!g->a_rot) load_row(io, io->tok[i] / g->top_k, 0, C, wk->x);
            for (int p = 0; p < 2; ++p) {                   /* gate with suh, up with suh.1 */
                if (g->a_rot) {
                    load_row(io, i, p * C, C, wk->xr);      /* prep_gu's xr: this half, rotated */
                } else {
                    const int rc = prologue(wk->x, p ? suh1 : suh, C, wk->xr);
                    if (rc != RAD_OK) return rc;
                }
                gemv(wk->xr, wk->what + p * W * C, W, C, wk->acc + p * W);
                if (exl3_had128(wk->acc + p * W, W) != EXL3_OK) return RAD_E_SHAPE;
                for (int64_t n = 0; n < W; ++n)
                    wk->acc[p * W + n] = wk->acc[p * W + n] * h2f(svh[p * W + n]);
            }
            uint16_t* h = (uint16_t*)io->y->data + i * io->y_pitch;
            if (!g->h_rot) {
                for (int64_t n = 0; n < W; ++n) h[n] = f2h(silu(wk->acc[n]) * wk->acc[W + n]);
                continue;
            }
            /* h_rot = 1: h, then down's first step with down's suh of this expert -- the same
             * prologue down would run, so the pair's bits do not depend on which op runs it. */
            for (int64_t n = 0; n < W; ++n) wk->x[n] = h2f(f2h(silu(wk->acc[n]) * wk->acc[W + n]));
            const int rc = prologue(wk->x, entry(io->tab[c][EXL3_P_DN_SUH], j), W, wk->xr);
            if (rc != RAD_OK) return rc;
            for (int64_t n = 0; n < W; ++n) h[n] = f2h(wk->xr[n]);
        } else {
            if (g->h_rot) {
                load_row(io, i, 0, C, wk->xr);              /* already rotated by gate_up */
            } else {
                load_row(io, i, 0, C, wk->x);
                const int rc = prologue(wk->x, suh, C, wk->xr);
                if (rc != RAD_OK) return rc;
            }
            gemv(wk->xr, wk->what, R, C, wk->acc);
            if (exl3_had128(wk->acc, R) != EXL3_OK) return RAD_E_SHAPE;
            uint16_t* y = (uint16_t*)io->y->data + i * io->y_pitch;
            for (int64_t n = 0; n < R; ++n) y[n] = rad_f32_to_bf16(wk->acc[n] * h2f(svh[n]));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ the launch */

/* moe_exl3_gate_up_down (version 5): its definition, run -- gate_up(a_rot = 0, h_rot = 1) into a host h, then
 * down(h_rot = 1) into y (include/exl3_abi.h, the two views). Every refusal is the halves' own. Version 7's
 * moe_exl3_gate_up_down_mod4 (op EXL3_GU_DN4) the same over its four-class views. */
static int launch_gu_dn(const RadArgs* a, int op) {
    const int mod4 = op == EXL3_GU_DN4;
    const Exl3Positions* gd = exl3_positions(op);
    if (a->n_t < gd->n_opd || !rad_arg_in(a, gd->y)) return RAD_E_INVAL;
    RadParam pg[15], pd[14];
    int ng = 0, nd = 0;
    if (mod4) exl3_gd4_params(a->p, a->n_p, pg, &ng, pd, &nd);
    else exl3_gd_params(a->p, a->n_p, pg, &ng, pd, &nd);
    const int64_t T = rad_tensor_rows(rad_arg_in(a, gd->y));
    int64_t W = 0;
    if (mod4) {
        static const char* const nq[EXL3_MAX_CLS] = { "N", "N_1", "N_2", "N_3" };
        const int64_t N = rad_param_geti(a->p, a->n_p, "N", 0);
        for (int c = 0; c < EXL3_MAX_CLS; ++c) {
            const int64_t n = rad_param_geti(a->p, a->n_p, nq[c], N) / 2;
            W = n > W ? n : W;
        }
    } else {
        const int64_t N = rad_param_geti(a->p, a->n_p, "N", 0), No = rad_param_geti(a->p, a->n_p, "N_odd", N);
        W = (N > No ? N : No) / 2;
    }
    if (T < 0 || W <= 0) return RAD_E_SHAPE;
    RadTensor h;
    memset(&h, 0, sizeof h);
    h.data = calloc((size_t)(T > 0 ? T : 1) * (size_t)W, 2);
    if (!h.data) return RAD_E_NOMEM;
    h.dtype = RAD_F16;
    h.rank = 2;
    h.shape[0] = T;
    h.shape[1] = W;
    rad_tensor_pack(&h);
    RadTensor tg[24], td[16];
    if (mod4) exl3_gd4_operands(a->t, &h, tg, td);
    else exl3_gd_operands(a->t, &h, tg, td);
    RadArgs ag = *a, ad = *a;
    ag.t = tg; ag.n_t = exl3_positions(mod4 ? EXL3_GATE_UP4 : EXL3_GATE_UP)->n_opd; ag.p = pg; ag.n_p = ng;
    ad.t = td; ad.n_t = exl3_positions(mod4 ? EXL3_DOWN4 : EXL3_DOWN)->n_opd; ad.p = pd; ad.n_p = nd;
    int rc = exl3ref_launch(&ag, mod4 ? EXL3_GATE_UP4 : EXL3_GATE_UP);
    if (rc == RAD_OK) rc = exl3ref_launch(&ad, mod4 ? EXL3_DOWN4 : EXL3_DOWN);
    free(h.data);
    return rc;
}

int exl3ref_launch(const RadArgs* a, int op) {
    if (exl3_op_base(op) == EXL3_GU_DN) return launch_gu_dn(a, op);
    const Exl3Positions* at = exl3_positions(op);
    if (a->n_t < at->n_opd) return RAD_E_INVAL;
    if (!rad_arg_in(a, at->x) || !rad_arg_in(a, at->sorted) || !rad_arg_in(a, at->offset) ||
        !rad_arg_in(a, at->y))
        return RAD_E_INVAL;
    /* The expert's even tables are required; its odd ones come all together or not at all. A _mod4 op's four classes
     * are all required. */
    int odd = 0, odd_planes = 0;
    for (int p = EXL3_P_CODES; p <= EXL3_P_SVH; ++p) {
        if (at->plane[0][p] < 0) continue;
        for (int c = 0; c < (at->n_cls == 4 ? 4 : 1); ++c)
            if (!rad_arg_in(a, at->plane[c][p])) return RAD_E_INVAL;
        ++odd_planes;
        odd += rad_arg_in(a, at->plane[1][p]) != 0;
    }
    if (odd != 0 && odd != odd_planes) return RAD_E_INVAL;

    Exl3Geom g;
    int rc = exl3_geom_parse(a->p, a->n_p, op, odd != 0, &g);
    if (rc != RAD_OK) return rc;
    /* gate_up with h_rot = 1 reads down's suh of every expert it serves: each class's table. */
    if (g.op == EXL3_GATE_UP && g.h_rot)
        for (int c = 0; c < g.n_class; ++c)
            if (!rad_arg_in(a, at->plane[c][EXL3_P_DN_SUH])) return RAD_E_INVAL;
    Io io;
    if ((rc = check(a, &g, at, &io)) != RAD_OK) return rc;

    /* The GEMMs: every row zero first -- dropped slots, rows past the live count, a narrower class's tail.
     * prep_gu writes the runs' rows only and leaves the rest as they were (include/exl3_abi.h, XR). */
    if (g.op != EXL3_PREP_GU)
        for (int64_t i = 0; i < io.T; ++i)
            memset((uint16_t*)io.y->data + i * io.y_pitch, 0, (size_t)g.y_cols * 2);

    int64_t most = 0, widest = 0;
    for (int c = 0; c < g.n_class; ++c) {
        if (g.rows[c] * g.cols[c] > most) most = g.rows[c] * g.cols[c];
        if (g.rows[c] > widest) widest = g.rows[c];
    }
    /* x and xr hold an input row (x_cols) and, with h_rot, gate_up's h row (widest / 2). */
    const int64_t row = g.x_cols > widest ? g.x_cols : widest;
    Work wk = { malloc((size_t)(most + 1) * sizeof(float)), malloc((size_t)row * sizeof(float)),
                malloc((size_t)row * sizeof(float)), malloc((size_t)(widest + 1) * sizeof(float)) };
    rc = wk.what && wk.x && wk.xr && wk.acc ? RAD_OK : RAD_E_NOMEM;
    for (int64_t e = 0; e < g.n_expert && rc == RAD_OK; ++e) {
        int live = 0;
        for (int64_t i = io.eoff[e]; i < io.eoff[e + 1] && !live; ++i) live = io.tok[i] >= 0;
        if (live || (g.op == EXL3_PREP_GU && io.eoff[e + 1] > io.eoff[e]))
            rc = g.op == EXL3_PREP_GU ? prep_rows(&g, &io, e, &wk) : expert_rows(&g, &io, e, &wk);
    }
    free(wk.what); free(wk.x); free(wk.xr); free(wk.acc);
    return rc;
}
