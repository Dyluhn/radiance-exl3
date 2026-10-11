/* layout_test.c -- exl3r4d's stored codes form, checked on the host (label host; no card needed).
 *
 *  1. The fast K=4 conversion (kernels/exl3r4d/layout.c) against the oracle conversion (common/exl3_layout.c),
 *     both directions, on random planes: a TP slice's rectangle in a wider plane (row stride != dense), a stacked
 *     gate_up's two halves, the real expert shapes.
 *  2. The hooks: what they claim (tag, dtype, bytes, align), what they leave as stored (suh, suh.1, svh), what
 *     they refuse and with which code (R12: K != 4, mcg, 3inst, a stacked encoding on down, the wrong plane, the
 *     wrong extents), and relayout -> unrelayout through the hooks giving the plane back byte for byte.
 *  3. Controls: one flipped bit in a record must make the comparison fail (it can fail); relayout run twice gives
 *     the same bytes (nothing uninitialised).
 *  4. The relayout rate, printed (the loader runs it behind the disk read; rad_abi.h:197-199).
 *
 * Exit 0 all passed, 1 any failure. */
#include "exl3r4d.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int g_fail = 0, g_pass = 0;
#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (cond) { ++g_pass; }                           \
        else { ++g_fail; printf("FAIL "); printf(__VA_ARGS__); printf("\n"); } \
    } while (0)

static uint64_t rs = 0x243F6A8885A308D3ull;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static void fill(uint8_t* p, size_t n) { for (size_t i = 0; i < n; ++i) p[i] = (uint8_t)rnd(); }

/* 1. fast vs oracle on a [rows, cols] rectangle at column j0 of a plane `plane_cols` wide. */
static void fast_vs_oracle(int64_t rows, int64_t cols, int64_t plane_cols, int64_t j0) {
    const int64_t rb = exl3_codes_row_bytes(plane_cols, 4);
    const int64_t rec_n = exl3_record_elems(rows, cols, 4);   /* u16 */
    uint8_t* plane = malloc((size_t)(rows * rb));
    uint16_t* want = malloc((size_t)rec_n * 2);
    uint32_t* got = malloc((size_t)rec_n * 2);
    uint8_t* back = malloc((size_t)(rows * rb));
    fill(plane, (size_t)(rows * rb));
    const uint8_t* rect = plane + j0 * 4 / 8;
    CHECK(exl3_codes_to_record(rect, rb, rows, cols, 4, want) == EXL3_OK, "oracle to_record %lldx%lld",
          (long long)rows, (long long)cols);
    exl3r4d_k4_to_record(rect, rb, rows, cols, got);
    CHECK(memcmp(want, got, (size_t)rec_n * 2) == 0, "fast to_record == oracle, %lldx%lld at col %lld of %lld",
          (long long)rows, (long long)cols, (long long)j0, (long long)plane_cols);
    /* back: into a garbage plane, only the rectangle compared (both write exactly it) */
    fill(back, (size_t)(rows * rb));
    memcpy(back, plane, (size_t)(rows * rb));
    for (int64_t i = 0; i < rows; ++i) memset(back + i * rb + j0 / 2, 0x5A, (size_t)(cols / 2));
    exl3r4d_k4_from_record(got, rows, cols, back + j0 / 2, rb);
    CHECK(memcmp(back, plane, (size_t)(rows * rb)) == 0, "fast from_record inverts, %lldx%lld at col %lld",
          (long long)rows, (long long)cols, (long long)j0);
    /* negative control: a flipped bit in the record is seen */
    got[rec_n / 4] ^= 1u << 7;
    CHECK(memcmp(want, got, (size_t)rec_n * 2) != 0, "control: one flipped record bit is caught");
    free(plane); free(want); free(got); free(back);
}

/* 2. the hooks */
static RadParam P[16];
static int nP;
static void params(int64_t M, int64_t N, int64_t K, int64_t No, int64_t Ko, const char* dt, const char* cb) {
    nP = 0;
    P[nP++] = RAD_INT("M", M);
    P[nP++] = RAD_INT("N", N);
    P[nP++] = RAD_INT("K", K);
    P[nP++] = RAD_INT("n_expert", 8);
    P[nP++] = RAD_INT("top_k", 2);
    P[nP++] = RAD_STR("dtype", dt);
    P[nP++] = RAD_STR("codebook", cb);
    P[nP++] = RAD_INT("h_rot", 0);   /* v3: required on both ops */
    P[nP++] = RAD_INT("a_rot", 0);   /* v4: required on gate_up (down and prep_gu do not read it) */
    if (No) { P[nP++] = RAD_INT("N_odd", No); P[nP++] = RAD_INT("K_odd", Ko); }
}

static RadTensor plane_t(uint32_t dt, int64_t r, int64_t c, void* data) {
    RadTensor t;
    memset(&t, 0, sizeof t);
    t.data = data; t.dtype = dt; t.rank = 2; t.shape[0] = r; t.shape[1] = c;
    t.stride[0] = c; t.stride[1] = 1;
    return t;
}

static void hooks(void) {
    RadEncoding gu, dn, e;
    RadLayout L;
    exl3_encoding_make(4, EXL3_CB_MUL1, 1, &gu);
    exl3_encoding_make(4, EXL3_CB_MUL1, 0, &dn);
    const Exl3Positions* pg = exl3_positions(1);
    const Exl3Positions* pd = exl3_positions(0);
    int sel = EXL3_PLANE_CODES;

    /* gate_up tp2: class 0 [768, 2560], class 1 [512, 2560] */
    params(16, 768, 2560, 512, 2560, "exl3k4", "mul1");
    RadTensor c0 = plane_t(RAD_U4, 768, 2560, 0), c1 = plane_t(RAD_U4, 512, 2560, 0);
    int rc = exl3r4d_layout(1, P, nP, pg->plane[0][EXL3_P_CODES], &gu, &sel, &c0, 1, &L);
    CHECK(rc == RAD_OK && L.dtype == EXL3R4D_DT_TRELLIS && L.bytes == 768 * 2560 / 2 && L.align == 256 &&
          L.shape[0] == 768 && L.shape[1] == 2560 && strcmp(L.tag, "exl3r4d.trellis.k4") == 0,
          "gate_up class 0 codes: claimed (rc %d bytes %lld)", rc, (long long)L.bytes);
    rc = exl3r4d_layout(1, P, nP, pg->plane[1][EXL3_P_CODES], &gu, &sel, &c1, 1, &L);
    CHECK(rc == RAD_OK && L.bytes == 512 * 2560 / 2, "gate_up class 1 codes: claimed at its own extents");
    rc = exl3r4d_layout(1, P, nP, pg->plane[1][EXL3_P_CODES], &gu, &sel, &c0, 1, &L);
    CHECK(rc == RAD_E_SHAPE, "gate_up class 1 codes at class 0's extents: RAD_E_SHAPE (got %d)", rc);
    int s_suh = EXL3_PLANE_SUH, s_suh1 = EXL3_PLANE_SUH1, s_svh = EXL3_PLANE_SVH;
    RadTensor suh = plane_t(RAD_F16, 1, 2560, 0), svh = plane_t(RAD_F16, 768, 1, 0);
    /* L-ALIGN: every scale plane is claimed as the identity form at 256-B alignment, not left at 4 KiB */
#define VEC_CLAIMED(rc_, n_)                                                                            \
    ((rc_) == RAD_OK && L.tag && strcmp(L.tag, "exl3r4d.f16.vec") == 0 && L.align == 256 &&            \
     L.dtype == RAD_F16 && L.bytes == (n_) * 2)
    rc = exl3r4d_layout(1, P, nP, pg->plane[0][EXL3_P_SUH], &gu, &s_suh, &suh, 1, &L);
    CHECK(VEC_CLAIMED(rc, 2560), "suh: the f16 vector form, 256-B aligned (rc %d)", rc);
    rc = exl3r4d_layout(1, P, nP, pg->plane[0][EXL3_P_SUH1], &gu, &s_suh1, &suh, 1, &L);
    CHECK(VEC_CLAIMED(rc, 2560), "suh.1: the f16 vector form, 256-B aligned (rc %d)", rc);
    rc = exl3r4d_layout(1, P, nP, pg->plane[0][EXL3_P_SVH], &gu, &s_svh, &svh, 1, &L);
    CHECK(VEC_CLAIMED(rc, 768) && L.shape[0] == 768 && L.shape[1] == 1, "svh: the f16 vector form, its own shape");
    {   /* the identity form is a copy both ways */
        uint16_t src[768], dst[768], back[768];
        for (int i = 0; i < 768; ++i) src[i] = (uint16_t)(i * 2654435761u >> 7);
        RadTensor sv = plane_t(RAD_F16, 768, 1, src), bv = plane_t(RAD_F16, 768, 1, back);
        rc = exl3r4d_relayout(1, P, nP, pg->plane[0][EXL3_P_SVH], &gu, &s_svh, &sv, 1, dst, sizeof dst);
        CHECK(rc == RAD_OK && memcmp(src, dst, sizeof dst) == 0, "svh relayout: the same bytes (rc %d)", rc);
        rc = exl3r4d_unrelayout(1, P, nP, pg->plane[0][EXL3_P_SVH], &gu, &s_svh, dst, sizeof dst, &bv, 1);
        CHECK(rc == RAD_OK && memcmp(src, back, sizeof back) == 0, "svh unrelayout: the same bytes (rc %d)", rc);
        rc = exl3r4d_relayout(1, P, nP, pg->plane[0][EXL3_P_SVH], &gu, &s_svh, &sv, 1, dst, sizeof dst - 2);
        CHECK(rc == RAD_E_INVAL, "svh relayout into a short buffer refused (rc %d)", rc);
    }
    CHECK(exl3r4d_layout(1, P, nP, pg->plane[0][EXL3_P_SUH], &gu, &s_svh, &svh, 1, &L) == RAD_E_SHAPE,
          "svh selected where suh belongs: RAD_E_SHAPE");
    CHECK(exl3r4d_layout(1, P, nP, pg->x, &gu, &sel, &c0, 1, &L) == RAD_E_UNSUPPORTED,
          "a non-table operand: not ours to lay out");

    /* v4: prep_gu's tables are gate_up's suh / suh.1 runs (the stacked encoding), stored as they are; checked */
    const Exl3Positions* pp = exl3_positions(EXL3_PREP_GU);
    rc = exl3r4d_layout(EXL3_PREP_GU, P, nP, pp->plane[0][EXL3_P_SUH], &gu, &s_suh, &suh, 1, &L);
    CHECK(VEC_CLAIMED(rc, 2560), "prep_gu suh: the form gate_up asks (rc %d)", rc);
    rc = exl3r4d_layout(EXL3_PREP_GU, P, nP, pp->plane[1][EXL3_P_SUH1], &gu, &s_suh1, &suh, 1, &L);
    CHECK(VEC_CLAIMED(rc, 2560), "prep_gu suh.1 (odd class): the form gate_up asks (rc %d)", rc);
    CHECK(exl3r4d_layout(EXL3_PREP_GU, P, nP, pp->plane[0][EXL3_P_SUH], &dn, &s_suh, &suh, 1, &L) == RAD_E_DTYPE,
          "prep_gu on down's (unstacked) encoding: RAD_E_DTYPE");
    RadTensor suh_short = plane_t(RAD_F16, 1, 2432, 0);
    CHECK(exl3r4d_layout(EXL3_PREP_GU, P, nP, pp->plane[0][EXL3_P_SUH], &gu, &s_suh, &suh_short, 1, &L) == RAD_E_SHAPE,
          "prep_gu suh 128 short: RAD_E_SHAPE");
    CHECK(exl3r4d_layout(EXL3_PREP_GU, P, nP, pp->x, &gu, &sel, &c0, 1, &L) == RAD_E_UNSUPPORTED,
          "prep_gu's a: not ours to lay out");

    /* gate_up's dn_suh is DOWN's suh weight (one stored copy): both hooks must ask the same form of it */
    {
        RadTensor dsuh0 = plane_t(RAD_F16, 1, 384, 0), dsuh1 = plane_t(RAD_F16, 1, 256, 0);
        rc = exl3r4d_layout(1, P, nP, pg->plane[0][EXL3_P_DN_SUH], &dn, &s_suh, &dsuh0, 1, &L);
        CHECK(VEC_CLAIMED(rc, 384), "gate_up dn_suh (class 0): the vector form (rc %d)", rc);
        rc = exl3r4d_layout(1, P, nP, pg->plane[1][EXL3_P_DN_SUH], &dn, &s_suh, &dsuh1, 1, &L);
        CHECK(VEC_CLAIMED(rc, 256), "gate_up dn_suh (class 1): the vector form (rc %d)", rc);
        rc = exl3r4d_layout(1, P, nP, pg->plane[0][EXL3_P_DN_SUH], &gu, &s_suh, &dsuh0, 1, &L);
        CHECK(rc == RAD_E_DTYPE, "gate_up dn_suh on a stacked encoding: RAD_E_DTYPE (got %d)", rc);
        rc = exl3r4d_layout(1, P, nP, pg->plane[0][EXL3_P_DN_SUH], &dn, &s_suh, &dsuh1, 1, &L);
        CHECK(rc == RAD_E_SHAPE, "gate_up dn_suh at the other class's width: RAD_E_SHAPE (got %d)", rc);
        params(16, 2560, 384, 2560, 256, "exl3k4", "mul1");   /* down tp2, the same experts */
        rc = exl3r4d_layout(0, P, nP, pd->plane[0][EXL3_P_SUH], &dn, &s_suh, &dsuh0, 1, &L);
        CHECK(VEC_CLAIMED(rc, 384), "down suh (class 0): the same form as gate_up's dn_suh view (rc %d)", rc);
        params(16, 768, 2560, 512, 2560, "exl3k4", "mul1");
    }

    /* R12: every encoding this library does not decode is RAD_E_DTYPE, by the hook */
    for (int k = 2; k <= 8; ++k) {
        if (k == 4) continue;
        char dt[16];
        snprintf(dt, sizeof dt, "exl3k%d", k);
        params(16, 768, 2560, 512, 2560, dt, "mul1");
        exl3_encoding_make(k, EXL3_CB_MUL1, 1, &e);
        RadTensor ck = plane_t(exl3_codes_dtype(k), 768, 2560, 0);
        rc = exl3r4d_layout(1, P, nP, pg->plane[0][EXL3_P_CODES], &e, &sel, &ck, 1, &L);
        CHECK(rc == RAD_E_DTYPE, "K=%d refused RAD_E_DTYPE (got %d)", k, rc);
    }
    params(16, 768, 2560, 512, 2560, "exl3k4", "mcg");
    for (int cb = EXL3_CB_3INST; cb <= EXL3_CB_MCG; ++cb) {
        exl3_encoding_make(4, cb, 1, &e);
        rc = exl3r4d_layout(1, P, nP, pg->plane[0][EXL3_P_CODES], &e, &sel, &c0, 1, &L);
        CHECK(rc == RAD_E_DTYPE, "codebook %s refused RAD_E_DTYPE (got %d)", exl3_codebook_name(cb), rc);
    }
    params(16, 2560, 384, 2560, 256, "exl3k4", "mul1");
    RadTensor d0 = plane_t(RAD_U4, 2560, 384, 0);
    rc = exl3r4d_layout(0, P, nP, pd->plane[0][EXL3_P_CODES], &gu, &sel, &d0, 1, &L);
    CHECK(rc == RAD_E_DTYPE, "a stacked (gate_up) encoding on down: RAD_E_DTYPE (got %d)", rc);
    rc = exl3r4d_layout(0, P, nP, pd->plane[0][EXL3_P_CODES], &dn, &sel, &d0, 1, &L);
    CHECK(rc == RAD_OK && L.bytes == 2560 * 384 / 2, "down class 0 codes claimed (rc %d)", rc);

    /* relayout -> unrelayout through the hooks, both ops, byte for byte; relayout twice is the same bytes */
    for (int op = 0; op < 2; ++op) {
        const int64_t R = op ? 768 : 2560, C = op ? 2560 : 384;
        if (op) params(16, 768, 2560, 512, 2560, "exl3k4", "mul1");
        else params(16, 2560, 384, 2560, 256, "exl3k4", "mul1");
        const RadEncoding* en = op ? &gu : &dn;
        const int opd = (op ? pg : pd)->plane[0][EXL3_P_CODES];
        const size_t pb = (size_t)(R * C / 2);
        uint8_t* src = malloc(pb);
        uint8_t* back = malloc(pb);
        uint8_t* st1 = malloc(pb);
        uint8_t* st2 = malloc(pb);
        fill(src, pb);
        memset(st1, 0x11, pb);
        memset(st2, 0x77, pb);
        RadTensor t = plane_t(RAD_U4, R, C, src);
        rc = exl3r4d_relayout(op, P, nP, opd, en, &sel, &t, 1, st1, (int64_t)pb);
        CHECK(rc == RAD_OK, "%s relayout (rc %d)", op ? "gate_up" : "down", rc);
        exl3r4d_relayout(op, P, nP, opd, en, &sel, &t, 1, st2, (int64_t)pb);
        CHECK(memcmp(st1, st2, pb) == 0, "%s relayout twice: same bytes", op ? "gate_up" : "down");
        /* gate's record then up's: each half is the oracle record of its rows */
        const int64_t half = op ? R / 2 : R;
        uint16_t* want = malloc((size_t)exl3_record_elems(half, C, 4) * 2);
        int ok = 1;
        for (int m = 0; m < (op ? 2 : 1); ++m) {
            exl3_codes_to_record(src + m * half * (C / 2), C / 2, half, C, 4, want);
            ok &= memcmp(want, st1 + m * half * C / 2, (size_t)(half * C / 2)) == 0;
        }
        CHECK(ok, "%s stored form = the oracle's record%s", op ? "gate_up" : "down", op ? "s, gate then up" : "");
        RadTensor b = plane_t(RAD_U4, R, C, back);
        memset(back, 0xEE, pb);
        rc = exl3r4d_unrelayout(op, P, nP, opd, en, &sel, st1, (int64_t)pb, &b, 1);
        CHECK(rc == RAD_OK && memcmp(back, src, pb) == 0, "%s unrelayout inverts (rc %d)", op ? "gate_up" : "down", rc);
        rc = exl3r4d_relayout(op, P, nP, opd, en, &sel, &t, 1, st1, (int64_t)pb - 1);
        CHECK(rc == RAD_E_INVAL, "%s relayout into a short buffer refused (rc %d)", op ? "gate_up" : "down", rc);
        free(src); free(back); free(st1); free(st2); free(want);
    }
}

/* 4. relayout rate on the real gate_up shape */
static void rate(void) {
    const int64_t R = 1280, C = 2560, n = 64;
    const size_t pb = (size_t)(R * C / 2);
    uint8_t* src = malloc(pb);
    uint8_t* dst = malloc(pb);
    fill(src, pb);
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (int64_t i = 0; i < n; ++i) {
        exl3r4d_k4_to_record(src, C / 2, R / 2, C, (uint32_t*)dst);
        exl3r4d_k4_to_record(src + pb / 2, C / 2, R / 2, C, (uint32_t*)(dst + pb / 2));
    }
    clock_gettime(CLOCK_MONOTONIC, &b);
    const double s = (double)(b.tv_sec - a.tv_sec) + 1e-9 * (double)(b.tv_nsec - a.tv_nsec);
    printf("relayout rate: %.0f MB/s of codes, one thread (gate_up 1280x2560 K=4, %lld experts)\n",
           (double)pb * (double)n / s / 1e6, (long long)n);
    free(src); free(dst);
}

int main(void) {
    fast_vs_oracle(16, 16, 16, 0);
    fast_vs_oracle(640, 2560, 2560, 0);          /* a real gate (or up) half */
    fast_vs_oracle(2560, 384, 640, 0);           /* tp2 down, rank 0's columns [0, 384) of 640 */
    fast_vs_oracle(2560, 256, 640, 384);         /* rank 1's [384, 640) */
    fast_vs_oracle(128, 128, 512, 256);
    hooks();
    rate();
    printf("layout_test: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
