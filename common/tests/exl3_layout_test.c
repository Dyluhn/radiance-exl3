/* exl3_layout_test.c -- canonical plane <-> tile-stream record, property-tested with random symbols at
 * K = 1..8 on the real expert shapes (gate_up [1280, 2560] stacked, down [2560, 640]) and on the
 * 384/256 parity slices tensor parallelism cuts them into (PLAN.md D7): rows of each gate_up half,
 * columns of down. A slice must give back exactly the whole weight's tiles, and decode to exactly the
 * whole weight's rows / columns. */
#include "../exl3_decode.h"
#include "../exl3_layout.h"
#include "exl3_check.h"

#include <stdlib.h>
#include <string.h>

enum { HIDDEN = 2560, WIDTH = 640, CUT = 384 };   /* Flash-Next: 640 = 384 + 256 at TP2 */

static uint8_t* random_plane(int64_t rows, int64_t cols, int bits, int64_t* rb) {
    int64_t i;
    uint8_t* p;
    *rb = exl3_codes_row_bytes(cols, bits);
    p = (uint8_t*)malloc((size_t)(rows * *rb));
    for (i = 0; i < rows * *rb; ++i) p[i] = (uint8_t)exl3_rng();
    /* bits past the last element of a row are not part of the plane: keep them zero, as a writer does */
    if ((cols * bits) % 8)
        for (i = 0; i < rows; ++i) p[i * *rb + *rb - 1] &= (uint8_t)((1u << ((cols * bits) % 8)) - 1);
    return p;
}

/* Tiles [tk0, tk0+kt) x [tn0, tn0+nt) of a whole record (whole_nt tiles a k-row) vs a slice's record. */
static int64_t tiles_differ(const uint16_t* whole, int64_t whole_nt, const uint16_t* slice,
                            int64_t tk0, int64_t kt, int64_t tn0, int64_t nt, int bits) {
    int64_t tk, tn, bad = 0;
    for (tk = 0; tk < kt; ++tk)
        for (tn = 0; tn < nt; ++tn)
            bad += memcmp(whole + ((tk0 + tk) * whole_nt + tn0 + tn) * 16 * bits,
                          slice + (tk * nt + tn) * 16 * bits, (size_t)(32 * bits)) != 0;
    return bad;
}

static void round_trip_and_slices(int64_t rows, int64_t cols, int bits, int rows_cut, int cols_cut) {
    int64_t rb, i, bad;
    uint8_t* plane = random_plane(rows, cols, bits, &rb);
    uint8_t* back = (uint8_t*)calloc((size_t)(rows * rb), 1);
    const int64_t elems = exl3_record_elems(rows, cols, bits);
    uint16_t* rec = (uint16_t*)malloc((size_t)elems * 2);
    uint16_t* rec2 = (uint16_t*)malloc((size_t)elems * 2);
    uint16_t* part = (uint16_t*)malloc((size_t)elems * 2);

    /* both directions, whole weight */
    CHECK(exl3_codes_to_record(plane, rb, rows, cols, bits, rec) == EXL3_OK);
    CHECK(exl3_record_to_codes(rec, rows, cols, bits, back, rb) == EXL3_OK);
    CHECK(memcmp(plane, back, (size_t)(rows * rb)) == 0);
    for (i = 0; i < elems; ++i) rec2[i] = (uint16_t)exl3_rng();
    exl3_record_to_codes(rec2, rows, cols, bits, back, rb);
    exl3_codes_to_record(back, rb, rows, cols, bits, rec);
    CHECK(memcmp(rec, rec2, (size_t)elems * 2) == 0);
    exl3_codes_to_record(plane, rb, rows, cols, bits, rec);

    /* row slices: each stacked half cut at CUT -- [0,384) [384,640) of gate, then of up */
    if (rows_cut) {
        const int64_t half = rows / 2, lo[4] = { 0, CUT, half, half + CUT },
                      hi[4] = { CUT, half, half + CUT, rows };
        int s;
        for (s = 0; s < 4; ++s) {
            const int64_t n = hi[s] - lo[s];
            CHECK(exl3_codes_to_record(plane + lo[s] * rb, rb, n, cols, bits, part) == EXL3_OK);
            bad = tiles_differ(rec, rows / 16, part, 0, cols / 16, lo[s] / 16, n / 16, bits);
            CHECK_EQ_U(bad, 0);
        }
    }
    /* column slices at CUT: the plane pointer moves by CUT*K/8 bytes, the stride stays the plane's */
    if (cols_cut) {
        const int64_t lo[2] = { 0, CUT }, hi[2] = { CUT, cols };
        int s;
        for (s = 0; s < 2; ++s) {
            const int64_t n = hi[s] - lo[s];
            CHECK((lo[s] * bits) % 8 == 0);
            CHECK(exl3_codes_to_record(plane + lo[s] * bits / 8, rb, rows, n, bits, part) == EXL3_OK);
            bad = tiles_differ(rec, rows / 16, part, lo[s] / 16, n / 16, 0, rows / 16, bits);
            CHECK_EQ_U(bad, 0);
            /* and back into a scratch copy: the slice's bits land, nothing outside it moves */
            memcpy(back, plane, (size_t)(rows * rb));
            for (i = 0; i < exl3_record_elems(rows, n, bits); ++i) part[i] = (uint16_t)~part[i];
            exl3_record_to_codes(part, rows, n, bits, back + lo[s] * bits / 8, rb);
            for (i = 0, bad = 0; i < rows * cols; ++i) {
                const int64_t r = i / cols, c = i % cols;
                const uint32_t a = exl3_codes_get(plane, rb, bits, r, c),
                               b = exl3_codes_get(back, rb, bits, r, c);
                const int inside = c >= lo[s] && c < hi[s];
                bad += inside ? a != (~b & ((1u << bits) - 1)) : a != b;
            }
            CHECK_EQ_U(bad, 0);
        }
    }
    free(plane); free(back); free(rec); free(rec2); free(part);
}

TEST(layout_round_trips_and_slices_keep_tiles_whole_at_every_k) {
    int bits;
    for (bits = 1; bits <= 8; ++bits) {
        round_trip_and_slices(2 * WIDTH, HIDDEN, bits, 1, 0);   /* gate_up, stacked */
        round_trip_and_slices(HIDDEN, WIDTH, bits, 0, 1);       /* down */
    }
}

/* Negative control: one flipped plane bit changes exactly one tile of the record. */
TEST(one_flipped_bit_is_one_changed_tile) {
    const int bits = 4;
    int64_t rb;
    uint8_t* plane = random_plane(64, 64, bits, &rb);
    uint16_t a[4 * 4 * 64], b[4 * 4 * 64];
    int64_t t, changed = 0;
    exl3_codes_to_record(plane, rb, 64, 64, bits, a);
    plane[17 * rb + 9] ^= 0x10;   /* row 17 (n), element 18 (k): tile (tk 1, tn 1) */
    exl3_codes_to_record(plane, rb, 64, 64, bits, b);
    for (t = 0; t < 16; ++t) changed += memcmp(a + t * 64, b + t * 64, 128) != 0;
    CHECK_EQ_U(changed, 1);
    CHECK(memcmp(a + (1 * 4 + 1) * 64, b + (1 * 4 + 1) * 64, 128) != 0);
    free(plane);
}

TEST(layout_refuses_a_rectangle_that_splits_a_tile) {
    uint8_t plane[64 * 64] = { 0 };
    uint16_t rec[4 * 4 * 64];
    CHECK(exl3_codes_to_record(plane, 32, 64, 56, 4, rec) == EXL3_E_SHAPE);
    CHECK(exl3_codes_to_record(plane, 32, 40, 64, 4, rec) == EXL3_E_SHAPE);
    CHECK(exl3_codes_to_record(plane, 31, 64, 64, 4, rec) == EXL3_E_SHAPE);   /* stride < a row */
    CHECK(exl3_record_to_codes(rec, 64, 64, 9, plane, 64) == EXL3_E_ARG);
}

/* A TP rank's slice DECODES to the slice of the weight: gate_up rows [gate lo..hi ; up lo..hi] with
 * suh/suh.1 whole; down columns [lo, hi) with suh cut and svh whole. Bit-for-bit in f32, because the
 * rotations are block-local and every op in a 128-block is the same op in the same order. */
static void fill_scales(uint16_t* s, int64_t n) {
    int64_t i;
    for (i = 0; i < n; ++i)   /* +-(0.5..1.5) in f16, a real-checkpoint-like magnitude */
        s[i] = exl3_half_from_double(((exl3_rng() & 1) ? -1.0 : 1.0) * (0.5 + (exl3_rng() % 1024) / 1024.0));
}

TEST(tp_slices_decode_to_the_slices_of_the_weight) {
    enum { GU = 2 * WIDTH };
    const int bits = 4;
    int64_t rb, i, r, bad = 0;
    uint8_t* gu = random_plane(GU, HIDDEN, bits, &rb);
    uint16_t* suh = (uint16_t*)malloc(HIDDEN * 2), *suh1 = (uint16_t*)malloc(HIDDEN * 2);
    uint16_t* svh = (uint16_t*)malloc(GU * 2);
    float* whole = (float*)malloc((size_t)GU * HIDDEN * 4);
    float* part = (float*)malloc((size_t)GU * HIDDEN * 4);
    uint8_t* sl = (uint8_t*)malloc((size_t)(2 * CUT * rb));
    uint16_t svs[2 * CUT];
    Exl3View v;
    fill_scales(suh, HIDDEN); fill_scales(suh1, HIDDEN); fill_scales(svh, GU);
    memset(&v, 0, sizeof v);
    v.rows = GU; v.cols = HIDDEN; v.bits = bits; v.codebook = EXL3_CB_MUL1;
    v.codes = gu; v.suh = suh; v.suh1 = suh1; v.svh = svh;
    CHECK(exl3_decode_rows(&v, 0, GU, whole) == EXL3_OK);
    /* rank 0 of a stacked gate_up at 384/256: gate rows [0, 384) then up rows [640, 1024) */
    memcpy(sl, gu, (size_t)(CUT * rb));
    memcpy(sl + CUT * rb, gu + WIDTH * rb, (size_t)(CUT * rb));
    memcpy(svs, svh, CUT * 2);
    memcpy(svs + CUT, svh + WIDTH, CUT * 2);
    v.rows = 2 * CUT; v.codes = sl; v.svh = svs;
    CHECK(exl3_decode_rows(&v, 0, 2 * CUT, part) == EXL3_OK);
    for (r = 0; r < 2 * CUT; ++r) {
        const int64_t src = r < CUT ? r : WIDTH + (r - CUT);
        bad += memcmp(part + r * HIDDEN, whole + src * HIDDEN, HIDDEN * 4) != 0;
    }
    CHECK_EQ_U(bad, 0);
    /* negative control: the same slice read without suh.1 is NOT the weight's up rows */
    v.suh1 = 0;
    v.rows = 2 * CUT;
    {
        Exl3View w = v;
        w.suh1 = suh;   /* gate's scales on the up half: wrong on purpose */
        CHECK(exl3_decode_rows(&w, 0, 2 * CUT, part) == EXL3_OK);
        for (r = CUT, bad = 0; r < 2 * CUT; ++r)
            bad += memcmp(part + r * HIDDEN, whole + (WIDTH + r - CUT) * HIDDEN, HIDDEN * 4) != 0;
        CHECK(bad > 0);
    }
    free(gu); free(sl); free(svh);

    /* down [2560, 640] cut by columns at 384: the codes pointer moves, suh is cut, svh is whole */
    {
        uint8_t* dn = random_plane(HIDDEN, WIDTH, bits, &rb);
        uint16_t* dsvh = (uint16_t*)malloc(HIDDEN * 2);
        const int64_t lo[2] = { 0, CUT }, n[2] = { CUT, WIDTH - CUT };
        int s;
        fill_scales(dsvh, HIDDEN);
        memset(&v, 0, sizeof v);
        v.rows = HIDDEN; v.cols = WIDTH; v.bits = bits; v.codebook = EXL3_CB_MUL1;
        v.codes = dn; v.suh = suh; v.svh = dsvh;
        CHECK(exl3_decode_rows(&v, 0, HIDDEN, whole) == EXL3_OK);
        for (s = 0; s < 2; ++s) {
            Exl3View w = v;
            w.cols = n[s];
            w.codes = dn + lo[s] * bits / 8;
            w.codes_row_bytes = rb;
            w.suh = suh + lo[s];
            CHECK(exl3_decode_rows(&w, 0, HIDDEN, part) == EXL3_OK);
            for (r = 0, bad = 0; r < HIDDEN; ++r)
                for (i = 0; i < n[s]; ++i)
                    bad += part[r * n[s] + i] != whole[r * WIDTH + lo[s] + i];
            CHECK_EQ_U(bad, 0);
        }
        free(dn); free(dsvh);
    }
    free(suh); free(suh1); free(whole); free(part);
}

int main(void) {
    exl3_rng_state = 1384256;
    RUN(layout_round_trips_and_slices_keep_tiles_whole_at_every_k);
    RUN(one_flipped_bit_is_one_changed_tile);
    RUN(layout_refuses_a_rectangle_that_splits_a_tile);
    RUN(tp_slices_decode_to_the_slices_of_the_weight);
    return exl3_check_done();
}
