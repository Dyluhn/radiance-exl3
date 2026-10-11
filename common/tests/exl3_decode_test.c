/* exl3_decode_test.c -- the oracle against things it did not write: the vllm-exl3 golden, hand-computed
 * codebook values, the tile map's stated layout, and the algebra of the rotation. Every comparison has
 * a negative control showing it can fail. */
#include "../exl3_decode.h"
#include "../exl3_layout.h"
#include "exl3_check.h"
#include "exl3_golden.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int codebook_of(const char* s) { return exl3_codebook_from_name(s); }

static int64_t golden_mismatches(const Exl3GoldenCase* g, const uint16_t* trellis, int codebook) {
    const int64_t n = (int64_t)g->kt * 16 * g->nt * 16;
    uint16_t* w = (uint16_t*)malloc((size_t)n * sizeof *w);
    int64_t bad = 0, i;
    if (exl3_trellis_decode(trellis, g->kt, g->nt, g->bits, codebook, w) != EXL3_OK) bad = -1;
    for (i = 0; bad >= 0 && i < n; ++i) bad += w[i] != g->w_hat[i];
    free(w);
    return bad;
}

/* vllm-exl3's host decode, K = 3, 4 x mcg, mul1: every fp16 bit equal. */
TEST(decode_is_bit_exact_against_the_vllm_exl3_golden) {
    int c;
    CHECK(EXL3_GOLDEN_N == 4);
    for (c = 0; c < EXL3_GOLDEN_N; ++c)
        CHECK_EQ_U(golden_mismatches(&exl3_golden_cases[c], exl3_golden_cases[c].trellis,
                                     codebook_of(exl3_golden_cases[c].codebook)), 0);
}

/* Negative controls: one flipped trellis bit, or the other codebook, must be seen. */
TEST(the_golden_comparison_can_fail) {
    const Exl3GoldenCase* g = &exl3_golden_cases[3];
    const size_t n = (size_t)g->kt * g->nt * 16 * g->bits;
    uint16_t* t = (uint16_t*)malloc(n * sizeof *t);
    memcpy(t, g->trellis, n * sizeof *t);
    t[5] ^= 0x0100;
    CHECK(golden_mismatches(g, t, codebook_of(g->codebook)) > 0);
    CHECK(golden_mismatches(g, g->trellis, EXL3_CB_MCG) > 0);   /* case 3 is mul1 */
    CHECK(golden_mismatches(g, g->trellis, EXL3_CB_3INST) > 0);
    free(t);
}

/* Values computed by hand from the formulas (exact rational arithmetic, notes/p1.md), not by either
 * implementation: 3inst h = x*89226354 + 64248484, mcg h = x*0xCBAC1FED, both (h & 0x8FFF8FFF) ^
 * 0x3B603B60 then fp16(lo) + fp16(hi); mul1 fp16(1024 + bytesum(x*0x83DCD12D)) * K_INV + K_BIAS. */
TEST(codebooks_give_the_hand_computed_values) {
    static const struct { uint32_t state; uint16_t inst3, mcg, mul1; } v[] = {
        { 0x0000, 0x3A25, 0x3F60, 0xC2E8 },   /* 3inst 0.768066, mcg 1.84375, mul1 -3.453125 */
        { 0x0001, 0xBB5B, 0x304E, 0x3921 },
        { 0x1234, 0xC066, 0x3E0F, 0xB89E },
        { 0xFFFF, 0xB110, 0x3ACD, 0xB936 },
    };
    size_t i;
    for (i = 0; i < sizeof v / sizeof v[0]; ++i) {
        CHECK_EQ_U(exl3_codebook_value(v[i].state, EXL3_CB_3INST), v[i].inst3);
        CHECK_EQ_U(exl3_codebook_value(v[i].state, EXL3_CB_MCG), v[i].mcg);
        CHECK_EQ_U(exl3_codebook_value(v[i].state, EXL3_CB_MUL1), v[i].mul1);
    }
    CHECK_EQ_U(exl3_codebook_value(0x10000, EXL3_CB_MUL1), 0xC2E8);   /* only the low 16 bits */
    CHECK_EQ_U(exl3_codebook_value(0, 7), 0x7E00);                   /* not a codebook: nan */
}

/* 3inst through a whole tile: at K = 8 the state of p is (sym[p-1] << 8) | sym[p], circularly, so a
 * hand-built tile with sym[p] = p reads states (p-1, p) -- and position 0 reads (255, 0). */
TEST(a_hand_built_3inst_tile_decodes_by_the_formula) {
    uint8_t sym[256];
    uint16_t tile[16 * 8], st[256], w[256];
    int p, r, c;
    for (p = 0; p < 256; ++p) sym[p] = (uint8_t)p;
    exl3_tile_pack(sym, 8, tile);
    exl3_tile_states(sym, 8, st);
    CHECK_EQ_U(st[0], 0xFF00);
    CHECK_EQ_U(st[1], 0x0001);
    CHECK_EQ_U(st[0x35], 0x3435);
    CHECK(exl3_trellis_decode(tile, 1, 1, 8, EXL3_CB_3INST, w) == EXL3_OK);
    /* state 0x0001 is position 1 = tile (1, 0); its value is the table's 0xBB5B */
    exl3_pos_to_tile(1, &r, &c);
    CHECK_EQ_U(w[r * 16 + c], 0xBB5B);
    for (p = 0; p < 256; ++p) {
        exl3_pos_to_tile(p, &r, &c);
        CHECK_EQ_U(w[r * 16 + c], exl3_codebook_value(st[p], EXL3_CB_3INST));
    }
}

/* trellis.py: p = 8L + j at row 2(L%4) + (0,1,8,9)[j%4], col L/4 + 8(j/4) -- spot values and a bijection. */
TEST(the_tile_map_is_the_stated_one_and_a_bijection) {
    int seen[256] = { 0 }, p, r, c, ok = 1;
    exl3_pos_to_tile(0, &r, &c);  CHECK(r == 0 && c == 0);
    exl3_pos_to_tile(2, &r, &c);  CHECK(r == 8 && c == 0);
    exl3_pos_to_tile(3, &r, &c);  CHECK(r == 9 && c == 0);
    exl3_pos_to_tile(4, &r, &c);  CHECK(r == 0 && c == 8);
    exl3_pos_to_tile(8, &r, &c);  CHECK(r == 2 && c == 0);    /* L = 1 */
    exl3_pos_to_tile(32, &r, &c); CHECK(r == 0 && c == 1);    /* L = 4 */
    exl3_pos_to_tile(255, &r, &c); CHECK(r == 15 && c == 15);
    for (p = 0; p < 256; ++p) {
        exl3_pos_to_tile(p, &r, &c);
        ok &= r >= 0 && r < 16 && c >= 0 && c < 16 && exl3_tile_to_pos(r, c) == p;
        seen[r * 16 + c]++;
    }
    for (p = 0; p < 256; ++p) ok &= seen[p] == 1;
    CHECK(ok);
}

/* Stream order: symbol p is stream bits [pK, (p+1)K), MSB-first, word w = int16[2w] | int16[2w+1] << 16. */
TEST(the_tile_stream_is_msb_first_little_endian_words) {
    uint8_t sym[256] = { 0 }, back[256];
    uint16_t tile[16 * 8];
    int bits, p, ok = 1;
    sym[0] = 0x9;                            /* K = 4: bits 0..3 of the stream = 1001 */
    exl3_tile_pack(sym, 4, tile);
    CHECK_EQ_U(tile[1], 0x9000);             /* stream bit 0 = bit 31 of word 0 = bit 15 of int16[1] */
    CHECK_EQ_U(tile[0], 0x0000);
    for (bits = 1; bits <= 8; ++bits) {
        for (p = 0; p < 256; ++p) sym[p] = (uint8_t)(exl3_rng() & ((1u << bits) - 1));
        exl3_tile_pack(sym, bits, tile);
        exl3_tile_symbols(tile, bits, back);
        ok &= memcmp(sym, back, 256) == 0;
    }
    CHECK(ok);
}

/* The state is the 16-bit window ending at the symbol: K = 4 -> the last four symbols, K = 3 -> six
 * symbols' 18 bits, top two dropped; circular at p = 0. */
TEST(states_are_the_window_ending_at_each_symbol) {
    uint8_t sym[256];
    uint16_t st[256];
    int p;
    for (p = 0; p < 256; ++p) sym[p] = (uint8_t)((p * 7 + 3) & 0xF);
    exl3_tile_states(sym, 4, st);
    CHECK_EQ_U(st[10], (uint32_t)(sym[7] << 12 | sym[8] << 8 | sym[9] << 4 | sym[10]));
    CHECK_EQ_U(st[1], (uint32_t)(sym[254] << 12 | sym[255] << 8 | sym[0] << 4 | sym[1]));
    for (p = 0; p < 256; ++p) sym[p] = (uint8_t)((p * 5 + 1) & 0x7);
    exl3_tile_states(sym, 3, st);
    CHECK_EQ_U(st[20], (uint32_t)(((sym[15] << 15 | sym[16] << 12 | sym[17] << 9 | sym[18] << 6 |
                                    sym[19] << 3 | sym[20])) & 0xFFFF));
}

TEST(fp16_round_trips_every_finite_half_and_rounds_ties_to_even) {
    uint32_t h;
    int ok = 1;
    for (h = 0; h < 0x10000; ++h) {
        if ((h & 0x7C00) == 0x7C00 && (h & 0x3FF)) continue;   /* nan */
        ok &= exl3_half_from_double((double)exl3_half_to_float((uint16_t)h)) == h;
    }
    CHECK(ok);
    CHECK_EQ_U(exl3_half_from_double(1.0 + 0x1p-11), 0x3C00);       /* tie -> even (down) */
    CHECK_EQ_U(exl3_half_from_double(1.0 + 3 * 0x1p-11), 0x3C02);   /* tie -> even (up) */
    CHECK_EQ_U(exl3_half_from_double(65504.0), 0x7BFF);
    CHECK_EQ_U(exl3_half_from_double(65519.0), 0x7BFF);
    CHECK_EQ_U(exl3_half_from_double(65520.0), 0x7C00);
    CHECK_EQ_U(exl3_half_from_double(-0x1p-25), 0x8000);            /* tie to even zero */
    CHECK_EQ_U(exl3_half_from_double(0x1p-24 * 1.5), 0x0002);       /* subnormal tie -> even */
}

/* Normalised natural-order Sylvester: H e0 = all 1/sqrt(128); H e1 alternates sign; H H = I. */
TEST(had128_is_the_normalised_sylvester_hadamard) {
    float v[256], w[256];
    int i, ok = 1;
    double err = 0;
    memset(v, 0, sizeof v);
    v[0] = 1.0f;
    v[128 + 1] = 1.0f;
    CHECK(exl3_had128(v, 256) == EXL3_OK);
    for (i = 0; i < 128; ++i) {
        ok &= fabs(v[i] - 0.0883883476) < 1e-7;
        ok &= fabs(v[128 + i] - ((i & 1) ? -0.0883883476 : 0.0883883476)) < 1e-7;
    }
    CHECK(ok);
    for (i = 0; i < 256; ++i) v[i] = w[i] = (float)((int)(exl3_rng() % 2001) - 1000) / 1000.0f;
    exl3_had128(v, 256);
    exl3_had128(v, 256);
    for (i = 0; i < 256; ++i) err = fmax(err, fabs(v[i] - w[i]));
    CHECK(err < 1e-5);
    CHECK(exl3_had128(v, 100) == EXL3_E_SHAPE);
}

/* The codes plane's bit stream is the core's: element j of a K-bit row in bits [jK, (j+1)K), LSB first. */
TEST(codes_plane_elements_are_lsb_first_along_the_row) {
    uint8_t row[8] = { 0 };
    exl3_codes_set(row, 8, 3, 0, 0, 0x5);   /* bits 0..2 = 1,0,1 */
    exl3_codes_set(row, 8, 3, 0, 2, 0x7);   /* bits 6..8: straddles bytes 0 and 1 */
    CHECK_EQ_U(row[0], 0xC5);
    CHECK_EQ_U(row[1], 0x01);
    CHECK_EQ_U(exl3_codes_get(row, 8, 3, 0, 2), 7);
    CHECK_EQ_U(exl3_codes_get(row, 8, 3, 0, 1), 0);
    CHECK_EQ_U(exl3_codes_row_bytes(2560, 3), 960);
    CHECK_EQ_U(exl3_codes_row_bytes(640, 4), 320);
}

/* W_hat from the canonical plane == W_hat from the trellis it was made from (transposed). */
TEST(w_hat_from_the_plane_equals_w_hat_from_the_trellis) {
    enum { N = 64, K = 48 };
    int bits, cb;
    for (bits = 1; bits <= 8; ++bits)
        for (cb = 0; cb < 3; ++cb) {
            const int64_t rb = exl3_codes_row_bytes(K, bits);
            uint16_t* tr = (uint16_t*)malloc((size_t)exl3_record_elems(N, K, bits) * 2);
            uint8_t* codes = (uint8_t*)calloc((size_t)(N * rb), 1);
            uint16_t wh[K * N];
            float out[N * K];
            Exl3View v;
            int64_t i, j, bad = 0;
            for (i = 0; i < exl3_record_elems(N, K, bits); ++i) tr[i] = (uint16_t)exl3_rng();
            exl3_trellis_decode(tr, K / 16, N / 16, bits, cb, wh);   /* [k][n] */
            exl3_record_to_codes(tr, N, K, bits, codes, rb);
            memset(&v, 0, sizeof v);
            v.rows = N; v.cols = K; v.bits = bits; v.codebook = cb; v.codes = codes;
            CHECK(exl3_decode_what_rows(&v, 0, N, out) == EXL3_OK);
            for (i = 0; i < N; ++i)
                for (j = 0; j < K; ++j)
                    bad += exl3_half_from_double(out[i * K + j]) != wh[j * N + i];
            CHECK_EQ_U(bad, 0);
            /* a middle range of rows gives the same rows */
            CHECK(exl3_decode_what_rows(&v, 20, 9, out) == EXL3_OK);
            for (i = 0, bad = 0; i < 9; ++i)
                for (j = 0; j < K; ++j)
                    bad += exl3_half_from_double(out[i * K + j]) != wh[j * N + 20 + i];
            CHECK_EQ_U(bad, 0);
            free(tr);
            free(codes);
        }
}

TEST(decode_refuses_what_it_cannot_decode) {
    uint8_t codes[256 * 128];
    uint16_t s[256];
    float out[256 * 128];
    Exl3View v;
    memset(&v, 0, sizeof v);
    memset(s, 0, sizeof s);
    v.rows = 128; v.cols = 128; v.bits = 4; v.codebook = EXL3_CB_MUL1;
    v.codes = codes; v.suh = s; v.svh = s;
    memset(codes, 0, sizeof codes);
    CHECK(exl3_decode_rows(&v, 0, 128, out) == EXL3_OK);
    v.bits = 9;            CHECK(exl3_decode_rows(&v, 0, 128, out) == EXL3_E_ARG);
    v.bits = 0;            CHECK(exl3_decode_rows(&v, 0, 128, out) == EXL3_E_ARG);
    v.bits = 4; v.codebook = 3; CHECK(exl3_decode_rows(&v, 0, 128, out) == EXL3_E_ARG);
    v.codebook = EXL3_CB_MUL1;
    v.cols = 112;          CHECK(exl3_decode_rows(&v, 0, 128, out) == EXL3_E_SHAPE);  /* not % 128 */
    v.cols = 128;
    CHECK(exl3_decode_rows(&v, 100, 29, out) == EXL3_E_SHAPE);                      /* past the end */
    v.suh1 = s;            CHECK(exl3_decode_rows(&v, 0, 128, out) == EXL3_E_SHAPE);  /* 128 rows can't halve */
    v.suh1 = 0; v.svh = 0; CHECK(exl3_decode_rows(&v, 0, 128, out) == EXL3_E_ARG);
}

int main(void) {
    exl3_rng_state = 20261011;
    RUN(decode_is_bit_exact_against_the_vllm_exl3_golden);
    RUN(the_golden_comparison_can_fail);
    RUN(codebooks_give_the_hand_computed_values);
    RUN(a_hand_built_3inst_tile_decodes_by_the_formula);
    RUN(the_tile_map_is_the_stated_one_and_a_bijection);
    RUN(the_tile_stream_is_msb_first_little_endian_words);
    RUN(states_are_the_window_ending_at_each_symbol);
    RUN(fp16_round_trips_every_finite_half_and_rounds_ties_to_even);
    RUN(had128_is_the_normalised_sylvester_hadamard);
    RUN(codes_plane_elements_are_lsb_first_along_the_row);
    RUN(w_hat_from_the_plane_equals_w_hat_from_the_trellis);
    RUN(decode_refuses_what_it_cannot_decode);
    return exl3_check_done();
}
