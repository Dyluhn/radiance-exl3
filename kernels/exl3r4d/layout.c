/* layout.c -- what exl3r4d stores for each plane table, and the conversions both ways.
 *
 * THE CODES PLANE is stored as exllamav3's tile-stream record (common/exl3_layout.h): the form the trellis decode
 * streams, tile (tk, tn) of 16 x 16 weights as 8*K little-endian u32 words, tiles k-major. A stacked gate_up is
 * gate's record then up's ([2][K/16][W/16][8*K] u32), which is how the kernel indexes matrix `mat`. Lossless and
 * derived from nothing but the bits (rad_abi.h: a relayout moves bits). ONE stored form for every band (R4): the
 * prefill band reads the same record (P5).
 *
 * suh, suh.1 and svh (and gate_up's view of down's suh, dn_suh) are claimed as an IDENTITY form, "exl3r4d.f16.vec":
 * the same f16 bytes, relayout a copy, but at 256-B alignment (RAD_ALIGN_SUB, the .rad sub-plane alignment) instead
 * of the 4 KiB (RAD_ALIGN_UNIT) every weight the kernel reads "as it is" is given. The engine packs an expert's
 * weights into one movement unit, each at its own alignment (core/place/planner.cpp, unit bytes; rad_builder.cpp:
 * w.align = RAD_ALIGN_UNIT unless the hook returns L.align), so five small vectors at 4 KiB cost ~12 KiB a slot:
 * ~1% of an expert at tp2, which was the whole of EXL3's size advantage over the baseline in slab slots (lever
 * L-ALIGN, notes/p4.md). Every row that reads these weights -- gate_up and down in both bands, prep_gu -- goes
 * through these hooks, so they all ask the same stored form (a weight is stored once).
 *
 * WHY A FAST PATH. The loader runs relayout behind the disk read and it has to keep pace (rad_abi.h:197-199).
 * common/exl3_layout.c walks bits one at a time -- the oracle, written to be read -- and at K = 4 a whole tile is
 * a byte shuffle: tile position p = 8L + j holds W_hat[k = 2(L%4) + {0,1,8,9}[j%4]][n = L/4 + 8(j/4)], so word L
 * = 4 cc + l (cc = n % 8) takes its eight 4-bit symbols from bytes l and l + 4 of canonical rows cc and cc + 8
 * (two k-nibbles a byte, low first), each byte's nibbles swapped into the MSB-first stream:
 *     word = swap(R[cc][l]) << 24 | swap(R[cc][l+4]) << 16 | swap(R[cc+8][l]) << 8 | swap(R[cc+8][l+4]).
 * The property test (tests/gpu/layout_test.c) checks it against common/exl3_layout.c both ways. */
#include "exl3r4d.h"

#include <string.h>

/* ------------------------------------------------------------------ K = 4, fast */

static uint8_t swapnib(uint8_t b) { return (uint8_t)((b << 4) | (b >> 4)); }

void exl3r4d_k4_to_record(const uint8_t* codes, int64_t row_bytes, int64_t rows, int64_t cols,
                          uint32_t* record) {
    const int64_t nt = rows / 16, kt = cols / 16;
    for (int64_t tk = 0; tk < kt; ++tk)
        for (int64_t tn = 0; tn < nt; ++tn) {
            const uint8_t* R = codes + tn * 16 * row_bytes + tk * 8;  /* row c at R + c*row_bytes */
            uint32_t* w = record + (tk * nt + tn) * 32;
            for (int cc = 0; cc < 8; ++cc) {
                const uint8_t* a = R + cc * row_bytes;
                const uint8_t* b = R + (cc + 8) * row_bytes;
                for (int l = 0; l < 4; ++l)
                    w[4 * cc + l] = (uint32_t)swapnib(a[l]) << 24 | (uint32_t)swapnib(a[l + 4]) << 16 |
                                    (uint32_t)swapnib(b[l]) << 8 | (uint32_t)swapnib(b[l + 4]);
            }
        }
}

void exl3r4d_k4_from_record(const uint32_t* record, int64_t rows, int64_t cols, uint8_t* codes,
                            int64_t row_bytes) {
    const int64_t nt = rows / 16, kt = cols / 16;
    for (int64_t tk = 0; tk < kt; ++tk)
        for (int64_t tn = 0; tn < nt; ++tn) {
            uint8_t* R = codes + tn * 16 * row_bytes + tk * 8;
            const uint32_t* w = record + (tk * nt + tn) * 32;
            for (int cc = 0; cc < 8; ++cc) {
                uint8_t* a = R + cc * row_bytes;
                uint8_t* b = R + (cc + 8) * row_bytes;
                for (int l = 0; l < 4; ++l) {
                    const uint32_t x = w[4 * cc + l];
                    a[l] = swapnib((uint8_t)(x >> 24));
                    a[l + 4] = swapnib((uint8_t)(x >> 16));
                    b[l] = swapnib((uint8_t)(x >> 8));
                    b[l + 4] = swapnib((uint8_t)x);
                }
            }
        }
}

/* ------------------------------------------------------------------ any K (R23), fast enough for the loader
 *
 * The same permutation for every K: tile (tk, tn) holds 256 symbols in trellis-position order, position p at
 * (k = 16 tk + r, n = 16 tn + c) with (r, c) = exl3_pos_to_tile(p), each symbol K bits MSB-first in a stream of
 * u32 words that are themselves MSB-first (stream bit b is bit 31 - b % 32 of word b / 32; common/exl3_decode.c
 * exl3_tile_pack). A canonical row keeps its K-bit symbols LSB-first from bit k K (exl3_codes_get), so a tile's
 * 16 symbols of one row are 2K whole bytes at byte 2K tk. One tile: its 16 row segments read once, then the 256
 * symbols through a 64-bit accumulator. Checked against common/exl3_layout.c both ways (tests/gpu/layout_test.c). */
static void tile_order(int tr[256], int tc[256]) {
    for (int p = 0; p < 256; ++p) exl3_pos_to_tile(p, &tr[p], &tc[p]);
}

void exl3r4d_kx_to_record(const uint8_t* codes, int64_t row_bytes, int64_t rows, int64_t cols, int bits,
                          uint32_t* record) {
    int tr[256], tc[256];
    tile_order(tr, tc);
    const int64_t nt = rows / 16, kt = cols / 16, seg = 2 * bits;   /* a row's 16 symbols: 2K bytes */
    const uint32_t mask = (1u << bits) - 1u;
    for (int64_t tk = 0; tk < kt; ++tk)
        for (int64_t tn = 0; tn < nt; ++tn) {
            uint8_t sym[16][16];   /* [c][r]: row n = 16 tn + c, column k = 16 tk + r */
            for (int c = 0; c < 16; ++c) {
                uint8_t b[18] = { 0 };
                memcpy(b, codes + (tn * 16 + c) * row_bytes + tk * seg, (size_t)seg);
                for (int r = 0; r < 16; ++r) {
                    const int o = r * bits;
                    sym[c][r] = (uint8_t)(((uint32_t)b[o >> 3] | (uint32_t)b[(o >> 3) + 1] << 8) >> (o & 7) & mask);
                }
            }
            uint32_t* w = record + (tk * nt + tn) * 8 * bits;
            uint64_t acc = 0;
            int nb = 0;
            for (int p = 0; p < 256; ++p) {
                acc = acc << bits | sym[tc[p]][tr[p]];
                nb += bits;
                if (nb >= 32) {
                    nb -= 32;
                    *w++ = (uint32_t)(acc >> nb);
                    acc &= (1ull << nb) - 1ull;
                }
            }
        }
}

void exl3r4d_kx_from_record(const uint32_t* record, int64_t rows, int64_t cols, int bits, uint8_t* codes,
                            int64_t row_bytes) {
    int tr[256], tc[256];
    tile_order(tr, tc);
    const int64_t nt = rows / 16, kt = cols / 16, seg = 2 * bits;
    const uint32_t mask = (1u << bits) - 1u;
    for (int64_t tk = 0; tk < kt; ++tk)
        for (int64_t tn = 0; tn < nt; ++tn) {
            uint8_t sym[16][16];
            const uint32_t* w = record + (tk * nt + tn) * 8 * bits;
            uint64_t acc = 0;
            int nb = 0;
            for (int p = 0; p < 256; ++p) {
                if (nb < bits) {
                    acc = acc << 32 | *w++;
                    nb += 32;
                }
                nb -= bits;
                sym[tc[p]][tr[p]] = (uint8_t)(acc >> nb & mask);
                acc &= (1ull << nb) - 1ull;
            }
            for (int c = 0; c < 16; ++c) {
                uint8_t b[18] = { 0 };
                for (int r = 0; r < 16; ++r) {
                    const int o = r * bits;
                    const uint32_t v = (uint32_t)sym[c][r] << (o & 7);
                    b[o >> 3] |= (uint8_t)v;
                    b[(o >> 3) + 1] |= (uint8_t)(v >> 8);
                }
                memcpy(codes + (tn * 16 + c) * row_bytes + tk * seg, b, (size_t)seg);
            }
        }
}

/* ------------------------------------------------------------------ which plane an operand is */

/* Operand -> (class, plane), or -1 when the operand is not a plane table of this op. */
static int which_plane(int gate_up, int operand, int* cls) {
    const Exl3Positions* pos = exl3_positions(gate_up);
    for (int c = 0; c < 2; ++c)
        for (int q = 0; q < EXL3_N_P; ++q)
            if (pos->plane[c][q] == operand) {
                *cls = c;
                return q;
            }
    return -1;
}

static const char* const k_plane_name[EXL3_N_P] = { "codes", "suh", "suh.1", "svh", "suh" };

/* The checks every hook makes, and the class geometry. RAD_OK, or the refusal the hook returns:
 * RAD_E_DTYPE for an encoding this library does not decode (not exl3, K outside 2..6, not mul1, a stacked
 * encoding on down or an unstacked one on gate_up), RAD_E_SHAPE for the wrong plane or extents. */
static int check(int gate_up, const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                 const int* sel, const RadTensor* planes, int n_planes, int* plane, int64_t* rows,
                 int64_t* cols, int* kbits) {
    int c = 0, bits, cb, stacked;
    const int q = which_plane(gate_up, operand, &c);
    Exl3Geom g;
    int64_t er, ec;
    if (q < 0) return RAD_E_UNSUPPORTED;
    /* v3: gate_up's dn_suh is a view of DOWN's suh plane (down's own, unstacked encoding): a scale vector, stored as
     * it is like every scale plane, so there is nothing to lay out and nothing of gate_up's encoding to check -- but
     * its SIZE is checked here, at declare, so a mis-wired table (another class's, or down's other parity's width) is
     * refused by name at load rather than read wrong in a serving step: rows / 2 elements, the class's width. */
    if (q == EXL3_P_DN_SUH) {
        /* down's own (unstacked) encoding, its "suh" plane, at the size gate_up's class implies; claimed in the same
         * form down's hook gives that weight */
        if (exl3_geom_parse(p, n_p, gate_up, exl3_geom_two_classes(p, n_p), &g) != RAD_OK) return RAD_E_SHAPE;
        if (c >= g.n_class || n_planes != 1 || !planes || planes[0].rank != 2) return RAD_E_SHAPE;
        exl3_plane_extents(g.rows[c], g.cols[c], q, &er, &ec);
        if (planes[0].shape[0] * planes[0].shape[1] != er * ec) return RAD_E_SHAPE;
        if (exl3_encoding_read(enc, &bits, &cb, &stacked) != RAD_OK) return RAD_E_DTYPE;
        if (!exl3r4d_bits_ok(bits) || cb != EXL3_CB_MUL1 || stacked) return RAD_E_DTYPE;
        if (!sel || sel[0] < 0 || sel[0] >= (int)enc->n_planes || strcmp(enc->plane[sel[0]].role, "suh") != 0)
            return RAD_E_SHAPE;
        if (planes[0].dtype != RAD_F16) return RAD_E_DTYPE;
        *plane = q;
        *rows = g.rows[c];
        *cols = g.cols[c];
        *kbits = bits;
        return RAD_OK;
    }
    if (exl3_encoding_read(enc, &bits, &cb, &stacked) != RAD_OK) return RAD_E_DTYPE;
    /* `gate_up` is the op here (EXL3_DOWN, EXL3_GATE_UP, EXL3_PREP_GU; v4): prep_gu reads gate_up's stacked runs */
    if (!exl3r4d_bits_ok(bits) || cb != EXL3_CB_MUL1 || stacked != (gate_up != EXL3_DOWN)) return RAD_E_DTYPE;
    if (exl3_geom_parse(p, n_p, gate_up, exl3_geom_two_classes(p, n_p), &g) != RAD_OK) return RAD_E_SHAPE;
    if (c >= g.n_class) return RAD_E_SHAPE;
    if (n_planes != 1 || !sel || !planes || sel[0] < 0 || sel[0] >= (int)enc->n_planes ||
        strcmp(enc->plane[sel[0]].role, k_plane_name[q]) != 0)
        return RAD_E_SHAPE;
    exl3_plane_extents(g.rows[c], g.cols[c], q, &er, &ec);
    if (planes[0].rank != 2) return RAD_E_SHAPE;
    /* codes: exactly the class's [rows, cols]; a scale vector by its element count (include/exl3_abi.h, THE PLANES:
     * the declared shape is what an issue carries, any rank whose extents multiply to C or R) */
    if (q == EXL3_P_CODES ? planes[0].shape[0] != er || planes[0].shape[1] != ec
                          : planes[0].shape[0] * planes[0].shape[1] != er * ec)
        return RAD_E_SHAPE;
    if (q != EXL3_P_CODES && planes[0].dtype != RAD_F16) return RAD_E_DTYPE;
    /* the codes plane handed is the encoding's K (R23: a uK plane of another K than the encoding names is refused,
     * not read as the encoding's K) */
    if (q == EXL3_P_CODES && planes[0].dtype != exl3_codes_dtype(bits)) return RAD_E_DTYPE;
    /* the op's dtype names the same K as the weight's encoding (a K-bit op handed another K's weight) */
    if (g.bits != bits) return RAD_E_DTYPE;
    *plane = q;
    *rows = g.rows[c];
    *cols = g.cols[c];
    *kbits = bits;
    return RAD_OK;
}

static int64_t record_bytes(int64_t rows, int64_t cols, int bits) { return rows * cols * bits / 8; }
static const char* const k_tag[EXL3R4D_BITS_MAX + 1] = { 0, 0, "exl3r4d.trellis.k2", "exl3r4d.trellis.k3",
                                                         "exl3r4d.trellis.k4", "exl3r4d.trellis.k5",
                                                         "exl3r4d.trellis.k6" };
static int64_t vec_bytes(const RadTensor* t) { return t->shape[0] * t->shape[1] * 2; }   /* a scale plane, f16 */

/* ------------------------------------------------------------------ the hooks */

/* moe_exl3_gate_up_down (v5) reads gate_up's planes and down's: each of its plane operands is laid out exactly as
 * the half that reads it lays out the same weight (the stored form must be one for every op reading it), so its
 * hooks redirect to that half -- gate_up's view (h_rot 1, a_rot 0) or down's ([K, N / 2], h_rot 1). */
typedef struct GdView { int op, operand, n_p; RadParam p[11]; } GdView;
static int gd_view(const RadParam* p, int n_p, int operand, GdView* v) {
    const Exl3Positions* gd = exl3_positions(EXL3_GU_DN);
    const Exl3Positions* gu = exl3_positions(EXL3_GATE_UP);
    const Exl3Positions* dn = exl3_positions(EXL3_DOWN);
    RadParam pg[11], pd[10];
    int ng = 0, nd = 0;
    exl3_gd_params(p, n_p, pg, &ng, pd, &nd);
    for (int c = 0; c < 2; ++c)
        for (int q = 0; q < EXL3_N_P; ++q) {
            if (gd->plane[c][q] != operand) continue;
            if (q <= EXL3_P_DN_SUH) {
                v->op = EXL3_GATE_UP;
                v->operand = gu->plane[c][q];
                v->n_p = ng;
                memcpy(v->p, pg, sizeof(RadParam) * (size_t)ng);
            } else {
                v->op = EXL3_DOWN;
                v->operand = dn->plane[c][q == EXL3_P_DN_CODES ? EXL3_P_CODES : EXL3_P_SVH];
                v->n_p = nd;
                memcpy(v->p, pd, sizeof(RadParam) * (size_t)nd);
            }
            return RAD_OK;
        }
    return RAD_E_UNSUPPORTED;
}

int exl3r4d_layout(int gate_up, const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                   const int* sel, const RadTensor* planes, int n_planes, RadLayout* out) {
    if (gate_up == EXL3_GU_DN) {
        GdView v;
        if (gd_view(p, n_p, operand, &v) != RAD_OK) return RAD_E_UNSUPPORTED;
        return exl3r4d_layout(v.op, v.p, v.n_p, v.operand, enc, sel, planes, n_planes, out);
    }
    int q, bits;
    int64_t rows, cols;
    const int rc = check(gate_up, p, n_p, operand, enc, sel, planes, n_planes, &q, &rows, &cols, &bits);
    if (rc != RAD_OK) return rc;
    if (!out) return RAD_E_INVAL;
    memset(out, 0, sizeof *out);
    if (q != EXL3_P_CODES) {   /* suh, suh.1, svh, dn_suh: the same bytes, 256-B aligned (L-ALIGN, above) */
        out->tag = "exl3r4d.f16.vec";
        out->dtype = RAD_F16;
        out->rank = 2;
        out->shape[0] = planes[0].shape[0];
        out->shape[1] = planes[0].shape[1];
        out->align = 256;
        out->bytes = vec_bytes(&planes[0]);
        return RAD_OK;
    }
    out->tag = k_tag[bits];
    out->dtype = EXL3R4D_DT_TRELLIS;
    out->rank = 2;
    out->shape[0] = rows;
    out->shape[1] = cols;
    out->align = 256;   /* the .rad sub-plane alignment; every tile (32 K bytes) starts on 16 B */
    out->bytes = record_bytes(rows, cols, bits);
    return RAD_OK;
}

int exl3r4d_relayout(int gate_up, const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                     const int* sel, const RadTensor* planes, int n_planes, void* dst,
                     int64_t dst_bytes) {
    if (gate_up == EXL3_GU_DN) {
        GdView v;
        if (gd_view(p, n_p, operand, &v) != RAD_OK) return RAD_E_UNSUPPORTED;
        return exl3r4d_relayout(v.op, v.p, v.n_p, v.operand, enc, sel, planes, n_planes, dst, dst_bytes);
    }
    int q, bits;
    int64_t rows, cols;
    const int rc = check(gate_up, p, n_p, operand, enc, sel, planes, n_planes, &q, &rows, &cols, &bits);
    if (rc != RAD_OK) return rc;
    if (q != EXL3_P_CODES) {   /* the identity form: a copy */
        if (!dst || !planes[0].data || dst_bytes < vec_bytes(&planes[0])) return RAD_E_INVAL;
        memcpy(dst, planes[0].data, (size_t)vec_bytes(&planes[0]));
        return RAD_OK;
    }
    if (!dst || !planes[0].data || dst_bytes < record_bytes(rows, cols, bits)) return RAD_E_INVAL;
    const int64_t rb = exl3_codes_row_bytes(cols, bits);
    const int64_t half = gate_up ? rows / 2 : rows;   /* gate rows then up rows: two records */
    for (int m = 0; m < (gate_up ? 2 : 1); ++m) {
        const uint8_t* src = (const uint8_t*)planes[0].data + m * half * rb;
        uint32_t* rec = (uint32_t*)dst + m * record_bytes(half, cols, bits) / 4;
        if (bits == 4) exl3r4d_k4_to_record(src, rb, half, cols, rec);
        else exl3r4d_kx_to_record(src, rb, half, cols, bits, rec);
    }
    return RAD_OK;
}

int exl3r4d_unrelayout(int gate_up, const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                       const int* sel, const void* src, int64_t src_bytes, const RadTensor* planes,
                       int n_planes) {
    if (gate_up == EXL3_GU_DN) {
        GdView v;
        if (gd_view(p, n_p, operand, &v) != RAD_OK) return RAD_E_UNSUPPORTED;
        return exl3r4d_unrelayout(v.op, v.p, v.n_p, v.operand, enc, sel, src, src_bytes, planes, n_planes);
    }
    int q, bits;
    int64_t rows, cols;
    const int rc = check(gate_up, p, n_p, operand, enc, sel, planes, n_planes, &q, &rows, &cols, &bits);
    if (rc != RAD_OK) return rc;
    if (q != EXL3_P_CODES) {
        if (!src || !planes[0].data || src_bytes < vec_bytes(&planes[0])) return RAD_E_INVAL;
        memcpy(planes[0].data, src, (size_t)vec_bytes(&planes[0]));
        return RAD_OK;
    }
    if (!src || !planes[0].data || src_bytes < record_bytes(rows, cols, bits)) return RAD_E_INVAL;
    const int64_t rb = exl3_codes_row_bytes(cols, bits);
    const int64_t half = gate_up ? rows / 2 : rows;
    for (int m = 0; m < (gate_up ? 2 : 1); ++m) {
        const uint32_t* rec = (const uint32_t*)src + m * record_bytes(half, cols, bits) / 4;
        uint8_t* dstp = (uint8_t*)planes[0].data + m * half * rb;
        if (bits == 4) exl3r4d_k4_from_record(rec, half, cols, dstp, rb);
        else exl3r4d_kx_from_record(rec, half, cols, bits, dstp, rb);
    }
    return RAD_OK;
}
