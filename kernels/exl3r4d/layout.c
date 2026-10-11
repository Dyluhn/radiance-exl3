/* layout.c -- what exl3r4d stores for each plane table, and the conversions both ways.
 *
 * THE CODES PLANE is stored as exllamav3's tile-stream record (common/exl3_layout.h): the form the trellis decode
 * streams, tile (tk, tn) of 16 x 16 weights as 8*K little-endian u32 words, tiles k-major. A stacked gate_up is
 * gate's record then up's ([2][K/16][W/16][8*K] u32), which is how the kernel indexes matrix `mat`. Lossless and
 * derived from nothing but the bits (rad_abi.h: a relayout moves bits). ONE stored form for every band (R4): the
 * prefill band reads the same record (P5).
 *
 * suh, suh.1 and svh are stored AS THEY ARE (RAD_E_UNSUPPORTED): the kernel reads the f16 vectors in place.
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
 * RAD_E_DTYPE for an encoding this library does not decode (not exl3, K != 4, not mul1, a stacked
 * encoding on down or an unstacked one on gate_up), RAD_E_SHAPE for the wrong plane or extents. */
static int check(int gate_up, const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                 const int* sel, const RadTensor* planes, int n_planes, int* plane, int64_t* rows,
                 int64_t* cols) {
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
        if (exl3_geom_parse(p, n_p, gate_up, exl3_geom_two_classes(p, n_p), &g) != RAD_OK) return RAD_E_SHAPE;
        if (c >= g.n_class || n_planes != 1 || !planes || planes[0].rank != 2) return RAD_E_SHAPE;
        exl3_plane_extents(g.rows[c], g.cols[c], q, &er, &ec);
        if (planes[0].shape[0] * planes[0].shape[1] != er * ec) return RAD_E_SHAPE;
        return RAD_E_UNSUPPORTED;
    }
    if (exl3_encoding_read(enc, &bits, &cb, &stacked) != RAD_OK) return RAD_E_DTYPE;
    /* `gate_up` is the op here (EXL3_DOWN, EXL3_GATE_UP, EXL3_PREP_GU; v4): prep_gu reads gate_up's stacked runs */
    if (bits != EXL3R4D_BITS || cb != EXL3_CB_MUL1 || stacked != (gate_up != EXL3_DOWN)) return RAD_E_DTYPE;
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
    *plane = q;
    *rows = g.rows[c];
    *cols = g.cols[c];
    return RAD_OK;
}

static int64_t record_bytes(int64_t rows, int64_t cols) { return rows * cols * EXL3R4D_BITS / 8; }

/* ------------------------------------------------------------------ the hooks */

int exl3r4d_layout(int gate_up, const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                   const int* sel, const RadTensor* planes, int n_planes, RadLayout* out) {
    int q;
    int64_t rows, cols;
    const int rc = check(gate_up, p, n_p, operand, enc, sel, planes, n_planes, &q, &rows, &cols);
    if (rc != RAD_OK) return rc;
    if (q != EXL3_P_CODES) return RAD_E_UNSUPPORTED;   /* suh, suh.1, svh: stored as they are */
    if (!out) return RAD_E_INVAL;
    memset(out, 0, sizeof *out);
    out->tag = "exl3r4d.trellis.k4";
    out->dtype = EXL3R4D_DT_TRELLIS;
    out->rank = 2;
    out->shape[0] = rows;
    out->shape[1] = cols;
    out->align = 256;   /* the .rad sub-plane alignment; every tile (128 B at K=4) starts on 16 B */
    out->bytes = record_bytes(rows, cols);
    return RAD_OK;
}

int exl3r4d_relayout(int gate_up, const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                     const int* sel, const RadTensor* planes, int n_planes, void* dst,
                     int64_t dst_bytes) {
    int q;
    int64_t rows, cols;
    const int rc = check(gate_up, p, n_p, operand, enc, sel, planes, n_planes, &q, &rows, &cols);
    if (rc != RAD_OK) return rc;
    if (q != EXL3_P_CODES) return RAD_E_UNSUPPORTED;
    if (!dst || !planes[0].data || dst_bytes < record_bytes(rows, cols)) return RAD_E_INVAL;
    const int64_t rb = exl3_codes_row_bytes(cols, EXL3R4D_BITS);
    const int64_t half = gate_up ? rows / 2 : rows;   /* gate rows then up rows: two records */
    for (int m = 0; m < (gate_up ? 2 : 1); ++m)
        exl3r4d_k4_to_record((const uint8_t*)planes[0].data + m * half * rb, rb, half, cols,
                             (uint32_t*)dst + m * record_bytes(half, cols) / 4);
    return RAD_OK;
}

int exl3r4d_unrelayout(int gate_up, const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                       const int* sel, const void* src, int64_t src_bytes, const RadTensor* planes,
                       int n_planes) {
    int q;
    int64_t rows, cols;
    const int rc = check(gate_up, p, n_p, operand, enc, sel, planes, n_planes, &q, &rows, &cols);
    if (rc != RAD_OK) return rc;
    if (q != EXL3_P_CODES) return RAD_E_UNSUPPORTED;
    if (!src || !planes[0].data || src_bytes < record_bytes(rows, cols)) return RAD_E_INVAL;
    const int64_t rb = exl3_codes_row_bytes(cols, EXL3R4D_BITS);
    const int64_t half = gate_up ? rows / 2 : rows;
    for (int m = 0; m < (gate_up ? 2 : 1); ++m)
        exl3r4d_k4_from_record((const uint32_t*)src + m * record_bytes(half, cols) / 4, half, cols,
                               (uint8_t*)planes[0].data + m * half * rb, rb);
    return RAD_OK;
}
