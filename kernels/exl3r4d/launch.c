/* launch.c -- the launch adapters of both rows: RadArgs -> the device argument block (exl3r4d_dev.h), and the
 * scratch hook. Plain C against the ABI; everything is read off the operands and the resolved parameters, nothing
 * is allocated, nothing synchronises, no environment is read (REQUIREMENTS R6; its grep over this directory is empty).
 *
 * WHAT A LAUNCH REFUSES, before anything is written, each by name (rad_types.h codes):
 *   RAD_E_INVAL    a required operand absent, or the odd class partly absent
 *   RAD_E_SCHEMA   a required parameter absent (exl3_geom_parse)
 *   RAD_E_DTYPE    K != 4, a codebook other than mul1, an operand of the wrong dtype
 *   RAD_E_SHAPE    a geometry the format or these kernels cannot hold; tables shorter
 *                  than the class; an output narrower or wider than the wider class
 *   RAD_E_STRIDE   a non-unit inner stride; routing operands not contiguous
 *   RAD_E_ALIGN    rows that the 8-byte activation loads or the 16-byte output stores cannot take
 *   RAD_E_SCRATCH  the prep path needs scratch the call was not given
 * The routing VALUES are not checked: they live in device memory, and reading them would be a synchronisation.
 * The kernels clamp every run to [0, T) instead, so a malformed table gives wrong rows, never a stray write. */
#include <pthread.h>
#include <stddef.h>
#include <stdlib.h>
#include "exl3r4d.h"
#include "exl3r4d_dev.h"
#include "exl3r4d_prefill.h"

/* Decode defaults, from Dylan's measured choice (exl3-moe hip.py:594-601 and FACTS-exl3 §2.4): two column tiles
 * a wave, LDS split-K 2 ("measured best at M 1..5, 384 and 256 wide"). Re-taken on this card (runs/shape2, prep
 * path): within 4% of the best of NPW {2,4} x SK {1,2,4,8} at M=1 for both ops; NPW 4 / SK 1 is 5.7% faster at
 * gate_up M=16 and NPW 2 / SK 1 5.2% at down M=16 -- tunables for rad-tune (PLAN P7), not a hand-picked change. */
#ifndef EXL3R4D_PF_ANY_K
#define EXL3R4D_PF_ANY_K 1   /* R23: the M > 16 band's kernels read every K the decode band does (P5) */
#endif
#ifndef EXL3R4D_NPW
#define EXL3R4D_NPW 2
#endif
#ifndef EXL3R4D_SK
#define EXL3R4D_SK 1  /* adopted (round 2, OCC-gu): gate_up split-K 1 -- tp2 M=4 -6.8%, M=1 +3.9% */
#endif
/* Per op (round 2, D.8 retune): down's own column tiles a wave and split-K; default the shared ones. */
#ifndef EXL3R4D_CTR_LAZY   /* TEST builds: a lazily allocated counter pool for tools that skip init (never shipped) */
#define EXL3R4D_CTR_LAZY 0
#endif
#ifndef EXL3R4D_NPW_DN
#define EXL3R4D_NPW_DN EXL3R4D_NPW
#endif
/* gate_up's LDS split-K at FEW tokens (D16, deterministic: the two halves combined in a fixed order through LDS;
 * approved 2026-10-12): a launch of at most EXL3R4D_SK_SMALL_M tokens takes split-K 2 (round 2's OCC-gu: split-K 1 is
 * +3.9% at tp2 M=1). Tokens = sorted rows / top_k, so the engine's band-top M parameter does not matter. 0: off; 2
 * by default (engine v14). The
 * bits of those launches move (within the tolerance suites); the fused gate_up_down launch is split-K 1 only. */
#ifndef EXL3R4D_SK_SMALL_M
#define EXL3R4D_SK_SMALL_M 2
#endif
#ifndef EXL3R4D_SK_SMALL_V   /* the split-K those launches take */
#define EXL3R4D_SK_SMALL_V 2
#endif
/* L7 (EXL3R4D_SK_FIT): ...but only when every unit the grid can hold -- its column blocks times min(T, n_expert) live
 * experts -- is resident at once at split-K 2 (exl3r4d_gu_resident: the device's WGPs times the workgroups a WGP holds).
 * Split-K 2 doubles a workgroup's waves, so a WGP holds half as many (174 VGPRs: 2 sixteen-wave workgroups a WGP, 64
 * on a 32-WGP card), and a grid past that runs a second round of latency-bound workgroups. Reading (P9-L7 sB-1,
 * 0000:03:00.0, gate_up GEMM alone): tp1 M=2 = 5 blocks x 20 rows -> 100 > 64 -> split-K 2 66.3 us, split-K 1 59.8
 * (libr4d 62.6); tp2 M=2 (3 x 20 = 60) and M=1 at tp1 / tp2 (50 / 30) fit and keep split-K 2 (tp2 M=1 is +7 % at
 * split-K 1). Deterministic in the device, the geometry and the token count, never the routing. Measured (sD-1, ABAB
 * x3): chain tp1 M=2 104.8 -> 98.6 us (libr4d 103.2), MTP-like 2-token runs 93.0 -> 86.5 (libr4d 90.9); every other
 * shape within 0.07 us and its y byte-identical (tp1 M=1/3/4, tp2 M=1..4). NUMERICS (accepted by the orchestrator,
 * 2026-10-14, as a change of summation order only): a launch that leaves split-K 2 -- tp1 M=2 here -- sums each output
 * over all of K in one chain (split-K 1's fp32 order, the M >= 3 form) instead of two halves combined through LDS; the
 * quantised weights, the activation format and the rotation are untouched; the suites pass at their tolerances
 * (sG-1: device 946/0, real 32/0, tp 360/0, P2 cases 457/0, band tops 80/0). tp2 bits never move. 0: the token rule. */
#ifndef EXL3R4D_SK_FIT
#define EXL3R4D_SK_FIT 1
#endif
#ifndef EXL3R4D_SK_ONE_V     /* reading arm: another split-K for 1-token launches only (0: as above) */
#define EXL3R4D_SK_ONE_V 0
#endif
/* ...and down's, likewise (D16; round 2's D.8 chose down split-K 1 from M = 1 / 4 / 16 -- M = 2 never measured).
 * P9-L7 sM-1 (split-K 2 at every decode M; tp2's 24-k-step units are latency chains, two rounds): SLOWER everywhere --
 * down tp2 M=1..6 +1.2/+3.6/+3.9/+3.5/+3.4/+3.9 us, MTP-verify M=4 +2.4, tp1 M=1..4 +1.2..+3.9; chain tp2 M=1/2/3
 * +1.3/+4.0/+4.9. Halving each wave's chain does not pay for half the resident units. Off. */
#ifndef EXL3R4D_SK_DN_SMALL_M
#define EXL3R4D_SK_DN_SMALL_M 0
#endif
#ifndef EXL3R4D_SK_DN   /* adopted (round 2, D.8): down at split-K 1 -- tp2 M=1 -4.9%, M=4 -8.6%, M=16 -11.2% */
#define EXL3R4D_SK_DN 1
#endif
/* The most rows a fused-prep pass may take (<= 16 and what LDS holds); above it the prep kernel runs. 0: the prep
 * kernel at every M -- MEASURED FASTER at decode on gfx1201 (lever L3, lab notes in exl3r4d.hip: gate_up M=1 -15%,
 * M=4 -26%; down M=1 -21%, M=16 -29%), so the fused prep is the switched-off arm. Reopen it if a change makes the
 * in-loop A fragments as cheap from LDS as from L2 (then the fused prep saves the prep dispatch at M <= 6). */
/* Lever L8 (the split epilogue): 1 = GEMM workgroups of EXL3R4D_TPW 16-column tiles of one matrix write raw f32
 * sums to scratch and a finisher kernel does the rotation, scales and SiLU / bf16 (lab notes in exl3r4d.hip). */
#ifndef EXL3R4D_SPLIT
#define EXL3R4D_SPLIT 0
#endif
#ifndef EXL3R4D_TPW
#define EXL3R4D_TPW 4
#endif
#ifndef EXL3R4D_FUSED_MAX
#define EXL3R4D_FUSED_MAX 0
#endif

/* Tokens a run can hold at most: an expert takes each token once (top-k picks distinct experts). */
static int64_t run_rows_max(const Exl3Geom* g, int64_t n_tok, int64_t T) {
    const int64_t t = g->gate_up ? n_tok : (T + g->top_k - 1) / g->top_k;
    return t < 16 ? t : 16;
}

/* One plane table of one class into the device form. `want` elements an entry (the plane's extents). */
static int table(const RadTensor* t, int plane, int64_t entries, int64_t rows, int64_t cols, int bits,
                 const void** base, int64_t* s0) {
    int64_t er, ec, per = 1;
    exl3_plane_extents(rows, cols, plane, &er, &ec);
    if (!t || t->rank < 2) return RAD_E_SHAPE;
    if (t->shape[0] < entries) return RAD_E_SHAPE;
    for (uint32_t d = 1; d < t->rank; ++d) per *= t->shape[d];
    if (plane == EXL3_P_CODES) {
        /* One entry is this library's stored record (layout.c), described [rows, cols] of EXL3R4D_DT_TRELLIS:
         * the engine's pointer table, or a tool's stacked records ([entries, rows, cols] packed, or u8 [entries,
         * record bytes]). Canonical codes (uK) are not read here: they go through the layout hook first. */
        const int64_t bytes = rows * cols * bits / 8;
        if (t->dtype == EXL3R4D_DT_TRELLIS) {   /* the engine's pointer table, or records back to back */
            if (per != rows * cols || (t->stride[0] != 0 && t->stride[0] != per)) return RAD_E_SHAPE;
            *s0 = t->stride[0] != 0 ? bytes : 0;
        } else if (t->dtype == RAD_U8) {
            if (per != bytes || (t->stride[0] != 0 && t->stride[0] % 256)) return RAD_E_SHAPE;
            *s0 = t->stride[0];
        } else {
            return RAD_E_DTYPE;
        }
    } else {
        if (t->dtype != RAD_F16) return RAD_E_DTYPE;
        if (per != er * ec) return RAD_E_SHAPE;
        if (t->stride[0] != 0 && (t->stride[0] * 2) % 16) return RAD_E_ALIGN;
        *s0 = t->stride[0] * 2;
    }
    if (t->stride[0] != 0 && !rad_ptr_aligned(t->data, 256)) return RAD_E_ALIGN;
    *base = t->data;
    return RAD_OK;
}

/* Everything but the scratch decision: operands checked and the argument block filled. */
/* `op`: EXL3_DOWN, EXL3_GATE_UP or EXL3_PREP_GU (v4; 0 and 1 are v3's gate_up flag), or a _mod4 form (v6), whose
 * classes 2 and 3 go to `hi` (the four-class block is packed from both by gemm4). prep_gu shares gate_up's suh /
 * suh.1 tables and input and writes xr, so `gate_up` below is true for it too. */
static int fill(const RadArgs* a, int op, Exl3Geom* g, Exl3r4dGemm* d, Exl3r4dCls hi[2]) {
    const int base = exl3_op_base(op), gate_up = base != EXL3_DOWN, prep = base == EXL3_PREP_GU;
    const Exl3Positions* pos = exl3_positions(op);
    const RadTensor *x = rad_arg_in(a, pos->x), *y = rad_arg_in(a, pos->y);
    const RadTensor *srt = rad_arg_in(a, pos->sorted), *off = rad_arg_in(a, pos->offset);
    int n_odd = 0, rc;
    memset(d, 0, sizeof *d);
    if (hi) memset(hi, 0, 2 * sizeof *hi);   /* a plane the op does not read stays null, as in d */
    if (!x || !y || !srt || !off) return RAD_E_INVAL;
    for (int q = 0; q < EXL3_N_P; ++q) {
        if (pos->plane[0][q] < 0 || q == EXL3_P_DN_SUH) continue;   /* dn_suh: optional, required by h_rot below */
        for (int c = 0; c < (pos->n_cls == 4 ? 4 : 1); ++c)        /* a _mod4 op's classes are all required */
            if (!rad_arg_in(a, pos->plane[c][q])) return RAD_E_INVAL;
        n_odd += rad_arg_in(a, pos->plane[1][q]) != 0;
    }
    const int n_planes = prep ? 2 : gate_up ? 4 : 3;
    if (n_odd != 0 && n_odd != n_planes) return RAD_E_INVAL;   /* a partly absent odd class (exl3ref's code) */
    if ((rc = exl3_geom_parse(a->p, a->n_p, op, n_odd != 0, g)) != RAD_OK) return rc;
    if (!exl3r4d_bits_ok(g->bits) || (!prep && g->codebook != EXL3_CB_MUL1)) return RAD_E_DTYPE;
    if (g->n_expert > 4096) return RAD_E_SHAPE;
    if (g->n_class == 4 && !hi) return RAD_E_INVAL;
    /* v3 EPI 1: with h_rot gate_up reads down's suh tables, every class's (dn_suh_odd with the odd class) */
    const int dn_rd = base == EXL3_GATE_UP && g->h_rot;
    for (int c = 0; dn_rd && c < g->n_class; ++c)
        if (!rad_arg_in(a, pos->plane[c][EXL3_P_DN_SUH])) return RAD_E_INVAL;

    /* activations and output */
    /* v4: gate_up's `a` at a_rot = 1 is prep_gu's xr, f16 [T][2 * x_cols] in sorted-row order */
    if (gate_up && !g->a_rot ? (x->dtype != RAD_F16 && x->dtype != RAD_BF16) : x->dtype != RAD_F16) return RAD_E_DTYPE;
    if (y->dtype != (gate_up ? RAD_F16 : RAD_BF16)) return RAD_E_DTYPE;
    if (srt->dtype != RAD_I32 || off->dtype != RAD_I32) return RAD_E_DTYPE;
    if (!rad_tensor_is_contiguous(srt) || !rad_tensor_is_contiguous(off)) return RAD_E_STRIDE;
    if (x->rank < 2 || y->rank < 2) return RAD_E_SHAPE;
    if (x->stride[x->rank - 1] != 1 || y->stride[y->rank - 1] != 1) return RAD_E_STRIDE;
    const int64_t T = rad_tensor_rows(y), n_tok = rad_tensor_rows(x);
    if (rad_tensor_lastdim(y) != g->y_cols || rad_tensor_lastdim(x) < g->x_cols) return RAD_E_SHAPE;
    if (rad_tensor_numel(srt) < T || rad_tensor_numel(off) < g->n_expert + 1) return RAD_E_SHAPE;
    if ((!gate_up || g->a_rot) && n_tok < T) return RAD_E_SHAPE;
    if (g->a_rot && rad_tensor_lastdim(x) < 2 * g->x_cols) return RAD_E_SHAPE;
    if (T > 0x7fffffff / 2 || n_tok > 0x7fffffff / 2) return RAD_E_SHAPE;
    /* rotate128 loads 8 bytes a lane (x rows 8-B aligned); the epilogue stores 16 bytes a lane */
    if (!rad_ptr_aligned(x->data, 8) || (rad_tensor_row_pitch(x) * 2) % 8) return RAD_E_ALIGN;
    /* an input that arrives rotated (down's h' at h_rot = 1, gate_up's xr at a_rot = 1) is read as the prep kernel's
     * rows: 16 bytes a lane (exl3r4d.hip, the A fragment loads) */
    if (((!gate_up && g->h_rot) || g->a_rot) &&
        (!rad_ptr_aligned(x->data, 16) || (rad_tensor_row_pitch(x) * 2) % 16))
        return RAD_E_ALIGN;
    if (!rad_ptr_aligned(y->data, 16) || (rad_tensor_row_pitch(y) * 2) % 16) return RAD_E_ALIGN;

    /* the plane tables */
    for (int c = 0; c < g->n_class; ++c) {
        Exl3r4dCls* k = c < 2 ? &d->cls[c] : &hi[c - 2];
        const int64_t ent = exl3_class_entries(g, c);
        const void** base[EXL3_N_P] = { &k->codes, &k->suh, &k->suh1, &k->svh, &k->dn_suh };
        int64_t* s0[EXL3_N_P] = { &k->codes_s0, &k->suh_s0, &k->suh1_s0, &k->svh_s0, &k->dn_suh_s0 };
        for (int q = 0; q < EXL3_N_P; ++q) {
            if (pos->plane[c][q] < 0 || (q == EXL3_P_DN_SUH && !dn_rd)) continue;
            rc = table(rad_arg_in(a, pos->plane[c][q]), q, ent, g->rows[c], g->cols[c], g->bits, base[q], s0[q]);
            if (rc != RAD_OK) return rc;
        }
        k->rows = (int32_t)g->rows[c];
        k->cols = (int32_t)g->cols[c];
    }
    if (g->n_class == 1) d->cls[1] = d->cls[0];

    d->x = x->data;
    d->y = y->data;
    d->sorted = (const int32_t*)srt->data;
    d->off = (const int32_t*)off->data;
    d->x_pitch = rad_tensor_row_pitch(x);
    d->y_pitch = rad_tensor_row_pitch(y);
    d->T = (int32_t)T;
    d->n_tok = (int32_t)n_tok;
    d->n_expert = (int32_t)g->n_expert;
    d->top_k = (int32_t)g->top_k;
    d->n_class = g->n_class;
    d->h_rot = g->h_rot;
    d->a_rot = g->a_rot;
    d->y_cols = (int32_t)g->y_cols;
    d->x_bf16 = x->dtype == RAD_BF16;
    d->sk = gate_up ? EXL3R4D_SK : EXL3R4D_SK_DN;
    if (gate_up && EXL3R4D_SK_SMALL_M > 0 && (T + g->top_k - 1) / g->top_k <= EXL3R4D_SK_SMALL_M) d->sk = EXL3R4D_SK_SMALL_V;
    if (gate_up && EXL3R4D_SK_ONE_V > 0 && (T + g->top_k - 1) / g->top_k <= 1) d->sk = EXL3R4D_SK_ONE_V;
    if (gate_up && EXL3R4D_SK_FIT && d->sk > 1) {
        const int64_t units = (g->y_cols + 127) / 128 * (T < g->n_expert ? T : g->n_expert);
        const int fit = exl3r4d_gu_resident(g->bits, EXL3R4D_NPW, d->sk, g->h_rot);
        if (fit > 0 && units > fit) d->sk = 1;
    }
    if (!gate_up && EXL3R4D_SK_DN_SMALL_M > 0 && (T + g->top_k - 1) / g->top_k <= EXL3R4D_SK_DN_SMALL_M) d->sk = 2;
    /* grid.y: every live expert (at most one a sorted row) and one spare that zeroes the rows no run owns */
    d->nslot = (int32_t)((T < g->n_expert ? T : g->n_expert) + 1);
    return RAD_OK;
}

/* P5: the prefill band's adapter (prefill.c, rows.c's M > 16 section) checks its operands and fills the same argument
 * block through this same function, so the two bands cannot accept different operands. */
int exl3r4d_fill(const RadArgs* a, int op, Exl3Geom* g, Exl3r4dGemm* d, Exl3r4dCls hi[2]) {
    const int rc = fill(a, op, g, d, hi);
    /* R23: both bands read K 2..6 (EXL3R4D_PF_ANY_K 0 with rows.c's EXL3R4D_PREFILL_DTYPES "exl3k4": the M > 16 band
     * back at K = 4 alone) */
    return rc == RAD_OK && g->bits != 4 && !EXL3R4D_PF_ANY_K ? RAD_E_DTYPE : rc;
}

/* Prep-path scratch: the rotated rows, f16 [T][mats * x_cols]. */
/* P1c: one copy of the rotated rows per 128-column block of the output, each written and read by its own GEMM
 * workgroups only (exl3r4d_dev.h). */
static int64_t prep_bytes(const Exl3Geom* g, int64_t T) {
    const int gscr = g->gate_up ? EXL3R4D_GSCR_GU : EXL3R4D_GSCR_DN;
    const int64_t ncb = (g->y_cols + 127) / 128, grp = g->gate_up ? 1 : EXL3R4D_DNG;
    const int64_t copies = gscr && T <= EXL3R4D_GSCR_MAXT ? (ncb + grp - 1) / grp : 1;
    return (copies * T * (g->gate_up ? 2 : 1) * g->x_cols * 2 + 255) / 256 * 256;
}

/* Split-epilogue scratch after the prep's: the raw f32 sums [T][the widest class's N]. */
static int64_t raw_pitch(const Exl3Geom* g) {
    int64_t w = 0;
    for (int c = 0; c < EXL3_MAX_CLS; ++c) w = g->rows[c] > w ? g->rows[c] : w;
    return w;
}

/* THE FOUR-CLASS ARGUMENT BLOCK (v6), packed from fill's two-class block and its classes 2 and 3: the same fields after
 * the class records (exl3r4d_gemm_fields.h, one list for both), and each record's entry strides narrowed to 32 bits --
 * RAD_E_SHAPE for a stacked table whose stride does not fit (the engine's pointer tables have none). */
_Static_assert(sizeof(Exl3r4dGemm) - offsetof(Exl3r4dGemm, x) == sizeof(Exl3r4dGemm4) - offsetof(Exl3r4dGemm4, x),
               "the two argument blocks share every field after their class records");
static int s0_32(int64_t s0, int32_t* out) {
    if (s0 < 0 || s0 > 0x7fffffff) return RAD_E_SHAPE;
    *out = (int32_t)s0;
    return RAD_OK;
}
int exl3r4d_gemm4(const Exl3r4dGemm* d, const Exl3r4dCls hi[2], Exl3r4dGemm4* o) {
    memcpy(&o->x, &d->x, sizeof(Exl3r4dGemm) - offsetof(Exl3r4dGemm, x));
    for (int c = 0; c < 4; ++c) {
        const Exl3r4dCls* k = c < 2 ? &d->cls[c] : &hi[c - 2];
        Exl3r4dCls4* q = &o->cls[c];
        q->codes = k->codes; q->suh = k->suh; q->suh1 = k->suh1; q->svh = k->svh; q->dn_suh = k->dn_suh;
        q->rows = k->rows;
        q->cols = k->cols;
        if (s0_32(k->codes_s0, &q->codes_s0) || s0_32(k->suh_s0, &q->suh_s0) || s0_32(k->suh1_s0, &q->suh1_s0) ||
            s0_32(k->svh_s0, &q->svh_s0) || s0_32(k->dn_suh_s0, &q->dn_suh_s0))
            return RAD_E_SHAPE;
    }
    return RAD_OK;
}
/* The decode GEMM launch over the block fill made: the two-class build's, or the four-class build's (v6). */
static int launch_gemm(const Exl3r4dGemm* d, const Exl3r4dCls hi[2], int gate_up, int bits, int npw, RadStream s) {
    if (d->n_class != 4) return exl3r4d_launch_gemm(d, gate_up, bits, npw, s);
    Exl3r4dGemm4 d4;
    const int rc = exl3r4d_gemm4(d, hi, &d4);
    return rc != RAD_OK ? rc : exl3r4d_launch_gemm4(&d4, gate_up, bits, npw, s);
}
static int64_t raw_bytes(const Exl3Geom* g, int64_t T) {
    return EXL3R4D_SPLIT ? (T * raw_pitch(g) * 4 + 255) / 256 * 256 : 0;
}

/* ------------------------------------------------------------------ D18: the counter blocks
 *
 * gate_up's single launch (prep + GEMM, EXL3R4D_FUSE1) orders its workgroups through a counter block that is zero at
 * the launch's entry and that its last workgroup zeroes again. The blocks come from a pool the ROW INSTANCE's init
 * allocates and zeroes (RadKernelInfo::init: once per resolved instance, after selection, before load -- never the
 * launch path, R6), EXL3R4D_CTR_BLOCKS of them, one taken a (stream, output) pair on first use: two launches that
 * may overlap -- on two streams, or on one stream when the engine starts a launch before independent ones finish
 * (rad_kernel_concurrent) -- write disjoint outputs or run on distinct streams, so they never share a block, and
 * two that share one are ordered (libr4d's split_ctr, r4d_gemm_fp8a8.hip:1026-1048). A launch past the pool, or
 * with no instance (a tool that calls launch cold), takes the two-dispatch path: the same bytes. */
typedef struct Exl3r4dCtrPool {
    unsigned*       base;
    int*            wq_map;                      /* WGQ: HW_ID1 key -> dense WGP id (exl3r4d_wgp_probe) */
    int             wq_nl;                       /* WGQ: WGPs; 0: WGQ off for this instance (the grid) */
    int64_t         words;                       /* a block: 2 + the band's largest slot count, rounded */
    int             used;
    RadStream       key_s[EXL3R4D_CTR_BLOCKS];
    const void*     key_o[EXL3R4D_CTR_BLOCKS];
    pthread_mutex_t mu;
#if EXL3R4D_SUHC
    struct Exl3r4dSuhc* suhc;                    /* gate_up_down's resident suh (its prologue prep), or null */
#endif
} Exl3r4dCtrPool;

#if EXL3R4D_SUHC
/* ------------------------------------------------------------------ SUHC: the decode prep's resident suh
 *
 * Every expert's suh / suh.1 copied, as the prep first meets it, into VRAM the row instance owns (f16 [n_expert][2][K],
 * 5.2 MB a layer a rank at K 2560, 512 experts), so a pooled expert's vectors cross the link once, not at every step.
 * The bytes are the planes' own, so the output is the same bits. A SLOT is one layer's copy, keyed by the layer's
 * suh / suh.1 tables (one table per declared op and operand, fixed for the life of the instance); the engine resolves
 * an instance per declared op, so one slot is enough there. A launch past the slots reads the planes as before. */
#ifndef EXL3R4D_SUHC_SLOTS   /* tools that run many layers through one instance (perfmoe) build with more */
#define EXL3R4D_SUHC_SLOTS 1
#endif
typedef struct Exl3r4dSuhc {
    unsigned char*  data;                             /* [slots][n_expert][2][K] f16 */
    unsigned*       ok;                               /* [slots][n_expert] wave-slice bits */
    int64_t         n_expert, K;
    const void*     key[EXL3R4D_SUHC_SLOTS][2];
    int             used;
    pthread_mutex_t mu;
} Exl3r4dSuhc;

static Exl3r4dSuhc* suhc_new(const RadParam* p, int n_p) {
    const int64_t n = rad_param_geti(p, n_p, "n_expert", 0), k0 = rad_param_geti(p, n_p, "K", 0);
    const int64_t k1 = rad_param_geti(p, n_p, "K_odd", k0), K = k0 > k1 ? k0 : k1;
    if (n <= 0 || K <= 0) return NULL;
    Exl3r4dSuhc* c = (Exl3r4dSuhc*)calloc(1, sizeof *c);
    if (!c) return NULL;
    c->n_expert = n;
    c->K = K;
    c->data = (unsigned char*)exl3r4d_dev_zalloc((size_t)(EXL3R4D_SUHC_SLOTS * n * 2 * K) * 2);
    c->ok = (unsigned*)exl3r4d_dev_zalloc((size_t)(EXL3R4D_SUHC_SLOTS * n) * sizeof(unsigned));
    if (!c->data || !c->ok) {
        exl3r4d_dev_free(c->data);
        exl3r4d_dev_free(c->ok);
        free(c);
        return NULL;   /* no copy: every launch reads the planes (the same bits) */
    }
    pthread_mutex_init(&c->mu, NULL);
    return c;
}
static void suhc_free(Exl3r4dSuhc* c) {
    if (!c) return;
    exl3r4d_dev_free(c->data);
    exl3r4d_dev_free(c->ok);
    pthread_mutex_destroy(&c->mu);
    free(c);
}
/* This launch's slot, by its suh / suh.1 tables; none (the planes) past the slots or for another geometry. */
static void suhc_bind(Exl3r4dSuhc* c, Exl3r4dGemm* d) {
    d->suhc = NULL;
    d->suhc_ok = NULL;
    if (!c || d->n_expert > c->n_expert || d->cls[0].cols != c->K || (d->n_class > 1 && d->cls[1].cols != c->K)) return;
    const void* k0 = d->cls[0].suh;
    const void* k1 = d->n_class > 1 ? d->cls[1].suh : NULL;
    int s = -1;
    pthread_mutex_lock(&c->mu);
    for (int i = 0; i < c->used && s < 0; ++i)
        if (c->key[i][0] == k0 && c->key[i][1] == k1) s = i;
    if (s < 0 && c->used < EXL3R4D_SUHC_SLOTS) {
        s = c->used++;
        c->key[s][0] = k0;
        c->key[s][1] = k1;
    }
    pthread_mutex_unlock(&c->mu);
    if (s < 0) return;
    d->suhc = c->data + (size_t)s * (size_t)(c->n_expert * 2 * c->K) * 2;
    d->suhc_ok = c->ok + (size_t)s * (size_t)c->n_expert;
}
int exl3r4d_suhc_init(const RadParam* p, int n_p, void** out) {
    *out = suhc_new(p, n_p);
    return RAD_OK;
}
void exl3r4d_suhc_fini(void* inst) { suhc_free((Exl3r4dSuhc*)inst); }
#endif

/* A pool of `blocks` counter blocks of `words` words, zeroed; with WGQ also the WGP map (probed). NULL on failure. */
static Exl3r4dCtrPool* ctr_pool_new(int64_t words) {
    Exl3r4dCtrPool* q = (Exl3r4dCtrPool*)calloc(1, sizeof *q);
    if (!q) return NULL;
    q->words = (words + EXL3R4D_CTR_ROUND - 1) / EXL3R4D_CTR_ROUND * EXL3R4D_CTR_ROUND;
    q->base = (unsigned*)exl3r4d_dev_zalloc((size_t)(q->words * EXL3R4D_CTR_BLOCKS) * sizeof(unsigned));
    if (!q->base) { free(q); return NULL; }
#if EXL3R4D_WGQ
    q->wq_map = (int*)exl3r4d_dev_zalloc(EXL3R4D_WGQ_KEYS * sizeof(int));
    const int nl = q->wq_map ? exl3r4d_wgp_probe(q->wq_map) : 0;
    q->wq_nl = nl > 0 && 2 * nl + 1 <= q->words ? nl : 0;   /* a failed probe: the grid, same bytes */
#endif
    pthread_mutex_init(&q->mu, NULL);
    return q;
}

int exl3r4d_ctr_init(const RadParam* p, int n_p, void** out) {
    Exl3Geom g;
    *out = NULL;
    if (!EXL3R4D_FUSE1 && !EXL3R4D_WGQ) return RAD_OK;
    if (exl3_geom_parse(p, n_p, 1, exl3_geom_two_classes(p, n_p), &g) != RAD_OK) return RAD_OK;  /* launch refuses */
    const int64_t slots = g.M * g.top_k + 1, wq = 2 * EXL3R4D_WGQ_KEYS + 1;
    *out = ctr_pool_new(EXL3R4D_WGQ && wq > 2 + slots ? wq : 2 + slots);   /* NULL: every launch takes the grid */
    return RAD_OK;
}

/* PDN: the down row's pool -- tickets only (two words a block), no WGP map. */
int exl3r4d_ctr_init_dn(const RadParam* p, int n_p, void** out) {
    *out = NULL;
    if (!EXL3R4D_PDN) return RAD_OK;
    (void)p; (void)n_p;
    Exl3r4dCtrPool* q = (Exl3r4dCtrPool*)calloc(1, sizeof *q);
    if (!q) return RAD_OK;
    q->words = EXL3R4D_CTR_ROUND;
    q->base = (unsigned*)exl3r4d_dev_zalloc((size_t)(q->words * EXL3R4D_CTR_BLOCKS) * sizeof(unsigned));
    if (!q->base) { free(q); return RAD_OK; }        /* no blocks: every launch takes the grid */
    pthread_mutex_init(&q->mu, NULL);
    *out = q;
    return RAD_OK;
}

void exl3r4d_ctr_fini(void* inst) {
    Exl3r4dCtrPool* q = (Exl3r4dCtrPool*)inst;
    if (!q) return;
    exl3r4d_dev_free(q->wq_map);
    exl3r4d_dev_free(q->base);
#if EXL3R4D_SUHC
    suhc_free(q->suhc);
#endif
    pthread_mutex_destroy(&q->mu);
    free(q);
}

/* exl3r4d.hip: the fused decode launch (v5's moe_exl3_gate_up_down; `workgroups` 0 = the resident capacity) */
int exl3r4d_launch_fused(const Exl3r4dGemm* gu, const Exl3r4dGemm* dn, unsigned* ctr, int workgroups, int prep,
                         void* stream);

/* This launch's block (host bookkeeping only), or null: the two-dispatch path. */
#if EXL3R4D_CTR_LAZY
/* TEST BUILDS ONLY (never shipped): a tool that calls launch without init (P2's harness) gets one lazily allocated
 * pool, so the counter-block kernels are what it checks. */
static Exl3r4dCtrPool* ctr_lazy_pool(void) {
    static Exl3r4dCtrPool* lazy = NULL;
    if (!lazy) lazy = ctr_pool_new(2 * EXL3R4D_WGQ_KEYS + 1 > 2 + 4097 ? 2 * EXL3R4D_WGQ_KEYS + 1 : 2 + 4097);
    return lazy;
}
#endif

static unsigned* ctr_block(void* inst, RadStream s, const void* out, int64_t nslot) {
    Exl3r4dCtrPool* q = (Exl3r4dCtrPool*)inst;
#if EXL3R4D_CTR_LAZY
    if (!q) q = ctr_lazy_pool();
#endif
    if (!q || 2 + nslot > q->words) return NULL;
    unsigned* b = NULL;
    pthread_mutex_lock(&q->mu);
    for (int i = 0; i < q->used; ++i)
        if (q->key_s[i] == s && q->key_o[i] == out) { b = q->base + (int64_t)i * q->words; break; }
    if (!b && q->used < EXL3R4D_CTR_BLOCKS) {
        q->key_s[q->used] = s;
        q->key_o[q->used] = out;
        b = q->base + (int64_t)q->used * q->words;
        ++q->used;
    }
    pthread_mutex_unlock(&q->mu);
    return b;
}

/* `op`: EXL3_GATE_UP or EXL3_DOWN, or its _mod4 form (v6). */
int exl3r4d_launch(const RadArgs* a, RadStream s, int op) {
    Exl3Geom g;
    Exl3r4dGemm d;
    Exl3r4dCls hi[2];
    const int gate_up = exl3_op_base(op) != EXL3_DOWN;
    int rc = fill(a, op, &g, &d, hi);
    if (rc != RAD_OK) return rc;
    if (d.T == 0) return RAD_OK;   /* no rows: nothing to write */
    if (gate_up && EXL3R4D_FUSE1) d.ctr = ctr_block(a->instance, s, d.y, d.nslot);
    if (!gate_up && EXL3R4D_PDN) d.ctr = ctr_block(a->instance, s, d.y, 0);   /* PDN's ticket block, or the grid */
#if EXL3R4D_WGQ
    if (gate_up) {   /* WGQ: this (stream, output)'s counter block and the instance's WGP map */
        unsigned* b = ctr_block(a->instance, s, d.y, d.nslot);
        const Exl3r4dCtrPool* q = (const Exl3r4dCtrPool*)a->instance;
#if EXL3R4D_CTR_LAZY
        if (!q && b) q = ctr_lazy_pool();
#endif
        if (b && q && q->wq_nl > 0) { d.wq = b; d.wq_map = q->wq_map; d.wq_nl = q->wq_nl; }
    }
#endif
    if ((!gate_up && d.h_rot) || (gate_up && d.a_rot)) {
        /* v3 EPI 1 (down) / v4 a_rot (gate_up): the input is already rotated -- gate_up wrote h' for down, or
         * moe_exl3_prep_gu wrote xr for gate_up -- and the GEMM reads it as the prep kernel's rows */
        if (EXL3R4D_SPLIT) return RAD_E_UNSUPPORTED;
        d.xr = (void*)d.x;
        d.xr_pitch = d.x_pitch;
        d.chunk = 16;
        return launch_gemm(&d, hi, gate_up, g.bits, gate_up ? EXL3R4D_NPW : EXL3R4D_NPW_DN, s);
    }
    const int fit = exl3r4d_fused_rows(gate_up, g.x_cols);
    const int64_t need = run_rows_max(&g, d.n_tok, d.T);
    if (need >= 1 && need <= fit && need <= EXL3R4D_FUSED_MAX) {
        d.chunk = (int32_t)need;          /* the fused prep: rows rotated into LDS, `need` a pass */
    } else {
        if (!a->scratch || a->scratch_bytes < prep_bytes(&g, d.T) + raw_bytes(&g, d.T)) return RAD_E_SCRATCH;
        if (!rad_ptr_aligned(a->scratch, 16)) return RAD_E_ALIGN;
        d.xr = a->scratch;                /* the prep kernel, then the GEMM reads its rows */
        d.xr_pitch = (gate_up ? 2 : 1) * g.x_cols;
        d.chunk = 16;
        if (EXL3R4D_SPLIT) {
            d.raw = (float*)((char*)a->scratch + prep_bytes(&g, d.T));
            d.raw_pitch = raw_pitch(&g);
            d.tpw = EXL3R4D_TPW;
        }
    }
    return launch_gemm(&d, hi, gate_up, g.bits, gate_up ? EXL3R4D_NPW : EXL3R4D_NPW_DN, s);
}

#if EXL3R4D_FUSED_PROTO
/* PROTOTYPE (reading builds; perfmoe --fused): gate_up (a_rot 1, h_rot 1) and down (h_rot 1) as ONE launch
 * (exl3r4d.hip, exl3r4d_fused_kernel). `ctr`: a zeroed block of 2 + 256 + 2 + 256 + 256 u32 the caller owns, zero again after
 * except the last word, the kernel's give-up flag (a down unit that waited ~1 s for its gate_up units). */
int exl3r4d_fused_proto(const RadArgs* gu, const RadArgs* dn, unsigned* ctr, int workgroups, RadStream s) {
    Exl3Geom gg, gd;
    Exl3r4dGemm a, b;
    int rc = fill(gu, EXL3_GATE_UP, &gg, &a, NULL);
    if (rc == RAD_OK) rc = fill(dn, EXL3_DOWN, &gd, &b, NULL);
    if (rc != RAD_OK) return rc;
    if (!gg.h_rot || !gd.h_rot || gg.bits != 4 || gd.bits != 4) return RAD_E_UNSUPPORTED;
    if (a.T == 0) return RAD_OK;
    const int prep = !gg.a_rot;   /* a_rot 0: the launch rotates the token rows itself, into gate_up's scratch */
    if (prep) {
        if (!gu->scratch || gu->scratch_bytes < prep_bytes(&gg, a.T)) return RAD_E_SCRATCH;
        a.xr = gu->scratch; a.xr_pitch = 2 * gg.x_cols; a.chunk = 16;
    } else {
        a.xr = (void*)a.x; a.xr_pitch = a.x_pitch; a.chunk = 16;   /* prep_gu's xr, read as the prep kernel's rows */
    }
    b.xr = (void*)b.x; b.xr_pitch = b.x_pitch; b.chunk = 16;   /* gate_up's h', read so (h_rot) */
    return exl3r4d_launch_fused(&a, &b, ctr, workgroups, prep, s);
}
#endif

/* ------------------------------------------------------------------ v5: moe_exl3_gate_up_down
 *
 * The decode chain as one op (include/exl3_abi.h): read as its two halves (exl3_gd_params / exl3_gd_operands), each
 * filled by the same fill() as the separate ops, so it refuses exactly what they refuse. With a counter block from
 * the row's pool it is ONE launch (exl3r4d.hip, exl3r4d_fused_kernel: prep in the prologue, gate_up's and down's
 * units by D18 tickets); without one (a tool that skips init) or past the fused kernel's live-expert list it is the
 * separate ops' kernels in a row. Both are the separate ops' bits. Scratch: xr [T][2 K] then h [T][max(N) / 2]. */
enum { EXL3R4D_GD_MAX_LIVE = 256, EXL3R4D_GD_WORDS = 2 + 256 + 2 + 256 + 2 + 256 };
/* The fewest tokens the fused launch takes: below, the separate kernels are faster (runs/p4b-s23, chain tp2: M=1
 * 40.7 fused vs 39.7, M=2 63.4 vs 60.3; M=4 93.3 vs 101.5). TEST builds set 1 so the suites check the fused launch at
 * every M. */
#ifndef EXL3R4D_GD_MIN_TOK
#define EXL3R4D_GD_MIN_TOK EXL3_GD_MIN_ROWS   /* include/exl3_abi.h: the arch's issue band starts there too */
#endif
/* AP: where a run's rows fit a gate_up workgroup's LDS and are at most EXL3R4D_GD_AP, the fused launch's gate_up units
 * rotate them there themselves (exl3r4d_fused_kernel<true>) instead of the prologue's prep and its wait -- the grid's
 * fused prep, which loses to the prep kernel there (EXL3R4D_FUSED_MAX 0, lever L3) but here saves the prologue's prep
 * and wait. 0: the prologue at every M. */
#ifndef EXL3R4D_GD_AP
#define EXL3R4D_GD_AP 0   /* reading arm until measured */
#endif
static int64_t gd_xr_bytes(int64_t T, int64_t x_cols) { return (T * 2 * x_cols * 2 + 255) / 256 * 256; }

/* The scratch's first region: the fused launch's xr, or what gate_up's own launch stages (the separate kernels). */
static int64_t gd_head_bytes(const RadParam* p, int n_p, const Exl3Geom* g) {
    RadParam pg[11], pd[10];
    int ng = 0, nd = 0;
    exl3_gd_params(p, n_p, pg, &ng, pd, &nd);
    RadArgs ag;
    memset(&ag, 0, sizeof ag);
    ag.p = pg;
    ag.n_p = ng;
    const int64_t xr = gd_xr_bytes(g->M * g->top_k, g->x_cols), own = exl3r4d_scratch(&ag, 1);
    return ((xr > own ? xr : own) + 255) / 256 * 256;
}

int64_t exl3r4d_scratch_gd(const RadArgs* a) {
    Exl3Geom g;
    if (exl3_geom_parse(a->p, a->n_p, EXL3_GU_DN, exl3_geom_two_classes(a->p, a->n_p), &g) != RAD_OK) return 0;
    const int64_t T = g.M * g.top_k, w = (g.rows[0] > g.rows[1] ? g.rows[0] : g.rows[1]) / 2;
    return gd_head_bytes(a->p, a->n_p, &g) + (T * w * 2 + 255) / 256 * 256;
}

int exl3r4d_ctr_init_gd(const RadParam* p, int n_p, void** out) {
    (void)p; (void)n_p;
    *out = ctr_pool_new(EXL3R4D_GD_WORDS);   /* NULL: every launch takes the separate kernels (the same bits) */
#if EXL3R4D_SUHC
    if (*out) ((Exl3r4dCtrPool*)*out)->suhc = suhc_new(p, n_p);
#endif
    return RAD_OK;
}

int exl3r4d_launch_gd(const RadArgs* a, RadStream s) {
    const Exl3Positions* gd = exl3_positions(EXL3_GU_DN);
    if (a->n_t < gd->n_opd || !rad_arg_in(a, gd->y)) return RAD_E_INVAL;
    RadParam pg[11], pd[10];
    int ng = 0, nd = 0, rc;
    exl3_gd_params(a->p, a->n_p, pg, &ng, pd, &nd);
    const int64_t T = rad_tensor_rows(rad_arg_in(a, gd->y));
    const int64_t N = rad_param_geti(a->p, a->n_p, "N", 0), No = rad_param_geti(a->p, a->n_p, "N_odd", N);
    const int64_t K = rad_param_geti(a->p, a->n_p, "K", 0), Ko = rad_param_geti(a->p, a->n_p, "K_odd", K);
    const int64_t w = (N > No ? N : No) / 2, xc = K > Ko ? K : Ko;
    if (T <= 0 || w <= 0 || xc <= 0) return T == 0 ? RAD_OK : RAD_E_SHAPE;
    Exl3Geom gx;
    if ((rc = exl3_geom_parse(a->p, a->n_p, EXL3_GU_DN, exl3_geom_two_classes(a->p, a->n_p), &gx)) != RAD_OK) return rc;
    gx.M = (T + gx.top_k - 1) / gx.top_k;   /* this launch's rows, not the declared band top */
    const int64_t xrb = gd_head_bytes(a->p, a->n_p, &gx);
    if (gd_xr_bytes(T, xc) > xrb) return RAD_E_SCRATCH;
    if (!a->scratch || a->scratch_bytes < xrb + T * w * 2) return RAD_E_SCRATCH;
    if (!rad_ptr_aligned(a->scratch, 256)) return RAD_E_ALIGN;
    RadTensor h;
    memset(&h, 0, sizeof h);
    h.data = (char*)a->scratch + xrb;
    h.dtype = RAD_F16;
    h.rank = 2;
    h.shape[0] = T;
    h.shape[1] = w;
    rad_tensor_pack(&h);
    RadTensor tg[14], td[10];
    exl3_gd_operands(a->t, &h, tg, td);
    RadArgs ag = *a, ad = *a;
    ag.t = tg; ag.n_t = 14; ag.p = pg; ag.n_p = ng;
    ad.t = td; ad.n_t = 10; ad.p = pd; ad.n_p = nd;
    Exl3Geom gg, gdd;
    Exl3r4dGemm A, B;
    if ((rc = fill(&ag, EXL3_GATE_UP, &gg, &A, NULL)) != RAD_OK || (rc = fill(&ad, EXL3_DOWN, &gdd, &B, NULL)) != RAD_OK)
        return rc;
    A.xr = a->scratch; A.xr_pitch = 2 * gg.x_cols; A.chunk = 16;   /* the rotated rows */
#if EXL3R4D_SUHC
    suhc_bind(a->instance ? ((Exl3r4dCtrPool*)a->instance)->suhc : NULL, &A);
#endif
    B.xr = (void*)B.x; B.xr_pitch = B.x_pitch; B.chunk = 16;       /* h, written rotated by gate_up (h_rot) */
    if (EXL3R4D_SPLIT) return RAD_E_UNSUPPORTED;   /* down reads h' rotated (h_rot): no split epilogue */
    /* nslot - 1: the most live experts this launch can have (the kernel's list holds 256) */
    unsigned* ctr = gg.bits == 4 && gdd.bits == 4 && A.nslot - 1 <= EXL3R4D_GD_MAX_LIVE && A.sk == 1 &&
                            A.n_tok >= EXL3R4D_GD_MIN_TOK
                        ? ctr_block(a->instance, s, B.y, 0) : NULL;
    if (ctr) {
        const int fit = exl3r4d_fused_rows(1, gg.x_cols);
        const int64_t need = run_rows_max(&gg, A.n_tok, A.T);
        const int ap = EXL3R4D_GD_AP && need >= 1 && need <= fit && need <= EXL3R4D_GD_AP;   /* AP: the most rows */
        Exl3r4dGemm Af = A;
        if (ap) {
            Af.xr = NULL;
            Af.chunk = (int32_t)need;
        }
        rc = exl3r4d_launch_fused(&Af, &B, ctr, 0, !ap, s);
#ifdef EXL3R4D_GD_DEBUG
        if (rc != RAD_OK)
            fprintf(stderr, "exl3r4d gd: fused rc %d (T %lld n_tok %d nslot %d x_cols %lld)\n", rc, (long long)T,
                    (int)A.n_tok, (int)A.nslot, (long long)gg.x_cols);
#endif
        if (rc != RAD_E_UNSUPPORTED) return rc;   /* UNSUPPORTED: down's view does not travel compact (exl3r4d.hip) */
    }
    /* The separate ops' own launches, as their rows run them: gate_up staging in the head region, then down. */
    ag.scratch = a->scratch;
    ag.scratch_bytes = xrb;
    ag.instance = ad.instance = NULL;
    ad.scratch = NULL;
    ad.scratch_bytes = 0;
    rc = exl3r4d_launch(&ag, s, EXL3_GATE_UP);
    if (rc == RAD_OK) rc = exl3r4d_launch(&ad, s, EXL3_DOWN);
#ifdef EXL3R4D_GD_DEBUG
    if (rc != RAD_OK) fprintf(stderr, "exl3r4d gd: separate rc %d (T %lld head %lld)\n", rc, (long long)T, (long long)xrb);
#endif
    return rc;
}

/* v4: moe_exl3_prep_gu -- gate_up's input rotation as an op: the prep kernel, writing the op's own xr operand
 * (rows of the experts' runs; a dropped slot inside a run zero; nothing else written). */
/* The prep_gu op's form of the prep kernel (decode band only): 1 = the 2-D grid, every (row, matrix, block) task in
 * one round (lever L9; runs/p4b-s48: order-1 chain tp2 M=1 -0.5 us, M=2 -1.2..-1.8; engine h15, 3 passes: C=4 -0.10
 * ms a step, C=1/2/8 within noise); the same bits. 0 = the build's form, as gate_up's own prep and the M > 16 band
 * keep -- THE DEFAULT (P5b adoption ed530a6: the locked-screen build that met every decode target ran without it, and
 * it costs C=1 ~0.13 ms; from 3 tokens EXL3R4D_PREP_ROW takes the prep_gu op anyway, exl3r4d.hip). */
#ifndef EXL3R4D_PREPOP_GRID
#define EXL3R4D_PREPOP_GRID 0
#endif
/* `op`: EXL3_PREP_GU, or its _mod4 form (v6). */
int exl3r4d_launch_prep_op(const RadArgs* a, RadStream s, int op) {
    Exl3Geom g;
    Exl3r4dGemm d;
    Exl3r4dCls hi[2];
    int rc = fill(a, op, &g, &d, hi);
    if (rc != RAD_OK) return rc;
    if (d.T == 0) return RAD_OK;
    d.xr = d.y;
    d.xr_pitch = d.y_pitch;
    if (g.n_class == 4) {
        Exl3r4dGemm4 d4;
        rc = exl3r4d_gemm4(&d, hi, &d4);
        return rc != RAD_OK ? rc : exl3r4d_launch_prep_form4(&d4, 1, s, EXL3R4D_PREPOP_GRID);
    }
#if EXL3R4D_SUHC
    suhc_bind((Exl3r4dSuhc*)a->instance, &d);
#endif
    return exl3r4d_launch_prep_form(&d, 1, s, EXL3R4D_PREPOP_GRID);
}

/* Sized at declare for the band top, from the parameters alone (STEP0 S0.9): zero when the band's largest
 * run fits the fused prep, else the prep's rows for T = M * top_k. `op` as exl3r4d_launch's. */
int64_t exl3r4d_scratch(const RadArgs* a, int op) {
    Exl3Geom g;
    const int gate_up = exl3_op_base(op) != EXL3_DOWN;
    if (exl3_geom_parse(a->p, a->n_p, op, exl3_geom_two_classes(a->p, a->n_p), &g) != RAD_OK) return 0;
    const int64_t T = g.M * g.top_k;
    const int64_t need = run_rows_max(&g, g.M, T);
    if ((!gate_up && g.h_rot) || (gate_up && g.a_rot)) return 0;   /* the input arrives rotated: nothing to stage */
    if (need <= exl3r4d_fused_rows(gate_up, g.x_cols) && need <= EXL3R4D_FUSED_MAX) return 0;
    /* P1c copies the rows once a column block up to EXL3R4D_GSCR_MAXT rows, so a smaller step than the band top
     * can need more: the larger of the two regimes. */
    const int64_t tg = T < EXL3R4D_GSCR_MAXT ? T : EXL3R4D_GSCR_MAXT;
    const int64_t pb = prep_bytes(&g, T) > prep_bytes(&g, tg) ? prep_bytes(&g, T) : prep_bytes(&g, tg);
    return pb + raw_bytes(&g, T);
}
