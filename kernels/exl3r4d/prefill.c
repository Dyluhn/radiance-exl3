/* prefill.c -- the launch and scratch hooks of exl3r4d's M > 16 rows (rows.c, the prefill section). Owner: P5.
 * Plain C against the ABI: nothing allocated, nothing synchronised, no environment read (R6).
 *
 * THE FORM IS CHOSEN BY SLOTS PER EXPERT, NOT BY M. A band of M > 16 tokens holds two different workloads: a decode
 * step at C >= 5 with MTP depth 3 (M = 20..64; on the real model T = 200..640 rows over 512 experts, runs of a row
 * or two) and a prefill chunk (runs of tens of rows). The first is the decode band's problem and takes its kernels
 * (P4's exl3r4d_launch: one workgroup an expert's run, MTX 1); the second takes the run-aligned MTX GEMM
 * (exl3r4d_prefill.hip). The crossover is libr4d's, a run length: decode-like while 2 T <= 3 n_expert -- an average
 * of 1.5 routed rows an expert (r4d_moe.hip r4d_moe_decode_band at 1.3.0). Re-measured for these kernels: lab notes in
 * exl3r4d_prefill.hip.
 *
 * WHAT A LAUNCH REFUSES is the decode adapter's list (launch.c, via exl3r4d_fill), plus RAD_E_SCRATCH when the
 * prefill form was not given the scratch its map and rows need, and RAD_E_SHAPE when the prep's rows would not be
 * addressable by the 32-bit lane offsets (T * mats * K * 2 bytes >= 4 GiB). */
#include "exl3r4d_prefill.h"

/* Run ownership: 16-row blocks a workgroup owns at most (Dylan's MTX 4 / 8) and column tiles a wave (NPW).
 * Defaults are Dylan's prefill choice (exl3-moe hip.py:594-601: MT_MAX 4 at more than 16 tokens, npw 2); MTX 8 is
 * built beside it and measured (lab notes). Compile-time, not tunables, until rad-tune can see a WTAB op (P7). */
#ifndef EXL3R4D_PF_MTX
#define EXL3R4D_PF_MTX 4
#endif
#ifndef EXL3R4D_PF_NPW
#define EXL3R4D_PF_NPW 2
#endif

/* The GEMM's A operand (exl3r4d_launch_prefill's `am`), a digit per op: 1 gate_up, 2 down are the low and high digit
 * of EXL3R4D_PF_AM (gate_up + 4 * down). 0 (fragments from the prep's rows in global memory each K step) is the
 * measured default; 1 (Q3's rotate-in-loop) and 2 (the rows staged through LDS once a workgroup) were measured slower
 * and stay as arms (lab notes in exl3r4d_prefill.hip). EXL3R4D_PF_PREP: 1 the prep by map entry (measured, adopted),
 * 0 P4's prep kernel (one workgroup an expert). */
#ifndef EXL3R4D_PF_AM
#define EXL3R4D_PF_AM 0
#endif
#ifndef EXL3R4D_PF_PREP
#define EXL3R4D_PF_PREP 1
#endif
static int am_of(int gate_up) { return gate_up ? EXL3R4D_PF_AM % 4 : EXL3R4D_PF_AM / 4; }
static int arot(int gate_up) { return am_of(gate_up) == 1; }

/* The crossover, rows an expert on average: decode-like while XD * T <= XN * n_expert (libr4d's 2 T <= 3 n_expert). A
 * pair per op (gate_up, down); XN 0 puts the whole band on the prefill form (a reading build). */
#ifndef EXL3R4D_PF_XN_GU
#define EXL3R4D_PF_XN_GU 0
#endif
#ifndef EXL3R4D_PF_XN_DN
#define EXL3R4D_PF_XN_DN 0
#endif
#ifndef EXL3R4D_PF_XD
#define EXL3R4D_PF_XD 2
#endif
static int decode_like(int gate_up, int64_t T, int64_t n_expert) {
    return EXL3R4D_PF_XD * T <= (gate_up ? EXL3R4D_PF_XN_GU : EXL3R4D_PF_XN_DN) * n_expert;
}

static int64_t round256(int64_t b) { return (b + 255) / 256 * 256; }

/* The prep's rows (f16 [T][mats * x_cols]) then the map (a 16-byte header and 16 bytes an entry). */
static int64_t rows_bytes(const Exl3Geom* g, int64_t T) {
    return arot(g->gate_up) ? 0 : round256(T * (g->gate_up ? 2 : 1) * g->x_cols * 2);
}
static int64_t map_bytes(int64_t cap) { return round256(16 + 16 * cap); }

static int64_t pf_bytes(const Exl3Geom* g, int64_t T) {
    return rows_bytes(g, T) + map_bytes(exl3r4d_pf_map_cap(T, g->n_expert, 16 * EXL3R4D_PF_MTX));
}

int exl3r4d_pf_launch(const RadArgs* a, RadStream s, int gate_up) {
    Exl3Geom g;
    Exl3r4dPrefill d;
    memset(&d, 0, sizeof d);
    int rc = exl3r4d_fill(a, gate_up, &g, &d.g);
    if (rc != RAD_OK) return rc;
    if (d.g.T == 0) return RAD_OK;   /* no rows: nothing to write */
    if (decode_like(gate_up, d.g.T, g.n_expert)) return exl3r4d_launch(a, s, gate_up);
    const int64_t T = d.g.T, xr_pitch = (gate_up ? 2 : 1) * g.x_cols;
    if (!arot(gate_up) && T * xr_pitch * 2 >= ((int64_t)1 << 32)) return RAD_E_SHAPE;
    if (!a->scratch || a->scratch_bytes < pf_bytes(&g, T)) return RAD_E_SCRATCH;
    if (!rad_ptr_aligned(a->scratch, 16)) return RAD_E_ALIGN;   /* the map's 16-byte entries */
    d.g.xr = arot(gate_up) ? 0 : a->scratch;
    d.g.xr_pitch = xr_pitch;
    d.g.chunk = 16;
    d.g.sk = 1;
    d.map = (uint32_t*)((char*)a->scratch + rows_bytes(&g, T));
    d.fm = 16 * EXL3R4D_PF_MTX;
    d.cap = (int32_t)exl3r4d_pf_map_cap(T, g.n_expert, d.fm);
    return exl3r4d_launch_prefill(&d, gate_up, EXL3R4D_PF_MTX, EXL3R4D_PF_NPW, am_of(gate_up), EXL3R4D_PF_PREP, s);
}

/* Sized at declare for the band's top (M = its upper bound, T = M * top_k), from the parameters alone: the larger of
 * the two forms' needs, because a launch anywhere in the band may take either -- the decode form below the
 * crossover (P4's hook answers for it), the prefill form above it. Both grow with T, so the top bounds the band. */
int64_t exl3r4d_pf_scratch(const RadArgs* a, int gate_up) {
    Exl3Geom g;
    if (exl3_geom_parse(a->p, a->n_p, gate_up, exl3_geom_two_classes(a->p, a->n_p), &g) != RAD_OK) return 0;
    const int64_t T = g.M * g.top_k, dec = exl3r4d_scratch(a, gate_up);
    if (decode_like(gate_up, T, g.n_expert)) return dec;
    const int64_t pf = pf_bytes(&g, T);
    return pf > dec ? pf : dec;
}
