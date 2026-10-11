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

/* The configuration (exl3r4d_prefill.h) as this file uses it. Compile-time, not tunables, until rad-tune can see a
 * WTAB op (P7). */
/* AM 1 and 3 rotate in the GEMM: no prep, no rows in scratch. */
static int rot_am(int am) { return am == 1 || am == 3; }
static int am_of(int gate_up, int run) {
    const int am = run ? EXL3R4D_PF_AM_RUN : EXL3R4D_PF_AM_MAP;
    return gate_up ? am % 4 : am / 4;
}

/* RUN FORM OR MAP FORM, by slots per expert: the run form (no map: one workgroup an expert's whole run) while
 * RD * T <= RN * n_expert, libr4d's decode band rule (2 T <= 3 n_expert) as the default. */
#ifndef EXL3R4D_PF_RN_GU
#define EXL3R4D_PF_RN_GU 3
#endif
#ifndef EXL3R4D_PF_RN_DN
#define EXL3R4D_PF_RN_DN 0   /* down: the map form at every T -- the run form was measured +8..+14% (lab notes) */
#endif
#ifndef EXL3R4D_PF_RD
#define EXL3R4D_PF_RD 2
#endif
static int64_t run_rn(int gate_up) { return gate_up ? EXL3R4D_PF_RN_GU : EXL3R4D_PF_RN_DN; }
static int run_form(int gate_up, int64_t T, int64_t n_expert) {
    return EXL3R4D_PF_RD * T <= run_rn(gate_up) * n_expert;
}

/* P4'S DECODE KERNELS AT M > 16 (exl3r4d_launch), an arm: while XD * T <= XN * n_expert. MEASURED SLOWER than the MTX
 * GEMM at every M > 16 (down M=32 -15%, M=64 -20% on the MTX form; gate_up 0..-2%), so XN is 0. */
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
static int64_t rows_bytes(const Exl3Geom* g, int64_t T, int run) {
    return rot_am(am_of(g->gate_up, run)) ? 0 : round256(T * (g->gate_up ? 2 : 1) * g->x_cols * 2);
}
static int64_t map_bytes(const Exl3Geom* g, int64_t T, int run) {
    return run ? 0 : round256(16 + 16 * exl3r4d_pf_map_cap(T, g->n_expert, 16 * EXL3R4D_PF_MTX));
}
static int64_t pf_bytes(const Exl3Geom* g, int64_t T) {
    const int run = run_form(g->gate_up, T, g->n_expert);
    return rows_bytes(g, T, run) + map_bytes(g, T, run);
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
    const int run = run_form(gate_up, T, g.n_expert), am = am_of(gate_up, run);
    if (!rot_am(am) && T * xr_pitch * 2 >= ((int64_t)1 << 32)) return RAD_E_SHAPE;
    const int64_t need = rows_bytes(&g, T, run) + map_bytes(&g, T, run);
    if (need > 0 && (!a->scratch || a->scratch_bytes < need)) return RAD_E_SCRATCH;
    if (need > 0 && !rad_ptr_aligned(a->scratch, 16)) return RAD_E_ALIGN;   /* the map's 16-byte entries */
    d.g.xr = rot_am(am) ? 0 : a->scratch;
    d.g.xr_pitch = xr_pitch;
    d.g.chunk = 16;
    d.g.sk = 1;
    d.map = run ? 0 : (uint32_t*)((char*)a->scratch + rows_bytes(&g, T, run));
    d.fm = 16 * EXL3R4D_PF_MTX;
    d.cap = run ? 0 : (int32_t)exl3r4d_pf_map_cap(T, g.n_expert, d.fm);
    const int prep = gate_up ? EXL3R4D_PF_PREP % 4 : EXL3R4D_PF_PREP / 4;
    return exl3r4d_launch_prefill(&d, gate_up, am, prep, run, s);
}

/* Sized at declare for the band's top (M = its upper bound, T = M * top_k), from the parameters alone: the larger of
 * the two forms' needs, because a launch anywhere in the band may take either -- the decode form below the
 * crossover (P4's hook answers for it), the prefill form above it. Both grow with T, so the top bounds the band. */
int64_t exl3r4d_pf_scratch(const RadArgs* a, int gate_up) {
    Exl3Geom g;
    if (exl3_geom_parse(a->p, a->n_p, gate_up, exl3_geom_two_classes(a->p, a->n_p), &g) != RAD_OK) return 0;
    /* P4's decode kernels only where the XN arm can send a launch to them (XN 0: never). */
    const int64_t T = g.M * g.top_k;
    const int64_t dec = (gate_up ? EXL3R4D_PF_XN_GU : EXL3R4D_PF_XN_DN) ? exl3r4d_scratch(a, gate_up) : 0;
    if (decode_like(gate_up, T, g.n_expert)) return dec;
    /* Every form a launch in the band can take, at its largest T: the run form up to the crossover, the map form
     * above it; both grow with T. */
    int64_t need = pf_bytes(&g, T);
    const int64_t Tr = run_rn(gate_up) * g.n_expert / EXL3R4D_PF_RD;   /* the run form's largest T */
    if (Tr < T) {
        const int64_t r = rows_bytes(&g, Tr, 1);
        need = r > need ? r : need;
    }
    return need > dec ? need : dec;
}
