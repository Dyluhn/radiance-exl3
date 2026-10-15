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
 * addressable by the 32-bit lane offsets (T rows, or a chunk's under the scratch cap, x mats x K x 2 bytes >= 4 GiB). */
#include <stddef.h>

#include "exl3r4d_prefill.h"

/* The configuration (exl3r4d_prefill.h) as this file uses it. Compile-time, not tunables, until rad-tune can see a
 * WTAB op (P7). */
/* AM 1 and 3 rotate in the GEMM: no prep, no rows in scratch. */
static int rot_am(int am) { return am == 1 || am == 3 || am == 4 || am == 5; }
/* AM 4 / 5 (gate_up's map form, fused: a workgroup an entry across every column) are taken only where they hold:
 * widths of at most 3 column blocks (tp2's 384 / 256: AM 4's registers, AM 5's 24 waves and LDS); elsewhere AM 0. */
static int am_of(const Exl3Geom* g, int run) {
    const int am = run ? EXL3R4D_PF_AM_RUN : EXL3R4D_PF_AM_MAP, a = g->gate_up ? am % 8 : am / 8;
    int wide = 0;
    for (int c = 0; c < g->n_class; ++c) wide |= g->rows[c] > 768;
    if ((a == 4 || a == 5) && (!g->gate_up || run || wide)) return 0;
    return a;
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
/* R23: gate_up's run form at K = 4 only. At K = 3 its MT 1 K loop compiles dependency-bound (256 instructions, 22
 * s_delay_alu, against the map form's 236 / 1) and runs +33..+37% at M = 17..64, where the map form is at K = 4's time
 * (runs/kperf3); K 2, 5, 6 take the map form likewise (RN_GU_KX). */
#ifndef EXL3R4D_PF_RN_GU_KX
#define EXL3R4D_PF_RN_GU_KX 0
#endif
static int64_t run_rn(int gate_up, int bits) {
    return gate_up ? (bits == 4 ? EXL3R4D_PF_RN_GU : EXL3R4D_PF_RN_GU_KX) : EXL3R4D_PF_RN_DN;
}
static int run_form(const Exl3Geom* g, int64_t T) {
    return EXL3R4D_PF_RD * T <= run_rn(g->gate_up, g->bits) * g->n_expert;
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

/* P10-K1, THE STREAMED y: down's map form writes y with whole-row nontemporal stores (exl3r4d_prefill.hip, the YS
 * kernels) when y (T rows x y_cols bf16) is at least YNT_MB MiB; 0: never. Measured, tp2 / tp1, us a layer, the _pf
 * chain gate_up -> down plus the next op (P10-K1 runs/probeC-1, probeD-1): M=256 -6.5 / -14.2, 512 -34.7 / -46.3,
 * 1024 -34.9 / -35.1, 2048 -26.1 / -15.9; below that it does not pay (M=32 / 64 / 128 at tp2: -2.3 / +1.2 / +0.4,
 * single op down M=32 +4.9) and y would leave the L2 its reader (the gather) finds it in. 8: gfx1201's L2 is 8 MiB -- a
 * y larger than that cannot stay there for its reader anyway (M=128: 6.6 MB, M=256: 13 MB at 2560 columns). */
#ifndef EXL3R4D_PF_YNT_MB
#define EXL3R4D_PF_YNT_MB 8
#endif
static int ystream(const Exl3Geom* g, int64_t T, int64_t y_cols, int run, int am) {
    return EXL3R4D_PF_YNT_MB > 0 && !g->gate_up && !run && am == 0 &&
           T * y_cols * 2 >= ((int64_t)EXL3R4D_PF_YNT_MB << 20);
}

static int64_t round256(int64_t b) { return (b + 255) / 256 * 256; }

/* Rows a map entry at most: 16 MTX, twice that with PAIR in gate_up's map form (AM 0). */
static int64_t fm_of(const Exl3Geom* g, int run) {
    return 16 * EXL3R4D_PF_MTX * (EXL3R4D_PF_PAIR && g->gate_up && !run && am_of(g, run) == 0 ? 2 : 1);
}

/* The prep's rows (f16 [T][mats * x_cols]) then the map (a 16-byte header and 16 bytes an entry), then -- chunked --
 * the chunk table (16 bytes a chunk). */
/* v3 EPI 1: with h_rot = 1 down's input arrives rotated (gate_up's epilogue did down's step 1), so down has no prep:
 * its GEMM reads h itself as the prep's rows. */
static int dn_prerot(const Exl3Geom* g) { return !g->gate_up && g->h_rot; }
static int64_t row_bytes(const Exl3Geom* g) { return (g->gate_up ? 2 : 1) * g->x_cols * 2; }
static int64_t map_entries(const Exl3Geom* g, int64_t T) {
    return exl3r4d_pf_map_cap(T, g->n_expert, fm_of(g, run_form(g, T)));
}
/* THE SCRATCH CAP (EXL3R4D_PF_SCRATCH_MB): gate_up's map form chunked by row when T rows would not fit. A chunk is the
 * entries that START in R consecutive rows; its rows fit R + fm (an entry is at most fm rows), so the scratch holds
 * RMAX + fm rows whatever T is, and a launch takes the fewest chunks of at most RMAX rows, balanced (equal R). */
static int64_t chunk_rmax(const Exl3Geom* g, int64_t T, int run) {   /* 0: not chunked */
    const int64_t fm = fm_of(g, run), cap = (int64_t)EXL3R4D_PF_SCRATCH_MB << 20, rb = row_bytes(g);
    if (EXL3R4D_PF_SCRATCH_MB <= 0 || run || !g->gate_up || rot_am(am_of(g, run)) || T * rb <= cap) return 0;
    const int64_t R = cap / rb - fm;
    return R < fm ? fm : R;
}
static int64_t n_chunks(const Exl3Geom* g, int64_t T, int run) {
    const int64_t R = chunk_rmax(g, T, run);
    return R ? (T + R - 1) / R : 1;
}
static int64_t chunk_rows(const Exl3Geom* g, int64_t T, int run) {
    const int64_t n = n_chunks(g, T, run);
    return chunk_rmax(g, T, run) ? (T + n - 1) / n : 0;
}
static int64_t rows_bytes(const Exl3Geom* g, int64_t T, int run) {
    if (dn_prerot(g) || rot_am(am_of(g, run))) return 0;
    const int64_t R = chunk_rmax(g, T, run);
    return round256((R ? R + fm_of(g, run) : T) * row_bytes(g));
}
static int64_t map_only_bytes(const Exl3Geom* g, int64_t T) { return round256(16 + 16 * map_entries(g, T)); }
static int64_t map_bytes(const Exl3Geom* g, int64_t T, int run) {
    if (run) return 0;
    return map_only_bytes(g, T) + (chunk_rows(g, T, run) ? round256(16 * n_chunks(g, T, run)) : 0);
}
static int64_t pf_bytes(const Exl3Geom* g, int64_t T) {
    const int run = run_form(g, T);
    return rows_bytes(g, T, run) + map_bytes(g, T, run);
}

/* `op`: EXL3_GATE_UP or EXL3_DOWN, or its _mod4 form (v6). */
int exl3r4d_pf_launch(const RadArgs* a, RadStream s, int op) {
    Exl3Geom g;
    Exl3r4dPrefill d;
    Exl3r4dCls hi[2];
    const int gate_up = exl3_op_base(op) != EXL3_DOWN;
    memset(&d, 0, sizeof d);
    int rc = exl3r4d_fill(a, op, &g, &d.g, hi);
    if (rc != RAD_OK) return rc;
    if (d.g.T == 0) return RAD_OK;   /* no rows: nothing to write */
    if (decode_like(gate_up, d.g.T, g.n_expert)) return exl3r4d_launch(a, s, op);
    const int64_t T = d.g.T, xr_pitch = dn_prerot(&g) ? d.g.x_pitch : (gate_up ? 2 : 1) * g.x_cols;
    const int run = run_form(&g, T), am = am_of(&g, run);
    /* A rotation in the GEMM would rotate an h_rot input a second time: those arms refuse it by name. */
    if (dn_prerot(&g) && rot_am(am)) return RAD_E_UNSUPPORTED;
    {   /* the A operand's 32-bit lane offsets: T rows, or a chunk's at most (the scratch cap) */
        const int64_t R = chunk_rmax(&g, T, run), xrows = R ? R + fm_of(&g, run) : T;
        if (!rot_am(am) && xrows * xr_pitch * 2 >= ((int64_t)1 << 32)) return RAD_E_SHAPE;
    }
    const int64_t need = rows_bytes(&g, T, run) + map_bytes(&g, T, run);
    if (need > 0 && (!a->scratch || a->scratch_bytes < need)) return RAD_E_SCRATCH;
    if (need > 0 && !rad_ptr_aligned(a->scratch, 16)) return RAD_E_ALIGN;   /* the map's 16-byte entries */
    d.g.xr = dn_prerot(&g) ? (void*)d.g.x : rot_am(am) ? 0 : a->scratch;
    d.g.xr_pitch = xr_pitch;
    d.g.chunk = 16;
    d.g.sk = 1;
    d.map = run ? 0 : (uint32_t*)((char*)a->scratch + rows_bytes(&g, T, run));
    d.fm = (int32_t)fm_of(&g, run);
    d.cap = run ? 0 : (int32_t)exl3r4d_pf_map_cap(T, g.n_expert, d.fm);
    d.chunk_n = 1;
    d.ystream = ystream(&g, T, d.g.y_cols, run, am);
    {
        const int64_t R = chunk_rows(&g, T, run);
        if (R) {   /* the scratch cap: chunks by row; a chunk's entries <= R / fm + 1 whole-or-partial entries of the
                    * runs it cuts, + one partial entry a run (expert or zero range) ending in it -- and <= cap */
            const int64_t fm = d.fm, runs = g.n_expert + 2 < R ? g.n_expert + 2 : R;
            const int64_t gy = R / fm + 1 + runs;
            d.chunk_n = (int32_t)n_chunks(&g, T, run);
            d.chunk_rows = (int32_t)R;
            d.chunk_gy = (int32_t)(gy < d.cap ? gy : d.cap);
            d.ctab = (uint32_t*)((char*)d.map + map_only_bytes(&g, T));
        }
    }
    /* prep 3: none (h_rot's down reads h as it is) */
    const int prep = dn_prerot(&g) ? 3 : gate_up ? EXL3R4D_PF_PREP % 4 : EXL3R4D_PF_PREP / 4;
    if (g.n_class != 4) return exl3r4d_launch_prefill(&d, gate_up, g.bits, am, prep, run, s);
    /* v6: the four-class block -- the same fields after the decode-band block (exl3r4d_prefill_fields.h) */
    Exl3r4dPrefill4 d4;
    if ((rc = exl3r4d_gemm4(&d.g, hi, &d4.g)) != RAD_OK) return rc;
    memcpy(&d4.map, &d.map, sizeof d - offsetof(Exl3r4dPrefill, map));
    return exl3r4d_launch_prefill4(&d4, gate_up, g.bits, am, prep, run, s);
}
_Static_assert(sizeof(Exl3r4dPrefill) - offsetof(Exl3r4dPrefill, map) ==
                   sizeof(Exl3r4dPrefill4) - offsetof(Exl3r4dPrefill4, map),
               "the two prefill blocks share every field after their decode-band blocks");

/* Sized at declare for the band's top (M = its upper bound, T = M * top_k), from the parameters alone: the larger of
 * the two forms' needs, because a launch anywhere in the band may take either -- the decode form below the
 * crossover (P4's hook answers for it), the prefill form above it. Both grow with T, so the top bounds the band. */
int64_t exl3r4d_pf_scratch(const RadArgs* a, int op) {
    Exl3Geom g;
    const int gate_up = exl3_op_base(op) != EXL3_DOWN;
    if (exl3_geom_parse(a->p, a->n_p, op, exl3_geom_two_classes(a->p, a->n_p), &g) != RAD_OK) return 0;
    /* P4's decode kernels only where the XN arm can send a launch to them (XN 0: never). */
    const int64_t T = g.M * g.top_k;
    const int64_t dec = (gate_up ? EXL3R4D_PF_XN_GU : EXL3R4D_PF_XN_DN) ? exl3r4d_scratch(a, op) : 0;
    if (decode_like(gate_up, T, g.n_expert)) return dec;
    /* Every form a launch in the band can take, at its largest T: the run form up to the crossover, the map form
     * above it; both grow with T. */
    int64_t need = pf_bytes(&g, T);
    const int64_t Tr = run_rn(gate_up, g.bits) * g.n_expert / EXL3R4D_PF_RD;   /* the run form's largest T */
    if (Tr < T) {
        const int64_t r = rows_bytes(&g, Tr, 1);
        need = r > need ? r : need;
    }
    return need > dec ? need : dec;
}
