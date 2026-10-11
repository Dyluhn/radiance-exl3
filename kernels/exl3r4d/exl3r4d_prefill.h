/* exl3r4d_prefill.h -- the prefill band (M > 16) of exl3r4d: what its host adapter (prefill.c) hands its device code
 * (exl3r4d_prefill.hip), and the row hooks rows.c registers. Owner: P5. The decode band's interface is
 * exl3r4d_dev.h (P4); this one only adds to it.
 *
 * TWO FORMS IN ONE BAND, chosen at launch by SLOTS PER EXPERT, not by M (prefill.c exl3r4d_pf_decode_like):
 *   decode-like  runs of a row or two an expert (decode at C >= 5 with MTP depth 3 lands at M = 20..64): P4's decode
 *                kernels (exl3r4d_launch, MTX 1, one workgroup an expert's run);
 *   prefill      runs of tens of rows: the run-aligned map in scratch, one workgroup a map entry of up to 16 * MTX
 *                rows of one expert, one decoded weight fragment feeding MTX WMMAs (Dylan's MTX 4 / 8). */
#ifndef EXL3R4D_PREFILL_H
#define EXL3R4D_PREFILL_H

#include <stdint.h>

#include "exl3r4d_dev.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The prefill form's argument block: the decode band's (operands, routing, the prep's rows in g.xr) plus the
 * run map. Passed by value to every kernel of the form; ~260 bytes, under the 768-byte kernel-argument limit
 * (docs/PLUGIN.md §5). */
typedef struct Exl3r4dPrefill {
    Exl3r4dGemm g;      /* g.xr: the prep kernel's rotated rows (scratch); g.sk is 1 */
    uint32_t*   map;    /* scratch: words 0..3 a header (word 0 = entries written), then `cap` entries of four
                         * words {expert or EXL3R4D_PF_ZERO, first row, end row, 0} */
    int32_t     cap;    /* entries the map can hold = grid.y of the GEMM */
    int32_t     fm;     /* rows an entry at most: 16 * MTX */
} Exl3r4dPrefill;

/* A map entry that owns no expert: rows [r0, r1) no run covers ([0, off[0]) and [off[n_expert], T)), zeroed. */
#define EXL3R4D_PF_ZERO 0xffffffffu

/* The device side: the map kernel, the prep kernel (P4's, exl3r4d_launch_prep) and the MTX GEMM, on `stream`.
 * mtx 4 or 8, npw 1 or 2. arot 1 (Q3's arm): no prep kernel -- the GEMM rotates its entry's rows into LDS a 128-wide
 * K block at a time, and g.xr is not read. RAD_OK or a negative RAD_E_*; must not allocate, must not synchronise. */
int exl3r4d_launch_prefill(const Exl3r4dPrefill* a, int gate_up, int mtx, int npw, int arot, void* stream);

/* Entries a map can need at most for T rows over n_expert experts at fm rows an entry: one partial entry an
 * expert, T / fm whole ones, and the two zero ranges' partials. The scratch hook and the launch use this one
 * function, so they cannot disagree about the map's size. */
static inline int64_t exl3r4d_pf_map_cap(int64_t T, int64_t n_expert, int64_t fm) {
    return n_expert + T / fm + 3;
}

#ifdef __cplusplus
}  /* extern "C" */
#endif

/* The host side (C, against the ABI): the prefill rows' hooks (prefill.c), registered in rows.c's M > 16 section,
 * and the decode band's operand checks they reuse (launch.c). Not seen by the device code. */
#ifndef __HIP__
#include "exl3r4d.h"
int     exl3r4d_pf_launch(const RadArgs* a, RadStream s, int gate_up);
int64_t exl3r4d_pf_scratch(const RadArgs* a, int gate_up);
/* launch.c's operand checks and argument block (the decode adapter's `fill`), exported for the prefill adapter. */
int     exl3r4d_fill(const RadArgs* a, int gate_up, Exl3Geom* g, Exl3r4dGemm* d);
#endif

#endif /* EXL3R4D_PREFILL_H */
