/* exl3r4d_dev.h -- what the C host side (launch.c) hands the device side (exl3r4d.hip): one plain
 * struct of kernel arguments and the extern "C" entry that launches it. No radiance types cross
 * this line except RadStream (an opaque pointer), so the .hip file needs nothing of rad_plugin.h.
 *
 * The struct is the kernel's argument block, passed by value: it has to stay under the engine's
 * 768-byte kernel-argument limit (docs/PLUGIN.md §5). Today it is about 240 bytes. */
#ifndef EXL3R4D_DEV_H
#define EXL3R4D_DEV_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* WGQ (P4b): gate_up's decode GEMM drains per-WGP unit lists (exl3r4d.hip); its counter blocks come from the same
 * instance pool as D18's. 0 = the grid. Set it for the C and the HIP side alike (the argument block depends on it). */
#ifndef EXL3R4D_WGQ
#define EXL3R4D_WGQ 0
#endif
/* PDN (P4b): down's decode GEMM (the h_rot path) as a persistent grid: each workgroup lists the live experts once
 * in LDS and claims (expert, column block) units from one ticket in `ctr` (D18 form; the down row's instance pool).
 * 0 = the grid. */
#ifndef EXL3R4D_PDN
#define EXL3R4D_PDN 0
#endif

/* One parity class of experts: the four plane tables (suh1 only for gate_up) and the class's
 * weight geometry. A table is the engine's pointer array when its *_s0 is 0, else a stacked base
 * with *_s0 bytes an entry (the form a tool holding plain buffers passes). */
typedef struct Exl3r4dCls {
    const void* codes;      /* the stored trellis record (layout.c), one entry an expert */
    const void* suh;        /* f16 [cols] */
    const void* suh1;       /* f16 [cols], gate_up only (the up half's input scales) */
    const void* svh;        /* f16 [rows] */
    const void* dn_suh;     /* f16 [rows / 2], gate_up with h_rot only: DOWN's suh of the same experts (v3 EPI 1) */
    int64_t     codes_s0, suh_s0, suh1_s0, svh_s0, dn_suh_s0;
    int32_t     rows, cols; /* this class's [N, K]; gate_up rows = 2 * width (gate then up) */
} Exl3r4dCls;

/* EXL3R4D_SUHC (P5b): the decode prep (prep_gu's PREP_ROW kernel, gate_up_down's GD_PROW prologue) reads each
 * expert's suh / suh.1 from a resident copy its row instance owns, never across the link for a pooled expert. Give it
 * to BOTH the C and the HIP flags: it changes this argument block. */
#ifndef EXL3R4D_SUHC   /* OFF: its 5.2 MB an instance (~504 MB a rank, allocated at init, before load) costs the expert
                        * pool ~490 resident units -- miss 0.347 -> 0.402 %, engine C=1 10.89 -> 11.00 ms/step */
#define EXL3R4D_SUHC 0
#endif
typedef struct Exl3r4dGemm {
    Exl3r4dCls     cls[2];
    const void*    x;        /* gate_up: a [n_tok][x_pitch] f16|bf16 (token order);
                              * down: h [>= T][x_pitch] f16 (sorted order) */
    void*          y;        /* gate_up: h f16 [T][y_pitch]; down: y bf16 [T][y_pitch] */
    const int32_t* sorted;   /* [>= T] flat (token, slot) index of sorted row i, -1 dropped */
    const int32_t* off;      /* [n_expert + 1] expert e's rows are [off[e], off[e+1]) */
    void*          xr;       /* the prep kernel's rotated rows, f16 [T][xr_pitch] (scratch); null
                              * when the GEMM rotates its own rows (fused prep) */
    int64_t        x_pitch, y_pitch, xr_pitch;   /* elements */
    int32_t        T;        /* sorted rows = the output's rows */
    int32_t        n_tok;    /* rows of x (gate_up: tokens; down: >= T) */
    int32_t        n_expert, top_k, n_class;
    int32_t        y_cols;   /* the output's width: the wider class's */
    int32_t        x_bf16;   /* gate_up's a is bf16 (else f16) */
    int32_t        sk;       /* LDS split-K: k-slices a column group of waves */
    int32_t        chunk;    /* fused prep: rows a pass (what fits in LDS, <= 16); else 16 */
    int32_t        nslot;    /* grid.y: workgroups a column block (live experts + spares) */
    int32_t        h_rot;    /* v3 EPI 1: gate_up writes h rotated for down (dn_suh); down reads it so */
    int32_t        a_rot;    /* v4: gate_up's `a` is moe_exl3_prep_gu's xr (sorted rows, already rotated) */
    unsigned*      ctr;      /* D18 single launch (gate_up): this launch's counter block -- [0] tickets, [1] done,
                              * [2 + slot] slot ready -- zero at entry, reset to zero by the last workgroup; null:
                              * the prep kernel and the GEMM as two dispatches */
    int32_t        tpw;      /* split epilogue: 16-column tiles a GEMM workgroup owns (of one matrix) */
    float*         raw;      /* split epilogue (lever L8): the GEMM's raw f32 sums [T][raw_pitch] (scratch), which
                              * the finisher rotates, scales and writes out; null: the GEMM's own epilogue */
    int64_t        raw_pitch;
#if EXL3R4D_SUHC  /* (two fields: the argument block grows 16 B; gate_up_down's 744 B < 768) -- with it on */
    void*          suhc;     /* SUHC: this layer's resident copy of every expert's suh, suh.1 -- f16 [n_expert][2][K] --
                              * filled by the prep as it meets each expert; null: the planes, wherever they lie */
    unsigned*      suhc_ok;  /* SUHC: [n_expert] one bit a wave's slice of the copy written (0xFF: the whole) */
#endif
#if EXL3R4D_WGQ   /* (the arm's fields: the shipped argument block is unchanged while it is off) */
    unsigned*      wq;       /* WGQ (gate_up): this launch's counter block -- [0, nl) list cursors, [nl, 2 nl) active
                              * workgroups a WGP, [2 nl] done -- zero at entry, reset by the last workgroup; null: the grid */
    const int*     wq_map;   /* WGQ: HW_ID1 key (exl3r4d.hip, wgp_key) -> dense WGP id, or -1; probed at instance init */
    int32_t        wq_nl;    /* WGQ: WGPs (lists) */
#endif
} Exl3r4dGemm;

/* The two ops' device launches. `gate_up` 1: moe_exl3_gate_up (two matrices, SiLU epilogue, f16 h);
 * 0: moe_exl3_down (one matrix, bf16 y). `npw` column tiles a wave (1, 2, 4). With a->xr set the
 * prep kernel runs first (into a->xr) and the GEMM reads it; else the GEMM rotates its rows into
 * LDS. RAD_OK, or a negative RAD_E_* naming what the device side refuses; launch errors come back
 * as RAD_E_DEVICE. Must not allocate and must not synchronise. */
int exl3r4d_launch_gemm(const Exl3r4dGemm* a, int gate_up, int bits, int npw, void* stream);

/* The prep kernel alone: every row of every live run of a->sorted / a->off rotated into a->xr (f16 [T][xr_pitch],
 * matrix m at m * cols), dropped rows zeros -- what the GEMM's A fragments read when a->xr is set. One workgroup a
 * live expert (grid a->nslot). exl3r4d_launch_gemm calls it; a second kernel of this library (the prefill band)
 * may too. RAD_OK, RAD_E_INVAL without a->xr, RAD_E_DEVICE on a launch error. */
int exl3r4d_launch_prep(const Exl3r4dGemm* a, int gate_up, void* stream);

/* WGQ: every WGP's HW_ID1 key, found by a probe dispatch (instance init only: it synchronises), written to map_dev
 * [EXL3R4D_WGQ_KEYS] as dense ids 0.. in key order (-1: no such WGP). Returns the WGP count, or <= 0. */
#define EXL3R4D_WGQ_KEYS 256
int exl3r4d_wgp_probe(int* map_dev);

/* D18's counter blocks: device memory allocated and zeroed (hipMalloc + hipMemset), and freed -- for a row
 * instance's init / fini only, never the launch path. Null when the device refuses. */
void* exl3r4d_dev_zalloc(size_t bytes);
void  exl3r4d_dev_free(void* p);

/* The bytes of LDS one GEMM launch of this shape asks for, or a negative RAD_E_* when it does not
 * fit (the launch refuses the same shape the same way). */
int64_t exl3r4d_gemm_lds(int gate_up, int npw, int sk, int fused_rows, int64_t cols_max);

/* Workgroups of the decode gate_up GEMM (K = bits, npw, split-K sk, h_rot's staged block) resident at once on the
 * current device, or 0 when unknown (EXL3R4D_SK_FIT, launch.c). Cached a device after the first call. */
int exl3r4d_gu_resident(int bits, int npw, int sk, int h_rot);

/* Rows a fused-prep pass can hold in LDS for this op at K = cols_max (0: none), at most 16. */
int exl3r4d_fused_rows(int gate_up, int64_t cols_max);

/* Dynamic LDS a launch may ask for: 64 KiB less room for the kernels' static __shared__ (a launch
 * asking for all 64 KiB aborts the queue; exl3_moe_wmma.hip:100-102, FACTS-exl3 trap 13). */
#define EXL3R4D_LDS_MAX (64 * 1024 - 256)

/* P1c (round 2): no prep kernel for an op -- each GEMM workgroup rotates its rows into its own copy of the
 * scratch rows (one copy a 128-column block of the output) -- when that op's flag is set and the sorted rows are
 * at most EXL3R4D_GSCR_MAXT; above it the prep kernel runs. Shared by the launcher (scratch size) and the kernel. */
#ifndef EXL3R4D_GSCR_GU
#define EXL3R4D_GSCR_GU 0
#endif
#ifndef EXL3R4D_GSCR_DN   /* adopted for down (round 2): tp2 M=1 -21%, M=4 -9%, M=8 -4%; M=16 +1.7%, hence the cap */
#define EXL3R4D_GSCR_DN 1
#endif
#ifndef EXL3R4D_GSCR_MAXT
#define EXL3R4D_GSCR_MAXT 80
#endif
/* E2 (round 2, playbook D.7): a down workgroup owns EXL3R4D_DNG consecutive 128-column blocks of one run -- the
 * expert search, the plane pointers and (P1c) the rotation paid once for all of them, one scratch copy a workgroup.
 * 1 = a block a workgroup. */
/* D18 (PLAN, 2026-10-11): gate_up's prep and GEMM in ONE launch, ordered by a counter block a (stream, output)
 * taken from a pool the row instance's init allocates and zeroes (never the launch path). 0 = two dispatches. */
#ifndef EXL3R4D_FUSE1
#define EXL3R4D_FUSE1 0
#endif
#define EXL3R4D_CTR_ROUND 64    /* a counter block's words rounded up to this (256 B): no two blocks share a line */
#define EXL3R4D_CTR_BLOCKS 64   /* counter blocks an instance: distinct (stream, output) pairs it serves fused */
#ifndef EXL3R4D_DNG
#define EXL3R4D_DNG 1
#endif


#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* EXL3R4D_DEV_H */
