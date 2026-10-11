// exl3r4d.hip -- the EXL3 routed-expert GEMMs (moe_exl3_gate_up, moe_exl3_down) on the gfx1201 matrix cores,
// decode band (M <= 16 tokens).
//
// LINEAGE. The device code is a port of Dylan's exl3-moe kernel, exl3_moe/kernels/gfx1201/exl3_moe_wmma.hip
// (exl3-moe main @ 8428c31, 1386 lines; from the kernel repo r9v-exl3-rdna4-kb @ ff5ede6, which is what R9V
// v0.5.0 ships), whose skeleton is tcclaviger's libr4d fork (codeberg tcclaviger/libr4d e7b9cec, from
// StillDeadcode/libr4d): the transposed formulation wmma(weight, activation), split-K through LDS in wave order,
// and the gfx1201 16-bit fragment law idx = lane % 16, k = 8 (e >> 2) + 4 (lane >> 4) + (e & 3). Kept AS IS
// (function for function, same names): the trellis decode in registers (Plan, decode_run, decode_frag,
// codebook_pair -- mul1 with K a template parameter, no LUT), rotate128, the epilogue's 128-point output
// rotation (had16_lane, had_tiles), the LDS split-K reduction, run ownership MTX 1, the EXL3_PF = 2 prefetch
// with unconditional clamped loads, SGPR base + 32-bit lane offsets, nontemporal global loads, the fused prep
// (rotating a workgroup's rows into LDS) and the prep kernel (rotating every row into memory).
//
// WHAT CHANGED, AND WHY (each forced by radiance's op contract, include/exl3_abi.h v2, docs/OPS-exl3.md):
//   * ROUTING. Radiance hands sorted_tok + expert_offset: expert e's rows are the CONTIGUOUS run
//     [off[e], off[e+1]) of the sorted order, unpadded (the in-tree scatter's contract). Dylan's kernel read
//     vLLM's block-16 padded sort. Here a workgroup owns ONE EXPERT'S RUN: blockIdx.y names the y-th expert
//     with a non-empty run (find_run, a scan of expert_offset in LDS), and it walks its run 16 rows (or the
//     fused prep's `chunk`) at a time. In the decode band a run is at most M <= 16 rows (top-k picks distinct
//     experts per token), so that is one pass and every expert's weight is read once a column block -- the
//     incidence count is exactly the number of live experts. The workgroups past the live count zero the rows
//     no expert owns (dropped slots, rows past off[n_expert]: libr4d's rule, exl3_abi.h).
//   * WEIGHTS ARE TABLES, one per plane per parity class (PLAN.md D15): codes (the tile-stream record, this
//     library's stored form -- layout.c), suh, suh.1 (gate_up), svh. Expert e is entry e / n_class of class
//     e % n_class. This replaces Dylan's base + slot delta (his row table) with a pointer per expert per plane.
//   * TWO PARITY CLASSES IN ONE GRID. The grid spans the wider class's columns; a workgroup of the narrower
//     class past its own width writes zeros there and leaves.
//   * EPI 1 STOPS AT h = f16(silu(gate) * up). Dylan's EPI 1 went on to rotate h by the DOWN projection's suh
//     (one launch less), but gate_up does not receive down's planes and its output is the plain activation by
//     schema; so the down GEMM rotates its own input (fused prep, its rows always fit LDS at K = width).
//   * EPI 2 does not multiply the routing weight and does not sum top-k (no counters): the in-tree moe_gather /
//     ar_gather_hc_write do both. No counters, no atomics, no cross-workgroup split-K (skg): nothing persists
//     between launches (PLAN.md D9; rad_kernel_concurrent may answer 1, STEP0 S0.9).
//   * Epilogue roundings follow the reference (kernels/exl3ref/ref.c) exactly where the order is ours to choose:
//     (had * 1/sqrt(128)) * svh, silu(g) = g * (1 / (1 + expf(-g))) -- Dylan's used __expf and __frcp_rn.
//
// NOT PORTED (out of scope or forbidden here, each named): MTX 4 / 8 run ownership and NPW for prefill (P5 adds
// them on this same stored layout), skg, the fused top-k sum, dual pools, down pieces (radiance's width split is
// whole 128-blocks per rank, D7), EXL3_LOAD64, the EXL3_PERMLANE swap, EPI 0, `seg`. EXL3_MUL24 is here as lever L7.
//
// SHIPPED PATH (lever L3, measured): the prep kernel rotates every row once into scratch and the GEMM reads its A
// fragments from there, at every decode M; the fused prep (rows rotated into LDS by each workgroup) is the
// switched-off arm (launch.c EXL3R4D_FUSED_MAX).
//
// READINGS, NUMBERS, REJECTED ARMS: see the lab notes at the end of this file (the radiance convention -- read
// them before proposing anything about this kernel).
#include <hip/hip_runtime.h>
#include <stdint.h>

#include "exl3r4d_common.hip.h"   // the device code a second kernel of this library shares

#ifndef EXL3_PF_GU  // gate_up's (round 2, L1b on buffer loads: PF 4 is -5.2% at tp2 M=1, -1.9% at M=4; down loses)
#define EXL3_PF_GU 4
#endif
#ifndef EXL3_PF
#define EXL3_PF 2  // k steps of weights / activations in flight per wave (2 = double buffering); down's
#endif
#ifndef EXL3R4D_NT
#define EXL3R4D_NT 1  // 1: the weight words are nontemporal loads (as ported); 0: plain loads (lever L2, lab notes)
#endif
#ifndef EXL3R4D_ABLATE_WLOAD   // 1: delete the weight loads (the words are register values that vary by step)
#define EXL3R4D_ABLATE_WLOAD 0
#endif
#ifndef EXL3R4D_ABLATE_ALOAD   // 1: delete the activation fragment loads (a register value that varies by step)
#define EXL3R4D_ABLATE_ALOAD 0
#endif
// Deletion-ablation builds (readings only, never shipped; tests/gpu/build_variant.sh): each removes ONE component and
// produces wrong numbers on purpose, so its time can be read against the full kernel's.
#ifndef EXL3R4D_ABLATE_NOPREP  // 1: delete the fused prep (no x / suh loads, no rotation, LDS rows left as they are)
#define EXL3R4D_ABLATE_NOPREP 0
#endif
// Staged exit (readings only): the workgroup leaves after stage S, writing nothing -- 1 the expert scan, 2 the plane
// pointers, 3 the prep, 4 the K loop and split-K reduction. A fake use keeps each stage's work from being deleted.
#ifndef EXL3R4D_EXIT_STAGE
#define EXL3R4D_EXIT_STAGE 0
#endif


// ----------------------------------------------------------------------------- the LDS-staged K loop (lever L10)
//
// libr4d's decode structure (r4d_moe.hip:1374-1612): a workgroup stages a K slab of every operand into LDS with wide
// coalesced loads, the NEXT slab's loads are issued into registers right after this slab is stored (so a slab's
// global latency hides behind the previous slab's compute), and the waves compute from LDS. Here a slab is FKS
// k-steps of every k-split group: raw trellis tiles (16-byte loads; each (group, k-step, matrix) of the 128-column
// block is a contiguous 1 KB) and the A fragments' rows from the prep kernel's scratch. Each wave keeps exactly its
// old k-step sequence (ks0 .. ks0 + kps), so the sums -- and the output bytes -- are the per-wave register loop's.
// EXL3R4D_STAGED 1 switches the decode GEMM to it (prep-kernel path, NPW 2, SK 2).
#ifndef EXL3R4D_STAGED
#define EXL3R4D_STAGED 0
#endif
#ifndef EXL3R4D_EW
#define EXL3R4D_EW 0
#endif
#ifndef EXL3R4D_BUF
#define EXL3R4D_BUF 1  // adopted (V1b): buffer loads in the gather loop; 0 = the global form
#endif
#ifndef EXL3R4D_SVH_EARLY
#define EXL3R4D_SVH_EARLY 0
#endif
#ifndef EXL3R4D_TSTAMP
#define EXL3R4D_TSTAMP 0
#endif
#ifndef EXL3R4D_ROTFIRST  // P1i: the in-GEMM rotation before the K loop's setup (register pressure)
#define EXL3R4D_ROTFIRST 0
#endif
#ifndef EXL3R4D_SGB  // pin the gather loop's per-k-step schedule with sched_group_barrier (VMEM, VALU, WMMA)
#define EXL3R4D_SGB 0
#endif
#ifndef EXL3R4D_TOKL  // P1h: the in-GEMM rotation takes each row's flat index from the pass's own tok registers
#define EXL3R4D_TOKL 0
#endif
#ifndef EXL3R4D_ROT2  // the rotation's batch size in rotate_rows (prep kernel and P1c); 0 = the earlier forms
#define EXL3R4D_ROT2 0
#endif
#ifndef EXL3R4D_GSCR_ABL
#define EXL3R4D_GSCR_ABL 0
#endif
#ifndef EXL3R4D_PREPB  // the prep kernel's tasks a batch, loads first (P1d); 0 = task by task
#define EXL3R4D_PREPB 0
#endif
#ifndef EXL3R4D_APAD  // halfs added to the fused prep's LDS row pitch (P1b); 8 moves row r to bank 4 r
#define EXL3R4D_APAD 0
#endif
#ifndef EXL3R4D_ROTB  // the fused prep's rotation tasks a batch, loads first (P1b); 0 = task by task
#define EXL3R4D_ROTB 0
#endif
#ifndef EXL3R4D_ABLATE_PREPK
#define EXL3R4D_ABLATE_PREPK 0
#endif
#ifndef EXL3R4D_STAGED_FKS
#define EXL3R4D_STAGED_FKS 4
#endif
constexpr int kStagedFks = EXL3R4D_STAGED_FKS;  // k-steps a k-split group stages a slab

template <int MATS>
static __host__ __device__ constexpr int staged_lds_bytes() {  // weights [2][FKS][MATS][8][128 B] + A rows
  return 2 * kStagedFks * MATS * 8 * 128 + 2 * MATS * 16 * (kStagedFks * 16 + 8) * 2;
}

template <int BITS, int NPW, int MATS, bool ROT>
static __device__ __forceinline__ void kloop_staged(const Exl3r4dGemm& p, const Planes& w, int rc, int r1, int mat,
                                                    int tl, int ksplit, int lane, float* lds, v8f (&acc)[NPW]) {
  static_assert(BITS == 4 && NPW == 2, "the staged slab is sized for K=4 tiles and two tiles a wave");
  constexpr int SKC = 2, FKS = kStagedFks, TB = 8 * BITS * 4, WV = 8 * MATS / NPW, NT = WV * SKC * WAVE32;
  constexpr int WUNITS = SKC * FKS * MATS * 8 * TB / 16, AUNITS = SKC * MATS * 16 * (FKS * 16 * 2 / 16);
  constexpr int WPT = WUNITS / NT, APT = AUNITS / NT, APITCH = FKS * 16 + 8;
  static_assert(WUNITS % NT == 0 && AUNITS % NT == 0 && WPT >= 1 && APT >= 1, "whole units a thread");
  using P = Plan<BITS>;
  constexpr int NW = P::NW;
  const int tid = threadIdx.x, h = lane >> 4, nl = lane & 15;
  const int width = MATS == 2 ? w.rows / 2 : w.rows, ntiles = width / 16, ksteps = w.cols / 16;
  const int kps = ksteps / SKC, nit = (kps + FKS - 1) / FKS, cb = blockIdx.x;
  const int nrows = min(EXL3_BLOCK, r1 - rc);
  unsigned char* wl = (unsigned char*)lds;
  _Float16* al = (_Float16*)((unsigned char*)lds + SKC * FKS * MATS * 8 * TB);
  // Global address space, not generic: a flat load also counts on the LDS counter, so the s_wait_dscnt in front of
  // each barrier would drain the prefetch right after it is issued (seen in the disassembly of the first build).
  typedef unsigned v4w __attribute__((ext_vector_type(4)));
  typedef const __attribute__((address_space(1))) v4w* gu4;
  const gu8 codes = (gu8)(const unsigned char*)w.codes;
  const gu8 xr = (gu8)(const unsigned char*)p.xr;

  // ROT (lever L11): the A slab is rotated here from x instead of read from the prep kernel's rows. With FKS 8 a
  // split's slab is exactly one 128-feature block, so a (row, matrix, split) task is one rotate128 by one wave: its
  // x and suh loads ride with the weight slab's, its math runs where the slab is stored. Up to TPW tasks a wave.
  constexpr int NWV = WV * SKC, TPW = (SKC * MATS * 16 + NWV - 1) / NWV;
  typedef const __attribute__((address_space(1))) v2u* gv2;
  [[maybe_unused]] gu8 rx_base[TPW], rs_base[TPW];
  [[maybe_unused]] int rrow[TPW], rblk0[TPW];
  [[maybe_unused]] bool rlive[TPW], rok[TPW];
  [[maybe_unused]] v2u rxv[TPW], rsv[TPW];
  if constexpr (ROT) {
    static_assert(FKS == 8, "a fused-rotation slab is one 128-feature block a split");
    const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE32), ntask = SKC * MATS * nrows;
#pragma unroll
    for (int q = 0; q < TPW; q++) {
      const int j0 = wave + q * NWV, j = j0 < ntask ? j0 : 0;  // a dead slot loads task 0's bytes, never stores
      const int i = j % nrows, sm = j / nrows, sp = sm / MATS, m = sm % MATS;
      const int t = p.sorted[rc + i];
      const int xrow = MATS == 2 ? min(t / p.top_k, p.n_tok - 1) : rc + i;
      rok[q] = j0 < ntask;
      rlive[q] = t >= 0;
      rrow[q] = sm * 16 + i;
      rblk0[q] = sp * (kps / 8);
      rx_base[q] = (gu8)(const unsigned char*)p.x + ((size_t)(t >= 0 ? xrow : 0) * p.x_pitch + 4 * lane) * 2;
      rs_base[q] = (gu8)(const unsigned char*)(m ? w.suh1 : w.suh) + (size_t)(4 * lane) * 2;
    }
    // rows past the chunk's are never written by a task: zero once, so their sums are zeros, not stale LDS
    for (int u = tid; u < SKC * MATS * 16 * APITCH / 8; u += NT) *(v4w*)(al + u * 8) = (v4w)(0u);
  }

  // A slab's k-steps past the split's last are clamped to it (loaded, never multiplied): kps need not divide by FKS.
  // The prefetch is three NAMED registers, not arrays: an array of them indexed in a loop the compiler did not
  // unroll went to scratch (48 B private segment, scratch_store_b128 a slab).
  static_assert(WPT == FKS / 2 && APT == FKS / 4 && (FKS == 4 || FKS == 8), "FKS 4: 2 + 1 units a thread, 8: 4 + 2");
  v4w wr0, wr1, wr2, wr3, ar0, ar1;
  auto wload = [&](int it, int u) __attribute__((always_inline)) -> v4w {
    const int c = u / 64, within = u % 64;
    const int s = c / (FKS * MATS), kk = (c / MATS) % FKS, m = c % MATS;
    const int tk = s * kps + min(it * FKS + kk, kps - 1);
    return *(gu4)(codes + ((size_t)(m * ksteps + tk) * ntiles + cb * 8) * TB + within * 16);
  };
  constexpr int APR = FKS * 16 * 2 / 16;  // 16-B units an A row of a slab
  auto aload = [&](int it, int v) __attribute__((always_inline)) -> v4w {
    const int part = v % APR, i = (v / APR) % 16, m = (v / (APR * 16)) % MATS, s = v / (APR * 16 * MATS);
    const int arow = rc + min(i, nrows - 1), tk = s * kps + min(it * FKS + part / 2, kps - 1);
    return *(gu4)(xr + ((size_t)arow * p.xr_pitch + (size_t)m * w.cols + tk * 16 + (part % 2) * 8) * 2);
  };
  auto issue = [&](int it) __attribute__((always_inline)) {
    wr0 = wload(it, tid);
    wr1 = wload(it, tid + NT);
    if constexpr (FKS == 8) {
      wr2 = wload(it, tid + 2 * NT);
      wr3 = wload(it, tid + 3 * NT);
    }
    if constexpr (ROT) {
#pragma unroll
      for (int q = 0; q < TPW; q++) {
        const size_t bo = (size_t)(rblk0[q] + it) * 256;
        rxv[q] = *(gv2)(rx_base[q] + bo);
        rsv[q] = *(gv2)(rs_base[q] + bo);
      }
    } else {
      ar0 = aload(it, tid);
      if constexpr (FKS == 8) ar1 = aload(it, tid + NT);
    }
  };
  const P plan(lane);
  unsigned wvo[2][NW];
#pragma unroll
  for (int r = 0; r < 2; r++)
#pragma unroll
    for (int x = 0; x < NW; x++) wvo[r][x] = 4u * (unsigned)plan.woff[r][x];

  issue(0);
  for (int it = 0; it < nit; ++it) {
    lds_barrier();  // every wave is done reading the previous slab
    // libr4d's collection: ONE explicit wait for the whole prefetch where the stores begin (r4d_moe.hip:1552-1558).
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "global");
    *(v4w*)(wl + tid * 16) = wr0;
    *(v4w*)(wl + (tid + NT) * 16) = wr1;
    if constexpr (FKS == 8) {
      *(v4w*)(wl + (tid + 2 * NT) * 16) = wr2;
      *(v4w*)(wl + (tid + 3 * NT) * 16) = wr3;
    }
    if constexpr (ROT) {
#pragma unroll
      for (int q = 0; q < TPW; q++)
        if (rok[q]) {  // wave-uniform
          const v2u o = rlive[q] ? rotate128_math(rxv[q], rsv[q], p.x_bf16 == 0, lane) : (v2u)(0u);
          *(v2u*)(al + rrow[q] * APITCH + 4 * lane) = o;
        }
    } else {
      *(v4w*)(al + (tid / APR) * APITCH + (tid % APR) * 8) = ar0;  // row (s * MATS + m) * 16 + i, part v % APR
      if constexpr (FKS == 8) *(v4w*)(al + ((tid + NT) / APR) * APITCH + ((tid + NT) % APR) * 8) = ar1;
    }
    if (it + 1 < nit) issue(it + 1);  // the next slab, in flight across this slab's compute (libr4d's form)
    lds_barrier();
    const int kn = min(FKS, kps - it * FKS);
#pragma unroll
    for (int kk = 0; kk < FKS; kk++) {
      if (kk >= kn) break;
      const v8h af = frag_lds(al + ((ksplit * MATS + mat) * 16 + nl) * APITCH + kk * 16 + 4 * h);
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        const unsigned char* tb = wl + ((((ksplit * FKS + kk) * MATS + mat) * 8 + tl + j) * TB);
        unsigned wd[2][NW];
#pragma unroll
        for (int r = 0; r < 2; r++)
#pragma unroll
          for (int x = 0; x < NW; x++) wd[r][x] = *(const unsigned*)(tb + wvo[r][x]);
        acc[j] = mma(decode_frag<BITS, NW>(wd, plan.shift), af, acc[j]);
      }
    }
  }
  lds_barrier();  // the reduction and the epilogue reuse the LDS
}

// ----------------------------------------------------------------------------- the streamed K loop (playbook S1a)
//
// EXL3R4D_STREAM 1: a wave streams ITS OWN NPW tiles a k-step as whole 128-B tiles -- 16-B loads, a uniform base and a
// 32-bit lane offset -- a pass of EXL3R4D_STREAM_P k-steps one pass ahead in registers, collected by ONE explicit wait
// a pass (libr4d's fence spelling, r4d_moe.hip:1552-1558), stored into a WAVE-PRIVATE LDS slot (no workgroup barrier:
// one wave writes and reads it, and LDS is in order within a wave) and decoded from there. The pass's A fragments are
// loaded with it. The words, the decode and the WMMA order are the gather loop's: same sums, same bits. Why: the
// gather loop's waitcnt drains every iteration (s_wait_loadcnt 0 at the top, R1), so its prefetch is one k-step deep
// whatever EXL3_PF says -- ~4-8 KB in flight a WGP against the ~21-32 KB the card's latency needs (playbook B.1).
#ifndef EXL3R4D_STREAM
#define EXL3R4D_STREAM 0
#endif
#ifndef EXL3R4D_STREAM_P
#define EXL3R4D_STREAM_P 8
#endif
constexpr int kStreamP = EXL3R4D_STREAM_P;

static __host__ __device__ constexpr int stream_slot_bytes(int npw) { return npw * kStreamP * 128; }

template <int BITS, int NPW>
static __device__ __forceinline__ void kloop_stream(gu8 wt0, unsigned kbytes, gu8 abase, unsigned avo, int kps,
                                                    int lane, unsigned char* slot, const unsigned (&wvo)[2][Plan<BITS>::NW],
                                                    const unsigned (&shift)[2], v8f (&acc)[NPW]) {
  constexpr int P = kStreamP, NW = Plan<BITS>::NW, KPQ = 4 / NPW, NQ = NPW * P / 4;
  static_assert(BITS == 4 && (NPW == 1 || NPW == 2 || NPW == 4) && (NPW * P) % 4 == 0 && P % KPQ == 0,
                "a 16-B load a lane covers 4 / NPW k-steps of the wave's NPW contiguous tiles");
  typedef const __attribute__((address_space(1))) v4u* gv4;
  // Load q of pass ps: lanes 0..8 NPW - 1 read k-step ps P + q KPQ, the next 8 NPW lanes the step after, ... each
  // lane 16 B of the wave's NPW * 128 contiguous bytes of that step; it lands at slot byte q * 512 + lane * 16, i.e.
  // slot[k-step in pass][tile j][128 B]. A k-step past the split's last re-reads the last one (min on the offset,
  // which is monotone in the k-step for a fixed lane): loaded, never multiplied.
  const unsigned lo = (unsigned)(lane % (NPW * 8)) * 16u;
  const unsigned loff = (unsigned)(lane / (NPW * 8)) * kbytes + lo, llast = (unsigned)(kps - 1) * kbytes + lo;
  const int npass = (kps + P - 1) / P;
  v4u wr[NQ];
  v8h an[P];
  auto issue = [&](int ps) __attribute__((always_inline)) {
#pragma unroll
    for (int q = 0; q < NQ; q++) {
      const unsigned u = (unsigned)(ps * P + q * KPQ) * kbytes;  // uniform
      wr[q] = *(gv4)(wt0 + min(u + loff, llast));
    }
#pragma unroll
    for (int kk = 0; kk < P; kk++) an[kk] = frag_a8(abase + (size_t)min(ps * P + kk, kps - 1) * 32, avo);
  };
  issue(0);
  for (int ps = 0; ps < npass; ++ps) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "global");  // the ONE wait of the pass
#pragma unroll
    for (int q = 0; q < NQ; q++) *(v4u*)(slot + q * 512 + lane * 16) = wr[q];
    v8h ac[P];
#pragma unroll
    for (int kk = 0; kk < P; kk++) ac[kk] = an[kk];
    if (ps + 1 < npass) issue(ps + 1);
    const int kn = min(P, kps - ps * P);
#pragma unroll
    for (int kk = 0; kk < P; kk++) {
      if (kk >= kn) break;
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        const unsigned char* tb = slot + (kk * NPW + j) * 128;
        unsigned wd[2][NW];
#pragma unroll
        for (int r = 0; r < 2; r++)
#pragma unroll
          for (int x = 0; x < NW; x++) wd[r][x] = *(const unsigned*)(tb + wvo[r][x]);
        acc[j] = mma(decode_frag<BITS, NW>(wd, shift), ac[kk], acc[j]);
      }
    }
  }
}

// ----------------------------------------------------------------------------- batched row rotation (P1c / P1d)
//
// A wave's rotation tasks (row i, matrix m, 128-block k of rows [r0, r0 + nrows)) NB at a time, written to
// dst[(r0 + i) * pitch + m * cols + k * 128 + 4 lane]. Every load of a batch is issued before any of its math and
// none is under a branch -- the task indices made scalar first (a division the compiler does in VALU loses the
// uniformity, and a load whose only use sits under the store's branch is sunk into it: both seen in the first form,
// which then waited out a row-index load and an x load per task, each behind the GEMM's weight stream). The math
// runs for every slot of the batch; only the store is conditional. Bit-identical to prep_rotate.
#ifndef EXL3R4D_ROTNI  // P1j: rotate_rows compiled as its own function (not inlined into the GEMM)
#define EXL3R4D_ROTNI 0
#endif
#if EXL3R4D_ROTNI
#define EXL3R4D_ROT_INLINE __attribute__((noinline))
#else
#define EXL3R4D_ROT_INLINE __forceinline__
#endif
template <int MATS, int NB>
static __device__ EXL3R4D_ROT_INLINE void rotate_rows(const Exl3r4dGemm& p, const Planes& w, int r0, int nrows, int wave,
                                                   int nwaves, int lane, _Float16* dst, size_t pitch,
                                                   const int* tokl = nullptr) {
  // tokl (P1h): lane i < 16 holds sorted[r0 + i], loaded by the caller earlier -- the row's flat index is read from
  // it (readlane) instead of loaded here, so no x address waits on a load issued in this function.
  typedef const __attribute__((address_space(1))) v2u* gv2;
  const int blocks = w.cols / 128, per_row = MATS * blocks, tasks = nrows * per_row;
  const gu8 xb = (gu8)(const unsigned char*)p.x;
  const gu8 s0 = (gu8)(const unsigned char*)w.suh;
  const gu8 s1 = (gu8)(const unsigned char*)(MATS == 2 ? w.suh1 : w.suh);
  for (int t0 = wave; t0 < tasks; t0 += NB * nwaves) {
    int bi[NB], bm[NB], bk[NB], bt[NB];
    v2u bx[NB], bs[NB];
#pragma unroll
    for (int b = 0; b < NB; b++) {
      const int t1 = t0 + b * nwaves, tsk = t1 < tasks ? t1 : t0;
      const int i = tsk / per_row, rem = tsk - i * per_row;
      bi[b] = __builtin_amdgcn_readfirstlane(i);
      bm[b] = __builtin_amdgcn_readfirstlane(rem / blocks);
      bk[b] = __builtin_amdgcn_readfirstlane(rem - (rem / blocks) * blocks);
    }
#pragma unroll
    for (int b = 0; b < NB; b++)
      bt[b] = tokl ? __builtin_amdgcn_readlane(*tokl, bi[b]) : __builtin_amdgcn_readfirstlane(p.sorted[r0 + bi[b]]);
#pragma unroll
    for (int b = 0; b < NB; b++) {
      const int xrow = MATS == 2 ? min(bt[b] / p.top_k, p.n_tok - 1) : r0 + bi[b];
      // EXL3R4D_GSCR_ABL 6 / 7 (reading, timing only): no x loads / no suh loads (a lane-made value instead)
      bx[b] = EXL3R4D_GSCR_ABL == 6 ? (v2u){(unsigned)lane * 0x3C013C01u, (unsigned)b}
            : EXL3R4D_GSCR_ABL == 15 ? *(gv2)(xb + (size_t)(bk[b] * 128 + 4 * lane) * 2)   // ABL 15: x row 0 always
            : EXL3R4D_GSCR_ABL == 16 ? *(gv2)((gu8)(const unsigned char*)dst + (size_t)(bk[b] * 128 + 4 * lane) * 2)  // 16: own region
                                    : *(gv2)(xb + ((size_t)(bt[b] >= 0 ? xrow : 0) * p.x_pitch + bk[b] * 128 + 4 * lane) * 2);
      // ABL 8: suh's load from x's row instead (two x loads a task); ABL 9: x's load from suh instead (two suh loads)
      if constexpr (EXL3R4D_GSCR_ABL == 9)
        bx[b] = *(gv2)((bm[b] ? s0 : s1) + (size_t)(bk[b] * 128 + 4 * lane) * 2);
      if constexpr (EXL3R4D_GSCR_ABL == 13)  // x's load from the codes plane instead (two weight allocations, no x)
        bx[b] = *(gv2)((gu8)(const unsigned char*)w.codes + (size_t)(bk[b] * 128 + 4 * lane) * 2);
      bs[b] = EXL3R4D_GSCR_ABL == 12 ? *(gv2)(s0 + (size_t)(bk[b] * 128 + 4 * lane) * 2)  // ABL 12: suh table only
            : EXL3R4D_GSCR_ABL == 7 ? (v2u)(0x3C003C00u)
            : EXL3R4D_GSCR_ABL == 8 ? *(gv2)(xb + ((size_t)(bt[b] >= 0 ? xrow : 0) * p.x_pitch + ((bk[b] + 1) % blocks) * 128 + 4 * lane) * 2)
                                    : *(gv2)((bm[b] ? s1 : s0) + (size_t)(bk[b] * 128 + 4 * lane) * 2);
    }
    v2u o[NB];
#pragma unroll
    for (int b = 0; b < NB; b++) {
      o[b] = rotate128_math(bx[b], bs[b], p.x_bf16 == 0, lane);
      if (bt[b] < 0) o[b] = (v2u)(0u);
    }
#pragma unroll
    for (int b = 0; b < NB; b++)
      if (t0 + b * nwaves < tasks)  // wave-uniform; only the store
        *(v2u*)(dst + (size_t)(r0 + bi[b]) * pitch + bm[b] * w.cols + bk[b] * 128 + 4 * lane) = o[b];
  }
}

// ----------------------------------------------------------------------------- body

// One pass over a chunk of an expert's run: rows [rc, rc + chunk) of [rc, r1), one 16-row WMMA block (MTX 1).
// MATS 2: gate/up (EPI 1), MATS 1: down (EPI 2). NPW column tiles a wave; WV = 8 MATS / NPW waves across the
// workgroup's 128 output columns (x MATS matrices), p.sk k-splits; wave = col_local * SK + ksplit.
// RAW (lever L8, the split epilogue): the workgroup owns p.tpw tiles of the class's [MATS][ntiles] tile space instead
// of a 128-column block of every matrix, and stores its raw f32 sums for the finisher (exl3r4d_epi_kernel).
template <int BITS, int NPW, int MATS, bool APREP, bool RAW = false, int STAGED = 0>
static __device__ __forceinline__ void moe_pass(const Exl3r4dGemm& p, const Planes& w, int rc, int r1, float* lds,
                                                unsigned long long tk0 = 0, int cbx = -1, bool rot = true) {
  // EXL3R4D_TSTAMP (reading, timing only): workgroup (0, 0)'s thread 0 writes the realtime counter (100 MHz) at the
  // kernel's entry and at this pass's phase boundaries into the output's first 48 bytes, after its own epilogue.
  [[maybe_unused]] unsigned long long tsA = EXL3R4D_TSTAMP ? __builtin_readsteadycounter() : 0, tsB = 0, tsC = 0, tsD = 0;
  using P = Plan<BITS>;
  constexpr int NW = P::NW, WORDS = P::WORDS;
  const int SK = p.sk;
  const int tid = threadIdx.x, lane = tid % WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE32);  // keep tile bases in SGPRs
  const int col_local = wave / SK, ksplit = wave % SK;
  const int h = lane >> 4, nl = lane & 15;
  const int width = MATS == 2 ? w.rows / 2 : w.rows;  // this class's output columns
  const int ntiles = width / 16, ksteps = w.cols / 16;
  const int kps = ksteps / SK, ks0 = ksplit * kps;
  const int cb = cbx >= 0 ? cbx : (int)blockIdx.x;  // E2: the workgroup's g-th block
  const int vl = RAW ? (int)blockIdx.x * p.tpw + col_local * NPW : col_local * NPW;
  // non-RAW: local tile 0 .. 8 MATS - 1 of this 128-column block; RAW: a tile of the [MATS][ntiles] space
  const int mat = RAW ? vl / ntiles : vl / 8, nt0 = RAW ? vl % ntiles : cb * 8 + vl % 8;
  const int chunk = APREP ? p.chunk : EXL3_BLOCK;

  // This lane's row of the 16-row block: in the run or not, and live (sorted_tok >= 0) or dropped. The load is
  // unconditional (clamped to the chunk's first row, which is in the run).
  const int row = rc + nl;
  const bool in_run = nl < chunk && row < r1;
  const int tok = p.sorted[in_run ? row : rc];
  const bool live = in_run && tok >= 0;

  // P1i (EXL3R4D_ROTFIRST): the in-GEMM rotation (P1c, STAGED 4) FIRST, before any K-loop state exists -- with it
  // below the K loop's setup (tile bases, descriptors), both sets of scalars are live at once and the compiler
  // spills ~29 SGPRs into VGPR lanes (gate_up: ~20 us at tp2 M=1, readings R7-R16).
  if constexpr (STAGED == 4 && EXL3R4D_ROTFIRST) {
    if (rot) {
      const int nwaves = (int)blockDim.x / WAVE32;
      _Float16* xo = (_Float16*)p.xr + (size_t)blockIdx.x * p.T * p.xr_pitch;
      rotate_rows<MATS, (EXL3R4D_ROT2 > 0 ? EXL3R4D_ROT2 : 4)>(p, w, rc, min(EXL3_BLOCK, r1 - rc), wave, nwaves, lane, xo,
                                                              (size_t)p.xr_pitch, EXL3R4D_TOKL ? &tok : nullptr);
      __syncthreads();
    }
  }

  const P plan(lane);
  unsigned wvo[2][NW];
#pragma unroll
  for (int r = 0; r < 2; r++)
#pragma unroll
    for (int x = 0; x < NW; x++) wvo[r][x] = 4u * (unsigned)plan.woff[r][x];

  // The record: matrix mat, tile (tk, tn) at (mat * ksteps * ntiles + tk * ntiles + tn) * WORDS u32 (layout.c).
  gu8 wt[NPW];
  const unsigned* wexp = w.codes + (size_t)mat * ksteps * ntiles * WORDS;
#pragma unroll
  for (int j = 0; j < NPW; j++) wt[j] = (gu8)(const unsigned char*)(wexp + ((size_t)ks0 * ntiles + nt0 + j) * WORDS);
  const size_t kbytes = (size_t)ntiles * WORDS * 4;

  // EXL3R4D_EW (playbook P1, libr4d's early weight: "the block's loads go out in the order their waits want them"):
  // with the fused prep, the first EXL3_PF - 1 k-steps' weight words are issued BEFORE the rows are rotated, and the
  // barrier after the rotation orders LDS only, so they stay in flight across it; the gather loop's prologue takes
  // them instead of loading.
  constexpr int PF = MATS == 2 ? EXL3_PF_GU : EXL3_PF;  // per op (L1b)
  [[maybe_unused]] unsigned ew[PF > 1 ? PF - 1 : 1][NPW][2][NW];
  if constexpr (APREP && EXL3R4D_EW) {
#pragma unroll
    for (int u = 0; u < PF - 1; u++)
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        const gu8 tb = wt[j] + (size_t)(u < kps ? u : kps - 1) * kbytes;
#pragma unroll
        for (int r = 0; r < 2; r++)
#pragma unroll
          for (int x = 0; x < NW; x++) ew[u][j][r][x] = *(gu32)(tb + wvo[r][x]);
      }
  }

  // Activation addressing: uniform base + per-lane 32-bit byte offset (libr4d's rowGo note).
  gu8 abase = nullptr;
  unsigned avo = 0;
  const _Float16* alds = (const _Float16*)lds;
  unsigned aoff = 0;
  if constexpr (APREP) {
    // This pass's rows rotated from x into LDS, [row][mat][cols] fp16; the lane reads row nl (row 0 for lanes
    // past the chunk). A dropped row is zeros.
    const int nrows = min(chunk, r1 - rc);
    const int blocks = w.cols / 128, per_row = MATS * blocks, tasks = nrows * per_row;
    const int nwaves = (int)blockDim.x / WAVE32;
    const int rp = MATS * w.cols + EXL3R4D_APAD;  // the LDS row pitch in halfs (APAD: off the bank multiple)
    if constexpr (EXL3R4D_ROTB > 0) {
      // P1b: a wave's tasks EXL3R4D_ROTB at a time, every load of the batch issued before any rotation math (the
      // rows' tokens, then x and suh), so a batch waits about two latencies, not two a task.
      constexpr int NB = EXL3R4D_ROTB > 0 ? EXL3R4D_ROTB : 1;
      typedef const __attribute__((address_space(1))) v2u* gv2;
      const gu8 xb = (gu8)(const unsigned char*)p.x;
      for (int t0 = wave; t0 < tasks; t0 += NB * nwaves) {
        int bi[NB], bm[NB], bk[NB], bt[NB];
        v2u bx[NB], bs[NB];
#pragma unroll
        for (int b = 0; b < NB; b++) {
          const int t1 = t0 + b * nwaves, tsk = t1 < tasks ? t1 : t0;
          bi[b] = tsk / per_row;
          const int rem = tsk - bi[b] * per_row;
          bm[b] = rem / blocks;
          bk[b] = rem - bm[b] * blocks;
          bt[b] = p.sorted[rc + bi[b]];
        }
#pragma unroll
        for (int b = 0; b < NB; b++) {
          const int xrow = MATS == 2 ? min(bt[b] / p.top_k, p.n_tok - 1) : rc + bi[b];
          bx[b] = *(gv2)(xb + ((size_t)(bt[b] >= 0 ? xrow : 0) * p.x_pitch + bk[b] * 128 + 4 * lane) * 2);
          bs[b] = *(gv2)((gu8)(const unsigned char*)(bm[b] ? w.suh1 : w.suh) + (size_t)(bk[b] * 128 + 4 * lane) * 2);
        }
#pragma unroll
        for (int b = 0; b < NB; b++)
          if (t0 + b * nwaves < tasks) {  // wave-uniform
            const v2u o = bt[b] >= 0 ? rotate128_math(bx[b], bs[b], p.x_bf16 == 0, lane) : (v2u)(0u);
            *(v2u*)((_Float16*)lds + ((size_t)bi[b] * rp + bm[b] * w.cols + bk[b] * 128 + 4 * lane)) = o;
          }
      }
    } else {
    for (int tsk = wave; tsk < (EXL3R4D_ABLATE_NOPREP ? 0 : tasks); tsk += nwaves) {
      const int i = tsk / per_row, rem = tsk - i * per_row, m = rem / blocks, blk = rem - m * blocks;
      const v2u o = prep_rotate<MATS>(p, w, rc + i, m, blk, lane);
      *(v2u*)((_Float16*)lds + ((size_t)i * rp + m * w.cols + blk * 128 + 4 * lane)) = o;
    }
    }
    if constexpr (EXL3R4D_EW) lds_barrier();  // LDS only: the early weight words stay in flight
    else __syncthreads();
    const int lrow = nl < nrows ? nl : 0;
    aoff = (unsigned)(lrow * rp + mat * w.cols + ks0 * 16 + 4 * h);  // in halfs
  } else {
    // The prep kernel's rows: xr[row][mat * cols + k], row pitch xr_pitch.
    // P1c (EXL3R4D_GSCR, STAGED 4): no prep kernel -- this workgroup rotates its pass's rows itself into ITS OWN copy
    // of those rows (region cb of grid.x copies of [T][xr_pitch]; no other workgroup writes it), tasks in batches of
    // 4 with every load of a batch first, then one workgroup barrier, and the K loop reads them as it reads the prep
    // kernel's: buffer loads from L2, no LDS in the loop.
    // one copy a WORKGROUP (with E2 a workgroup's G blocks share it, rotated by its first block's pass)
    const size_t region = STAGED == 4 ? (size_t)blockIdx.x * p.T * p.xr_pitch : 0;
    if (STAGED == 4 && rot && !EXL3R4D_ROTFIRST) {
      const int nrows = min(EXL3_BLOCK, r1 - rc);
      const int blocks = w.cols / 128, per_row = MATS * blocks, tasks = nrows * per_row;
      const int nwaves = (int)blockDim.x / WAVE32;
      constexpr int NB = 4;
      typedef const __attribute__((address_space(1))) v2u* gv2;
      const gu8 xb = (gu8)(const unsigned char*)p.x;
      _Float16* xo = (_Float16*)p.xr + region;
      if constexpr (EXL3R4D_ROT2) {  // the batched helper (loads first, scalar indices)
        rotate_rows<MATS, EXL3R4D_ROT2>(p, w, rc, nrows, wave, nwaves, lane, xo, (size_t)p.xr_pitch,
                                        EXL3R4D_TOKL ? &tok : nullptr);
#if EXL3R4D_EXIT_STAGE == 5  // reading: the kernel up to and through the in-GEMM rotation, nothing after
        __syncthreads();
        return;
#endif
      } else
      // EXL3R4D_GSCR_ABL (reading, timing only): 1 = no rotation at all, 2 = no stores, 3 = no barrier.
      for (int t0 = wave; t0 < (EXL3R4D_GSCR_ABL == 1 ? 0 : tasks); t0 += NB * nwaves) {
        int bi[NB], bm[NB], bk[NB], bt[NB];
        v2u bx[NB], bs[NB];
#pragma unroll
        for (int b = 0; b < NB; b++) {
          const int t1 = t0 + b * nwaves, tsk = t1 < tasks ? t1 : t0;
          bi[b] = tsk / per_row;
          const int rem = tsk - bi[b] * per_row;
          bm[b] = rem / blocks;
          bk[b] = rem - bm[b] * blocks;
          bt[b] = p.sorted[rc + bi[b]];
        }
#pragma unroll
        for (int b = 0; b < NB; b++) {
          const int xrow = MATS == 2 ? min(bt[b] / p.top_k, p.n_tok - 1) : rc + bi[b];
          bx[b] = *(gv2)(xb + ((size_t)(bt[b] >= 0 ? xrow : 0) * p.x_pitch + bk[b] * 128 + 4 * lane) * 2);
          bs[b] = *(gv2)((gu8)(const unsigned char*)(bm[b] ? w.suh1 : w.suh) + (size_t)(bk[b] * 128 + 4 * lane) * 2);
        }
#pragma unroll
        for (int b = 0; b < NB; b++)
          if (t0 + b * nwaves < tasks) {  // wave-uniform
            // ABL 4: loads, no math (the loaded words xor'd); ABL 5: math on lane-made inputs, no x / suh loads
            const v2u o = EXL3R4D_GSCR_ABL == 4 ? (bx[b] ^ bs[b])
                        : EXL3R4D_GSCR_ABL == 5 ? rotate128_math((v2u){(unsigned)lane * 0x3C013C01u, (unsigned)b},
                                                                 (v2u)(0x3C003C00u), p.x_bf16 == 0, lane)
                        : bt[b] >= 0 ? rotate128_math(bx[b], bs[b], p.x_bf16 == 0, lane) : (v2u)(0u);
            if (EXL3R4D_GSCR_ABL != 2)
              *(v2u*)(xo + (size_t)(rc + bi[b]) * p.xr_pitch + bm[b] * w.cols + bk[b] * 128 + 4 * lane) = o;
            else if (o[0] == 0x12345678u) *(int*)p.y = 1;
          }
      }
      if constexpr (EXL3R4D_GSCR_ABL != 3) __syncthreads();  // workgroup release/acquire on global: rows visible
    }
    const int arow = in_run ? row : rc;
    avo = (unsigned)(((size_t)arow * p.xr_pitch + 4 * h) * 2);
    abase = (gu8)(const unsigned char*)((const _Float16*)p.xr + region + (size_t)mat * w.cols + (size_t)ks0 * 16);
  }

#if EXL3R4D_EXIT_STAGE == 3
  if (APREP && alds[aoff] == (_Float16)12345.f) *(int*)p.y = 1;
  return;
#endif
  if constexpr (EXL3R4D_TSTAMP) tsB = __builtin_readsteadycounter();
  if constexpr (STAGED == 1 || STAGED == 2) {  // lever L10 (refuted), kept with its number
    static_assert(!APREP && !RAW, "the staged K loop reads the prep kernel's rows and ends in the epilogue");
    v8f sacc[NPW];
#pragma unroll
    for (int j = 0; j < NPW; j++) sacc[j] = (v8f)(0.f);
    kloop_staged<BITS, NPW, MATS, STAGED == 2>(p, w, rc, r1, mat, vl % 8, ksplit, lane, lds, sacc);
    if (SK > 1) splitk_reduce<NPW>(sacc, lds, col_local, ksplit, SK, lane);
    epilogue_block<NPW, MATS>(p, w, sacc, col_local, ksplit, lane, cb, row, in_run, live, lds);
    return;
  }

  v8f acc[NPW];
#pragma unroll
  for (int j = 0; j < NPW; j++) acc[j] = (v8f)(0.f);
  [[maybe_unused]] v8h svp[epi_opw<NPW, MATS>()][MATS];
  if constexpr (EXL3R4D_SVH_EARLY && !RAW) svh_early<NPW, MATS>(w, col_local, lane, cb, svp);

  if constexpr (STAGED == 3) {  // playbook S1a: the streamed K loop, the wave's LDS slot past the reduction's region
    const int sbase = max(MATS * 8192, (8 * MATS / NPW) * NPW * (SK - 1) * 1024);
    kloop_stream<BITS, NPW>(wt[0], (unsigned)kbytes, abase, avo, kps, lane,
                            (unsigned char*)lds + sbase + wave * stream_slot_bytes(NPW), wvo, plan.shift, acc);
  } else {
  // EXL3_PF stages of weight words and activation fragments in flight: step ks + PF - 1 is loaded while step ks
  // is decoded and multiplied (2 = double buffering).
  unsigned wb[PF][NPW][2][NW];
  v8h ab[PF];

  // EXL3R4D_BUF (playbook V1b): the weight words and the A fragments as BUFFER loads -- a uniform resource, a
  // per-lane offset fixed for the whole loop and the k-step's offset in an SGPR -- so a k-step's addresses cost no
  // VALU (the global form spends ~5 a tile on 64-bit v_add_co pairs, R1). At K = 4 a lane's two runs touch three
  // words: woff[1][x] = woff[0][x] + 1 (run 1 ends 32 bits later), so W[0] = {w-1, w0}, W[1] = {w0, w+1}.
  // Descriptor word 3: gfx11/12 raw buffer (0x31004000: OOB_SELECT raw, no stride, no swizzle).
  [[maybe_unused]] const __amdgpu_buffer_rsrc_t wrs =
      __builtin_amdgcn_make_buffer_rsrc((void*)wexp, (short)0, 0x7fffffff, 0x31004000);
  [[maybe_unused]] const __amdgpu_buffer_rsrc_t ars =
      __builtin_amdgcn_make_buffer_rsrc((void*)abase, (short)0, 0x7fffffff, 0x31004000);
  [[maybe_unused]] const unsigned bso = (unsigned)(((size_t)ks0 * ntiles + nt0) * WORDS * 4);
  [[maybe_unused]] const unsigned bwm = wvo[0][0], bw0 = wvo[0][1], bwp = wvo[1][NW - 1];
  static_assert(!EXL3R4D_BUF || (BITS == 4 && NW == 2), "the three-word form is K = 4's");
  constexpr int kAux = EXL3R4D_NT ? 1 : 0;  // TH_LOAD_NT, as the global form's __builtin_nontemporal_load

#define WOFF_(r, x) wvo[r][x]
#define EXL3_LOAD(ks, W_, A_)                                                        \
  do {                                                                               \
    if constexpr (EXL3R4D_BUF) {                                                     \
      const unsigned so_ = bso + (unsigned)(ks) * (unsigned)kbytes;                  \
      _Pragma("unroll") for (int j = 0; j < NPW; j++) {                              \
        const unsigned m_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwm + j * 128, so_, kAux); \
        const unsigned z_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bw0 + j * 128, so_, kAux); \
        const unsigned q_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwp + j * 128, so_, kAux); \
        W_[j][0][0] = m_; W_[j][0][1] = z_; W_[j][1][0] = z_; W_[j][1][1] = q_;     \
      }                                                                              \
      if constexpr (APREP) A_ = frag_lds(alds + aoff + (ks) * 16);                   \
      else {                                                                         \
        union { v2u w[2]; v8h f; } x_;                                               \
        x_.w[0] = __builtin_amdgcn_raw_buffer_load_b64(ars, avo, (unsigned)(ks) * 32u, 0);      \
        x_.w[1] = __builtin_amdgcn_raw_buffer_load_b64(ars, avo + 16u, (unsigned)(ks) * 32u, 0); \
        A_ = x_.f;                                                                   \
      }                                                                              \
      break;                                                                         \
    }                                                                                \
    _Pragma("unroll") for (int j = 0; j < NPW; j++) {                                \
      const gu8 tb_ = wt[j] + (size_t)(ks) * kbytes;                                 \
      _Pragma("unroll") for (int r = 0; r < 2; r++)                                  \
        _Pragma("unroll") for (int x = 0; x < NW; x++)                               \
          W_[j][r][x] = EXL3R4D_ABLATE_WLOAD ? (unsigned)((ks) * 0x9E3779B1u + wvo[r][x] + j) \
                      : EXL3R4D_NT ? __builtin_nontemporal_load((gu32)(tb_ + WOFF_(r, x))) \
                                   : *(gu32)(tb_ + WOFF_(r, x));                       \
    }                                                                                \
    if constexpr (EXL3R4D_ABLATE_ALOAD) {                                            \
      union { unsigned u[4]; v8h f; } ab_;                                           \
      for (int q_ = 0; q_ < 4; q_++) ab_.u[q_] = (unsigned)((ks) * 0x3C003C00u + avo + q_); \
      A_ = ab_.f;                                                                    \
    } else if constexpr (APREP) A_ = frag_lds(alds + aoff + (ks) * 16);              \
    else A_ = frag_a8(abase + (size_t)(ks) * 32, avo);                              \
  } while (0)

#define EXL3_COMPUTE(W_, A_)                                                         \
  do {                                                                               \
    _Pragma("unroll") for (int j = 0; j < NPW; j++) {                                \
      const v8h wf = decode_frag<BITS, NW>(W_[j], plan.shift);                       \
      acc[j] = mma(wf, A_, acc[j]);                                                  \
    }                                                                                \
    if constexpr (EXL3R4D_SGB) { /* pin: the step's loads, then its decode, then its WMMAs (libr4d's OPT 2048) */ \
      __builtin_amdgcn_sched_group_barrier(0x020, 3 * NPW + 2, 0);                  \
      __builtin_amdgcn_sched_group_barrier(0x002, 48 * NPW, 0);                     \
      __builtin_amdgcn_sched_group_barrier(0x008, NPW, 0);                          \
    }                                                                                \
  } while (0)

  // Loads are unconditional (the step index clamps to the last one): with a branch around the prefetch, the
  // compiler cannot count the loads in flight on the join and waits for ALL of them before the next WMMA, which
  // serialized the prefetch with the math every other step (exl3_moe_wmma.hip:714-716; FACTS-exl3 trap 11).
#pragma unroll
  for (int u = 0; u < PF - 1; u++) {
    if constexpr (APREP && EXL3R4D_EW) {
#pragma unroll
      for (int j = 0; j < NPW; j++)
#pragma unroll
        for (int r = 0; r < 2; r++)
#pragma unroll
          for (int x = 0; x < NW; x++) wb[u][j][r][x] = ew[u][j][r][x];
      ab[u] = frag_lds(alds + aoff + (u < kps ? u : kps - 1) * 16);
    } else {
      EXL3_LOAD(u < kps ? u : kps - 1, wb[u], ab[u]);
    }
  }
  int ks = 0;
  for (; ks + PF <= kps; ks += PF) {
#pragma unroll
    for (int u = 0; u < PF; u++) {
      const int ls = ks + u + PF - 1 < kps ? ks + u + PF - 1 : kps - 1;
      EXL3_LOAD(ls, wb[(u + PF - 1) % PF], ab[(u + PF - 1) % PF]);
      EXL3_COMPUTE(wb[u], ab[u]);
    }
  }
#pragma unroll
  for (int u = 0; u < PF - 1; u++)  // the last kps % PF steps, already loaded into stages 0..
    if (ks + u < kps) EXL3_COMPUTE(wb[u], ab[u]);
#undef EXL3_LOAD
#undef EXL3_COMPUTE
#undef WOFF_
  }  // STAGED != 3
  if constexpr (APREP) __syncthreads();  // the rotated rows' LDS is reused by the reductions and the epilogue

#if EXL3R4D_EXIT_STAGE == 4
  if (SK > 1) lds_barrier();
  if (acc[0][0] == 12345.678f) *(int*)p.y = 1;
  return;
#endif
  if constexpr (EXL3R4D_TSTAMP) tsC = __builtin_readsteadycounter();
  if (SK > 1) splitk_reduce<NPW>(acc, lds, col_local, ksplit, SK, lane);
  if constexpr (EXL3R4D_TSTAMP) tsD = __builtin_readsteadycounter();
  if constexpr (RAW) {  // the raw sums, row-major [row][mat * width + n]; the finisher does everything else
    if (!ksplit && in_run) raw_store<NPW>(p, acc, row, mat * width + nt0 * 16 + 8 * h);
    if (SK > 1) lds_barrier();  // the next pass's reduction reuses the LDS
    return;
  }
  epilogue_block<NPW, MATS, EXL3R4D_SVH_EARLY != 0>(p, w, acc, col_local, ksplit, lane, cb, row, in_run, live, lds,
                                                     svp);
  if constexpr (EXL3R4D_TSTAMP) {
    const unsigned long long tsE = __builtin_readsteadycounter();
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
      unsigned long long* o = (unsigned long long*)p.y;
      o[0] = tk0; o[1] = tsA; o[2] = tsB; o[3] = tsC; o[4] = tsD; o[5] = tsE;
    }
  }
}

// One workgroup = (128-column block cb = blockIdx.x, the blockIdx.y-th live expert). Its expert's run, a pass a
// chunk; a narrower class's workgroup past its width zero-fills; a spare zeroes rows no expert owns.
// EXL3R4D_WPE (round 2 occupancy arm, L6 reopened after V1/PF): amdgpu_waves_per_eu on the GEMM kernel -- 12 caps
// the VGPRs at 128, three 16-wave gate_up workgroups a WGP instead of two. 0 = no attribute.
#ifndef EXL3R4D_WPE
#define EXL3R4D_WPE 0
#endif
#ifndef EXL3R4D_WPE_MAX  // the occupancy hint's MAXIMUM (P1k): amdgpu_waves_per_eu(1, N) -- the compiler may schedule
#define EXL3R4D_WPE_MAX 0  // for ILP within N waves a SIMD instead of for register pressure
#endif
#if EXL3R4D_WPE > 0
#define EXL3R4D_WPE_ATTR __attribute__((amdgpu_waves_per_eu(EXL3R4D_WPE)))
#elif EXL3R4D_WPE_MAX > 0
#define EXL3R4D_WPE_ATTR __attribute__((amdgpu_waves_per_eu(1, EXL3R4D_WPE_MAX)))
#else
#define EXL3R4D_WPE_ATTR
#endif
template <int BITS, int NPW, int MATS, bool APREP, bool RAW = false, int STAGED = 0>
__global__ void __launch_bounds__(1024) EXL3R4D_WPE_ATTR exl3r4d_gemm_kernel(Exl3r4dGemm p) {
  extern __shared__ float lds[];
  __shared__ int s_find[33];
  [[maybe_unused]] const unsigned long long tk0 = EXL3R4D_TSTAMP ? __builtin_readsteadycounter() : 0;
#if EXL3R4D_EXIT_STAGE == 99
  if (p.T == -1) *(int*)p.y = 1;   // stage "0": the dispatch alone, this grid and block
  return;
#endif
  const Run run = find_run(p, blockIdx.y, s_find);
#if EXL3R4D_EXIT_STAGE == 1
  if (run.e == 123456) *(int*)p.y = run.r0;
  return;
#endif
  if (run.e < 0) {
    if constexpr (MATS == 1 && !RAW && !APREP && (STAGED == 0 || STAGED == 4) && EXL3R4D_DNG > 1) {
      const int ncb = (p.y_cols + 127) / 128;
      for (int g = 0; g < EXL3R4D_DNG; ++g)
        if ((int)blockIdx.x * EXL3R4D_DNG + g < ncb)
          zero_dead_rows(p, (int)blockIdx.y - run.nlive, (int)gridDim.y - run.nlive, blockIdx.x * EXL3R4D_DNG + g);
    } else if (!RAW) {
      zero_dead_rows(p, (int)blockIdx.y - run.nlive, (int)gridDim.y - run.nlive, blockIdx.x);
    }
    return;  // RAW: the finisher writes every output row
  }
  const Planes w = planes_of(p, run.e);
#if EXL3R4D_EXIT_STAGE == 2
  if (w.codes == nullptr && w.suh == nullptr && w.svh == nullptr && w.suh1 == nullptr) *(int*)p.y = 1;
  return;
#endif
  const int width = MATS == 2 ? w.rows / 2 : w.rows;
  constexpr int G = (MATS == 1 && !RAW && !APREP && (STAGED == 0 || STAGED == 4)) ? EXL3R4D_DNG : 1;
  if constexpr (G > 1) {  // E2: blocks blockIdx.x * G .. + G - 1 of this run, one after the other
    const int ncb = (p.y_cols + 127) / 128;
    for (int g = 0; g < G; ++g) {
      const int cb = (int)blockIdx.x * G + g;
      if (cb >= ncb) break;
      if (cb * 128 >= width) { zero_rows(p, run.r0, run.r1, cb); continue; }
      for (int rc = run.r0; rc < run.r1; rc += EXL3_BLOCK)
        moe_pass<BITS, NPW, MATS, APREP, RAW, STAGED>(p, w, rc, run.r1, lds, tk0, cb, g == 0);
    }
    return;
  }
  if (RAW ? (int)blockIdx.x * p.tpw >= MATS * (width / 16) : (int)blockIdx.x * 128 >= width) {
    if (!RAW) zero_rows(p, run.r0, run.r1, blockIdx.x);  // the narrower parity class: zeros past its own width
    return;
  }
  const int chunk = APREP ? p.chunk : EXL3_BLOCK;
  for (int rc = run.r0; rc < run.r1; rc += chunk)
    moe_pass<BITS, NPW, MATS, APREP, RAW, STAGED>(p, w, rc, run.r1, lds, tk0);
}

// The prep kernel: every row of every live run rotated into xr (f16 [T][xr_pitch], matrix m at m * cols), one
// wave per (row, matrix, 128-block); dropped rows are zeros. One workgroup per live expert (find_run's numbering).
// EXL3R4D_PREPGRID (lever L9): 1 = a 2-D grid, y over live experts and x over groups of a row's (matrix, 128-block)
// tasks, two waves a workgroup, every row of the run a wave's loop -- all of a step's tasks in one round; 0 = one
// 256-thread workgroup a live expert striding over every task (as ported: ~5 serial rounds at gate_up M=1).
#ifndef EXL3R4D_PREPGRID
#define EXL3R4D_PREPGRID 0
#endif
template <int MATS>
__global__ void __launch_bounds__(256) exl3r4d_prep_kernel(Exl3r4dGemm p) {
  __shared__ int s_find[33];
  const Run run = find_run(p, EXL3R4D_PREPGRID ? blockIdx.y : blockIdx.x, s_find);
  if (run.e < 0) return;
  const Planes w = planes_of(p, run.e);
  const int lane = threadIdx.x % WAVE32, nwaves = (int)blockDim.x / WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane((int)(threadIdx.x / WAVE32));
  const int blocks = w.cols / 128, per_row = MATS * blocks, tasks = (run.r1 - run.r0) * per_row;
#if EXL3R4D_PREPGRID
  // task (row i, group g = blockIdx.x * nwaves + wave): one (mat, block) a wave, every row of the run
  const int g = (int)blockIdx.x * nwaves + wave;
  if (g >= per_row) return;
  for (int tsk = g; tsk < tasks; tsk += per_row) {
#else
  if constexpr (EXL3R4D_ROT2 > 0) {  // the batched helper (loads first, scalar indices)
    rotate_rows<MATS, EXL3R4D_ROT2>(p, w, run.r0, run.r1 - run.r0, wave, nwaves, lane, (_Float16*)p.xr,
                                    (size_t)p.xr_pitch);
    return;
  }
  if constexpr (EXL3R4D_PREPB > 0) {
    // P1d (EXL3R4D_PREPB): the wave's tasks EXL3R4D_PREPB at a time, every load of a batch (the rows' tokens, then x
    // and suh) issued before any rotation math -- about two latencies a batch instead of two a task.
    constexpr int NB = EXL3R4D_PREPB > 0 ? EXL3R4D_PREPB : 1;
    typedef const __attribute__((address_space(1))) v2u* gv2;
    const gu8 xb = (gu8)(const unsigned char*)p.x;
    for (int t0 = wave; t0 < tasks; t0 += NB * nwaves) {
      int bi[NB], bm[NB], bk[NB], bt[NB];
      v2u bx[NB], bs[NB];
#pragma unroll
      for (int b = 0; b < NB; b++) {
        const int t1 = t0 + b * nwaves, tsk = t1 < tasks ? t1 : t0;
        bi[b] = tsk / per_row;
        const int rem = tsk - bi[b] * per_row;
        bm[b] = rem / blocks;
        bk[b] = rem - bm[b] * blocks;
        bt[b] = p.sorted[run.r0 + bi[b]];
      }
#pragma unroll
      for (int b = 0; b < NB; b++) {
        const int xrow = MATS == 2 ? min(bt[b] / p.top_k, p.n_tok - 1) : run.r0 + bi[b];
        bx[b] = *(gv2)(xb + ((size_t)(bt[b] >= 0 ? xrow : 0) * p.x_pitch + bk[b] * 128 + 4 * lane) * 2);
        bs[b] = *(gv2)((gu8)(const unsigned char*)(bm[b] ? w.suh1 : w.suh) + (size_t)(bk[b] * 128 + 4 * lane) * 2);
      }
#pragma unroll
      for (int b = 0; b < NB; b++)
        if (t0 + b * nwaves < tasks) {  // wave-uniform
          const v2u o = bt[b] >= 0 ? rotate128_math(bx[b], bs[b], p.x_bf16 == 0, lane) : (v2u)(0u);
          *(v2u*)((_Float16*)p.xr + (size_t)(run.r0 + bi[b]) * p.xr_pitch + (size_t)bm[b] * w.cols + bk[b] * 128 +
                  4 * lane) = o;
        }
    }
    return;
  }
  for (int tsk = wave; tsk < tasks; tsk += nwaves) {
#endif
    const int i = tsk / per_row, rem = tsk - i * per_row, m = rem / blocks, blk = rem - m * blocks;
    const int r = run.r0 + i;
    const v2u o = prep_rotate<MATS>(p, w, r, m, blk, lane);
    *(v2u*)((_Float16*)p.xr + (size_t)r * p.xr_pitch + (size_t)m * w.cols + blk * 128 + 4 * lane) = o;
  }
}


// P1f (EXL3R4D_PREPE): the prep kernel as one workgroup an EXPERT -- the expert's offsets and its table pointers
// are loaded together (both depend only on the expert index), an expert with no rows leaves at once, and the rows
// are rotated by rotate_rows (every load of a batch first). About two dependent round trips where the live-slot form
// takes five (scan, two barriers, offsets again, pointers, then x / suh). Same rows, same rotate128: bit-identical.
#ifndef EXL3R4D_PREPE
#define EXL3R4D_PREPE 0
#endif
template <int MATS>
__global__ void __launch_bounds__(256) exl3r4d_prep_e_kernel(Exl3r4dGemm p) {
  const int e = blockIdx.x;
  const int a = clamp_row(p.off[e], p.T), b = clamp_row(p.off[e + 1], p.T);
  const Planes w = planes_of(p, e);  // independent of the offsets: issued with them
  if (b <= a) return;
  const int lane = threadIdx.x % WAVE32, nwaves = (int)blockDim.x / WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane((int)(threadIdx.x / WAVE32));
  rotate_rows<MATS, 4>(p, w, a, b - a, wave, nwaves, lane, (_Float16*)p.xr, (size_t)p.xr_pitch);
}

// ============================================================================= D18: prep + GEMM in one launch
//
// THE SINGLE LAUNCH (EXL3R4D_FUSE1, gate_up, the decode default path). Every workgroup takes a TICKET from the
// launch's counter block (Exl3r4dGemm::ctr). Tickets 0 .. nslot - 1 do the prep kernel's work for live slot `ticket`
// -- in prep_role, NOT inlined and called first, so nothing of the GEMM is live across it and the GEMM path's code
// (and its K loop's schedule) is the two-dispatch path's: an inlined rotation re-schedules gate_up's K loop into a
// dependency-bound order (R18: 448 instructions / 36 s_delay_alu against 405 / 2) -- then publish ctr[2 + slot] = 1.
// Tickets from nslot on are GEMM workgroups (column block, slot), numbered as the two-dispatch grid numbers them;
// one waits on its slot's flag before its K loop reads the rotated rows.
//
// FORWARD PROGRESS, without assuming any dispatch order: a workgroup takes its ticket only once it is running, and
// a prep ticket's holder waits on nothing (its slot's rows and planes are inputs), so it finishes. A waiter holds a
// ticket >= nslot, so every prep ticket was taken BEFORE its own -- by a workgroup that is already running and
// will finish -- and every flag it can wait on is set in finite time. (A waiter cannot starve a prep workgroup of a
// slot on the device: that prep workgroup already holds one.)
//
// THE RESET: each workgroup adds one to ctr[1] as its very last act; the one that takes it to gridDim.x zeroes the
// tickets, the count and the flags for the next launch on the same block. Only launches that share a block need be
// ordered (D18: allocated at the instance's init, one block a (stream, output) -- libr4d's split_ctr argument:
// overlapping launches on one stream write disjoint outputs). The output is the two-dispatch path's to the bit.
#ifndef EXL3R4D_PREP_NOINLINE  // D18's prep role compiled as its own function (1) or inlined in its branch (0)
#define EXL3R4D_PREP_NOINLINE 0
#endif
#if EXL3R4D_PREP_NOINLINE
#define EXL3R4D_PREP_ATTR __attribute__((noinline))
#else
#define EXL3R4D_PREP_ATTR __forceinline__
#endif
template <int MATS>
static __device__ EXL3R4D_PREP_ATTR void prep_role(const Exl3r4dGemm& p, int slot, int* s_find) {
  // BY REFERENCE: taking the kernel argument's address (&p) makes every field a load through a pointer the compiler
  // cannot prove uniform, and the GEMM's buffer loads then waterfall (24 readfirstlane loops in the K loop)
  const Run run = find_run(p, slot, s_find);
  if (run.e < 0) return;
  const Planes w = planes_of(p, run.e);
  const int lane = threadIdx.x % WAVE32, nwaves = (int)blockDim.x / WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane((int)(threadIdx.x / WAVE32));
  const int blocks = w.cols / 128, per_row = MATS * blocks, tasks = (run.r1 - run.r0) * per_row;
  for (int tsk = wave; tsk < tasks; tsk += nwaves) {
    const int i = tsk / per_row, rem = tsk - i * per_row, m = rem / blocks, blk = rem - m * blocks;
    const int r = run.r0 + i;
    const v2u o = prep_rotate<MATS>(p, w, r, m, blk, lane);
    *(v2u*)((_Float16*)p.xr + (size_t)r * p.xr_pitch + (size_t)m * w.cols + blk * 128 + 4 * lane) = o;
  }
}

// The run and the planes made provably uniform (readfirstlane): in the single-launch kernel the compiler otherwise
// keeps the planes' pointers in VGPRs and every buffer load of the K loop becomes a readfirstlane waterfall.
static __device__ __forceinline__ const void* uni_ptr(const void* q) {
  const unsigned long long v = (unsigned long long)q;
  const unsigned lo = (unsigned)__builtin_amdgcn_readfirstlane((int)(unsigned)v);
  const unsigned hi = (unsigned)__builtin_amdgcn_readfirstlane((int)(unsigned)(v >> 32));
  return (const void*)(((unsigned long long)hi << 32) | lo);
}
static __device__ __forceinline__ Planes uni_planes(Planes w) {
  w.codes = (const unsigned*)uni_ptr(w.codes);
  w.suh = (const _Float16*)uni_ptr(w.suh);
  w.suh1 = (const _Float16*)uni_ptr(w.suh1);
  w.svh = (const _Float16*)uni_ptr(w.svh);
  w.dn_suh = (const _Float16*)uni_ptr(w.dn_suh);
  w.rows = __builtin_amdgcn_readfirstlane(w.rows);
  w.cols = __builtin_amdgcn_readfirstlane(w.cols);
  return w;
}
static __device__ __forceinline__ Run uni_run(Run r) {
  r.e = __builtin_amdgcn_readfirstlane(r.e);
  r.r0 = __builtin_amdgcn_readfirstlane(r.r0);
  r.r1 = __builtin_amdgcn_readfirstlane(r.r1);
  r.nlive = __builtin_amdgcn_readfirstlane(r.nlive);
  return r;
}

template <int BITS, int NPW, int MATS>
__global__ void __launch_bounds__(1024) exl3r4d_gemm_fused_kernel(Exl3r4dGemm p) {
  extern __shared__ float lds[];
  __shared__ int s_find[33];
  __shared__ unsigned s_ticket, s_done;
  if (threadIdx.x == 0) s_ticket = __hip_atomic_fetch_add(&p.ctr[0], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  __syncthreads();
  // readfirstlane: the ticket (and the column block and slot from it) must be KNOWN uniform, or the K loop's tile
  // bases fall into VGPRs and its schedule doubles (830 instructions / 57 s_delay_alu against 405 / 2)
  const unsigned ticket = (unsigned)__builtin_amdgcn_readfirstlane((int)s_ticket), nslot = (unsigned)p.nslot;
  if (ticket < nslot) {
    prep_role<MATS>(p, (int)ticket, s_find);
    __threadfence();  // every rotated row visible device-wide before the flag, from every thread
    __syncthreads();
    if (threadIdx.x == 0) __hip_atomic_store(&p.ctr[2 + ticket], 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  } else {
    const int ncb = p.y_cols / 128, g = (int)(ticket - nslot), cb = g % ncb, slot = g / ncb;
    const Run run = uni_run(find_run(p, slot, s_find));
    if (run.e < 0) {
      zero_dead_rows(p, slot - run.nlive, p.nslot - run.nlive, cb);
    } else {
      const Planes w = uni_planes(planes_of(p, run.e));
      const int width = MATS == 2 ? w.rows / 2 : w.rows;
      if (cb * 128 >= width) {
        zero_rows(p, run.r0, run.r1, cb);
      } else {
        if (threadIdx.x == 0)
          while (__hip_atomic_load(&p.ctr[2 + slot], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == 0u)
            __builtin_amdgcn_s_sleep(1);
        __syncthreads();
        __atomic_thread_fence(__ATOMIC_ACQUIRE);  // every wave: no stale line of the rotated rows
        for (int rc = run.r0; rc < run.r1; rc += EXL3_BLOCK)
          moe_pass<BITS, NPW, MATS, false, false, 0>(p, w, rc, run.r1, lds, 0, cb);
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) s_done = __hip_atomic_fetch_add(&p.ctr[1], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
  __syncthreads();
  if (s_done + 1u == gridDim.x) {  // the last workgroup: zero the block for the next launch on it
    for (unsigned u = threadIdx.x; u < 2u + nslot; u += blockDim.x)
      __hip_atomic_store(&p.ctr[u], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
}

// The finisher of the split epilogue (lever L8): workgroup = (128-column block, live expert), one wave a row: the raw
// sums' output rotation in the reference's own butterfly order (exl3_had128: stages 1, 2, .., 64 in f32, then x
// 1/sqrt(128)), x svh, then h = f16(silu(gate) * up) or y = bf16; dropped rows, rows no run owns and a narrower
// class's columns are written as zeros. Four consecutive columns a lane: bits 0, 1 in the lane, 2..5 through DPP
// row_xmask, 6 through v_permlanex16 -- rotate128's transform, on f32 sums instead of f16(x * suh).
template <int MATS>
__global__ void __launch_bounds__(256) exl3r4d_epi_kernel(Exl3r4dGemm p) {
  __shared__ int s_find[33];
  const Run run = find_run(p, blockIdx.y, s_find);
  if (run.e < 0) {
    zero_dead_rows(p, (int)blockIdx.y - run.nlive, (int)gridDim.y - run.nlive, blockIdx.x);
    return;
  }
  const Planes w = planes_of(p, run.e);
  const int width = MATS == 2 ? w.rows / 2 : w.rows;
  if ((int)blockIdx.x * 128 >= width) {
    zero_rows(p, run.r0, run.r1, blockIdx.x);
    return;
  }
  constexpr float RS = 0.08838834764831845f;
  const int lane = threadIdx.x % WAVE32, nw = (int)blockDim.x / WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane((int)(threadIdx.x / WAVE32));
  const int c0 = (int)blockIdx.x * 128 + 4 * lane;
  for (int r = run.r0 + wave; r < run.r1; r += nw) {
    const int t = p.sorted[r];
    const float* src = p.raw + (size_t)r * p.raw_pitch + c0;
    float g[4], sv[4];
    const v4f gv = *(const v4f*)src;
    g[0] = gv[0]; g[1] = gv[1]; g[2] = gv[2]; g[3] = gv[3];
    had128_f32(g, lane);
    { const v2u s2 = *(const v2u*)(w.svh + c0);
      for (int q = 0; q < 4; q++) sv[q] = (float)__builtin_bit_cast(_Float16, (unsigned short)(s2[q >> 1] >> (16 * (q & 1)))); }
    if constexpr (MATS == 2) {
      float u[4], su[4];
      const v4f uv4 = *(const v4f*)(src + width);
      u[0] = uv4[0]; u[1] = uv4[1]; u[2] = uv4[2]; u[3] = uv4[3];
      had128_f32(u, lane);
      { const v2u s2 = *(const v2u*)(w.svh + width + c0);
        for (int q = 0; q < 4; q++) su[q] = (float)__builtin_bit_cast(_Float16, (unsigned short)(s2[q >> 1] >> (16 * (q & 1)))); }
      union { _Float16 hv[4]; v2u u; } o;
#pragma unroll
      for (int q = 0; q < 4; q++) {
        const float gq = (g[q] * RS) * sv[q], uq = (u[q] * RS) * su[q];
        o.hv[q] = t >= 0 ? (_Float16)((gq * (1.0f / (1.0f + expf(-gq)))) * uq) : (_Float16)0.f;
      }
      *(v2u*)((_Float16*)p.y + (size_t)r * p.y_pitch + c0) = o.u;
    } else {
      v2u o;
#pragma unroll
      for (int q = 0; q < 2; q++)
        o[q] = t >= 0 ? (unsigned)f32_to_bf16((g[2 * q] * RS) * sv[2 * q]) |
                            ((unsigned)f32_to_bf16((g[2 * q + 1] * RS) * sv[2 * q + 1]) << 16)
                      : 0u;
      *(v2u*)((unsigned short*)p.y + (size_t)r * p.y_pitch + c0) = o;
    }
  }
}

// ----------------------------------------------------------------------------- host

static int64_t red_bytes(int wv, int npw, int sk) { return (int64_t)wv * npw * (sk - 1) * 256 * 4; }

extern "C" int exl3r4d_fused_rows(int gate_up, int64_t cols_max) {
  const int64_t per = ((int64_t)(gate_up ? 2 : 1) * cols_max + EXL3R4D_APAD) * 2;
  const int64_t n = per > 0 ? EXL3R4D_LDS_MAX / per : 0;
  return (int)(n < EXL3_BLOCK ? n : EXL3_BLOCK);
}

extern "C" int64_t exl3r4d_gemm_lds(int gate_up, int npw, int sk, int fused_rows, int64_t cols_max) {
  const int mats = gate_up ? 2 : 1, wv = 8 * mats / npw;
  if (npw != 1 && npw != 2 && npw != 4) return RAD_E_SHAPE;
  if (sk < 1 || wv * sk * WAVE32 > 1024) return RAD_E_SHAPE;
  int64_t need = (int64_t)mats * 8 * 256 * 4;  // the epilogue's tiles
  const int64_t red = red_bytes(wv, npw, sk), rows = (int64_t)fused_rows * (mats * cols_max + EXL3R4D_APAD) * 2;
  need = red > need ? red : need;
  need = rows > need ? rows : need;
  return need > EXL3R4D_LDS_MAX ? RAD_E_SHAPE : need;
}

template <int BITS, int MATS>
static const void* gemm_fn(int npw, bool aprep) {
#define EXL3R4D_FN(NPWv) \
  (aprep ? (const void*)exl3r4d_gemm_kernel<BITS, NPWv, MATS, true> : (const void*)exl3r4d_gemm_kernel<BITS, NPWv, MATS, false>)
  switch (npw) {
    case 1: return EXL3R4D_FN(1);
    case 2: return EXL3R4D_FN(2);
    case 4: return EXL3R4D_FN(4);
    default: return nullptr;
  }
#undef EXL3R4D_FN
}

extern "C" void* exl3r4d_dev_zalloc(size_t bytes) {  // instance init only (D18), never a launch
  void* p = nullptr;
  if (hipMalloc(&p, bytes) != hipSuccess) return nullptr;
  if (hipMemset(p, 0, bytes) != hipSuccess || hipDeviceSynchronize() != hipSuccess) {
    (void)hipFree(p);
    return nullptr;
  }
  return p;
}
extern "C" void exl3r4d_dev_free(void* p) {
  if (p) (void)hipFree(p);
}

extern "C" int exl3r4d_launch_prep(const Exl3r4dGemm* a, int gate_up, void* stream) {
  if (!a->xr) return RAD_E_INVAL;
  Exl3r4dGemm p = *a;
  void* args[] = {&p};
  hipStream_t st = (hipStream_t)stream;
  const void* pf = gate_up ? (const void*)exl3r4d_prep_kernel<2> : (const void*)exl3r4d_prep_kernel<1>;
  if (EXL3R4D_PREPE) {  // P1f: a workgroup an expert
    const void* pe = gate_up ? (const void*)exl3r4d_prep_e_kernel<2> : (const void*)exl3r4d_prep_e_kernel<1>;
    if (hipLaunchKernel(pe, dim3((unsigned)a->n_expert), dim3(256), args, 0, st) != hipSuccess) return RAD_E_DEVICE;
    return RAD_OK;
  }
#if EXL3R4D_PREPGRID
  const int64_t cols_max = a->cls[0].cols > a->cls[1].cols ? a->cls[0].cols : a->cls[1].cols;
  const unsigned groups = (unsigned)(((gate_up ? 2 : 1) * (cols_max / 128) + 1) / 2);   // two waves a WG
  if (hipLaunchKernel(pf, dim3(groups, (unsigned)a->nslot), dim3(2 * WAVE32), args, 0, st) != hipSuccess)
    return RAD_E_DEVICE;
#else
  if (hipLaunchKernel(pf, dim3(a->nslot), dim3(256), args, 0, st) != hipSuccess) return RAD_E_DEVICE;
#endif
  return RAD_OK;
}

template <int BITS, int MATS>
static const void* gemm_raw_fn(int npw) {
  switch (npw) {
    case 1: return (const void*)exl3r4d_gemm_kernel<BITS, 1, MATS, false, true>;
    case 2: return (const void*)exl3r4d_gemm_kernel<BITS, 2, MATS, false, true>;
    case 4: return (const void*)exl3r4d_gemm_kernel<BITS, 4, MATS, false, true>;
    default: return nullptr;
  }
}

extern "C" int exl3r4d_launch_gemm(const Exl3r4dGemm* a, int gate_up, int bits, int npw, void* stream) {
  if (bits != 4) return RAD_E_DTYPE;  // K = 4 is the only instantiation (exl3k4); refused by name upstream too
  const int mats = gate_up ? 2 : 1, wv = 8 * mats / npw;
  const bool aprep = a->xr == nullptr;
  const int64_t cols_max = a->cls[0].cols > a->cls[1].cols ? a->cls[0].cols : a->cls[1].cols;
  int64_t smem = exl3r4d_gemm_lds(gate_up, npw, a->sk, aprep ? a->chunk : 0, cols_max);
  if (smem < 0) return (int)smem;
  if (gate_up && a->h_rot && smem < kHrotStage + 4096) smem = kHrotStage + 4096;  // v3 EPI 1's staged h block
  const bool hrot_dn = !gate_up && a->h_rot;  // down reads gate_up's rotated h: no prep, no in-kernel rotation
  const void* fn = gate_up ? gemm_fn<4, 2>(npw, aprep) : gemm_fn<4, 1>(npw, aprep);
  if (!fn) return RAD_E_SHAPE;
  Exl3r4dGemm p = *a;
  void* args[] = {&p};
  hipStream_t st = (hipStream_t)stream;
  // Lever L10/L11: the LDS-staged K loop (NPW 2, SK 2); STAGED 2 also rotates in it (no prep kernel) where every
  // class's split is whole 128-feature blocks.
  const bool staged = EXL3R4D_STAGED && !aprep && !a->raw && npw == 2 && a->sk == 2;
#if EXL3R4D_STAGED == 2
  static_assert(EXL3R4D_STAGED_FKS == 8, "STAGED 2 (rotation in the K loop) needs EXL3R4D_STAGED_FKS 8");
  const bool rot = staged && a->cls[0].cols % 256 == 0 && a->cls[1].cols % 256 == 0;
#define EXL3R4D_ROTK(M) (const void*)exl3r4d_gemm_kernel<4, 2, M, false, false, 2>
#else
  const bool rot = false;
#define EXL3R4D_ROTK(M) (const void*)nullptr
#endif
  const bool gscr = !hrot_dn && (gate_up ? EXL3R4D_GSCR_GU : EXL3R4D_GSCR_DN) && a->T <= EXL3R4D_GSCR_MAXT && !aprep && !a->raw &&
                    !staged && !EXL3R4D_STREAM;  // P1c (exl3r4d_dev.h)
  // D18: gate_up's prep and GEMM as one launch when the instance gave this launch a counter block
  const bool fuse1 = EXL3R4D_FUSE1 && gate_up && a->ctr && !aprep && !a->raw && !staged && !gscr && !rot &&
                     !EXL3R4D_STREAM;
  if (!aprep && !rot && !gscr && !hrot_dn && !fuse1 && !EXL3R4D_ABLATE_PREPK) {  // ABLATE_PREPK (timing only, wrong output): the prep kernel's cost
    const int rc = exl3r4d_launch_prep(a, gate_up, stream);
    if (rc != RAD_OK) return rc;
  }
  if (a->raw) {  // the split epilogue: GEMM workgroups of p.tpw tiles of one matrix, then the finisher
    if (aprep || a->tpw < npw || a->tpw % npw) return RAD_E_SHAPE;
    const int wvr = a->tpw / npw;
    int64_t tiles = 0;
    for (int c = 0; c < 2; ++c) {
      const int64_t nt = (gate_up ? a->cls[c].rows / 2 : a->cls[c].rows) / 16;  // tiles a matrix
      if (nt % a->tpw) return RAD_E_SHAPE;   // a workgroup never straddles gate and up
      tiles = mats * nt > tiles ? mats * nt : tiles;
    }
    if (wvr * a->sk * WAVE32 > 1024) return RAD_E_SHAPE;
    const void* rf = gate_up ? gemm_raw_fn<4, 2>(npw) : gemm_raw_fn<4, 1>(npw);
    const size_t red = (size_t)red_bytes(wvr, npw, a->sk);
    if (hipLaunchKernel(rf, dim3((unsigned)(tiles / a->tpw), (unsigned)a->nslot, 1),
                        dim3((unsigned)(wvr * a->sk * WAVE32)), args, red > 0 ? red : 0, st) != hipSuccess)
      return RAD_E_DEVICE;
    const void* ef = gate_up ? (const void*)exl3r4d_epi_kernel<2> : (const void*)exl3r4d_epi_kernel<1>;
    if (hipLaunchKernel(ef, dim3((unsigned)(a->y_cols / 128), (unsigned)a->nslot, 1), dim3(128), args, 0, st) !=
        hipSuccess)
      return RAD_E_DEVICE;
    return RAD_OK;
  }
  // E2: a down workgroup owns EXL3R4D_DNG column blocks (the default path and P1c only; exl3r4d_dev.h)
  const int dng = (!gate_up && !aprep && !a->raw && !staged && !EXL3R4D_STREAM) ? EXL3R4D_DNG : 1;
  const dim3 grid((unsigned)((a->y_cols / 128 + dng - 1) / dng), (unsigned)a->nslot, 1);
  const dim3 block((unsigned)(wv * a->sk * WAVE32));
  if (fuse1) {
    const void* ff = npw == 1 ? (const void*)exl3r4d_gemm_fused_kernel<4, 1, 2>
                   : npw == 2 ? (const void*)exl3r4d_gemm_fused_kernel<4, 2, 2>
                              : (const void*)exl3r4d_gemm_fused_kernel<4, 4, 2>;
    const unsigned wgs = (unsigned)a->nslot + (unsigned)(a->y_cols / 128) * (unsigned)a->nslot;
    if (hipLaunchKernel(ff, dim3(wgs), block, args, (size_t)smem, st) != hipSuccess) return RAD_E_DEVICE;
    return RAD_OK;
  }
  if (gscr) {  // P1c: the rotation in the GEMM workgroup, into its own scratch rows
    const void* gf = nullptr;
    switch (npw) {
      case 1: gf = gate_up ? (const void*)exl3r4d_gemm_kernel<4, 1, 2, false, false, 4>
                           : (const void*)exl3r4d_gemm_kernel<4, 1, 1, false, false, 4>; break;
      case 2: gf = gate_up ? (const void*)exl3r4d_gemm_kernel<4, 2, 2, false, false, 4>
                           : (const void*)exl3r4d_gemm_kernel<4, 2, 1, false, false, 4>; break;
      case 4: gf = gate_up ? (const void*)exl3r4d_gemm_kernel<4, 4, 2, false, false, 4>
                           : (const void*)exl3r4d_gemm_kernel<4, 4, 1, false, false, 4>; break;
      default: return RAD_E_SHAPE;
    }
    if (hipLaunchKernel(gf, grid, block, args, (size_t)smem, st) != hipSuccess) return RAD_E_DEVICE;
    return RAD_OK;
  }
  if (EXL3R4D_STREAM && !aprep) {  // playbook S1a
    const void* sf = nullptr;
    switch (npw) {
      case 1: sf = gate_up ? (const void*)exl3r4d_gemm_kernel<4, 1, 2, false, false, 3>
                           : (const void*)exl3r4d_gemm_kernel<4, 1, 1, false, false, 3>; break;
      case 2: sf = gate_up ? (const void*)exl3r4d_gemm_kernel<4, 2, 2, false, false, 3>
                           : (const void*)exl3r4d_gemm_kernel<4, 2, 1, false, false, 3>; break;
      case 4: sf = gate_up ? (const void*)exl3r4d_gemm_kernel<4, 4, 2, false, false, 3>
                           : (const void*)exl3r4d_gemm_kernel<4, 4, 1, false, false, 3>; break;
      default: return RAD_E_SHAPE;
    }
    const int64_t ssm = smem + (int64_t)wv * a->sk * stream_slot_bytes(npw);
    if (ssm > EXL3R4D_LDS_MAX) return RAD_E_SHAPE;
    if (hipLaunchKernel(sf, grid, block, args, (size_t)ssm, st) != hipSuccess) return RAD_E_DEVICE;
    return RAD_OK;
  }
  if (staged) {
    const void* sf = rot ? (gate_up ? EXL3R4D_ROTK(2) : EXL3R4D_ROTK(1))
                         : (gate_up ? (const void*)exl3r4d_gemm_kernel<4, 2, 2, false, false, 1>
                                    : (const void*)exl3r4d_gemm_kernel<4, 2, 1, false, false, 1>);
    const int64_t sl = gate_up ? staged_lds_bytes<2>() : staged_lds_bytes<1>();
    const size_t ssm = (size_t)(sl > smem ? sl : smem);
    if (ssm > EXL3R4D_LDS_MAX) return RAD_E_SHAPE;
    if (hipLaunchKernel(sf, grid, block, args, ssm, st) != hipSuccess) return RAD_E_DEVICE;
    return RAD_OK;
  }
#undef EXL3R4D_ROTK
  if (hipLaunchKernel(fn, grid, block, args, (size_t)smem, st) != hipSuccess) return RAD_E_DEVICE;
  return RAD_OK;
}

// ============================================================================= lab notes
//
// READ BEFORE PROPOSING ANYTHING ABOUT THIS KERNEL. Every number: ONE gfx1201 card of 64 CUs (32 WGPs) -- which one,
// and its sessions, are in the plan folder's notes/p4.md -- radiance-build:1.3.0 (e052a2ab), memory clock 1349 MHz
// under load (that card's memory overclock; its measured stream ceiling 679.5 GB/s, S0.11; R22's 92% line 625 GB/s),
// shader clock ~2.9-3.2 GHz in the timed windows, precheck and kernel-log watchdog clean. Harness: tools/perfmoe.cpp -- 48
// layers of routing draws over 2 expert pools of real size (an expert comes round again only after ~840 MB), 500 ms
// clock warm-up, libr4d's moe_gemm_w4a8 (w4nl64a8h, the baseline's served experts) on the same routing in the same
// process, 5 reps, ABAB. Raw logs: AI-Work/radiance-exl3-p4-decode-20261011/runs/<name>, ledger beside them.
//
// HARNESS TRAPS FOUND (both cost a day of wrong readings if missed):
//   * no warm-up: a ~15 ms timed burst after one pass runs on a ramping shader clock (LACT busy samples ~1.3 GHz);
//     exl3r4d read 237 GB/s cold vs 270 warm at gate_up M=1, libr4d 588 vs 658. Readings taken before the fix
//     (runs/perf1, ablate1, stages1, lever1-pf, lever2-nt) are SUPERSEDED.
//   * one routing per pool: only the routed experts recycle, 4 x 15.5 MB at down tp2 M=4 -- inside the 64 MB
//     last-level cache: libr4d read 701-741 GB/s, above the ceiling. `perfmoe --pools 1 --layers 1` is that case,
//     kept as the control: libr4d jumps to 670-984 GB/s; exl3r4d does NOT move (270 vs 268) -- itself a reading:
//     this kernel is not bound by DRAM at decode.
//
// READING 1 -- ratio to ceiling, hot experts (runs/perf3; as ported, fused prep at M <= 6):
//   op tp M          exl3r4d GB/s   libr4d GB/s        op tp M           exl3r4d   libr4d
//   gate_up 1 1         268 (39%)     473 (70%)          down 1 1           236       383
//   gate_up 1 4         341           594                down 1 4           315       532
//   gate_up 1 16        516 (76%)     596 (88%)          down 1 16          338       587
//   gate_up 2 1         224           428                down 2 1           182       327
//   gate_up 2 16        488           579                down 2 16          270       578
//   SHIPPED (lever L3 below), runs/lever9-prepgrid base: gate_up 1 M=1 323, M=4 462, M=16 511; down 1 M=1 302,
//   M=16 474; gate_up 2 M=1 242; down 2 M=1 207 GB/s.
// READING 2 (res.sh): gate_up NPW 2 prep path 154 VGPR / fused 166, down 91 / 102; no scratch; WGP mode.
// READING 3 (disassembly): a 2-step K iteration ~245 instructions a step a wave; ~50 VALU a tile decode incl. 8
//   quarter-rate v_mul_lo_u32; weight loads with 64-bit vector addresses (v_add_co pair per load) although the base
//   is scalar -- the activation loads beside them do get saddr+voffset, and neither readfirstlane on the plane
//   pointers (code object byte-identical) nor an opaque per-step base with a masked offset changes the weight loads
//   (L4, L5 below); s_wait_loadcnt 0 at the top of every K iteration (the in-order counter: the next compute needs
//   the youngest group), so PF 2 behaves like PF 1.
// READING 4 (deletion ablation, warm, DRAM-representative; runs/ablate2, ablate3-loads), gate_up tp1 M=1, base
//   60.8 us (fused): dispatch floor 2.6, +expert scan 1.1, +plane pointers 0.3, +fused prep 1.7, +K LOOP 51.8,
//   +split-K reduce and epilogue 3.3. Deletions: prep -6.2%, rotation math -3.6%, epilogue rotation -0.6%, decode
//   +1.3% (two rounds mask it; single-round runs: -15%), PF 1 -2.3%, PF 4 -2.1%, split-K off -1.2%, NT off +1.6%,
//   weight loads -8.7% (M=16 -14.8%), activation loads +1.8%. At M=16: decode -13.6%, weight loads -14.8%. Down M=16:
//   the fused prep is the cost (prep kernel instead: -28.8%).
// READING 5 (K-sweep, M=1, runs/ksweep1, cold-clock era, slopes still informative): exl3r4d gate_up fixed 13.2 us +
//   296 GB/s slope, libr4d 2.4 us + 506; down 19.4 + 377 vs 3.0 + 581.
// READING W (workgroup count; runs/wgsweep*): gate_up time is a STAIRCASE in live workgroups -- 32 us for <= 30
//   (one round on 32 WGPs), 49 us for 35-60, 66 for 70-85 -- for every shape tried (16- or 8-wave WGs, capped
//   registers); deleting the decode flattens the first step (35 WGs 51.5 -> 33.4 us). libr4d is smooth.
// TRACE (rocprofv3 --kernel-trace; needs libdw.so.1, absent from the build image: Ubuntu's libdw1t64 extracted
//   beside the build and put on LD_LIBRARY_PATH -- notes/p4.md), gate_up tp1
//   M=1: prep kernel 5.2 us, GEMM 42.1 us (392 GB/s alone) vs libr4d's one kernel 34.0 us; one expert (5 WGs):
//   GEMM 24.6 us -- a workgroup streams its 327 KB block at ~13-19 GB/s, latency-bound.
//
// LEVERS (each pre-registered with a kill condition; ledger seq in brackets):
//   L1 [11] deeper prefetch EXL3_PF 3/4/6 -- REFUTED (gate_up M=1 +1..+5%, down PF 4/6 +8..+13%): PF 4 issues four
//      steps at once and drains them all at the next top. Kept: EXL3_PF (default 2).
//   L2 [13] plain loads instead of nontemporal (EXL3R4D_NT 0) -- REFUTED on the governing point (gate_up tp1 M=1
//      -3.9%); post-hoc: down tp1 M=1 -12.9%, gate_up tp2 M=1 -12.8% (cold-clock session). Reopen: stack test.
//   L3 [21] the prep kernel at every decode M instead of the fused prep (EXL3R4D_FUSED_MAX 0) -- CONFIRMED, ADOPTED:
//      gate_up tp1 M=1 -15.3%, M=4 -25.8%, M=16 0; down tp1 M=1 -21.0%, M=4 -25.9%, M=16 -28.8%; tp2 M=1 -7..-10%.
//      Byte-identical rotation. The fused prep stays (FUSED_MAX 16 brings it back) for when in-loop LDS fragments
//      get as cheap as L2 ones.
//   L4 [24] readfirstlane on the plane pointers (libr4d's UW) -- REFUSED AT BUILD: code object byte-identical (the
//      pointers were already scalar). Toggle removed.
//   L5 [26] opaque per-step tile base + masked lane offset (saddr) -- REFUSED AT BUILD: the weight loads keep their
//      v_add_co pairs. Toggle removed. Reopen: a toolchain that selects saddr here, or buffer loads.
//   L6 [30] register cap for two gate_up WGs a WGP (amdgpu_waves_per_eu) -- REFUTED BY READING: the staircase is the
//      same for 8-wave WGs and capped registers (spills only). Toggle removed.
//   L7 [32] state hash as two v_mul_u32_u24 (EXL3_MUL24; needs inline asm -- the combiner folds the C and __umul24
//      forms back into v_mul_lo_u32) -- REFUTED (gate_up M=1 -2.5%, M=16 -1.6%); post-hoc tp2 M=1 -8.4%.
//   L8 [34] split epilogue: GEMM workgroups of EXL3R4D_TPW tiles of one matrix write raw f32 sums, a finisher kernel
//      rotates (in the reference's butterfly order), scales and applies SiLU / bf16 (EXL3R4D_SPLIT 1) -- correct
//      (device suite 245/0/0) but REFUTED: gate_up M=1 +1.4%, M=16 -3.9%, down M=1 +13.6%. Reopen for P5's prefill
//      band, where MTX > 1 makes the epilogue's share and the WG count different.
//   L9 [37] 2-D prep grid (EXL3R4D_PREPGRID 1) -- REFUTED: prep kernel 5.2 -> 4.0 us traced, total +2.8% (every extra
//      WG pays the expert scan).
//
//
// ROUND 2 (2026-10-11, the playbook notes/radiance-moe-playbook.md §D; governing points tp2 M=1/4/16, every arm ABAB in
// one process against the shipped arm; runs/<name>; ledger seq in brackets). Per layer (both ops, tp2) the shipped
// kernel was 54.6 / 129.8 / 351.5 us at M=1/4/16 against libr4d's 32.4 / 93.1 / 275.9.
// READING R1 [48] (tests/gpu/isa.sh, static): the shipped K loop spends ~75 cycles a 128-B tile -- 8 v_mul_lo_u32
//   (32 cycles), ~27 other VALU (6 bfe, 4 + 4 sad, 4 pk_fma, ~5 for 64-bit load addresses, 2 and, 2 alignbit), the
//   16-cycle f16 WMMA, 3 VMEM -- and s_wait_loadcnt 0 at every iteration top (one k-step of prefetch).
// READING R6 [54] (tests/gpu/oprate.hip): issue cycles vs v_add_nc_u32 on gfx1201: v_mul_lo_u32 3.3; v_mul_u32_u24,
//   v_mad_u32_u24, v_mad_u32_u16 (op_sel too), v_pk_mul_lo_u16, v_perm_b32, v_bfe, v_alignbit, v_sad_u8/_hi,
//   v_pk_fma_f16, v_dot2_f32_f16, v_permlanex16 all ~1; v_mov_b32_dpp 1.65.
// READING R3' [52] (EXL3R4D_ABLATE_PREPK: the prep dispatch skipped, timing only): the prep kernel costs, us an op,
//   gate_up / down: M=1 5.8 / 5.2, M=4 7.1 / 6.2, M=16 11.6 / 7.0.
//   L10 [43, 46] workgroup-shared LDS slab of 4 / 8 k-steps, 16-B loads one slab ahead, two barriers a slab
//      (EXL3R4D_STAGED 1; 2 = rotation in the slab) -- correct, REFUTED: +55..+114% at every point. At M <= 16 a
//      tile is used by one wave, so staging saves no traffic, and the barriers put the 16 waves in lockstep.
//   S1a [49, 51] playbook S1a, wave-private LDS slot, 16-B loads one 8-k-step pass ahead, one wait a pass, no barrier
//      (EXL3R4D_STREAM 1) -- correct, REFUTED: tp2 M=4 gate_up +3.3%, down +9.2%; M=1 +17..+19%. An LDS round trip
//      is exposed every k-step (16 s_wait_dscnt a pass), the next pass's A fragments are copied (32 VGPRs), and
//      down at tp2 has 1-2 passes. Bytes in flight were not the binding constraint.
//   V1a [53, 56] the state hash two states at a time (EXL3_HASH 2): states extracted in packed pairs by v_perm_b32, one
//      v_pk_mul_lo_u16 + two v_mad_u32_u16 (op_sel) a pair -- CONFIRMED, ADOPTED (bit-identical: 245/0/0, real
//      18/0/0): -3.5..-9.2% everywhere; tp2 M=16 gate_up -5.9%, down -5.7%; M=1 gate_up -9.2%. L7's 24-bit form
//      beside it: -1..-2.4%.
//   V1b [57, 60] buffer loads (EXL3R4D_BUF 1): the three words a tile and the A fragments from a uniform descriptor, a
//      per-lane offset fixed for the loop and the k-step's offset in an SGPR -- no 64-bit address VALU, no
//      s_wait_alu (18 an iteration -> 0); ~40 VALU + the WMMA a tile -- CONFIRMED, ADOPTED (bit-identical): over V1a,
//      tp2 M=16 gate_up -6.6%, down -4.5%; M=4 down -6.3% (gate_up 0); M=1 gate_up -10.7%. L5 reopened and closed.
//      With it PF 3 / 4 no longer drain at the top (L1's cause): queued.
//   P1 [58, 61] the fused prep at every M it fits (FUSED_MAX 16) with the first k-step's weights issued before the
//      rotation and an LDS-only barrier after it (EXL3R4D_EW 1) -- correct, REFUTED: +11..+57% (EW ~1%). The rotation
//      is task by task (a latency a task) and the LDS row pitch is a bank multiple (16-way fragment conflicts at
//      M > 1). P1b (EXL3R4D_ROTB, EXL3R4D_APAD) is the next arm.
//   After V1a + V1b, per layer tp2: 47.0 / 120.3 / 307.0 us at M=1/4/16 (sclk ~2.91 GHz; libr4d in the same session
//   from here on).
//   P1b [62, 66] the fused prep with batched rotation loads and the LDS pitch padded off the bank multiple
//      (EXL3R4D_ROTB 4, EXL3R4D_APAD 8) -- correct, REFUTED: +17..+67% (pad alone the same within 3%).
//   E1 [63, 67] svh loads in the prologue (EXL3R4D_SVH_EARLY) -- correct, REFUTED: -0.3% at down M=4 (kill -1%).
//   L1b [64, 68] EXL3_PF 3 / 4 on buffer loads (no longer drained at the top) -- gate_up only: PF 4 tp2 M=1 -5.2%,
//      M=4 -1.9%; down +9..+22% (its K is 8-12 k-steps a split). ADOPTED per op: EXL3_PF_GU 4, EXL3_PF 2 for down.
//   D8 [70, 73] NPW / split-K per op -- down NPW 2 / SK 1 CONFIRMED, ADOPTED (EXL3R4D_SK_DN 1): tp2 M=1 -4.9%, M=4
//      -8.6%, M=16 -11.2%; NUMERICS: down sums each output over all of K in one WMMA chain instead of two halves
//      through LDS -- the bits move, both pass the tolerance suites. gate_up NPW 4 +18..+62%: stays 2 / 2.
//   P1c [69, 72, 74, 77] no prep dispatch: each GEMM workgroup rotates its rows (batched loads) into its OWN copy of
//      the scratch rows, one barrier, then the K loop reads them as the prep kernel's (EXL3R4D_GSCR_DN / _GU,
//      EXL3R4D_GSCR_MAXT; exl3r4d_dev.h). Down, on SK 1: tp2 M=1 -21.3%, M=4 -9.0%, M=8 -4.0%, M=16 +1.7% --
//      ADOPTED for down up to 80 sorted rows (bit-identical). gate_up: +20..+39% -- off (reading R7 below).
//   P1d [75, 78] the prep kernel with batched loads (EXL3R4D_PREPB) -- REFUTED: +-1%; its cost is the dispatch.
//   R7 [79, 81] gate_up's in-kernel rotation, deletions: none of it = the no-prep floor; without its stores or its
//      barrier = no change. The cost is the rotation's loads and math themselves, ~16 us at M=1 for ~3 tasks a wave,
//      far above a rotate128's ~80 cycles + one latency. Under reading.
//   Defaults now, per layer tp2 (libr4d same session, sclk 3.0 GHz): M=1 41.2 vs 32.2, M=4 112.8 vs 92.8, M=8
//   177.4 vs 158.7, M=16 294.4 vs 278.2 us.
//   R8-R13 [82-101] gate_up's in-GEMM rotation, continued: its LOADS are the cost (loads-only = full cost, math-only
//      cheap); batching them for real (rotate_rows, EXL3R4D_ROT2; ISA: 8 loads back to back) changes nothing; x+x or
//      suh+suh loads are cheap, x with any suh table slow; not a region count (x + suh only is slow), not the data
//      (perfmoe --rand-data: same), and the rotation phase alone is cheap (EXIT_STAGE 5: 6.2 us) -- the ~14 us
//      lands after it. In-kernel stamps (EXL3R4D_TSTAMP, perfmoe --dump-stamps) perturb the kernel 3x; this image's
//      rocprofv3 has no thread trace. Mechanism OPEN; gate_up keeps the prep kernel. rotate_rows in the prep kernel
//      (EXL3R4D_ROT2): = base (the prep's cost is its dispatch).
//   E2 [103, 105] a down workgroup owns G column blocks (EXL3R4D_DNG) -- REFUTED: best -1.6% at M=16 (G 2), small M
//      +16..+70%.
//   EPI 1 (interface v3, PLAN D-rulings 2026-10-11): h_rot = 1 -- gate_up's epilogue stages its h block in LDS
//      (kHrotStage) and rotates each row's 128-column block with DOWN's suh (Planes::dn_suh), rotate128's exact
//      rounding; down reads that as the prep kernel's rows (no prep, no P1c). Device suite 279/0/0 (both flags),
//      real 18/0/0, P2's band-top chains: the pair at h_rot 1 == the pair at h_rot 0 to the bit (M=16). Same-process
//      pair, tp2: -1.7 / -1.3 / -1.7 us a layer at M=1/4/16 (down -1.8 / -2.0 / -3.0, gate_up +0.1 / +0.7 / +1.4).
//   OCC [106, 108] gate_up NPW 2 / split-K 1 -- CONFIRMED, ADOPTED (EXL3R4D_SK 1): tp2 M=4 -6.8%, M=8 -0.9%, M=16
//      -1.4%, M=1 +3.9%; one setting for every decode M (a row's bits must not depend on the batch). NUMERICS: as
//      down's, the sum over all of K in one chain. A VGPR cap to 128 (EXL3R4D_WPE 12) spills: +7..+68%, refuted.
//
// STOP (optimize-to-ceiling §3, R22): NOT reached. gate_up tp1 M=1 at 323 GB/s = 48% of 679.5 (libr4d 473 = 70%);
// M=16 511 = 75% (libr4d 596). No single deleted component exceeds 15%; the rest of the K loop's per-workgroup latency
// is "not one thing" by deletion, so the next reading is counter-level (rocprofv3 --pmc / ATT now runnable), and the
// candidate structure is libr4d's: whole-tile 16-byte loads staged through LDS with explicit vmcnt pipelining.
