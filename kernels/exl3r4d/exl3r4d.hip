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
#ifdef EXL3R4D_GD_DEBUG
#include <cstdio>
#endif

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
// EXL3R4D_DLY_{GU,DN,PG,GD} (reading builds only, never shipped; 0 = off): a fixed delay of that many microseconds
// injected into ONE decode op -- gate_up's GEMM, down's GEMM, the prep kernels, the fused gate_up_down -- so an engine
// pass's step time against the plain build reads how much of that op is on the step's critical path (a causal
// profile: the slope per microsecond injected, with nothing else perturbed). Statically gridded kernels delay
// workgroup 0 at entry (the dispatch ends ~D later however many rounds it takes); the fused kernel takes work
// tickets, so every one of its workgroups waits.
#ifndef EXL3R4D_DLY_GU
#define EXL3R4D_DLY_GU 0
#endif
#ifndef EXL3R4D_DLY_DN
#define EXL3R4D_DLY_DN 0
#endif
#ifndef EXL3R4D_DLY_PG
#define EXL3R4D_DLY_PG 0
#endif
#ifndef EXL3R4D_DLY_GD
#define EXL3R4D_DLY_GD 0
#endif
template <int US>
static __device__ __forceinline__ void inject_delay(bool here) {
  if constexpr (US > 0) {
    if (here) {
      const unsigned long long t0 = __builtin_readsteadycounter();  // 100 MHz
      while (__builtin_readsteadycounter() - t0 < 100ull * US) __builtin_amdgcn_s_sleep(2);
    }
  }
}
#define EXL3R4D_WG0 (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
#ifndef EXL3R4D_ABLATE_W1      // 1: K = 4's buffer form loads only the middle word a tile (w-1 and w+1 derived from it)
#define EXL3R4D_ABLATE_W1 0
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
#ifndef EXL3R4D_XSK_ABL
#define EXL3R4D_XSK_ABL 0
#endif
#ifndef EXL3R4D_KX_BUF
#define EXL3R4D_KX_BUF 1
#endif
#ifndef EXL3R4D_EPIC  // down: both output tiles of a wave from one pass over the staged tiles (epilogue_block)
#define EXL3R4D_EPIC 1  // ADOPTED (runs/epic): down -0.12/+0.04/-0.74/-0.09/-1.00 us at tp2 M=1/2/4/8/16; bits move
#endif                  // inside the tolerance (paired across-tile sum)
#ifndef EXL3R4D_DNW  // down: EXL3R4D_DNW 128-column blocks a workgroup side by side (the kernel's DNW form); 0/1 off
#define EXL3R4D_DNW 0  // REFUTED at 2 (runs/dnw): +0.5/+0.4/+0.6/-0.3/-0.8 us at down tp2 M=1/2/4/8/16
#endif
constexpr int kDnwB = EXL3R4D_DNW > 1 ? EXL3R4D_DNW : 2;
#ifndef EXL3R4D_MSPLIT_ABL  // REFUTED (runs/msp, msp2): +11.8 us at gate_up tp2 M=1 before any exchange (PF 2 or 4)
#define EXL3R4D_MSPLIT_ABL 0
#endif
#ifndef EXL3R4D_BLK_ABL
#define EXL3R4D_BLK_ABL 0
#endif
#ifndef EXL3R4D_EPIB  // down's output rotation as a butterfly over the tiles (exl3r4d_common.hip.h, epilogue_block)
#define EXL3R4D_EPIB 0  // REFUTED (runs/epib): +1.1/+3.5/+4.9/+7.2/+11.8 us at down tp2 M=1/2/4/8/16 -- two dependent
#endif                  // exchange stages lengthen the latency chain and 16 KB of LDS halves the workgroups a WGP
template <int BITS, int NPW, int MATS, bool APREP, bool RAW = false, int STAGED = 0, bool PIN = false>
static __device__ __forceinline__ void moe_pass(const Exl3r4dGemm& p, const Planes& w, int rc, int r1, float* lds,
                                                unsigned long long tk0 = 0, int cbx = -1, bool rot = true,
                                                int wsub = 0) {
  // EXL3R4D_TSTAMP (reading, timing only): workgroup (0, 0)'s thread 0 writes the realtime counter (100 MHz) at the
  // kernel's entry and at this pass's phase boundaries into the output's first 48 bytes, after its own epilogue.
  [[maybe_unused]] unsigned long long tsA = EXL3R4D_TSTAMP ? __builtin_readsteadycounter() : 0, tsB = 0, tsC = 0, tsD = 0;
  using P = Plan<BITS>;
  constexpr int NW = P::NW, WORDS = P::WORDS;
  const int SK = p.sk;
  const int tid = (int)threadIdx.x - wsub * WAVE32, lane = tid % WAVE32;  // DNW: a half's own wave numbers
  const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE32);  // keep tile bases in SGPRs
  const int col_local = wave / SK, ksplit = wave % SK;
  const int h = lane >> 4, nl = lane & 15;
  const int width = MATS == 2 ? w.rows / 2 : w.rows;  // this class's output columns
  const int ntiles = width / 16, ksteps = w.cols / 16;
  // EXL3R4D_XSK_ABL (reading, TIMING ONLY, wrong output): gate_up's K split across XS workgroups (grid.z), each its
  // slice and then the epilogue on its partial sum -- the K loop's balance and per-workgroup cost with D16's
  // cross-workgroup split, before its reduction exists. REFUTED (runs/xsk1, XSK-abl), gate_up tp2 us at M=1/2/4/8:
  // xs1 27.4/43.3/70.6/115.7, xs2 26.1/43.0/70.8/112.8, xs4 26.3/45.3/70.8/116.6, xs5 27.3/44.7/72.0/119.8, xs10
  // 29.7/49.7/83.8/146.0 -- the per-workgroup fixed chain paid XS times outweighs the balance gain, before the
  // reduction is even added. D16 is not the small-M lever for this kernel.
  constexpr int XS = (MATS == 2 && EXL3R4D_XSK_ABL > 1) ? EXL3R4D_XSK_ABL : 1;
  const int kps = ksteps / (SK * XS), ks0 = ((XS > 1 ? (int)blockIdx.z : 0) * SK + ksplit) * kps;
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
  // EXL3R4D_BLK_ABL (reading, TIMING ONLY, wrong output): the record as if stored column-block-major -- a 128-column
  // block's 8 tiles a k-step back to back and its k-steps consecutive, so a workgroup streams one contiguous run a
  // matrix instead of 1 KB pieces ntiles x 128 B apart. Same bytes read; the indexing a blocked layout would need.
  // REFUTED (runs/blk1): no difference at any decode M for either op (within 0.5 us): the record's order is not a cost.
  constexpr bool kBlk = EXL3R4D_BLK_ABL && !RAW;
  const size_t tile0 = kBlk ? ((size_t)(nt0 / 8) * ksteps + ks0) * 8 + nt0 % 8 : (size_t)ks0 * ntiles + nt0;
#pragma unroll
  for (int j = 0; j < NPW; j++) wt[j] = (gu8)(const unsigned char*)(wexp + (tile0 + j) * WORDS);
  const size_t kbytes = (size_t)(kBlk ? 8 : ntiles) * WORDS * 4;

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
  [[maybe_unused]] const unsigned bso = (unsigned)(tile0 * WORDS * 4);
  [[maybe_unused]] const unsigned bwm = wvo[0][0], bw0 = wvo[0][1], bwp = wvo[1][NW - 1];
  // R23: the three-word buffer form is K = 4's (a lane's two runs share their middle word only there); every other K
  // takes the global form
  // R23 (EXL3R4D_KX_BUF): another K takes the buffer form too, four words a tile (each run's two), addresses still
  // free of VALU; K = 4 keeps its three.
  constexpr bool kBuf = EXL3R4D_BUF && NW == 2 && (BITS == 4 || EXL3R4D_KX_BUF);
  constexpr int kAux = EXL3R4D_NT ? 1 : 0;  // TH_LOAD_NT, as the global form's __builtin_nontemporal_load

#define WOFF_(r, x) wvo[r][x]
#define EXL3_LOAD(ks, W_, A_)                                                        \
  do {                                                                               \
    if constexpr (kBuf) {                                                            \
      const unsigned so_ = bso + (unsigned)(ks) * (unsigned)kbytes;                  \
      _Pragma("unroll") for (int j = 0; j < NPW; j++) {                              \
        if constexpr (BITS == 4) {                                                   \
          unsigned m_, z_, q_;                                                       \
          if constexpr (EXL3R4D_ABLATE_WLOAD) { /* the buffer form's deletion too */ \
            z_ = (unsigned)(ks) * 0x9E3779B1u + bw0 + j; m_ = z_ ^ bwm; q_ = z_ + bwp; \
          } else if constexpr (EXL3R4D_ABLATE_W1) {                                  \
            z_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bw0 + j * 128, so_, kAux); \
            m_ = __builtin_rotateleft32(z_, 8); q_ = z_ ^ 0x5A5A5A5Au;               \
          } else {                                                                   \
            m_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwm + j * 128, so_, kAux); \
            z_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bw0 + j * 128, so_, kAux); \
            q_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwp + j * 128, so_, kAux); \
          }                                                                          \
          W_[j][0][0] = m_; W_[j][0][1] = z_; W_[j][1][0] = z_; W_[j][1][1] = q_;   \
        } else {                                                                     \
          _Pragma("unroll") for (int r = 0; r < 2; r++)                              \
            _Pragma("unroll") for (int x = 0; x < NW; x++)                           \
              W_[j][r][x] = __builtin_amdgcn_raw_buffer_load_b32(wrs, wvo[r][x] + j * WORDS * 4, so_, kAux); \
        }                                                                            \
      }                                                                              \
      if constexpr (EXL3R4D_ABLATE_ALOAD) {                                          \
        union { unsigned u[4]; v8h f; } ab_;                                         \
        for (int q_ = 0; q_ < 4; q_++) ab_.u[q_] = (unsigned)((ks) * 0x3C003C00u + avo + q_); \
        A_ = ab_.f;                                                                  \
      } else if constexpr (APREP) A_ = frag_lds(alds + aoff + (ks) * 16);            \
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
    if constexpr (EXL3R4D_SGB || PIN) { /* pin: the step's loads, its decode, its WMMAs (libr4d's OPT 2048) */ \
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
  epilogue_block<NPW, MATS, EXL3R4D_SVH_EARLY != 0, EXL3R4D_EPIB != 0, EXL3R4D_EPIC != 0>(
      p, w, acc, col_local, ksplit, lane, cb, row, in_run, live, lds,
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
// EXL3R4D_WGSTAMP (reading, timing only, never shipped): every GEMM workgroup's timeline -- the realtime counter
// (100 MHz) when its wave 0 enters and the latest of its waves' exits, the hardware slot it ran on (HW_ID1: SE, SA,
// WGP) and whether it had work -- into a module array per op, read by perfmoe --wgstamps through the two exported
// functions below. Names whether a dispatch's time is a tail of few workgroups or uniformly slow ones.
#if EXL3R4D_NCLS == 4 && defined(EXL3R4D_WGSTAMP)   // a reading build's stamps are the two-class compile's alone
#undef EXL3R4D_WGSTAMP
#endif
#ifndef EXL3R4D_WGSTAMP
#define EXL3R4D_WGSTAMP 0
#endif
#if EXL3R4D_WGSTAMP
constexpr int kWgsMax = 8192;
__device__ unsigned long long g_wgs_t[2][kWgsMax][2];  // [down, gate_up][linear workgroup][entry, exit]
__device__ unsigned g_wgs_hw[2][kWgsMax];               // HW_ID1 | live << 31
// entry: wave 0 lane 0 stamps the counter and the slot; exit: every wave's lane 0 (atomicMax), live workgroups only
static __device__ __forceinline__ void wgs_entry(int op, int wg) {
  if (threadIdx.x == 0 && wg < kWgsMax) {
    unsigned hw;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(hw));
    g_wgs_t[op][wg][0] = __builtin_readsteadycounter();
    g_wgs_hw[op][wg] = hw;
  }
}
static __device__ __forceinline__ void wgs_exit(int op, int wg) {
  if (threadIdx.x % WAVE32 == 0 && wg < kWgsMax) {
    atomicMax(&g_wgs_t[op][wg][1], __builtin_readsteadycounter());
    if (threadIdx.x == 0) atomicOr(&g_wgs_hw[op][wg], 1u << 31);
  }
}
#endif

#ifndef EXL3R4D_LB  // the GEMM kernel's launch bound (reading builds: 256 lets a deep prefetch hold its registers)
#define EXL3R4D_LB 1024
#endif
template <int BITS, int NPW, int MATS, bool APREP, bool RAW = false, int STAGED = 0, bool DNW = false>
__global__ void __launch_bounds__(EXL3R4D_LB) EXL3R4D_WPE_ATTR exl3r4d_gemm_kernel(Exl3r4dGemm p) {
  inject_delay<MATS == 2 ? EXL3R4D_DLY_GU : EXL3R4D_DLY_DN>(EXL3R4D_WG0);
  extern __shared__ float lds[];
  __shared__ int s_find[35];  // find_run_regs also hands r0, r1 through [33], [34]
#if EXL3R4D_WGSTAMP
  if (EXL3R4D_WGSTAMP & 1) wgs_entry(MATS - 1, (int)(blockIdx.y * gridDim.x + blockIdx.x));
#endif
  [[maybe_unused]] const unsigned long long tk0 = EXL3R4D_TSTAMP ? __builtin_readsteadycounter() : 0;
#if EXL3R4D_EXIT_STAGE == 99
  if (p.T == -1) *(int*)p.y = 1;   // stage "0": the dispatch alone, this grid and block
  return;
#endif
  // L7-2 (EXL3R4D_FRP, exl3r4d_common.hip.h): the plane pointers requested during the search, not after it
  Planes wpre = {};
  const bool frp = EXL3R4D_FRP && p.n_expert <= 8 * (int)blockDim.x;   // uniform
  const Run run = frp ? find_run_planes(p, blockIdx.y, s_find, wpre) : find_run_any(p, blockIdx.y, s_find);
#if EXL3R4D_EXIT_STAGE == 1
  if (run.e == 123456) *(int*)p.y = run.r0;
  return;
#endif
  // DNW (down, the plain K loop): a workgroup is TWO 128-column blocks of the run side by side, each half of its
  // waves one block (cb = 2 blockIdx.x + half), each half its own epilogue LDS -- the search, the plane pointers and
  // the launch paid once a pair, the blocks still in parallel. The same arithmetic a block: bit-identical.
  static_assert(!DNW || (MATS == 1 && !RAW && !APREP && STAGED == 0), "DNW is down's plain K loop");
  if (run.e < 0) {
    if constexpr (DNW) {
      for (int g = 0; g < kDnwB; ++g)
        zero_dead_rows(p, (int)blockIdx.y - run.nlive, (int)gridDim.y - run.nlive, kDnwB * blockIdx.x + g);
    } else if constexpr (MATS == 1 && !RAW && !APREP && (STAGED == 0 || STAGED == 4) && EXL3R4D_DNG > 1) {
      const int ncb = (p.y_cols + 127) / 128;
      for (int g = 0; g < EXL3R4D_DNG; ++g)
        if ((int)blockIdx.x * EXL3R4D_DNG + g < ncb)
          zero_dead_rows(p, (int)blockIdx.y - run.nlive, (int)gridDim.y - run.nlive, blockIdx.x * EXL3R4D_DNG + g);
    } else if (!RAW) {
      zero_dead_rows(p, (int)blockIdx.y - run.nlive, (int)gridDim.y - run.nlive, blockIdx.x);
    }
    return;  // RAW: the finisher writes every output row
  }
  const Planes w = frp ? wpre : planes_of(p, run.e);
#if EXL3R4D_EXIT_STAGE == 2
  if (w.codes == nullptr && w.suh == nullptr && w.svh == nullptr && w.suh1 == nullptr) *(int*)p.y = 1;
  return;
#endif
  const int width = MATS == 2 ? w.rows / 2 : w.rows;
  if constexpr (DNW) {  // the host launches it only with an even block count and both classes as wide as y
    const int part = __builtin_amdgcn_readfirstlane((int)(threadIdx.x / (blockDim.x / kDnwB)));
    const int hw = (int)blockDim.x / kDnwB / WAVE32;
    for (int rc = run.r0; rc < run.r1; rc += EXL3_BLOCK)
      moe_pass<BITS, NPW, MATS, APREP, RAW, STAGED>(p, w, rc, run.r1, lds + part * 2048, tk0,
                                                    kDnwB * (int)blockIdx.x + part, true, part * hw);
    return;
  }
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
#if EXL3R4D_WGSTAMP
  if (EXL3R4D_WGSTAMP & 4) wgs_entry(MATS - 1, (int)(blockIdx.y * gridDim.x + blockIdx.x));   // after the search
#endif
  for (int rc = run.r0; rc < run.r1; rc += chunk)
    moe_pass<BITS, NPW, MATS, APREP, RAW, STAGED>(p, w, rc, run.r1, lds, tk0);
#if EXL3R4D_WGSTAMP
  if (EXL3R4D_WGSTAMP & 2) wgs_exit(MATS - 1, (int)(blockIdx.y * gridDim.x + blockIdx.x));
#endif
}

// ----------------------------------------------------------------------------- WGB: the persistent decode grid
//
// EXL3R4D_WGB 1 (reading WGSTAMP, runs/p4b-s2): at tp2 M=8 gate_up has ~192 workgroup units (live experts x column
// blocks) for 128 resident slots (4 a WGP, VGPR-capped). The hardware hands the 64 second-round units to whichever
// slots free first, so WGPs end up with 5..7 units and the 7-unit ones set the kernel's end (106.5 us vs ~97 for the
// 6-unit ones) -- each WGP streams at the same rate with 2 or 4 resident, so the loss is the imbalance alone. Here
// the launch has G = resident-capacity workgroups and workgroup g takes units g, g + G, ...: the dispatcher
// interleaves workgroups over the WGPs, so every WGP gets ceil-balanced units. A unit is exactly the grid kernel's
// workgroup (blockIdx.x = cb, blockIdx.y = slot): the same code on the same data, bit-identical. Correctness does
// not depend on where a workgroup lands (each unit has one owner); only the balance does.
// REFUTED (runs/p4b-s6; capped test build 825/32/414/70 green, 312/312 outputs byte-identical to the grid kernel's):
// chain tp2 M=1/2/4/8/16 -0.2/-0.4/+6.4/+1.0/+8.7 us. The dispatcher does NOT place the persistent workgroups evenly
// (wgbs timeline M=8: 4..8 units a WGP), and HIP's occupancy query assumes 64 KB of LDS a WGP (G = 96, 89 resident).
// Balance needs WGP-aware claiming (HW_ID1), not a static stride. Kept off; the codegen notes below are what any
// persistent form of this K loop needs.
#ifndef EXL3R4D_WGB
#define EXL3R4D_WGB 0
#endif
#ifndef EXL3R4D_WGB_PIN  // the persistent kernel's K loop with the sched_group_barrier pin (EXL3R4D_SGB's)
#define EXL3R4D_WGB_PIN 1
#endif
#ifndef EXL3R4D_WGB_KARG  // the unit loop re-reads the argument block through an opaque kernarg pointer
#define EXL3R4D_WGB_KARG 1
#endif
#ifndef EXL3R4D_WGB_UPTR  // the plane pointers made uniform by readfirstlane
#define EXL3R4D_WGB_UPTR 1
#endif
#ifndef EXL3R4D_WGB_CAP  // TEST builds: at most this many persistent workgroups, so small suite shapes loop (0: off)
#define EXL3R4D_WGB_CAP 0
#endif
// readfirstlane returns a signed int: each half goes through unsigned before it is widened (a sign-extended low half
// made 0xffff....... addresses -- the page faults of runs/p4b-s4, found by bisection in runs/p4b-s5)
static __device__ __forceinline__ const void* uniform_ptr(const void* q) {
  const unsigned long long v = (unsigned long long)q;
  const unsigned lo = (unsigned)__builtin_amdgcn_readfirstlane((int)(unsigned)v);
  const unsigned hi = (unsigned)__builtin_amdgcn_readfirstlane((int)(unsigned)(v >> 32));
  return (const void*)((unsigned long long)hi << 32 | lo);
}
template <int BITS, int NPW, int MATS>
static __device__ __forceinline__ void gemm_unit(const Exl3r4dGemm& p, int cb, int slot, float* lds, int* s_find) {
  const Run run = find_run_any(p, slot, s_find);
  if (run.e < 0) {
    zero_dead_rows(p, slot - run.nlive, p.nslot - run.nlive, cb);
    return;
  }
  // The plane pointers made scalar by hand: after the previous unit's stores the compiler cannot prove the tables
  // unchanged, loads them as vectors, and the buffer descriptors built from them go through waterfall loops (a K loop
  // of 789 instructions and 23 spilled SGPRs instead of the grid kernel's 354). They are uniform: readfirstlane.
  Planes w = planes_of(p, run.e);
  if (EXL3R4D_WGB_UPTR) {
    w.codes = (const unsigned*)uniform_ptr(w.codes);
    w.suh = (const _Float16*)uniform_ptr(w.suh);
    w.suh1 = (const _Float16*)uniform_ptr(w.suh1);
    w.svh = (const _Float16*)uniform_ptr(w.svh);
    w.dn_suh = (const _Float16*)uniform_ptr(w.dn_suh);
  }
  const int width = MATS == 2 ? w.rows / 2 : w.rows;
  if (cb * 128 >= width) {
    zero_rows(p, run.r0, run.r1, cb);  // the narrower parity class: zeros past its own width
    return;
  }
#if EXL3R4D_WGSTAMP
  if (EXL3R4D_WGSTAMP & 4) wgs_entry(MATS - 1, slot * ((p.y_cols + 127) / 128) + cb);
#endif
  for (int rc = run.r0; rc < run.r1; rc += EXL3_BLOCK)
    moe_pass<BITS, NPW, MATS, false, false, 0, EXL3R4D_WGB_PIN != 0>(p, w, rc, run.r1, lds, 0, cb);
#if EXL3R4D_WGSTAMP
  if (EXL3R4D_WGSTAMP & 2) wgs_exit(MATS - 1, slot * ((p.y_cols + 127) / 128) + cb);
#endif
}

template <int BITS, int NPW, int MATS>
__global__ void __launch_bounds__(1024) EXL3R4D_WPE_ATTR exl3r4d_gemm_pu_kernel(Exl3r4dGemm p) {
  inject_delay<EXL3R4D_DLY_GU>(EXL3R4D_WG0);
  extern __shared__ float lds[];
  __shared__ int s_find[35];
  const int ncb = (p.y_cols + 127) / 128, units = ncb * p.nslot;
  typedef const __attribute__((address_space(4))) Exl3r4dGemm* karg;
  karg pp = (karg)__builtin_amdgcn_kernarg_segment_ptr();
  for (int u = blockIdx.x; u < units; u += gridDim.x) {
    // the argument block re-read every unit through a (scalar, constant) pointer the compiler cannot follow across
    // iterations: otherwise every argument field the body uses stays live across the loop (107 SGPRs, 32 spilled)
    if (EXL3R4D_WGB_KARG) asm volatile("" : "+s"(pp));
    gemm_unit<BITS, NPW, MATS>(EXL3R4D_WGB_KARG ? *(const Exl3r4dGemm*)pp : p, u % ncb, u / ncb, lds, s_find);
    __syncthreads();  // s_find and the LDS are the next unit's
  }
}

// gate_up's column blocks of expert e (its class's width / 128), and a resolved unit (WGQ, the fused prototype).
static __device__ __forceinline__ int unit_ncb(const Exl3r4dGemm& p, int e) {
  const int c = e % p.n_class, width = EXL3R4D_CLS_F(p, c, rows) / 2;
  return (width + 127) / 128;
}
struct Unit { int e, cb, r0, r1, ncb; };

// ----------------------------------------------------------------------------- WGQ: WGP-aware unit claiming
//
// EXL3R4D_WGQ 1 (WGB's lesson: the dispatcher places workgroups unevenly, so the balance must follow the WGP a
// workgroup actually runs on). Units u = slot * ncb + cb (the grid kernel's workgroups) are dealt round-robin to
// one list a WGP (list u % nl). A workgroup reads its WGP from HW_ID1 (wgp_key -> the dense id the instance probed
// at init), registers as active there, and drains that WGP's list; when it is empty it takes units only from lists
// whose WGP has NO active workgroup (so a WGP the dispatcher left empty, or crowded by another kernel, cannot strand
// units), then leaves. Each WGP thus processes ceil / floor(U / nl) units with however many workgroups it holds.
// The launch is the grid's U workgroups; the ones the GPU starts late find nothing and leave. The counter block is
// D18's form: zero at entry (instance init), the last workgroup to leave zeroes it again. A unit is the grid
// kernel's workgroup on the same data: bit-identical whichever workgroup runs it.
// REFUTED, all three forms (lazy-pool test builds with a 5-workgroup cap: suites green, 312/312 outputs identical):
// v1 lists over the whole grid (cheap units dealt too: real units/WGP still 4..8) chain tp2 M=8 +11.5 us (runs/p4b-
// s10); v2 dense live units + capacity grid: every WGP exactly 6 units, gate_up span -5.9 us, but the persistent
// prologue/exit +8.7 -> M=8 +2.8, M=16 +3.4 (s11); v3 claim folded into the search, next claim in flight: M=8 +4.0,
// M=16 +3.5 (s14). Persistent claiming does not pay for gate_up at decode.
static __device__ __forceinline__ int wgp_key() {  // HW_ID1: SE [20:18], SA [16], WGP [13:10] -> 8 bits
  unsigned hw;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(hw));
  return (int)(((hw >> 18) & 7) << 5 | ((hw >> 16) & 1) << 4 | ((hw >> 10) & 15));
}

#if EXL3R4D_WGQ
__global__ void __launch_bounds__(64) exl3r4d_wgp_probe_kernel(unsigned* bits) {
  if (threadIdx.x == 0) {
    const int k = wgp_key();
    atomicOr(&bits[k >> 5], 1u << (k & 31));
  }
  for (int i = 0; i < 8; ++i) __builtin_amdgcn_s_sleep(64);  // hold the slot a moment so the probe spreads
}

// Dense live units (v2: the cheap units -- the narrow class's zero blocks, spare slots -- are not dealt; dealing
// them made the REAL units per WGP vary 4..8, runs/p4b-s10): live expert e (id order) owns ncb_e = width_e / 128
// consecutive unit numbers. unit_count: their total; find_unit: the expert and column block of unit `idx` (the
// expert search weighted by ncb_e; two barriers, the same answer in every workgroup). sh: >= 36 ints of LDS.
static __device__ Unit find_unit(const Exl3r4dGemm& p, int idx, int* sh, int* total_out) {
  const int tid = threadIdx.x, nt = blockDim.x, lane = tid % WAVE32, wave = tid / WAVE32, nw = nt / WAVE32;
  const int per = (p.n_expert + nt - 1) / nt, e0 = tid * per;
  int cnt = 0;
  for (int e = e0; e < e0 + per && e < p.n_expert; e++)
    cnt += clamp_row(p.off[e + 1], p.T) > clamp_row(p.off[e], p.T) ? unit_ncb(p, e) : 0;
  int incl = cnt;
#pragma unroll
  for (int o = 1; o < WAVE32; o <<= 1) {
    const int v = __shfl_up(incl, o, WAVE32);
    if (lane >= o) incl += v;
  }
  if (tid == 0) sh[32] = -1;
  if (lane == WAVE32 - 1) sh[wave] = incl;
  __syncthreads();
  int base = 0, total = 0;
  for (int w = 0; w < nw; w++) {
    const int c = sh[w];
    base += w < wave ? c : 0;
    total += c;
  }
  const int excl = base + incl - cnt;
  if (idx == -2) idx = sh[35];  // WGQ v3: the claim tid 0 published before the first barrier
  if (cnt && idx >= excl && idx < excl + cnt) {
    int k = idx - excl;
    for (int e = e0; e < e0 + per && e < p.n_expert; e++) {
      if (clamp_row(p.off[e + 1], p.T) <= clamp_row(p.off[e], p.T)) continue;
      const int n = unit_ncb(p, e);
      if (k < n) { sh[32] = e; sh[33] = k; break; }
      k -= n;
    }
  }
  __syncthreads();
  Unit u;
  u.e = sh[32];
  u.cb = u.e >= 0 ? sh[33] : 0;
  u.r0 = u.e >= 0 ? clamp_row(p.off[u.e], p.T) : 0;
  u.r1 = u.e >= 0 ? clamp_row(p.off[u.e + 1], p.T) : 0;
  u.ncb = u.e >= 0 ? unit_ncb(p, u.e) : 0;
  *total_out = total;
  return u;
}

// One dense unit: the grid kernel's workgroup (cb of expert e's run), and, from the expert's last real block, the
// zero blocks of the narrower class (the grid's cheap workgroups for this expert).
template <int BITS, int NPW, int MATS>
static __device__ __forceinline__ void gemm_unit_run(const Exl3r4dGemm& p, const Unit& un, float* lds) {
  Planes w = planes_of(p, un.e);
  w.codes = (const unsigned*)uniform_ptr(w.codes);
  w.suh = (const _Float16*)uniform_ptr(w.suh);
  w.suh1 = (const _Float16*)uniform_ptr(w.suh1);
  w.svh = (const _Float16*)uniform_ptr(w.svh);
  w.dn_suh = (const _Float16*)uniform_ptr(w.dn_suh);
#if EXL3R4D_WGSTAMP
  if (EXL3R4D_WGSTAMP & 4) wgs_entry(MATS - 1, un.e * 4 + un.cb);
#endif
  for (int rc = un.r0; rc < un.r1; rc += EXL3_BLOCK)
    moe_pass<BITS, NPW, MATS, false, false, 0, EXL3R4D_WGB_PIN != 0>(p, w, rc, un.r1, lds, 0, un.cb);
  if (un.cb == un.ncb - 1)
    for (int cb = un.ncb; cb * 128 < p.y_cols; ++cb) zero_rows(p, un.r0, un.r1, cb);
#if EXL3R4D_WGSTAMP
  if (EXL3R4D_WGSTAMP & 2) wgs_exit(MATS - 1, un.e * 4 + un.cb);
#endif
}

// v3: the steal / leave path alone (wave 0): a list with units left and no workgroup on its WGP, or -1.
static __device__ int wgq_claim_steal(const Exl3r4dGemm& p, int units, int* sh) {
  if (threadIdx.x < WAVE32) {
    const int lane = (int)threadIdx.x, nl = p.wq_nl;
    unsigned* next = p.wq;
    const unsigned* active = p.wq + nl;
    int u = -1;
    for (;;) {
      bool cand = false;
      int j0 = -1;
      for (int j = lane; j < nl; j += WAVE32) {
        const unsigned nx = __atomic_load_n(&next[j], __ATOMIC_RELAXED);
        const unsigned ac = __atomic_load_n(&active[j], __ATOMIC_RELAXED);
        const bool cj = ac == 0 && (long long)j + (long long)nl * nx < units;
        if (cj && !cand) { cand = true; j0 = j; }
      }
      const unsigned long long any = __builtin_amdgcn_ballot_w32(cand);
      if (!any) break;
      const int j = __builtin_amdgcn_readlane(j0, __builtin_ctzll(any));
      if (lane == 0) {
        const long long cc = (long long)j + (long long)nl * atomicAdd(&next[j], 1u);
        u = cc < units ? (int)cc : -1;
      }
      u = __builtin_amdgcn_readfirstlane(u);
      if (u >= 0) break;
    }
    if (lane == 0) sh[0] = u;
  }
  __syncthreads();
  const int u = sh[0];
  __syncthreads();
  return u;
}

// v3 (orchestrator's point: v2 lost its balance win to the prologue): the claim's atomic is issued before the
// search and its answer travels with the search's first barrier (no separate count pass: the search returns the
// total), and each next claim is issued when a unit starts, so its latency hides behind the K loop. A claim past
// the workgroup's own list falls to wgq_claim's steal / leave path.
template <int BITS, int NPW, int MATS>
__global__ void __launch_bounds__(1024) EXL3R4D_WPE_ATTR exl3r4d_gemm_wq_kernel(Exl3r4dGemm p) {
  inject_delay<EXL3R4D_DLY_GU>(EXL3R4D_WG0);
  extern __shared__ float lds[];
  __shared__ int s_find[36];
  __shared__ int s_claim[1];
  const int nl = p.wq_nl;
  const int key = wgp_key();
  const int w0 = p.wq_map[key];
  const int w = w0 >= 0 && w0 < nl ? w0 : key % nl;  // a key the probe did not see: any list (correct either way)
  long long c = -1;  // tid 0: this workgroup's claim in flight (a unit number, unchecked against the total)
  if (threadIdx.x == 0) {
    atomicAdd(&p.wq[nl + w], 1u);
    c = (long long)w + (long long)nl * atomicAdd(&p.wq[w], 1u);
  }
  typedef const __attribute__((address_space(4))) Exl3r4dGemm* karg;
  karg pp = (karg)__builtin_amdgcn_kernarg_segment_ptr();
  for (;;) {
    asm volatile("" : "+s"(pp));  // the argument block re-read a unit (WGB's codegen note)
    const Exl3r4dGemm& q = *(const Exl3r4dGemm*)pp;
    if (threadIdx.x == 0) s_find[35] = (int)(c < 0x7fffffff ? c : 0x7fffffff);
    // find_unit's first barrier publishes s_find[35]; every thread reads it after the second
    int units;
    Unit un = find_unit(q, -2, s_find, &units);   // -2: the unit is s_find[35], read inside after barrier 1
    int u = s_find[35];
    if (u >= units) {  // own list exhausted: steal (only from WGPs with no workgroup) or leave
      __syncthreads();
      u = wgq_claim_steal(q, units, s_claim);
      if (u < 0) break;
      un = find_unit(q, u, s_find, &units);
    }
    if (threadIdx.x == 0) c = (long long)w + (long long)nl * atomicAdd(&q.wq[w], 1u);  // the next claim, in flight
    if (u == 0) {  // the rows no run covers (the grid's spare workgroups), every column block
      for (int cb = 0; cb * 128 < q.y_cols; ++cb) zero_dead_rows(q, 0, 1, cb);
    }
    gemm_unit_run<BITS, NPW, MATS>(q, un, lds);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicSub(&p.wq[nl + w], 1u);
    if (atomicAdd(&p.wq[2 * nl], 1u) == gridDim.x * gridDim.y - 1) {  // the last to leave: zero the block
      for (int j = 0; j < 2 * nl + 1; ++j) __atomic_store_n(&p.wq[j], 0u, __ATOMIC_RELAXED);
    }
  }
}
#endif

// ----------------------------------------------------------------------------- PDN: the persistent down grid
//
// EXL3R4D_PDN 1 (readings runs/p4b-s2: down at M=8 runs ~3.4 synchronized rounds of ~450 four-wave workgroups, each
// paying its own expert search and plane pointers before its first weight load; 524 GB/s inside its span): G =
// resident-capacity workgroups; each lists the live experts once (one search, ids into LDS), then claims units
// u = slot * ncb + cb from one ticket -- the next ticket taken while the current unit runs -- so units start
// back to back and the rounds desynchronise. Down's classes are as wide as y, so every live expert has ncb real
// units (no zero blocks). A unit is the grid kernel's workgroup on the same data: bit-identical.
// MEASURED (test build cap 5 + lazy pool: suites green, 312/312 identical; chain tp2 h_rot 1): always on M=8 -4.5/-4.9,
// M=16 -4.4/-5.1, M=4 +0.2/+0.5, M=1/2 +1.3/+1.9 (runs/p4b-s12); persistent past 2 grid rounds: M=4 +1.0 (s13); past 3
// rounds (below): M=8 -2.5..-2.7, M=16 -3.6..-3.9 but M=4 +0.8 on the GRID path (s15) -- cause outside the kernel,
// open. Off until that is understood.
// Live experts the LDS list holds; the host takes the grid when min(T, n_expert) could exceed it (16 x top-16). 512 B:
// a 2 KB list cost three resident workgroups a WGP (11 instead of 14, runs/p4b-s12).
constexpr int kPdnMaxLive = 256;
#if EXL3R4D_PDN
template <int BITS, int NPW, int MATS>
__global__ void __launch_bounds__(1024) EXL3R4D_WPE_ATTR exl3r4d_gemm_pdn_kernel(Exl3r4dGemm p) {
  inject_delay<EXL3R4D_DLY_DN>(EXL3R4D_WG0);
  extern __shared__ float lds[];
  __shared__ int s_find[36];
  __shared__ unsigned short s_live[kPdnMaxLive];
  const int tid = threadIdx.x, nt = blockDim.x, lane = tid % WAVE32, wave = tid / WAVE32, nw = nt / WAVE32;
  unsigned* tk = p.ctr;
  int next = 0;
  if (tid == 0) next = (int)atomicAdd(&tk[0], 1u);  // the first ticket, beside the search
  // the live experts in id order (find_run's scan, every one written out)
  const int per = (p.n_expert + nt - 1) / nt, e0 = tid * per;
  int cnt = 0;
  for (int e = e0; e < e0 + per && e < p.n_expert; e++) cnt += clamp_row(p.off[e + 1], p.T) > clamp_row(p.off[e], p.T);
  int incl = cnt;
#pragma unroll
  for (int o = 1; o < WAVE32; o <<= 1) {
    const int v = __shfl_up(incl, o, WAVE32);
    if (lane >= o) incl += v;
  }
  if (lane == WAVE32 - 1) s_find[wave] = incl;
  if (tid == 0) s_find[32] = next;
  __syncthreads();
  int base = 0, nlive = 0;
  for (int w = 0; w < nw; w++) {
    const int c = s_find[w];
    base += w < wave ? c : 0;
    nlive += c;
  }
  for (int e = e0, k = base + incl - cnt; e < e0 + per && e < p.n_expert; e++)
    if (clamp_row(p.off[e + 1], p.T) > clamp_row(p.off[e], p.T)) {
      if (k < kPdnMaxLive) s_live[k] = (unsigned short)e;
      ++k;
    }
  int u = s_find[32];
  __syncthreads();
  const int ncb = (p.y_cols + 127) / 128, units = (nlive < kPdnMaxLive ? nlive : kPdnMaxLive) * ncb;
  if (u == 0) {  // the rows no run covers (the grid's spare workgroups), every column block
    for (int cb = 0; cb < ncb; ++cb) zero_dead_rows(p, 0, 1, cb);
  }
  typedef const __attribute__((address_space(4))) Exl3r4dGemm* karg;
  karg pp = (karg)__builtin_amdgcn_kernarg_segment_ptr();
  while (u < units) {
    if (tid == 0) next = (int)atomicAdd(&tk[0], 1u);  // the next ticket, in flight across this unit
    asm volatile("" : "+s"(pp));  // the argument block re-read a unit (WGB's codegen note)
    const Exl3r4dGemm& q = *(const Exl3r4dGemm*)pp;
    const int e = s_live[u / ncb], cb = u % ncb;
    const int r0 = clamp_row(q.off[e], q.T), r1 = clamp_row(q.off[e + 1], q.T);
    Planes w = planes_of(q, e);
    w.codes = (const unsigned*)uniform_ptr(w.codes);
    w.suh = (const _Float16*)uniform_ptr(w.suh);
    w.suh1 = (const _Float16*)uniform_ptr(w.suh1);
    w.svh = (const _Float16*)uniform_ptr(w.svh);
    w.dn_suh = (const _Float16*)uniform_ptr(w.dn_suh);
#if EXL3R4D_WGSTAMP
    if (EXL3R4D_WGSTAMP & 4) wgs_entry(MATS - 1, u);
#endif
    for (int rc = r0; rc < r1; rc += EXL3_BLOCK)
      moe_pass<BITS, NPW, MATS, false, false, 0, EXL3R4D_WGB_PIN != 0>(q, w, rc, r1, lds, 0, cb);
#if EXL3R4D_WGSTAMP
    if (EXL3R4D_WGSTAMP & 2) wgs_exit(MATS - 1, u);
#endif
    if (tid == 0) s_find[33] = next;
    __syncthreads();
    u = s_find[33];
    __syncthreads();
  }
  if (tid == 0 && atomicAdd(&tk[1], 1u) == gridDim.x - 1) {  // the last to leave: zero the block
    __atomic_store_n(&tk[0], 0u, __ATOMIC_RELAXED);
    __atomic_store_n(&tk[1], 0u, __ATOMIC_RELAXED);
  }
}
#endif

// ----------------------------------------------------------------------------- FUSED: gate_up + down, one launch
//
// The v5 row moe_exl3_gate_up_down launches it (launch.c exl3r4d_launch_gd: a_rot 0, the prep in the prologue);
// EXL3R4D_FUSED_PROTO reading builds also export exl3r4d_fused_proto for perfmoe --fused (either a_rot). The
// decode pair (h_rot = 1) as ONE persistent launch. Tickets from one D18 counter are dealt in order:
// first gate_up's dense live units (expert k's ncb_gu column blocks), then down's units (expert k's column-block
// PAIRS, two 4-wave halves of a workgroup: DNW's bit-identical form). A finished gate_up unit counts itself into
// gdone[k] after its h' stores are visible; a down unit of expert k waits until gdone[k] == ncb_gu(k). Every
// waiter holds a later ticket than every unit it waits for, and those were claimed by running workgroups, so it
// always progresses (PLAN D18's argument). Each unit is the grid kernels' workgroup on the same data: bit-identical.
// What it buys (readings runs/p4b-s2): no gate_up -> down dispatch boundary (7.1 / 7.4 us at M=1 / 8), down units
// fill gate_up's imbalance tail (~9.5 us at M=8), down starts expert by expert instead of in synchronized rounds.
#ifndef EXL3R4D_FUSED_PROTO
#define EXL3R4D_FUSED_PROTO 0
#endif
static int wq_capacity(const void* fn, int threads, size_t smem, int nwgp);
#if EXL3R4D_FUSED_PROTO || EXL3R4D_PDN
// reading hook: what wq_capacity computes for the persistent down kernel, and the raw attributes it reads
extern "C" __attribute__((visibility("default"))) int exl3r4d_debug_capacity(int* regs, int* shared, int* nwgp);
#endif
// THE ARGUMENT BLOCK FITS THE ENGINE'S KERNARG SLOT: radiance 1.3.0 dispatches with 768 bytes a kernel's argument
// segment, 256 of them HIP's hidden arguments (the K loop reads blockDim), so the explicit block is at most 512 B (the
// static_asserts below, kRadKernargSlot). Two
// full views (2 x 320 B) did not fit (runs/p4b-s25b: "its kernel argument segment is 912 bytes and the backend's slot
// holds 768"), so down travels as what differs from gate_up's view -- its planes, output, width and split -- and the
// kernel rebuilds it (dn_view); the launcher checks the rebuild is the view fill() made, byte for byte.
#if EXL3R4D_NCLS == 2   // the fused gate_up_down launch is two-class only (interface v6: no four-class form)
struct Exl3r4dDnCls {
  const void* codes;
  const void* suh;
  const void* svh;
  int64_t     codes_s0, suh_s0, svh_s0;
  int32_t     rows, cols;
};
struct Exl3r4dDnDelta {
  Exl3r4dDnCls cls[2];
  void*        y;
  int64_t      y_pitch;
  int32_t      y_cols, sk;
};
struct Exl3r4dFused {
  Exl3r4dGemm gu;
  Exl3r4dDnDelta dnd;   // down's view less what it shares with gate_up's (dn_view)
  int nwgp;       // WGPs (the arrival rule's divisor)
  int prep;       // 1: gu.x is the token rows and the launch also rotates them into gu.xr (prep_gu's work, P units)
  unsigned* ctr;  // [0] tickets, [1] done workgroups, [2 + k] gate_up units done of live expert k, [kFuErr]
};
#endif
// radiance 1.3.0's kernarg slot (the dispatch refuses a larger argument segment by name) and what HIP appends to every
// kernel here (code object v5's hidden block: the kernels read blockDim). tests/gpu/kernarg_check.sh checks the built
// code objects against the slot too.
constexpr size_t kRadKernargSlot = 768, kHipHiddenArgs = 256;
static_assert(sizeof(Exl3r4dGemm) + kHipHiddenArgs <= kRadKernargSlot, "a single-view kernel's arguments must fit");
static_assert(sizeof(Exl3r4dPrefill) + kHipHiddenArgs <= kRadKernargSlot, "the prefill form's arguments must fit");
#if EXL3R4D_NCLS == 2
static_assert(sizeof(Exl3r4dFused) + kHipHiddenArgs <= kRadKernargSlot, "the fused launch's arguments must fit (above)");
// Down's view: gate_up's (routing, rows, counts) with down's planes, h as its rotated input (h_rot) and its own output.
__host__ __device__ __forceinline__ Exl3r4dGemm dn_view(const Exl3r4dGemm& gu, const Exl3r4dDnDelta& dd) {
  Exl3r4dGemm d = gu;
  for (int c = 0; c < 2; ++c) {
    d.cls[c].codes = dd.cls[c].codes;
    d.cls[c].suh = dd.cls[c].suh;
    d.cls[c].suh1 = nullptr;
    d.cls[c].svh = dd.cls[c].svh;
    d.cls[c].dn_suh = nullptr;
    d.cls[c].codes_s0 = dd.cls[c].codes_s0;
    d.cls[c].suh_s0 = dd.cls[c].suh_s0;
    d.cls[c].suh1_s0 = 0;
    d.cls[c].svh_s0 = dd.cls[c].svh_s0;
    d.cls[c].dn_suh_s0 = 0;
    d.cls[c].rows = dd.cls[c].rows;
    d.cls[c].cols = dd.cls[c].cols;
  }
  d.x = gu.y;
  d.y = dd.y;
  d.xr = gu.y;
  d.x_pitch = gu.y_pitch;
  d.y_pitch = dd.y_pitch;
  d.xr_pitch = gu.y_pitch;
  d.n_tok = gu.T;
  d.y_cols = dd.y_cols;
  d.x_bf16 = 0;
  d.sk = dd.sk;
  d.chunk = 16;
  d.h_rot = 1;
  d.a_rot = 0;
  d.ctr = nullptr;
  return d;
}
// ctr words: [0] gate_up tickets, [1] done workgroups, [2, 2 + kFuMaxLive) gdone, [kFuErr] the give-up flag,
// [kFuDn] down tickets, [kFuArr, + 256) arrivals a WGP key, [kFuPd] prep task tickets, [kFuPd + 1] prep tasks done
constexpr int kFuMaxLive = 256, kFuErr = 2 + kFuMaxLive, kFuDn = kFuErr + 1, kFuArr = kFuDn + 1, kFuPd = kFuArr + 256,
              kFuWords = kFuPd + 2 + kFuMaxLive;
#endif
// EXL3R4D_GD_PX (reading arm): each prep task counts itself into its expert's [kFuPd + 2 + k] and a gate_up unit waits
// for its own expert's rows only, so early experts' units stream while later rows rotate (fup7's one-task claims; s21's
// fup8 also batched the claims, which cost at small M). 0: every workgroup waits for every task once.
#ifndef EXL3R4D_GD_PX
#define EXL3R4D_GD_PX 0
#endif
#ifndef EXL3R4D_GD_WGS
#define EXL3R4D_GD_WGS 100
#endif
// EXL3R4D_GD_EARLY (P10-K2, target T3: the fused launch's prologue, ~7-8 us a layer at M=4..16 before any weight streams
// -- P9-L7's timeline: scan -> prep ticket -> table pointers + sorted_tok -> x / suh -> rotate / store -> next ticket ->
// count -> poll, then claim -> offsets + planes -> first weights). Bits, each moving dependent round trips off that chain;
// the same units, rows, rotations and sums whichever bits are set (bit-identical):
//   1  the scan loads its offsets once, unconditionally, before any use (find_run_regs's form), and beside them the
//      experts' suh / suh.1 table entries and sorted_tok, kept in LDS for the live ones: a prep task's lookups are LDS
//      reads instead of a table round trip and a sorted_tok round trip. REFUTED (P10-K2 s2a, 0000:03:00.0, ABAB x3,
//      perfmoe gd us a layer vs v1.2.0): +3.2 tp2 M=4 MTP-verify routing, +2.5 tp2 M=4, +4.8 tp2 M=8 verify, neutral
//      at M=16; stamped, the scan ENDS later (3.7 vs 2.7 us): every thread of every workgroup's table-entry loads.
//   2  each wave's first prep ticket taken at entry (in flight across the scan), and every next ticket as soon as the
//      current one is known (in flight across the task) instead of after it. REFUTED (s2a): +0.6..+6.5 us -- 1024
//      waves' entry claims land on ONE counter word at once and serialise at the atomic unit.
//   4  the workgroup's first unit ticket taken right after the scan (in flight across the prep) and, for a gate_up unit,
//      its rows and planes requested before the prep wait (in flight across the poll). CONFIRMED (s4, ABAB x3, every
//      GD-CHECK 0 bytes differ): tp2 M=4 verify 71.00 -> 69.07, M=4 95.03 -> 91.87, M=8 160.72 -> 158.46, M=16 276.87 ->
//      274.86, M=8 verify 114.27 -> 111.84, M=16 verify 196.31 -> 194.92; tp1 M=4 verify 123.75 -> 122.66, M=4 167.52 ->
//      166.59, M=8 297.32 -> 293.93. Stamped (tp2 M=4 verify) the wait-to-first-unit gap 1.5 -> ~1.0 us and the claim
//      no longer follows the wait. The claim block itself stays where it was (moving it ahead of the prep, with no
//      early claim, cost +0.6..+1.5 us: SGPRs 99 -> 103), so EARLY 0 is v1.2.0's code object byte for byte.
#ifndef EXL3R4D_GD_EARLY
#define EXL3R4D_GD_EARLY 0
#endif
#if EXL3R4D_SUHC
// SUHC: the resident copy of expert e's suh (m 0) / suh.1 (m 1), and whether a wave may read it: every wave's slice
// (wave = WPM m + kw: blocks kw, kw + WPM, ... of matrix m) written -- its bit in ok[e] -- by whichever prep met e
// first. The flag is loaded in the round the expert's planes are, so a resident expert pays no extra trip.
static __device__ __forceinline__ unsigned suhc_ok(const Exl3r4dGemm& p, int e) {
  return p.suhc_ok ? __hip_atomic_load(p.suhc_ok + e, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) : 0u;
}
static __device__ __forceinline__ _Float16* suhc_vec(const Exl3r4dGemm& p, int e, int m, int cols) {
  return (_Float16*)p.suhc + ((size_t)e * 2 + m) * (size_t)cols;
}
// the wave's loaded slice into the copy, then its bit (after the stores are visible)
template <int NBW, int WPM>
static __device__ __forceinline__ void suhc_fill(const Exl3r4dGemm& p, int e, int m, int kw, int blocks, int cols,
                                                 int wave, int lane, const v2u (&bs)[NBW]) {
  _Float16* dst = suhc_vec(p, e, m, cols);
#pragma unroll
  for (int b = 0; b < NBW; b++) {
    const int k = kw + b * WPM;
    if (k < blocks) *(v2u*)(dst + k * 128 + 4 * lane) = bs[b];
  }
  __threadfence();
  if (lane == 0) atomicOr(p.suhc_ok + e, 1u << wave);
}
#endif
// One sorted row r of the prep (EXL3R4D_PREP_ROW's workgroup, GD_PROW 2's prologue): 8 waves, MATS matrices; false
// for a row no live run covers (left as it is).
template <int MATS>
static __device__ __forceinline__ bool prep_row(const Exl3r4dGemm& p, int r, int wave, int lane) {
  typedef const __attribute__((address_space(1))) v2u* gv2;
  constexpr int WPM = 8 / MATS, NBW = 8;   // waves a matrix; a wave's 128-blocks at most (cols <= 128 * NBW * WPM)
  const int n = p.n_expert, m = wave / WPM, kw = wave - m * WPM;
  const int t = __builtin_amdgcn_readfirstlane(p.sorted[r]);
  int cnt = 0;
  for (int i0 = 0; i0 <= n; i0 += 17 * WAVE32) {
#pragma unroll
    for (int u = 0; u < 17; u++) {
      const int i = i0 + u * WAVE32 + lane;
      cnt += (i <= n && clamp_row(p.off[min(i, n)], p.T) <= r) ? 1 : 0;
    }
  }
#pragma unroll
  for (int o = WAVE32 / 2; o > 0; o >>= 1) cnt += __shfl_xor(cnt, o, WAVE32);
  const int e = __builtin_amdgcn_readfirstlane(cnt - 1);
  if (e < 0 || e >= n) return false;
  const Planes w = planes_of(p, e);
  const int blocks = w.cols / 128;
  const gu8 xb = (gu8)(const unsigned char*)p.x;
#if EXL3R4D_SUHC
  const bool hit = __builtin_amdgcn_readfirstlane(suhc_ok(p, e)) == 0xFFu;
  const gu8 sb = (gu8)(const unsigned char*)(hit ? suhc_vec(p, e, m, w.cols) : (m ? w.suh1 : w.suh));
#else
  const gu8 sb = (gu8)(const unsigned char*)(m ? w.suh1 : w.suh);
#endif
  const size_t xrow = (size_t)(t < 0 ? 0 : MATS == 2 ? min(t / p.top_k, p.n_tok - 1) : r);
  v2u bx[NBW], bs[NBW];
#pragma unroll
  for (int b = 0; b < NBW; b++) {   // unconditional: past the last block, the last block again
    const int k = min(kw + b * WPM, blocks - 1);
    bx[b] = *(gv2)(xb + (xrow * p.x_pitch + k * 128 + 4 * lane) * 2);
    bs[b] = *(gv2)(sb + (size_t)(k * 128 + 4 * lane) * 2);
  }
#if EXL3R4D_SUHC
  if (p.suhc && !hit) suhc_fill<NBW, WPM>(p, e, m, kw, blocks, w.cols, wave, lane, bs);   // wave-uniform
#endif
#pragma unroll
  for (int b = 0; b < NBW; b++) {
    const int k = kw + b * WPM;
    if (k < blocks) {   // wave-uniform
      const v2u o = t >= 0 ? rotate128_math(bx[b], bs[b], p.x_bf16 == 0, lane) : (v2u)(0u);
      *(v2u*)((_Float16*)p.xr + (size_t)r * p.xr_pitch + (size_t)m * w.cols + k * 128 + 4 * lane) = o;
    }
  }
  return true;
}
// EXL3R4D_GD_PROW (P5b): the prologue's prep by ROW -- a (row, matrix, every fourth 128-block) slice a wave ticket,
// its token and its expert's planes in one round, then every x / suh load of the slice before any rotation --
// instead of a ticket and a dependent load chain a (row, matrix, 128-block) task. Same rotation and stores:
// bit-identical. The classes must share K (gate_up's do), else the ticket form. Until 2026-10-14 the rows went to
// workgroups statically (row q to workgroup q % gridDim.x) on the argument that every launched workgroup runs; it
// does not when other kernels hold slots, and resident workgroups then waited on rows nobody could run (the PROW
// branch below says what that cost). GD_PROW 2 (a reading arm) still deals rows statically: do not ship it.
#ifndef EXL3R4D_GD_PROW   /* ADOPTED (P5b): with PREP_ROW, engine (locked) C=1/2/4/8 10.89/10.90/11.89/14.16 vs v14 10.89/10.92/12.03/14.32 */
#define EXL3R4D_GD_PROW 1
#endif
#ifndef EXL3R4D_PREP_ROW  /* ADOPTED (P5b, with GD_PROW): every decode prep a workgroup a sorted row (prep_row, below) */
#define EXL3R4D_PREP_ROW 1
#endif
#ifndef EXL3R4D_PREP_ROW_MIN   /* ...from this many tokens; fewer take the live-slot prep kernel (the 1-2 token path:
                                * PREP_ROW + SUHC put engine C=1 at 10.98-11.02 against 10.89-10.90 ms/step) */
#define EXL3R4D_PREP_ROW_MIN 3
#endif
#if EXL3R4D_SUHC && (!EXL3R4D_PREP_ROW || EXL3R4D_GD_PROW != 1)
#error "EXL3R4D_SUHC is read by the PREP_ROW kernel and the GD_PROW 1 prologue: build it with both"
#endif
#if EXL3R4D_GD_EARLY && (EXL3R4D_SUHC || EXL3R4D_GD_PROW != 1)
#error "EXL3R4D_GD_EARLY takes the GD_PROW 1 prologue without SUHC"
#endif
#if EXL3R4D_NCLS == 2   // the fused gate_up_down kernel and its launch (two-class only)
// EXL3R4D_FU_LDSB (reading arm, radiance 1.4.0's lds_barrier): the persistent loop's workgroup handshakes order LDS
// only -- each wave makes its own global writes visible (__threadfence) or acquires (agent fence) on its own side --
// so they need not be __syncthreads(), which on gfx12 also waits for every load and invalidates the vector cache.
#ifndef EXL3R4D_FU_LDSB
#define EXL3R4D_FU_LDSB 0
#endif
#define FU_BAR() do { if (EXL3R4D_FU_LDSB) lds_barrier(); else __syncthreads(); } while (0)
// tid 0: wait until *c >= need -- bounded (~1 s) so a broken dependency cannot hang the GPU (a hung kernel's queue
// teardown can take the card down; giving up raises *err, which the caller checks), RELAXED AGENT-scope polls:
// acquire loads invalidate the CU's L0/GL1 each time (M=1 +90 us, runs/p4b-s15) and HIP's __atomic_load_n is system
// scope, past L2 to memory (gate_up units beside three pollers a WGP ran 2x slower, runs/p4b-s17). The caller
// barriers and acquires (every wave) after it.
static __device__ __forceinline__ void fu_wait(unsigned* c, unsigned need, unsigned* err) {
  for (unsigned spin = 0; __hip_atomic_load(c, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < need; ++spin) {
    if (spin > (1u << 22)) { atomicOr(err, 1u); break; }
    __builtin_amdgcn_s_sleep(16);
  }
}

// AP (small M): gate_up units rotate their expert's rows into LDS themselves, as the grid gate_up's fused prep does
// (APREP, f.gu.chunk rows a pass, f.gu.xr unused): no prologue prep and no wait for it. A kernel of its own, so the
// other's K loops and registers are what they were.
template <bool AP>
__global__ void __launch_bounds__(1024) EXL3R4D_WPE_ATTR exl3r4d_fused_kernel(Exl3r4dFused f) {
  inject_delay<EXL3R4D_DLY_GD>(true);
#if EXL3R4D_WGSTAMP
  // bit 8: this workgroup's entry, and (below) the end of its prologue wait, in the down table's top slots
  if ((EXL3R4D_WGSTAMP & 8) && threadIdx.x == 0) g_wgs_t[0][kWgsMax - 1 - blockIdx.x][0] = __builtin_readsteadycounter();
#endif
  extern __shared__ float lds[];
  __shared__ int s_find[36];
  __shared__ unsigned short s_live[kFuMaxLive], s_gpre[kFuMaxLive + 1];
  __shared__ int s_tpre[kFuMaxLive + 1];   // f.prep: prefix of the live experts' prep tasks (rows x 2 x blocks)
#if EXL3R4D_GD_PROW
  __shared__ int s_r0[kFuMaxLive];   // each live expert's first row
#endif
#if EXL3R4D_GD_EARLY & 1
  __shared__ const void* s_psuh[2][kFuMaxLive];   // EARLY 1: each live expert's suh / suh.1 table entry
  __shared__ int s_srt[kFuMaxLive];                // EARLY 1: sorted_tok[0, T) (T <= blockDim.x <= kFuMaxLive)
#endif
  const int tid = threadIdx.x, nt = blockDim.x, lane = tid % WAVE32, wave = tid / WAVE32, nw = nt / WAVE32;
  unsigned* ctr = f.ctr;
  int next = 0;
  // this workgroup's arrival on its WGP, in flight across the scan (the arrival rule below reads it)
  if (tid == 0) s_find[34] = (int)atomicAdd(&ctr[kFuArr + wgp_key()], 1u);
#if EXL3R4D_GD_EARLY & 2
  // EARLY 2: this wave's first prep ticket (the PROW form's row slices), in flight across the scan; one taken where
  // there turns out to be no prep is harmless (the last workgroup to leave zeroes the counter)
  int ptk = 0;
  if (f.prep && f.gu.cls[0].cols == f.gu.cls[1].cols && nw == 8 && lane == 0) ptk = (int)atomicAdd(&ctr[kFuPd], 1u);
#endif
  // the live experts (id order) and the prefix of their gate_up units, once a workgroup
  const Exl3r4dGemm& g0 = f.gu;
#if EXL3R4D_GD_PROW == 2
  // GD_PROW 2: the prep's rows FIRST, beside the scan below (each row finds its own expert): row r to workgroup
  // r % gridDim.x; the live ones counted for the wait in PREP
  int prow_rows = 0;
  const bool prow_on = f.prep && g0.cls[0].cols == g0.cls[1].cols && nw == 8;
  if (prow_on)
    for (int r = blockIdx.x; r < g0.T; r += gridDim.x)
      prow_rows += prep_row<2>(g0, r, __builtin_amdgcn_readfirstlane(wave), lane) ? 1 : 0;
#endif
  const int per = (g0.n_expert + nt - 1) / nt, e0 = tid * per;
  auto ptasks = [&](int e) {   // expert e's prep tasks: rows x 2 matrices x 128-blocks of its class's K
    const int c = e % g0.n_class, cols = c ? g0.cls[1].cols : g0.cls[0].cols;
    return (clamp_row(g0.off[e + 1], g0.T) - clamp_row(g0.off[e], g0.T)) * 2 * (cols / 128);
  };
#if EXL3R4D_GD_EARLY & 1
  // EARLY 1: the thread's per + 1 offsets, its experts' suh / suh.1 entries and one sorted_tok value, all requested
  // before any is used (one round trip), where the loops below take a load and a wait an expert
  constexpr int kScanP = 2;   // per = n_expert / blockDim.x: 2 at 512 experts; more takes the loops below
  const bool fast = per <= kScanP && g0.T <= nt && nt <= kFuMaxLive;   // uniform
  int so[kScanP + 1];
  const void* sp[kScanP][2];
  int srt = 0;
  if (fast) {
#pragma unroll
    for (int i = 0; i <= kScanP; i++) so[i] = clamp_row(g0.off[min(e0 + min(i, per), g0.n_expert)], g0.T);
    if (f.prep) {
#pragma unroll
      for (int i = 0; i < kScanP; i++) {
        const int e = min(e0 + min(i, per - 1), g0.n_expert - 1), c = e % g0.n_class, j = e / g0.n_class;
        const Exl3r4dCls& C = c ? g0.cls[1] : g0.cls[0];
        sp[i][0] = entry_at(C.suh, C.suh_s0, j);
        sp[i][1] = C.suh1 ? entry_at(C.suh1, C.suh1_s0, j) : sp[i][0];
      }
    }
    srt = g0.sorted[min(tid, g0.T - 1)];
  }
  auto ptasks_o = [&](int i) {   // expert e0 + i's prep tasks, from the loaded offsets
    const int c = (e0 + i) % g0.n_class, cols = c ? g0.cls[1].cols : g0.cls[0].cols;
    return (so[i + 1] - so[i]) * 2 * (cols / 128);
  };
#else
  constexpr bool fast = false;
#endif
  int cnt = 0, cntg = 0, cntt = 0;
  if (fast) {
#if EXL3R4D_GD_EARLY & 1
#pragma unroll
    for (int i = 0; i < kScanP; i++)
      if (i < per && e0 + i < g0.n_expert && so[i + 1] > so[i]) {
        cnt++;
        cntg += unit_ncb(g0, e0 + i);
        if (f.prep) cntt += ptasks_o(i);
      }
#endif
  } else {
  for (int e = e0; e < e0 + per && e < g0.n_expert; e++)
    if (clamp_row(g0.off[e + 1], g0.T) > clamp_row(g0.off[e], g0.T)) {
      cnt++;
      cntg += unit_ncb(g0, e);
      if (f.prep) cntt += ptasks(e);
    }
  }
  int incl = cnt, inclg = cntg, inclt = cntt;
#pragma unroll
  for (int o = 1; o < WAVE32; o <<= 1) {
    const int v = __shfl_up(incl, o, WAVE32), vg = __shfl_up(inclg, o, WAVE32), vt = __shfl_up(inclt, o, WAVE32);
    if (lane >= o) { incl += v; inclg += vg; inclt += vt; }
  }
  if (lane == WAVE32 - 1) { s_find[wave] = incl; s_find[16 + wave] = inclg; s_tpre[kFuMaxLive - nw + wave] = inclt; }
  __syncthreads();
  int base = 0, baseg = 0, baset = 0, nlive = 0, ug = 0, ntasks = 0;
  for (int w = 0; w < nw; w++) {
    const int ct = s_tpre[kFuMaxLive - nw + w];
    base += w < wave ? s_find[w] : 0;
    baseg += w < wave ? s_find[16 + w] : 0;
    baset += w < wave ? ct : 0;
    nlive += s_find[w];
    ug += s_find[16 + w];
    ntasks += ct;
  }
  __syncthreads();   // s_tpre's tail held the wave totals; the prefix below overwrites it
  if (fast) {
#if EXL3R4D_GD_EARLY & 1
    int k = base + incl - cnt, kg = baseg + inclg - cntg, kt = baset + inclt - cntt;
#pragma unroll
    for (int i = 0; i < kScanP; i++)
      if (i < per && e0 + i < g0.n_expert && so[i + 1] > so[i]) {
        if (k < kFuMaxLive) {
          s_live[k] = (unsigned short)(e0 + i); s_gpre[k] = (unsigned short)kg; s_tpre[k] = kt; s_r0[k] = so[i];
          s_psuh[0][k] = sp[i][0]; s_psuh[1][k] = sp[i][1];
        }
        ++k;
        kg += unit_ncb(g0, e0 + i);
        if (f.prep) kt += ptasks_o(i);
      }
    if (tid < g0.T) s_srt[tid] = srt;
#endif
  } else {
  for (int e = e0, k = base + incl - cnt, kg = baseg + inclg - cntg, kt = baset + inclt - cntt;
       e < e0 + per && e < g0.n_expert; e++)
    if (clamp_row(g0.off[e + 1], g0.T) > clamp_row(g0.off[e], g0.T)) {
      if (k < kFuMaxLive) { s_live[k] = (unsigned short)e; s_gpre[k] = (unsigned short)kg; s_tpre[k] = kt; }
#if EXL3R4D_GD_PROW
      if (k < kFuMaxLive) s_r0[k] = clamp_row(g0.off[e], g0.T);
#endif
      ++k;
      kg += unit_ncb(g0, e);
      if (f.prep) kt += ptasks(e);
    }
  }
  if (tid == 0) {
    s_gpre[nlive < kFuMaxLive ? nlive : kFuMaxLive] = (unsigned short)ug;
    s_tpre[nlive < kFuMaxLive ? nlive : kFuMaxLive] = ntasks;
  }
  __syncthreads();
#if EXL3R4D_GD_EARLY & 4
  // EARLY 4: the first unit ticket now, in flight across the prep (tid 0 keeps it in a register until the prep is
  // counted); pre_resolve hands it to every wave and, for a gate_up unit, requests its rows and planes before the poll
  // claim()'s first ticket under the same arrival rule (below): the gate_up counter's atomic issued now and its value
  // held, unread, by tid 0; a workgroup it leaves without a gate_up unit takes its down ticket at the handover
  int t_first = 0;
  if (tid == 0) t_first = s_find[34] < (ug + f.nwgp - 1) / (f.nwgp > 0 ? f.nwgp : 1) ? (int)atomicAdd(&ctr[0], 1u) : ug;
  auto first_ticket = [&]() -> int {   // tid 0 only
    return t_first < ug ? t_first : ug + (int)atomicAdd(&ctr[kFuDn], 1u);
  };
  bool pre = false, pre_lds = false;
  int pr0 = 0, pr1 = 0;
  Planes pw = {};
  auto pre_resolve = [&]() {
    if (tid == 0) s_find[32] = first_ticket();
    FU_BAR();
    pre_lds = true;
    const int t0 = __builtin_amdgcn_readfirstlane(s_find[32]);
    if (t0 < ug) {   // uniform
      int lo = 0, hi = nlive - 1;
      while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (s_gpre[mid] <= t0) lo = mid; else hi = mid - 1; }
      const int e = __builtin_amdgcn_readfirstlane((int)s_live[__builtin_amdgcn_readfirstlane(lo)]);
      pr0 = __builtin_amdgcn_readfirstlane(clamp_row(f.gu.off[e], f.gu.T));
      pr1 = __builtin_amdgcn_readfirstlane(clamp_row(f.gu.off[e + 1], f.gu.T));
      pw = planes_of(f.gu, e);
      pre = true;
    }
  };
#endif
#if EXL3R4D_WGSTAMP
  // bit 16 (prologue phases, P10-K2): the scan done (gate_up table's top slots, entry); the workgroup's own prep tasks
  // done and counted (same slot, exit); its first ticket known (down table, 256 slots lower, entry) and its first unit's
  // planes resolved, just before its K loop (same, exit)
  if ((EXL3R4D_WGSTAMP & 16) && tid == 0) g_wgs_t[1][kWgsMax - 1 - blockIdx.x][0] = __builtin_readsteadycounter();
#endif
  if (f.prep && ntasks > 0) {
    // PREP (prep_gu's work, a_rot 0): every row of every live run rotated into xr -- exactly the prep kernel's
    // rotation and store -- by tasks (row, matrix, 128-block) that waves claim one at a time from a ticket; a wave's
    // count goes in after its stores are visible, and every workgroup waits for all of them once. A claimed task is
    // done by a running wave that never waits while holding one, so the wait always ends (D18's argument).
    const int nlv = nlive < kFuMaxLive ? nlive : kFuMaxLive;
    const int wv = __builtin_amdgcn_readfirstlane(wave);
    int done = 0;
#if EXL3R4D_GD_PROW
    bool prow = false;
    {
#if EXL3R4D_GD_PROW == 2
      if (prow_on) {   // the rows were rotated at entry: count them and wait for every workgroup's
        __threadfence();
        FU_BAR();
        if (tid == 0 && prow_rows) atomicAdd(&ctr[kFuPd + 1], (unsigned)(prow_rows * 2 * (g0.cls[0].cols / 128)));
        if (tid == 0) fu_wait(&ctr[kFuPd + 1], (unsigned)ntasks, &ctr[kFuErr]);
        FU_BAR();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        prow = true;
      }
#else
      if (g0.cls[0].cols == g0.cls[1].cols && nw == 8) {   // uniform
        typedef const __attribute__((address_space(1))) v2u* gv2;
        const int blocks = g0.cls[0].cols / 128, per_row = 2 * blocks, nrows = ntasks / per_row;
        // A WAVE A TICKET, never a row a workgroup by blockIdx (the v1.0.0-v1.1.0 form, `q = blockIdx.x; q +=
        // gridDim.x`): that form gave rows to workgroups that may not be resident -- the grid is the card's whole
        // capacity and the engine runs other kernels beside this one (the shared expert's GEMMs, the all-reduce) -- while
        // every resident workgroup waited below for all rows: the wait ran to fu_wait's ~1.7 s give-up (the 2 s MTP
        // C=4 steps, the "hang" capped at 8 rows in v1.1.0). A ticket is taken by a running wave that never waits
        // while holding one (D18). Slice sl of row q is what wave sl did in the static form: matrix sl / 4, blocks
        // sl % 4, + 4, ...; same rotation and stores, bit-identical.
#if EXL3R4D_GD_EARLY & 2
        int tk = __builtin_amdgcn_readfirstlane(ptk);
#endif
        for (;;) {
#if EXL3R4D_GD_EARLY & 2
          if (tk >= nrows * 8) break;
          int nx = 0;
          if (lane == 0) nx = (int)atomicAdd(&ctr[kFuPd], 1u);   // the next ticket, in flight across this task
#else
          int tk = 0;
          if (lane == 0) tk = (int)atomicAdd(&ctr[kFuPd], 1u);
          tk = __builtin_amdgcn_readfirstlane(tk);
          if (tk >= nrows * 8) break;
#endif
          const int q = tk >> 3, sl = tk & 7, m = sl >> 2, kw = sl & 3;
          const int tq = q * per_row;
          int lo = 0, hi = nlv - 1;
          while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (s_tpre[mid] <= tq) lo = mid; else hi = mid - 1; }
          const int k = __builtin_amdgcn_readfirstlane(lo);
          const int e = __builtin_amdgcn_readfirstlane((int)s_live[k]);
          const int r = __builtin_amdgcn_readfirstlane(s_r0[k] + (tq - s_tpre[k]) / per_row);
          const gu8 xb = (gu8)(const unsigned char*)g0.x;
#if EXL3R4D_GD_EARLY & 1
          // the classes share K here (the branch's condition), so w.cols is either class's
          struct { int cols; } w = { g0.cls[0].cols };
          int t;
          gu8 sb;
          if (fast) {   // uniform
            t = __builtin_amdgcn_readfirstlane(s_srt[r]);
            sb = (gu8)(const unsigned char*)ptr_uniform(s_psuh[m][k]);
          } else {
            const Planes wp = planes_of(g0, e);
            t = __builtin_amdgcn_readfirstlane(g0.sorted[r]);
            sb = (gu8)(const unsigned char*)(m ? wp.suh1 : wp.suh);
          }
#else
          const Planes w = planes_of(g0, e);
          const int t = __builtin_amdgcn_readfirstlane(g0.sorted[r]);
#if EXL3R4D_SUHC
          const bool hit = __builtin_amdgcn_readfirstlane(suhc_ok(g0, e)) == 0xFFu;
          const gu8 sb = (gu8)(const unsigned char*)(hit ? suhc_vec(g0, e, m, w.cols) : (m ? w.suh1 : w.suh));
#else
          const gu8 sb = (gu8)(const unsigned char*)(m ? w.suh1 : w.suh);
#endif
#endif
          const size_t xrow = (size_t)(t < 0 ? 0 : min(t / g0.top_k, g0.n_tok - 1));
          v2u bx[8], bs[8];
#pragma unroll
          for (int b = 0; b < 8; b++) {   // unconditional: past the last block, the last block again
            const int kb = min(kw + b * 4, blocks - 1);
            bx[b] = *(gv2)(xb + (xrow * g0.x_pitch + kb * 128 + 4 * lane) * 2);
            bs[b] = *(gv2)(sb + (size_t)(kb * 128 + 4 * lane) * 2);
          }
#if EXL3R4D_SUHC
          if (g0.suhc && !hit) suhc_fill<8, 4>(g0, e, m, kw, blocks, w.cols, sl, lane, bs);   // wave-uniform
#endif
#pragma unroll
          for (int b = 0; b < 8; b++) {
            const int kb = kw + b * 4;
            if (kb < blocks) {   // wave-uniform
              const v2u o = t >= 0 ? rotate128_math(bx[b], bs[b], g0.x_bf16 == 0, lane) : (v2u)(0u);
              *(v2u*)((_Float16*)g0.xr + (size_t)r * g0.xr_pitch + (size_t)m * w.cols + kb * 128 + 4 * lane) = o;
            }
          }
          done += (blocks - kw + 3) / 4;   // this slice's blocks: the four slices of a matrix sum to `blocks`
#if EXL3R4D_GD_EARLY & 2
          tk = __builtin_amdgcn_readfirstlane(nx);
#endif
        }
        __threadfence();   // this wave's stores visible before its count
        if (lane == 0 && done) atomicAdd(&ctr[kFuPd + 1], (unsigned)done);
#if EXL3R4D_WGSTAMP
        if ((EXL3R4D_WGSTAMP & 16) && tid == 0) g_wgs_t[1][kWgsMax - 1 - blockIdx.x][1] = __builtin_readsteadycounter();
#endif
#if EXL3R4D_GD_EARLY & 4
        pre_resolve();   // the first unit's rows and planes in flight across the poll below
#endif
        if (tid == 0) fu_wait(&ctr[kFuPd + 1], (unsigned)ntasks, &ctr[kFuErr]);
        FU_BAR();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        prow = true;
      }
#endif
    }
#else
    constexpr bool prow = false;
#endif
    for (; !prow;) {
      int tk = 0;
      if (lane == 0) tk = (int)atomicAdd(&ctr[kFuPd], 1u);
      tk = __builtin_amdgcn_readfirstlane(tk);
      if (tk >= ntasks) break;
      int lo = 0, hi = nlv - 1;
      while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (s_tpre[mid] <= tk) lo = mid; else hi = mid - 1; }
      const int k = __builtin_amdgcn_readfirstlane(lo);
      const int e = __builtin_amdgcn_readfirstlane((int)s_live[k]);
      Planes w = planes_of(g0, e);
      const int blocks = w.cols / 128, per_row = 2 * blocks, local = tk - __builtin_amdgcn_readfirstlane(s_tpre[k]);
      const int i = local / per_row, rem = local - i * per_row, m = rem / blocks, blk = rem - m * blocks;
      const int r = clamp_row(g0.off[e], g0.T) + i;
      const v2u o = prep_rotate<2>(g0, w, r, m, blk, lane);
      *(v2u*)((_Float16*)g0.xr + (size_t)r * g0.xr_pitch + (size_t)m * w.cols + blk * 128 + 4 * lane) = o;
      ++done;
      if (EXL3R4D_GD_PX) {   // this task's rows visible, then counted for its expert
        __threadfence();
        if (lane == 0) atomicAdd(&ctr[kFuPd + 2 + k], 1u);
      }
    }
    (void)wv;
    if (!EXL3R4D_GD_PX && !prow) {
      __threadfence();
      if (lane == 0 && done) atomicAdd(&ctr[kFuPd + 1], (unsigned)done);
      if (tid == 0) fu_wait(&ctr[kFuPd + 1], (unsigned)ntasks, &ctr[kFuErr]);
      FU_BAR();
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
  }
#if EXL3R4D_WGSTAMP
  if ((EXL3R4D_WGSTAMP & 8) && threadIdx.x == 0) g_wgs_t[0][kWgsMax - 1 - blockIdx.x][1] = __builtin_readsteadycounter();
#endif
  const int ncb_dn = (f.dnd.y_cols + 127) / 128, npair = ncb_dn / 2, units = ug + nlive * npair;
  // Who may take gate_up units. Tickets go to whichever workgroups start first, and those sit 2-3 a WGP: at M=1
  // the 22 gate_up units then shared WGPs and ran 2-3x slower (30-42 us against the grid kernel's 15.9,
  // runs/p4b-s16). So a workgroup counts its arrival on its own WGP (HW_ID1 key) and only the first ag =
  // ceil(ug / nwgp) arrivals of each WGP take gate_up tickets (one counter, gate_up's units first); the others take
  // down tickets (a second counter) and wait there. Every gate_up unit is claimed by a workgroup that never waits
  // (it takes gate_up tickets until they run out, then down ones), so every waiter still progresses (D18).
  const int ag = (ug + f.nwgp - 1) / (f.nwgp > 0 ? f.nwgp : 1);
  bool gw = __builtin_amdgcn_readfirstlane(s_find[34]) < ag;   // this workgroup takes gate_up tickets (for now)
  auto claim = [&]() -> int {   // tid 0 only
    if (gw) {
      const int c = (int)atomicAdd(&ctr[0], 1u);
      if (c < ug) return c;
    }
    return ug + (int)atomicAdd(&ctr[kFuDn], 1u);
  };
#if EXL3R4D_GD_EARLY & 4
  if (!pre_lds) {   // the paths that did not hand the first ticket over before the wait
    if (tid == 0) s_find[32] = first_ticket();
    FU_BAR();
  }
#else
  if (tid == 0) s_find[32] = claim();
  FU_BAR();
#endif
  int t = __builtin_amdgcn_readfirstlane(s_find[32]);
  gw = gw && t < ug;
  FU_BAR();
#if EXL3R4D_WGSTAMP
  if ((EXL3R4D_WGSTAMP & 16) && tid == 0) g_wgs_t[0][kWgsMax - 257 - blockIdx.x][0] = __builtin_readsteadycounter();
  bool first_unit = true;
#endif
  typedef const __attribute__((address_space(4))) Exl3r4dFused* karg;
  karg pp = (karg)__builtin_amdgcn_kernarg_segment_ptr();
  if (t == 0) {  // the output rows no run covers (the grid's spare workgroups), every column block of y
    const Exl3r4dGemm dz = dn_view(f.gu, f.dnd);
    for (int cb = 0; cb < ncb_dn; ++cb) zero_dead_rows(dz, 0, 1, cb);
  }
  while (t < units) {
    if (tid == 0) next = claim();  // the next ticket, in flight across this unit
    asm volatile("" : "+s"(pp));
    const Exl3r4dFused& q = *(const Exl3r4dFused*)pp;
    if (t < ug) {  // gate_up unit t: the live expert whose prefix holds it
      const int tg = t;
      int lo = 0, hi = nlive - 1;
      while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (s_gpre[mid] <= tg) lo = mid; else hi = mid - 1; }
      // every coordinate made scalar by hand: values read back from LDS in a data-dependent search are not seen as
      // uniform, and the buffer descriptors built from them went through waterfall loops (K loop 1028 instr)
      const int k = __builtin_amdgcn_readfirstlane(lo);
      const int e = __builtin_amdgcn_readfirstlane((int)s_live[k]);
      const int g0k = __builtin_amdgcn_readfirstlane((int)s_gpre[k]), g1k = __builtin_amdgcn_readfirstlane((int)s_gpre[k + 1]);
      const int cb = tg - g0k;
      if (EXL3R4D_GD_PX && f.prep) {   // this expert's rows rotated: all its tasks counted (they hold earlier tickets)
        const unsigned need_p = (unsigned)__builtin_amdgcn_readfirstlane(s_tpre[k + 1] - s_tpre[k]);
        if (tid == 0) fu_wait(&ctr[kFuPd + 2 + k], need_p, &ctr[kFuErr]);
        __syncthreads();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
      }
      Unit un;
      un.e = e; un.cb = cb; un.ncb = g1k - g0k;
#if EXL3R4D_GD_EARLY & 4
      Planes w;
      if (pre) {   // uniform: the first unit, requested before the prep wait (the same ticket, the same expert)
        un.r0 = pr0;
        un.r1 = pr1;
        w = pw;
        pre = false;
      } else {
        un.r0 = __builtin_amdgcn_readfirstlane(clamp_row(q.gu.off[e], q.gu.T));
        un.r1 = __builtin_amdgcn_readfirstlane(clamp_row(q.gu.off[e + 1], q.gu.T));
        w = planes_of(q.gu, e);
      }
#else
      un.r0 = __builtin_amdgcn_readfirstlane(clamp_row(q.gu.off[e], q.gu.T));
      un.r1 = __builtin_amdgcn_readfirstlane(clamp_row(q.gu.off[e + 1], q.gu.T));
      Planes w = planes_of(q.gu, e);
#endif
      w.codes = (const unsigned*)uniform_ptr(w.codes);
      w.svh = (const _Float16*)uniform_ptr(w.svh);
      w.dn_suh = (const _Float16*)uniform_ptr(w.dn_suh);
      w.suh = (const _Float16*)uniform_ptr(w.suh);
      w.suh1 = (const _Float16*)uniform_ptr(w.suh1);
#if EXL3R4D_WGSTAMP
      if (EXL3R4D_WGSTAMP & 4) wgs_entry(1, tg);
      if ((EXL3R4D_WGSTAMP & 16) && first_unit && tid == 0)
        g_wgs_t[0][kWgsMax - 257 - blockIdx.x][1] = __builtin_readsteadycounter();
      first_unit = false;
#endif
      if constexpr (AP) {
        const int chunk = __builtin_amdgcn_readfirstlane(q.gu.chunk);
        for (int rc = un.r0; rc < un.r1; rc += chunk)
          moe_pass<4, 2, 2, true, false, 0, true>(q.gu, w, rc, un.r1, lds, 0, cb);
      } else {
        for (int rc = un.r0; rc < un.r1; rc += EXL3_BLOCK)
          moe_pass<4, 2, 2, false, false, 0, true>(q.gu, w, rc, un.r1, lds, 0, cb);
      }
#if EXL3R4D_WGSTAMP
      if (EXL3R4D_WGSTAMP & 2) wgs_exit(1, tg);
#endif
      __threadfence();   // h' visible device-wide before the count
      FU_BAR();
      if (tid == 0) atomicAdd(&ctr[2 + k], 1u);
    } else {  // down unit: expert k's column-block pair, a 4-wave half each
      const int d = t - ug, k = d / npair, pair = d - k * npair;
      const int e = __builtin_amdgcn_readfirstlane((int)s_live[k]);
      const unsigned need = (unsigned)__builtin_amdgcn_readfirstlane((int)s_gpre[k + 1] - (int)s_gpre[k]);
      if (tid == 0) fu_wait(&ctr[2 + k], need, &ctr[kFuErr]);
      FU_BAR();
      // every wave acquires: in WGP mode a workgroup's waves sit on two CUs with their own L0 caches
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
      const Exl3r4dGemm qdn = dn_view(q.gu, q.dnd);
      const int r0 = __builtin_amdgcn_readfirstlane(clamp_row(qdn.off[e], qdn.T));
      const int r1 = __builtin_amdgcn_readfirstlane(clamp_row(qdn.off[e + 1], qdn.T));
      Planes w = planes_of(qdn, e);
      w.codes = (const unsigned*)uniform_ptr(w.codes);
      w.svh = (const _Float16*)uniform_ptr(w.svh);
      w.suh = (const _Float16*)uniform_ptr(w.suh);
      w.suh1 = (const _Float16*)uniform_ptr(w.suh1);
      w.dn_suh = nullptr;
      const int part = __builtin_amdgcn_readfirstlane(tid / (nt / 2)), hw = nt / 2 / WAVE32;
#if EXL3R4D_WGSTAMP
      if (EXL3R4D_WGSTAMP & 4) wgs_entry(0, d);
      if ((EXL3R4D_WGSTAMP & 16) && first_unit && tid == 0)
        g_wgs_t[0][kWgsMax - 257 - blockIdx.x][1] = __builtin_readsteadycounter();
      first_unit = false;
#endif
      for (int rc = r0; rc < r1; rc += EXL3_BLOCK)
        moe_pass<4, 2, 1, false, false, 0, true>(qdn, w, rc, r1, lds + part * 2048, 0, 2 * pair + part, true, part * hw);
#if EXL3R4D_WGSTAMP
      if (EXL3R4D_WGSTAMP & 2) wgs_exit(0, d);
#endif
    }
    if (tid == 0) s_find[33] = next;
    FU_BAR();
    t = __builtin_amdgcn_readfirstlane(s_find[33]);
    gw = gw && t < ug;
    FU_BAR();
  }
  if (tid == 0 && atomicAdd(&ctr[1], 1u) == gridDim.x - 1) {  // the last to leave: zero the block
    for (int j = 0; j < 2 + nlive && j < 2 + kFuMaxLive; ++j) __atomic_store_n(&ctr[j], 0u, __ATOMIC_RELAXED);
    __atomic_store_n(&ctr[kFuDn], 0u, __ATOMIC_RELAXED);
    for (int j = 0; j < 256; ++j) __atomic_store_n(&ctr[kFuArr + j], 0u, __ATOMIC_RELAXED);
    __atomic_store_n(&ctr[kFuPd], 0u, __ATOMIC_RELAXED);
    __atomic_store_n(&ctr[kFuPd + 1], 0u, __ATOMIC_RELAXED);
    if (EXL3R4D_GD_PX)
      for (int j = 0; j < nlive && j < kFuMaxLive; ++j) __atomic_store_n(&ctr[kFuPd + 2 + j], 0u, __ATOMIC_RELAXED);
  }
}

extern "C" int exl3r4d_launch_fused(const Exl3r4dGemm* gu, const Exl3r4dGemm* dn, unsigned* ctr, int workgroups,
                                    int prep, void* stream) {
  Exl3r4dFused f = {};
  f.gu = *gu;
  for (int c = 0; c < 2; ++c) {
    f.dnd.cls[c].codes = dn->cls[c].codes;
    f.dnd.cls[c].suh = dn->cls[c].suh;
    f.dnd.cls[c].svh = dn->cls[c].svh;
    f.dnd.cls[c].codes_s0 = dn->cls[c].codes_s0;
    f.dnd.cls[c].suh_s0 = dn->cls[c].suh_s0;
    f.dnd.cls[c].svh_s0 = dn->cls[c].svh_s0;
    f.dnd.cls[c].rows = dn->cls[c].rows;
    f.dnd.cls[c].cols = dn->cls[c].cols;
  }
  f.dnd.y = dn->y;
  f.dnd.y_pitch = dn->y_pitch;
  f.dnd.y_cols = dn->y_cols;
  f.dnd.sk = dn->sk;
  {   // the rebuild must be the view fill() made: every field compared (the struct's padding is not a field)
    const Exl3r4dGemm r = dn_view(*gu, f.dnd);
    bool same = r.x == dn->x && r.y == dn->y && r.sorted == dn->sorted && r.off == dn->off && r.xr == dn->xr &&
                r.x_pitch == dn->x_pitch && r.y_pitch == dn->y_pitch && r.xr_pitch == dn->xr_pitch && r.T == dn->T &&
                r.n_tok == dn->n_tok && r.n_expert == dn->n_expert && r.top_k == dn->top_k &&
                r.n_class == dn->n_class && r.y_cols == dn->y_cols && r.x_bf16 == dn->x_bf16 && r.sk == dn->sk &&
                r.chunk == dn->chunk && r.nslot == dn->nslot && r.h_rot == dn->h_rot && r.a_rot == dn->a_rot &&
                r.ctr == dn->ctr && r.tpw == dn->tpw && r.raw == dn->raw && r.raw_pitch == dn->raw_pitch;
    for (int c = 0; c < 2; ++c) {
      const Exl3r4dCls &a = r.cls[c], &b = dn->cls[c];
      same = same && a.codes == b.codes && a.suh == b.suh && a.suh1 == b.suh1 && a.svh == b.svh &&
             a.dn_suh == b.dn_suh && a.codes_s0 == b.codes_s0 && a.suh_s0 == b.suh_s0 && a.suh1_s0 == b.suh1_s0 &&
             a.svh_s0 == b.svh_s0 && a.dn_suh_s0 == b.dn_suh_s0 && a.rows == b.rows && a.cols == b.cols;
    }
    if (!same) return RAD_E_UNSUPPORTED;   // the caller takes the separate kernels
  }
  f.ctr = ctr;
  f.prep = prep;
  static int nwgp_dev[16];   // per device, once (a host query a launch was measurable at M=1)
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 16) dev = 0;
  if (nwgp_dev[dev] <= 0) {
    int n = 0;
    (void)hipDeviceGetAttribute(&n, hipDeviceAttributeMultiprocessorCount, dev);
    nwgp_dev[dev] = n > 0 ? n : 1;
  }
  f.nwgp = nwgp_dev[dev];
  const int64_t cols_max = gu->cls[0].cols > gu->cls[1].cols ? gu->cls[0].cols : gu->cls[1].cols;
  const bool ap = gu->xr == nullptr;   // the rows rotated in each gate_up unit's LDS (gu->chunk a pass)
  if (ap && prep) return RAD_E_INVAL;
  int64_t smem = exl3r4d_gemm_lds(1, 2, gu->sk, ap ? gu->chunk : 0, cols_max);
  if (smem < 0) return (int)smem;
  if (smem < kHrotStage + 4096) smem = kHrotStage + 4096;
  if (smem < 2 * 8192) smem = 2 * 8192;  // down's two halves' epilogue tiles
  const void* fn = ap ? (const void*)exl3r4d_fused_kernel<true> : (const void*)exl3r4d_fused_kernel<false>;
  if (workgroups <= 0) {
    workgroups = wq_capacity(fn, 256, (size_t)smem, f.nwgp);
    // EXL3R4D_GD_WGS: percent of the resident capacity the persistent launch takes, so independent work the engine
    // issues beside it (the shared expert's GEMMs) finds slots; the tickets are dealt dynamically, so fewer workers
    // finish the same units. 100: every slot.
    if (EXL3R4D_GD_WGS < 100 && workgroups > 0) workgroups = workgroups * EXL3R4D_GD_WGS / 100 > f.nwgp
                                                                ? workgroups * EXL3R4D_GD_WGS / 100 : f.nwgp;
  }
#ifdef EXL3R4D_GD_DEBUG   // reading builds: why a fused launch failed
  if (workgroups <= 0) {
    hipFuncAttributes at;
    const hipError_t e = hipFuncGetAttributes(&at, fn);
    fprintf(stderr, "exl3r4d fused: no capacity (smem %lld nwgp %d attr %s regs %d shared %zu)\n", (long long)smem,
            f.nwgp, hipGetErrorString(e), e == hipSuccess ? at.numRegs : -1, e == hipSuccess ? at.sharedSizeBytes : 0);
  }
#endif
  if (workgroups <= 0) return RAD_E_DEVICE;
  void* args[] = {&f};
  const hipError_t le = hipLaunchKernel(fn, dim3((unsigned)workgroups), dim3(256), args, (size_t)smem, (hipStream_t)stream);
#ifdef EXL3R4D_GD_DEBUG
  if (le != hipSuccess)
    fprintf(stderr, "exl3r4d fused: launch %s (wg %d smem %lld)\n", hipGetErrorString(le), workgroups, (long long)smem);
#endif
  if (le != hipSuccess)
    return RAD_E_DEVICE;
  return RAD_OK;
}

#endif   // EXL3R4D_NCLS == 2: the fused gate_up_down launch

// The prep kernel: every row of every live run rotated into xr (f16 [T][xr_pitch], matrix m at m * cols), one
// wave per (row, matrix, 128-block); dropped rows are zeros. One workgroup per live expert (find_run's numbering).
// EXL3R4D_PREPGRID (lever L9): 1 = a 2-D grid, y over live experts and x over groups of a row's (matrix, 128-block)
// tasks, two waves a workgroup, every row of the run a wave's loop -- all of a step's tasks in one round; 0 = one
// 256-thread workgroup a live expert striding over every task (as ported: ~5 serial rounds at gate_up M=1).
#ifndef EXL3R4D_PREPGRID
#define EXL3R4D_PREPGRID 0
#endif
// EXL3R4D_PREP_ABL (P5b, reading only -- timing, wrong output): the prep kernel returns at entry (1), after find_run
// (2), after its planes (3); 4 runs every task with lane-made x / suh values instead of their loads.
#ifndef EXL3R4D_PREP_ABL
#define EXL3R4D_PREP_ABL 0
#endif
template <int MATS>
__global__ void __launch_bounds__(256) exl3r4d_prep_kernel(Exl3r4dGemm p) {
  if constexpr (EXL3R4D_PREP_ABL == 1) return;
  inject_delay<EXL3R4D_DLY_PG>(EXL3R4D_WG0);
  __shared__ int s_find[35];  // find_run_regs also hands r0, r1 through [33], [34]
  const Run run = find_run_any(p, EXL3R4D_PREPGRID ? blockIdx.y : blockIdx.x, s_find);
  if (run.e < 0) return;
  if constexpr (EXL3R4D_PREP_ABL == 2) {
    if (threadIdx.x == 0 && run.r1 < run.r0) *(int*)p.xr = run.e;   // never true: keeps the search
    return;
  }
  const Planes w = planes_of(p, run.e);
  if constexpr (EXL3R4D_PREP_ABL == 3) {
    if (threadIdx.x == 0 && (w.suh == nullptr || w.suh1 == nullptr)) *(int*)p.xr = run.e;   // keeps the loads
    return;
  }
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
    v2u o;
    if constexpr (EXL3R4D_PREP_ABL == 4) {
      const int t = p.sorted[r];
      o = t >= 0 ? rotate128_math((v2u){(unsigned)lane * 0x3C013C01u, (unsigned)(blk + m)},
                                  (v2u)(0x3C003C00u), p.x_bf16 == 0, lane) : (v2u)(0u);
    } else {
      o = prep_rotate<MATS>(p, w, r, m, blk, lane);
    }
    *(v2u*)((_Float16*)p.xr + (size_t)r * p.xr_pitch + (size_t)m * w.cols + blk * 128 + 4 * lane) = o;
  }
}

// The prep_gu op's 2-D form (L9, EXL3R4D_PREPOP_GRID in launch.c): a kernel of its own so the one above -- gate_up's
// own prep and the M > 16 band's -- is untouched. y over live experts, x over groups of a row's (matrix, 128-block)
// tasks, two waves a workgroup, every row of the run a wave's loop: all of a step's tasks in one round. Same bits.
template <int MATS>
__global__ void __launch_bounds__(64) exl3r4d_prep_grid_kernel(Exl3r4dGemm p) {
  inject_delay<EXL3R4D_DLY_PG>(EXL3R4D_WG0);
  __shared__ int s_find[35];
  const Run run = find_run_any(p, blockIdx.y, s_find);
  if (run.e < 0) return;
  const Planes w = planes_of(p, run.e);
  const int lane = threadIdx.x % WAVE32, nwaves = (int)blockDim.x / WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane((int)(threadIdx.x / WAVE32));
  const int blocks = w.cols / 128, per_row = MATS * blocks, tasks = (run.r1 - run.r0) * per_row;
  const int g = (int)blockIdx.x * nwaves + wave;
  if (g >= per_row) return;
  for (int tsk = g; tsk < tasks; tsk += per_row) {
    const int i = tsk / per_row, rem = tsk - i * per_row, m = rem / blocks, blk = rem - m * blocks;
    const int r = run.r0 + i;
    const v2u o = prep_rotate<MATS>(p, w, r, m, blk, lane);
    *(v2u*)((_Float16*)p.xr + (size_t)r * p.xr_pitch + (size_t)m * w.cols + blk * 128 + 4 * lane) = o;
  }
}

// P5b (EXL3R4D_PREP_ROW): the prep as one 256-thread workgroup a SORTED ROW (grid T) and no run search -- each wave
// counts the run table's offsets at or below its row in one round of unconditional loads (expert = that count - 1:
// the offsets are non-decreasing), the row's token loaded beside them; then the expert's planes; then every x and suh
// load of the wave's tasks (one matrix, every 8 / MATS-th 128-block) before any rotation. Three dependent rounds where
// the live-slot form takes about twelve (traced, tp2 M=4: find_run 2.2 of its 5.8 us, the task loop 2.2-3.3). Same
// rows, same rotate128_math: bit-identical; a row no live run covers is left as it is, as before.
template <int MATS>
__global__ void __launch_bounds__(256) exl3r4d_prep_row_kernel(Exl3r4dGemm p) {
  (void)prep_row<MATS>(p, blockIdx.x, __builtin_amdgcn_readfirstlane((int)(threadIdx.x / WAVE32)),
                       threadIdx.x % WAVE32);
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
  inject_delay<EXL3R4D_DLY_PG>(EXL3R4D_WG0);
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
  __shared__ int s_find[35];  // find_run_regs also hands r0, r1 through [33], [34]
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
  __shared__ int s_find[35];  // find_run_regs also hands r0, r1 through [33], [34]
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

#if EXL3R4D_NCLS == 2   // the library's class-free host entries: once, in the two-class build (the other calls these)
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
  if (!gate_up && EXL3R4D_EPIB) need = 2 * 8 * 256 * 4;  // EPIB: two exchange buffers
  const int64_t red = red_bytes(wv, npw, sk), rows = (int64_t)fused_rows * (mats * cols_max + EXL3R4D_APAD) * 2;
  need = red > need ? red : need;
  need = rows > need ? rows : need;
  return need > EXL3R4D_LDS_MAX ? RAD_E_SHAPE : need;
}
#endif

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

#if EXL3R4D_NCLS == 2
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
#endif

#if EXL3R4D_WGQ
extern "C" int exl3r4d_wgp_probe(int* map_dev) {  // instance init only (it synchronises)
  unsigned* bits = nullptr;
  void* pargs[] = {&bits};
  constexpr int kWords = EXL3R4D_WGQ_KEYS / 32;
  if (hipMalloc((void**)&bits, kWords * sizeof(unsigned)) != hipSuccess) return RAD_E_DEVICE;
  int rc = RAD_E_DEVICE;
  unsigned hb[kWords];
  int map[EXL3R4D_WGQ_KEYS];
  if (hipMemset(bits, 0, kWords * sizeof(unsigned)) == hipSuccess &&
      hipLaunchKernel((const void*)exl3r4d_wgp_probe_kernel, dim3(4096), dim3(64), pargs, 0, nullptr) == hipSuccess &&
      hipDeviceSynchronize() == hipSuccess &&
      hipMemcpy(hb, bits, sizeof hb, hipMemcpyDeviceToHost) == hipSuccess) {
    int n = 0;
    for (int k = 0; k < EXL3R4D_WGQ_KEYS; ++k) map[k] = (hb[k >> 5] >> (k & 31)) & 1 ? n++ : -1;
    rc = n > 0 && hipMemcpy(map_dev, map, sizeof map, hipMemcpyHostToDevice) == hipSuccess ? n : RAD_E_DEVICE;
  }
  (void)hipFree(bits);
  return rc;
}
#endif

#if EXL3R4D_WGSTAMP  // the reading's two entries (perfmoe --wgstamps finds them by dlsym; absent from shipped builds)
extern "C" __attribute__((visibility("default"))) int exl3r4d_wgstamp_clear(void* stream) {
  void *t = nullptr, *h = nullptr;
  if (hipGetSymbolAddress(&t, HIP_SYMBOL(g_wgs_t)) != hipSuccess ||
      hipGetSymbolAddress(&h, HIP_SYMBOL(g_wgs_hw)) != hipSuccess)
    return RAD_E_DEVICE;
  if (hipMemsetAsync(t, 0, sizeof g_wgs_t, (hipStream_t)stream) != hipSuccess ||
      hipMemsetAsync(h, 0, sizeof g_wgs_hw, (hipStream_t)stream) != hipSuccess)
    return RAD_E_DEVICE;
  return RAD_OK;
}
// op 1 gate_up, 0 down: n entries of {entry, exit, hw} as three u64 each
extern "C" __attribute__((visibility("default"))) int exl3r4d_wgstamp_read(int op, unsigned long long* out, int n) {
  static unsigned long long t[2][kWgsMax][2];
  static unsigned hw[2][kWgsMax];
  if (n > kWgsMax || hipMemcpyFromSymbol(t, HIP_SYMBOL(g_wgs_t), sizeof t) != hipSuccess ||
      hipMemcpyFromSymbol(hw, HIP_SYMBOL(g_wgs_hw), sizeof hw) != hipSuccess)
    return RAD_E_DEVICE;
  for (int i = 0; i < n; ++i) {
    out[3 * i] = t[op][i][0];
    out[3 * i + 1] = t[op][i][1];
    out[3 * i + 2] = hw[op][i];
  }
  return RAD_OK;
}
#endif

// grid2d: 1 = the 2-D form (L9) for this launch, 0 = the build's (EXL3R4D_PREPGRID)
extern "C" int exl3r4d_launch_prep_form(const Exl3r4dGemm* a, int gate_up, void* stream, int grid2d) {
  if (!a->xr) return RAD_E_INVAL;
  Exl3r4dGemm p = *a;
  void* args[] = {&p};
  hipStream_t st = (hipStream_t)stream;
  if (EXL3R4D_PREP_ROW && a->n_tok >= EXL3R4D_PREP_ROW_MIN) {   // P5b: a workgroup a sorted row
    const void* pr = gate_up ? (const void*)exl3r4d_prep_row_kernel<2> : (const void*)exl3r4d_prep_row_kernel<1>;
    if (a->T > 0 && hipLaunchKernel(pr, dim3((unsigned)a->T), dim3(256), args, 0, st) != hipSuccess)
      return RAD_E_DEVICE;
    return RAD_OK;
  }
  if (grid2d && !EXL3R4D_PREPGRID && !EXL3R4D_PREPE) {
    const void* pg = gate_up ? (const void*)exl3r4d_prep_grid_kernel<2> : (const void*)exl3r4d_prep_grid_kernel<1>;
    const int64_t cm = EXL3R4D_CLS_MAX(*a, cols);
    const unsigned groups = (unsigned)(((gate_up ? 2 : 1) * (cm / 128) + 1) / 2);   // two waves a WG
    if (hipLaunchKernel(pg, dim3(groups, (unsigned)a->nslot), dim3(2 * WAVE32), args, 0, st) != hipSuccess)
      return RAD_E_DEVICE;
    return RAD_OK;
  }
  const void* pf = gate_up ? (const void*)exl3r4d_prep_kernel<2> : (const void*)exl3r4d_prep_kernel<1>;
  if (EXL3R4D_PREPE) {  // P1f: a workgroup an expert
    const void* pe = gate_up ? (const void*)exl3r4d_prep_e_kernel<2> : (const void*)exl3r4d_prep_e_kernel<1>;
    if (hipLaunchKernel(pe, dim3((unsigned)a->n_expert), dim3(256), args, 0, st) != hipSuccess) return RAD_E_DEVICE;
    return RAD_OK;
  }
#if EXL3R4D_PREPGRID
  const int64_t cols_max = EXL3R4D_CLS_MAX(*a, cols);
  const unsigned groups = (unsigned)(((gate_up ? 2 : 1) * (cols_max / 128) + 1) / 2);   // two waves a WG
  if (hipLaunchKernel(pf, dim3(groups, (unsigned)a->nslot), dim3(2 * WAVE32), args, 0, st) != hipSuccess)
    return RAD_E_DEVICE;
#else
  if (hipLaunchKernel(pf, dim3(a->nslot), dim3(256), args, 0, st) != hipSuccess) return RAD_E_DEVICE;
#endif
  return RAD_OK;
}
extern "C" int exl3r4d_launch_prep(const Exl3r4dGemm* a, int gate_up, void* stream) {
  return exl3r4d_launch_prep_form(a, gate_up, stream, 0);
}

// WGB: workgroups of `fn` the device holds at once (occupancy a CU x CUs), per device and launch shape; 0 if the
// runtime cannot say. Host-side queries, cached: no allocation, no synchronisation.
#ifndef EXL3R4D_WGQ_CAP
#define EXL3R4D_WGQ_CAP 0
#endif
#ifndef EXL3R4D_PDN_CAP  // TEST builds: at most this many persistent down workgroups
#define EXL3R4D_PDN_CAP 0
#endif
// Workgroups of `fn` resident at once on `nwgp` WGPs, from its VGPR count (cached a shape): gfx12 wave32 has 1536
// VGPRs a SIMD lane-slice, allocated in 8s, at most 16 waves a SIMD, 4 SIMDs a WGP, and 128 KiB of LDS a WGP in WGP
// mode (hipOccupancy* assumes 64 KiB a multiprocessor there and under-counts: WGB's G = 96 against 128 resident).
static int wq_capacity(const void* fn, int threads, size_t smem, int nwgp) {
  static const void* c_fn[4];
  static int c_thr[4], c_cap[4];
  static size_t c_smem[4];
  for (int i = 0; i < 4; ++i)
    if (c_fn[i] == fn && c_cap[i] > 0 && c_thr[i] == threads && c_smem[i] == smem) return c_cap[i] * nwgp;
  hipFuncAttributes at;
  if (hipFuncGetAttributes(&at, fn) != hipSuccess || at.numRegs <= 0) return 0;
  const int regs = (at.numRegs + 7) / 8 * 8, wps = 1536 / regs < 16 ? 1536 / regs : 16;
  const int wpg = threads / WAVE32, by_vgpr = wps * 4 / wpg;
  const size_t lds_wg = smem + (size_t)at.sharedSizeBytes;
  const int by_lds = lds_wg > 0 ? (int)((128 * 1024) / lds_wg) : by_vgpr;
  int i = 0;   // the first free or matching entry, else the last
  while (i < 3 && c_fn[i] != nullptr && c_fn[i] != fn) ++i;
  c_fn[i] = fn;
  c_thr[i] = threads;
  c_smem[i] = smem;
  c_cap[i] = by_vgpr < by_lds ? by_vgpr : by_lds;
  return c_cap[i] * nwgp;
}
#if EXL3R4D_WGB
static int resident_wgs(const void* fn, int threads, size_t smem) {
  static int cache_thr[16], cache_g[16];
  static size_t cache_smem[16];
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 16) return 0;
  if (cache_g[dev] > 0 && cache_thr[dev] == threads && cache_smem[dev] == smem) return cache_g[dev];
  int per_cu = 0, cus = 0;
  if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, fn, threads, smem) != hipSuccess ||
      hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess)
    return 0;
  cache_thr[dev] = threads;
  cache_smem[dev] = smem;
  cache_g[dev] = per_cu * cus;
  return cache_g[dev];
}
#endif

template <int BITS, int MATS>
static const void* gemm_raw_fn(int npw) {
  switch (npw) {
    case 1: return (const void*)exl3r4d_gemm_kernel<BITS, 1, MATS, false, true>;
    case 2: return (const void*)exl3r4d_gemm_kernel<BITS, 2, MATS, false, true>;
    case 4: return (const void*)exl3r4d_gemm_kernel<BITS, 4, MATS, false, true>;
    default: return nullptr;
  }
}

// R23: K = 2, 3, 5, 6 -- the shipped configuration's kernels only (NPW 2, the prep kernel's rows, down's P1c form),
// instantiated per K; every lever toggle stays K = 4's and refuses another K by name (RAD_E_UNSUPPORTED).
template <int B>
static const void* kx_fn(bool gate_up, bool gscr) {
  if (gate_up) {
    if constexpr (EXL3R4D_GSCR_GU) {
      if (gscr) return (const void*)exl3r4d_gemm_kernel<B, 2, 2, false, false, 4>;
    }
    return (const void*)exl3r4d_gemm_kernel<B, 2, 2, false>;
  }
  if constexpr (EXL3R4D_GSCR_DN) {
    if (gscr) return (const void*)exl3r4d_gemm_kernel<B, 2, 1, false, false, 4>;
  }
  return (const void*)exl3r4d_gemm_kernel<B, 2, 1, false>;
}

#if EXL3R4D_NCLS == 2   // a class-free host entry: once, in the two-class build (the four-class kernels are its twins, twin_check)
// L7 (EXL3R4D_SK_FIT, launch.c): workgroups of the decode gate_up GEMM (K = bits, npw tiles a wave, split-K sk, the
// prep kernel's rows) resident at once on the current device -- wq_capacity's model times the WGPs; 0 when unknown.
extern "C" int exl3r4d_gu_resident(int bits, int npw, int sk, int h_rot) {
  const void* fn = nullptr;
  switch (bits) {
    case 2: fn = kx_fn<2>(true, false); break;
    case 3: fn = kx_fn<3>(true, false); break;
    case 4: fn = gemm_fn<4, 2>(npw, false); break;
    case 5: fn = kx_fn<5>(true, false); break;
    case 6: fn = kx_fn<6>(true, false); break;
    default: return 0;
  }
  int64_t smem = exl3r4d_gemm_lds(1, npw, sk, 0, 0);
  if (!fn || smem < 0) return 0;
  if (h_rot && smem < kHrotStage + 4096) smem = kHrotStage + 4096;  // as exl3r4d_launch_gemm sizes it
  static int nwgp_dev[16];
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 16) return 0;
  if (nwgp_dev[dev] <= 0) {
    int n = 0;
    (void)hipDeviceGetAttribute(&n, hipDeviceAttributeMultiprocessorCount, dev);
    nwgp_dev[dev] = n;
  }
  return nwgp_dev[dev] > 0 ? wq_capacity(fn, 8 * 2 / npw * sk * WAVE32, (size_t)smem, nwgp_dev[dev]) : 0;
}
#endif

extern "C" int exl3r4d_launch_gemm(const Exl3r4dGemm* a, int gate_up, int bits, int npw, void* stream) {
  if (bits < 2 || bits > 6) return RAD_E_DTYPE;  // R23: exl3k2 .. exl3k6; refused by name upstream too
  const int mats = gate_up ? 2 : 1, wv = 8 * mats / npw;
  const bool aprep = a->xr == nullptr;
  const int64_t cols_max = EXL3R4D_CLS_MAX(*a, cols);
  int64_t smem = exl3r4d_gemm_lds(gate_up, npw, a->sk, aprep ? a->chunk : 0, cols_max);
  if (smem < 0) return (int)smem;
  if (gate_up && a->h_rot && smem < kHrotStage + 4096) smem = kHrotStage + 4096;  // v3 EPI 1's staged h block
  // the input arrives rotated -- down reads gate_up's h' (v3 h_rot), gate_up reads prep_gu's xr (v4 a_rot): no prep
  const bool hrot_dn = (!gate_up && a->h_rot) || (gate_up && a->a_rot);
  // K = 4's kernel for this NPW (every lever form); another K picks its own below (kx_fn)
  const void* fn = bits != 4 ? nullptr : gate_up ? gemm_fn<4, 2>(npw, aprep) : gemm_fn<4, 1>(npw, aprep);
  if (bits == 4 && !fn) return RAD_E_SHAPE;
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
  const bool fuse1 = EXL3R4D_FUSE1 && gate_up && !hrot_dn && a->ctr && !aprep && !a->raw && !staged && !gscr && !rot &&
                     !EXL3R4D_STREAM;
  if (bits != 4 && (aprep || a->raw || staged || rot || fuse1 || EXL3R4D_STREAM || EXL3R4D_ABLATE_PREPK || npw != 2))
    return RAD_E_UNSUPPORTED;  // R23: another K has the shipped configuration's kernels only (kx_fn)
  if (!aprep && !rot && !gscr && !hrot_dn && !fuse1 && !EXL3R4D_ABLATE_PREPK) {  // ABLATE_PREPK (timing only, wrong output): the prep kernel's cost
    const int rc = exl3r4d_launch_prep(a, gate_up, stream);
    if (rc != RAD_OK) return rc;
  }
  // DNW: down's plain K loop, two column blocks a workgroup (its own kernel instantiation; every class as wide as y
  // and an even block count, else the one-block form -- the same bits either way)
  const int64_t ncb_y = a->y_cols / 128;
  const bool dnw = EXL3R4D_DNW > 1 && !gate_up && !aprep && !a->raw && !staged && !rot && !gscr && !fuse1 &&
                   !EXL3R4D_STREAM && EXL3R4D_DNG == 1 && ncb_y % kDnwB == 0 && a->cls[0].rows == a->y_cols &&
                   a->cls[1].rows == a->y_cols && npw == 2;
  if constexpr (EXL3R4D_DNW > 1) if (dnw) {
    const void* df = bits == 2 ? (const void*)exl3r4d_gemm_kernel<2, 2, 1, false, false, 0, true>
                   : bits == 3 ? (const void*)exl3r4d_gemm_kernel<3, 2, 1, false, false, 0, true>
                   : bits == 4 ? (const void*)exl3r4d_gemm_kernel<4, 2, 1, false, false, 0, true>
                   : bits == 5 ? (const void*)exl3r4d_gemm_kernel<5, 2, 1, false, false, 0, true>
                               : (const void*)exl3r4d_gemm_kernel<6, 2, 1, false, false, 0, true>;
    const size_t dsm = (size_t)(kDnwB * smem);
    if (dsm > EXL3R4D_LDS_MAX) return RAD_E_SHAPE;
    if (hipLaunchKernel(df, dim3((unsigned)(ncb_y / kDnwB), (unsigned)a->nslot, 1),
                        dim3((unsigned)(kDnwB * wv * a->sk * WAVE32)), args, dsm, st) != hipSuccess)
      return RAD_E_DEVICE;
    return RAD_OK;
  }
  // EXL3R4D_MSPLIT_ABL (reading, TIMING ONLY, wrong output): gate_up with its two matrices in separate workgroups --
  // the one-matrix (down-form) kernel over gate_up's stacked [gate; up] rows, twice the column blocks, each a 4-wave
  // workgroup of one matrix's 8 tiles over the whole K, its epilogue into the prep rows (no exchange, no SiLU). The
  // upper bound of a mat-split gate_up at small M (more workgroups over the WGPs, no K split, no reduction).
  if constexpr (EXL3R4D_MSPLIT_ABL) if (gate_up && bits == 4 && !aprep && !a->raw && !fuse1 && !staged && !gscr) {
    Exl3r4dGemm q = *a;
    q.y = a->xr;   // scratch: T x 2K f16 holds T x 2w bf16
    q.y_pitch = 2 * a->y_cols;
    q.y_cols = 2 * a->y_cols;
    q.h_rot = 0;
    void* qa[] = {&q};
    const int64_t qsm = exl3r4d_gemm_lds(0, 2, a->sk, 0, cols_max);
    if (hipLaunchKernel((const void*)exl3r4d_gemm_kernel<4, 2, 1, false>, dim3((unsigned)(2 * a->y_cols / 128),
                        (unsigned)a->nslot, 1), dim3((unsigned)(4 * a->sk * WAVE32)), qa, (size_t)qsm, st) != hipSuccess)
      return RAD_E_DEVICE;
    return RAD_OK;
  }
  if (bits != 4) {
    const int dngk = gate_up ? 1 : EXL3R4D_DNG;
    const dim3 gk((unsigned)((a->y_cols / 128 + dngk - 1) / dngk), (unsigned)a->nslot, 1);
    const void* kf = bits == 2 ? kx_fn<2>(gate_up, gscr) : bits == 3 ? kx_fn<3>(gate_up, gscr)
                   : bits == 5 ? kx_fn<5>(gate_up, gscr) : kx_fn<6>(gate_up, gscr);
    if (hipLaunchKernel(kf, gk, dim3((unsigned)(wv * a->sk * WAVE32)), args, (size_t)smem, st) != hipSuccess)
      return RAD_E_DEVICE;
    return RAD_OK;
  }
  if (a->raw) {  // the split epilogue: GEMM workgroups of p.tpw tiles of one matrix, then the finisher
    if (aprep || a->tpw < npw || a->tpw % npw) return RAD_E_SHAPE;
    const int wvr = a->tpw / npw;
    int64_t tiles = 0;
    for (int c = 0; c < EXL3R4D_NCLS; ++c) {
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
  const dim3 grid((unsigned)((a->y_cols / 128 + dng - 1) / dng), (unsigned)a->nslot,
                  gate_up && EXL3R4D_XSK_ABL > 1 ? (unsigned)EXL3R4D_XSK_ABL : 1u);
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
#if EXL3R4D_PDN
  if (!gate_up && hrot_dn && npw == 2 && grid.z == 1 && a->ctr) {
    const void* df = (const void*)exl3r4d_gemm_pdn_kernel<4, 2, 1>;
    int nwgp = 0, dev = 0;
    if (hipGetDevice(&dev) == hipSuccess) (void)hipDeviceGetAttribute(&nwgp, hipDeviceAttributeMultiprocessorCount, dev);
    const int cap0 = nwgp > 0 ? wq_capacity(df, (int)block.x, (size_t)smem, nwgp) : 0;
    const int cap = EXL3R4D_PDN_CAP > 0 && cap0 > EXL3R4D_PDN_CAP ? EXL3R4D_PDN_CAP : cap0;   // TEST builds
    // persistent only past 3 rounds of the grid -- measured always on: M=8/16 -4.7 us, M=4 +0.2..+1.0, M=1/2
    // +1.3/+1.9 (runs/p4b-s12); a 2-round boundary still took M=4 (~1.8 rounds) persistent, +1.0 (runs/p4b-s13) --
    // and only while every live expert fits the LDS list
    const int64_t live_max = a->nslot - 1, units_max = (int64_t)grid.x * live_max;
    if (cap > 0 && live_max <= kPdnMaxLive && (units_max > 3 * (int64_t)cap0 || EXL3R4D_PDN_CAP > 0)) {
      if (hipLaunchKernel(df, dim3((unsigned)cap), block, args, (size_t)smem, st) != hipSuccess) return RAD_E_DEVICE;
      return RAD_OK;
    }
  }
#endif
#if EXL3R4D_WGQ
  if (gate_up && !aprep && npw == 2 && grid.z == 1 && a->wq && a->wq_map && a->wq_nl > 0) {
    const void* qf = (const void*)exl3r4d_gemm_wq_kernel<4, 2, 2>;
    const int cap0 = wq_capacity(qf, (int)block.x, (size_t)smem, a->wq_nl);
    // TEST builds (EXL3R4D_WGQ_CAP): at most this many workgroups, so small suite shapes take this path -- and, with
    // fewer workgroups than WGPs, the steal path
    const int cap = EXL3R4D_WGQ_CAP > 0 && cap0 > EXL3R4D_WGQ_CAP ? EXL3R4D_WGQ_CAP : cap0;
    if (cap > 0 && (int64_t)grid.x * (a->nslot - 1) > cap) {  // live units can exceed the resident slots
      if (hipLaunchKernel(qf, dim3((unsigned)cap), block, args, (size_t)smem, st) != hipSuccess) return RAD_E_DEVICE;
      return RAD_OK;
    }
  }
#endif
#if EXL3R4D_WGB
  if (gate_up && !aprep && npw == 2 && grid.z == 1) {  // WGB: more units than resident slots
    const void* pf = (const void*)exl3r4d_gemm_pu_kernel<4, 2, 2>;
    const int r = resident_wgs(pf, (int)block.x, (size_t)smem);
    const int g = EXL3R4D_WGB_CAP > 0 && r > EXL3R4D_WGB_CAP ? EXL3R4D_WGB_CAP : r;
    if (g > 0 && (int64_t)grid.x * grid.y > g) {
      if (hipLaunchKernel(pf, dim3((unsigned)g), block, args, (size_t)smem, st) != hipSuccess) return RAD_E_DEVICE;
      return RAD_OK;
    }
  }
#endif
  if (hipLaunchKernel(fn, grid, block, args, (size_t)smem, st) != hipSuccess) return RAD_E_DEVICE;
  return RAD_OK;
}

#if EXL3R4D_PDN
extern "C" int exl3r4d_debug_capacity(int* regs, int* shared, int* nwgp) {
  hipFuncAttributes at;
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess || hipFuncGetAttributes(&at, (const void*)exl3r4d_gemm_pdn_kernel<4, 2, 1>) != hipSuccess)
    return -1;
  *regs = at.numRegs;
  *shared = (int)at.sharedSizeBytes;
  (void)hipDeviceGetAttribute(nwgp, hipDeviceAttributeMultiprocessorCount, dev);
  return wq_capacity((const void*)exl3r4d_gemm_pdn_kernel<4, 2, 1>, 128, 8192, *nwgp);
}
#endif

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
//
// P9-L7 (2026-10-14; card 0000:03:00.0, stock mclk, stream 632.5 GB/s; perfmoe, libr4d 1.5.0 f27fd398 same session;
// tables and raw logs: the plan folder's notes/p9-l7.md). THE DECODE BAND AT M = 2..4.
// READING 1, split chain (prep_gu -> gate_up a_rot 1 -> down) vs libr4d's, us a layer: tp1 M=2 104.2 / 103.1, M=3 145.6
//   / 142.8, M=4 177.1 / 179.7, M=4 MTP-verify routing (4-token runs, 25.9 live) 133.1 / 130.1; tp2 M=2 59.7 / 60.7, M=3
//   80.6 / 78.7, M=4 100.8 / 100.5, M=4 verify 77.4 / 72.2. Fused gate_up_down (0 bytes differ): tp2 M=3 75.4, M=4 verify
//   69.8; tp1 M=3 138.6, M=4 verify 126.5 -- ahead of both. In the split form tp2's DOWN is 12-15 % behind at M >= 3.
// READING 5, size sweeps: we STREAM as fast or faster (down tp2 M=4 ~640 vs 547 GB/s, gate_up tp2 M=4 verify 565 vs 526)
//   and lose on the intercept (down 10.3 vs 1.7 us, gate_up 6.3 vs 3.75): fixed cost a dispatch and ROUNDS.
// READING W (WGSTAMP): second rounds of latency-bound units -- tp1 M=2 gate_up at split-K 2 (64 sixteen-wave slots, ~98
//   units), tp1 M=4 verify (128 slots, 130-145 units: a 17-unit tail, +17 us), tp2 down (~450 slots, 580-800 units).
// READING 4 (deletions, K loops checked statically first): decode <= 10 % at M=1..6; A loads free; the expert search ~1 us
//   of a gate_up dispatch, 2.3-3.7 of a down one. WLOAD / EPI / EXIT_STAGE 4 re-schedule the K loop (33-38 s_delay_alu vs
//   2): void as deletions.
// LEVERS: L7-1 EXL3R4D_SK_FIT (launch.c) ADOPTED -- tp1 M=2 chain 104.8 -> 98.6 (numerics: summation order, accepted);
//   L7-2 EXL3R4D_FRP (exl3r4d_common.hip.h) REFUTED +1..2 us; H5 waves_per_eu 10 (5 WGs a WGP) REFUTED (spills: K loop
//   re-scheduled, fused +6..+27 us); EXL3R4D_ALIGN_PAGE (CMakeLists.txt) ADOPTED -- instruction-identical builds read the
//   tp1 M=3 chain 146.0 vs 155.2 by where the kernels landed; 4 KB alignment read 146.0.
// LEFT: the fused prologue (~7-8 us before any weight streams at M=4..16: 4-5 dependent round trips), the fused tail at tp1
//   M=4 verify (128-slot quantisation), prep_gu's dependent dispatch at M=1..3 (L3), split down at tp4 (K a rank 160).

// The four-class build (EXL3R4D_NCLS 4, exl3r4d_mod4.hip) serves the shipped configuration: the reading arms that read
// a class outside planes_of / EXL3R4D_CLS_F, or keep state of their own, are two-class only.
#if EXL3R4D_NCLS == 4 && (EXL3R4D_WGQ || EXL3R4D_PDN || EXL3R4D_WGSTAMP || EXL3R4D_FUSED_PROTO || EXL3R4D_STAGED || \
                          EXL3R4D_DNW > 1 || EXL3R4D_WGB || EXL3R4D_SUHC)
#error "the four-class build (EXL3R4D_NCLS 4) takes none of the reading arms WGQ, PDN, WGSTAMP, FUSED_PROTO, STAGED, DNW, WGB, SUHC"
#endif
