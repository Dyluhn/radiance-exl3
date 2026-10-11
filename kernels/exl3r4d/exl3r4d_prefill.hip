// exl3r4d_prefill.hip -- the EXL3 routed-expert GEMMs (moe_exl3_gate_up, moe_exl3_down) on gfx1201, prefill form of
// the M > 16 band. Owner: P5. The decode band and every shared device helper are P4's (exl3r4d.hip,
// exl3r4d_common.hip.h); this file adds the run map and the MTX GEMM and nothing else.
//
// LINEAGE. A port of the MTX 4 / 8 path of Dylan's exl3-moe kernel (exl3_moe/kernels/gfx1201/exl3_moe_wmma.hip,
// moe_body<MT, MTX, ...> and moe_dispatch, :555-967), itself from tcclaviger's libr4d run ownership: a workgroup owns
// up to MTX consecutive 16-row blocks of ONE expert, so one decoded weight fragment feeds up to MTX WMMAs (the trellis
// decode costs more than an MXFP4 fragment, hence MTX 8 where Dylan's prefill experts ran ~80 rows, FACTS-exl3 trap
// 17); NPW column tiles a wave; no split-K (sk 1, Dylan's prefill default). The K loop, the decode, the prefetch and the
// epilogue are the decode band's (exl3r4d_common.hip.h), with MT A fragments a step instead of one.
//
// WHAT CHANGED, AND WHY (radiance's routing contract, include/exl3_abi.h):
//   * OWNERSHIP FROM A RUN-ALIGNED MAP, not vLLM's padded block-16 sort. Radiance hands sorted_tok + expert_offset:
//     expert e's rows are the contiguous run [off[e], off[e+1]), unpadded. One workgroup of one launch builds the map
//     in scratch (libr4d's moe_blockmap_kernel pattern, r4d_moe.hip at 1.3.0): an entry {expert, r0, r1} for every
//     16 * MTX rows of every run, and zero entries for the rows no run owns ([0, off[0]) and [off[n_expert], T): dropped
//     slots, protected experts' rows -- libr4d's rule). The GEMM's grid is (column block, entry); a workgroup past the
//     map's count exits at its head. Every row of [0, T) is in exactly one entry, so every output row is written once a
//     column block. The map is sized at declare from the band top (exl3r4d_pf_map_cap; the scratch hook).
//   * WEIGHTS ARE TABLES (planes_of): codes / suh / suh.1 / svh per parity class; the narrower class zero-fills its
//     columns past its width (the grid spans the wider class).
//   * A = the prep kernel's rows in scratch (P4's exl3r4d_launch_prep: had128(f16(x * suh)) per (row, matrix)), the
//     arm P4 measured faster at decode (lever L3). Whether to rotate in the K loop instead at prefill is Q3 (lab notes).
//   * Epilogue per 16-row block = the decode band's epilogue_block (EPI 1: h = f16(silu(gate) * up); EPI 2: y = bf16),
//     so the two bands produce the same bits for the same row (the order of every sum is the same: one wave owns a
//     16x16 output tile and walks K in order, whatever MT is).
//
// NOT PORTED (each named): skg, the fused top-k sum, down pieces, dual pools, EPI 0, seg, the fused prep (its rows do
// not fit LDS at MTX > 1), MTX 8 with NPW 4 (its accumulators alone would be 256 VGPRs -- Dylan refused it too).
//
// READINGS, NUMBERS, REJECTED ARMS: lab notes at the end of this file. Read them before proposing anything here.
#include "exl3r4d_common.hip.h"
#include "exl3r4d_prefill.h"

#ifndef EXL3R4D_PF_STAGES
#define EXL3R4D_PF_STAGES 2  // k steps of weights and activations in flight a wave (2 = double buffering, as decode)
#endif
// Deletion-ablation build (a reading, never shipped): 1 replaces each WMMA by one VALU add that keeps the decoded
// fragment and the activation live, so the decode and the loads stay in the loop and only the matrix work goes.
#ifndef EXL3R4D_PF_ABLATE_MMA
#define EXL3R4D_PF_ABLATE_MMA 0
#endif
// 1 replaces the weight loads by register values that vary by step (a reading: the K loop without its memory stream).
#ifndef EXL3R4D_PF_ABLATE_WLOAD
#define EXL3R4D_PF_ABLATE_WLOAD 0
#endif
#ifndef EXL3R4D_PF_NT
#define EXL3R4D_PF_NT 1  // 1: the weight words are nontemporal loads (as ported); 0: plain loads
#endif

// ----------------------------------------------------------------------------- the run map

// ONE workgroup of 1024 threads: each thread owns a contiguous range of expert ids (n_expert <= 4096 by the rows'
// constraint: at most 4 a thread), counts the entries its runs need, a wave prefix and a cross-wave prefix in LDS
// place them, and each thread writes its own. Thread 0 owns the head zero range [0, off[0]) and the last thread the
// tail [off[n_expert], T). Integers only: the map has one answer whatever order anything runs in. Writes past `cap`
// are dropped (only a malformed offset table can ask for more; its rows come out wrong, never written out of place).
__global__ void __launch_bounds__(1024) exl3r4d_pf_map_kernel(Exl3r4dPrefill a) {
  __shared__ int s_w[32];
  const Exl3r4dGemm& p = a.g;
  const int tid = threadIdx.x, nt = blockDim.x, lane = tid % WAVE32, wave = tid / WAVE32, nw = nt / WAVE32;
  const int T = p.T, fm = a.fm, cap = a.cap;
  const int lo = clamp_row(p.off[0], T), hi = clamp_row(p.off[p.n_expert], T);
  const int per = (p.n_expert + nt - 1) / nt, e0 = tid * per, e1 = min(e0 + per, p.n_expert);
  int cnt = 0;
  if (tid == 0) cnt += (lo + fm - 1) / fm;
  for (int e = e0; e < e1; e++) {
    const int a0 = clamp_row(p.off[e], T), a1 = clamp_row(p.off[e + 1], T);
    cnt += a1 > a0 ? (a1 - a0 + fm - 1) / fm : 0;
  }
  if (tid == nt - 1) cnt += (T - hi + fm - 1) / fm;
  int incl = cnt;
#pragma unroll
  for (int o = 1; o < WAVE32; o <<= 1) {
    const int v = __shfl_up(incl, o, WAVE32);
    if (lane >= o) incl += v;
  }
  if (lane == WAVE32 - 1) s_w[wave] = incl;
  __syncthreads();
  int base = 0, total = 0;
  for (int w = 0; w < nw; w++) {
    base += w < wave ? s_w[w] : 0;
    total += s_w[w];
  }
  int o = base + incl - cnt;
  uint4* ent = (uint4*)a.map + 1;
  auto put = [&](unsigned e, int r0, int r1) {
    if (o < cap) ent[o] = make_uint4(e, (unsigned)r0, (unsigned)r1, 0u);
    o++;
  };
  if (tid == 0)
    for (int r = 0; r < lo; r += fm) put(EXL3R4D_PF_ZERO, r, min(r + fm, lo));
  for (int e = e0; e < e1; e++) {
    const int a0 = clamp_row(p.off[e], T), a1 = clamp_row(p.off[e + 1], T);
    for (int r = a0; r < a1; r += fm) put((unsigned)e, r, min(r + fm, a1));
  }
  if (tid == nt - 1)
    for (int r = hi; r < T; r += fm) put(EXL3R4D_PF_ZERO, r, min(r + fm, T));
  if (tid == 0) a.map[0] = (unsigned)min(total, cap);
}

// ----------------------------------------------------------------------------- the MTX GEMM

// One map entry: rows [r0, r1) of expert w, MT = ceil((r1 - r0) / 16) blocks, in column block cb = blockIdx.x of every
// matrix. WV = 8 MATS / NPW waves, wave w owns local tiles [w NPW, w NPW + NPW) of the block's 8 MATS tiles (matrix
// w NPW / 8). The A fragment of block i, lane l is row r0 + 16 i + l % 16 of the prep's rows (clamped into the entry;
// rows past it are computed and not stored).
template <int MT, int NPW, int MATS, int AM>
static __device__ __forceinline__ void pf_body(const Exl3r4dGemm& p, const Planes& w, int r0, int r1, float* lds) {
  using P = Plan<4>;
  constexpr int NW = P::NW, WORDS = P::WORDS;
  const int tid = threadIdx.x, lane = tid % WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE32);  // keep tile bases in SGPRs
  const int h = lane >> 4, nl = lane & 15;
  const int width = MATS == 2 ? w.rows / 2 : w.rows;
  const int ntiles = width / 16, ksteps = w.cols / 16;
  const int cb = blockIdx.x;
  const int vl = wave * NPW;
  const int mat = vl / 8, nt0 = cb * 8 + vl % 8;

  // THE A OPERAND, three arms (AM):
  //   0  every wave loads its own fragments of the prep's rows from global memory each K step (as ported: the four
  //      waves of a matrix load the SAME rows -- ~4.9 GB of L2 traffic at gate_up tp2 M=2048, the cost R2 found);
  //   1  Q3's arm: the entry's rows rotated into LDS a 128-wide K block at a time (no prep kernel) -- REFUTED;
  //   2  the prep's rows of a 128-wide K block copied ONCE a workgroup into LDS, 16 B a thread-chunk, loaded into
  //      registers one pass ahead (libr4d's staging, r4d_moe.hip:1381-1403) -- every wave reads its fragments from LDS.
  // LDS rows are [row][mat][128 + 8] f16: the 8 halfs of padding put the 16 rows a fragment read touches in 16
  // different bank groups. `aoff` is the lane's offset in halfs.
  unsigned avo[MT];
  const _Float16* alds = (const _Float16*)lds;
  constexpr int APITCH = MATS * 128 + 8;   // halfs a row
  unsigned aoff = 0;
  gu8 abase = nullptr;
  if constexpr (AM != 0) {
    aoff = (unsigned)(nl * APITCH + mat * 128 + 4 * h);
  }
  if constexpr (AM == 2) {
    abase = (gu8)(const unsigned char*)p.xr;
  }
  if constexpr (AM == 0) {
#pragma unroll
    for (int i = 0; i < MT; i++) {
      const int row = r0 + 16 * i + nl;
      avo[i] = (unsigned)(((size_t)(row < r1 ? row : r1 - 1) * p.xr_pitch + 4 * h) * 2);
    }
    abase = (gu8)(const unsigned char*)((const _Float16*)p.xr + (size_t)mat * w.cols);
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
  for (int j = 0; j < NPW; j++) wt[j] = (gu8)(const unsigned char*)(wexp + (size_t)(nt0 + j) * WORDS);
  const size_t kbytes = (size_t)ntiles * WORDS * 4;

  v8f acc[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);

  constexpr int PF = EXL3R4D_PF_STAGES;
  unsigned wb[PF][NPW][2][NW];
  v8h ab[PF][MT];

#define PF_WLOAD(ks, W_)                                                                             \
  do {                                                                                               \
    _Pragma("unroll") for (int j = 0; j < NPW; j++) {                                                \
      const gu8 tb_ = wt[j] + (size_t)(ks) * kbytes;                                                 \
      _Pragma("unroll") for (int r = 0; r < 2; r++)                                                  \
        _Pragma("unroll") for (int x = 0; x < NW; x++)                                               \
          W_[j][r][x] = EXL3R4D_PF_ABLATE_WLOAD ? (unsigned)((ks) * 0x9E3779B1u + wvo[r][x] + j)     \
                      : EXL3R4D_PF_NT ? __builtin_nontemporal_load((gu32)(tb_ + wvo[r][x]))          \
                                      : *(gu32)(tb_ + wvo[r][x]);                                    \
    }                                                                                                \
  } while (0)
#define PF_LOAD(ks, W_, A_)                                                                          \
  do {                                                                                               \
    PF_WLOAD(ks, W_);                                                                                \
    const gu8 ab_ = abase + (size_t)(ks) * 32;                                                       \
    _Pragma("unroll") for (int i = 0; i < MT; i++) A_[i] = frag_a8(ab_, avo[i]);                     \
  } while (0)
#define PF_COMPUTE(W_, A_)                                                                           \
  do {                                                                                               \
    _Pragma("unroll") for (int j = 0; j < NPW; j++) {                                                \
      const v8h wf = decode_frag<4, NW>(W_[j], plan.shift);                                          \
      _Pragma("unroll") for (int i = 0; i < MT; i++)                                                 \
        acc[j][i] = EXL3R4D_PF_ABLATE_MMA ? acc[j][i] + (float)(wf[i & 7] + A_[i][0])                \
                                          : mma(wf, A_[i], acc[j][i]);                               \
    }                                                                                                \
  } while (0)

  if constexpr (AM == 0) {
    // Unconditional (clamped) loads, PF stages in flight: the decode band's loop (trap 11).
#pragma unroll
    for (int u = 0; u < PF - 1; u++) PF_LOAD(u < ksteps ? u : ksteps - 1, wb[u], ab[u]);
    int ks = 0;
    for (; ks + PF <= ksteps; ks += PF) {
#pragma unroll
      for (int u = 0; u < PF; u++) {
        const int ls = ks + u + PF - 1 < ksteps ? ks + u + PF - 1 : ksteps - 1;
        PF_LOAD(ls, wb[(u + PF - 1) % PF], ab[(u + PF - 1) % PF]);
        PF_COMPUTE(wb[u], ab[u]);
      }
    }
#pragma unroll
    for (int u = 0; u < PF - 1; u++)
      if (ks + u < ksteps) PF_COMPUTE(wb[u], ab[u]);
  } else if constexpr (AM == 2) {
    // A pass = one 128-wide K block = 8 K steps. Thread t owns chunks c = t + q NTHR of the pass's MT*16 rows x MATS
    // matrices x 16 chunks of 16 B; their global offsets (row clamped into the entry) and LDS offsets are fixed for the
    // whole loop, the pass adds 256 bytes. The stage registers for pass kb + 1 are loaded right after pass kb's are
    // stored, so the one wait for them lands at the next pass head (clamped, unconditional: trap 11).
    constexpr int NTHR = 8 * MATS / NPW * WAVE32;
    constexpr int CH = MT * 16 * MATS * 16, CPT = CH / NTHR;
    static_assert(CH % NTHR == 0, "a pass's chunks divide over the workgroup");
    const int tid_ = threadIdx.x;
    unsigned goff[CPT], loff[CPT];
#pragma unroll
    for (int q = 0; q < CPT; q++) {
      const int c = tid_ + q * NTHR, i = c / (MATS * 16), m = (c / 16) % MATS, part = c % 16;
      const int row = r0 + i < r1 ? r0 + i : r1 - 1;
      goff[q] = (unsigned)(((size_t)row * p.xr_pitch + (size_t)m * w.cols + part * 8) * 2);
      loff[q] = (unsigned)(i * APITCH + m * 128 + part * 8);
    }
    const int nkb = ksteps / 8;
    uint4 stg[CPT];
#define PF_SLOAD(kb_)                                                                                \
  do {                                                                                               \
    const gu8 sb_ = abase + (size_t)(kb_) * 256;                                                     \
    _Pragma("unroll") for (int q = 0; q < CPT; q++) {                                                \
      const v4u v_ = *(__attribute__((address_space(1))) const v4u*)(sb_ + goff[q]);                 \
      stg[q] = make_uint4(v_[0], v_[1], v_[2], v_[3]);                                               \
    }                                                                                                \
  } while (0)
    PF_SLOAD(0);
    PF_WLOAD(0, wb[0]);
    for (int kb = 0; kb < nkb; kb++) {
      if (kb) lds_barrier();   // every wave is done reading pass kb - 1
#pragma unroll
      for (int q = 0; q < CPT; q++) *(uint4*)((_Float16*)lds + loff[q]) = stg[q];
      PF_SLOAD(kb + 1 < nkb ? kb + 1 : kb);
      lds_barrier();
#pragma unroll
      for (int kk = 0; kk < 8; kk++) {
        const int ks = kb * 8 + kk, ln = ks + 1 < ksteps ? ks + 1 : ksteps - 1;
        PF_WLOAD(ln, wb[(kk + 1) % 2]);
        v8h af[MT];
#pragma unroll
        for (int i = 0; i < MT; i++) af[i] = frag_lds(alds + aoff + i * 16 * APITCH + kk * 16);
        PF_COMPUTE(wb[kk % 2], af);
      }
    }
#undef PF_SLOAD
    lds_barrier();   // the epilogue reuses the LDS
  } else {
    // One 128-wide K block at a time: every wave rotates a share of the entry's (row, matrix) pairs into LDS, a
    // barrier, then the block's 8 K steps read their A fragments from LDS. The weight words keep streaming across the
    // block boundary (double-buffered, clamped), so the barrier waits on the rotation, not on memory.
    const int nrows = r1 - r0, nw = (int)blockDim.x / WAVE32, tasks = nrows * MATS;
    PF_WLOAD(0, wb[0]);
    for (int kb = 0; kb < ksteps / 8; kb++) {
      if (kb) lds_barrier();   // every wave is done reading block kb - 1
      for (int t = wave; t < tasks; t += nw) {
        const int i = t / MATS, m = t - i * MATS;
        const v2u o = prep_rotate<MATS>(p, w, r0 + i, m, kb, lane);
        *(v2u*)((_Float16*)lds + i * APITCH + m * 128 + 4 * lane) = o;
      }
      lds_barrier();
#pragma unroll
      for (int kk = 0; kk < 8; kk++) {
        const int ks = kb * 8 + kk, ln = ks + 1 < ksteps ? ks + 1 : ksteps - 1;
        PF_WLOAD(ln, wb[(kk + 1) % 2]);
        v8h af[MT];
#pragma unroll
        for (int i = 0; i < MT; i++) af[i] = frag_lds(alds + aoff + i * 16 * APITCH + kk * 16);
        PF_COMPUTE(wb[kk % 2], af);
      }
    }
    lds_barrier();   // the epilogue reuses the LDS
  }
#undef PF_WLOAD
#undef PF_LOAD
#undef PF_COMPUTE

  // The epilogue, a 16-row block at a time (two barriers each; the LDS holds one block's 8 MATS tiles).
#pragma unroll
  for (int i = 0; i < MT; i++) {
    const int row = r0 + 16 * i + nl;
    const bool in_run = row < r1;
    const int tok = p.sorted[in_run ? row : r0];   // unconditional, clamped into the entry
    v8f blk[NPW];
#pragma unroll
    for (int j = 0; j < NPW; j++) blk[j] = acc[j][i];
    epilogue_block<NPW, MATS>(p, w, blk, wave, 0, lane, cb, row, in_run, in_run && tok >= 0, lds);
  }
}

// mt (1..MTX, workgroup-uniform) -> the body compiled for exactly that many blocks (Dylan's moe_dispatch).
template <int M, int MTX, int NPW, int MATS, int AM>
static __device__ __forceinline__ void pf_dispatch(int mt, const Exl3r4dGemm& p, const Planes& w, int r0, int r1,
                                                   float* lds) {
  if constexpr (M == MTX) {
    pf_body<M, NPW, MATS, AM>(p, w, r0, r1, lds);
  } else {
    if (mt <= M)
      pf_body<M, NPW, MATS, AM>(p, w, r0, r1, lds);
    else
      pf_dispatch<M + 1, MTX, NPW, MATS, AM>(mt, p, w, r0, r1, lds);
  }
}

// One workgroup = (128-column block blockIdx.x, map entry blockIdx.y).
template <int MTX, int NPW, int MATS, int AM>
__global__ void __launch_bounds__(512) exl3r4d_pf_gemm_kernel(Exl3r4dPrefill a) {
  extern __shared__ float lds[];
  const Exl3r4dGemm& p = a.g;
  if (blockIdx.y >= a.map[0]) return;
  const uint4 ent = ((const uint4*)a.map)[1 + blockIdx.y];
  const int r0 = (int)ent.y, r1 = (int)ent.z;
  if (ent.x == EXL3R4D_PF_ZERO) {
    zero_rows(p, r0, r1, blockIdx.x);
    return;
  }
  const Planes w = planes_of(p, (int)ent.x);
  const int width = MATS == 2 ? w.rows / 2 : w.rows;
  if ((int)blockIdx.x * 128 >= width) {  // the narrower parity class: zeros past its own width
    zero_rows(p, r0, r1, blockIdx.x);
    return;
  }
  pf_dispatch<1, MTX, NPW, MATS, AM>((r1 - r0 + 15) / 16, p, w, r0, r1, lds);
}

// ----------------------------------------------------------------------------- the prep, by map entry

// The prefill form's input rotation: the prep's rows (P4's prep_rotate, the same bits as the decode band's prep
// kernel) for the rows the map's entries own. Workgroup = (8 work units, entry blockIdx.y); a unit = one (matrix,
// 128-feature block) task of the entry's rows r0 + rs, r0 + rs + PF_PREP_RS, ..: every task of every row is one
// wave's work, spread over the device by entry and task. P4's prep kernel takes one workgroup an expert, so a hot
// expert's hundreds of rows ran on one workgroup: 743.5 us of gate_up tp2 M=2048 in the trace (R2).
#define EXL3R4D_PF_PREP_RS 4
template <int MATS>
__global__ void __launch_bounds__(256) exl3r4d_pf_prep_kernel(Exl3r4dPrefill a) {
  const Exl3r4dGemm& p = a.g;
  if (blockIdx.y >= a.map[0]) return;
  const uint4 ent = ((const uint4*)a.map)[1 + blockIdx.y];
  if (ent.x == EXL3R4D_PF_ZERO) return;
  const Planes w = planes_of(p, (int)ent.x);
  const int lane = threadIdx.x % WAVE32, nw = (int)blockDim.x / WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane((int)(threadIdx.x / WAVE32));
  const int blocks = w.cols / 128, tasks = MATS * blocks, u = (int)blockIdx.x * nw + wave;
  const int t = u % tasks, rs = u / tasks;
  if (rs >= EXL3R4D_PF_PREP_RS) return;
  const int m = t / blocks, blk = t - m * blocks;
  for (int r = (int)ent.y + rs; r < (int)ent.z; r += EXL3R4D_PF_PREP_RS) {
    const v2u o = prep_rotate<MATS>(p, w, r, m, blk, lane);
    *(v2u*)((_Float16*)p.xr + (size_t)r * p.xr_pitch + (size_t)m * w.cols + blk * 128 + 4 * lane) = o;
  }
}

// ----------------------------------------------------------------------------- host

template <int MATS, int AM>
static const void* pf_gemm_fn(int mtx, int npw) {
  switch (mtx * 16 + npw) {
    case 4 * 16 + 1: return (const void*)exl3r4d_pf_gemm_kernel<4, 1, MATS, AM>;
    case 4 * 16 + 2: return (const void*)exl3r4d_pf_gemm_kernel<4, 2, MATS, AM>;
    case 8 * 16 + 1: return (const void*)exl3r4d_pf_gemm_kernel<8, 1, MATS, AM>;
    case 8 * 16 + 2: return (const void*)exl3r4d_pf_gemm_kernel<8, 2, MATS, AM>;
    default: return nullptr;
  }
}

template <int MATS>
static const void* pf_gemm_am(int am, int mtx, int npw) {
  return am == 0 ? pf_gemm_fn<MATS, 0>(mtx, npw) : am == 1 ? pf_gemm_fn<MATS, 1>(mtx, npw)
       : am == 2 ? pf_gemm_fn<MATS, 2>(mtx, npw) : nullptr;
}

// LDS a GEMM launch asks for: the epilogue's tiles of one 16-row block, or with AM 1 / 2 the rows of one 128-wide K
// block if larger (the two never live at once).
static int64_t pf_lds(int mats, int mtx, int am) {
  const int64_t epi = (int64_t)mats * 8 * 256 * 4, rows = am ? (int64_t)mtx * 16 * (mats * 128 + 8) * 2 : 0;
  return epi > rows ? epi : rows;
}

extern "C" int exl3r4d_launch_prefill(const Exl3r4dPrefill* a, int gate_up, int mtx, int npw, int am, int prep,
                                      void* stream) {
  if (!a->map || a->cap < 1 || a->fm != 16 * mtx || a->g.sk != 1 || (am != 1 && !a->g.xr)) return RAD_E_INVAL;
  if (a->g.n_expert < 1 || a->g.n_expert > 4096) return RAD_E_SHAPE;   // the map kernel: <= 4 experts a thread
  const int mats = gate_up ? 2 : 1, wv = 8 * mats / npw;
  const void* fn = gate_up ? pf_gemm_am<2>(am, mtx, npw) : pf_gemm_am<1>(am, mtx, npw);
  if (!fn) return RAD_E_SHAPE;
  const int64_t smem = pf_lds(mats, mtx, am);
  if (smem > EXL3R4D_LDS_MAX) return RAD_E_SHAPE;
  Exl3r4dPrefill p = *a;
  void* args[] = {&p};
  hipStream_t st = (hipStream_t)stream;
  if (hipLaunchKernel((const void*)exl3r4d_pf_map_kernel, dim3(1), dim3(1024), args, 0, st) != hipSuccess)
    return RAD_E_DEVICE;
  if (am != 1 && prep == 0) {   // P4's prep kernel: one workgroup a live expert
    const int rc = exl3r4d_launch_prep(&a->g, gate_up, stream);
    if (rc != RAD_OK) return rc;
  } else if (am != 1) {          // by map entry
    const int64_t cols = a->g.cls[0].cols > a->g.cls[1].cols ? a->g.cls[0].cols : a->g.cls[1].cols;
    const int64_t units = (int64_t)mats * (cols / 128) * EXL3R4D_PF_PREP_RS;
    const void* pk = gate_up ? (const void*)exl3r4d_pf_prep_kernel<2> : (const void*)exl3r4d_pf_prep_kernel<1>;
    if (hipLaunchKernel(pk, dim3((unsigned)((units + 7) / 8), (unsigned)a->cap), dim3(256), args, 0, st) != hipSuccess)
      return RAD_E_DEVICE;
  }
  if (hipLaunchKernel(fn, dim3((unsigned)(a->g.y_cols / 128), (unsigned)a->cap), dim3((unsigned)(wv * WAVE32)), args,
                      (size_t)smem, st) != hipSuccess)
    return RAD_E_DEVICE;
  return RAD_OK;
}

// ============================================================================= lab notes
//
// READ BEFORE PROPOSING ANYTHING ABOUT THIS KERNEL. (Filled in as P5 measures; plan folder notes/p5.md has the raw.)
