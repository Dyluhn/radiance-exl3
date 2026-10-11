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
// 1 (P4's V1b, adopted at decode, cc71419): the weight words and the A fragments as BUFFER loads -- a uniform
// resource, a per-lane offset fixed for the loop, the k-step's offset in an SGPR: no 64-bit address VALU a load.
#ifndef EXL3R4D_PF_BUF
#define EXL3R4D_PF_BUF 1
#endif
// A reading, never shipped (wrong output): 1 folds every prep row onto row % 256, so the rotated rows live in a 2.6 MB
// region that stays cache-resident -- the GEMM's and the prep's time without the round trip of the rows through DRAM.
#ifndef EXL3R4D_PF_ABLATE_XRCACHE
#define EXL3R4D_PF_ABLATE_XRCACHE 0
#endif
#define PF_XR_ROW(r) (EXL3R4D_PF_ABLATE_XRCACHE ? ((r) & 255) : (r))
#ifndef EXL3R4D_PF_NT
#define EXL3R4D_PF_NT 1  // 1: the weight words are nontemporal loads (as ported); 0: plain loads
#endif

// ----------------------------------------------------------------------------- the run map

// ONE workgroup (1024 threads alone, 256 inside the fused map + prep kernel): each thread owns a contiguous range of
// expert ids (n_expert <= 4096 by the rows' constraint: at most 4 or 16 a thread), counts the entries its runs need, a wave prefix and a cross-wave prefix in LDS
// place them, and each thread writes its own. Thread 0 owns the head zero range [0, off[0]) and the last thread the
// tail [off[n_expert], T). Integers only: the map has one answer whatever order anything runs in. Writes past `cap`
// are dropped (only a malformed offset table can ask for more; its rows come out wrong, never written out of place).
static __device__ void pf_map_build(const Exl3r4dPrefill& a, int* s_w) {
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

__global__ void __launch_bounds__(1024) exl3r4d_pf_map_kernel(Exl3r4dPrefill a) {
  __shared__ int s_w[32];
  pf_map_build(a, s_w);
}

// Chunk chunk_c of chunk_n of the map's E entries: [c E / n, (c + 1) E / n), and the first row they cover -- the row
// the chunk's prep rows are stored from (row 0 of the scratch rows). One chunk: [0, E) and row 0.
static __device__ __forceinline__ void pf_chunk(const Exl3r4dPrefill& a, int& lo, int& hi, int& rbase) {
  const int E = (int)a.map[0];
  lo = (int)((long long)a.chunk_c * E / a.chunk_n);
  hi = (int)((long long)(a.chunk_c + 1) * E / a.chunk_n);
  rbase = a.chunk_n > 1 && lo < E ? (int)((const uint4*)a.map)[1 + lo].y : 0;
}

// ----------------------------------------------------------------------------- the MTX GEMM

// One map entry: rows [r0, r1) of expert w, MT = ceil((r1 - r0) / 16) blocks, in column block cb = blockIdx.x of every
// matrix. WV = 8 MATS / NPW waves, wave w owns local tiles [w NPW, w NPW + NPW) of the block's 8 MATS tiles (matrix
// w NPW / 8). The A fragment of block i, lane l is row r0 + 16 i + l % 16 of the prep's rows (clamped into the entry;
// rows past it are computed and not stored).
template <int MT, int NPW, int MATS, int AM>
static __device__ __forceinline__ void pf_body(const Exl3r4dGemm& p, const Planes& w, int r0, int r1, float* lds,
                                               int rbase) {
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
      avo[i] = (unsigned)(((size_t)PF_XR_ROW((row < r1 ? row : r1 - 1) - rbase) * p.xr_pitch + 4 * h) * 2);
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
  // Buffer form (EXL3R4D_PF_BUF): at K = 4 a lane's two runs touch three words of a tile -- woff[1][x] = woff[0][x] +
  // 1 -- so W[0] = {w-1, w0}, W[1] = {w0, w+1} (P4's V1b, exl3r4d.hip). Descriptor word 3 0x31004000: gfx11/12 raw.
  static_assert(!EXL3R4D_PF_BUF || NW == 2, "the three-word form is K = 4's");
  [[maybe_unused]] const __amdgpu_buffer_rsrc_t wrs =
      __builtin_amdgcn_make_buffer_rsrc((void*)wexp, (short)0, 0x7fffffff, 0x31004000);
  [[maybe_unused]] const __amdgpu_buffer_rsrc_t ars =
      __builtin_amdgcn_make_buffer_rsrc((void*)abase, (short)0, 0x7fffffff, 0x31004000);
  [[maybe_unused]] const unsigned bso = (unsigned)nt0 * WORDS * 4;
  [[maybe_unused]] const unsigned bwm = wvo[0][0], bw0 = wvo[0][1], bwp = wvo[1][NW - 1];
  constexpr int kAux = EXL3R4D_PF_NT ? 1 : 0;   // TH_LOAD_NT, as __builtin_nontemporal_load

#define PF_WLOAD(ks, W_)                                                                             \
  do {                                                                                               \
    if constexpr (EXL3R4D_PF_BUF && !EXL3R4D_PF_ABLATE_WLOAD) {                                      \
      const unsigned so_ = bso + (unsigned)(ks) * (unsigned)kbytes;                                  \
      _Pragma("unroll") for (int j = 0; j < NPW; j++) {                                              \
        const unsigned m_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwm + j * 128, so_, kAux);     \
        const unsigned z_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bw0 + j * 128, so_, kAux);     \
        const unsigned q_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwp + j * 128, so_, kAux);     \
        W_[j][0][0] = m_; W_[j][0][1] = z_; W_[j][1][0] = z_; W_[j][1][1] = q_;                     \
      }                                                                                              \
      break;                                                                                         \
    }                                                                                                \
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
    if constexpr (EXL3R4D_PF_BUF) {                                                                  \
      _Pragma("unroll") for (int i = 0; i < MT; i++) {                                               \
        union { v2u w[2]; v8h f; } x_;                                                               \
        x_.w[0] = __builtin_amdgcn_raw_buffer_load_b64(ars, avo[i], (unsigned)(ks) * 32u, 0);        \
        x_.w[1] = __builtin_amdgcn_raw_buffer_load_b64(ars, avo[i] + 16u, (unsigned)(ks) * 32u, 0);  \
        A_[i] = x_.f;                                                                                \
      }                                                                                              \
    } else {                                                                                         \
      const gu8 ab_ = abase + (size_t)(ks) * 32;                                                     \
      _Pragma("unroll") for (int i = 0; i < MT; i++) A_[i] = frag_a8(ab_, avo[i]);                   \
    }                                                                                                \
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
  } else if constexpr (AM == 3) {
    // THE ROTATION IN THE K LOOP, PIPELINED (Q3's arm done the way libr4d stages its operands): wave w rotates the
    // (row, matrix) pairs m = w % MATS, rows i = w / MATS + S j (S = WV / MATS) of every 128-wide K block -- RPW rows a
    // wave, all of one matrix, so its suh slice is one load a pass. The NEXT block's x and suh are loaded into
    // registers right after this block's rotation consumed them, so their round trip runs under the block's 8 K steps
    // (the first arm, AM 1, loaded them task by task between two barriers: +5..+64%). Buffer loads: the row's x base
    // and the pass's K offset are uniform (SGPR), the lane's 8 bytes a fixed voffset.
    constexpr int WV = 8 * MATS / NPW, S = WV / MATS, RPW = MT * 16 / S;
    static_assert(WV % MATS == 0 && (MT * 16) % S == 0, "rotation tasks divide over the waves");
    const int rm = wave % MATS, ri0 = wave / MATS, nkb = ksteps / 8, nrows = r1 - r0;
    const __amdgpu_buffer_rsrc_t xrs = __builtin_amdgcn_make_buffer_rsrc((void*)p.x, (short)0, 0x7fffffff, 0x31004000);
    const __amdgpu_buffer_rsrc_t srs =
        __builtin_amdgcn_make_buffer_rsrc((void*)(rm ? w.suh1 : w.suh), (short)0, 0x7fffffff, 0x31004000);
    unsigned xoff[RPW];   // the task's x row base in bytes (uniform): gate_up the token's row, down the sorted row
    bool rlive[RPW];
#pragma unroll
    for (int j = 0; j < RPW; j++) {
      const int i = ri0 + S * j, r = r0 + (i < nrows ? i : nrows - 1);
      const int t = __builtin_amdgcn_readfirstlane(p.sorted[r]);
      const int xrow = MATS == 2 ? (t >= 0 ? min(t / p.top_k, p.n_tok - 1) : 0) : r;
      xoff[j] = (unsigned)((size_t)xrow * p.x_pitch * 2);
      rlive[j] = t >= 0;
    }
    v2u xv[RPW], sv;
#define PF_XLOAD(kb_)                                                                                \
  do {                                                                                               \
    const unsigned ko_ = (unsigned)(kb_) * 256u;                                                     \
    _Pragma("unroll") for (int j = 0; j < RPW; j++)                                                  \
      xv[j] = __builtin_bit_cast(v2u, __builtin_amdgcn_raw_buffer_load_b64(xrs, (unsigned)lane * 8u, \
                                                                            xoff[j] + ko_, 0));     \
    sv = __builtin_bit_cast(v2u, __builtin_amdgcn_raw_buffer_load_b64(srs, (unsigned)lane * 8u, ko_, 0)); \
  } while (0)
    PF_XLOAD(0);
    PF_WLOAD(0, wb[0]);
    for (int kb = 0; kb < nkb; kb++) {
      if (kb) lds_barrier();   // every wave is done reading block kb - 1
#pragma unroll
      for (int j = 0; j < RPW; j++) {
        const v2u o = rlive[j] ? rotate128_math(xv[j], sv, p.x_bf16 == 0, lane) : (v2u)(0u);
        *(v2u*)((_Float16*)lds + (ri0 + S * j) * APITCH + rm * 128 + 4 * lane) = o;
      }
      PF_XLOAD(kb + 1 < nkb ? kb + 1 : kb);
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
#undef PF_XLOAD
    lds_barrier();   // the epilogue reuses the LDS
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
                                                   float* lds, int rbase) {
  if constexpr (M == MTX) {
    pf_body<M, NPW, MATS, AM>(p, w, r0, r1, lds, rbase);
  } else {
    if (mt <= M)
      pf_body<M, NPW, MATS, AM>(p, w, r0, r1, lds, rbase);
    else
      pf_dispatch<M + 1, MTX, NPW, MATS, AM>(mt, p, w, r0, r1, lds, rbase);
  }
}

// One workgroup = (128-column block blockIdx.x, blockIdx.y): the map form (RUN 0) -- map entry blockIdx.y; the run form
// (RUN 1) -- the blockIdx.y-th live expert (P4's find_run, a scan of expert_offset; the workgroups past the live count
// zero the rows no run owns, P4's zero_dead_rows), its whole run in passes of 16 MTX rows.
template <int MTX, int NPW, int MATS, int AM, bool RUN>
__global__ void __launch_bounds__(512) exl3r4d_pf_gemm_kernel(Exl3r4dPrefill a) {
  extern __shared__ float lds[];
  const Exl3r4dGemm& p = a.g;
  int e, r0, r1, rbase = 0;
  if constexpr (RUN) {
    __shared__ int s_find[33];
    const Run run = find_run(p, blockIdx.y, s_find);
    if (run.e < 0) {
      zero_dead_rows(p, (int)blockIdx.y - run.nlive, (int)gridDim.y - run.nlive, blockIdx.x);
      return;
    }
    e = run.e, r0 = run.r0, r1 = run.r1;
  } else {
    int lo, hi;
    pf_chunk(a, lo, hi, rbase);
    const int k = lo + (int)blockIdx.y;
    if (k >= hi) return;
    const uint4 ent = ((const uint4*)a.map)[1 + k];
    r0 = (int)ent.y, r1 = (int)ent.z;
    if (ent.x == EXL3R4D_PF_ZERO) {
      zero_rows(p, r0, r1, blockIdx.x);
      return;
    }
    e = (int)ent.x;
  }
  const Planes w = planes_of(p, e);
  const int width = MATS == 2 ? w.rows / 2 : w.rows;
  if ((int)blockIdx.x * 128 >= width) {  // the narrower parity class: zeros past its own width
    zero_rows(p, r0, r1, blockIdx.x);
    return;
  }
  for (int rc = r0; rc < r1; rc += 16 * MTX) {   // one pass in the map form (an entry is at most 16 MTX rows)
    const int re = rc + 16 * MTX < r1 ? rc + 16 * MTX : r1;
    pf_dispatch<1, MTX, NPW, MATS, AM>((re - rc + 15) / 16, p, w, rc, re, lds, rbase);
  }
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
  int lo, hi, rbase;
  pf_chunk(a, lo, hi, rbase);
  const int k = lo + (int)blockIdx.y;
  if (k >= hi) return;
  const uint4 ent = ((const uint4*)a.map)[1 + k];
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
    *(v2u*)((_Float16*)p.xr + (size_t)PF_XR_ROW(r - rbase) * p.xr_pitch + (size_t)m * w.cols + blk * 128 + 4 * lane) = o;
  }
}

// The map AND the prep in ONE launch (PREP 2): workgroup 0 builds the map (pf_map_build), every other workgroup
// rotates PF_ROWS consecutive sorted rows -- each row's expert found by a binary search of expert_offset (rows no run
// owns are left alone: the GEMM zeroes them), every (matrix, 128-feature block) task of the rows one wave's work.
// Saves the map's own dispatch and its dependent-launch gap (~3 + ~2.8 us a call, the M = 32 trace).
#define EXL3R4D_PF_ROWS 16
template <int MATS>
__global__ void __launch_bounds__(256) exl3r4d_pf_mapprep_kernel(Exl3r4dPrefill a) {
  __shared__ int s_w[32];
  __shared__ int s_e[EXL3R4D_PF_ROWS];
  if (blockIdx.x == 0) {
    pf_map_build(a, s_w);
    return;
  }
  const Exl3r4dGemm& p = a.g;
  const int tid = threadIdx.x, lane = tid % WAVE32, nw = (int)blockDim.x / WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE32);
  const int base = ((int)blockIdx.x - 1) * EXL3R4D_PF_ROWS;
  if (tid < EXL3R4D_PF_ROWS) {
    const int r = base + tid;
    int e = -1;
    if (r < p.T && r >= clamp_row(p.off[0], p.T) && r < clamp_row(p.off[p.n_expert], p.T)) {
      int lo = 0, hi = p.n_expert;   // the largest e with off[e] <= r (r is below off[n_expert])
      while (lo + 1 < hi) {
        const int mid = (lo + hi) >> 1;
        if (clamp_row(p.off[mid], p.T) <= r) lo = mid; else hi = mid;
      }
      e = lo;
    }
    s_e[tid] = e;
  }
  __syncthreads();
  const int64_t cmax = p.cls[0].cols > p.cls[1].cols ? p.cls[0].cols : p.cls[1].cols;
  const int tmax = MATS * (int)(cmax / 128);
  for (int u = wave; u < EXL3R4D_PF_ROWS * tmax; u += nw) {
    const int i = u / tmax, t = u - i * tmax, r = base + i;
    const int e = __builtin_amdgcn_readfirstlane(s_e[i]);
    if (e < 0) continue;
    const Planes w = planes_of(p, e);
    const int blocks = w.cols / 128;
    if (t >= MATS * blocks) continue;   // the narrower class's K
    const int m = t / blocks, blk = t - m * blocks;
    const v2u o = prep_rotate<MATS>(p, w, r, m, blk, lane);
    *(v2u*)((_Float16*)p.xr + (size_t)PF_XR_ROW(r) * p.xr_pitch + (size_t)m * w.cols + blk * 128 + 4 * lane) = o;
  }
}

// ----------------------------------------------------------------------------- host

// The kernels this build serves: the configured MTX and NPW (exl3r4d_prefill.h), every A arm, both forms.
template <int MATS, bool RUN>
static const void* pf_gemm_fn(int am) {
  constexpr int X = EXL3R4D_PF_MTX, N = EXL3R4D_PF_NPW;
  return am == 0 ? (const void*)exl3r4d_pf_gemm_kernel<X, N, MATS, 0, RUN>
       : am == 1 ? (const void*)exl3r4d_pf_gemm_kernel<X, N, MATS, 1, RUN>
       : am == 2 ? (const void*)exl3r4d_pf_gemm_kernel<X, N, MATS, 2, RUN>
       : am == 3 ? (const void*)exl3r4d_pf_gemm_kernel<X, N, MATS, 3, RUN> : nullptr;
}

// LDS a GEMM launch asks for: the epilogue's tiles of one 16-row block, or with AM 1 / 2 the rows of one 128-wide K
// block if larger (the two never live at once).
static int64_t pf_lds(int mats, int mtx, int am) {
  const int64_t epi = (int64_t)mats * 8 * 256 * 4, rows = am ? (int64_t)mtx * 16 * (mats * 128 + 8) * 2 : 0;
  return epi > rows ? epi : rows;
}

extern "C" int exl3r4d_launch_prefill(const Exl3r4dPrefill* a, int gate_up, int am, int prep, int run, void* stream) {
  constexpr int mtx = EXL3R4D_PF_MTX, npw = EXL3R4D_PF_NPW;
  const bool rot = am == 1 || am == 3;   // the rotation in the GEMM: no prep, the prep's rows not read
  if (a->fm != 16 * mtx || a->g.sk != 1 || (!rot && !a->g.xr) || (!run && (!a->map || a->cap < 1)))
    return RAD_E_INVAL;
  if (a->g.n_expert < 1 || a->g.n_expert > 4096) return RAD_E_SHAPE;   // the map kernel: <= 4 experts a thread
  const int mats = gate_up ? 2 : 1, wv = 8 * mats / npw;
  const void* fn = run ? (gate_up ? pf_gemm_fn<2, true>(am) : pf_gemm_fn<1, true>(am))
                       : (gate_up ? pf_gemm_fn<2, false>(am) : pf_gemm_fn<1, false>(am));
  if (!fn) return RAD_E_SHAPE;
  int64_t smem = pf_lds(mats, mtx, am);
  if (gate_up && a->g.h_rot && smem < kHrotStage + 4096) smem = kHrotStage + 4096;   // v3 EPI 1's staged h block
  if (smem > EXL3R4D_LDS_MAX) return RAD_E_SHAPE;
  Exl3r4dPrefill p = *a;
  void* args[] = {&p};
  hipStream_t st = (hipStream_t)stream;
  if (prep == 3 && (gate_up || !a->g.h_rot || rot || a->g.xr != a->g.x)) return RAD_E_INVAL;   // no prep: h_rot's down
  if (!run && prep == 2 && !rot) {   // the map and the prep by row in one launch
    const void* mk = gate_up ? (const void*)exl3r4d_pf_mapprep_kernel<2> : (const void*)exl3r4d_pf_mapprep_kernel<1>;
    if (hipLaunchKernel(mk, dim3((unsigned)(1 + (a->g.T + EXL3R4D_PF_ROWS - 1) / EXL3R4D_PF_ROWS)), dim3(256), args, 0,
                        st) != hipSuccess)
      return RAD_E_DEVICE;
  } else if (!run && hipLaunchKernel((const void*)exl3r4d_pf_map_kernel, dim3(1), dim3(1024), args, 0, st) !=
             hipSuccess) {
    return RAD_E_DEVICE;
  }
  if (!run && a->chunk_n > 1) {   // the map form in chunks of entries: prep then GEMM a chunk
    if (prep != 1 || rot) return RAD_E_UNSUPPORTED;
    const int64_t cols = a->g.cls[0].cols > a->g.cls[1].cols ? a->g.cls[0].cols : a->g.cls[1].cols;
    const int64_t units = (int64_t)mats * (cols / 128) * EXL3R4D_PF_PREP_RS;
    const void* pk = gate_up ? (const void*)exl3r4d_pf_prep_kernel<2> : (const void*)exl3r4d_pf_prep_kernel<1>;
    const unsigned gy = (unsigned)(a->cap / a->chunk_n + 1);
    for (int c = 0; c < a->chunk_n; c++) {
      p.chunk_c = c;
      if (hipLaunchKernel(pk, dim3((unsigned)((units + 7) / 8), gy), dim3(256), args, 0, st) != hipSuccess ||
          hipLaunchKernel(fn, dim3((unsigned)(a->g.y_cols / 128), gy), dim3((unsigned)(wv * WAVE32)), args,
                          (size_t)smem, st) != hipSuccess)
        return RAD_E_DEVICE;
    }
    return RAD_OK;
  }
  if ((!run && prep == 2 && !rot) || prep == 3) {
    // done above (2), or none (3)
  } else if (!rot && (run || prep == 0)) {   // P4's prep kernel: one workgroup a live expert (find_run)
    const int rc = exl3r4d_launch_prep(&a->g, gate_up, stream);
    if (rc != RAD_OK) return rc;
  } else if (!rot) {                     // by map entry
    const int64_t cols = a->g.cls[0].cols > a->g.cls[1].cols ? a->g.cls[0].cols : a->g.cls[1].cols;
    const int64_t units = (int64_t)mats * (cols / 128) * EXL3R4D_PF_PREP_RS;
    const void* pk = gate_up ? (const void*)exl3r4d_pf_prep_kernel<2> : (const void*)exl3r4d_pf_prep_kernel<1>;
    if (hipLaunchKernel(pk, dim3((unsigned)((units + 7) / 8), (unsigned)a->cap), dim3(256), args, 0, st) != hipSuccess)
      return RAD_E_DEVICE;
  }
  const unsigned gy = run ? (unsigned)a->g.nslot : (unsigned)a->cap;
  if (hipLaunchKernel(fn, dim3((unsigned)(a->g.y_cols / 128), gy), dim3((unsigned)(wv * WAVE32)), args, (size_t)smem,
                      st) != hipSuccess)
    return RAD_E_DEVICE;
  return RAD_OK;
}

// ============================================================================= lab notes
//
// READ BEFORE PROPOSING ANYTHING ABOUT THIS KERNEL. Every number: one gfx1201 card of 64 CUs (which one, and its
// sessions, are in the plan folder's notes/p5.md), radiance-build:1.3.0 (e052a2ab), memory clock 1349 MHz under load,
// precheck and kernel-log watchdog clean. Harness: tools/perfmoe.cpp --exl3-row exl3r4d_moe_*_pf (P4's: 48 routing
// draws over 2 real-size pools, 500 ms warm-up, libr4d moe_gemm_w4a8 on the same routing in the same process, ABAB),
// tp2 = rank 0's parity split (even experts 384 wide, odd 256), 512 experts, top-10, us a dispatch, median of 5.
// Ledger: AI-Work/radiance-exl3-p5-prefill-20261011 (labbook; raw logs runs/<name>).
//
// WHERE THE BAND STANDS (runs/suite2, build ad457483 = commit 39a126d), exl3r4d / libr4d us:
//   tp2      M=17         32           64           512          2048          4096
//   gate_up  196.1/196.2  303.8/304.8  457.5/463.7  799.9/724.9  1633.2/1008.1 2945.8/1738.5
//   down     103.4/97.1   151.1/146.7  227.4/217.9  404.6/428.8  719.5/749.4   1222.5/1201.1
//   tp1: gate_up 0.94-1.00x libr4d up to M=256, 1.04x at 512, 1.24x at 1024, 1.32x at 2048, 1.43x at 4096; down
//   0.89-0.99x except 1.03x at M=2048. The decode band's M=32 pair (C=8 with MTP depth 3) is at parity counting the
//   gq_exp dispatch the baseline issues and EXL3 does not; prefill gate_up at M >= 512 is the open gap.
//
// THE FORMS (prefill.c chooses, by slots per expert):
//   run form  gate_up while 2 T <= 3 n_expert: one workgroup a (column block, live expert) via P4's find_run, the whole
//             run in passes of 16 MTX rows, P4's prep kernel. 2 launches.
//   map form  everything else: the map, the prep (gate_up: by map entry; down: map + prep by row in ONE launch), the
//             GEMM by map entry. 3 / 2 launches.
//
// READINGS (each pre-registered; ledger seq in brackets):
//   R2 trace [rocprofv3, runs/trace1] gate_up tp2 M=2048 as first built: GEMM 1329 us + P4's prep kernel 744 us (one
//      workgroup an expert: the hottest expert's 443 rows x 40 tasks ran serially) + map 3.4 us; libr4d 938 us.
//   R1 deletions [12, perf3] gate_up tp2 M=2048: decode -3%, WMMA -7% (of the total); M=512 -11% / -9%.
//   R3 deletions [27, perf5] weight loads deleted -12% at M=32/512/2048: no single component is the cost.
//   trace2 [runs/trace2, after V1/prep]: M=32 gate_up GEMM 283 us vs libr4d 301 (prep 22.5 + map 3.0 were the excess);
//      M=512 GEMM 674.5 vs 709.7 (prep 99.5); M=2048 GEMM 1291 vs 944, prep 314 = writing the rotated rows: the prep
//      writes T x 2 x 2560 x 2 B (210 MB at M=2048) and the GEMM reads them back -- 420 MB of DRAM beside 432 MB of
//      weights. THAT ROUND TRIP AND THE GEMM'S ISSUE (~1.0 ms of decode + f16 WMMA by count at M=2048) ARE THE GAP.
//
// LEVERS AND ARMS:
//   ADOPTED  map form's prep by map entry (EXL3R4D_PF_PREP digit 1) [22]: gate_up M=2048 -16%, M=4096 -20%.
//   ADOPTED  one form for the band, P4's decode kernels off (EXL3R4D_PF_XN_* 0) [6, 14]: down M=32 -15%, M=64 -20%.
//   ADOPTED  P4's V1b buffer loads (EXL3R4D_PF_BUF 1) and V1a hash (shared header) [30]: -2..-9% each, both ops.
//   ADOPTED  run form for gate_up below the crossover (EXL3R4D_PF_RN_GU 3) [33]: M=17 -3.6%, M=32 -2.1%.
//   ADOPTED  down: map + prep in one launch (EXL3R4D_PF_PREP digit 2) [38]: -1.3..-1.7% (small, consistent).
//   REFUTED  Q3 rotate-in-K-loop (AM 1) [4]: gate_up M=2048 +5%, down +21..+38%; in the run form +56..+60% [34]; in
//            down's map form +25..+64% [39]. As written the rotation sits between two barriers each 128-K block with
//            its loads exposed. Reopen ONLY with the next block's x staged ahead (16-B loads into registers one pass
//            early) -- the A round trip above is the remaining prefill gap and this is its only cure.
//   REFUTED  A staged through LDS once a workgroup (AM 2) [23]: -1..+4%: the four waves' redundant A loads are L2 hits.
//   REFUTED  MTX 8 NPW 1 [5]: +26% (16-wave workgroups at 217 VGPRs: one a WGP). MTX 8 NPW 2 spills (res.sh 177).
//   REFUTED  MUL24 hash [13]: -0.1..-2% (superseded by V1a). PF 4: -3%; plain loads: -1.5% [27] (not adopted).
//   REFUTED  run form for down [33]: +8..+14% (20 column blocks x every live expert scanning expert_offset).
