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
// k steps of WEIGHTS in flight a wave (AM 0), a multiple of EXL3R4D_PF_STAGES (which stays the activations' depth), per
// op. ADOPTED round 2 [ledger perf-r1, r2, m1..m4]: gate_up 4 (-1.0/-2.4/-4.0/-9.0% at M=512/1024/2048/4096, replicated
// five sessions; the ATT trace then shows its GEMM issue-bound, load waits 54% -> 7.5%); down stays 2 (4: +0.2..+3.2%,
// 208 VGPRs). EXL3R4D_PF_WSTAGES, if given, sets both (the arms).
#ifdef EXL3R4D_PF_WSTAGES
#define EXL3R4D_PF_WSTAGES_GU EXL3R4D_PF_WSTAGES
#define EXL3R4D_PF_WSTAGES_DN EXL3R4D_PF_WSTAGES
#endif
#ifndef EXL3R4D_PF_WSTAGES_GU
#define EXL3R4D_PF_WSTAGES_GU 4
#endif
#ifndef EXL3R4D_PF_WSTAGES_DN
#define EXL3R4D_PF_WSTAGES_DN EXL3R4D_PF_STAGES
#endif
// Deletion-ablation build (a reading, never shipped): 1 replaces each WMMA by one VALU add that keeps the decoded
// fragment and the activation live, so the decode and the loads stay in the loop and only the matrix work goes.
#ifndef EXL3R4D_PF_ABLATE_MMA
#define EXL3R4D_PF_ABLATE_MMA 0
#endif
// 1 replaces the weight loads by register values that vary by step (a reading: the K loop without its memory stream).
#ifndef EXL3R4D_PF_SGB   // an arm: the K step's schedule pinned with sched_group_barrier (1 grouped, 2 a tile at a time)
#define EXL3R4D_PF_SGB 0
#endif
#ifndef EXL3R4D_PF_ABLATE_EPI   // TIMING ONLY: the GEMM's epilogue deleted (no output)
#define EXL3R4D_PF_ABLATE_EPI 0
#endif
#ifndef EXL3R4D_PF_ABLATE_ALOAD   // TIMING ONLY: the A fragments made from the k-step and the lane's offset, no load
#define EXL3R4D_PF_ABLATE_ALOAD 0
#endif
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
#ifndef EXL3R4D_PF_KX_BUF   // R23: K != 4 weight words through the buffer form (1) or the global form (0)
#define EXL3R4D_PF_KX_BUF 1
#endif
#ifndef EXL3R4D_PF_ABLATE_ROTMATH   // TIMING ONLY: AM 3 stores x ^ suh in place of the rotation (wrong answers)
#define EXL3R4D_PF_ABLATE_ROTMATH 0
#endif
#ifndef EXL3R4D_PF_ABLATE_XRCACHE
#define EXL3R4D_PF_ABLATE_XRCACHE 0
#endif
#define PF_XR_ROW(r) (EXL3R4D_PF_ABLATE_XRCACHE ? ((r) & 255) : (r))
// The epilogue's svh loaded before the K loop (P4's E1 in the shared header), per op. ADOPTED round 2 for down
// [m2-perf: -0.9/-0.4/-0.7/-1.5% at M=512..4096, bit-identical]; gate_up +-0.2% (off). EXL3R4D_PF_SVH_EARLY, if
// given, sets both (the arms).
#ifdef EXL3R4D_PF_SVH_EARLY
#define EXL3R4D_PF_SVH_EARLY_GU EXL3R4D_PF_SVH_EARLY
#define EXL3R4D_PF_SVH_EARLY_DN EXL3R4D_PF_SVH_EARLY
#else
#define EXL3R4D_PF_SVH_EARLY 0
#endif
#ifndef EXL3R4D_PF_SVH_EARLY_GU
#define EXL3R4D_PF_SVH_EARLY_GU 0
#endif
#ifndef EXL3R4D_PF_SVH_EARLY_DN
#define EXL3R4D_PF_SVH_EARLY_DN 1
#endif
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
  if (a.chunk_n <= 1) return;
  // THE CHUNK TABLE (the scratch cap): chunk c = the entries whose first row is in [c R, (c + 1) R) -- entries are in
  // row order (the zero head, the experts in id order, the zero tail), so a chunk is [lo, hi) of them, found by a
  // lower bound each end. Its rows start at its first entry's row (rbase) and end before rbase + R + fm.
  __threadfence_block();
  __syncthreads();
  const int E = (int)a.map[0], R = a.chunk_rows;
  const uint4* ent = (const uint4*)a.map + 1;
  auto first_at = [&](long long row) {   // the first entry whose first row is >= row
    int lo = 0, hi = E;
    while (lo < hi) {
      const int mid = (lo + hi) >> 1;
      if ((long long)ent[mid].y < row) lo = mid + 1; else hi = mid;
    }
    return lo;
  };
  for (int c = threadIdx.x; c < a.chunk_n; c += blockDim.x) {
    const int lo = first_at((long long)c * R), hi = c == a.chunk_n - 1 ? E : first_at((long long)(c + 1) * R);
    ((uint4*)a.ctab)[c] = make_uint4((unsigned)lo, (unsigned)hi, lo < E ? ent[lo].y : 0u, 0u);
  }
}

// Chunk c's entries [lo, hi) and the row its scratch rows start at (the chunk table, the scratch cap); one chunk: every
// entry, from row 0.
static __device__ __forceinline__ void pf_chunk(const Exl3r4dPrefill& a, int c, int& lo, int& hi, int& rbase) {
  if (a.chunk_n <= 1) {
    lo = 0, hi = (int)a.map[0], rbase = 0;
    return;
  }
  const uint4 t = ((const uint4*)a.ctab)[c];
  lo = (int)t.x, hi = (int)t.y, rbase = (int)t.z;
}

// ----------------------------------------------------------------------------- gate_up's epilogue, operands early
// EXL3R4D_PF_EPI_PRE: the shared header's epilogue_block<NPW, 2, PRE = true> (EPI 1, split-K 1, no EPIB / EPIC) copied
// statement for statement, with down's suh slice for the h_rot rotation passed in like svh (`sr`, loaded before the K
// loop) instead of a global load after the epilogue's barrier -- each call exposed two dependent round trips. Same
// helpers, same order of every sum: the same bits (checked by output hash against the shared path). Keep in step with
// epilogue_block (exl3r4d_common.hip.h, P4's).
#ifndef EXL3R4D_PF_EPI_PRE
#define EXL3R4D_PF_EPI_PRE 0
#endif
// wv / nwv: this wave's index in the waves that share `lds` and their count (the h_rot rows' split); a workgroup's
// whole wave set for AM 0 / 4, one column block's group of 8 for AM 5.
template <int NPW>
static __device__ __forceinline__ void pf_epi_gu(const Exl3r4dGemm& p, const Planes& w, const v8f (&acc)[NPW],
                                                 int col_local, int lane, int cb, int row, bool in_run, bool live,
                                                 float* lds, const v8h (*svp)[2], v2u sr, int wv, int nwv) {
  constexpr int MATS = 2, OPW = epi_opw<NPW, MATS>();
  constexpr float RS = 0.08838834764831845f;  // 1 / sqrt(128)
  const int h = lane >> 4;
  const bool writer = col_local * OPW < 8;
  float* xs = lds;  // [8 MATS tiles][32 lanes][8]
  const float sgn_h = h ? -1.f : 1.f;
#pragma unroll
  for (int j = 0; j < NPW; j++) {
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; e++) v[e] = acc[j][e];
    had16_lane(v, sgn_h);
    lds_put(xs + (col_local * NPW + j) * 256 + lane * 8, v);
  }
  lds_barrier();
  if (writer && in_run) {
#pragma unroll
    for (int o = 0; o < OPW; o++) {
      const int t = col_local * OPW + o;
      const int n0 = cb * 128 + 16 * t + 8 * h;
      float g[8], u[8], sg[8], su[8];
      had_tiles(xs, t, lane, g);
      had_tiles(xs + 8 * 256, t, lane, u);
#pragma unroll
      for (int e = 0; e < 8; e++) { sg[e] = (float)svp[o][0][e]; su[e] = (float)svp[o][1][e]; }
      union { _Float16 hv[8]; v4u u; } out;
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const float gv = (g[e] * RS) * sg[e], uv = (u[e] * RS) * su[e];
        out.hv[e] = live ? (_Float16)((gv * (1.0f / (1.0f + expf(-gv)))) * uv) : (_Float16)0.f;
      }
      if (p.h_rot)
        *(v4u*)((_Float16*)((unsigned char*)lds + kHrotStage) + (lane & 15) * 128 + 16 * t + 8 * h) = out.u;
      else
        *(v4u*)((_Float16*)p.y + (size_t)row * p.y_pitch + n0) = out.u;
    }
  }
  if (p.h_rot) {
    lds_barrier();
    const _Float16* hst = (const _Float16*)((const unsigned char*)lds + kHrotStage);
    const int rc = row - (lane & 15);
    for (int i = wv; i < 16; i += nwv) {
      if (!__builtin_amdgcn_readlane((int)in_run, i)) continue;  // wave-uniform
      const bool li = __builtin_amdgcn_readlane((int)live, i);
      const v2u xr = *(const v2u*)(hst + i * 128 + 4 * lane);
      const v2u o = li ? rotate128_math(xr, sr, true, lane) : (v2u)(0u);
      *(v2u*)((_Float16*)p.y + (size_t)(rc + i) * p.y_pitch + cb * 128 + 4 * lane) = o;
    }
  }
  lds_barrier();  // xs (and, next pass, the rotated rows) are rewritten
}
// Down's suh slice of column block cb for the h_rot rotation (zeros without h_rot: never read then).
static __device__ __forceinline__ v2u pf_dn_suh(const Exl3r4dGemm& p, const Planes& w, int cb, int lane) {
  typedef const __attribute__((address_space(1))) v2u* gv2;
  return p.h_rot && w.dn_suh ? *(gv2)((const unsigned char*)w.dn_suh + (size_t)(cb * 128 + 4 * lane) * 2) : (v2u)(0u);
}

// ----------------------------------------------------------------------------- the MTX GEMM

// One map entry: rows [r0, r1) of expert w, MT = ceil((r1 - r0) / 16) blocks, in column block cb = blockIdx.x of every
// matrix. WV = 8 MATS / NPW waves, wave w owns local tiles [w NPW, w NPW + NPW) of the block's 8 MATS tiles (matrix
// w NPW / 8). The A fragment of block i, lane l is row r0 + 16 i + l % 16 of the prep's rows (clamped into the entry;
// rows past it are computed and not stored).
template <int BITS, int MT, int NPW, int MATS, int AM>
static __device__ __forceinline__ void pf_body(const Exl3r4dGemm& p, const Planes& w, int r0, int r1, float* lds,
                                               int rbase) {
  using P = Plan<BITS>;   // R23: any K the shared decode serves (2..6 at mul1); the buffer form below is K = 4's
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
  // P4's E1 (EXL3R4D_PF_SVH_EARLY): the epilogue's svh in registers from the prologue, not a global round trip after
  // the epilogue's barrier in every 16-row block.
  constexpr bool SVE = MATS == 2 ? EXL3R4D_PF_SVH_EARLY_GU : EXL3R4D_PF_SVH_EARLY_DN;
  [[maybe_unused]] v8h svp[epi_opw<NPW, MATS>()][MATS];
  [[maybe_unused]] v2u dsr = {0u, 0u};
  if constexpr (SVE || EXL3R4D_PF_EPI_PRE) svh_early<NPW, MATS>(w, wave, lane, cb, svp);
  if constexpr (EXL3R4D_PF_EPI_PRE && MATS == 2) dsr = pf_dn_suh(p, w, cb, lane);

  constexpr int PF = EXL3R4D_PF_STAGES, PW = MATS == 2 ? EXL3R4D_PF_WSTAGES_GU : EXL3R4D_PF_WSTAGES_DN;
  static_assert(PW >= PF && PW % PF == 0, "the weights' depth is a multiple of the activations'");
  unsigned wb[PW][NPW][2][NW];
  v8h ab[PF][MT];
  // Buffer form (EXL3R4D_PF_BUF): at K = 4 a lane's two runs touch three words of a tile -- woff[1][x] = woff[0][x] +
  // 1 -- so W[0] = {w-1, w0}, W[1] = {w0, w+1} (P4's V1b, exl3r4d.hip). Descriptor word 3 0x31004000: gfx11/12 raw.
  // R23: every K takes the buffer form (P4's EXL3R4D_KX_BUF pattern): K = 4 three words a tile (its two runs share the
  // middle one), any other K four (each run's two), lane offsets fixed, the k-step in an SGPR -- the global form's
  // 64-bit address VALU a tile is what made K != 4 slow, with the shared header's packed-pair hash (EXL3R4D_KX_PAIR).
  constexpr bool WBUF = EXL3R4D_PF_BUF && NW == 2 && (BITS == 4 || EXL3R4D_PF_KX_BUF);
  [[maybe_unused]] const __amdgpu_buffer_rsrc_t wrs =
      __builtin_amdgcn_make_buffer_rsrc((void*)wexp, (short)0, 0x7fffffff, 0x31004000);
  [[maybe_unused]] const __amdgpu_buffer_rsrc_t ars =
      __builtin_amdgcn_make_buffer_rsrc((void*)abase, (short)0, 0x7fffffff, 0x31004000);
  [[maybe_unused]] const unsigned bso = (unsigned)nt0 * WORDS * 4;
  [[maybe_unused]] const unsigned bwm = wvo[0][0], bw0 = wvo[0][1], bwp = wvo[1][NW - 1];
  constexpr int kAux = EXL3R4D_PF_NT ? 1 : 0;   // TH_LOAD_NT, as __builtin_nontemporal_load

#define PF_WLOAD(ks, W_)                                                                             \
  do {                                                                                               \
    if constexpr (WBUF && !EXL3R4D_PF_ABLATE_WLOAD) {                                                \
      const unsigned so_ = bso + (unsigned)(ks) * (unsigned)kbytes;                                  \
      _Pragma("unroll") for (int j = 0; j < NPW; j++) {                                              \
        if constexpr (BITS == 4) {                                                                   \
          const unsigned m_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwm + j * 128, so_, kAux);   \
          const unsigned z_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bw0 + j * 128, so_, kAux);   \
          const unsigned q_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwp + j * 128, so_, kAux);   \
          W_[j][0][0] = m_; W_[j][0][1] = z_; W_[j][1][0] = z_; W_[j][1][1] = q_;                   \
        } else {                                                                                     \
          _Pragma("unroll") for (int r = 0; r < 2; r++)                                              \
            _Pragma("unroll") for (int x = 0; x < NW; x++)                                           \
              W_[j][r][x] = __builtin_amdgcn_raw_buffer_load_b32(wrs, wvo[r][x] + j * WORDS * 4, so_, kAux); \
        }                                                                                            \
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
#define PF_ALOAD(ks, A_)                                                                             \
  do {                                                                                               \
    if constexpr (EXL3R4D_PF_ABLATE_ALOAD) {                                                         \
      _Pragma("unroll") for (int i = 0; i < MT; i++) {                                               \
        union { unsigned u[4]; v8h f; } x_;                                                          \
        _Pragma("unroll") for (int q_ = 0; q_ < 4; q_++) x_.u[q_] = (unsigned)(ks) * 0x3C003C00u + avo[i] + q_; \
        A_[i] = x_.f;                                                                                \
      }                                                                                              \
    } else if constexpr (EXL3R4D_PF_BUF) {                                                           \
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
#define PF_LOAD(ks, W_, A_) \
  do {                      \
    PF_WLOAD(ks, W_);       \
    PF_ALOAD(ks, A_);       \
  } while (0)
#define PF_COMPUTE(W_, A_)                                                                           \
  do {                                                                                               \
    _Pragma("unroll") for (int j = 0; j < NPW; j++) {                                                \
      const v8h wf = decode_frag<BITS, NW>(W_[j], plan.shift);                                          \
      _Pragma("unroll") for (int i = 0; i < MT; i++)                                                 \
        acc[j][i] = EXL3R4D_PF_ABLATE_MMA ? acc[j][i] + (float)(wf[i & 7] + A_[i][0])                \
                                          : mma(wf, A_[i], acc[j][i]);                               \
    }                                                                                                \
    if constexpr (EXL3R4D_PF_SGB == 1) {   /* the step's loads, its decode, its WMMAs (P4's EXL3R4D_SGB) */   \
      __builtin_amdgcn_sched_group_barrier(0x020, 3 * NPW + 2 * MT, 0);                              \
      __builtin_amdgcn_sched_group_barrier(0x002, 44 * NPW, 0);                                      \
      __builtin_amdgcn_sched_group_barrier(0x008, NPW * MT, 0);                                      \
    } else if constexpr (EXL3R4D_PF_SGB == 2) {   /* a tile at a time: its loads, decode, WMMAs */      \
      _Pragma("unroll") for (int j_ = 0; j_ < NPW; j_++) {                                           \
        __builtin_amdgcn_sched_group_barrier(0x020, 3 + 2 * MT / NPW, 0);                            \
        __builtin_amdgcn_sched_group_barrier(0x002, 44, 0);                                          \
        __builtin_amdgcn_sched_group_barrier(0x008, MT, 0);                                          \
      }                                                                                              \
    }                                                                                                \
  } while (0)

  if constexpr (AM == 0 && PW == PF) {
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
  } else if constexpr (AM == 0) {
    // EXL3R4D_PF_WSTAGES > PF: the weights PW k-steps ahead, the activations PF (the ATT reading: the K loop waits on
    // loads issued one step earlier -- too few weight bytes in flight a WGP to cover DRAM's loaded latency). Steps
    // past the loop's last whole PW run (ksteps % PW of them) take the same body under a uniform guard.
#pragma unroll
    for (int u = 0; u < PW - 1; u++) PF_WLOAD(u < ksteps ? u : ksteps - 1, wb[u]);
#pragma unroll
    for (int u = 0; u < PF - 1; u++) PF_ALOAD(u < ksteps ? u : ksteps - 1, ab[u]);
    auto step = [&](int ks, int u) __attribute__((always_inline)) {
      const int lw = ks + u + PW - 1 < ksteps ? ks + u + PW - 1 : ksteps - 1;
      const int la = ks + u + PF - 1 < ksteps ? ks + u + PF - 1 : ksteps - 1;
      PF_WLOAD(lw, wb[(u + PW - 1) % PW]);
      PF_ALOAD(la, ab[(u + PF - 1) % PF]);
      PF_COMPUTE(wb[u % PW], ab[u % PF]);
    };
    int ks = 0;
    for (; ks + PW <= ksteps; ks += PW) {
#pragma unroll
      for (int u = 0; u < PW; u++) step(ks, u);
    }
#pragma unroll
    for (int u = 0; u < PW - 1; u++)
      if (ks + u < ksteps) step(ks, u);
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
        const v2u o = !rlive[j]                  ? (v2u)(0u)
                      : EXL3R4D_PF_ABLATE_ROTMATH ? (xv[j] ^ sv)
                                                  : rotate128_math(xv[j], sv, p.x_bf16 == 0, lane);
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
      goff[q] = (unsigned)(((size_t)PF_XR_ROW(row - rbase) * p.xr_pitch + (size_t)m * w.cols + part * 8) * 2);
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
#undef PF_ALOAD
#undef PF_COMPUTE

  if constexpr (EXL3R4D_PF_ABLATE_EPI) {   // TIMING ONLY: no epilogue; one sum kept so the K loop is not dead code
    float s_ = 0.f;
#pragma unroll
    for (int j = 0; j < NPW; j++)
#pragma unroll
      for (int i = 0; i < MT; i++)
#pragma unroll
        for (int t = 0; t < 8; t++) s_ += acc[j][i][t];
    if (s_ == 1.2345e-30f) ((float*)p.y)[lane] = s_;
    return;
  }
  // The epilogue, a 16-row block at a time (two barriers each; the LDS holds one block's 8 MATS tiles).
#pragma unroll
  for (int i = 0; i < MT; i++) {
    const int row = r0 + 16 * i + nl;
    const bool in_run = row < r1;
    const int tok = p.sorted[in_run ? row : r0];   // unconditional, clamped into the entry
    v8f blk[NPW];
#pragma unroll
    for (int j = 0; j < NPW; j++) blk[j] = acc[j][i];
    if constexpr (EXL3R4D_PF_EPI_PRE && MATS == 2)
      pf_epi_gu<NPW>(p, w, blk, wave, lane, cb, row, in_run, in_run && tok >= 0, lds, svp, dsr, wave,
                     (int)blockDim.x / WAVE32);
    else if constexpr (SVE || EXL3R4D_PF_EPI_PRE)
      epilogue_block<NPW, MATS, true>(p, w, blk, wave, 0, lane, cb, row, in_run, in_run && tok >= 0, lds, svp);
    else
      epilogue_block<NPW, MATS>(p, w, blk, wave, 0, lane, cb, row, in_run, in_run && tok >= 0, lds);
  }
}

// DOWN, G COLUMN BLOCKS A WORKGROUP (EXL3R4D_PF_DN_G; round 2). down's K is short (16 / 24 k-steps at tp2) and its GEMM
// is memory-latency bound (ATT runs/att2: load waits 56%): every (entry, cb) workgroup pays its startup chain and its
// pipeline fill. Here a workgroup takes G consecutive column blocks of one entry and the weight / activation ring runs
// on across them: the next block's first loads are issued during this block's last steps and stay in flight through
// its epilogue. Each tile's sum is the shipped one's (one wave, K in order), so the bits are the same.
#ifndef EXL3R4D_PF_DN_G
#define EXL3R4D_PF_DN_G 1
#endif
template <int BITS, int MT, int G>
static __device__ __forceinline__ void pf_body_dg(const Exl3r4dGemm& p, const Planes& w, int r0, int r1, float* lds,
                                                  int rbase) {
  using P = Plan<BITS>;
  constexpr int NW = P::NW, WORDS = P::WORDS, NPW = 2, PF = 2;
  static_assert(NW == 2, "the buffer form");
  const int tid = threadIdx.x, lane = tid % WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE32);
  const int h = lane >> 4, nl = lane & 15;
  const int width = w.rows, ntiles = width / 16, ksteps = w.cols / 16;
  const int vl = wave * NPW, cb0 = (int)blockIdx.x * G;
  const int gv = min(G, max(0, width / 128 - cb0));   // valid column blocks of this group (the narrower class)
  unsigned avo[MT];
#pragma unroll
  for (int i = 0; i < MT; i++) {
    const int row = r0 + 16 * i + nl;
    avo[i] = (unsigned)(((size_t)PF_XR_ROW((row < r1 ? row : r1 - 1) - rbase) * p.xr_pitch + 4 * h) * 2);
  }
  const __amdgpu_buffer_rsrc_t ars = __builtin_amdgcn_make_buffer_rsrc((void*)p.xr, (short)0, 0x7fffffff, 0x31004000);
  const P plan(lane);
  unsigned wvo[2][NW];
#pragma unroll
  for (int r = 0; r < 2; r++)
#pragma unroll
    for (int x = 0; x < NW; x++) wvo[r][x] = 4u * (unsigned)plan.woff[r][x];
  const __amdgpu_buffer_rsrc_t wrs = __builtin_amdgcn_make_buffer_rsrc((void*)w.codes, (short)0, 0x7fffffff, 0x31004000);
  const unsigned kbytes = (unsigned)ntiles * WORDS * 4;
  const unsigned bwm = wvo[0][0], bw0 = wvo[0][1], bwp = wvo[1][NW - 1];
  constexpr int kAux = EXL3R4D_PF_NT ? 1 : 0;
  unsigned wb[PF][NPW][2][NW];
  v8h ab[PF][MT];
  // Step (g, ks) of the group's concatenated stream: weights of column block cb0 + g, activations of k-step ks.
#define DG_LOAD(g_, ks_, W_, A_)                                                                     \
  do {                                                                                               \
    const unsigned so_ = (unsigned)((cb0 + (g_)) * 8 + vl) * WORDS * 4 + (unsigned)(ks_) * kbytes;   \
    _Pragma("unroll") for (int j = 0; j < NPW; j++) {                                                \
      if constexpr (BITS == 4) {                                                                     \
        const unsigned m_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwm + j * 128, so_, kAux);     \
        const unsigned z_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bw0 + j * 128, so_, kAux);     \
        const unsigned q_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwp + j * 128, so_, kAux);     \
        W_[j][0][0] = m_; W_[j][0][1] = z_; W_[j][1][0] = z_; W_[j][1][1] = q_;                     \
      } else {                                                                                       \
        _Pragma("unroll") for (int r = 0; r < 2; r++)                                                \
          _Pragma("unroll") for (int x = 0; x < NW; x++)                                             \
            W_[j][r][x] = __builtin_amdgcn_raw_buffer_load_b32(wrs, wvo[r][x] + j * WORDS * 4, so_, kAux); \
      }                                                                                              \
    }                                                                                                \
    _Pragma("unroll") for (int i = 0; i < MT; i++) {                                                 \
      union { v2u w[2]; v8h f; } x_;                                                                 \
      x_.w[0] = __builtin_amdgcn_raw_buffer_load_b64(ars, avo[i], (unsigned)(ks_) * 32u, 0);         \
      x_.w[1] = __builtin_amdgcn_raw_buffer_load_b64(ars, avo[i] + 16u, (unsigned)(ks_) * 32u, 0);   \
      A_[i] = x_.f;                                                                                  \
    }                                                                                                \
  } while (0)
  if (gv > 0) DG_LOAD(0, 0, wb[0], ab[0]);   // PF - 1 = 1 step ahead
  for (int g = 0; g < gv; g++) {
    const int cb = cb0 + g;
    v8h svp[epi_opw<NPW, 1>()][1];
    svh_early<NPW, 1>(w, wave, lane, cb, svp);
    v8f acc[NPW][MT];
#pragma unroll
    for (int j = 0; j < NPW; j++)
#pragma unroll
      for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);
    for (int ks = 0; ks + PF <= ksteps; ks += PF) {   // ksteps even (the launch checks)
#pragma unroll
      for (int u = 0; u < PF; u++) {
        const int s = ks + u + 1;   // the step loaded: this block's next, else the next block's first
        const bool nxt = s >= ksteps, more = g + 1 < gv;
        const int lg = nxt && more ? g + 1 : g, lk = nxt ? (more ? s - ksteps : ksteps - 1) : s;
        DG_LOAD(lg, lk, wb[(u + 1) % PF], ab[(u + 1) % PF]);
#pragma unroll
        for (int j = 0; j < NPW; j++) {
          const v8h wf = decode_frag<BITS, NW>(wb[u][j], plan.shift);
#pragma unroll
          for (int i = 0; i < MT; i++) acc[j][i] = mma(wf, ab[u][i], acc[j][i]);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < MT; i++) {
      const int row = r0 + 16 * i + nl;
      const bool in_run = row < r1;
      const int tok = p.sorted[in_run ? row : r0];
      v8f blk[NPW];
#pragma unroll
      for (int j = 0; j < NPW; j++) blk[j] = acc[j][i];
      epilogue_block<NPW, 1, true>(p, w, blk, wave, 0, lane, cb, row, in_run, in_run && tok >= 0, lds, svp);
    }
  }
#undef DG_LOAD
  const int ncol = (int)(p.y_cols / 128);
  for (int g = gv; g < G && cb0 + g < ncol; g++) zero_rows(p, r0, r1, cb0 + g);   // the narrower class's zeros
}
template <int BITS, int M, int MTX, int G>
static __device__ __forceinline__ void pf_dg_dispatch(int mt, const Exl3r4dGemm& p, const Planes& w, int r0, int r1,
                                                      float* lds, int rbase) {
  if constexpr (M == MTX) {
    pf_body_dg<BITS, M, G>(p, w, r0, r1, lds, rbase);
  } else {
    if (mt <= M)
      pf_body_dg<BITS, M, G>(p, w, r0, r1, lds, rbase);
    else
      pf_dg_dispatch<BITS, M + 1, MTX, G>(mt, p, w, r0, r1, lds, rbase);
  }
}

// mt (1..MTX, workgroup-uniform) -> the body compiled for exactly that many blocks (Dylan's moe_dispatch).
template <int BITS, int M, int MTX, int NPW, int MATS, int AM>
static __device__ __forceinline__ void pf_dispatch(int mt, const Exl3r4dGemm& p, const Planes& w, int r0, int r1,
                                                   float* lds, int rbase) {
  if constexpr (M == MTX) {
    pf_body<BITS, M, NPW, MATS, AM>(p, w, r0, r1, lds, rbase);
  } else {
    if (mt <= M)
      pf_body<BITS, M, NPW, MATS, AM>(p, w, r0, r1, lds, rbase);
    else
      pf_dispatch<BITS, M + 1, MTX, NPW, MATS, AM>(mt, p, w, r0, r1, lds, rbase);
  }
}

// One workgroup = (128-column block blockIdx.x, blockIdx.y): the map form (RUN 0) -- map entry blockIdx.y; the run form
// (RUN 1) -- the blockIdx.y-th live expert (P4's find_run, a scan of expert_offset; the workgroups past the live count
// zero the rows no run owns, P4's zero_dead_rows), its whole run in passes of 16 MTX rows.
// EXL3R4D_PF_WPE (an arm): the GEMM's minimum waves a SIMD (amdgpu_waves_per_eu), i.e. a VGPR cap -- 7: <= 219, 8:
// <= 192. 0: the compiler's choice (231-238 at MT 4: 6 waves).
#ifndef EXL3R4D_PF_WPE
#define EXL3R4D_PF_WPE 0
#endif
#if EXL3R4D_PF_WPE
#define PF_WPE_ATTR __attribute__((amdgpu_waves_per_eu(EXL3R4D_PF_WPE)))
#else
#define PF_WPE_ATTR
#endif
template <int BITS, int MTX, int NPW, int MATS, int AM, bool RUN>
__global__ void __launch_bounds__(512) PF_WPE_ATTR exl3r4d_pf_gemm_kernel(Exl3r4dPrefill a) {
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
    pf_chunk(a, a.chunk_c, lo, hi, rbase);
    const int k = lo + (int)blockIdx.y;
    if (k >= hi) return;
    const uint4 ent = ((const uint4*)a.map)[1 + k];
    r0 = (int)ent.y, r1 = (int)ent.z;
    constexpr int GD = MATS == 1 && AM == 0 ? EXL3R4D_PF_DN_G : 1;
    if (ent.x == EXL3R4D_PF_ZERO) {
      for (int g = 0; g < GD && (int)blockIdx.x * GD + g < (int)(p.y_cols / 128); g++)
        zero_rows(p, r0, r1, (int)blockIdx.x * GD + g);
      return;
    }
    e = (int)ent.x;
    if constexpr (GD > 1) {   // down, G column blocks a workgroup (entries are at most 16 MTX rows)
      const Planes w = planes_of(p, e);
      pf_dg_dispatch<BITS, 1, MTX, GD>((r1 - r0 + 15) / 16, p, w, r0, r1, lds, rbase);
      return;
    }
  }
  const Planes w = planes_of(p, e);
  const int width = MATS == 2 ? w.rows / 2 : w.rows;
  if ((int)blockIdx.x * 128 >= width) {  // the narrower parity class: zeros past its own width
    zero_rows(p, r0, r1, blockIdx.x);
    return;
  }
  for (int rc = r0; rc < r1; rc += 16 * MTX) {   // one pass in the map form (an entry is at most 16 MTX rows)
    const int re = rc + 16 * MTX < r1 ? rc + 16 * MTX : r1;
    pf_dispatch<BITS, 1, MTX, NPW, MATS, AM>((re - rc + 15) / 16, p, w, rc, re, lds, rbase);
  }
}

// ----------------------------------------------------------------------------- AM 4: gate_up fused, a row rotated once
//
// WHY (round 2, notes/p5.md): the map form's gate_up moves ~843 MB at tp2 M=2048 -- 423 MB of weights plus the prep's
// rotated rows written (210 MB) and read back (210 MB) -- at ~79% of DRAM peak; the prep alone is write-bandwidth bound
// (341 us, batching its loads changed nothing). The practical-peak target (weights at ~85% of DRAM) needs the rows' round
// trip gone. Rotating in the column-block GEMM (AM 1 / 3) redoes each row's rotation in every column block (x3 even, x2
// odd) and lost on VALU. Here ONE workgroup owns a map entry's rows across ALL N of its expert, so each (row, matrix,
// 128-wide K block) is rotated once, into LDS, and x (10.5 MB, cache-resident) is the only activation stream.
//
// Shape: 16 waves; wave w owns matrix w / 8 (gate, up) and that matrix's tile w % 8 of every 128-column block cb
// (NCB = width / 128 tiles a wave: 3 at tp2's 384, 2 at 256) x MT 16-row blocks -- so for each cb the 16 waves hold
// its 8 gate + 8 up tiles, one each: the decode band's epilogue_block<1, 2> as is (the same bits as every other path:
// one wave owns a 16x16 output tile and walks K in order). A K pass = 128-wide block: the waves rotate its rows into
// LDS ([row][gate 128 | up 128 | 8 pad] f16; wave w: matrix w / 8, rows w % 8 + 8 q), barrier, 8 K steps read the A
// fragments from LDS, the NEXT block's x and suh loads in flight under them, barrier, rotate, barrier. The weights
// stream one k-step ahead across the passes, in the buffer form, as the map form's.
#ifndef EXL3R4D_PF_FU_MID
#define EXL3R4D_PF_FU_MID 1
#endif
#ifndef EXL3R4D_PF_FU_ABLATE_BAR   // TIMING ONLY (wrong output): AM 4 without its two barriers a K pass
#define EXL3R4D_PF_FU_ABLATE_BAR 0
#endif
#ifndef EXL3R4D_PF_FU_WS   // AM 4: k-steps of weights in flight a wave (2, 4 or 8)
#define EXL3R4D_PF_FU_WS 2
#endif
template <int BITS, int MT, int NCB, bool XF>
static __device__ __forceinline__ void pf_fused_body(const Exl3r4dGemm& p, const Planes& w, int r0, int r1,
                                                     float* lds) {
  using P = Plan<BITS>;
  constexpr int NW = P::NW, WORDS = P::WORDS;
  constexpr int APITCH = 2 * 128 + 8, RPW = 2 * MT;
  static_assert(NW == 2, "the fused GEMM takes the buffer form (K = 2..6)");
  const int tid = threadIdx.x, lane = tid % WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE32);
  const int h = lane >> 4, nl = lane & 15, mat = wave >> 3, wl = wave & 7;
  const int width = w.rows / 2, ntiles = width / 16, ksteps = w.cols / 16, nkb = ksteps / 8, nrows = r1 - r0;
  _Float16* const alds = (_Float16*)lds;
  const unsigned aoff = (unsigned)(nl * APITCH + mat * 128 + 4 * h);

  // The rotation tasks: rows wl + 8 q of the entry, matrix mat. The token row is uniform (an SGPR offset).
  const __amdgpu_buffer_rsrc_t xrs = __builtin_amdgcn_make_buffer_rsrc((void*)p.x, (short)0, 0x7fffffff, 0x31004000);
  const __amdgpu_buffer_rsrc_t srs =
      __builtin_amdgcn_make_buffer_rsrc((void*)(mat ? w.suh1 : w.suh), (short)0, 0x7fffffff, 0x31004000);
  unsigned xoff[RPW];
  bool rlive[RPW];
#pragma unroll
  for (int q = 0; q < RPW; q++) {
    const int i = wl + 8 * q, r = r0 + (i < nrows ? i : nrows - 1);
    const int t = __builtin_amdgcn_readfirstlane(p.sorted[r]);
    xoff[q] = (unsigned)((size_t)(t >= 0 ? min(t / p.top_k, p.n_tok - 1) : 0) * p.x_pitch * 2);
    rlive[q] = i < nrows && t >= 0;
  }
  v2u xv[RPW], sv;
  auto xload = [&](int kb) __attribute__((always_inline)) {
    const unsigned ko = (unsigned)kb * 256u;
#pragma unroll
    for (int q = 0; q < RPW; q++)
      xv[q] = __builtin_bit_cast(v2u, __builtin_amdgcn_raw_buffer_load_b64(xrs, (unsigned)lane * 8u, xoff[q] + ko, 0));
    sv = __builtin_bit_cast(v2u, __builtin_amdgcn_raw_buffer_load_b64(srs, (unsigned)lane * 8u, ko, 0));
  };
  // The rotation in two halves (EXL3R4D_PF_FU_MID): its math into registers in the middle of a pass, under the other
  // waves' WMMAs, and only the LDS stores between the pass's two barriers. xv and ro are never live at once.
  v2u ro[RPW];
  auto rot_math = [&]() __attribute__((always_inline)) {
#pragma unroll
    for (int q = 0; q < RPW; q++) {
      const v2u r_ = EXL3R4D_PF_ABLATE_ROTMATH ? xv[q] ^ sv : rotate128_math(xv[q], sv, XF, lane);   // XF: no branch
      ro[q] = rlive[q] ? r_ : (v2u)(0u);
    }
  };
  auto rot_put = [&]() __attribute__((always_inline)) {
#pragma unroll
    for (int q = 0; q < RPW; q++) *(v2u*)(alds + (wl + 8 * q) * APITCH + mat * 128 + 4 * lane) = ro[q];
  };

  // The weights: matrix mat, tile (tk, j * 8 + wl) at ((mat * ksteps + tk) * ntiles + j * 8 + wl) * WORDS u32.
  const P plan(lane);
  unsigned wvo[2][NW];
#pragma unroll
  for (int r = 0; r < 2; r++)
#pragma unroll
    for (int x = 0; x < NW; x++) wvo[r][x] = 4u * (unsigned)plan.woff[r][x];
  const unsigned* wexp = w.codes + (size_t)mat * ksteps * ntiles * WORDS;
  const __amdgpu_buffer_rsrc_t wrs = __builtin_amdgcn_make_buffer_rsrc((void*)wexp, (short)0, 0x7fffffff, 0x31004000);
  const unsigned bso = (unsigned)wl * WORDS * 4, kbytes = (unsigned)ntiles * WORDS * 4;
  constexpr unsigned TS = 8u * WORDS * 4u;   // bytes from a wave's tile in one column block to the next
  const unsigned bwm = wvo[0][0], bw0 = wvo[0][1], bwp = wvo[1][NW - 1];
  constexpr int kAux = EXL3R4D_PF_NT ? 1 : 0;
#define FU_WLOAD(ks, W_)                                                                             \
  do {                                                                                               \
    const unsigned so_ = bso + (unsigned)(ks) * kbytes;                                              \
    _Pragma("unroll") for (int j = 0; j < NCB; j++) {                                                \
      if constexpr (EXL3R4D_PF_ABLATE_WLOAD) {                                                       \
        _Pragma("unroll") for (int r = 0; r < 2; r++)                                                \
          _Pragma("unroll") for (int x = 0; x < NW; x++)                                             \
            W_[j][r][x] = (unsigned)((ks) * 0x9E3779B1u + wvo[r][x] + j + so_);                      \
      } else if constexpr (BITS == 4) {                                                                     \
        const unsigned m_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwm + j * TS, so_, kAux);      \
        const unsigned z_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bw0 + j * TS, so_, kAux);      \
        const unsigned q_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwp + j * TS, so_, kAux);      \
        W_[j][0][0] = m_; W_[j][0][1] = z_; W_[j][1][0] = z_; W_[j][1][1] = q_;                     \
      } else {                                                                                       \
        _Pragma("unroll") for (int r = 0; r < 2; r++)                                                \
          _Pragma("unroll") for (int x = 0; x < NW; x++)                                             \
            W_[j][r][x] = __builtin_amdgcn_raw_buffer_load_b32(wrs, wvo[r][x] + j * TS, so_, kAux);  \
      }                                                                                              \
    }                                                                                                \
  } while (0)

  v8f acc[NCB][MT];
#pragma unroll
  for (int j = 0; j < NCB; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);
  constexpr int WS = EXL3R4D_PF_FU_WS;
  static_assert(8 % WS == 0, "a pass's 8 steps hold whole weight rings");
  unsigned wb[WS][NCB][2][NW];

  xload(0);
#pragma unroll
  for (int u = 0; u < WS - 1; u++) FU_WLOAD(u < ksteps ? u : ksteps - 1, wb[u]);
  rot_math();
  rot_put();
  xload(nkb > 1 ? 1 : 0);
  lds_barrier();
  for (int kb = 0; kb < nkb; kb++) {
#pragma unroll
    for (int kk = 0; kk < 8; kk++) {
      if constexpr (EXL3R4D_PF_FU_MID)
        if (kk == 4) rot_math();   // block kb + 1's rows (kk is a constant: no branch)
      const int ks = kb * 8 + kk, ln = ks + WS - 1 < ksteps ? ks + WS - 1 : ksteps - 1;
      FU_WLOAD(ln, wb[(kk + WS - 1) % WS]);
      v8h af[MT];
#pragma unroll
      for (int i = 0; i < MT; i++) af[i] = frag_lds(alds + aoff + i * 16 * APITCH + kk * 16);
#pragma unroll
      for (int j = 0; j < NCB; j++) {
        const v8h wf = decode_frag<BITS, NW>(wb[kk % WS][j], plan.shift);
#pragma unroll
        for (int i = 0; i < MT; i++)
          acc[j][i] = EXL3R4D_PF_ABLATE_MMA ? acc[j][i] + (float)(wf[i & 7] + af[i][0]) : mma(wf, af[i], acc[j][i]);
      }
      // A fence a k-step: unfenced, the scheduler hoists all 8 steps' loads of the pass to its top and spills them
      // (1,100 registers at MT 4 / NCB 3).
      __builtin_amdgcn_sched_barrier(0);
    }
    // Unconditional (the last pass rotates its own block again, unread): a branch here splits the pass into a block
    // of loads and decodes and a block of WMMAs, and the decoded fragments of all 8 steps spill.
    if constexpr (!EXL3R4D_PF_FU_ABLATE_BAR) lds_barrier();   // every wave is done reading block kb
    if constexpr (!EXL3R4D_PF_FU_MID) rot_math();
    rot_put();       // block kb + 1, its loads issued a pass ago
    xload(kb + 2 < nkb ? kb + 2 : nkb - 1);
    if constexpr (!EXL3R4D_PF_FU_ABLATE_BAR) lds_barrier();
  }
#undef FU_WLOAD
  lds_barrier();   // the epilogue reuses the LDS
  if constexpr (EXL3R4D_PF_ABLATE_EPI) {   // TIMING ONLY: no epilogue; one sum kept so the K loop is not dead code
    float s_ = 0.f;
#pragma unroll
    for (int j = 0; j < NCB; j++)
#pragma unroll
      for (int i = 0; i < MT; i++)
#pragma unroll
        for (int t = 0; t < 8; t++) s_ += acc[j][i][t];
    if (s_ == 1.2345e-30f) ((float*)p.y)[lane] = s_;
    return;
  }

  // The epilogue, a 16-row block and a column block at a time: the decode band's epilogue_block, 16 waves of one tile.
  // EXL3R4D_PF_SVH_EARLY: every column block's svh loaded once here (one round trip, not one a call).
  [[maybe_unused]] v8h svp[NCB][1][2];
  [[maybe_unused]] v2u dsr[NCB];
  if constexpr (EXL3R4D_PF_SVH_EARLY || EXL3R4D_PF_EPI_PRE)
#pragma unroll
    for (int j = 0; j < NCB; j++) svh_early<1, 2>(w, wave, lane, j, svp[j]);
  if constexpr (EXL3R4D_PF_EPI_PRE)
#pragma unroll
    for (int j = 0; j < NCB; j++) dsr[j] = pf_dn_suh(p, w, j, lane);
#pragma unroll
  for (int i = 0; i < MT; i++) {
    const int row = r0 + 16 * i + nl;
    const bool in_run = row < r1;
    const int tok = p.sorted[in_run ? row : r0];
#pragma unroll
    for (int j = 0; j < NCB; j++) {
      const v8f blk[1] = {acc[j][i]};
      if constexpr (EXL3R4D_PF_EPI_PRE)
        pf_epi_gu<1>(p, w, blk, wave, lane, j, row, in_run, in_run && tok >= 0, lds, svp[j], dsr[j], wave, 16);
      else if constexpr (EXL3R4D_PF_SVH_EARLY)
        epilogue_block<1, 2, true>(p, w, blk, wave, 0, lane, j, row, in_run, in_run && tok >= 0, lds, svp[j]);
      else
        epilogue_block<1, 2>(p, w, blk, wave, 0, lane, j, row, in_run, in_run && tok >= 0, lds);
    }
  }
}

template <int BITS, int M, int MTX, int NCB, bool XF>
static __device__ __forceinline__ void pf_fused_dispatch(int mt, const Exl3r4dGemm& p, const Planes& w, int r0, int r1,
                                                         float* lds) {
  if constexpr (M == MTX) {
    pf_fused_body<BITS, M, NCB, XF>(p, w, r0, r1, lds);
  } else {
    if (mt <= M)
      pf_fused_body<BITS, M, NCB, XF>(p, w, r0, r1, lds);
    else
      pf_fused_dispatch<BITS, M + 1, MTX, NCB, XF>(mt, p, w, r0, r1, lds);
  }
}
// x's format (f16 or bf16) as a template constant: a runtime one makes rotate128_math branch inside the K loop.
template <int BITS, int MLO, int MHI, int NCB>
static __device__ __forceinline__ void pf_fused_dispatch_x(int mt, const Exl3r4dGemm& p, const Planes& w, int r0,
                                                           int r1, float* lds) {
  if (p.x_bf16)
    pf_fused_dispatch<BITS, MLO, MHI, NCB, false>(mt, p, w, r0, r1, lds);
  else
    pf_fused_dispatch<BITS, MLO, MHI, NCB, true>(mt, p, w, r0, r1, lds);
}

// One workgroup = map entry blockIdx.y, every column of its expert (grid.x 1). Never chunked: no rows in scratch.
// The entries of MLO..MHI 16-row blocks (EXL3R4D_PF_FU_SPLIT: two launches over the same map, so the kernel of the
// short entries is allocated its own -- fewer -- registers; a kernel's VGPRs are its largest variant's). The launch with
// MLO = 1 also zeroes the rows no run owns. WPE: the minimum waves a SIMD asked of the compiler (0: its choice).
template <int BITS, int MLO, int MHI, int WPE>
__global__ void __launch_bounds__(512) __attribute__((amdgpu_waves_per_eu(WPE > 0 ? WPE : 1)))
exl3r4d_pf_fused_kernel(Exl3r4dPrefill a) {
  extern __shared__ float lds[];
  const Exl3r4dGemm& p = a.g;
  const int k = (int)blockIdx.y;
  if (k >= (int)a.map[0]) return;
  const uint4 ent = ((const uint4*)a.map)[1 + k];
  const int r0 = (int)ent.y, r1 = (int)ent.z, ncol = (int)(p.y_cols / 128), mt = (r1 - r0 + 15) / 16;
  if (ent.x == EXL3R4D_PF_ZERO) {
    if (MLO == 1)
      for (int cb = 0; cb < ncol; cb++) zero_rows(p, r0, r1, cb);
    return;
  }
  if (mt < MLO || mt > MHI) return;   // the other launch's entry
  const Planes w = planes_of(p, (int)ent.x);
  const int ncb = w.rows / 256;
  if (ncb == 3)
    pf_fused_dispatch_x<BITS, MLO, MHI, 3>(mt, p, w, r0, r1, lds);
  else if (ncb == 2)
    pf_fused_dispatch_x<BITS, MLO, MHI, 2>(mt, p, w, r0, r1, lds);
  else
    pf_fused_dispatch_x<BITS, MLO, MHI, 1>(mt, p, w, r0, r1, lds);
  for (int cb = ncb; cb < ncol; cb++) zero_rows(p, r0, r1, cb);   // the narrower class: zeros past its width
}
#ifndef EXL3R4D_PF_FU_SPLIT   // AM 4: entries of <= SPLIT 16-row blocks in a launch of their own (0: one launch)
#define EXL3R4D_PF_FU_SPLIT 0
#endif
#ifndef EXL3R4D_PF_FU_WPE     // AM 4 split: the short entries' kernel's minimum waves a SIMD (8: <= 192 VGPRs)
#define EXL3R4D_PF_FU_WPE 0
#endif

// ----------------------------------------------------------------------------- AM 5: gate_up, column-block groups
//
// AM 4's lesson (round 2): one 16-wave workgroup a WGP at 246+ VGPRs leaves its prologue, epilogue and every wait
// exposed (deletions: skeleton ~0.5 ms + epilogue ~0.22 ms at M=2048). AM 5 keeps the shipped map form's per-wave work
// (NPW 2 tiles x MT 16-row blocks, 6 waves a SIMD) and merges the expert's column-block workgroups into ONE workgroup
// of NCB groups of 8 waves (group g = column block g: the shipped (entry, cb) workgroup's waves) that share the entry's
// rotated rows in LDS -- each (row, matrix, 128-K block) rotated once, no prep, no rows scratch. One launch a class
// width (NCB = width / 128: 3 and 2 at tp2), each skipping the other class's entries. LDS: the K pass's rows [64][264]
// f16 (33.8 KB), then each group's epilogue region (16 KB tiles + 4 KB h stage) side by side.
#ifndef EXL3R4D_PF_G_WS   // AM 5: k-steps of weights in flight a wave
#define EXL3R4D_PF_G_WS 4
#endif
constexpr int kGrpEpi = 16384 + 4096;   // bytes of LDS a group's epilogue takes (kHrotStage + the 16x128 h block)
template <int BITS, int MT, int NCB, bool XF>
static __device__ __forceinline__ void pf_grp_body(const Exl3r4dGemm& p, const Planes& w, int r0, int r1,
                                                   float* lds) {
  using P = Plan<BITS>;
  constexpr int NW = P::NW, WORDS = P::WORDS, NPW = 2, WV = 8 * NCB;
  constexpr int APITCH = 2 * 128 + 8, NT = 2 * 16 * MT, RPW = (NT + WV - 1) / WV;
  static_assert(NW == 2, "AM 5 takes the buffer form (K = 2..6)");
  const int tid = threadIdx.x, lane = tid % WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE32);
  const int h = lane >> 4, nl = lane & 15, cb = wave >> 3, wl = wave & 7;
  const int vl = wl * NPW, mat = vl / 8, nt0 = cb * 8 + vl % 8;
  const int width = w.rows / 2, ntiles = width / 16, ksteps = w.cols / 16, nkb = ksteps / 8, nrows = r1 - r0;
  _Float16* const alds = (_Float16*)lds;
  const unsigned aoff = (unsigned)(nl * APITCH + mat * 128 + 4 * h);

  // Rotation tasks t = wave + WV q: row t / 2, matrix t % 2 (uniform; x row offsets in SGPRs).
  const __amdgpu_buffer_rsrc_t xrs = __builtin_amdgcn_make_buffer_rsrc((void*)p.x, (short)0, 0x7fffffff, 0x31004000);
  const __amdgpu_buffer_rsrc_t srs0 = __builtin_amdgcn_make_buffer_rsrc((void*)w.suh, (short)0, 0x7fffffff, 0x31004000);
  const __amdgpu_buffer_rsrc_t srs1 = __builtin_amdgcn_make_buffer_rsrc((void*)w.suh1, (short)0, 0x7fffffff, 0x31004000);
  unsigned xoff[RPW];
  bool rlive[RPW];
#pragma unroll
  for (int q = 0; q < RPW; q++) {
    const int t = wave + WV * q, i = t >> 1, r = r0 + (i < nrows ? i : nrows - 1);
    const int tk = __builtin_amdgcn_readfirstlane(p.sorted[r]);
    xoff[q] = (unsigned)((size_t)(tk >= 0 ? min(tk / p.top_k, p.n_tok - 1) : 0) * p.x_pitch * 2);
    rlive[q] = t < NT && i < nrows && tk >= 0;
  }
  v2u xv[RPW], sv0, sv1, ro[RPW];
  auto xload = [&](int kb) __attribute__((always_inline)) {
    const unsigned ko = (unsigned)kb * 256u;
#pragma unroll
    for (int q = 0; q < RPW; q++)
      xv[q] = __builtin_bit_cast(v2u, __builtin_amdgcn_raw_buffer_load_b64(xrs, (unsigned)lane * 8u, xoff[q] + ko, 0));
    sv0 = __builtin_bit_cast(v2u, __builtin_amdgcn_raw_buffer_load_b64(srs0, (unsigned)lane * 8u, ko, 0));
    sv1 = __builtin_bit_cast(v2u, __builtin_amdgcn_raw_buffer_load_b64(srs1, (unsigned)lane * 8u, ko, 0));
  };
  auto rot_math = [&]() __attribute__((always_inline)) {
#pragma unroll
    for (int q = 0; q < RPW; q++) {
      const int t = wave + WV * q;   // its matrix: t % 2 = wave % 2 (WV is even)
      const v2u r_ = rotate128_math(xv[q], (wave & 1) ? sv1 : sv0, XF, lane);
      ro[q] = rlive[q] ? r_ : (v2u)(0u);
      (void)t;
    }
  };
  // Unconditional (a branch here splits the pass and spills, AM 4's lesson): the surplus tasks t >= NT write rows
  // 64.. of the pass buffer, inside the launch's LDS (>= 72 rows' worth whatever NCB) and never read.
  static_assert(WV * RPW <= 2 * (16 * EXL3R4D_PF_MTX + 8), "the surplus rows fit the LDS the launch asks for");
  auto rot_put = [&]() __attribute__((always_inline)) {
#pragma unroll
    for (int q = 0; q < RPW; q++) {
      const int t = wave + WV * q;
      *(v2u*)(alds + (t >> 1) * APITCH + (t & 1) * 128 + 4 * lane) = ro[q];
    }
  };

  // The weights: the shipped map form's (matrix mat, tiles nt0, nt0 + 1 of every k-step), buffer form.
  const P plan(lane);
  unsigned wvo[2][NW];
#pragma unroll
  for (int r = 0; r < 2; r++)
#pragma unroll
    for (int x = 0; x < NW; x++) wvo[r][x] = 4u * (unsigned)plan.woff[r][x];
  const unsigned* wexp = w.codes + (size_t)mat * ksteps * ntiles * WORDS;
  const __amdgpu_buffer_rsrc_t wrs = __builtin_amdgcn_make_buffer_rsrc((void*)wexp, (short)0, 0x7fffffff, 0x31004000);
  const unsigned bso = (unsigned)nt0 * WORDS * 4, kbytes = (unsigned)ntiles * WORDS * 4;
  const unsigned bwm = wvo[0][0], bw0 = wvo[0][1], bwp = wvo[1][NW - 1];
  constexpr int kAux = EXL3R4D_PF_NT ? 1 : 0;
#define GR_WLOAD(ks, W_)                                                                             \
  do {                                                                                               \
    const unsigned so_ = bso + (unsigned)(ks) * kbytes;                                              \
    _Pragma("unroll") for (int j = 0; j < NPW; j++) {                                                \
      if constexpr (BITS == 4) {                                                                     \
        const unsigned m_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwm + j * 128, so_, kAux);     \
        const unsigned z_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bw0 + j * 128, so_, kAux);     \
        const unsigned q_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwp + j * 128, so_, kAux);     \
        W_[j][0][0] = m_; W_[j][0][1] = z_; W_[j][1][0] = z_; W_[j][1][1] = q_;                     \
      } else {                                                                                       \
        _Pragma("unroll") for (int r = 0; r < 2; r++)                                                \
          _Pragma("unroll") for (int x = 0; x < NW; x++)                                             \
            W_[j][r][x] = __builtin_amdgcn_raw_buffer_load_b32(wrs, wvo[r][x] + j * WORDS * 4, so_, kAux); \
      }                                                                                              \
    }                                                                                                \
  } while (0)

  v8f acc[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);
  constexpr int WS = EXL3R4D_PF_G_WS;
  static_assert(8 % WS == 0, "a pass's 8 steps hold whole weight rings");
  unsigned wb[WS][NPW][2][NW];

  xload(0);
#pragma unroll
  for (int u = 0; u < WS - 1; u++) GR_WLOAD(u < ksteps ? u : ksteps - 1, wb[u]);
  rot_math();
  rot_put();
  xload(nkb > 1 ? 1 : 0);
  lds_barrier();
  for (int kb = 0; kb < nkb; kb++) {
#pragma unroll
    for (int kk = 0; kk < 8; kk++) {
      if (kk == 4) rot_math();   // block kb + 1's rows, under the other waves' work (kk a constant)
      const int ks = kb * 8 + kk, ln = ks + WS - 1 < ksteps ? ks + WS - 1 : ksteps - 1;
      GR_WLOAD(ln, wb[(kk + WS - 1) % WS]);
      v8h af[MT];
#pragma unroll
      for (int i = 0; i < MT; i++) af[i] = frag_lds(alds + aoff + i * 16 * APITCH + kk * 16);
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        const v8h wf = decode_frag<BITS, NW>(wb[kk % WS][j], plan.shift);
#pragma unroll
        for (int i = 0; i < MT; i++) acc[j][i] = mma(wf, af[i], acc[j][i]);
      }
      __builtin_amdgcn_sched_barrier(0);   // AM 4's lesson: keep the pass's 8 steps from being hoisted and spilled
    }
    lds_barrier();   // every wave is done reading block kb
    rot_put();       // block kb + 1 (the last pass rewrites its own, unread)
    xload(kb + 2 < nkb ? kb + 2 : nkb - 1);
    lds_barrier();
  }
#undef GR_WLOAD

  // The epilogue: each group its own LDS region, the shared one's statements (pf_epi_gu) with svh and down's suh
  // loaded here, once.
  v8h svp[epi_opw<NPW, 2>()][2];
  svh_early<NPW, 2>(w, wl, lane, cb, svp);
  const v2u dsr = pf_dn_suh(p, w, cb, lane);
  float* elds = (float*)((unsigned char*)lds + cb * kGrpEpi);
#pragma unroll
  for (int i = 0; i < MT; i++) {
    const int row = r0 + 16 * i + nl;
    const bool in_run = row < r1;
    const int tok = p.sorted[in_run ? row : r0];
    v8f blk[NPW];
#pragma unroll
    for (int j = 0; j < NPW; j++) blk[j] = acc[j][i];
    pf_epi_gu<NPW>(p, w, blk, wl, lane, cb, row, in_run, in_run && tok >= 0, elds, svp, dsr, wl, 8);
  }
}

template <int BITS, int M, int MTX, int NCB, bool XF>
static __device__ __forceinline__ void pf_grp_dispatch(int mt, const Exl3r4dGemm& p, const Planes& w, int r0, int r1,
                                                       float* lds) {
  if constexpr (M == MTX) {
    pf_grp_body<BITS, M, NCB, XF>(p, w, r0, r1, lds);
  } else {
    if (mt <= M)
      pf_grp_body<BITS, M, NCB, XF>(p, w, r0, r1, lds);
    else
      pf_grp_dispatch<BITS, M + 1, MTX, NCB, XF>(mt, p, w, r0, r1, lds);
  }
}

// One workgroup = map entry blockIdx.y if its expert's class is NCB column blocks wide (the other class's launch takes
// the rest); the zero entries in the launch with ZERO = true.
template <int BITS, int MTX, int NCB, bool ZERO>
__global__ void __launch_bounds__(8 * NCB * WAVE32) exl3r4d_pf_grp_kernel(Exl3r4dPrefill a) {
  extern __shared__ float lds[];
  const Exl3r4dGemm& p = a.g;
  const int k = (int)blockIdx.y;
  if (k >= (int)a.map[0]) return;
  const uint4 ent = ((const uint4*)a.map)[1 + k];
  const int r0 = (int)ent.y, r1 = (int)ent.z, ncol = (int)(p.y_cols / 128);
  if (ent.x == EXL3R4D_PF_ZERO) {
    if (ZERO)
      for (int cb = 0; cb < ncol; cb++) zero_rows(p, r0, r1, cb);
    return;
  }
  const Planes w = planes_of(p, (int)ent.x);
  if (w.rows / 256 != NCB) return;   // the other class's launch
  const int mt = (r1 - r0 + 15) / 16;
  if (p.x_bf16)
    pf_grp_dispatch<BITS, 1, MTX, NCB, false>(mt, p, w, r0, r1, lds);
  else
    pf_grp_dispatch<BITS, 1, MTX, NCB, true>(mt, p, w, r0, r1, lds);
  for (int cb = NCB; cb < ncol; cb++) zero_rows(p, r0, r1, cb);   // the narrower class: zeros past its width
}

// AM 5's launches: one a class width present (1..3 column blocks), the first also zeroing the rows no run owns.
template <int BITS>
static int pf_grp_launch_k(const Exl3r4dPrefill* a, void** args, hipStream_t st) {
  constexpr int X = EXL3R4D_PF_MTX;
  bool seen[4] = {false, false, false, false}, zero = true;
  for (int c = 0; c < a->g.n_class; c++) {
    const int ncb = (int)(a->g.cls[c].rows / 256);
    if (ncb < 1 || ncb > 3) return RAD_E_UNSUPPORTED;
    if (seen[ncb]) continue;
    seen[ncb] = true;
    const void* fn = ncb == 3 ? (zero ? (const void*)exl3r4d_pf_grp_kernel<BITS, X, 3, true>
                                      : (const void*)exl3r4d_pf_grp_kernel<BITS, X, 3, false>)
                   : ncb == 2 ? (zero ? (const void*)exl3r4d_pf_grp_kernel<BITS, X, 2, true>
                                      : (const void*)exl3r4d_pf_grp_kernel<BITS, X, 2, false>)
                              : (zero ? (const void*)exl3r4d_pf_grp_kernel<BITS, X, 1, true>
                                      : (const void*)exl3r4d_pf_grp_kernel<BITS, X, 1, false>);
    const size_t rows = (size_t)(16 * X + 8) * (2 * 128 + 8) * 2, epi = (size_t)ncb * kGrpEpi;   // + surplus rows
    if (hipLaunchKernel(fn, dim3(1u, (unsigned)a->cap), dim3((unsigned)(8 * ncb * WAVE32)), args,
                        rows > epi ? rows : epi, st) != hipSuccess)
      return RAD_E_DEVICE;
    zero = false;
  }
  return RAD_OK;
}
static int pf_grp_launch(int bits, const Exl3r4dPrefill* a, void** args, hipStream_t st) {
  switch (bits) {
    case 2: return pf_grp_launch_k<2>(a, args, st);
    case 3: return pf_grp_launch_k<3>(a, args, st);
    case 4: return pf_grp_launch_k<4>(a, args, st);
    case 5: return pf_grp_launch_k<5>(a, args, st);
    case 6: return pf_grp_launch_k<6>(a, args, st);
    default: return RAD_E_DTYPE;
  }
}

// ----------------------------------------------------------------------------- the prep, by map entry

// The prefill form's input rotation: the prep's rows (P4's prep_rotate, the same bits as the decode band's prep
// kernel) for the rows the map's entries own. Workgroup = (8 work units, entry blockIdx.y); a unit = one (matrix,
// 128-feature block) task of the entry's rows r0 + rs, r0 + rs + PF_PREP_RS, ..: every task of every row is one
// wave's work, spread over the device by entry and task. P4's prep kernel takes one workgroup an expert, so a hot
// expert's hundreds of rows ran on one workgroup: 743.5 us of gate_up tp2 M=2048 in the trace (R2).
#define EXL3R4D_PF_PREP_RS 4
#ifndef EXL3R4D_PF_PREP_B   // rows a wave's prep batch (1: one row at a time, as shipped)
#define EXL3R4D_PF_PREP_B 1
#endif
template <int MATS>
__global__ void __launch_bounds__(256) exl3r4d_pf_prep_kernel(Exl3r4dPrefill a) {
  const Exl3r4dGemm& p = a.g;
  int lo, hi, rbase;
  pf_chunk(a, a.chunk_c, lo, hi, rbase);
  const int lane = threadIdx.x % WAVE32, nw = (int)blockDim.x / WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane((int)(threadIdx.x / WAVE32));
  // grid.y = the map's capacity unchunked (one entry a workgroup row), EXL3R4D_PF_PGY chunked (a loop over them)
  for (int k = lo + (int)blockIdx.y; k < hi; k += (int)gridDim.y) {
    const uint4 ent = ((const uint4*)a.map)[1 + k];
    if (ent.x == EXL3R4D_PF_ZERO) continue;
    const Planes w = planes_of(p, (int)ent.x);
    const int blocks = w.cols / 128, tasks = MATS * blocks, u = (int)blockIdx.x * nw + wave;
    const int t = u % tasks, rs = u / tasks;
    if (rs >= EXL3R4D_PF_PREP_RS) continue;
    const int m = t / blocks, blk = t - m * blocks;
    if constexpr (EXL3R4D_PF_PREP_B > 1) {
      // PREP_B rows a batch: their sorted entries, then their x loads, then the rotations (prep_rotate's math and
      // operands, so the same bits) -- the ATT reading: one row at a time a wave waits on every row's loads.
      const v2u sr = *(const v2u*)((m ? w.suh1 : w.suh) + blk * 128 + 4 * lane);
      for (int rb = (int)ent.y + rs; rb < (int)ent.z; rb += EXL3R4D_PF_PREP_RS * EXL3R4D_PF_PREP_B) {
        int tk[EXL3R4D_PF_PREP_B];
        v2u xv[EXL3R4D_PF_PREP_B];
#pragma unroll
        for (int i = 0; i < EXL3R4D_PF_PREP_B; i++) {
          const int r = rb + EXL3R4D_PF_PREP_RS * i;
          tk[i] = r < (int)ent.z ? p.sorted[r] : -1;
        }
#pragma unroll
        for (int i = 0; i < EXL3R4D_PF_PREP_B; i++) {
          const int xrow = MATS == 2 ? min(max(tk[i], 0) / p.top_k, p.n_tok - 1) : rb + EXL3R4D_PF_PREP_RS * i;
          xv[i] = tk[i] >= 0 ? *(const v2u*)((const unsigned short*)p.x + (size_t)xrow * p.x_pitch + blk * 128 + 4 * lane)
                             : (v2u)(0u);
        }
#pragma unroll
        for (int i = 0; i < EXL3R4D_PF_PREP_B; i++) {
          const int r = rb + EXL3R4D_PF_PREP_RS * i;
          if (r >= (int)ent.z) break;
          const v2u o = tk[i] >= 0 ? rotate128_math(xv[i], sr, p.x_bf16 == 0, lane) : (v2u)(0u);
          *(v2u*)((_Float16*)p.xr + (size_t)PF_XR_ROW(r - rbase) * p.xr_pitch + (size_t)m * w.cols + blk * 128 +
                  4 * lane) = o;
        }
      }
      continue;
    }
    for (int r = (int)ent.y + rs; r < (int)ent.z; r += EXL3R4D_PF_PREP_RS) {
      const v2u o = prep_rotate<MATS>(p, w, r, m, blk, lane);
      *(v2u*)((_Float16*)p.xr + (size_t)PF_XR_ROW(r - rbase) * p.xr_pitch + (size_t)m * w.cols + blk * 128 + 4 * lane) =
          o;
    }
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

// The kernels this build serves: the configured MTX and NPW (exl3r4d_prefill.h), both forms; at K = 4 every A arm,
// at the other K (R23: 2, 3, 5, 6) the shipped one (AM 0) -- the arms are K = 4 readings.
template <int BITS, int MATS, bool RUN>
static const void* pf_gemm_fn(int am) {
  constexpr int X = EXL3R4D_PF_MTX, N = EXL3R4D_PF_NPW;
  if constexpr (MATS == 2 && !RUN)
    if (am == 4) return (const void*)exl3r4d_pf_fused_kernel<BITS, 1, X, EXL3R4D_PF_FU_WPE>;
  if constexpr (BITS != 4) return am == 0 ? (const void*)exl3r4d_pf_gemm_kernel<BITS, X, N, MATS, 0, RUN> : nullptr;
  return am == 0 ? (const void*)exl3r4d_pf_gemm_kernel<BITS, X, N, MATS, 0, RUN>
       : am == 1 ? (const void*)exl3r4d_pf_gemm_kernel<BITS, X, N, MATS, 1, RUN>
       : am == 2 ? (const void*)exl3r4d_pf_gemm_kernel<BITS, X, N, MATS, 2, RUN>
       : am == 3 ? (const void*)exl3r4d_pf_gemm_kernel<BITS, X, N, MATS, 3, RUN> : nullptr;
}
template <int BITS>
static const void* pf_gemm_fn_k(int gate_up, int run, int am) {
  return run ? (gate_up ? pf_gemm_fn<BITS, 2, true>(am) : pf_gemm_fn<BITS, 1, true>(am))
             : (gate_up ? pf_gemm_fn<BITS, 2, false>(am) : pf_gemm_fn<BITS, 1, false>(am));
}
static const void* pf_gemm_select(int bits, int gate_up, int run, int am) {
  switch (bits) {
    case 2: return pf_gemm_fn_k<2>(gate_up, run, am);
    case 3: return pf_gemm_fn_k<3>(gate_up, run, am);
    case 4: return pf_gemm_fn_k<4>(gate_up, run, am);
    case 5: return pf_gemm_fn_k<5>(gate_up, run, am);
    case 6: return pf_gemm_fn_k<6>(gate_up, run, am);
    default: return nullptr;
  }
}

// AM 4 split (EXL3R4D_PF_FU_SPLIT): the kernel of the short entries (h 0) or of the rest (h 1).
template <int BITS>
static const void* pf_fused_split_k(int h) {
  constexpr int S = EXL3R4D_PF_FU_SPLIT > 0 && EXL3R4D_PF_FU_SPLIT < EXL3R4D_PF_MTX ? EXL3R4D_PF_FU_SPLIT : 1;
  if constexpr (EXL3R4D_PF_FU_SPLIT == 0)
    return nullptr;
  else
    return h ? (const void*)exl3r4d_pf_fused_kernel<BITS, S + 1, EXL3R4D_PF_MTX < S + 1 ? S + 1 : EXL3R4D_PF_MTX, 0>
             : (const void*)exl3r4d_pf_fused_kernel<BITS, 1, S, EXL3R4D_PF_FU_WPE>;
}
static const void* pf_fused_split(int bits, int h) {
  switch (bits) {
    case 2: return pf_fused_split_k<2>(h);
    case 3: return pf_fused_split_k<3>(h);
    case 4: return pf_fused_split_k<4>(h);
    case 5: return pf_fused_split_k<5>(h);
    case 6: return pf_fused_split_k<6>(h);
    default: return nullptr;
  }
}

// LDS a GEMM launch asks for: the epilogue's tiles of one 16-row block, or with AM 1 / 2 the rows of one 128-wide K
// block if larger (the two never live at once).
static int64_t pf_lds(int mats, int mtx, int am) {
  const int64_t epi = (int64_t)mats * 8 * 256 * 4, rows = am ? (int64_t)mtx * 16 * (mats * 128 + 8) * 2 : 0;
  return epi > rows ? epi : rows;
}

extern "C" int exl3r4d_launch_prefill(const Exl3r4dPrefill* a, int gate_up, int bits, int am, int prep, int run,
                                      void* stream) {
  constexpr int mtx = EXL3R4D_PF_MTX, npw = EXL3R4D_PF_NPW;
  const bool rot = am == 1 || am == 3 || am == 4 || am == 5;   // the rotation in the GEMM: no prep, no rows read
  if (a->fm != 16 * mtx || a->g.sk != 1 || (!rot && !a->g.xr) || (!run && (!a->map || a->cap < 1)))
    return RAD_E_INVAL;
  if (a->g.n_expert < 1 || a->g.n_expert > 4096) return RAD_E_SHAPE;   // the map kernel: <= 4 experts a thread
  const int mats = gate_up ? 2 : 1, wv = am == 4 ? 16 : 8 * mats / npw;
  if (am >= 4 && (!gate_up || run || a->chunk_n > 1)) return RAD_E_UNSUPPORTED;   // the fused GEMMs: gate_up's map form
  const void* fn = am == 5 ? (const void*)exl3r4d_pf_map_kernel : pf_gemm_select(bits, gate_up, run, am);
  if (!fn) return bits < 2 || bits > 6 ? RAD_E_DTYPE : RAD_E_UNSUPPORTED;
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
  if (!run && a->chunk_n > 1) {   // the scratch cap: the map form in chunks by row, prep then GEMM a chunk
    if (prep != 1 || rot || !a->ctab || a->chunk_rows < 1 || a->chunk_gy < 1) return RAD_E_UNSUPPORTED;
    const int64_t cols = a->g.cls[0].cols > a->g.cls[1].cols ? a->g.cls[0].cols : a->g.cls[1].cols;
    const int64_t units = (int64_t)mats * (cols / 128) * EXL3R4D_PF_PREP_RS;
    const void* pk = gate_up ? (const void*)exl3r4d_pf_prep_kernel<2> : (const void*)exl3r4d_pf_prep_kernel<1>;
    for (int c = 0; c < a->chunk_n; c++) {
      p.chunk_c = c;
      if (hipLaunchKernel(pk, dim3((unsigned)((units + 7) / 8), (unsigned)EXL3R4D_PF_PGY), dim3(256), args, 0, st) !=
              hipSuccess ||
          hipLaunchKernel(fn, dim3((unsigned)(a->g.y_cols / 128), (unsigned)a->chunk_gy), dim3((unsigned)(wv * WAVE32)),
                          args, (size_t)smem, st) != hipSuccess)
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
  if (am == 5) return pf_grp_launch(bits, a, args, st);
  if (am == 4 && EXL3R4D_PF_FU_SPLIT > 0 && EXL3R4D_PF_FU_SPLIT < mtx) {   // short entries, then the rest
    const void* f2[2] = {pf_fused_split(bits, 0), pf_fused_split(bits, 1)};
    for (int h = 0; h < 2; h++)
      if (!f2[h] || hipLaunchKernel(f2[h], dim3(1u, gy), dim3(512u), args, (size_t)smem, st) != hipSuccess)
        return RAD_E_DEVICE;
    return RAD_OK;
  }
  // down, G column blocks a workgroup: the last group may be partial (K is a multiple of 128: even step counts)
  const int gdn = !gate_up && !run && am == 0 ? EXL3R4D_PF_DN_G : 1;
  if (hipLaunchKernel(fn, dim3(am == 4 ? 1u : (unsigned)((a->g.y_cols / 128 + gdn - 1) / gdn), gy), dim3((unsigned)(wv * WAVE32)), args, (size_t)smem,
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
// WHERE THE BAND STANDS (runs/perf12, h_rot 1 = interface v3, build 486f9994 = main 70dda1c), exl3r4d / libr4d us:
//   tp2      M=17         32           64           512          2048          4096
//   gate_up  197.0/195.6  305.2/304.9  458.8/463.1  811.1/724.9  1640.8/1004.6 2930.8/1736.2
//   down     103.7/97.0   151.6/145.3  225.5/217.9  399.1/428.5  698.7/750.1   1174.9/1196.5
//   rocprofv3 (runs/trace3): the GEMMS are at or below libr4d below M ~ 1024 (gate_up M=512 GEMM 672.7 vs 710.9, M=32
//   287.6 vs 302.1; down M=32 144.4 vs 142.0); what is over is gate_up's PREP: 15 / 110.6 / 329 us at M = 32 / 512 /
//   2048, i.e. writing the rotated rows (52 / 210 MB) at DRAM speed, plus the GEMM reading them back at M=2048.
//   The issue floor at M=2048 (minimum WMMA 0.38 ms + decode 0.46 ms at 100% issue) is ~0.84 ms against libr4d's whole
//   0.95-1.0 ms: the prefill gate_up target needs the rows' round trip gone, not tuning.
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
//   REFUTED  AM 3, the rotation in the K loop pipelined one pass ahead [45-47]: gate_up M=2048 +25%, run form M=32 +81%,
//            down +49..+62% (P4: the compiler schedules such a loop dependency-bound). Kept as an arm.
//   READING  rows folded onto a cache-resident region (EXL3R4D_PF_ABLATE_XRCACHE) [52]: gate_up M=2048 -15%, M=4096 -13%:
//            the round trip is worth at most that even if free.
//   REFUTED  MTX 6 / MTX 8 + PF 1 [55]: +4.7% at M=512 (6 waves a SIMD, not 7), -3.3% at 4096; MTX 8 PF 1 +11..+35%.
//   REFUTED  down on P4's decode kernels with h_rot [59]: M=32 +2.6%, M=64 +6.3%.
//   REFUTED  the map form in expert chunks of 16 / 32 MiB of rows (EXL3R4D_PF_CHUNK_MB) [63]: +4.5..+21%: each chunk's
//            prep -> GEMM is serialised with a short-grid tail. Reopen only with prep and GEMM in ONE launch (D18).
//   REFUTED  G 128-column blocks a GEMM workgroup [66]: down run form G 4 +2.0%, G 10 +12.4%; and the loop around the
//            body costs gate_up's run form +30% even at G 1 (VGPRs 215 -> 241). Reverted.
//   SETTLED  the rotation in the GEMM is its MATH [69, perf16]: per wave per K pass (64 WMMA) AM 0 960 instructions,
//            AM 3 3228 (490 s_delay_alu), AM 3 with the math deleted 1030 (18); timed, M=512 810 / 1099 / 878 us, M=2048
//            1642 / 2037 / 1497. ~137 instructions a 128-wide row block, done again in each of the 3 (2) column-block
//            workgroups: a schedule fix removes the stalls, not the work. Even FREE it is worth <= 9% at M=2048 and
//            loses at 512; a ring rotating each row once keeps 1/3 of it (~break-even). Dead for gate_up prefill;
//            reopen only with a rotation several times cheaper than rotate128_math.
//   ADOPTED  THE SCRATCH CAP (EXL3R4D_PF_SCRATCH_MB 46) [72, perf18]: above 46 MiB of rows the map form runs in
//            balanced chunks by row (prep then GEMM a chunk, one rows buffer; the map kernel writes the chunk table).
//            Declared scratch at the 2048-token band top 200 -> ~46 MiB a card (the baseline's: 47): that VRAM is KV the
//            cache lends the expert slab. Per op vs uncapped, tp2: M=512 +3.4%, 768 -6.8%, 1024 -4.0%, 2048 +0.2%,
//            4096 +3.6% (tp1 -0.4..+3.0%); 32 MiB +5.6% at 2048. The first form (chunks of RMAX rows, the last short):
//            M=512 +6.0%.
//   R23      the kernels are templated on K (2..6, mul1): Plan<K> / decode_frag<K> are the shared decode's; the A arms
//            (AM 1..3) are instantiated at K = 4 only. First form (global word loads, the general state hash): K loop
//            per wave (MT 4, 16 WMMA) K 2 219 instructions, 3 255, 4 240, 6 272, no spills -- but K=3 ran +24..31%
//            SLOWER than K=4 at M=17..64 on 25% fewer bytes (48% of bandwidth, runs/kperf1 [76]). Now P4's pair: the
//            buffer form at every K (EXL3R4D_PF_KX_BUF) + the shared header's packed-pair hash (EXL3R4D_KX_PAIR).
//            Final (runs/kperf4 [83]), tp2 us K=3 vs K=4: gate_up +2.7/+0.5/-1.7/+0.3/+2.3%, down +3.5/+3.9/+2.3/+1.1/
//            +3.3% at M=17/32/64/512/2048 -- K=3 matches K=4 but is issue-bound (decode 262 vs 240 instructions a step)
//            on its 25% fewer bytes; K=2 0..-13%, K=5 +18..30%, K=6 +35..45%. gate_up's run form at K = 4 only
//            (EXL3R4D_PF_RN_GU_KX 0): at K = 3 its MT 1 loop compiles to 256 instructions with 22 s_delay_alu (the
//            map form's: 236 / 1) and ran +35% [kperf3].
//   READING  deletions at h_rot 1 [86, del5], gate_up tp2: decode deleted -10.6% (M=2048) / -9.8% (4096) / -3.5% (512);
//            WMMA deleted -4.2 / -4.5 / +2.4%; weight loads deleted -8.8 / -12.0 / -13.3%. No single term is the cost.
//            MTX 8 (one decode for a 128-row entry) saves 14% of decodes at 2048 and 31% at 4096: <= -1.5% / -3.0%
//            before its own LDS and barrier costs -- NOT BUILT.
//   REFUTED  prep of chunk c + 1 by extra workgroups of chunk c's GEMM launch, two rows buffers in the cap [89, copa1]:
//            +26% (64 workgroup rows) .. +67% (16) at M=2048, the same at 512..4096. The prep needs ~1/5 of the device
//            for its time and the latency-bound GEMM leaves no idle slots; dispatched after the GEMM's, the prep
//            workgroups extend every launch's tail.
//   PLAYBOOK ROUND 1 (libr4d's prefill structure read: notes/p5.md "Prefill playbook"), gate_up tp2 h_rot 1 M=2048,
//   every arm through the main group's bounds first (runs/arm-*), timed in runs/occ2 [93, 94]:
//   REFUTED  more waves a SIMD: EXL3R4D_PF_WPE 7 (216 VGPRs, 14 spills) +0.6%, 8 (192, 96 spills) +5.0%.
//   REFUTED  A staged once a WG a pass in LDS (AM 2, its rows fixed for the scratch cap -- it read past the buffer and
//            faulted the card once, 2026-10-12 02:32Z, before the fix) +23.5%.
//   READING  A loads deleted (EXL3R4D_PF_ABLATE_ALOAD) +0.2% at 2048 (-5.3% at 4096): A traffic is NOT the cost;
//            A + W loads deleted -5.7% / -9.9%.
//   STATIC   sched_group_barrier (EXL3R4D_PF_SGB 1 grouped: 239 instructions / 0 s_delay_alu; 2 a tile at a time:
//            246 / 3) gives nothing over the compiler's 240 / 0 -- not timed.
//   ROUND 2 (2026-10-12, goal now PRACTICAL PEAK; gate_up tp2 h_rot 1, us at M=512/1024/2048/4096, every arm through
//   the main group's bounds and byte-identical to base by output hash; runs/perf-r1, r2, r4):
//   READING  ATT thread trace of this GEMM at 2048 (runs/att1; recipe notes/tooling.md): s_wait_loadcnt 54% of summed
//            cycles, WMMA 7.8%, VALU 15% -- the waits are for loads issued one k-step earlier.
//   REFUTED  weights deeper (EXL3R4D_PF_WSTAGES) [99]: 4 steps ahead -1.0/-2.4/-4.0/-9.0% (replicated -4.2% / -9.1%),
//            down +0.2..+3.2%; 8 steps +13%, both operands 4 deep (PF_STAGES 4) +6%. Predicted -10..-25% at 2048.
//   REFUTED  the prep's rows batched (EXL3R4D_PF_PREP_B 4 / 8) [100]: prep 341 -> 330 us, op unchanged: the prep writes
//            210 MB at ~630 GB/s, write-bandwidth bound.
//   READING  tile shapes (the handoff's queue) vs base 826/1595/3001: MTX 3 825/1614/2937, MTX 2 826/1707/3132, MTX 2
//            NPW 4 832/1691/3105, MTX 3 NPW 4 831/1604/2869 -- none beats MTX 4 / NPW 2.
//   REFUTED  AM 4, gate_up fused (each row rotated once, no prep, no rows scratch) [106]: fu0 1103/1344/1879/3085, the
//            rotation math mid-pass 1123/1363/1899/3113, + weights 4 deep 1045/1288/1817/3006 vs base 826/1055/1599/3005.
//            Kept (off) for the readings below; see notes/p5.md round 2.
//   READING  f16 WMMA and VALU SERIALIZE on a SIMD at saturation (tests/gpu/prefill/mix.hip, runs/m1-C: same waves
//            0.94-0.99 of the sum, split waves half the sum) -> the floor is WMMA + decode + rotation VALU, not DRAM.
//   READING  AM 4 deletions at 2048 (fa0 1834): no WMMA 1326, no decode 1297, no weight loads 1760, no barriers 1842,
//            no epilogue 1615, all four K-loop terms 709 -- serial sums; MT-class split (FU_SPLIT) -1%.
//   SMALL    svh before the K loop (EXL3R4D_PF_SVH_EARLY): gate_up +-0.2%, down -0.4..-1.5%; + down's suh too (local
//            epilogue pf_epi_gu, EXL3R4D_PF_EPI_PRE, byte-identical to epilogue_block): gate_up -0.3% (w4 + it, 2048).
//   REFUTED  AM 5 (EXL3R4D_PF_AM_MAP=5: the shipped (entry, cb) workgroups merged into one of NCB groups of 8 waves
//            sharing each row's rotation in LDS; 213 VGPRs): 1076/1313/1812/2950 vs base 826/1057/1595/2987 (runs/m4-C).
//            Byte-identical on main/tops/hrot.
//   ADOPTED  gate_up weights 4 k-steps ahead (EXL3R4D_PF_WSTAGES_GU 4) and down's svh before the K loop
//            (EXL3R4D_PF_SVH_EARLY_DN 1) -- runs/ad-C: gate_up 826/1053/1592/2978 -> 818/1027/1525/2700, down
//            388/461/671/1137 -> 385/460/667/1123 at M=512/1024/2048/4096; bit-identical.
//   READING  ATT (runs/att2): the adopted gate_up GEMM is ISSUE-bound (VALU 62% of cycles, the decode ~51%, WMMA 11%,
//            epilogue barriers 11%, load waits 7.5%); down's GEMM is memory-latency bound (load waits 56%).
//   REFUTED  down G column blocks a workgroup with the ring across them (EXL3R4D_PF_DN_G, 204 VGPRs): G2 +0.3..+1.4%,
//            G4 +2..+3%, G5 +3..+4% vs the adopted build (runs/dg-C).
