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

#include "rad_types.h"
#include "exl3r4d_dev.h"

#ifndef EXL3_PF
#define EXL3_PF 2  // k steps of weights / activations in flight per wave (2 = double buffering)
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
#ifndef EXL3R4D_ABLATE_EPI     // 1: skip the output rotation (had16_lane + had_tiles), store raw sums
#define EXL3R4D_ABLATE_EPI 0
#endif
#ifndef EXL3R4D_ABLATE_PREP    // 1: skip the input rotation (fused prep copies x into LDS unrotated)
#define EXL3R4D_ABLATE_PREP 0
#endif
#ifndef EXL3R4D_ABLATE_DECODE  // 1: skip the trellis decode (the raw words are the fragment)
#define EXL3R4D_ABLATE_DECODE 0
#endif
#ifndef EXL3R4D_ABLATE_NOPREP  // 1: delete the fused prep (no x / suh loads, no rotation, LDS rows left as they are)
#define EXL3R4D_ABLATE_NOPREP 0
#endif
// Staged exit (readings only): the workgroup leaves after stage S, writing nothing -- 1 the expert scan, 2 the plane
// pointers, 3 the prep, 4 the K loop and split-K reduction. A fake use keeps each stage's work from being deleted.
#ifndef EXL3R4D_EXIT_STAGE
#define EXL3R4D_EXIT_STAGE 0
#endif

#define WAVE32 32
#define EXL3_BLOCK 16

typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef _Float16 v2h __attribute__((ext_vector_type(2)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef unsigned v4u __attribute__((ext_vector_type(4)));
typedef unsigned v2u __attribute__((ext_vector_type(2)));

// Global-qualified so loads stay global_load (a flat load also bumps dscnt on gfx12; libr4d's note).
typedef __attribute__((address_space(1))) const unsigned* gu32;
typedef __attribute__((address_space(1))) const unsigned long long* gu64;
typedef __attribute__((address_space(1))) const unsigned char* gu8;

static __device__ __forceinline__ void lds_barrier() {
  // libr4d's lds_barrier: order LDS only, so the barrier does not invalidate the vector cache.
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

static __device__ __forceinline__ float swap16(float x) { return __shfl_xor(x, 16, 32); }

static __device__ __forceinline__ unsigned short f32_to_bf16(float f) {  // round to nearest even (libr4d)
  const unsigned u = __builtin_bit_cast(unsigned, f);
  return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

template <int CTRL>
static __device__ __forceinline__ float dpp_f(float v) {  // row_xmask: lane ^ mask within each row of 16
  return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), CTRL, 0xF, 0xF, false));
}

static __device__ __forceinline__ float x16_f(float v) {  // lane ^ 16
  return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(__builtin_bit_cast(int, v), __builtin_bit_cast(int, v),
                                                                0x76543210, (int)0xFEDCBA98, false, false));
}

template <int CTRL>
static __device__ __forceinline__ void fwht_lane_stage(float (&v)[4], float sgn) {
#pragma unroll
  for (int q = 0; q < 4; q++) v[q] = fmaf(v[q], sgn, dpp_f<CTRL>(v[q]));
}

// One 128-feature block of one row's input rotation, by one wave: had128(fp16(x * suh)) * 1/sqrt(128) as fp16,
// four consecutive features per lane (rounding points: x * suh rounded to fp16, the transform in fp32, the result
// rounded to fp16 -- exl3_abi.h's f16(had(f16(x * suh)))). The stage order is exl3_had128's (h = 1, 2, .. 64) and
// each stage is exact-sign a +- b, so the result is bit-identical to the reference's prologue. Feature bits 0, 1
// in the lane, 2..5 through DPP row_xmask, 6 through v_permlanex16_b32. Verbatim from exl3_moe_wmma.hip:275-306.
static __device__ __forceinline__ v2u rotate128(const unsigned short* x_blk, bool x_f16, const _Float16* suh_blk,
                                                int lane) {
  float v[4];
  const v2u xr = *(const v2u*)(x_blk + 4 * lane);
  const v2u sr = *(const v2u*)(suh_blk + 4 * lane);
  const unsigned xs[2] = {xr[0], xr[1]}, ss[2] = {sr[0], sr[1]};
#pragma unroll
  for (int q = 0; q < 4; q++) {
    const unsigned short xb = (unsigned short)(xs[q >> 1] >> (16 * (q & 1)));
    const unsigned short sb = (unsigned short)(ss[q >> 1] >> (16 * (q & 1)));
    const float xv = x_f16 ? (float)__builtin_bit_cast(_Float16, xb) : __builtin_bit_cast(float, (unsigned)xb << 16);
    v[q] = (float)(_Float16)(xv * (float)__builtin_bit_cast(_Float16, sb));
  }
#if EXL3R4D_ABLATE_PREP
  { union { _Float16 h[4]; v2u u; } o;
    for (int q = 0; q < 4; q++) o.h[q] = (_Float16)v[q];
    return o.u; }
#endif
  { const float a = v[0], b = v[1], c = v[2], d = v[3];
    v[0] = a + b; v[1] = a - b; v[2] = c + d; v[3] = c - d;
    const float e0 = v[0], e1 = v[1];
    v[0] = e0 + v[2]; v[1] = e1 + v[3]; v[2] = e0 - v[2]; v[3] = e1 - v[3]; }
  fwht_lane_stage<0x161>(v, (lane & 1) ? -1.f : 1.f);
  fwht_lane_stage<0x162>(v, (lane & 2) ? -1.f : 1.f);
  fwht_lane_stage<0x164>(v, (lane & 4) ? -1.f : 1.f);
  fwht_lane_stage<0x168>(v, (lane & 8) ? -1.f : 1.f);
  {
    const float sgn = (lane & 16) ? -1.f : 1.f;
#pragma unroll
    for (int q = 0; q < 4; q++) v[q] = fmaf(v[q], sgn, x16_f(v[q]));
  }
  constexpr float RS = 0.08838834764831845f;
  union { _Float16 h[4]; v2u u; } o;
#pragma unroll
  for (int q = 0; q < 4; q++) o.h[q] = (_Float16)(v[q] * RS);
  return o.u;
}

// ----------------------------------------------------------------------------- trellis decode (verbatim)

#ifndef EXL3_MUL24
#define EXL3_MUL24 0  // 1: the state hash as two v_mul_u32_u24 + v_lshl_add_u32 instead of one v_mul_lo_u32 (lever L7)
#endif
static __device__ __forceinline__ unsigned state_hash(unsigned x) {  // x < 2^16
#if EXL3_MUL24
  // Both products fit v_mul_u32_u24 (x has 16 known bits). Written as plain multiplies -- or with __umul24, which
  // lowers to masked multiplies -- the combiner folds them straight back into one v_mul_lo_u32 (measured: the same
  // code object, sha256 06f291ec...), and clang has no 24-bit multiply builtin; so the two are inline assembly.
  unsigned lo, hi;
  asm("v_mul_u32_u24 %0, 0xdcd12d, %1" : "=v"(lo) : "v"(x));
  asm("v_mul_u32_u24 %0, 0x83, %1" : "=v"(hi) : "v"(x));
  return lo + (hi << 24);
#else
  return x * 0x83DCD12Du;
#endif
}

static __device__ __forceinline__ unsigned sad_lo(unsigned x, unsigned acc) {  // acc + byte sum of x
  return __builtin_amdgcn_sad_u8(x, 0u, acc);
}

static __device__ __forceinline__ unsigned sad_hi(unsigned x, unsigned acc) {  // acc + (byte sum of x) << 16
  return __builtin_amdgcn_sad_hi_u8(x, 0u, acc);
}

// Two states -> one dword of two fp16 weights (first state in the low half), exllamav3's mul1 codebook:
// fp16(1024 + bytesum(state * 0x83DCD12D)) * K_INV + K_BIAS in one fp16 FMA each (exact as exllamav3's __hfma).
static __device__ __forceinline__ unsigned codebook_pair(unsigned s0, unsigned s1) {
  unsigned acc = sad_lo(state_hash(s0), 0x64006400u);  // 1024.0 in both halves
  acc = sad_hi(state_hash(s1), acc);
  const _Float16 kinv = __builtin_bit_cast(_Float16, (unsigned short)0x1EEE);
  const _Float16 kbias = __builtin_bit_cast(_Float16, (unsigned short)0xC931);
  const v2h r = __builtin_elementwise_fma(__builtin_bit_cast(v2h, acc), (v2h){kinv, kinv}, (v2h){kbias, kbias});
  return __builtin_bit_cast(unsigned, r);
}

template <int BITS>
struct Plan {
  static constexpr int WORDS = 8 * BITS;       // u32 per tile
  static constexpr int NW = BITS == 7 ? 3 : 2;  // words a run of four positions touches
  int woff[2][NW];                             // word index in the tile, per run
  unsigned shift[2];                           // funnel shift that ends the run's bits at bit 0

  __device__ __forceinline__ explicit Plan(int lane) {
    const int n = lane & 15, h = lane >> 4;
    const int base = 32 * (n & 7) + 4 * (n >> 3) + 16 * h;
#pragma unroll
    for (int r = 0; r < 2; r++) {
      const int e3 = (base + 8 * r + 4) * BITS;  // end bit of the run's last state
      const int de = (e3 - 1) >> 5;              // word holding that bit
      shift[r] = (unsigned)(32 * (de + 1) - e3);
#pragma unroll
      for (int x = 0; x < NW; x++) woff[r][x] = (de - (NW - 1) + x + WORDS) % WORDS;
    }
  }
};

// The run's four positions q0..q0 + 3 -> two dwords of fp16 pairs (q0, q0 + 1), (q0 + 2, q0 + 3).
template <int BITS, int NW>
static __device__ __forceinline__ void decode_run(const unsigned (&w)[NW], unsigned s, unsigned& p01, unsigned& p23) {
  const unsigned lo = __builtin_amdgcn_alignbit(w[NW - 2], w[NW - 1], s);  // the 32 stream bits ending at e3
  unsigned mid = 0;
  if constexpr (3 * BITS + 16 > 32) {
    const unsigned hi = NW == 3 ? __builtin_amdgcn_alignbit(w[0], w[1], s) : (w[0] >> s);
    mid = __builtin_amdgcn_alignbit(hi, lo, 16);  // bits 16..47 above e3 - 48
  }
  unsigned st[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int off = (3 - i) * BITS;  // state i ends off bits below e3
    st[i] = off + 16 <= 32 ? __builtin_amdgcn_ubfe(lo, off, 16) : __builtin_amdgcn_ubfe(mid, off - 16, 16);
  }
  p01 = codebook_pair(st[0], st[1]);
  p23 = codebook_pair(st[2], st[3]);
}

// The lane's 16-bit A fragment of one tile: element e is W_hat[k = 8 (e >> 2) + 4 h + (e & 3)][n = lane % 16].
template <int BITS, int NW>
static __device__ __forceinline__ v8h decode_frag(const unsigned (&w)[2][NW], const unsigned (&shift)[2]) {
  union { unsigned u[4]; v8h f; } x;
#if EXL3R4D_ABLATE_DECODE
  x.u[0] = w[0][0]; x.u[1] = w[1][0]; x.u[2] = w[0][NW - 1]; x.u[3] = w[1][NW - 1];
  (void)shift;
  return x.f;
#endif
  decode_run<BITS, NW>(w[0], shift[0], x.u[0], x.u[2]);  // positions base + 0, 1 | base + 2, 3  -> e 0, 1 | 4, 5
  decode_run<BITS, NW>(w[1], shift[1], x.u[1], x.u[3]);  // positions base + 8, 9 | base + 10, 11 -> e 2, 3 | 6, 7
  return x.f;
}

// libr4d's fragA (two 8-byte groups 16 bytes apart) from a uniform base plus a per-lane byte offset.
static __device__ __forceinline__ v8h frag_a8(gu8 base, unsigned off) {
  union { unsigned long long w[2]; v8h f; } x;
  x.w[0] = *(gu64)(base + off);
  x.w[1] = *(gu64)(base + off + 16);
  return x.f;
}

static __device__ __forceinline__ v8h frag_lds(const _Float16* p) {  // fragA from LDS (halfs 0..3 and 8..11)
  union { unsigned long long w[2]; v8h f; } x;
  x.w[0] = *(const unsigned long long*)p;
  x.w[1] = *(const unsigned long long*)(p + 8);
  return x.f;
}

static __device__ __forceinline__ v8f mma(v8h wf, v8h af, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(wf, af, c);
}

// ----------------------------------------------------------------------------- epilogue helpers (verbatim)

static __device__ __forceinline__ void had8(float (&v)[8]) {  // unnormalized natural-order 8-point Hadamard
#pragma unroll
  for (int s = 1; s < 8; s <<= 1)
#pragma unroll
    for (int e = 0; e < 8; e++)
      if (!(e & s)) {
        const float a = v[e], b = v[e + s];
        v[e] = a + b;
        v[e + s] = a - b;
      }
}

// In-lane over e, then over h: after this, v is the tile's 16-point transform (bits 0..3 of n). Over h the lane
// pair holds a (h = 0) and b (h = 1) and wants a + b and a - b: lane h = 0 computes a * 1 + b, lane h = 1
// computes b * -1 + a.
static __device__ __forceinline__ void had16_lane(float (&v)[8], float sgn_h) {
  had8(v);
#pragma unroll
  for (int e = 0; e < 8; e++) v[e] = fmaf(v[e], sgn_h, swap16(v[e]));
}

static __device__ __forceinline__ void lds_put(float* dst, const float (&v)[8]) {
  *(v4f*)dst = (v4f){v[0], v[1], v[2], v[3]};
  *(v4f*)(dst + 4) = (v4f){v[4], v[5], v[6], v[7]};
}

// The 8-point transform over t: out = sum over t2 of (-1)^popcount(t & t2) * tile t2 (this lane's 8 values).
static __device__ __forceinline__ void had_tiles(const float* xs, int t, int lane, float (&out)[8]) {
#pragma unroll
  for (int e = 0; e < 8; e++) out[e] = 0.f;
#pragma unroll
  for (int t2 = 0; t2 < 8; t2++) {
    const float sg = (__builtin_popcount(t & t2) & 1) ? -1.f : 1.f;
    const v4f a = *(const v4f*)(xs + t2 * 256 + lane * 8);
    const v4f b = *(const v4f*)(xs + t2 * 256 + lane * 8 + 4);
#pragma unroll
    for (int e = 0; e < 4; e++) {
      out[e] = fmaf(sg, a[e], out[e]);
      out[4 + e] = fmaf(sg, b[e], out[4 + e]);
    }
  }
}

static __device__ __forceinline__ void load_h8(const _Float16* p, float (&v)[8]) {
  const v8h x = *(const v8h*)p;
#pragma unroll
  for (int e = 0; e < 8; e++) v[e] = (float)x[e];
}

// ----------------------------------------------------------------------------- radiance's routing

// An entry of a weight-table operand (libr4d's wtab_at): the pointer array when s0 is 0, a stacked plane otherwise.
// The table lives in device memory and the index is workgroup-uniform, so this is one scalar load.
static __device__ __forceinline__ const unsigned char* entry_at(const void* tab, long long s0, int j) {
  if (s0 == 0) return (const unsigned char*)((const void* const*)tab)[j];
  return (const unsigned char*)tab + (size_t)j * (size_t)s0;
}

struct Run {
  int e;      // the expert, or -1: a spare workgroup
  int r0, r1; // its rows [r0, r1), clamped to [0, T)
  int nlive;  // experts with a non-empty run
};

// The y-th expert (in id order) whose run [off[e], off[e+1]) holds at least one row of [0, T). Every thread
// counts the live experts of a contiguous id range, a wave prefix and a cross-wave prefix in LDS number them,
// and the thread whose range holds number `slot` names the expert. Two barriers, one load of expert_offset a
// thread; the same answer in every workgroup of the column (it is arithmetic on the same table).
static __device__ __forceinline__ int clamp_row(int v, int T) { return v < 0 ? 0 : (v > T ? T : v); }

static __device__ Run find_run(const Exl3r4dGemm& p, int slot, int* sh) {
  const int tid = threadIdx.x, nt = blockDim.x, lane = tid % WAVE32, wave = tid / WAVE32, nw = nt / WAVE32;
  const int per = (p.n_expert + nt - 1) / nt, e0 = tid * per;
  int cnt = 0;
  for (int e = e0; e < e0 + per && e < p.n_expert; e++)
    cnt += clamp_row(p.off[e + 1], p.T) > clamp_row(p.off[e], p.T);
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
  if (cnt && slot >= excl && slot < excl + cnt) {
    int k = slot - excl;
    for (int e = e0; e < e0 + per && e < p.n_expert; e++)
      if (clamp_row(p.off[e + 1], p.T) > clamp_row(p.off[e], p.T) && k-- == 0) {
        sh[32] = e;
        break;
      }
  }
  __syncthreads();
  Run r;
  r.e = sh[32];
  r.nlive = total;
  r.r0 = r.e >= 0 ? clamp_row(p.off[r.e], p.T) : 0;
  r.r1 = r.e >= 0 ? clamp_row(p.off[r.e + 1], p.T) : 0;
  return r;
}

// Zero rows [a, b) of the output in column block cb (128 columns), 16 bytes a thread a step.
static __device__ __forceinline__ void zero_rows(const Exl3r4dGemm& p, int a, int b, int cb) {
  const int units = (b - a) * 16;  // 128 columns x 2 bytes = 16 units of 16 bytes
  for (int u = threadIdx.x; u < units; u += blockDim.x) {
    const int r = a + u / 16, c = cb * 128 + (u % 16) * 8;
    *(v4u*)((unsigned short*)p.y + (size_t)r * p.y_pitch + c) = (v4u){0u, 0u, 0u, 0u};
  }
}

// A spare workgroup (no expert): its share of the rows no run covers -- [0, off[0]) and [off[n_expert], T) --
// zeroed in its column block. Spares split them round-robin by row.
static __device__ void zero_dead_rows(const Exl3r4dGemm& p, int spare, int nspare, int cb) {
  const int lo = clamp_row(p.off[0], p.T), hi = clamp_row(p.off[p.n_expert], p.T);
  const int dead = lo + (p.T - hi);
  for (int u = threadIdx.x; u < dead * 16; u += blockDim.x) {
    const int d = u / 16;
    if (d % nspare != spare) continue;
    const int r = d < lo ? d : hi + (d - lo), c = cb * 128 + (u % 16) * 8;
    *(v4u*)((unsigned short*)p.y + (size_t)r * p.y_pitch + c) = (v4u){0u, 0u, 0u, 0u};
  }
}

// One expert's planes, resolved once per workgroup (uniform: scalar registers).
struct Planes {
  const unsigned* codes;
  const _Float16 *suh, *suh1, *svh;
  int rows, cols;
};

static __device__ __forceinline__ Planes planes_of(const Exl3r4dGemm& p, int e) {
  // A class picked field by field: indexing the argument block by a runtime index would copy it to the stack.
  const int c = e % p.n_class, j = e / p.n_class;
  const Exl3r4dCls& C0 = p.cls[0];
  const Exl3r4dCls& C1 = p.cls[1];
  Planes q;
  q.codes = (const unsigned*)entry_at(c ? C1.codes : C0.codes, c ? C1.codes_s0 : C0.codes_s0, j);
  q.suh = (const _Float16*)entry_at(c ? C1.suh : C0.suh, c ? C1.suh_s0 : C0.suh_s0, j);
  const void* s1 = c ? C1.suh1 : C0.suh1;
  q.suh1 = s1 ? (const _Float16*)entry_at(s1, c ? C1.suh1_s0 : C0.suh1_s0, j) : q.suh;
  q.svh = (const _Float16*)entry_at(c ? C1.svh : C0.svh, c ? C1.svh_s0 : C0.svh_s0, j);
  q.rows = c ? C1.rows : C0.rows;
  q.cols = c ? C1.cols : C0.cols;
  return q;
}

// ----------------------------------------------------------------------------- body

// One pass over a chunk of an expert's run: rows [rc, rc + chunk) of [rc, r1), one 16-row WMMA block (MTX 1).
// MATS 2: gate/up (EPI 1), MATS 1: down (EPI 2). NPW column tiles a wave; WV = 8 MATS / NPW waves across the
// workgroup's 128 output columns (x MATS matrices), p.sk k-splits; wave = col_local * SK + ksplit.
// RAW (lever L8, the split epilogue): the workgroup owns p.tpw tiles of the class's [MATS][ntiles] tile space instead
// of a 128-column block of every matrix, and stores its raw f32 sums for the finisher (exl3r4d_epi_kernel).
template <int BITS, int NPW, int MATS, bool APREP, bool RAW = false>
static __device__ __forceinline__ void moe_pass(const Exl3r4dGemm& p, const Planes& w, int rc, int r1, float* lds) {
  using P = Plan<BITS>;
  constexpr int NW = P::NW, WORDS = P::WORDS;
  constexpr int WV = RAW ? 8 : 8 * MATS / NPW;   // RAW: p.tpw / NPW at launch; WV is the epilogue's, unused there
  const int SK = p.sk;
  const int tid = threadIdx.x, lane = tid % WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE32);  // keep tile bases in SGPRs
  const int col_local = wave / SK, ksplit = wave % SK;
  const int h = lane >> 4, nl = lane & 15;
  const int width = MATS == 2 ? w.rows / 2 : w.rows;  // this class's output columns
  const int ntiles = width / 16, ksteps = w.cols / 16;
  const int kps = ksteps / SK, ks0 = ksplit * kps;
  const int cb = blockIdx.x;
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
    for (int tsk = wave; tsk < (EXL3R4D_ABLATE_NOPREP ? 0 : tasks); tsk += nwaves) {
      const int i = tsk / per_row, rem = tsk - i * per_row, m = rem / blocks, blk = rem - m * blocks;
      const int r = rc + i, t = p.sorted[r];
      v2u o = {0u, 0u};
      if (t >= 0) {
        // gate_up reads its token's row (a_order token), down the sorted row itself (a_order sorted)
        const int xrow = MATS == 2 ? min(t / p.top_k, p.n_tok - 1) : r;  // clamped: a bad table never reads past a
        o = rotate128((const unsigned short*)p.x + (size_t)xrow * p.x_pitch + blk * 128, p.x_bf16 == 0,
                      (m ? w.suh1 : w.suh) + blk * 128, lane);
      }
      *(v2u*)((_Float16*)lds + ((size_t)(i * MATS + m) * w.cols + blk * 128 + 4 * lane)) = o;
    }
    __syncthreads();
    const int lrow = nl < nrows ? nl : 0;
    aoff = (unsigned)((lrow * MATS + mat) * w.cols + ks0 * 16 + 4 * h);  // in halfs
  } else {
    // The prep kernel's rows: xr[row][mat * cols + k], row pitch xr_pitch.
    const int arow = in_run ? row : rc;
    avo = (unsigned)(((size_t)arow * p.xr_pitch + 4 * h) * 2);
    abase = (gu8)(const unsigned char*)((const _Float16*)p.xr + (size_t)mat * w.cols + (size_t)ks0 * 16);
  }

#if EXL3R4D_EXIT_STAGE == 3
  if (APREP && alds[aoff] == (_Float16)12345.f) *(int*)p.y = 1;
  return;
#endif
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

  v8f acc[NPW];
#pragma unroll
  for (int j = 0; j < NPW; j++) acc[j] = (v8f)(0.f);

  // EXL3_PF stages of weight words and activation fragments in flight: step ks + PF - 1 is loaded while step ks
  // is decoded and multiplied (2 = double buffering).
  constexpr int PF = EXL3_PF;
  unsigned wb[PF][NPW][2][NW];
  v8h ab[PF];

#define WOFF_(r, x) wvo[r][x]
#define EXL3_LOAD(ks, W_, A_)                                                        \
  do {                                                                               \
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
  } while (0)

  // Loads are unconditional (the step index clamps to the last one): with a branch around the prefetch, the
  // compiler cannot count the loads in flight on the join and waits for ALL of them before the next WMMA, which
  // serialized the prefetch with the math every other step (exl3_moe_wmma.hip:714-716; FACTS-exl3 trap 11).
#pragma unroll
  for (int u = 0; u < PF - 1; u++) EXL3_LOAD(u < kps ? u : kps - 1, wb[u], ab[u]);
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
  if constexpr (APREP) __syncthreads();  // the rotated rows' LDS is reused by the reductions and the epilogue

#if EXL3R4D_EXIT_STAGE == 4
  if (SK > 1) lds_barrier();
  if (acc[0][0] == 12345.678f) *(int*)p.y = 1;
  return;
#endif
  if (SK > 1) {  // libr4d's split-K: partials of ksplit > 0 through LDS into the ksplit-0 waves, in wave order
    if (ksplit) {
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        float* dst = lds + ((col_local * NPW + j) * (SK - 1) + (ksplit - 1)) * 256 + lane * 8;
#pragma unroll
        for (int e = 0; e < 8; e++) dst[e] = acc[j][e];
      }
    }
    lds_barrier();
    if (!ksplit) {
#pragma unroll
      for (int j = 0; j < NPW; j++)
        for (int s = 0; s < SK - 1; s++) {
          const float* src = lds + ((col_local * NPW + j) * (SK - 1) + s) * 256 + lane * 8;
#pragma unroll
          for (int e = 0; e < 8; e++) acc[j][e] += src[e];
        }
    }
    lds_barrier();
  }

  if constexpr (RAW) {  // the raw sums, row-major [row][mat * width + n]; the finisher does everything else
    if (!ksplit && in_run) {
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        float* dst = p.raw + (size_t)row * p.raw_pitch + (size_t)mat * width + (nt0 + j) * 16 + 8 * h;
        *(v4f*)dst = (v4f){acc[j][0], acc[j][1], acc[j][2], acc[j][3]};
        *(v4f*)(dst + 4) = (v4f){acc[j][4], acc[j][5], acc[j][6], acc[j][7]};
      }
    }
    if (SK > 1) lds_barrier();  // the next pass's reduction reuses the LDS
    return;
  }
  constexpr float RS = 0.08838834764831845f;  // 1 / sqrt(128)
  // Output tiles t (0..7 of the block) per wave: 8 / WV, or one for the first 8 waves when WV = 16.
  constexpr int OPW = WV >= 8 ? 1 : 8 / WV;
  static_assert(WV >= 1 && WV <= 16 && (8 % WV == 0 || WV == 16), "EPI 1 / 2: 8 * MATS / NPW waves");
  const bool writer = col_local * OPW < 8;
  float* xs = lds;  // [8 MATS tiles][32 lanes][8]
  const float sgn_h = h ? -1.f : 1.f;
  if (!ksplit) {
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      float v[8];
#pragma unroll
      for (int e = 0; e < 8; e++) v[e] = acc[j][e];
#if !EXL3R4D_ABLATE_EPI
      had16_lane(v, sgn_h);
#endif
      lds_put(xs + (col_local * NPW + j) * 256 + lane * 8, v);
    }
  }
  lds_barrier();
  if (!ksplit && writer && in_run) {
#pragma unroll
    for (int o = 0; o < OPW; o++) {
      const int t = col_local * OPW + o;
      const int n0 = cb * 128 + 16 * t + 8 * h;
      if constexpr (MATS == 2) {  // EPI 1: h = f16(silu(gate) * up), gate rows [0, width), up rows [width, 2 width)
        float g[8], u[8], sg[8], su[8];
#if EXL3R4D_ABLATE_EPI
        for (int e = 0; e < 8; e++) { g[e] = xs[t * 256 + lane * 8 + e]; u[e] = xs[(8 + t) * 256 + lane * 8 + e]; }
#else
        had_tiles(xs, t, lane, g);
        had_tiles(xs + 8 * 256, t, lane, u);
#endif
        load_h8(w.svh + n0, sg);
        load_h8(w.svh + width + n0, su);
        union { _Float16 hv[8]; v4u u; } out;
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const float gv = (g[e] * RS) * sg[e], uv = (u[e] * RS) * su[e];
          out.hv[e] = live ? (_Float16)((gv * (1.0f / (1.0f + expf(-gv)))) * uv) : (_Float16)0.f;
        }
        *(v4u*)((_Float16*)p.y + (size_t)row * p.y_pitch + n0) = out.u;
      } else {  // EPI 2: y = bf16((had * 1/sqrt(128)) * svh)
        float r[8], sv[8];
#if EXL3R4D_ABLATE_EPI
        for (int e = 0; e < 8; e++) r[e] = xs[t * 256 + lane * 8 + e];
#else
        had_tiles(xs, t, lane, r);
#endif
        load_h8(w.svh + n0, sv);
        v4u out;
#pragma unroll
        for (int q = 0; q < 4; q++)
          out[q] = live ? (unsigned)f32_to_bf16((r[2 * q] * RS) * sv[2 * q]) |
                              ((unsigned)f32_to_bf16((r[2 * q + 1] * RS) * sv[2 * q + 1]) << 16)
                        : 0u;
        *(v4u*)((unsigned short*)p.y + (size_t)row * p.y_pitch + n0) = out;
      }
    }
  }
  lds_barrier();  // xs (and, next pass, the rotated rows) are rewritten
}

// One workgroup = (128-column block cb = blockIdx.x, the blockIdx.y-th live expert). Its expert's run, a pass a
// chunk; a narrower class's workgroup past its width zero-fills; a spare zeroes rows no expert owns.
template <int BITS, int NPW, int MATS, bool APREP, bool RAW = false>
__global__ void __launch_bounds__(1024) exl3r4d_gemm_kernel(Exl3r4dGemm p) {
  extern __shared__ float lds[];
  __shared__ int s_find[33];
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
    if (!RAW) zero_dead_rows(p, (int)blockIdx.y - run.nlive, (int)gridDim.y - run.nlive, blockIdx.x);
    return;  // RAW: the finisher writes every output row
  }
  const Planes w = planes_of(p, run.e);
#if EXL3R4D_EXIT_STAGE == 2
  if (w.codes == nullptr && w.suh == nullptr && w.svh == nullptr && w.suh1 == nullptr) *(int*)p.y = 1;
  return;
#endif
  const int width = MATS == 2 ? w.rows / 2 : w.rows;
  if (RAW ? (int)blockIdx.x * p.tpw >= MATS * (width / 16) : (int)blockIdx.x * 128 >= width) {
    if (!RAW) zero_rows(p, run.r0, run.r1, blockIdx.x);  // the narrower parity class: zeros past its own width
    return;
  }
  const int chunk = APREP ? p.chunk : EXL3_BLOCK;
  for (int rc = run.r0; rc < run.r1; rc += chunk) moe_pass<BITS, NPW, MATS, APREP, RAW>(p, w, rc, run.r1, lds);
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
  for (int tsk = wave; tsk < tasks; tsk += nwaves) {
#endif
    const int i = tsk / per_row, rem = tsk - i * per_row, m = rem / blocks, blk = rem - m * blocks;
    const int r = run.r0 + i, t = p.sorted[r];
    v2u o = {0u, 0u};
    if (t >= 0) {
      const int xrow = MATS == 2 ? min(t / p.top_k, p.n_tok - 1) : r;
      o = rotate128((const unsigned short*)p.x + (size_t)xrow * p.x_pitch + blk * 128, p.x_bf16 == 0,
                    (m ? w.suh1 : w.suh) + blk * 128, lane);
    }
    *(v2u*)((_Float16*)p.xr + (size_t)r * p.xr_pitch + (size_t)m * w.cols + blk * 128 + 4 * lane) = o;
  }
}

// The finisher of the split epilogue (lever L8): workgroup = (128-column block, live expert), one wave a row: the raw
// sums' output rotation in the reference's own butterfly order (exl3_had128: stages 1, 2, .., 64 in f32, then x
// 1/sqrt(128)), x svh, then h = f16(silu(gate) * up) or y = bf16; dropped rows, rows no run owns and a narrower
// class's columns are written as zeros. Four consecutive columns a lane: bits 0, 1 in the lane, 2..5 through DPP
// row_xmask, 6 through v_permlanex16 -- rotate128's transform, on f32 sums instead of f16(x * suh).
static __device__ __forceinline__ void had128_f32(float (&v)[4], int lane) {
  { const float a = v[0], b = v[1], c = v[2], d = v[3];
    v[0] = a + b; v[1] = a - b; v[2] = c + d; v[3] = c - d;
    const float e0 = v[0], e1 = v[1];
    v[0] = e0 + v[2]; v[1] = e1 + v[3]; v[2] = e0 - v[2]; v[3] = e1 - v[3]; }
  fwht_lane_stage<0x161>(v, (lane & 1) ? -1.f : 1.f);
  fwht_lane_stage<0x162>(v, (lane & 2) ? -1.f : 1.f);
  fwht_lane_stage<0x164>(v, (lane & 4) ? -1.f : 1.f);
  fwht_lane_stage<0x168>(v, (lane & 8) ? -1.f : 1.f);
  const float sgn = (lane & 16) ? -1.f : 1.f;
#pragma unroll
  for (int q = 0; q < 4; q++) v[q] = fmaf(v[q], sgn, x16_f(v[q]));
}

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
  const int64_t per = (int64_t)(gate_up ? 2 : 1) * cols_max * 2;
  const int64_t n = per > 0 ? EXL3R4D_LDS_MAX / per : 0;
  return (int)(n < EXL3_BLOCK ? n : EXL3_BLOCK);
}

extern "C" int64_t exl3r4d_gemm_lds(int gate_up, int npw, int sk, int fused_rows, int64_t cols_max) {
  const int mats = gate_up ? 2 : 1, wv = 8 * mats / npw;
  if (npw != 1 && npw != 2 && npw != 4) return RAD_E_SHAPE;
  if (sk < 1 || wv * sk * WAVE32 > 1024) return RAD_E_SHAPE;
  int64_t need = (int64_t)mats * 8 * 256 * 4;  // the epilogue's tiles
  const int64_t red = red_bytes(wv, npw, sk), rows = (int64_t)fused_rows * mats * cols_max * 2;
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
  const int64_t smem = exl3r4d_gemm_lds(gate_up, npw, a->sk, aprep ? a->chunk : 0, cols_max);
  if (smem < 0) return (int)smem;
  const void* fn = gate_up ? gemm_fn<4, 2>(npw, aprep) : gemm_fn<4, 1>(npw, aprep);
  if (!fn) return RAD_E_SHAPE;
  Exl3r4dGemm p = *a;
  void* args[] = {&p};
  hipStream_t st = (hipStream_t)stream;
  if (!aprep) {
    const void* pf = gate_up ? (const void*)exl3r4d_prep_kernel<2> : (const void*)exl3r4d_prep_kernel<1>;
#if EXL3R4D_PREPGRID
    const unsigned groups = (unsigned)((mats * (cols_max / 128) + 1) / 2);   // two waves (task groups) a WG
    if (hipLaunchKernel(pf, dim3(groups, (unsigned)a->nslot), dim3(2 * WAVE32), args, 0, st) != hipSuccess)
      return RAD_E_DEVICE;
#else
    if (hipLaunchKernel(pf, dim3(a->nslot), dim3(256), args, 0, st) != hipSuccess) return RAD_E_DEVICE;
#endif
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
  const dim3 grid((unsigned)(a->y_cols / 128), (unsigned)a->nslot, 1);
  const dim3 block((unsigned)(wv * a->sk * WAVE32));
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
// STOP (optimize-to-ceiling §3, R22): NOT reached. gate_up tp1 M=1 at 323 GB/s = 48% of 679.5 (libr4d 473 = 70%);
// M=16 511 = 75% (libr4d 596). No single deleted component exceeds 15%; the rest of the K loop's per-workgroup latency
// is "not one thing" by deletion, so the next reading is counter-level (rocprofv3 --pmc / ATT now runnable), and the
// candidate structure is libr4d's: whole-tile 16-byte loads staged through LDS with explicit vmcnt pipelining.
