// exl3r4d_common.hip.h -- the device code every kernel of this library shares: types, the trellis decode in
// registers, the 128-point rotations, WTAB addressing and run finding, the input rotation of one prep task, the LDS
// split-K reduction and the per-16-row-block epilogue. Included by exl3r4d.hip (the decode band) and by the prefill
// band's kernel file (P5). Device code only; nothing here launches. Owner: P4 (exl3r4d.hip's owner) -- a change here
// moves every kernel's bits, so it is re-verified with tests/gpu's device suite and asked for in notes/p4.md.
//
// Provenance of every function: exl3r4d.hip's header (Dylan's exl3_moe_wmma.hip, tcclaviger's libr4d skeleton).
// Everything is `static __device__ __forceinline__`: each kernel file gets its own copy, so one translation unit's
// code object is the same whether or not the other includes this.
#ifndef EXL3R4D_COMMON_HIP_H
#define EXL3R4D_COMMON_HIP_H

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "rad_types.h"
#include "exl3r4d_dev.h"

// Deletion-ablation switches of the shared helpers (readings only, never shipped; tests/gpu/build_variant.sh).
#ifndef EXL3R4D_ABLATE_EPI     // 1: skip the output rotation (had16_lane + had_tiles), store raw sums
#define EXL3R4D_ABLATE_EPI 0
#endif
#ifndef EXL3R4D_ABLATE_PREP    // 1: skip the input rotation (fused prep copies x into LDS unrotated)
#define EXL3R4D_ABLATE_PREP 0
#endif
#ifndef EXL3R4D_ABLATE_DECODE  // 1: skip the trellis decode (the raw words are the fragment)
#define EXL3R4D_ABLATE_DECODE 0
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

// ----------------------------------------------------------------------------- shared pieces of a kernel body

// One prep task: row r's input rotation for matrix m (gate 0 with suh, up 1 with suh.1), 128-feature block blk, by
// one wave: f16(had128(f16(x * suh))), zeros for a dropped row (sorted_tok < 0). gate_up reads its token's row
// (a_order token), down the sorted row itself (a_order sorted); the token row is clamped, so a bad table never reads
// past `a`. The lane's 4 fp16 values packed in two dwords (rotate128's).
template <int MATS>
static __device__ __forceinline__ v2u prep_rotate(const Exl3r4dGemm& p, const Planes& w, int r, int m, int blk,
                                                  int lane) {
  const int t = p.sorted[r];
  v2u o = {0u, 0u};
  if (t >= 0) {
    const int xrow = MATS == 2 ? min(t / p.top_k, p.n_tok - 1) : r;
    o = rotate128((const unsigned short*)p.x + (size_t)xrow * p.x_pitch + blk * 128, p.x_bf16 == 0,
                  (m ? w.suh1 : w.suh) + blk * 128, lane);
  }
  return o;
}

// libr4d's LDS split-K: the partials of waves ksplit > 0 through LDS into the ksplit-0 waves, summed in wave order
// (deterministic). Workgroup-wide (two barriers); every wave calls it. Needs (SK - 1) * NPW * 256 floats a column
// group of LDS at `lds`.
template <int NPW>
static __device__ __forceinline__ void splitk_reduce(v8f (&acc)[NPW], float* lds, int col_local, int ksplit, int SK,
                                                     int lane) {
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

// The split epilogue's raw store (lever L8): this lane's NPW tiles of sums, row-major at raw[row][col0 + 16 j ..],
// col0 = the first tile's column + 8 h within the row's [MATS * width] sums.
template <int NPW>
static __device__ __forceinline__ void raw_store(const Exl3r4dGemm& p, const v8f (&acc)[NPW], int row, int col0) {
#pragma unroll
  for (int j = 0; j < NPW; j++) {
    float* dst = p.raw + (size_t)row * p.raw_pitch + col0 + j * 16;
    *(v4f*)dst = (v4f){acc[j][0], acc[j][1], acc[j][2], acc[j][3]};
    *(v4f*)(dst + 4) = (v4f){acc[j][4], acc[j][5], acc[j][6], acc[j][7]};
  }
}

// The epilogue of one 16-row block (after the split-K reduction), the workgroup owning 128 output columns cb of every
// matrix: WV = 8 MATS / NPW waves of NPW tiles, wave = col_local * SK + ksplit; this lane's row `row` (in the run or
// not; live = sorted_tok >= 0). The 16-point transform in-lane and over the lane pair, the 8-point over tiles through
// LDS (8 MATS tiles of 256 floats at `lds`), x 1/sqrt(128), x svh, then
//   MATS 2 (EPI 1): h = f16(silu(gate) * up), gate rows [0, width), up rows [width, 2 width), f16 [T][y_pitch];
//   MATS 1 (EPI 2): y = bf16(...), bf16 [T][y_pitch];
// a dropped row in the run is written as zeros, rows past the run are not written. Workgroup-wide (two barriers).
template <int NPW, int MATS>
static __device__ __forceinline__ void epilogue_block(const Exl3r4dGemm& p, const Planes& w, const v8f (&acc)[NPW],
                                                      int col_local, int ksplit, int lane, int cb, int row,
                                                      bool in_run, bool live, float* lds) {
  constexpr int WV = 8 * MATS / NPW;
  constexpr float RS = 0.08838834764831845f;  // 1 / sqrt(128)
  // Output tiles t (0..7 of the block) per wave: 8 / WV, or one for the first 8 waves when WV = 16.
  constexpr int OPW = WV >= 8 ? 1 : 8 / WV;
  static_assert(WV >= 1 && WV <= 16 && (8 % WV == 0 || WV == 16), "EPI 1 / 2: 8 * MATS / NPW waves");
  const int h = lane >> 4;
  const int width = MATS == 2 ? w.rows / 2 : w.rows;  // this class's output columns
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

#endif  // EXL3R4D_COMMON_HIP_H
