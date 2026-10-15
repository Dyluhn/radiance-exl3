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
#include "exl3r4d_prefill.h"

// THE CLASS COUNT THIS TRANSLATION UNIT IS BUILT FOR (interface v6). 2: the parity ops' kernels, this source as it
// always was. 4: the _mod4 ops' -- exl3r4d_mod4.hip and exl3r4d_prefill_mod4.hip compile the two kernel files again
// with EXL3R4D_NCLS 4, and the names below make the same code take the four-class argument blocks (exl3r4d_dev.h,
// Exl3r4dGemm4) and export its launches under their own names. Every place the code reads a class goes through
// EXL3R4D_CLS_F / EXL3R4D_CLS_MAX, which at 2 are the field-by-field expressions the parity code always used.
#ifndef EXL3R4D_NCLS
#define EXL3R4D_NCLS 2
#endif
#if EXL3R4D_NCLS == 4
#define Exl3r4dCls Exl3r4dCls4
#define Exl3r4dGemm Exl3r4dGemm4
#define Exl3r4dPrefill Exl3r4dPrefill4
#define exl3r4d_launch_gemm exl3r4d_launch_gemm4
#define exl3r4d_launch_prep exl3r4d_launch_prep4
#define exl3r4d_launch_prep_form exl3r4d_launch_prep_form4
#define exl3r4d_launch_prefill exl3r4d_launch_prefill4
// Class c's record, read straight from the kernel-argument segment (see planes_of): host code and constant indices
// read the block as they always did.
#define EXL3R4D_CLS_F(p, c, f) exl3r4d_karg_cls(c).f
static __host__ __device__ __forceinline__ int32_t exl3r4d_max4(int32_t a, int32_t b, int32_t c, int32_t d) {
  const int32_t ab = a > b ? a : b, cd = c > d ? c : d;
  return ab > cd ? ab : cd;
}
#define EXL3R4D_CLS_MAX(p, f) exl3r4d_max4((p).cls[0].f, (p).cls[1].f, (p).cls[2].f, (p).cls[3].f)
#elif EXL3R4D_NCLS == 2
#define EXL3R4D_CLS_F(p, c, f) ((c) ? (p).cls[1].f : (p).cls[0].f)
#define EXL3R4D_CLS_MAX(p, f) ((p).cls[0].f > (p).cls[1].f ? (p).cls[0].f : (p).cls[1].f)
#else
#error "EXL3R4D_NCLS is 2 (the parity ops) or 4 (the _mod4 ops)"
#endif

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

// lane ^ 16 for the epilogue's 16-point stage. __shfl_xor compiles to ds_bpermute_b32 (an LDS-crossbar round trip a
// value, 16 a wave a tile pair); EXL3R4D_EPI_PLX 1 uses v_permlanex16_b32, one VALU op, as rotate128's x16_f does --
// the same values moved, bit-identical.
#ifndef EXL3R4D_EPI_PLX  // REFUTED (runs/plx): down 0.00..+0.37 us, gate_up -0.11 at tp2 M=1..16 -- not on the path
#define EXL3R4D_EPI_PLX 0
#endif
static __device__ __forceinline__ float swap16(float x) {
  if constexpr (EXL3R4D_EPI_PLX)
    return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(__builtin_bit_cast(int, x), __builtin_bit_cast(int, x),
                                                                  0x76543210, (int)0xFEDCBA98, false, false));
  return __shfl_xor(x, 16, 32);
}

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
// each stage is exact-sign a +- b. NOT bit-identical to the reference's prologue: the final x 1/sqrt(128) and fp32
// rounding inside the transform differ in the last place on ~4e-5 of elements (1 f16 ulp, P5's measurement); every
// device path (prep kernel, in-GEMM rotation, gate_up's h_rot epilogue) uses this same function, so the device's
// h_rot pair is bit-identical to its h_rot = 0 pair. Feature bits 0, 1
// in the lane, 2..5 through DPP row_xmask, 6 through v_permlanex16_b32. Verbatim from exl3_moe_wmma.hip:275-306.
// rotate128 in two halves for a pipelined caller (the staged K loop, lever L11): the lane's loads, then the math on
// what they returned. rotate128 itself is the two back to back.
static __device__ __forceinline__ v2u rotate128_math(v2u xr, v2u sr, bool x_f16, int lane);
static __device__ __forceinline__ v2u rotate128(const unsigned short* x_blk, bool x_f16, const _Float16* suh_blk,
                                                int lane) {
  const v2u xr = *(const v2u*)(x_blk + 4 * lane);
  const v2u sr = *(const v2u*)(suh_blk + 4 * lane);
  return rotate128_math(xr, sr, x_f16, lane);
}
static __device__ __forceinline__ v2u rotate128_math(v2u xr, v2u sr, bool x_f16, int lane) {
  float v[4];
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

#ifndef EXL3R4D_KX_PAIR  // R23: K != 4 states hashed in packed pairs too (1) or one v_mul_lo_u32 each (0)
#define EXL3R4D_KX_PAIR 1
#endif
#ifndef EXL3_HASH
#define EXL3_HASH 2  // 1: one v_mul_lo_u32 a state (the reference form); 2 (adopted, V1a): states extracted in packed pairs (v_perm_b32) and hashed two at a time (playbook V1a)
#endif
// EXL3_HASH 2: two 16-bit states packed in one dword (a | b << 16) -> their two 32-bit hashes, exactly
// state * 0x83DCD12D mod 2^32 = state * 0xD12D + (lo16(state * 0x83DC) << 16): one v_pk_mul_lo_u16 for both high
// parts, then one v_mad_u32_u16 each (op_sel picks the high state). Five full-rate ops for two states against two
// quarter-rate v_mul_lo_u32 (8 cycles). Inline assembly: the combiner folds the plain-C form back into v_mul_lo_u32.
// V1CD (second opinion, notes/decode-second-opinion.md): V1c -- the two high products come out of v_pk_mul_lo_u16
// already in place: against 0x83DC0000 the low lane multiplies by 0, op_sel_hi:[0,1] puts lo16(a 0x83DC) in the high
// half and the default lo16(b 0x83DC) -- no shift, no mask (the same integers for every input, every K). V1d (K = 4,
// decode_run): the byte-aligned state pair straight from the two loaded words with one v_perm_b32 (s is 0 or 16 at
// K = 4, so the selector is a lane constant) and lo >> 4 as one alignbit at s + 4.
#ifndef EXL3R4D_V1CD  // ADOPTED (runs/v1cd): 312/312 suite outputs byte-identical; chain tp2 -1.13 / -2.10 / -2.56 us
#define EXL3R4D_V1CD 1  // at M=1/2/4 (M=8 unchanged); gate_up K loop 405 -> 354 a 8 tiles, down 206 -> 182 a 4
#endif
static __device__ __forceinline__ void hash_pair(unsigned ab, unsigned& ha, unsigned& hb) {
#if EXL3R4D_V1CD
  unsigned pa, pb;
  asm("v_pk_mul_lo_u16 %0, %1, %2 op_sel_hi:[0,1]" : "=v"(pa) : "v"(ab), "s"(0x83DC0000u));
  asm("v_pk_mul_lo_u16 %0, %1, %2" : "=v"(pb) : "v"(ab), "s"(0x83DC0000u));
#else
  unsigned ph;
  asm("v_pk_mul_lo_u16 %0, %1, %2" : "=v"(ph) : "v"(ab), "s"(0x83DC83DCu));
  const unsigned pa = ph << 16, pb = ph & 0xFFFF0000u;
#endif
  asm("v_mad_u32_u16 %0, %1, %2, %3" : "=v"(ha) : "v"(ab), "s"(0xD12Du), "v"(pa));
  asm("v_mad_u32_u16 %0, %1, %2, %3 op_sel:[1,0,0,0]" : "=v"(hb) : "v"(ab), "s"(0xD12Du), "v"(pb));
}

// The run's four positions q0..q0 + 3 -> two dwords of fp16 pairs (q0, q0 + 1), (q0 + 2, q0 + 3).
template <int BITS, int NW>
static __device__ __forceinline__ void decode_run(const unsigned (&w)[NW], unsigned s, unsigned& p01, unsigned& p23) {
  const unsigned lo = __builtin_amdgcn_alignbit(w[NW - 2], w[NW - 1], s);  // the 32 stream bits ending at e3
  if constexpr (EXL3_HASH == 2 && BITS == 4) {
    // K = 4: state i is lo[(3 - i) 4, +16). (s3 | s1 << 16) is bytes 0,1 | 1,2 of lo; (s2 | s0 << 16) the same of
    // lo >> 4. Then the codebook exactly as codebook_pair does it.
#if EXL3R4D_V1CD
    const unsigned p31 = __builtin_amdgcn_perm(w[NW - 2], w[NW - 1], 0x02010100u + (s >> 4) * 0x02020202u);
    const unsigned lo4 = __builtin_amdgcn_alignbit(w[NW - 2], w[NW - 1], s + 4u);
#else
    const unsigned p31 = __builtin_amdgcn_perm(lo, lo, 0x02010100u);
    const unsigned lo4 = lo >> 4;
#endif
    const unsigned p20 = __builtin_amdgcn_perm(lo4, lo4, 0x02010100u);
    unsigned h0, h1, h2, h3;
    hash_pair(p31, h3, h1);
    hash_pair(p20, h2, h0);
    const _Float16 kinv = __builtin_bit_cast(_Float16, (unsigned short)0x1EEE);
    const _Float16 kbias = __builtin_bit_cast(_Float16, (unsigned short)0xC931);
    const unsigned a01 = sad_hi(h1, sad_lo(h0, 0x64006400u)), a23 = sad_hi(h3, sad_lo(h2, 0x64006400u));
    p01 = __builtin_bit_cast(unsigned, __builtin_elementwise_fma(__builtin_bit_cast(v2h, a01), (v2h){kinv, kinv},
                                                                 (v2h){kbias, kbias}));
    p23 = __builtin_bit_cast(unsigned, __builtin_elementwise_fma(__builtin_bit_cast(v2h, a23), (v2h){kinv, kinv},
                                                                 (v2h){kbias, kbias}));
    return;
  }
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
  if constexpr (EXL3_HASH == 2 && EXL3R4D_KX_PAIR) {
    // R23, any K: V1a's packed hash on the extracted states -- (s3 | s1 << 16) and (s2 | s0 << 16), one
    // v_lshl_or each, then hash_pair and the codebook exactly as the K = 4 form (bit-identical to codebook_pair).
    unsigned h0, h1, h2, h3;
    hash_pair(st[3] | st[1] << 16, h3, h1);
    hash_pair(st[2] | st[0] << 16, h2, h0);
    const _Float16 kinv = __builtin_bit_cast(_Float16, (unsigned short)0x1EEE);
    const _Float16 kbias = __builtin_bit_cast(_Float16, (unsigned short)0xC931);
    const unsigned a01 = sad_hi(h1, sad_lo(h0, 0x64006400u)), a23 = sad_hi(h3, sad_lo(h2, 0x64006400u));
    p01 = __builtin_bit_cast(unsigned, __builtin_elementwise_fma(__builtin_bit_cast(v2h, a01), (v2h){kinv, kinv},
                                                                 (v2h){kbias, kbias}));
    p23 = __builtin_bit_cast(unsigned, __builtin_elementwise_fma(__builtin_bit_cast(v2h, a23), (v2h){kinv, kinv},
                                                                 (v2h){kbias, kbias}));
    return;
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

// EXL3R4D_FR_LDSB (radiance 1.4.0's lever 2; the default since engine v14): the run search's two barriers exchange LDS
// only, so they need not be __syncthreads() (on gfx12 also a wait for every load and a vector-cache invalidate at
// kernel start). 0: __syncthreads().
#ifndef EXL3R4D_FR_LDSB
#define EXL3R4D_FR_LDSB 1
#endif
#define FR_BAR() do { if (EXL3R4D_FR_LDSB) lds_barrier(); else __syncthreads(); } while (0)
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
  FR_BAR();
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
  FR_BAR();
  Run r;
  r.e = sh[32];
  r.nlive = total;
  r.r0 = r.e >= 0 ? clamp_row(p.off[r.e], p.T) : 0;
  r.r1 = r.e >= 0 ? clamp_row(p.off[r.e + 1], p.T) : 0;
  return r;
}

// FRW: the same answer as find_run with no LDS and no barrier, for n_expert <= 512 -- every wave on its own: lane l
// loads expert_offset[16 l .. 16 l + 16] (one round trip), counts its live experts, a wave prefix numbers them, the
// lane whose range holds `slot` (a ballot) picks the expert from its registers, readlane hands it to the wave. The
// old chain is two global round trips and two barriers before the first weight load can issue. Callers opt in
// (find_run_any); P5's prefill keeps find_run.
static __device__ __forceinline__ Run find_run_wave(const Exl3r4dGemm& p, int slot) {
  const int lane = (int)(threadIdx.x % WAVE32), n = p.n_expert, e0 = lane * 16;
  int o[17];
#pragma unroll
  for (int i = 0; i < 17; i++) o[i] = clamp_row(p.off[min(e0 + i, n)], p.T);
  int cnt = 0;
#pragma unroll
  for (int i = 0; i < 16; i++) cnt += (e0 + i < n && o[i + 1] > o[i]) ? 1 : 0;
  int incl = cnt;
#pragma unroll
  for (int d = 1; d < WAVE32; d <<= 1) {
    const int v = __shfl_up(incl, d, WAVE32);
    if (lane >= d) incl += v;
  }
  const int total = __builtin_amdgcn_readlane(incl, WAVE32 - 1), excl = incl - cnt;
  const unsigned hit = (unsigned)__builtin_amdgcn_ballot_w32(slot >= excl && slot < incl);
  Run r;
  r.nlive = total;
  r.e = -1;
  r.r0 = r.r1 = 0;
  if (hit) {
    const int L = __builtin_ctz(hit);
    int e = -1, a = 0, b = 0, k = slot - excl;
#pragma unroll
    for (int i = 0; i < 16; i++) {
      const bool lv = e0 + i < n && o[i + 1] > o[i];
      if (lv && k == 0 && e < 0) { e = e0 + i; a = o[i]; b = o[i + 1]; }
      k -= lv ? 1 : 0;
    }
    r.e = __builtin_amdgcn_readlane(e, L);
    r.r0 = __builtin_amdgcn_readlane(a, L);
    r.r1 = __builtin_amdgcn_readlane(b, L);
  }
  return r;
}

#ifndef EXL3R4D_FRW  // REFUTED (runs/frw): gate_up +1.5..+2.4 us, down +0.4..+3.9 at tp2 M=1..16
#define EXL3R4D_FRW 0
#endif

// FRS: find_run's block algorithm (the same two barriers, the same answer) without its serial memory chain: a
// thread's per + 1 offsets loaded once, unconditionally (clamped index) and all before any use, the count and the
// k-th live expert taken from registers, and the chosen expert's [r0, r1) handed through LDS with its id -- not
// re-loaded by every thread after the barrier. find_run walks its range with a load and a wait a step (per = 2 at
// 256 threads, 4 at 128), re-scans it from memory in the matching thread, and loads off[e], off[e + 1] after the
// second barrier: about six dependent round trips before the first weight load, against one here.
static __device__ __forceinline__ Run find_run_regs(const Exl3r4dGemm& p, int slot, int* sh) {
  constexpr int MAXP = 8;
  const int tid = threadIdx.x, nt = blockDim.x, lane = tid % WAVE32, wave = tid / WAVE32, nw = nt / WAVE32;
  const int n = p.n_expert, per = (n + nt - 1) / nt, e0 = tid * per;
  int o[MAXP + 1];
#pragma unroll
  for (int i = 0; i <= MAXP; i++) o[i] = clamp_row(p.off[min(e0 + min(i, per), n)], p.T);
  int cnt = 0;
#pragma unroll
  for (int i = 0; i < MAXP; i++) cnt += (i < per && e0 + i < n && o[i + 1] > o[i]) ? 1 : 0;
  int incl = cnt;
#pragma unroll
  for (int d = 1; d < WAVE32; d <<= 1) {
    const int v = __shfl_up(incl, d, WAVE32);
    if (lane >= d) incl += v;
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
    int k = slot - excl, e = -1, a = 0, b = 0;
#pragma unroll
    for (int i = 0; i < MAXP; i++) {
      const bool lv = i < per && e0 + i < n && o[i + 1] > o[i];
      if (lv && k == 0 && e < 0) { e = e0 + i; a = o[i]; b = o[i + 1]; }
      k -= lv ? 1 : 0;
    }
    sh[32] = e;
    sh[33] = a;
    sh[34] = b;
  }
  __syncthreads();
  Run r;
  r.e = sh[32];
  r.nlive = total;
  r.r0 = r.e >= 0 ? sh[33] : 0;
  r.r1 = r.e >= 0 ? sh[34] : 0;
  return r;
}
#ifndef EXL3R4D_FRS  // REFUTED (runs/frs): gate_up -0.02..-0.41 us, down -0.3..-0.7 / +0.8 at M=16: the search is
#define EXL3R4D_FRS 0   // already off the critical path
#endif

static __device__ __forceinline__ Run find_run_any(const Exl3r4dGemm& p, int slot, int* sh) {
  if (EXL3R4D_FRW && p.n_expert <= 16 * WAVE32) return find_run_wave(p, slot);   // uniform
  if (EXL3R4D_FRS && p.n_expert <= 8 * (int)blockDim.x) return find_run_regs(p, slot, sh);   // uniform
  return find_run(p, slot, sh);
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
  const _Float16* dn_suh;  // gate_up with h_rot (v3 EPI 1): down's suh of this expert; null otherwise
  int rows, cols;
};

// P10 L1 (EXL3R4D_KARG_PF, four-class build only): the class records into the scalar cache at every kernel's entry.
// planes_of reads record c only after the expert search names the expert, so without it that read is a dependent
// memory round trip on each kernel's critical path -- measured on the same work against the two-class twins (P10 R1,
// 0000:13:00.0, M=1, rocprofv3): +0.96 / +1.00 / +0.92 us in prep / gate_up / down, +2.8..+4.3 us a chain at M=1..4.
// The prefetch issues at entry and the search hides it. Bit-identical: it moves no data.
// REFUTED, OFF (P10 R2, 0000:13:00.0, ABAB x3 against the RC library): the four-class chain 2.5x SLOWER -- tp4 rank 0
// M=1 38.25 -> 99.35 us, M=4 G4 62.12 -> 172.33, M=16 176.16 -> 391.88; the two-class control unchanged (35.15 / 35.07).
// The prefetch takes the kernarg pointer as an operand, after which the class fields' loads are no longer invariant
// scalar loads (+8 VGPRs a GEMM kernel, twin_check 54 costlier): the divergent-pointer waterfall d02fd56 removed. Reopen
// only with a prefetch the compiler does not treat as an escape of the kernarg pointer.
#ifndef EXL3R4D_KARG_PF
#define EXL3R4D_KARG_PF 0
#endif
#if EXL3R4D_NCLS == 4 && EXL3R4D_KARG_PF
#define EXL3R4D_KARG_PREFETCH()                                                                                          \
  __builtin_amdgcn_s_prefetch_data((const __attribute__((address_space(4))) void*)__builtin_amdgcn_kernarg_segment_ptr(), \
                                   offsetof(Exl3r4dGemm4, cls) + sizeof(((Exl3r4dGemm4*)0)->cls))
#else
#define EXL3R4D_KARG_PREFETCH() ((void)0)
#endif
#if EXL3R4D_NCLS == 4
typedef const __attribute__((address_space(4))) Exl3r4dCls4 KargCls4;
// Record c of the argument block at the start of the kernel-argument segment (planes_of says why).
static __device__ __forceinline__ KargCls4& exl3r4d_karg_cls(int c) {
  typedef const __attribute__((address_space(4))) char KargByte;
  return ((KargCls4*)((KargByte*)__builtin_amdgcn_kernarg_segment_ptr() + offsetof(Exl3r4dGemm4, cls)))[c];
}
#endif
static __device__ __forceinline__ Planes planes_of(const Exl3r4dGemm& p, int e) {
  // A class picked field by field: indexing the argument block by a runtime index would copy it to the stack.
  Planes q;
#if EXL3R4D_NCLS == 4
  // AT FOUR CLASSES THE RECORD IS READ FROM THE KERNEL-ARGUMENT SEGMENT ITSELF, at a uniform index: a scalar load
  // from constant memory, nothing copied. Selecting among the four records' fields instead (a select chain, or
  // readfirstlane'd values) cost 2-3x the parity kernels' time, measured: LLVM folds a four-way select of loads into
  // a load at a selected address, which keeps the whole argument block in scratch (464 B), and readfirstlane'd
  // fields came back as divergent pointers with every buffer load in a waterfall loop. `p` is the kernel's own
  // argument block, which every four-class kernel takes first (the decode kernels' Exl3r4dGemm4, the prefill
  // kernels' Exl3r4dPrefill4 whose first member it is), so its records start where the segment does.
  (void)p;
  const int c = e & 3, j = e >> 2;   // the four-class blocks always carry four classes (launch.c)
  const KargCls4& C = exl3r4d_karg_cls(c);
  q.codes = (const unsigned*)entry_at(C.codes, C.codes_s0, j);
  q.suh = (const _Float16*)entry_at(C.suh, C.suh_s0, j);
  q.suh1 = C.suh1 ? (const _Float16*)entry_at(C.suh1, C.suh1_s0, j) : q.suh;
  q.svh = (const _Float16*)entry_at(C.svh, C.svh_s0, j);
  q.dn_suh = p.h_rot && C.dn_suh ? (const _Float16*)entry_at(C.dn_suh, C.dn_suh_s0, j) : nullptr;
  q.rows = C.rows;
  q.cols = C.cols;
#else
  const int c = e % p.n_class, j = e / p.n_class;
  const Exl3r4dCls& C0 = p.cls[0];
  const Exl3r4dCls& C1 = p.cls[1];
  q.codes = (const unsigned*)entry_at(c ? C1.codes : C0.codes, c ? C1.codes_s0 : C0.codes_s0, j);
  q.suh = (const _Float16*)entry_at(c ? C1.suh : C0.suh, c ? C1.suh_s0 : C0.suh_s0, j);
  const void* s1 = c ? C1.suh1 : C0.suh1;
  q.suh1 = s1 ? (const _Float16*)entry_at(s1, c ? C1.suh1_s0 : C0.suh1_s0, j) : q.suh;
  q.svh = (const _Float16*)entry_at(c ? C1.svh : C0.svh, c ? C1.svh_s0 : C0.svh_s0, j);
  const void* dn = c ? C1.dn_suh : C0.dn_suh;
  q.dn_suh = p.h_rot && dn ? (const _Float16*)entry_at(dn, c ? C1.dn_suh_s0 : C0.dn_suh_s0, j) : nullptr;
  q.rows = c ? C1.rows : C0.rows;
  q.cols = c ? C1.cols : C0.cols;
#endif
  return q;
}

// L7-2 (EXL3R4D_FRP): find_run_regs's search -- every offset of a thread's range loaded once, before any use -- with the
// chosen expert's PLANE POINTERS requested during it: each thread with a live expert loads that range's first live
// expert's table entries (planes_of) before the first barrier, the thread whose range holds the slot hands them through
// LDS with the id and [r0, r1) (loading them then only when the slot is not its range's first live expert), and the
// workgroup reads them back as uniform values. find_run then planes_of is the search's round trips and, after them, one
// more dependent round trip for the table entries before the first weight load can issue (P9-L7 H6, sH-1: ex2 - ex1 =
// 0.7-1.1 us a down dispatch, paid by every unit of both rounds). The same expert, rows and pointers: bit-identical.
// REFUTED (P9-L7 sJ-1, 0000:03:00.0, frp vs the same tree without it, ABAB x3; y byte-identical at tp1/tp2 M=1..4):
// SLOWER everywhere -- gate_up +0.2..+0.8 us, down +0.2..+0.6, split chain tp2 M=1/2/3 +1.1/+1.2/+1.0, tp1 M=1/2
// +1.1/+2.2. The round trip it removes is not on the dispatch's critical path (as FRS found: the search overlaps the
// other workgroups' streams); every live expert's thread now issues five table loads and the kernel grows ~1150
// instructions. Off. Reopen only with a reading that shows units waiting on their first weight load.
#ifndef EXL3R4D_FRP
#define EXL3R4D_FRP 0
#endif
static __device__ __forceinline__ const void* ptr_uniform(const void* v) {   // both halves unsigned (WGB's fault)
  const unsigned long long x = (unsigned long long)v;
  const unsigned lo = (unsigned)__builtin_amdgcn_readfirstlane((int)(unsigned)x);
  const unsigned hi = (unsigned)__builtin_amdgcn_readfirstlane((int)(unsigned)(x >> 32));
  return (const void*)(((unsigned long long)hi << 32) | lo);
}
static __device__ __forceinline__ Run find_run_planes(const Exl3r4dGemm& p, int slot, int* sh, Planes& w) {
  constexpr int MAXP = 8;
  __shared__ const void* s_pl[5];
  const int tid = threadIdx.x, nt = blockDim.x, lane = tid % WAVE32, wave = tid / WAVE32, nw = nt / WAVE32;
  const int n = p.n_expert, per = (n + nt - 1) / nt, e0 = tid * per;
  int o[MAXP + 1];
#pragma unroll
  for (int i = 0; i <= MAXP; i++) o[i] = clamp_row(p.off[min(e0 + min(i, per), n)], p.T);
  int cnt = 0, ef = -1;
#pragma unroll
  for (int i = 0; i < MAXP; i++) {
    const bool lv = i < per && e0 + i < n && o[i + 1] > o[i];
    cnt += lv ? 1 : 0;
    if (lv && ef < 0) ef = e0 + i;
  }
  Planes q = {};
  if (ef >= 0) q = planes_of(p, ef);  // in flight across the prefix sum and the barrier
  int incl = cnt;
#pragma unroll
  for (int d = 1; d < WAVE32; d <<= 1) {
    const int v = __shfl_up(incl, d, WAVE32);
    if (lane >= d) incl += v;
  }
  if (tid == 0) sh[32] = -1;
  if (lane == WAVE32 - 1) sh[wave] = incl;
  FR_BAR();
  int base = 0, total = 0;
  for (int v = 0; v < nw; v++) {
    const int c = sh[v];
    base += v < wave ? c : 0;
    total += c;
  }
  const int excl = base + incl - cnt;
  if (cnt && slot >= excl && slot < excl + cnt) {
    int k = slot - excl, e = -1, a = 0, b = 0;
#pragma unroll
    for (int i = 0; i < MAXP; i++) {
      const bool lv = i < per && e0 + i < n && o[i + 1] > o[i];
      if (lv && k == 0 && e < 0) { e = e0 + i; a = o[i]; b = o[i + 1]; }
      k -= lv ? 1 : 0;
    }
    if (e != ef) q = planes_of(p, e);  // the slot is a later live expert of this range (rare at decode M)
    sh[32] = e;
    sh[33] = a;
    sh[34] = b;
    s_pl[0] = q.codes;
    s_pl[1] = q.suh;
    s_pl[2] = q.suh1;
    s_pl[3] = q.svh;
    s_pl[4] = q.dn_suh;
  }
  FR_BAR();
  Run r;
  r.e = sh[32];
  r.nlive = total;
  r.r0 = r.e >= 0 ? sh[33] : 0;
  r.r1 = r.e >= 0 ? sh[34] : 0;
  if (r.e >= 0) {
    const int c = r.e % p.n_class;
    w.codes = (const unsigned*)ptr_uniform(s_pl[0]);
    w.suh = (const _Float16*)ptr_uniform(s_pl[1]);
    w.suh1 = (const _Float16*)ptr_uniform(s_pl[2]);
    w.svh = (const _Float16*)ptr_uniform(s_pl[3]);
    w.dn_suh = (const _Float16*)ptr_uniform(s_pl[4]);
    w.rows = c ? p.cls[1].rows : p.cls[0].rows;
    w.cols = c ? p.cls[1].cols : p.cls[0].cols;
  }
  return r;
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
#ifndef EXL3R4D_XROW0
#define EXL3R4D_XROW0 0
#endif
static __device__ __forceinline__ v2u prep_rotate(const Exl3r4dGemm& p, const Planes& w, int r, int m, int blk,
                                                  int lane) {
  const int t = p.sorted[r];
  v2u o = {0u, 0u};
  if (t >= 0) {
#if EXL3R4D_XROW0  // reading only (wrong output): the token -> row arithmetic deleted, to test its effect on the schedule
    const int xrow = 0;
#else
    const int xrow = MATS == 2 ? min(t / p.top_k, p.n_tok - 1) : r;
#endif
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
// (epilogue_block follows svh_early below.)
//
// v3 EPI 1: gate_up's h block staged at this LDS byte offset (16 rows x 128 f16 = 4 KB, past the epilogue's 16 KB
// of tiles; the launcher allocates kHrotStage + 4096 at least).
constexpr int kHrotStage = 16384;

// Output tiles t (0..7 of a 128-column block) a wave writes in the epilogue: 8 / WV, or one for the first 8 waves when
// WV = 8 MATS / NPW is 16.
template <int NPW, int MATS>
static __host__ __device__ constexpr int epi_opw() { return 8 * MATS / NPW >= 8 ? 1 : 8 / (8 * MATS / NPW); }

// Playbook E1 (EXL3R4D_SVH_EARLY): the epilogue's svh loads issued in the prologue, before the K loop, and carried
// as raw f16 (svp[o][m]: output tile o of the wave, matrix m), instead of a global round trip after the reduction
// barrier on every workgroup's tail. A wave past the block's 8 tiles (WV 16) loads tile 7's: never used.
template <int NPW, int MATS>
static __device__ __forceinline__ void svh_early(const Planes& w, int col_local, int lane, int cb,
                                                 v8h (&svp)[epi_opw<NPW, MATS>()][MATS]) {
  typedef const __attribute__((address_space(1))) v8h* gv8h;
  constexpr int OPW = epi_opw<NPW, MATS>();
  const int width = MATS == 2 ? w.rows / 2 : w.rows, h = lane >> 4;
#pragma unroll
  for (int o = 0; o < OPW; o++) {
    const int t = min(col_local * OPW + o, 7), n0 = cb * 128 + 16 * t + 8 * h;
#pragma unroll
    for (int m = 0; m < MATS; m++) svp[o][m] = *(gv8h)(w.svh + m * width + n0);
  }
}

// EPIB (opt-in, `EPIB`; down's decode GEMM at NPW 2 / split-K 1, the caller's choice): the 8-point transform over
// the block's tiles as a BUTTERFLY instead of had_tiles' 8-term sums -- a wave holds tiles 2c and 2c + 1, so the
// first stage (tile bit 0) is in registers and the other two are one LDS exchange each with the partner wave c ^ 1,
// then c ^ 2 (write its own 16 floats a lane, barrier, read the partner's; the half with the bit set takes partner
// - mine). 48 adds and 2 x 32 floats of LDS a lane instead of 128 FMAs and 16 + 128; each wave stores its own tiles.
// The sums are in the butterfly order (exllamav3's, the transcription's), not had_tiles' sequential one: bits move
// inside the tolerance (a numerics decision). LDS: two [8][32][8] f32 exchange buffers, 16 KB (exl3r4d_gemm_lds).
// Callers that do not ask (P5's prefill) keep had_tiles.
// EPIC (opt-in, `EPIC`; down, two output tiles a wave): the wave's tiles 2c and 2c + 1 differ only in tile bit 0, so
// both come from ONE pass over the 8 staged tiles -- E = the even tiles' signed sum, O = the odd tiles', out(2c) =
// E + O, out(2c + 1) = E - O: 16 LDS loads and 64 FMAs + 16 adds a lane instead of had_tiles twice (32 and 128).
// The sum is paired (E, O) instead of sequential over all 8: bits move inside the tolerance (a numerics decision).
template <int NPW, int MATS, bool PRE = false, bool EPIB = false, bool EPIC = false>
static __device__ __forceinline__ void epilogue_block(const Exl3r4dGemm& p, const Planes& w, const v8f (&acc)[NPW],
                                                      int col_local, int ksplit, int lane, int cb, int row,
                                                      bool in_run, bool live, float* lds,
                                                      const v8h (*svp)[MATS] = nullptr) {
  constexpr int WV = 8 * MATS / NPW;
  constexpr float RS = 0.08838834764831845f;  // 1 / sqrt(128)
  constexpr int OPW = epi_opw<NPW, MATS>();
  static_assert(WV >= 1 && WV <= 16 && (8 % WV == 0 || WV == 16), "EPI 1 / 2: 8 * MATS / NPW waves");
  const int h = lane >> 4;
  const int width = MATS == 2 ? w.rows / 2 : w.rows;  // this class's output columns
  const bool writer = col_local * OPW < 8;
  float* xs = lds;  // [8 MATS tiles][32 lanes][8]
  const float sgn_h = h ? -1.f : 1.f;
  if constexpr (EPIB && MATS == 1 && NPW == 2 && !PRE) {
    if (p.sk == 1) {  // workgroup-uniform: every wave holds two whole tiles
      float v[2][8];
#pragma unroll
      for (int j = 0; j < 2; j++) {
#pragma unroll
        for (int e = 0; e < 8; e++) v[j][e] = acc[j][e];
        had16_lane(v[j], sgn_h);
      }
#pragma unroll
      for (int e = 0; e < 8; e++) {  // tile bit 0: this wave's own pair
        const float a = v[0][e], b = v[1][e];
        v[0][e] = a + b;
        v[1][e] = a - b;
      }
#pragma unroll
      for (int s = 0; s < 2; s++) {  // tile bits 1 and 2: the partner waves c ^ 1, c ^ 2
        float* xb = lds + s * 2048;
        const int pc = col_local ^ (1 << s);
        const bool hi = (col_local >> s) & 1;
#pragma unroll
        for (int j = 0; j < 2; j++) lds_put(xb + (col_local * 2 + j) * 256 + lane * 8, v[j]);
        lds_barrier();
#pragma unroll
        for (int j = 0; j < 2; j++) {
          const v4f a = *(const v4f*)(xb + (pc * 2 + j) * 256 + lane * 8);
          const v4f b = *(const v4f*)(xb + (pc * 2 + j) * 256 + lane * 8 + 4);
#pragma unroll
          for (int e = 0; e < 4; e++) {
            v[j][e] = hi ? a[e] - v[j][e] : v[j][e] + a[e];
            v[j][4 + e] = hi ? b[e] - v[j][4 + e] : v[j][4 + e] + b[e];
          }
        }
      }
      if (in_run) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
          const int n0 = cb * 128 + 16 * (col_local * 2 + j) + 8 * h;
          float sv[8];
          load_h8(w.svh + n0, sv);
          v4u out;
#pragma unroll
          for (int q = 0; q < 4; q++)
            out[q] = live ? (unsigned)f32_to_bf16((v[j][2 * q] * RS) * sv[2 * q]) |
                                ((unsigned)f32_to_bf16((v[j][2 * q + 1] * RS) * sv[2 * q + 1]) << 16)
                          : 0u;
          *(v4u*)((unsigned short*)p.y + (size_t)row * p.y_pitch + n0) = out;
        }
      }
      lds_barrier();  // the exchange buffers are rewritten next pass
      return;
    }
  }
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
  [[maybe_unused]] float epc_e[8], epc_o[8];  // EPIC: the even / odd tiles' sums
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
        if constexpr (PRE) {
#pragma unroll
          for (int e = 0; e < 8; e++) { sg[e] = (float)svp[o][0][e]; su[e] = (float)svp[o][1][e]; }
        } else {
          load_h8(w.svh + n0, sg);
          load_h8(w.svh + width + n0, su);
        }
        union { _Float16 hv[8]; v4u u; } out;
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const float gv = (g[e] * RS) * sg[e], uv = (u[e] * RS) * su[e];
          out.hv[e] = live ? (_Float16)((gv * (1.0f / (1.0f + expf(-gv)))) * uv) : (_Float16)0.f;
        }
        if (p.h_rot)  // v3 EPI 1: staged, rotated below with down's suh before it is stored
          *(v4u*)((_Float16*)((unsigned char*)lds + kHrotStage) + (lane & 15) * 128 + 16 * t + 8 * h) = out.u;
        else
          *(v4u*)((_Float16*)p.y + (size_t)row * p.y_pitch + n0) = out.u;
      } else {  // EPI 2: y = bf16((had * 1/sqrt(128)) * svh)
        float r[8], sv[8];
#if EXL3R4D_ABLATE_EPI
        for (int e = 0; e < 8; e++) r[e] = xs[t * 256 + lane * 8 + e];
#else
        if constexpr (EPIC && OPW == 2) {  // both of the wave's tiles from one pass (o = 0 computes, o = 1 reuses)
          if (o == 0) {
#pragma unroll
            for (int e = 0; e < 8; e++) { epc_e[e] = 0.f; epc_o[e] = 0.f; }
#pragma unroll
            for (int t2 = 0; t2 < 8; t2++) {
              const float sg = (__builtin_popcount(t & t2) & 1) ? -1.f : 1.f;
              const v4f a = *(const v4f*)(xs + t2 * 256 + lane * 8);
              const v4f b = *(const v4f*)(xs + t2 * 256 + lane * 8 + 4);
              float* acc8 = (t2 & 1) ? epc_o : epc_e;
#pragma unroll
              for (int e = 0; e < 4; e++) {
                acc8[e] = fmaf(sg, a[e], acc8[e]);
                acc8[4 + e] = fmaf(sg, b[e], acc8[4 + e]);
              }
            }
          }
#pragma unroll
          for (int e = 0; e < 8; e++) r[e] = o == 0 ? epc_e[e] + epc_o[e] : epc_e[e] - epc_o[e];
        } else {
          had_tiles(xs, t, lane, r);
        }
#endif
        if constexpr (PRE) {
#pragma unroll
          for (int e = 0; e < 8; e++) sv[e] = (float)svp[o][0][e];
        } else {
          load_h8(w.svh + n0, sv);
        }
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
  if (MATS == 2 && p.h_rot) {  // (MATS is a template constant: down compiles none of this)
    // v3 EPI 1 (h_rot): the block's h rows, staged above, rotated with DOWN's suh -- exactly down's step 1
    // (prep_rotate for down: x = h f16, suh = down's, rotate128's rounding), so the pair is bit-identical to the
    // h_rot = 0 pair. One wave a row of the pass; a row not in the run is not written, a dropped row is zeros.
    lds_barrier();
    typedef const __attribute__((address_space(1))) v2u* gv2;
    const _Float16* hst = (const _Float16*)((const unsigned char*)lds + kHrotStage);
    const int nwv = (int)blockDim.x / WAVE32, wv = __builtin_amdgcn_readfirstlane((int)(threadIdx.x / WAVE32));
    const int rc = row - (lane & 15);
    const v2u sr = *(gv2)((const unsigned char*)w.dn_suh + (size_t)(cb * 128 + 4 * lane) * 2);
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

#endif  // EXL3R4D_COMMON_HIP_H
