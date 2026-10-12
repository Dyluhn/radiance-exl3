// issue.hip -- R-ISSUE (notes/decode-second-opinion.md §5.2): what the decode K loop's non-VALU instructions and its
// register-file pattern cost on gfx1201, as rates relative to a plain VALU stream measured the same way. The answers
// gate three bit-identical levers (SALU diet, bank-aware registers, one b128 A load a k-step).
//   issue [iters]     (inside a run_gpu.sh session; one row an arm)
// Arms (8 independent chains a wave, inline asm so nothing folds; every SIMD full unless the arm says otherwise):
//   base        32 v_mad_u32_u16 a step (A = chain, B = SGPR, C = a VGPR)
//   bank_same   the same with A and C hard-wired to one VGPR bank (reg % 4 equal)    } physical registers, clobbered
//   bank_diff   ... A and C in different banks                                       }
//   bank_none   ... C an SGPR too (one VGPR read)                                    }
//   salu        base + 8 s_add_co_i32 a step (one a 4 VALU)
//   vmem        base + 8 buffer_load_b32 a step from one hot dword (one a 4 VALU), no waits until the step's end
//   wmma / valu / mix   8 WMMA chains, 8 mad chains, or half the waves each (does WMMA on one wave overlap VALU on another)
//   lat_*       ONE wave, ONE chain: dependent latency of mad_u32_u16, pk_mul_lo_u16, sad_u8, perm, pk_fma_f16,
//               alignbit, v_add_nc_u32 (reported in ns an op and relative to v_add_nc_u32)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));

#define MAD(c) asm volatile("v_mad_u32_u16 %0, %0, %1, %2" : "+v"(c) : "s"(z), "v"(k));
#define CH8(S) S(c0) S(c1) S(c2) S(c3) S(c4) S(c5) S(c6) S(c7)

__global__ __launch_bounds__(256) void k_base(unsigned* out, int iters, unsigned kk) {
  unsigned c0 = threadIdx.x, c1 = c0 + 1, c2 = c0 + 2, c3 = c0 + 3, c4 = c0 + 4, c5 = c0 + 5, c6 = c0 + 6, c7 = c0 + 7;
  const unsigned k = kk ^ threadIdx.x, z = __builtin_amdgcn_readfirstlane(kk >> 7);
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int u = 0; u < 4; ++u) { CH8(MAD) }
  }
  if ((c0 ^ c1 ^ c2 ^ c3 ^ c4 ^ c5 ^ c6 ^ c7) == 0x12345678u) out[0] = 1;
}

// Physical registers: chains in v100..v107, the second VGPR source in v120..v127 (same bank as its chain: 120 - 100
// = 20 = 0 mod 4) or v121..v128 (bank + 1). 4 x 8 instructions a step.
#define BANK_STEP(SRC)                                                                                        \
  "v_mad_u32_u16 v100, v100, %0, " SRC(0) "\n v_mad_u32_u16 v101, v101, %0, " SRC(1) "\n"                    \
  "v_mad_u32_u16 v102, v102, %0, " SRC(2) "\n v_mad_u32_u16 v103, v103, %0, " SRC(3) "\n"                    \
  "v_mad_u32_u16 v104, v104, %0, " SRC(4) "\n v_mad_u32_u16 v105, v105, %0, " SRC(5) "\n"                    \
  "v_mad_u32_u16 v106, v106, %0, " SRC(6) "\n v_mad_u32_u16 v107, v107, %0, " SRC(7) "\n"
#define SRC_SAME(i) "v12" #i
#define SRC_DIFF(i) "v13" #i
#define SRC_NONE(i) "%0"
#define BANK_KERNEL(NAME, SRC)                                                                                \
  __global__ __launch_bounds__(256) void NAME(unsigned* out, int iters, unsigned kk) {                       \
    const unsigned z = __builtin_amdgcn_readfirstlane(kk >> 7);                                               \
    asm volatile("v_mov_b32 v100, %0\n v_mov_b32 v101, %0\n v_mov_b32 v102, %0\n v_mov_b32 v103, %0\n"       \
                 "v_mov_b32 v104, %0\n v_mov_b32 v105, %0\n v_mov_b32 v106, %0\n v_mov_b32 v107, %0\n"       \
                 "v_mov_b32 v120, %0\n v_mov_b32 v121, %0\n v_mov_b32 v122, %0\n v_mov_b32 v123, %0\n"       \
                 "v_mov_b32 v124, %0\n v_mov_b32 v125, %0\n v_mov_b32 v126, %0\n v_mov_b32 v127, %0\n"       \
                 "v_mov_b32 v130, %0\n v_mov_b32 v131, %0\n v_mov_b32 v132, %0\n v_mov_b32 v133, %0\n"       \
                 "v_mov_b32 v134, %0\n v_mov_b32 v135, %0\n v_mov_b32 v136, %0\n v_mov_b32 v137, %0\n"       \
                 :: "v"(threadIdx.x) : "v100", "v101", "v102", "v103", "v104", "v105", "v106", "v107", "v120", \
                   "v121", "v122", "v123", "v124", "v125", "v126", "v127", "v130", "v131", "v132", "v133",     \
                   "v134", "v135", "v136", "v137");                                                           \
    for (int i = 0; i < iters; ++i)                                                                           \
      asm volatile(BANK_STEP(SRC) BANK_STEP(SRC) BANK_STEP(SRC) BANK_STEP(SRC) :: "s"(z)                      \
                   : "v100", "v101", "v102", "v103", "v104", "v105", "v106", "v107");                         \
    unsigned r;                                                                                               \
    asm volatile("v_mov_b32 %0, v100" : "=v"(r));                                                             \
    if (r == 0x12345678u) out[0] = 1;                                                                         \
  }
// SRC_DIFF uses v130..v137: 130 - 100 = 30 = 2 mod 4, a different bank from every chain.
BANK_KERNEL(k_bank_same, SRC_SAME)
BANK_KERNEL(k_bank_diff, SRC_DIFF)
BANK_KERNEL(k_bank_none, SRC_NONE)

__global__ __launch_bounds__(256) void k_salu(unsigned* out, int iters, unsigned kk) {
  unsigned c0 = threadIdx.x, c1 = c0 + 1, c2 = c0 + 2, c3 = c0 + 3, c4 = c0 + 4, c5 = c0 + 5, c6 = c0 + 6, c7 = c0 + 7;
  const unsigned k = kk ^ threadIdx.x, z = __builtin_amdgcn_readfirstlane(kk >> 7);
  unsigned sa = z, sb = z + 1;
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      MAD(c0) MAD(c1) MAD(c2) MAD(c3)
      asm volatile("s_add_co_i32 %0, %0, 3" : "+s"(sa) : : "scc");
      MAD(c4) MAD(c5) MAD(c6) MAD(c7)
      asm volatile("s_add_co_i32 %0, %0, 5" : "+s"(sb) : : "scc");
    }
  }
  if ((c0 ^ c1 ^ c2 ^ c3 ^ c4 ^ c5 ^ c6 ^ c7 ^ sa ^ sb) == 0x12345678u) out[0] = 1;
}

__global__ __launch_bounds__(256) void k_vmem(unsigned* out, int iters, unsigned kk) {
  unsigned c0 = threadIdx.x, c1 = c0 + 1, c2 = c0 + 2, c3 = c0 + 3, c4 = c0 + 4, c5 = c0 + 5, c6 = c0 + 6, c7 = c0 + 7;
  const unsigned k = kk ^ threadIdx.x, z = __builtin_amdgcn_readfirstlane(kk >> 7);
  const __amdgpu_buffer_rsrc_t rs = __builtin_amdgcn_make_buffer_rsrc((void*)out, (short)0, 64, 0x31004000);
  unsigned acc = 0;
  for (int i = 0; i < iters; ++i) {
    unsigned l[8];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      MAD(c0) MAD(c1) MAD(c2) MAD(c3)
      l[2 * u] = __builtin_amdgcn_raw_buffer_load_b32(rs, 4u, 0, 0);
      MAD(c4) MAD(c5) MAD(c6) MAD(c7)
      l[2 * u + 1] = __builtin_amdgcn_raw_buffer_load_b32(rs, 8u, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < 8; ++u) acc ^= l[u];  // one wait for the step's 8 loads, at its end (the K loop's shape)
  }
  if ((c0 ^ c1 ^ c2 ^ c3 ^ c4 ^ c5 ^ c6 ^ c7 ^ acc) == 0x12345678u) out[0] = 1;
}

// ROLE 0: every wave 8 mad chains; 1: every wave 8 WMMA chains; 2: even waves WMMA, odd waves VALU. A step is 32
// mad or 8 WMMA a wave (one WMMA ~16 cycles, one mad ~1: comparable time a step).
template <int ROLE>
__global__ __launch_bounds__(256) void k_mix(unsigned* out, int iters, unsigned kk) {
  const int wave = threadIdx.x / 32;
  // ROLE 3: even waves WMMA, odd waves idle; ROLE 4: odd waves VALU, even waves idle -- the two halves of ROLE 2 alone
  if ((ROLE == 3 && (wave & 1)) || (ROLE == 4 && !(wave & 1))) return;
  const bool wm = ROLE == 1 || ((ROLE == 2 || ROLE == 3) && !(wave & 1));
  if (wm) {
    v8h a, b;
#pragma unroll
    for (int e = 0; e < 8; ++e) { a[e] = (_Float16)(threadIdx.x * 0.001f + e); b[e] = (_Float16)(kk * 1e-6f + e); }
    v8f c[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) c[j] = (v8f)(0.f);
    for (int i = 0; i < iters; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) c[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c[j]);
    }
    float s = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) s += c[j][0];
    if (s == 1234.5f) out[0] = 1;
  } else {
    unsigned c0 = threadIdx.x, c1 = c0 + 1, c2 = c0 + 2, c3 = c0 + 3, c4 = c0 + 4, c5 = c0 + 5, c6 = c0 + 6, c7 = c0 + 7;
    const unsigned k = kk ^ threadIdx.x, z = __builtin_amdgcn_readfirstlane(kk >> 7);
    for (int i = 0; i < iters; ++i) {
#pragma unroll
      for (int u = 0; u < 4; ++u) { CH8(MAD) }
    }
    if ((c0 ^ c1 ^ c2 ^ c3 ^ c4 ^ c5 ^ c6 ^ c7) == 0x12345678u) out[0] = 1;
  }
}

// Latency: one wave, one dependent chain, 32 ops a step.
#define LAT_KERNEL(NAME, OP)                                                                                  \
  __global__ __launch_bounds__(32) void NAME(unsigned* out, int iters, unsigned kk) {                        \
    unsigned c = threadIdx.x ^ kk;                                                                             \
    const unsigned k = kk ^ (threadIdx.x << 3), z = __builtin_amdgcn_readfirstlane(kk >> 7);                  \
    for (int i = 0; i < iters; ++i) {                                                                          \
      _Pragma("unroll") for (int u = 0; u < 32; ++u) asm volatile(OP : "+v"(c) : "s"(z), "v"(k));             \
    }                                                                                                          \
    if (c == 0x12345678u) out[0] = 1;                                                                          \
  }
LAT_KERNEL(lat_add, "v_add_nc_u32 %0, %0, %2")
LAT_KERNEL(lat_mad, "v_mad_u32_u16 %0, %0, %1, %2")
LAT_KERNEL(lat_pkmul, "v_pk_mul_lo_u16 %0, %0, %2")
LAT_KERNEL(lat_sad, "v_sad_u8 %0, %2, 0, %0")
LAT_KERNEL(lat_perm, "v_perm_b32 %0, %0, %2, 0x05040100")
LAT_KERNEL(lat_pkfma, "v_pk_fma_f16 %0, %0, %2, %2")
LAT_KERNEL(lat_alignbit, "v_alignbit_b32 %0, %0, %2, 7")

typedef void (*Kern)(unsigned*, int, unsigned);
static double run(Kern f, unsigned* out, int iters, int blocks, int threads) {
  hipEvent_t a, b;
  (void)hipEventCreate(&a);
  (void)hipEventCreate(&b);
  for (int w = 0; w < 3; ++w) hipLaunchKernelGGL(f, dim3(blocks), dim3(threads), 0, 0, out, iters, 0x3c013c01u);
  (void)hipEventRecord(a);
  hipLaunchKernelGGL(f, dim3(blocks), dim3(threads), 0, 0, out, iters, 0x3c013c01u);
  (void)hipEventRecord(b);
  (void)hipEventSynchronize(b);
  float ms = 0;
  (void)hipEventElapsedTime(&ms, a, b);
  (void)hipEventDestroy(a);
  (void)hipEventDestroy(b);
  return (double)ms;
}

int main(int argc, char** argv) {
  const int iters = argc > 1 ? atoi(argv[1]) : 4096;
  hipDeviceProp_t pr;
  (void)hipGetDeviceProperties(&pr, 0);
  const int blocks = pr.multiProcessorCount * 16;
  unsigned* out = nullptr;
  if (hipMalloc(&out, 256) != hipSuccess) return 2;
  (void)hipMemset(out, 0, 256);
  printf("issue: %s %s, %d CUs, %d blocks x 256, %d iters\n", pr.name, pr.gcnArchName, pr.multiProcessorCount, blocks,
         iters);
  // Throughput arms: 32 VALU a step a wave (plus the arm's extra instructions); ratio to base.
  const double t_base = run(k_base, out, iters, blocks, 256);
  struct { const char* n; Kern f; } arms[] = {
      {"base (32 mad a step)", k_base}, {"bank_none (1 VGPR read)", k_bank_none}, {"bank_diff (2 VGPR, 2 banks)", k_bank_diff},
      {"bank_same (2 VGPR, 1 bank)", k_bank_same}, {"salu (+8 s_add a step)", k_salu}, {"vmem (+8 buffer_load a step)", k_vmem}};
  for (auto& a : arms) {
    const double t = run(a.f, out, iters, blocks, 256);
    printf("RATE %-30s %8.3f ms   %5.3f x base\n", a.n, t, t / t_base);
  }
  // WMMA vs VALU across waves of one SIMD.
  const double tv = run(k_mix<0>, out, iters, blocks, 256), tw = run(k_mix<1>, out, iters, blocks, 256),
               tm = run(k_mix<2>, out, iters, blocks, 256);
  const double th_w = run(k_mix<3>, out, iters, blocks, 256), th_v = run(k_mix<4>, out, iters, blocks, 256);
  printf("MIX valu-only %.3f ms, wmma-only %.3f ms, half/half %.3f ms; the halves alone: wmma %.3f, valu %.3f ms -> "
         "serial %.3f, overlapped %.3f\n", tv, tw, tm, th_w, th_v, th_w + th_v, th_w > th_v ? th_w : th_v);
  // Latency: one wave on the whole GPU.
  struct { const char* n; Kern f; } lats[] = {{"v_add_nc_u32", lat_add}, {"v_mad_u32_u16", lat_mad},
                                              {"v_pk_mul_lo_u16", lat_pkmul}, {"v_sad_u8", lat_sad},
                                              {"v_perm_b32", lat_perm}, {"v_pk_fma_f16", lat_pkfma},
                                              {"v_alignbit_b32", lat_alignbit}};
  const int li = iters / 4 > 0 ? iters / 4 : 1;
  const double t_add = run(lat_add, out, li, 1, 32);
  for (auto& a : lats) {
    const double t = run(a.f, out, li, 1, 32);
    printf("LAT  %-16s %7.3f ns an op   %5.2f x v_add_nc_u32\n", a.n, t * 1e6 / ((double)li * 32.0), t / t_add);
  }
  (void)hipFree(out);
  return 0;
}
