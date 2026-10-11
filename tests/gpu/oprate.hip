// oprate.hip -- playbook D.0 R6: single-issue rate of the instructions the decode diet (V1a/V1b) and a dot path (V4)
// would use on gfx1201, each relative to v_add_nc_u32 measured the same way. Every arm runs one instruction in eight
// independent chains (nothing to stall on but issue), as inline assembly so the compiler neither folds nor pairs it
// (no VOPD), over enough waves to fill every SIMD. The RATIO is the reading; it does not depend on the clock.
//   oprate [iters]     (run inside a run_gpu.sh session; prints one row an arm)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHAINS(STMT)                                                                   \
  STMT(c0) STMT(c1) STMT(c2) STMT(c3) STMT(c4) STMT(c5) STMT(c6) STMT(c7)

// ARM(name, asm template on %0 (the chain) with %1 = k (a VGPR constant) and %2 = z (an SGPR zero))
#define ARM(NAME, TMPL)                                                                \
  __global__ __launch_bounds__(256) void NAME(unsigned* out, int iters, unsigned kk) { \
    unsigned c0 = threadIdx.x, c1 = c0 + 1, c2 = c0 + 2, c3 = c0 + 3, c4 = c0 + 4,    \
             c5 = c0 + 5, c6 = c0 + 6, c7 = c0 + 7;                                    \
    const unsigned k = kk ^ (threadIdx.x & 1u);                                        \
    const unsigned z = __builtin_amdgcn_readfirstlane(kk >> 31);                       \
    for (int i = 0; i < iters; ++i) {                                                  \
      _Pragma("unroll") for (int u = 0; u < 4; ++u) {                                  \
        CHAINS(ONE_##NAME)                                                             \
      }                                                                                \
    }                                                                                  \
    const unsigned s = c0 ^ c1 ^ c2 ^ c3 ^ c4 ^ c5 ^ c6 ^ c7;                          \
    if (s == 0x12345678u) out[0] = s;                                                  \
  }

#define ONE_add(c)      asm volatile("v_add_nc_u32 %0, %0, %1" : "+v"(c) : "v"(k), "s"(z));
#define ONE_mul_lo(c)   asm volatile("v_mul_lo_u32 %0, %0, %1" : "+v"(c) : "v"(k), "s"(z));
#define ONE_mul24(c)    asm volatile("v_mul_u32_u24 %0, %0, %1" : "+v"(c) : "v"(k), "s"(z));
#define ONE_mad24(c)    asm volatile("v_mad_u32_u24 %0, %0, %1, %0" : "+v"(c) : "v"(k), "s"(z));
#define ONE_mad16(c)    asm volatile("v_mad_u32_u16 %0, %0, %1, %0" : "+v"(c) : "v"(k), "s"(z));
#define ONE_mad16h(c)   asm volatile("v_mad_u32_u16 %0, %0, %1, %0 op_sel:[1,0,0,0]" : "+v"(c) : "v"(k), "s"(z));
#define ONE_pkmul16(c)  asm volatile("v_pk_mul_lo_u16 %0, %0, %1" : "+v"(c) : "v"(k), "s"(z));
#define ONE_lshladd(c)  asm volatile("v_lshl_add_u32 %0, %0, 3, %1" : "+v"(c) : "v"(k), "s"(z));
#define ONE_perm(c)     asm volatile("v_perm_b32 %0, %0, %1, 0x05040100" : "+v"(c) : "v"(k), "s"(z));
#define ONE_bfe(c)      asm volatile("v_bfe_u32 %0, %0, 4, 16" : "+v"(c) : "v"(k), "s"(z));
#define ONE_alignbit(c) asm volatile("v_alignbit_b32 %0, %0, %1, 7" : "+v"(c) : "v"(k), "s"(z));
#define ONE_sad(c)      asm volatile("v_sad_u8 %0, %1, 0, %0" : "+v"(c) : "v"(k), "s"(z));
#define ONE_sadhi(c)    asm volatile("v_sad_hi_u8 %0, %1, 0, %0" : "+v"(c) : "v"(k), "s"(z));
#define ONE_pkfma(c)    asm volatile("v_pk_fma_f16 %0, %0, %1, %1" : "+v"(c) : "v"(k), "s"(z));
#define ONE_dot2(c)     asm volatile("v_dot2_f32_f16 %0, %1, %1, %0" : "+v"(c) : "v"(k), "s"(z));
#define ONE_plx16(c)    asm volatile("v_permlanex16_b32 %0, %1, %2, %2" : "+v"(c) : "v"(k), "s"(z));
#define ONE_dppmov(c)   asm volatile("v_mov_b32_dpp %0, %1 row_xmask:1 row_mask:0xf bank_mask:0xf" : "+v"(c) : "v"(k), "s"(z));

ARM(add, _)
ARM(mul_lo, _)
ARM(mul24, _)
ARM(mad24, _)
ARM(mad16, _)
ARM(mad16h, _)
ARM(pkmul16, _)
ARM(lshladd, _)
ARM(perm, _)
ARM(bfe, _)
ARM(alignbit, _)
ARM(sad, _)
ARM(sadhi, _)
ARM(pkfma, _)
ARM(dot2, _)
ARM(plx16, _)
ARM(dppmov, _)

typedef void (*Kern)(unsigned*, int, unsigned);
static double run(Kern f, unsigned* out, int iters, int blocks) {
  hipEvent_t a, b;
  (void)hipEventCreate(&a);
  (void)hipEventCreate(&b);
  for (int w = 0; w < 3; ++w) hipLaunchKernelGGL(f, dim3(blocks), dim3(256), 0, 0, out, iters, 0x3c013c01u);
  (void)hipEventRecord(a);
  hipLaunchKernelGGL(f, dim3(blocks), dim3(256), 0, 0, out, iters, 0x3c013c01u);
  (void)hipEventRecord(b);
  (void)hipEventSynchronize(b);
  float ms = 0;
  (void)hipEventElapsedTime(&ms, a, b);
  const double inst = (double)blocks * (256 / 32) * iters * 32.0;  // wave-instructions (4 x 8 an iteration)
  return inst / (ms * 1e-3);
}

int main(int argc, char** argv) {
  const int iters = argc > 1 ? atoi(argv[1]) : 4096;
  hipDeviceProp_t pr;
  (void)hipGetDeviceProperties(&pr, 0);
  const int blocks = pr.multiProcessorCount * 16;  // 16 x 8 waves a CU: every SIMD full whatever the CU count means
  unsigned* out = nullptr;
  if (hipMalloc(&out, 64) != hipSuccess) return 2;
  printf("oprate: %s %s, %d CUs, %d blocks x 256, %d iters x 32 a wave\n", pr.name, pr.gcnArchName,
         pr.multiProcessorCount, blocks, iters);
  struct { const char* n; Kern f; } arms[] = {
      {"v_add_nc_u32", add},         {"v_mul_lo_u32", mul_lo},   {"v_mul_u32_u24", mul24},
      {"v_mad_u32_u24", mad24},      {"v_mad_u32_u16", mad16},   {"v_mad_u32_u16 op_sel", mad16h},
      {"v_pk_mul_lo_u16", pkmul16},  {"v_lshl_add_u32", lshladd}, {"v_perm_b32", perm},
      {"v_bfe_u32", bfe},            {"v_alignbit_b32", alignbit}, {"v_sad_u8", sad},
      {"v_sad_hi_u8", sadhi},        {"v_pk_fma_f16", pkfma},    {"v_dot2_f32_f16", dot2},
      {"v_permlanex16_b32", plx16},  {"v_mov_b32_dpp row_xmask", dppmov}};
  double ref = 0;
  for (auto& a : arms) {
    const double r = run(a.f, out, iters, blocks);
    if (ref == 0) ref = r;
    printf("RATE %-26s %8.1f Gwave-inst/s   %5.2f cycles vs v_add_nc_u32\n", a.n, r * 1e-9, ref / r);
  }
  (void)hipFree(out);
  return 0;
}
