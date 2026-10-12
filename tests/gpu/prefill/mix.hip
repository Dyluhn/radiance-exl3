// mix.hip -- does f16 WMMA throughput overlap VALU throughput on one SIMD when the SAME waves issue both (the prefill
// K loop's shape: a wave decodes, then feeds the matrix unit)? Sets the prefill binding floor: max(WMMA, VALU) if they
// overlap, WMMA + VALU if they serialize (notes/p5.md, round 2).
//
// Each wave runs `iters` steps of W independent WMMA (8 accumulator chains) and V independent v_mad_u32_u16 (8 chains),
// interleaved (a WMMA every V/W VALU) or not; every SIMD is filled with `waves` waves. Arms: WMMA only, VALU only, both
// in every wave, and both split across waves (even waves WMMA, odd waves VALU). Time by hipEvent over `reps` launches;
// prints ms and the ratio of the mixed arm to the sum and to the max of the single arms.
//   mix <waves a SIMD> <W a step> <V a step> <iters> <reps>      (W 8; V 32, 64 or 128)
// Build (the build image; not part of the CMake tree): hipcc -O3 --offload-arch=gfx1201 -o mix tests/gpu/prefill/mix.hip
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));

template <int MODE, int W, int V>   // MODE 0 WMMA only, 1 VALU only, 2 both in every wave, 3 even waves WMMA / odd VALU
__global__ void mix_kernel(float* out, int iters, unsigned seed) {
  const int wave = threadIdx.x / 32;
  const bool doW = MODE == 0 || MODE == 2 || (MODE == 3 && !(wave & 1));
  const bool doV = MODE == 1 || MODE == 2 || (MODE == 3 && (wave & 1));
  v8h a, b;
  for (int i = 0; i < 8; i++) { a[i] = (_Float16)(threadIdx.x * 0.001f + i); b[i] = (_Float16)(seed * 1e-6f + i); }
  v8f c[8];
  for (int j = 0; j < 8; j++) c[j] = (v8f)(0.f);
  unsigned m[8];
  for (int j = 0; j < 8; j++) m[j] = seed + threadIdx.x + j;
  constexpr int VPW = V / W;   // mads after each WMMA
  for (int it = 0; it < iters; it++) {
    if (MODE == 3 ? doW : true) {
#pragma unroll
      for (int w = 0; w < W; w++) {
        if (MODE != 1) c[w & 7] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c[w & 7]);
        if (MODE == 1 || MODE == 2) {
#pragma unroll
          for (int k = 0; k < VPW; k++) asm volatile("v_mad_u32_u16 %0, %0, %1, %0" : "+v"(m[k & 7]) : "v"(m[(k + 1) & 7]));
        }
      }
    } else {
#pragma unroll
      for (int k = 0; k < V; k++) asm volatile("v_mad_u32_u16 %0, %0, %1, %0" : "+v"(m[k & 7]) : "v"(m[(k + 1) & 7]));
    }
  }
  float s = 0.f;
  for (int j = 0; j < 8; j++) for (int e = 0; e < 8; e++) s += c[j][e];
  for (int j = 0; j < 8; j++) s += (float)m[j];
  if (s == 1.2345e-30f) out[threadIdx.x] = s;
}

int main(int argc, char** argv) {
  if (argc < 6) { fprintf(stderr, "usage: mix <waves a SIMD> <W> <V> <iters> <reps>\n"); return 2; }
  const int waves = atoi(argv[1]), W = atoi(argv[2]), V = atoi(argv[3]), iters = atoi(argv[4]), reps = atoi(argv[5]);
  if (W != 8 || (V != 64 && V != 128 && V != 32)) { fprintf(stderr, "W 8, V 32 / 64 / 128 only\n"); return 2; }
  hipDeviceProp_t pr;
  hipGetDeviceProperties(&pr, 0);
  const int cus = pr.multiProcessorCount;   // gfx1201: CUs; a CU has 2 SIMDs
  const int threads = 256, blocks = cus * 2 * waves * 32 / threads;   // `waves` waves on each SIMD
  float* out;
  hipMalloc(&out, 1024 * sizeof(float));
  hipEvent_t e0, e1;
  hipEventCreate(&e0); hipEventCreate(&e1);
  double ms[4];
  for (int mode = 0; mode < 4; mode++) {
    auto run = [&]() {
#define MIXL(M_, V_) mix_kernel<M_, 8, V_><<<blocks, threads>>>(out, iters, 1u)
#define MIXV(M_) (V == 32 ? MIXL(M_, 32) : V == 64 ? MIXL(M_, 64) : MIXL(M_, 128))
      switch (mode) {
        case 0: MIXV(0); break;
        case 1: MIXV(1); break;
        case 2: MIXV(2); break;
        default: MIXV(3); break;
      }
    };
    run(); run();
    hipDeviceSynchronize();
    hipEventRecord(e0);
    for (int r = 0; r < reps; r++) run();
    hipEventRecord(e1);
    hipEventSynchronize(e1);
    float t;
    hipEventElapsedTime(&t, e0, e1);
    ms[mode] = t / reps;
  }
  // split-waves does half of each (every wave one kind): compare it with half the sum / half the max
  printf("MIX waves/SIMD %d W %d V %d iters %d: wmma-only %.3f ms, valu-only %.3f ms, both-in-each-wave %.3f ms "
         "(sum %.3f, max %.3f; both/sum %.2f), split-waves %.3f ms (half sum %.3f, half max %.3f)\n",
         waves, W, V, iters, ms[0], ms[1], ms[2], ms[0] + ms[1], ms[0] > ms[1] ? ms[0] : ms[1],
         ms[2] / (ms[0] + ms[1]), ms[3], (ms[0] + ms[1]) / 2, (ms[0] > ms[1] ? ms[0] : ms[1]) / 2);
  return 0;
}
