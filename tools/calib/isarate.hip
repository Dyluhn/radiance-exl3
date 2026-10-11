// What one instruction slot is worth on gfx1201, so that an instruction COUNT taken out of a
// disassembly can be turned into a time.
//
// A GEMM's staged K tile is a few dozen v_wmma inside several hundred other instructions, so
// whether trimming the shell is worth anything depends entirely on what a wmma costs relative to a
// plain VALU op -- and that ratio is not in any public table for this part. So measure it: run each
// instruction back to back in independent chains (nothing to stall on but issue) and report the
// rate. The RATIO is what matters and it is clock-independent, which is why no clock is read here.
//
// Occupancy is a parameter because a real tile kernel runs at 2 waves a SIMD (36864 B of LDS) and a
// microbenchmark at 8 would answer a different question.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

typedef int   i2v  __attribute__((ext_vector_type(2)));
typedef float f8v  __attribute__((ext_vector_type(8)));

template <int LDS>
__global__ __launch_bounds__(256) void wmma_rate(float* out, int iters) {
    __shared__ char pad[LDS ? LDS : 4];
    pad[threadIdx.x & 3] = (char)threadIdx.x;
    i2v a, b;
    a[0] = 0x01010101; a[1] = 0x02020202;
    b[0] = 0x03030303; b[1] = 0x04040404;
    f8v c0{}, c1{}, c2{}, c3{}, c4{}, c5{}, c6{}, c7{};
    for (int i = 0; i < iters; ++i) {
        c0 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c0);
        c1 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c1);
        c2 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c2);
        c3 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c3);
        c4 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c4);
        c5 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c5);
        c6 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c6);
        c7 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c7);
    }
    float s = 0;
    for (int t = 0; t < 8; ++t) s += c0[t] + c1[t] + c2[t] + c3[t] + c4[t] + c5[t] + c6[t] + c7[t];
    if (s == -1.0f) out[0] = s + (float)pad[0];
}

template <int LDS>
__global__ __launch_bounds__(256) void valu_rate(float* out, int iters) {
    __shared__ char pad[LDS ? LDS : 4];
    pad[threadIdx.x & 3] = (char)threadIdx.x;
    float x0 = threadIdx.x, x1 = x0 + 1, x2 = x0 + 2, x3 = x0 + 3;
    float x4 = x0 + 4, x5 = x0 + 5, x6 = x0 + 6, x7 = x0 + 7;
    const float k = 1.0000001f;
    // Eight independent chains, eight ops an iteration -- the same shape as the wmma loop so the
    // two numbers differ only in the instruction being issued.
    for (int i = 0; i < iters; ++i) {
        x0 *= k; x1 *= k; x2 *= k; x3 *= k; x4 *= k; x5 *= k; x6 *= k; x7 *= k;
    }
    const float s = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
    if (s == -1.0f) out[0] = s + (float)pad[0];
}

static float bench(void (*k)(float*, int), float* d, int iters, int blocks, int reps) {
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    hipLaunchKernelGGL(k, dim3(blocks), dim3(256), 0, 0, d, iters);   // warm
    hipDeviceSynchronize();
    float best = 1e30f;
    for (int r = 0; r < reps; ++r) {
        hipEventRecord(a);
        hipLaunchKernelGGL(k, dim3(blocks), dim3(256), 0, 0, d, iters);
        hipEventRecord(b); hipEventSynchronize(b);
        float ms = 0; hipEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    return best;
}

int main(int argc, char** argv) {
    const int iters  = argc > 1 ? atoi(argv[1]) : 20000;
    const int blocks = argc > 2 ? atoi(argv[2]) : 1024;
    float* d = nullptr; hipMalloc(&d, 4);
    printf("  gfx1201 issue rate, %d blocks x 256 threads, %d iterations x 8 instructions\n\n",
           blocks, iters);
    printf("  %-22s %10s %14s %12s %10s\n", "instruction", "ms", "Ginstr/s", "TFLOP/s", "vs VALU");
    const double n = (double)iters * 8.0 * blocks * 8.0;   // 8 waves a block, 8 instr an iteration
    struct { const char* nm; float ms; double flop; } row[4];
    row[0] = { "v_wmma fp8 (0 LDS)",  bench(wmma_rate<0>,     d, iters, blocks, 5), 8192.0 };
    row[1] = { "v_wmma fp8 (36 KiB)", bench(wmma_rate<36864>, d, iters, blocks, 5), 8192.0 };
    row[2] = { "v_mul_f32  (0 LDS)",  bench(valu_rate<0>,     d, iters, blocks, 5), 64.0 };
    row[3] = { "v_mul_f32  (36 KiB)", bench(valu_rate<36864>, d, iters, blocks, 5), 64.0 };
    double vbase = 0;
    for (int i = 0; i < 4; ++i) {
        const double gi = n / (row[i].ms * 1e6);
        if (i == 2) vbase = gi;
        printf("  %-22s %10.3f %14.2f %12.1f %9.2fx\n", row[i].nm, row[i].ms, gi,
               n * row[i].flop / (row[i].ms * 1e-3) / 1e12, vbase > 0 ? vbase / gi : 0.0);
    }
    printf("\n  \"vs VALU\" is how many v_mul_f32 issue slots one of these costs.\n");
    return 0;
}
