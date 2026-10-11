// PROVENANCE: copied from codeberg.org/StillDeadcode/radiance libr4d/isa/isarate.hip at commit 6c11671b
// (v1.0.5), Apache-2.0 (LICENSE, NOTICE.radiance beside this file); the unmodified copy is this file
// at the commit that added it here. Changed for radiance-exl3 S0.11 (2026-10-11), nothing else:
//   - an f16 arm, v_wmma_f32_16x16x16_f16, with the same eight-independent-chains loop as the fp8 arm;
//   - occupancy as arguments: threads a block and a LIST of dynamic-LDS sizes (default 0 and 36864, the
//     author's two arms), with the occupancy HIP computes for each printed beside its row, so an arm can
//     be put at the occupancy a real tile runs;
//   - "vs VALU" against v_mul_f32 at the SAME LDS (same occupancy) and against the 0-LDS one. As shipped
//     the column was 0.00x on both wmma rows, because the base was set only after they had printed;
//   - the device line (devid.h) so the log names the card.
//
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
#include <vector>
#include "devid.h"

typedef int      i2v  __attribute__((ext_vector_type(2)));
typedef float    f8v  __attribute__((ext_vector_type(8)));
typedef _Float16 h8v  __attribute__((ext_vector_type(8)));

// LDS is dynamic so one binary can take any occupancy; the 4-byte touch keeps the allocation live
// exactly as the static pad[LDS ? LDS : 4] did (the 0-LDS arm launches with 4 bytes, as before).
__global__ __launch_bounds__(256) void wmma_rate(float* out, int iters) {
    extern __shared__ char pad[];
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

// The f16 arm: the instruction the EXL3 kernel issues (trellis-decoded f16 weight x f16 activation).
// Same loop, same eight chains, same flops an instruction (16x16x16x2); only the opcode differs.
__global__ __launch_bounds__(256) void wmma16_rate(float* out, int iters) {
    extern __shared__ char pad[];
    pad[threadIdx.x & 3] = (char)threadIdx.x;
    h8v a, b;
    for (int e = 0; e < 8; ++e) { a[e] = (_Float16)(0.01f * (e + 1)); b[e] = (_Float16)(0.02f * (e + 1)); }
    f8v c0{}, c1{}, c2{}, c3{}, c4{}, c5{}, c6{}, c7{};
    for (int i = 0; i < iters; ++i) {
        c0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c0);
        c1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c1);
        c2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c2);
        c3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c3);
        c4 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c4);
        c5 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c5);
        c6 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c6);
        c7 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c7);
    }
    float s = 0;
    for (int t = 0; t < 8; ++t) s += c0[t] + c1[t] + c2[t] + c3[t] + c4[t] + c5[t] + c6[t] + c7[t];
    if (s == -1.0f) out[0] = s + (float)pad[0];
}

__global__ __launch_bounds__(256) void valu_rate(float* out, int iters) {
    extern __shared__ char pad[];
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

typedef void (*rate_kernel)(float*, int);

static float bench(rate_kernel k, float* d, int iters, int blocks, int threads, size_t lds, int reps) {
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    hipLaunchKernelGGL(k, dim3(blocks), dim3(threads), lds, 0, d, iters);   // warm
    hipDeviceSynchronize();
    float best = 1e30f;
    for (int r = 0; r < reps; ++r) {
        hipEventRecord(a);
        hipLaunchKernelGGL(k, dim3(blocks), dim3(threads), lds, 0, d, iters);
        hipEventRecord(b); hipEventSynchronize(b);
        float ms = 0; hipEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    if (hipGetLastError() != hipSuccess) { std::printf("  launch failed (threads %d, lds %zu)\n", threads, lds); std::exit(1); }
    return best;
}

// Waves a SIMD that HIP will co-schedule for this kernel at this block size and LDS. On gfx12 (WGP mode)
// HIP's "multiprocessor" is a WGP: four SIMD32 sharing 64 KiB of LDS (measured: 32 of them on a 64-CU
// part, lds_per_cu 65536). So 36864 B holds a 256-thread block to one a WGP, 2 waves a SIMD.
static double waves_per_simd(rate_kernel k, int threads, size_t lds) {
    int nb = 0;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&nb, (const void*)k, threads, lds) != hipSuccess) return -1;
    return nb * (threads / 32) / 4.0;
}

// isarate [iters] [blocks] [threads] [lds_bytes ...]     defaults: 20000 1024 256 0 36864
int main(int argc, char** argv) {
    const int iters   = argc > 1 ? atoi(argv[1]) : 20000;
    const int blocks  = argc > 2 ? atoi(argv[2]) : 1024;
    const int threads = argc > 3 ? atoi(argv[3]) : 256;
    std::vector<size_t> ldss;
    for (int i = 4; i < argc; ++i) ldss.push_back((size_t)atol(argv[i]));
    if (ldss.empty()) ldss = {0, 36864};
    if (threads < 32 || threads > 256 || threads % 32) { std::printf("threads must be 32..256, a multiple of 32\n"); return 2; }
    if (calib_print_device(0)) return 1;
    float* d = nullptr; hipMalloc(&d, 4);
    printf("  gfx1201 issue rate, %d blocks x %d threads, %d iterations x 8 instructions\n\n",
           blocks, threads, iters);
    printf("  %-14s %7s %6s %9s %10s %9s %9s %9s\n", "instruction", "LDS", "w/SIMD", "ms", "Ginstr/s",
           "TFLOP/s", "slots", "slots@0");
    const double n = (double)iters * 8.0 * blocks * (threads / 32);   // waves x 8 instr an iteration
    double vbase0 = 0;
    for (size_t li = 0; li < ldss.size(); ++li) {
        const size_t lds = ldss[li] < 4 ? 4 : ldss[li];
        struct { const char* nm; rate_kernel k; double flop; float ms; } row[3] = {
            { "v_mul_f32",   valu_rate,   64.0,   0 },
            { "v_wmma fp8",  wmma_rate,   8192.0, 0 },
            { "v_wmma f16",  wmma16_rate, 8192.0, 0 } };
        for (auto& r : row) r.ms = bench(r.k, d, iters, blocks, threads, lds, 5);
        const double vbase = n / (row[0].ms * 1e6);
        if (li == 0) vbase0 = vbase;
        for (auto& r : row) {
            const double gi = n / (r.ms * 1e6);
            printf("  %-14s %7zu %6.1f %9.3f %10.2f %9.1f %8.2fx %8.2fx\n", r.nm, ldss[li],
                   waves_per_simd(r.k, threads, lds), r.ms, gi, n * r.flop / (r.ms * 1e-3) / 1e12,
                   vbase / gi, vbase0 / gi);
        }
    }
    printf("\n  \"slots\" is how many v_mul_f32 issue slots one of these costs at the SAME LDS (same occupancy);\n"
           "  \"slots@0\" against v_mul_f32 at the first LDS in the list. w/SIMD is HIP's occupancy model.\n");
    return 0;
}
