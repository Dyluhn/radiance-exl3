// stream.hip -- the achievable stream bandwidth of a card, at the row pitches the EXL3 kernels read.
// radiance-exl3 S0.11 (2026-10-11). Apache-2.0.
//
// The author quotes 638 GB/s for "a plain stream" on gfx1201 and a threads-per-row sweep at a 5120 B
// pitch (422/626/633/635/636, r4d_gemm_fp8a8.hip "THE STAGING ASSIGNMENT"), but the kernel is not in
// the repo. This is that kernel, written to the same description: one thread moves one 32-byte granule
// (two 16-byte loads) a step, TPR consecutive threads cover TPR*32 contiguous bytes of ONE row, so a
// wave's load asks for 32/TPR rows of TPR*32 bytes; a 256-thread block owns 256/TPR whole rows and walks
// them left to right. TPR*32 need not divide the pitch: the lanes past the row's end idle, as a real
// tile's would. Each launch reads 1 GiB, far past the last-level cache, so every byte comes from DRAM.
//
// "plain" is pitch 1024 at TPR 32: a block reads 8 KiB contiguous and consecutive blocks are adjacent.
// copy mode (plain only) writes what it reads and counts both directions, in case a quoted figure was a
// copy. The source is filled with a hash, not zeros, so no compression path can flatter the number.
//
//   hipcc -O3 -std=c++17 --offload-arch=gfx1201 stream.hip -o stream && ./stream [timed launches]
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "devid.h"

static const size_t kPool = (size_t)1 << 30;

__global__ void fill(unsigned* p, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = (unsigned)(i * 0x9e3779b97f4a7c15ull >> 32);
}

template <int TPR, bool COPY>
__global__ __launch_bounds__(256) void rows_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst,
                                                   unsigned pitch16, unsigned rows) {
    const unsigned row = blockIdx.x * (256 / TPR) + threadIdx.x / TPR;
    if (row >= rows) return;
    const size_t base = (size_t)row * pitch16;
    unsigned acc = 0;
    for (unsigned c = (threadIdx.x % TPR) * 2; c < pitch16; c += TPR * 2) {
        const uint4 v0 = src[base + c], v1 = src[base + c + 1];
        if (COPY) { dst[base + c] = v0; dst[base + c + 1] = v1; }
        else acc ^= v0.x ^ v0.y ^ v0.z ^ v0.w ^ v1.x ^ v1.y ^ v1.z ^ v1.w;
    }
    if (!COPY && acc == 0x2545f491u) dst[0].x = acc;   // practically never; keeps the loads alive
}

typedef void (*rows_fn)(const uint4*, uint4*, unsigned, unsigned);

template <bool COPY> static rows_fn pick(int tpr) {
    switch (tpr) {
        case 2: return rows_kernel<2, COPY>;   case 4: return rows_kernel<4, COPY>;
        case 8: return rows_kernel<8, COPY>;   case 16: return rows_kernel<16, COPY>;
        default: return rows_kernel<32, COPY>;
    }
}

// Median of `reps` event-timed launches after two warm ones, in GB/s of bytes moved.
static double run(rows_fn k, const uint4* src, uint4* dst, unsigned pitch, int tpr, double moved, int reps) {
    const unsigned rows = (unsigned)(kPool / pitch), per = 256 / tpr;
    const dim3 grid((rows + per - 1) / per);
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    std::vector<float> ms;
    for (int r = -2; r < reps; ++r) {
        hipEventRecord(a);
        hipLaunchKernelGGL(k, grid, dim3(256), 0, 0, src, dst, pitch / 16, rows);
        hipEventRecord(b); hipEventSynchronize(b);
        float t = 0; hipEventElapsedTime(&t, a, b);
        if (r >= 0) ms.push_back(t);
    }
    if (hipGetLastError() != hipSuccess) { std::printf("launch failed (pitch %u, tpr %d)\n", pitch, tpr); std::exit(1); }
    std::sort(ms.begin(), ms.end());
    return moved * (double)rows * pitch / (ms[ms.size() / 2] * 1e6);
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 10;
    if (calib_print_device(0)) return 1;
    uint4 *src = nullptr, *dst = nullptr;
    if (hipMalloc(&src, kPool) != hipSuccess || hipMalloc(&dst, kPool) != hipSuccess) { std::printf("hipMalloc 2 GiB\n"); return 1; }
    hipLaunchKernelGGL(fill, dim3(4096), dim3(256), 0, 0, (unsigned*)src, kPool / 4);
    hipDeviceSynchronize();
    std::printf("stream: 1 GiB a launch, median of %d launches\n", reps);
    const unsigned pitches[] = {1024, 5120, 1280, 640};   // plain; 2560 bf16; 2560 u4 codes; 1280 u4 codes
    const int tprs[] = {2, 4, 8, 16, 32};
    for (unsigned p : pitches)
        for (int t : tprs)
            std::printf("stream mode=read pitch=%u tpr=%d gbps=%.1f\n", p, t, run(pick<false>(t), src, dst, p, t, 1.0, reps));
    std::printf("stream mode=copy pitch=1024 tpr=32 gbps=%.1f\n", run(pick<true>(32), src, dst, 1024, 32, 2.0, reps));
    return 0;
}
