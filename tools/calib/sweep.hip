// sweep.hip -- the DEVICE-side fixed cost of a dispatch: the intercept of t = fixed + bytes / BW.
// radiance-exl3 S0.11 (2026-10-11). Apache-2.0.
//
// The author's ~4.5 us is the intercept of one line through the decode GEMM shapes and small ops in a
// trace. This takes the same reading off the simplest kernel there is: stream.hip's plain read (a
// 256-thread block reads 8 KiB contiguous, one 32-byte granule a thread), launched R times back to back
// on one stream for each size S. t(S) = event span over the R launches / R. A 1 GiB read is queued
// first, so the host has enqueued all R launches before the device reaches the first: the span is the
// device's time per dispatch with a full queue (a decode step's state), not the host's submit rate --
// graphrate.hip prices that. The blocker is a READ and not a spin because the memory clock follows load:
// behind a spin, or a run of tiny kernels, the governor drops it (seen: 456 MHz of a 1350 MHz top, which
// put a 2 MiB point at 16 us against a 7 us line), and a decode step never sees that state. Launch i
// reads the next S bytes of a 1 GiB pool, wrapping, so no launch finds its bytes in the last-level cache.
//
// The fit is weighted least squares with weights 1/t^2 (constant RELATIVE noise; unweighted, the
// 64 MiB point alone would set the intercept). Fit range fixed before any data: 1..64 MiB, 128..8192
// blocks, i.e. at least two blocks a CU on a 64-CU part. Smaller sizes and the empty kernel are printed
// with their residual against the line, outside the fit.
//
//   hipcc -O3 -std=c++17 --offload-arch=gfx1201 sweep.hip -o sweep && ./sweep [batches (5)] [KiB ...]
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "devid.h"

static const size_t kPool = (size_t)1 << 30, kBlockBytes = 8192;

__global__ __launch_bounds__(256) void read8k(const uint4* __restrict__ src, unsigned* out) {
    const uint4* p = src + blockIdx.x * (kBlockBytes / 16) + threadIdx.x * 2;
    const uint4 v0 = p[0], v1 = p[1];
    const unsigned acc = v0.x ^ v0.y ^ v0.z ^ v0.w ^ v1.x ^ v1.y ^ v1.z ^ v1.w;
    if (acc == 0x2545f491u) out[0] = acc;   // practically never; keeps the loads alive
}

// graphrate.hip's `tiny`: one block, one store -- the author's "empty" dispatch.
__global__ void tiny(unsigned* out) { if (threadIdx.x == 0) out[blockIdx.x] = 1u; }

__global__ void fill(unsigned* p, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = (unsigned)(i * 0x9e3779b97f4a7c15ull >> 32);
}

static size_t g_off = 0;   // where the next launch reads; runs on across batches and sizes

// Device microseconds per dispatch for R back-to-back launches of S bytes (S = 0: the empty kernel).
static double per_dispatch_us(const uint4* pool, unsigned* out, size_t S, int R) {
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    hipLaunchKernelGGL(read8k, dim3((unsigned)(kPool / kBlockBytes)), dim3(256), 0, 0, pool, out);   // blocker
    hipEventRecord(a);
    for (int i = 0; i < R; ++i) {
        if (S == 0) hipLaunchKernelGGL(tiny, dim3(1), dim3(64), 0, 0, out);
        else {
            if (g_off + S > kPool) g_off = 0;
            hipLaunchKernelGGL(read8k, dim3((unsigned)(S / kBlockBytes)), dim3(256), 0, 0, pool + g_off / 16, out);
            g_off += S;
        }
    }
    hipEventRecord(b); hipEventSynchronize(b);
    float ms = 0; hipEventElapsedTime(&ms, a, b);
    if (hipGetLastError() != hipSuccess) { std::printf("launch failed (S %zu)\n", S); std::exit(1); }
    hipEventDestroy(a); hipEventDestroy(b);
    return ms * 1e3 / R;
}

int main(int argc, char** argv) {
    const int inner = argc > 1 ? std::atoi(argv[1]) : 5;   // medians of this many batches a size
    if (calib_print_device(0)) return 1;
    uint4* pool = nullptr; unsigned* out = nullptr;
    if (hipMalloc(&pool, kPool) != hipSuccess || hipMalloc(&out, 64) != hipSuccess) { std::printf("hipMalloc\n"); return 1; }
    hipLaunchKernelGGL(fill, dim3(4096), dim3(256), 0, 0, (unsigned*)pool, kPool / 4);
    hipDeviceSynchronize();

    const size_t KiB = 1024, MiB = KiB * KiB;
    std::vector<size_t> sizes = {0, 64 * KiB, 256 * KiB, 512 * KiB, 1 * MiB, 2 * MiB, 4 * MiB, 8 * MiB,
                                 16 * MiB, 32 * MiB, 64 * MiB};
    if (argc > 2) {   // sweep [inner] [KiB ...]: other sizes, multiples of 8 KiB (0 = the empty kernel)
        sizes.clear();
        for (int i = 2; i < argc; ++i) sizes.push_back((size_t)std::atol(argv[i]) / 8 * 8 * KiB);
    }
    std::vector<double> xs, ts;
    for (size_t S : sizes) {
        const int R = (int)std::max<size_t>(20, std::min<size_t>(200, S ? 512 * MiB / S : 200));
        per_dispatch_us(pool, out, S, R);   // warm
        std::vector<double> v;
        for (int k = 0; k < inner; ++k) v.push_back(per_dispatch_us(pool, out, S, R));
        std::sort(v.begin(), v.end());
        xs.push_back((double)S); ts.push_back(v[v.size() / 2]);
    }
    // Weighted least squares, w = 1/t^2, over 1..64 MiB.
    double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (size_t i = 0; i < xs.size(); ++i) {
        if (xs[i] < 1 * MiB) continue;
        const double w = 1.0 / (ts[i] * ts[i]);
        sw += w; sx += w * xs[i]; sy += w * ts[i]; sxx += w * xs[i] * xs[i]; sxy += w * xs[i] * ts[i];
    }
    const double slope = (sw * sxy - sx * sy) / (sw * sxx - sx * sx);   // us a byte
    const double fixed = (sy - slope * sx) / sw;
    double rr = 0; int nf = 0;
    std::printf("sweep: per-dispatch device time, back to back, median of %d batches a size\n", inner);
    for (size_t i = 0; i < xs.size(); ++i) {
        const double fit = fixed + slope * xs[i], res = ts[i] - fit;
        const bool in = xs[i] >= 1 * MiB;
        if (in) { rr += (res / ts[i]) * (res / ts[i]); ++nf; }
        std::printf("sweep bytes=%zu us=%.3f fit=%.3f resid_us=%+.3f resid_pct=%+.2f%s\n", (size_t)xs[i], ts[i], fit, res,
                    100.0 * res / ts[i], in ? "" : " (outside fit)");
    }
    std::printf("sweep fixed_us=%.3f slope_gbps=%.1f rms_resid_pct=%.2f points=%d empty_us=%.3f\n", fixed,
                1e-3 / slope, 100.0 * std::sqrt(rr / nf), nf, ts[0]);
    return 0;
}
