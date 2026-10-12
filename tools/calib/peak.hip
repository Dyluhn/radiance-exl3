// peak.hip -- the card's PEAK streaming-read bandwidth (tuned: loads in flight a thread, occupancy, access order,
// buffer size beyond and inside the last-level cache) and a MEMORY-LEVEL-PARALLELISM sweep (bytes in flight a wave x
// waves a SIMD -> GB/s), the denominators of the decode kernels' targets. radiance-exl3 P4 (2026-10-12). Apache-2.0.
//
// stream.hip (S0.11) gives 679.5 GB/s for one form (two 16-B loads a thread a step, 1 GiB). This asks whether any
// form reads faster, and what a kernel must keep in flight to get there:
//   read   U independent 16-B loads a thread a step (U = 1..16), 256-thread blocks, B blocks a WGP's worth of grid,
//          order "stride" (a load instruction of a wave reads 512 contiguous bytes, the grid strides) or "chunk" (each
//          block walks its own contiguous slice), plain or nontemporal; a buffer of S bytes read R times a launch
//          (S = 1 GiB / 256 MiB: DRAM; 48 MiB: inside the 64 MB last-level cache).
//   mlp    a wave keeps exactly U loads of W bytes a lane in flight (issue U, consume all, issue the next U), over its
//          own contiguous region; 128-thread blocks (a wave a SIMD), 32 x V blocks resident (V waves a SIMD, nominal):
//          in flight = U x W x 32 bytes a wave, x 4 V a WGP.
// Every result line: mode, parameters, GB/s (median of the timed launches after two warm ones).
//
//   hipcc -O3 -std=c++17 --offload-arch=gfx1201 peak.hip -o peak && ./peak [read|mlp|all] [reps]
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "devid.h"

__global__ void fill(unsigned* p, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = (unsigned)(i * 0x9e3779b97f4a7c15ull >> 32);
}

typedef unsigned v4u __attribute__((ext_vector_type(4)));
typedef unsigned v2u __attribute__((ext_vector_type(2)));
template <bool NT>
__device__ __forceinline__ v4u ld16(const uint4* p) {
    if constexpr (NT) return __builtin_nontemporal_load((const v4u*)p);
    else return *(const v4u*)p;
}

// read: n16 16-B units, R passes. stride order: unit i + u * T of step s (T = threads in the grid); chunk order: block b
// owns units [b * per, (b + 1) * per) and its threads stride over them.
template <int U, bool NT, bool CHUNK>
__global__ __launch_bounds__(256) void read_kernel(const uint4* __restrict__ src, size_t n16, int passes, unsigned* sink) {
    unsigned acc = 0;
    size_t lo, hi, step, i0;
    if (CHUNK) {
        const size_t per = (n16 + gridDim.x - 1) / gridDim.x;
        lo = blockIdx.x * per; hi = min(lo + per, n16); step = blockDim.x; i0 = lo + threadIdx.x;
    } else {
        lo = 0; hi = n16; step = (size_t)gridDim.x * blockDim.x; i0 = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    }
    for (int r = 0; r < passes; ++r) {
        size_t i = i0;
        for (; i + (U - 1) * step < hi; i += U * step) {
            v4u v[U];
#pragma unroll
            for (int u = 0; u < U; ++u) v[u] = ld16<NT>(src + i + u * step);
#pragma unroll
            for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
        }
        for (; i < hi; i += step) { const v4u v = ld16<NT>(src + i); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    }
    if (acc == 0x2545f491u) sink[0] = acc;   // practically never; keeps the loads alive
}

// mlp: a wave's region is `per` bytes; each step its lanes issue U loads of W bytes (lane-contiguous within a load:
// one load instruction of the wave reads 32 W contiguous bytes, the U loads are consecutive such pieces), then use
// all of them before the next step's loads -- exactly U x W x 32 bytes in flight a wave.
template <int U, int W>
__global__ __launch_bounds__(128) void mlp_kernel(const unsigned char* __restrict__ src, size_t per, unsigned* sink) {
    const int lane = threadIdx.x % 32, wave = (int)(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
    const unsigned char* base = src + (size_t)wave * per;
    constexpr size_t kStep = (size_t)U * W * 32;
    unsigned acc = 0;
    for (size_t off = 0; off + kStep <= per; off += kStep) {
        if constexpr (W == 16) {
            v4u v[U];
#pragma unroll
            for (int u = 0; u < U; ++u) v[u] = __builtin_nontemporal_load((const v4u*)(base + off + (size_t)(u * 32 + lane) * 16));
#pragma unroll
            for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
        } else if constexpr (W == 8) {
            v2u v[U];
#pragma unroll
            for (int u = 0; u < U; ++u) v[u] = __builtin_nontemporal_load((const v2u*)(base + off + (size_t)(u * 32 + lane) * 8));
#pragma unroll
            for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].y;
        } else {
            unsigned v[U];
#pragma unroll
            for (int u = 0; u < U; ++u) v[u] = __builtin_nontemporal_load((const unsigned*)(base + off + (size_t)(u * 32 + lane) * 4));
#pragma unroll
            for (int u = 0; u < U; ++u) acc ^= v[u];
        }
        // the next step's loads must not be hoisted above this step's use: a compiler barrier on the accumulator
        asm volatile("" : "+v"(acc));
    }
    if (acc == 0x2545f491u) sink[0] = acc;
}

// pf: mlp's W = 16 stream (U loads in flight a wave) plus ONE scalar prefetch a step, s_prefetch_data of the wave's
// bytes D steps ahead (len field L: the instruction's own length operand), clamped into the region. Asks whether the
// scalar prefetch warms L2 for later vector loads: if it does, GB/s at low U x V rises toward the plain stream's.
template <int U, int L>
__global__ __launch_bounds__(128) void pf_kernel(const unsigned char* __restrict__ src, size_t per, int dist, unsigned* sink) {
    const int lane = threadIdx.x % 32, wave = __builtin_amdgcn_readfirstlane((int)(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32));
    const unsigned char* base = src + (size_t)wave * per;
    constexpr size_t kStep = (size_t)U * 16 * 32;
    unsigned acc = 0;
    for (size_t off = 0; off + kStep <= per; off += kStep) {
        if (dist > 0) __builtin_amdgcn_s_prefetch_data(base + min(off + (size_t)dist * kStep, per - kStep), L);
        v4u v[U];
#pragma unroll
        for (int u = 0; u < U; ++u) v[u] = __builtin_nontemporal_load((const v4u*)(base + off + (size_t)(u * 32 + lane) * 16));
#pragma unroll
        for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
        asm volatile("" : "+v"(acc));
    }
    if (acc == 0x2545f491u) sink[0] = acc;
}

// pat: the decode GEMMs' weight access pattern. A unit = (expert, column block): NSTEPS pieces of 1 KiB (4 waves x 32
// lanes x 8 B), piece j at expert_base + j * ncb KiB + cb KiB -- the tile-stream record's k-step rows (layout.c), a
// workgroup's 8 tiles a k-step; ncb 1 is a contiguous run a unit (the blocked layout). Expert bases are scattered over
// the buffer (a hash). Units are walked persistently, unit u = cb-fastest (u % ncb, as the grid dispatches) or, with
// order 1, expert-fastest; U pieces a wave in flight (issue U, consume, repeat).
template <int U>
__global__ __launch_bounds__(128) void pat_kernel(const unsigned char* __restrict__ src, size_t nexp, int ncb, int nsteps,
                                                  int units, int order, unsigned* sink) {
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const size_t rec = (size_t)ncb * nsteps * 1024;   // one expert's record
    unsigned acc = 0;
    for (int u = blockIdx.x; u < units; u += gridDim.x) {
        const int nex = units / ncb;
        const int e = order ? u % nex : u / ncb, cb = order ? u / nex : u % ncb;
        const size_t eb = (size_t)(((unsigned)e * 2654435761u) % (unsigned)nexp) * rec;
        const unsigned char* base = src + eb + (size_t)cb * 1024 + wave * 256 + lane * 8;
        for (int j = 0; j + U <= nsteps; j += U) {
            v2u v[U];
#pragma unroll
            for (int q = 0; q < U; ++q) v[q] = __builtin_nontemporal_load((const v2u*)(base + (size_t)(j + q) * ncb * 1024));
#pragma unroll
            for (int q = 0; q < U; ++q) acc ^= v[q].x ^ v[q].y;
            asm volatile("" : "+v"(acc));
        }
    }
    if (acc == 0x2545f491u) sink[0] = acc;
}

static float time_ms(void (*launch)(void*), void* ctx, int reps) {
    hipEvent_t a, b;
    (void)hipEventCreate(&a); (void)hipEventCreate(&b);
    std::vector<float> ms;
    for (int r = -2; r < reps; ++r) {
        (void)hipEventRecord(a);
        launch(ctx);
        (void)hipEventRecord(b); (void)hipEventSynchronize(b);
        float t = 0; (void)hipEventElapsedTime(&t, a, b);
        if (r >= 0) ms.push_back(t);
    }
    if (hipGetLastError() != hipSuccess) { std::printf("launch failed\n"); std::exit(1); }
    std::sort(ms.begin(), ms.end());
    (void)hipEventDestroy(a); (void)hipEventDestroy(b);
    return ms[ms.size() / 2];
}

struct ReadCtx { const void* fn; unsigned grid; const uint4* src; size_t n16; int passes; unsigned* sink; };
static void launch_read(void* c) {
    ReadCtx* r = (ReadCtx*)c;
    void* args[] = { &r->src, &r->n16, &r->passes, &r->sink };
    (void)hipLaunchKernel(r->fn, dim3(r->grid), dim3(256), args, 0, 0);
}
struct MlpCtx { const void* fn; unsigned grid; const unsigned char* src; size_t per; unsigned* sink; };
static void launch_mlp(void* c) {
    MlpCtx* m = (MlpCtx*)c;
    void* args[] = { &m->src, &m->per, &m->sink };
    (void)hipLaunchKernel(m->fn, dim3(m->grid), dim3(128), args, 0, 0);
}

struct PatCtx { const void* fn; unsigned grid; const unsigned char* src; size_t nexp; int ncb, nsteps, units, order; unsigned* sink; };
static void launch_pat(void* c) {
    PatCtx* m = (PatCtx*)c;
    void* args[] = { &m->src, &m->nexp, &m->ncb, &m->nsteps, &m->units, &m->order, &m->sink };
    (void)hipLaunchKernel(m->fn, dim3(m->grid), dim3(128), args, 0, 0);
}
struct PfCtx { const void* fn; unsigned grid; const unsigned char* src; size_t per; int dist; unsigned* sink; };
static void launch_pf(void* c) {
    PfCtx* m = (PfCtx*)c;
    void* args[] = { &m->src, &m->per, &m->dist, &m->sink };
    (void)hipLaunchKernel(m->fn, dim3(m->grid), dim3(128), args, 0, 0);
}
template <int U> static const void* pf_fn(int len) {
    switch (len) {
        case 0: return (const void*)pf_kernel<U, 0>;   case 3: return (const void*)pf_kernel<U, 3>;
        case 7: return (const void*)pf_kernel<U, 7>;   case 15: return (const void*)pf_kernel<U, 15>;
        default: return (const void*)pf_kernel<U, 31>;
    }
}

template <int U> static const void* read_fn(bool nt, bool chunk) {
    if (nt) return chunk ? (const void*)read_kernel<U, true, true> : (const void*)read_kernel<U, true, false>;
    return chunk ? (const void*)read_kernel<U, false, true> : (const void*)read_kernel<U, false, false>;
}
static const void* read_pick(int u, bool nt, bool chunk) {
    switch (u) {
        case 1: return read_fn<1>(nt, chunk);  case 2: return read_fn<2>(nt, chunk);
        case 4: return read_fn<4>(nt, chunk);  case 8: return read_fn<8>(nt, chunk);
        default: return read_fn<16>(nt, chunk);
    }
}
template <int W> static const void* mlp_fn(int u) {
    switch (u) {
        case 1: return (const void*)mlp_kernel<1, W>;   case 2: return (const void*)mlp_kernel<2, W>;
        case 3: return (const void*)mlp_kernel<3, W>;   case 4: return (const void*)mlp_kernel<4, W>;
        case 6: return (const void*)mlp_kernel<6, W>;   case 8: return (const void*)mlp_kernel<8, W>;
        case 12: return (const void*)mlp_kernel<12, W>; default: return (const void*)mlp_kernel<16, W>;
    }
}

int main(int argc, char** argv) {
    const char* what = argc > 1 ? argv[1] : "all";
    const int reps = argc > 2 ? std::atoi(argv[2]) : 10;
    if (calib_print_device(0)) return 1;
    hipDeviceProp_t prop;
    (void)hipGetDeviceProperties(&prop, 0);
    const int wgps = prop.multiProcessorCount;   // gfx12 in WGP mode reports its WGPs here (32 on a 64-CU part)
    const size_t kBuf = (size_t)1 << 30;
    unsigned char* src = nullptr;
    unsigned* sink = nullptr;
    if (hipMalloc(&src, kBuf) != hipSuccess || hipMalloc(&sink, 64) != hipSuccess) { std::printf("hipMalloc 1 GiB\n"); return 1; }
    hipLaunchKernelGGL(fill, dim3(4096), dim3(256), 0, 0, (unsigned*)src, kBuf / 4);
    (void)hipDeviceSynchronize();
    {   // warm the clocks: 30 plain reads of 1 GiB (the S0.11 ramp trap)
        ReadCtx w{ read_pick(4, false, false), (unsigned)(wgps * 16), (const uint4*)src, kBuf / 16, 1, sink };
        for (int i = 0; i < 30; ++i) launch_read(&w);
        (void)hipDeviceSynchronize();
    }
    std::printf("peak: %d WGPs, reps %d (median)\n", wgps, reps);
    if (!std::strcmp(what, "read") || !std::strcmp(what, "all")) {
        struct Size { size_t bytes; int passes; } sizes[] = { { kBuf, 1 }, { kBuf / 4, 2 }, { (size_t)48 << 20, 16 } };
        for (const Size& s : sizes)
            for (int chunk = 0; chunk < 2; ++chunk)
                for (int nt = 0; nt < 2; ++nt)
                    for (int u : { 1, 2, 4, 8, 16 })
                        for (int bpw : { 4, 8, 16, 32 }) {   // 256-thread blocks a WGP's share of the grid
                            ReadCtx c{ read_pick(u, nt, chunk), (unsigned)(wgps * bpw), (const uint4*)src, s.bytes / 16, s.passes, sink };
                            const float ms = time_ms(launch_read, &c, reps);
                            std::printf("read size_mib=%zu order=%s nt=%d U=%d blocks_per_wgp=%d gbps=%.1f\n", s.bytes >> 20,
                                        chunk ? "chunk" : "stride", nt, u, bpw, (double)s.bytes * s.passes / (ms * 1e6));
                        }
    }
    if (!std::strcmp(what, "mlp") || !std::strcmp(what, "all")) {
        for (int w : { 4, 16 })
            for (int v : { 1, 2, 4, 6, 8, 12, 16 })   // waves a SIMD (nominal: 32 V four-wave blocks, all resident)
                for (int u : { 1, 2, 3, 4, 6, 8, 12, 16 }) {
                    const unsigned grid = (unsigned)(wgps * v);
                    const size_t waves = (size_t)grid * 4, step = (size_t)u * w * 32;
                    const size_t per = kBuf / waves / step * step;
                    MlpCtx c{ w == 16 ? mlp_fn<16>(u) : w == 8 ? mlp_fn<8>(u) : mlp_fn<4>(u), grid, src, per, sink };
                    const float ms = time_ms(launch_mlp, &c, reps);
                    std::printf("mlp W=%d waves_per_simd=%d U=%d inflight_wave_B=%zu inflight_wgp_KiB=%.1f gbps=%.1f\n", w, v, u,
                                step, (double)step * 4 * v / 1024, (double)per * waves / (ms * 1e6));
                }
    }
    if (!std::strcmp(what, "pat") || !std::strcmp(what, "all")) {
        struct Shape { const char* name; int ncb, nsteps; } shapes[] = { { "down", 20, 24 }, { "down_blocked", 1, 24 },
                                                                        { "gate_up", 3, 160 }, { "gate_up_blocked", 1, 160 } };
        for (const Shape& sh : shapes)
            for (int order : { 0, 1 })
                for (int k : { 4, 8, 14 })
                    for (int u : { 2, 4 }) {
                        if (sh.ncb == 1 && order) continue;
                        const size_t rec = (size_t)sh.ncb * sh.nsteps * 1024, nexp = kBuf / rec;
                        // units: ~200 MB of pieces a launch, whole experts
                        const int nex = (int)std::min<size_t>(nexp, (size_t)(200u << 20) / rec), units = nex * sh.ncb;
                        PatCtx c{ u == 2 ? (const void*)pat_kernel<2> : (const void*)pat_kernel<4>, (unsigned)(wgps * k), src, nexp,
                                  sh.ncb, sh.nsteps, units, order, sink };
                        const float ms = time_ms(launch_pat, &c, reps);
                        std::printf("pat shape=%s order=%s wgs_per_wgp=%d U=%d gbps=%.1f\n", sh.name, order ? "expert" : "cb", k, u,
                                    (double)units * sh.nsteps * 1024 / (ms * 1e6));
                    }
    }
    if (!std::strcmp(what, "pf") || !std::strcmp(what, "all")) {
        for (int v : { 1, 2, 4 })
            for (int u : { 1, 2 })
                for (int len : { 0, 3, 7, 15, 31 })
                    for (int dist : { 0, 2, 4, 8, 16, 32 }) {
                        if (dist == 0 && len != 0) continue;
                        const unsigned grid = (unsigned)(wgps * v);
                        const size_t waves = (size_t)grid * 4, step = (size_t)u * 16 * 32;
                        const size_t per = kBuf / waves / step * step;
                        PfCtx c{ u == 1 ? pf_fn<1>(len) : pf_fn<2>(len), grid, src, per, dist, sink };
                        const float ms = time_ms(launch_pf, &c, reps);
                        std::printf("pf waves_per_simd=%d U=%d len=%d dist=%d step_B=%zu gbps=%.1f\n", v, u, len, dist, step,
                                    (double)per * waves / (ms * 1e6));
                    }
    }
    return 0;
}
