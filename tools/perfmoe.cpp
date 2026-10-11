// perfmoe -- the routed-expert decode dispatch, timed: exl3r4d's moe_exl3_gate_up / moe_exl3_down against libr4d's
// moe_gemm_w4a8 (the baseline's served w4nl64a8h form) on the SAME routing and geometry, in ONE harness, modelled on
// libr4d's `r4d_selftest --perfmoe` (libr4d/r4d_selftest.cpp:5653-5930 at radiance 1.3.0).
//
//   perfmoe --exl3 <exl3r4d.so> [--r4d <libr4d.so>] [--op gate_up|down] [--tp 1|2] [--M m] [--layers L] [--pools P]
//           [--experts E] [--topk k] [--hidden H] [--width W] [--iters n] [--reps r] [--ceiling GB/s]
//           [--warm-ms ms (500)] [--exl3-row <row name>] [--rand-data] [--h-rot 0|1] [--dump-stamps] [--pooled P]
//
// --pooled P: P% of each parity class's experts -- the least routed over the draws -- have every table entry (codes
// and scale planes, both libraries) in pinned, mapped host memory: the engine's non-resident tier, read over the link.
//
// --exl3-row picks the exl3r4d row by name (default: the decode band's exl3r4d_moe_gate_up_m16 / _down_m16), e.g.
// the prefill band's rows; it must implement the --op asked for.
//
// WHAT IS TIMED. `iters` passes over L LAYERS (default 48, the model's), launched back to back on one stream between
// two events; the time a dispatch is the elapsed time over iters * L. Every layer has its OWN routing draw, over one
// of P expert POOLS (layer l reads pool l % P; default 2; a pool is E experts of real size, its own allocations and
// pointer tables: 840 MB for gate_up tp1). An expert therefore comes round again only after ~E / live dispatches,
// i.e. after about a pool's worth of other bytes: every dispatch streams from DRAM, as a decode step does. (FACTS-exl3
// trap 2. The first version of this tool kept ONE routing per pool, so the bytes recycled were only the routed
// experts -- 4 x 15.5 MB at down tp2 M=4, inside the 64 MB last-level cache -- and libr4d read 701 GB/s, above the
// 679.5 GB/s stream ceiling. `--pools 1 --layers 1` is that cache-resident case, kept as the harness's control.)
// HOT means every expert is in VRAM (no pooled experts over the link). Before timing, --warm-ms of the same launches
// bring the clocks up; the libraries then alternate within every repetition (ABAB), so drift lands on both.
//
// THE BYTES are the weight bytes the dispatch must read: for every LIVE expert (one with a routed row), its whole
// codes plane plus its scale planes -- exl3r4d: codes R*C*4/8 + suh (+ suh.1) + svh; libr4d w4nl64a8h: codes R*C/2 +
// E4M3 scales R*2C/128. Activation and output bytes are not counted (they are small and the same for both). GB/s =
// bytes / time a dispatch, which INCLUDES the dispatch's fixed cost (the K-sweep separates the two: --hidden).
//
// THE ROUTING is libr4d's perfmoe draw: expert popularity lognormal (sigma 0.8), each token's top_k by a Gumbel draw
// over it, a stable sort; at tp2 both libraries see rank 0's parity split (even experts 384 wide, odd 256).
// Prints one line a (library, repetition) and a summary: median [min, max] us and GB/s, and GB/s / --ceiling.
// Exit 0, or 1 when a launch fails (its return code printed).
#include "rad_abi.h"
#include "rad_plugin.h"

#include <hip/hip_runtime_api.h>

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Plugin-private dtypes the two libraries' tables carry once laid out: exl3r4d's trellis record
// (kernels/exl3r4d/exl3r4d.h EXL3R4D_DT_TRELLIS) and libr4d's MoE w4 plane (libr4d/r4d_args.h:273 at 1.3.0).
constexpr uint32_t kDtTrellis = RAD_DT_PLUGIN_BASE + 0;
constexpr uint32_t kDtMoeW4 = RAD_DT_PLUGIN_BASE + 7;

[[noreturn]] void die(const std::string& m) {
    std::fprintf(stderr, "perfmoe: %s\n", m.c_str());
    std::exit(2);
}
void ck(hipError_t e, const char* what) {
    if (e != hipSuccess) die(std::string(what) + ": " + hipGetErrorString(e));
}
void* dalloc(size_t n) {
    void* p = nullptr;
    ck(hipMalloc(&p, n < 256 ? 256 : n), "hipMalloc");
    return p;
}
void put(void* d, const void* h, size_t n) { ck(hipMemcpy(d, h, n, hipMemcpyHostToDevice), "copy"); }

uint64_t g_rs = 0x9E3779B97F4A7C15ull;
uint64_t rnd() { g_rs ^= g_rs << 13; g_rs ^= g_rs >> 7; g_rs ^= g_rs << 17; return g_rs; }
double unif() { return ((double)(rnd() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }

RadParam PI(const char* k, long long v) { RadParam p = RAD_INT(k, v); return p; }
RadParam PS(const char* k, const char* v) { RadParam p = RAD_STR(k, v); return p; }
RadTensor T(void* d, uint32_t dt, std::vector<int64_t> s) {
    RadTensor t;
    std::memset(&t, 0, sizeof t);
    t.data = d;
    t.dtype = dt;
    t.rank = (uint32_t)s.size();
    for (size_t i = 0; i < s.size(); ++i) t.shape[i] = s[i];
    rad_tensor_pack(&t);
    return t;
}
RadTensor TAB(void* tab, uint32_t dt, std::vector<int64_t> s) {   // a pointer table: stride[0] = 0
    RadTensor t = T(tab, dt, s);
    t.stride[0] = 0;
    return t;
}

const RadKernelInfo* row_of(const char* lib, const char* name) {
    void* h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!h) die(std::string("dlopen ") + lib + ": " + dlerror());
    auto count = (int (*)())dlsym(h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    if (!count || !at) die(std::string(lib) + " is not a radiance kernel library");
    for (int i = 0; i < count(); ++i)
        if (at(i) && at(i)->name && std::strcmp(at(i)->name, name) == 0) return at(i);
    die(std::string(lib) + " has no row " + name);
}

struct Opts {
    std::string exl3, r4d, op = "gate_up", exl3_row;
    int tp = 1, layers = 48, pools = 2, iters = 4, reps = 5;
    int64_t M = 1, E = 512, topk = 10, hidden = 2560, width = 640;
    double ceiling = 0;
    float warm_ms = 500;
    bool rand_data = false;
    bool dump_stamps = false;
    int h_rot = 0;
    int pooled = 0;           // --pooled P: P% of each class's experts (the least routed) read from mapped host memory            // --h-rot 1: interface v3 EPI 1 (gate_up writes h rotated for down; down reads it so) // --dump-stamps: print exl3r4d's EXL3R4D_TSTAMP phase stamps (a timing-only build)   // --rand-data: x random, suh / suh.1 random +-1, svh random (the default: constants)
};

// One routing draw: sorted rows and offsets over E experts, and how many rows each expert got.
struct Route {
    std::vector<int32_t> sorted, off;
    std::vector<int64_t> rows;
};
Route draw(int64_t M, int64_t E, int64_t topk) {
    std::vector<double> logp((size_t)E);
    for (auto& v : logp) v = 0.8 * std::sqrt(-2.0 * std::log(unif())) * std::cos(6.283185307 * unif());
    std::vector<std::vector<int32_t>> by((size_t)E);
    std::vector<std::pair<double, int>> sc((size_t)E);
    for (int64_t t = 0; t < M; ++t) {
        for (int64_t e = 0; e < E; ++e) sc[(size_t)e] = { logp[(size_t)e] - std::log(-std::log(unif())), (int)e };
        std::partial_sort(sc.begin(), sc.begin() + topk, sc.end(),
                          [](const std::pair<double, int>& a, const std::pair<double, int>& b) { return a.first > b.first; });
        for (int64_t j = 0; j < topk; ++j) by[(size_t)sc[(size_t)j].second].push_back((int32_t)(t * topk + j));
    }
    Route r;
    r.off.assign((size_t)E + 1, 0);
    r.rows.assign((size_t)E, 0);
    for (int64_t e = 0; e < E; ++e) {
        r.rows[(size_t)e] = (int64_t)by[(size_t)e].size();
        r.off[(size_t)e + 1] = r.off[(size_t)e] + (int32_t)r.rows[(size_t)e];
        r.sorted.insert(r.sorted.end(), by[(size_t)e].begin(), by[(size_t)e].end());
    }
    return r;
}

// A class's entries in one buffer a plane and a device pointer table onto them.
// The engine's non-resident tier (playbook R5, libr4d's moe_perf pool_pct, r4d_selftest.cpp:5742-5782): a pooled
// entry points into pinned, mapped host memory (a copy of the device entry), which the GEMM reads over the link.
std::vector<char> g_pooled[2];   // [class][entry]: read from host memory
void* table_of(void* base, int64_t n, size_t stride, int cls = -1) {
    std::vector<void*> p((size_t)n);
    for (int64_t i = 0; i < n; ++i) {
        p[(size_t)i] = (char*)base + (size_t)i * stride;
        if (cls < 0 || g_pooled[cls].empty() || !g_pooled[cls][(size_t)i]) continue;
        void* h = nullptr;
        void* hd = nullptr;
        ck(hipHostMalloc(&h, stride, hipHostMallocMapped), "hipHostMalloc (pooled entry)");
        ck(hipMemcpy(h, p[(size_t)i], stride, hipMemcpyDeviceToHost), "copy (pooled entry)");
        ck(hipHostGetDevicePointer(&hd, h, 0), "hipHostGetDevicePointer");
        p[(size_t)i] = hd;
    }
    void* d = dalloc(p.size() * sizeof(void*));
    put(d, p.data(), p.size() * sizeof(void*));
    return d;
}
void* random_buf(size_t n, uint8_t lo = 0, uint8_t span = 255) {
    std::vector<uint8_t> h(n);
    if (lo == 0 && span == 255) {   // any bytes: eight a draw (the weight pools are gigabytes)
        for (size_t i = 0; i + 8 <= n; i += 8) { const uint64_t v = rnd(); std::memcpy(&h[i], &v, 8); }
        for (size_t i = n / 8 * 8; i < n; ++i) h[i] = (uint8_t)rnd();
    } else {
        for (auto& b : h) b = (uint8_t)(lo + (uint8_t)(rnd() % span));
    }
    void* d = dalloc(n);
    put(d, h.data(), n);
    return d;
}
void* f16_buf(size_t n, float v) {
    std::vector<uint16_t> h(n, rad_f32_to_f16(v));
    void* d = dalloc(n * 2);
    put(d, h.data(), n * 2);
    return d;
}
// kind 0: uniform in (-v, v); 1: +-v (the EXL3 sign vectors suh / suh.1 are +-1); 2: uniform in (v/2, 3v/2)
void* f16_rand(size_t n, float v, int kind) {
    std::vector<uint16_t> h(n);
    for (auto& e : h) {
        const double u = unif();
        const float f = kind == 1 ? (u < 0.5 ? -v : v) : kind == 2 ? (float)(v * (0.5 + u)) : (float)(v * (2 * u - 1));
        e = rad_f32_to_f16(f);
    }
    void* d = dalloc(n * 2);
    put(d, h.data(), n * 2);
    return d;
}

struct Lib {
    const char* name;
    const RadKernelInfo* row;
    std::vector<std::vector<RadTensor>> t;   // [layer] operands
    std::vector<RadParam> p;
    std::vector<double> bytes;               // [layer] weight bytes the dispatch reads
    void* scratch = nullptr;
    void* y = nullptr;
    int64_t scratch_bytes = 0;
    std::vector<double> us;                  // [rep]
};

}  // namespace

int main(int argc, char** argv) {
    Opts o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() -> const char* { if (i + 1 >= argc) die("missing value for " + a); return argv[++i]; };
        if (a == "--exl3") o.exl3 = nx();
        else if (a == "--r4d") o.r4d = nx();
        else if (a == "--exl3-row") o.exl3_row = nx();
        else if (a == "--op") o.op = nx();
        else if (a == "--tp") o.tp = std::atoi(nx());
        else if (a == "--M") o.M = std::atoll(nx());
        else if (a == "--layers") o.layers = std::atoi(nx());
        else if (a == "--pools") o.pools = std::atoi(nx());
        else if (a == "--rand-data") o.rand_data = true;
        else if (a == "--dump-stamps") o.dump_stamps = true;
        else if (a == "--h-rot") o.h_rot = std::atoi(nx());
        else if (a == "--pooled") o.pooled = std::atoi(nx());
        else if (a == "--experts") o.E = std::atoll(nx());
        else if (a == "--topk") o.topk = std::atoll(nx());
        else if (a == "--hidden") o.hidden = std::atoll(nx());
        else if (a == "--width") o.width = std::atoll(nx());
        else if (a == "--iters") o.iters = std::atoi(nx());
        else if (a == "--reps") o.reps = std::atoi(nx());
        else if (a == "--ceiling") o.ceiling = std::atof(nx());
        else if (a == "--warm-ms") o.warm_ms = (float)std::atof(nx());
        else die("unknown argument " + a);
    }
    if (o.exl3.empty()) die("--exl3 <exl3r4d.so> is required");
    if (o.op != "gate_up" && o.op != "down") die("--op is gate_up or down");
    if (o.tp != 1 && o.tp != 2) die("--tp is 1 or 2");
    const bool gu = o.op == "gate_up";
    // This rank's widths by parity: tp1 every expert whole; tp2 rank 0 holds 3 of 5 128-blocks of the even experts
    // and 2 of the odd ones (PLAN.md D7) -- the general rule is the arch's, this is the Qwen3.8 shape.
    const int64_t w0 = o.tp == 1 ? o.width : (o.width / 128 + 1) / 2 * 128, w1 = o.tp == 1 ? o.width : o.width - w0;
    const int64_t wq[2] = { w0, w1 };
    const int64_t R[2] = { gu ? 2 * wq[0] : o.hidden, gu ? 2 * wq[1] : o.hidden };   // [rows, cols] a class
    const int64_t C[2] = { gu ? o.hidden : wq[0], gu ? o.hidden : wq[1] };
    const int64_t nec[2] = { (o.E + 1) / 2, o.E / 2 };
    const int64_t Tr = o.M * o.topk;
    const int64_t xcols = std::max(C[0], C[1]), ycols = std::max(R[0], R[1]) / (gu ? 2 : 1);
    const int64_t Ma = gu ? o.M : Tr;

    int dev = 0;
    hipDeviceProp_t prop;
    ck(hipGetDevice(&dev), "hipGetDevice");
    ck(hipGetDeviceProperties(&prop, dev), "props");
    char bus[32] = {0};
    ck(hipDeviceGetPCIBusId(bus, sizeof bus, dev), "bus id");
    std::printf("perfmoe: device 0 pci %s %s; %s tp%d M=%lld (T=%lld rows), %lld experts top-%lld, hidden %lld, "
                "width %lld (this rank: %lld/%lld), %d layers over %d pools x %d iters x %d reps\n",
                bus, prop.gcnArchName, o.op.c_str(), o.tp, (long long)o.M, (long long)Tr, (long long)o.E,
                (long long)o.topk, (long long)o.hidden, (long long)o.width, (long long)wq[0], (long long)wq[1],
                o.layers, o.pools, o.iters, o.reps);

    std::vector<Route> route;
    double live_sum = 0;
    for (int l = 0; l < o.layers; ++l) {
        route.push_back(draw(o.M, o.E, o.topk));
        for (int64_t e = 0; e < o.E; ++e) live_sum += route.back().rows[(size_t)e] > 0;
    }
    std::printf("perfmoe: %.1f live experts a dispatch (mean over layers)\n", live_sum / o.layers);
    if (o.pooled > 0) {   // the least routed P% of each class's experts, over every draw (residency keeps the hot ones)
        for (int c = 0; c < 2; ++c) {
            std::vector<int64_t> cnt((size_t)nec[c], 0), order((size_t)nec[c]);
            for (const Route& r : route)
                for (int64_t x = 0; x < nec[c]; ++x) cnt[(size_t)x] += r.rows[(size_t)(2 * x + c)];
            for (int64_t x = 0; x < nec[c]; ++x) order[(size_t)x] = x;
            std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return cnt[(size_t)a] < cnt[(size_t)b]; });
            g_pooled[c].assign((size_t)nec[c], 0);
            int64_t np = nec[c] * o.pooled / 100, rows = 0, all = 0;
            for (int64_t i = 0; i < np; ++i) { g_pooled[c][(size_t)order[(size_t)i]] = 1; rows += cnt[(size_t)order[(size_t)i]]; }
            for (int64_t x = 0; x < nec[c]; ++x) all += cnt[(size_t)x];
            std::printf("perfmoe: pooled class %d: %lld of %lld experts in mapped host memory, %.2f%% of routed rows on them\n",
                        c, (long long)np, (long long)nec[c], all ? 100.0 * (double)rows / (double)all : 0.0);
        }
    }

    std::vector<Lib> libs;
    // ---------------------------------------------------------------- exl3r4d
    {
        const std::string rn = !o.exl3_row.empty() ? o.exl3_row : gu ? "exl3r4d_moe_gate_up_m16" : "exl3r4d_moe_down_m16";
        Lib L{ "exl3r4d", row_of(o.exl3.c_str(), rn.c_str()), {}, {}, {} };
        if (std::strcmp(L.row->op, gu ? "moe_exl3_gate_up" : "moe_exl3_down") != 0)
            die("--exl3-row " + rn + " implements " + L.row->op + ", not the --op asked for");
        L.p = { PI("M", o.M), PI("N", R[0]), PI("K", C[0]), PI("n_expert", o.E), PI("top_k", o.topk),
                PS("dtype", "exl3k4"), PS("codebook", "mul1"), PI("N_odd", R[1]), PI("K_odd", C[1]),
                PI("h_rot", o.h_rot) };
        void* x = o.rand_data ? f16_rand((size_t)(Ma * xcols), 1.0f, 0) : f16_buf((size_t)(Ma * xcols), 0.5f);
        void* y = dalloc((size_t)(Tr * ycols * 2));
        L.y = y;
        std::vector<std::array<std::array<RadTensor, 5>, 2>> pool((size_t)o.pools);
        for (auto& pc : pool)
            for (int c = 0; c < 2; ++c) {
                const size_t cb = (size_t)(R[c] * C[c] / 2);
                void* codes = random_buf(cb * (size_t)nec[c]);
                void* suh = o.rand_data ? f16_rand((size_t)(C[c] * nec[c]), 1.0f, 1) : f16_buf((size_t)(C[c] * nec[c]), 0.01f);
                void* suh1 = o.rand_data ? f16_rand((size_t)(C[c] * nec[c]), 1.0f, 1) : f16_buf((size_t)(C[c] * nec[c]), 0.01f);
                void* svh = o.rand_data ? f16_rand((size_t)(R[c] * nec[c]), 0.01f, 2) : f16_buf((size_t)(R[c] * nec[c]), 1.0f);
                pc[c][0] = TAB(table_of(codes, nec[c], cb, c), kDtTrellis, { nec[c], R[c], C[c] });
                pc[c][1] = TAB(table_of(suh, nec[c], (size_t)C[c] * 2, c), RAD_F16, { nec[c], 1, C[c] });
                pc[c][2] = TAB(table_of(suh1, nec[c], (size_t)C[c] * 2, c), RAD_F16, { nec[c], 1, C[c] });
                pc[c][3] = TAB(table_of(svh, nec[c], (size_t)R[c] * 2, c), RAD_F16, { nec[c], R[c], 1 });
                if (gu) {   // v3: down's suh of the same experts, [1, R / 2] an entry (read with --h-rot 1)
                    void* dsuh = o.rand_data ? f16_rand((size_t)(R[c] / 2 * nec[c]), 1.0f, 1)
                                             : f16_buf((size_t)(R[c] / 2 * nec[c]), 0.01f);
                    pc[c][4] = TAB(table_of(dsuh, nec[c], (size_t)R[c] / 2 * 2, c), RAD_F16, { nec[c], 1, R[c] / 2 });
                }
            }
        for (int l = 0; l < o.layers; ++l) {
            const Route& rt = route[(size_t)l];
            void* ds = dalloc((size_t)Tr * 4);
            void* doff = dalloc((size_t)(o.E + 1) * 4);
            put(ds, rt.sorted.data(), (size_t)Tr * 4);
            put(doff, rt.off.data(), (size_t)(o.E + 1) * 4);
            const auto& pl = pool[(size_t)(l % o.pools)];
            double bytes = 0;
            for (int c = 0; c < 2; ++c)
                for (int64_t e = c; e < o.E; e += 2)
                    if (rt.rows[(size_t)e]) bytes += (double)(R[c] * C[c] / 2) + 2.0 * ((gu ? 2 : 1) * C[c] + R[c]);
            L.bytes.push_back(bytes);
            const RadTensor xt = T(x, RAD_F16, { Ma, xcols }), yt = T(y, gu ? RAD_F16 : RAD_BF16, { Tr, ycols });
            const RadTensor st = T(ds, RAD_I32, { Tr }), ot = T(doff, RAD_I32, { o.E + 1 });
            if (gu)
                L.t.push_back({ xt, pl[0][0], pl[0][1], pl[0][2], pl[0][3], pl[0][4], st, ot, pl[1][0], pl[1][1],
                                pl[1][2], pl[1][3], pl[1][4], yt });
            else
                L.t.push_back({ xt, pl[0][0], pl[0][1], pl[0][3], st, ot, pl[1][0], pl[1][1], pl[1][3], yt });
        }
        libs.push_back(L);
    }
    // ---------------------------------------------------------------- libr4d, w4nl64a8h (the baseline's experts)
    if (!o.r4d.empty()) {
        Lib L{ "libr4d w4nl64a8h", row_of(o.r4d.c_str(), "moe_gemm_w4a8"), {}, {}, {} };
        L.p = { PI("M", o.M), PI("N", R[0]), PI("K", C[0]), PI("N_odd", R[1]), PI("K_odd", C[1]),
                PI("parts", gu ? 2 : 1), PI("n_expert", o.E), PI("top_k", o.topk), PI("group", 128),
                PS("a_order", gu ? "token" : "sorted"), PS("dtype", "w4nl64a8h") };
        void* a = random_buf((size_t)(Ma * xcols), 8, 111);
        std::vector<float> asc((size_t)(Ma * xcols / 128), 0.013f);
        void* as = dalloc(asc.size() * 4);
        put(as, asc.data(), asc.size() * 4);
        void* y = dalloc((size_t)(Tr * std::max(R[0], R[1]) * 2));
        std::vector<std::array<std::array<RadTensor, 2>, 2>> pool((size_t)o.pools);   // [pool][class] = {w, s}
        for (auto& pc : pool)
            for (int c = 0; c < 2; ++c) {
                const size_t cb = (size_t)(R[c] * C[c] / 2), sb = (size_t)(R[c] * 2 * C[c] / 128);
                void* codes = random_buf(cb * (size_t)nec[c]);
                void* sc = random_buf(sb * (size_t)nec[c], 0x30, 8);   // E4M3 near 1/64: finite, no NaN
                pc[c][0] = TAB(table_of(codes, nec[c], cb, c), kDtMoeW4, { nec[c], R[c], C[c] });
                pc[c][1] = TAB(table_of(sc, nec[c], sb, c), RAD_F8E4M3, { nec[c], R[c], 2 * C[c] / 128 });
            }
        for (int l = 0; l < o.layers; ++l) {
            const Route& rt = route[(size_t)l];
            void* ds = dalloc((size_t)Tr * 4);
            void* doff = dalloc((size_t)(o.E + 1) * 4);
            put(ds, rt.sorted.data(), (size_t)Tr * 4);
            put(doff, rt.off.data(), (size_t)(o.E + 1) * 4);
            const auto& pc = pool[(size_t)(l % o.pools)];
            double bytes = 0;
            for (int c = 0; c < 2; ++c)
                for (int64_t e = c; e < o.E; e += 2)
                    if (rt.rows[(size_t)e]) bytes += (double)(R[c] * C[c] / 2 + R[c] * 2 * C[c] / 128);
            L.bytes.push_back(bytes);
            L.t.push_back({ T(a, RAD_F8E4M3, { Ma, xcols }), T(as, RAD_F32, { Ma, xcols / 128 }), pc[0][0], pc[0][1],
                            T(ds, RAD_I32, { Tr }), T(doff, RAD_I32, { o.E + 1 }), pc[1][0], pc[1][1],
                            T(y, RAD_BF16, { Tr, std::max(R[0], R[1]) }) });
        }
        libs.push_back(L);
    }

    hipStream_t stream;
    ck(hipStreamCreate(&stream), "stream");
    auto args_of = [&](Lib& L, int l) {
        RadArgs a;
        std::memset(&a, 0, sizeof a);
        a.t = L.t[(size_t)l].data();
        a.n_t = (int)L.t[(size_t)l].size();
        a.p = L.p.data();
        a.n_p = (int)L.p.size();
        a.scratch = L.scratch;
        a.scratch_bytes = L.scratch_bytes;
        a.world_size = 1;
        return a;
    };
    for (Lib& L : libs) {
        RadArgs a = args_of(L, 0);
        L.scratch_bytes = L.row->scratch ? L.row->scratch(&a) : 0;
        if (L.scratch_bytes > 0) L.scratch = dalloc((size_t)L.scratch_bytes);
        for (int l = 0; l < o.layers; ++l) {   // warm: every layer once, and every launch must succeed
            a = args_of(L, l);
            const int rc = L.row->launch(&a, (RadStream)stream);
            if (rc != RAD_OK) { std::printf("perfmoe: %s launch failed rc=%d\n", L.name, rc); return 1; }
        }
        ck(hipStreamSynchronize(stream), "warm-up");
    }
    hipEvent_t e0, e1;
    ck(hipEventCreate(&e0), "event");
    ck(hipEventCreate(&e1), "event");
    // WARM THE CLOCKS FIRST: launch every library's layers back to back for --warm-ms (default 500) before anything is
    // timed, as r4d_selftest --perfmoe does (libr4d/r4d_selftest.cpp:5900-5910). A card coming out of idle runs its
    // first milliseconds at a low shader clock, and a timed burst of ~10 ms then measures the ramp: worst for a kernel
    // whose inner loop is VALU-heavy (the trellis decode), so it would bias the comparison against exl3r4d. The
    // repetitions below follow without a gap, alternating libraries, so the clocks stay up.
    {
        float wms = 0.0f;
        ck(hipEventRecord(e0, stream), "record");
        for (int guard = 0; guard < 1000000 && wms < o.warm_ms; ++guard) {
            for (Lib& L : libs)
                for (int l = 0; l < o.layers; ++l) {
                    RadArgs a = args_of(L, l);
                    L.row->launch(&a, (RadStream)stream);
                }
            ck(hipEventRecord(e1, stream), "record");
            ck(hipEventSynchronize(e1), "sync");
            ck(hipEventElapsedTime(&wms, e0, e1), "elapsed");
        }
        std::printf("perfmoe: warmed %.0f ms\n", wms);
    }
    for (int r = 0; r < o.reps; ++r)
        for (Lib& L : libs) {
            ck(hipEventRecord(e0, stream), "record");
            for (int it = 0; it < o.iters; ++it)
                for (int l = 0; l < o.layers; ++l) {
                    RadArgs a = args_of(L, l);
                    L.row->launch(&a, (RadStream)stream);
                }
            ck(hipEventRecord(e1, stream), "record");
            ck(hipEventSynchronize(e1), "sync");
            float ms = 0;
            ck(hipEventElapsedTime(&ms, e0, e1), "elapsed");
            L.us.push_back((double)ms * 1000.0 / (o.iters * o.layers));
        }
    for (Lib& L : libs) {
        double b = 0;
        for (double x : L.bytes) b += x;
        b /= (double)L.bytes.size();
        std::vector<double> u = L.us;
        std::sort(u.begin(), u.end());
        const double med = u[u.size() / 2];
        std::printf("RESULT %-17s %s tp%d M=%-3lld  %8.2f us [%.2f, %.2f]  %7.1f GB/s [%.1f, %.1f]  %.2f MB/dispatch",
                    L.name, o.op.c_str(), o.tp, (long long)o.M, med, u.front(), u.back(), b / (med * 1e3),
                    b / (u.back() * 1e3), b / (u.front() * 1e3), b / 1e6);
        if (o.ceiling > 0) std::printf("  %.1f%% of %.1f", 100.0 * b / (med * 1e3) / o.ceiling, o.ceiling);
        std::printf("\nREPS   %-17s %s tp%d M=%-3lld  GB/s", L.name, o.op.c_str(), o.tp, (long long)o.M);
        for (double x : L.us) std::printf(" %.1f", b / (x * 1e3));   // in run order, for the ledger
        std::printf("\n");
        if (o.dump_stamps && L.y) {   // workgroup (0, 0) of the last dispatch: entry, pass start, K loop start, K loop
            uint64_t t[6] = {};        // end, reduce end, epilogue end -- us after the entry
            ck(hipMemcpy(t, L.y, sizeof t, hipMemcpyDeviceToHost), "copy");
            std::printf("STAMPS %-16s %s tp%d M=%-3lld  us:", L.name, o.op.c_str(), o.tp, (long long)o.M);
            for (int k = 1; k < 6; ++k) std::printf(" %.2f", (double)(t[k] - t[0]) * 0.01);
            std::printf("\n");
        }
    }
    return 0;
}
