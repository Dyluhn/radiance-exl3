// perfmoe -- the routed-expert decode dispatch, timed: exl3r4d's moe_exl3_gate_up / moe_exl3_down against libr4d's
// moe_gemm_w4a8 (the baseline's served w4nl64a8h form) on the SAME routing and geometry, in ONE harness, modelled on
// libr4d's `r4d_selftest --perfmoe` (libr4d/r4d_selftest.cpp:5653-5930 at radiance 1.3.0).
//
//   perfmoe --exl3 <exl3r4d.so> [--r4d <libr4d.so>] [--op gate_up|down] [--tp 1|2] [--M m] [--layers L] [--pools P]
//           [--experts E] [--topk k] [--hidden H] [--width W] [--iters n] [--reps r] [--ceiling GB/s]
//           [--warm-ms ms (500)] [--exl3-row <row name>] [--rand-data] [--h-rot 0|1] [--dump-stamps] [--pooled P]
//           [--chain [--a-rot 1] [--gd] [--dump-y FILE]] [--bits K] [--skew S] [--group G --group-mix R]
//
// --gd (with --chain --h-rot 1, a_rot 0): also times interface v5's moe_exl3_gate_up_down row (exl3r4d_moe_gate_up_down_m16)
// as ONE op on the same tables, routing and token rows, after checking its y byte for byte against the serial chain's
// (GD-CHECK); RESULT "exl3r4d gd".
//
// --chain times one layer's routed-expert CHAIN on the same routing, dependent launches on one stream: exl3r4d gate_up
// (prep inside) -> down; libr4d gate_up -> gated_had_quant_fp8 -> down (the served baseline block's launches). The
// RESULT line is then a layer's chain, both ops' bytes. --op is ignored.
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
#include <chrono>
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
// --arena G: every device buffer carved from ONE G-GiB allocation (a bump pointer, 256-B aligned), as the engine's
// arenas place weights and activations, instead of a hipMalloc each (separate translation fragments).
size_t g_arena_bytes = 0, g_arena_used = 0;
char* g_arena = nullptr;
void* dalloc(size_t n) {
    void* p = nullptr;
    if (g_arena_bytes) {
        if (!g_arena) ck(hipMalloc((void**)&g_arena, g_arena_bytes), "hipMalloc (arena)");
        const size_t a = (n + 255) / 256 * 256;
        if (g_arena_used + a > g_arena_bytes) { std::fprintf(stderr, "perfmoe: --arena too small\n"); std::exit(2); }
        p = g_arena + g_arena_used;
        g_arena_used += a;
        return p;
    }
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
    bool gd = false;
    bool host_us = false;     // --host-us: host microseconds a layer spent inside the libraries' launch() calls          // --gd (with --chain --h-rot 1, a_rot 0): the v5 row as one op beside the chain
    int fused = 0;            // --fused W (with --chain --a-rot 1 --h-rot 1): an EXL3R4D_FUSED_PROTO build's gate_up + down
                              // as one launch of W workgroups; checked byte for byte against the serial chain first
    bool skip_prep = false;   // --skip-prep (with --chain --a-rot 1, TIMING ONLY, wrong output): no prep_gu dispatch -- the
                              // ceiling of folding it into the routed launch
    bool overlap = false;     // --overlap (with --chain, TIMING ONLY, wrong output): a library's chain launches beside each
                              // other on two streams, joined a layer -- the ceiling of fusing them into one dispatch
    std::string dump_y;       // --dump-y FILE (with --chain): layer 0's exl3r4d chain y after the timed passes, raw bytes --
                              // two builds' files compared byte for byte (same seed: same routing and data)
    std::string wgstamps;     // --wgstamps FILE: an EXL3R4D_WGSTAMP build's workgroup timeline of the last layer, CSV
    int wgstamps_lib = 0;     // --wgstamps-lib I: whose layers that pass launches (0 the first library; with --gd, 1 the v5 row)
    int h_rot = 0;
    int cu_pct = 100;         // --cu-pct P: the timed stream may use only the first P% of the card's CUs (hipExtStreamCreateWithCUMask)
                              // -- a persistent launch sized to the whole card then cannot have every workgroup resident,
                              // as in the engine when other kernels hold slots; a launch that needs co-residency stalls
    int pooled = 0;
    bool chain = false;
    double skew = -1;         // --skew S: one popularity for every layer, log-normal spread S (see draw)
    int bits = 4;             // --bits K (R23): the EXL3 experts' K, 2..6 (libr4d's side is its w4 form whatever K is)
    int a_rot = 0;            // --a-rot 1 (with --chain; or --op gate_up: the GEMM alone over xr): the v4 decode form, moe_exl3_prep_gu -> gate_up(a_rot 1) -> down       // --chain: the routed-expert CHAIN a layer -- exl3r4d gate_up -> down; libr4d gate_up ->
                              // gated_had_quant_fp8 -> down (the served baseline block's three launches)           // --pooled P: P% of each class's experts (the least routed) read from mapped host memory            // --h-rot 1: interface v3 EPI 1 (gate_up writes h rotated for down; down reads it so) // --dump-stamps: print exl3r4d's EXL3R4D_TSTAMP phase stamps (a timing-only build)   // --rand-data: x random, suh / suh.1 random +-1, svh random (the default: constants)
};

// One routing draw: sorted rows and offsets over E experts, and how many rows each expert got.
struct Route {
    std::vector<int32_t> sorted, off;
    std::vector<int64_t> rows;
};
// --skew S (>= 0): ONE expert popularity for every layer, log-normal with spread S, so the least routed experts are
// the same in every draw -- the engine's picture, where residency ranks by heat and the pooled tier holds the cold
// tail. Default (< 0): every draw its own popularity at spread 0.8 (no expert stays cold across layers).
// --group G --group-mix R: an MTP verify pass is G consecutive tokens of ONE sequence, and those share experts. Each run
// of G tokens draws one Gumbel vector its tokens share; a token's score is logp + R x shared + (1 - R) x its own. R 0 is
// the independent draw; R 0.53 gives ~26 live experts at M = 4, top-10 of 512 (P4b's verify reading), ~146 at M = 32.
std::vector<double> g_logp;
int64_t g_group = 1;
double g_group_mix = 0;
Route draw(int64_t M, int64_t E, int64_t topk) {
    std::vector<double> logp((size_t)E);
    if (!g_logp.empty()) logp = g_logp;
    else for (auto& v : logp) v = 0.8 * std::sqrt(-2.0 * std::log(unif())) * std::cos(6.283185307 * unif());
    std::vector<std::vector<int32_t>> by((size_t)E);
    std::vector<std::pair<double, int>> sc((size_t)E);
    std::vector<double> shared((size_t)E, 0.0);
    for (int64_t t = 0; t < M; ++t) {
        if (g_group > 1 && t % g_group == 0)
            for (auto& v : shared) v = -std::log(-std::log(unif()));
        const double own = g_group > 1 ? 1 - g_group_mix : 1;
        for (int64_t e = 0; e < E; ++e)
            sc[(size_t)e] = { logp[(size_t)e] + g_group_mix * shared[(size_t)e] - own * std::log(-std::log(unif())), (int)e };
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
void* g_fu_h = nullptr; void* g_fu_y = nullptr; size_t g_fu_hb = 0, g_fu_yb = 0;   // --fused: the chain's h and y
void* g_gd_y = nullptr;                                                               // --gd: the v5 row's y
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
    void* instance = nullptr;                 // the row's init() result (the engine calls init once a resolved row)
    std::vector<struct Lib> chain;            // --chain: dependent rows run in this order a layer, on one stream
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
        else if (a == "--wgstamps") o.wgstamps = nx();
        else if (a == "--dump-y") o.dump_y = nx();
        else if (a == "--wgstamps-lib") o.wgstamps_lib = std::atoi(nx());
        else if (a == "--overlap") o.overlap = true;
        else if (a == "--skip-prep") o.skip_prep = true;
        else if (a == "--fused") o.fused = std::atoi(nx());
        else if (a == "--h-rot") o.h_rot = std::atoi(nx());
        else if (a == "--pooled") o.pooled = std::atoi(nx());
        else if (a == "--chain") o.chain = true;
        else if (a == "--gd") o.gd = true;
        else if (a == "--cu-pct") o.cu_pct = std::atoi(nx());
        else if (a == "--host-us") o.host_us = true;
        else if (a == "--skew") o.skew = std::atof(nx());
        else if (a == "--group") g_group = std::atoll(nx());
        else if (a == "--group-mix") g_group_mix = std::atof(nx());
        else if (a == "--bits") o.bits = std::atoi(nx());
        else if (a == "--a-rot") o.a_rot = std::atoi(nx());
        else if (a == "--arena") g_arena_bytes = (size_t)std::atof(nx()) * (1ull << 30);
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
    if (o.bits < 2 || o.bits > 6) die("--bits is 2..6");
    if (g_group < 1 || g_group_mix < 0 || g_group_mix > 1) die("--group is >= 1, --group-mix 0..1");
    char kdt[16];
    std::snprintf(kdt, sizeof kdt, "exl3k%d", o.bits);
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

    if (o.skew >= 0) {
        g_logp.resize((size_t)o.E);
        for (auto& v : g_logp) v = o.skew * std::sqrt(-2.0 * std::log(unif())) * std::cos(6.283185307 * unif());
    }
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
    if (o.chain) {
        // ------------------------------------------------------------ THE CHAIN (--chain)
        // One layer's routed experts as the blocks issue them, on the same routing, dependent launches on one stream:
        //   exl3r4d (rad_block_moe_exl3.h:1475-1478): gate_up (its prep inside the op) -> down, both with h_rot.
        //   libr4d, the served baseline (rad_block_moe_fp8.h:1574-1605 with expert_rot and the rotated pair written by
        //   the hyper-connection read, qwen4exp_fp8.cpp:1197-1216, so no had_quant_act_fp8): moe_gemm_w4a8 gate_up ->
        //   gated_had_quant_fp8 (silu(gate) * up, rotated, E4M3 + scales) -> moe_gemm_w4a8 down.
        const int64_t wmax = std::max(wq[0], wq[1]);
        const int64_t Rg[2] = { 2 * wq[0], 2 * wq[1] }, Cg[2] = { o.hidden, o.hidden };   // gate_up [rows, cols]
        const int64_t Rd[2] = { o.hidden, o.hidden }, Cd[2] = { wq[0], wq[1] };            // down
        std::vector<void*> rs_sorted, rs_off;
        for (int l = 0; l < o.layers; ++l) {
            const Route& rt = route[(size_t)l];
            void* ds = dalloc((size_t)Tr * 4);
            void* doff = dalloc((size_t)(o.E + 1) * 4);
            put(ds, rt.sorted.data(), (size_t)Tr * 4);
            put(doff, rt.off.data(), (size_t)(o.E + 1) * 4);
            rs_sorted.push_back(ds);
            rs_off.push_back(doff);
        }
        auto layer_bytes = [&](int l, const int64_t* R_, const int64_t* C_, bool exl3, bool g) {
            double b = 0;
            for (int c = 0; c < 2; ++c)
                for (int64_t e = c; e < o.E; e += 2)
                    if (route[(size_t)l].rows[(size_t)e])
                        b += exl3 ? (double)(R_[c] * C_[c] * o.bits / 8) + 2.0 * ((g ? 2 : 1) * C_[c] + R_[c])
                                  : (double)(R_[c] * C_[c] / 2 + R_[c] * 2 * C_[c] / 128);
            return b;
        };
        {   // exl3r4d
            Lib E{ "exl3r4d chain", nullptr, {}, {}, {} };
            Lib G{ "exl3r4d gate_up", row_of(o.exl3.c_str(), "exl3r4d_moe_gate_up_m16"), {}, {}, {} };
            Lib D{ "exl3r4d down", row_of(o.exl3.c_str(), "exl3r4d_moe_down_m16"), {}, {}, {} };
            G.p = { PI("M", o.M), PI("N", Rg[0]), PI("K", Cg[0]), PI("n_expert", o.E), PI("top_k", o.topk),
                    PS("dtype", kdt), PS("codebook", "mul1"), PI("N_odd", Rg[1]), PI("K_odd", Cg[1]),
                    PI("h_rot", o.h_rot), PI("a_rot", o.a_rot) };
            Lib P{ "exl3r4d prep_gu", o.a_rot ? row_of(o.exl3.c_str(), "exl3r4d_moe_prep_gu_m16") : nullptr, {}, {}, {} };
            P.p = { PI("M", o.M), PI("K", Cg[0]), PI("n_expert", o.E), PI("top_k", o.topk), PS("dtype", kdt),
                    PI("K_odd", Cg[1]) };
            void* xr = o.a_rot ? dalloc((size_t)(Tr * 2 * o.hidden * 2)) : nullptr;   // v4: prep_gu's xr
            D.p = { PI("M", o.M), PI("N", Rd[0]), PI("K", Cd[0]), PI("n_expert", o.E), PI("top_k", o.topk),
                    PS("dtype", kdt), PS("codebook", "mul1"), PI("N_odd", Rd[1]), PI("K_odd", Cd[1]),
                    PI("h_rot", o.h_rot) };
            void* x = o.rand_data ? f16_rand((size_t)(o.M * o.hidden), 1.0f, 0) : f16_buf((size_t)(o.M * o.hidden), 0.5f);
            void* h = dalloc((size_t)(Tr * wmax * 2));
            void* y = dalloc((size_t)(Tr * o.hidden * 2));
            g_fu_h = h; g_fu_y = y; g_fu_hb = (size_t)(Tr * wmax * 2); g_fu_yb = (size_t)(Tr * o.hidden * 2);
            auto f16t = [&](size_t n, float v, int kind) { return o.rand_data ? f16_rand(n, kind == 1 ? 1.0f : v, kind)
                                                                              : f16_buf(n, v); };
            std::vector<std::array<std::array<RadTensor, 5>, 2>> pg((size_t)o.pools);
            std::vector<std::array<std::array<RadTensor, 3>, 2>> pd((size_t)o.pools);
            for (int q = 0; q < o.pools; ++q)
                for (int c = 0; c < 2; ++c) {
                    const size_t cbg = (size_t)(Rg[c] * Cg[c] * o.bits / 8), cbd = (size_t)(Rd[c] * Cd[c] * o.bits / 8);
                    pg[(size_t)q][c][0] = TAB(table_of(random_buf(cbg * (size_t)nec[c]), nec[c], cbg, c), kDtTrellis, { nec[c], Rg[c], Cg[c] });
                    pg[(size_t)q][c][1] = TAB(table_of(f16t((size_t)(Cg[c] * nec[c]), 0.01f, 1), nec[c], (size_t)Cg[c] * 2, c), RAD_F16, { nec[c], 1, Cg[c] });
                    pg[(size_t)q][c][2] = TAB(table_of(f16t((size_t)(Cg[c] * nec[c]), 0.01f, 1), nec[c], (size_t)Cg[c] * 2, c), RAD_F16, { nec[c], 1, Cg[c] });
                    pg[(size_t)q][c][3] = TAB(table_of(f16t((size_t)(Rg[c] * nec[c]), 1.0f, 2), nec[c], (size_t)Rg[c] * 2, c), RAD_F16, { nec[c], Rg[c], 1 });
                    void* dsuh = f16t((size_t)(Cd[c] * nec[c]), 0.01f, 1);   // DOWN's suh: gate_up's dn_suh and down's suh
                    pg[(size_t)q][c][4] = TAB(table_of(dsuh, nec[c], (size_t)Cd[c] * 2, c), RAD_F16, { nec[c], 1, Cd[c] });
                    pd[(size_t)q][c][0] = TAB(table_of(random_buf(cbd * (size_t)nec[c]), nec[c], cbd, c), kDtTrellis, { nec[c], Rd[c], Cd[c] });
                    pd[(size_t)q][c][1] = TAB(table_of(dsuh, nec[c], (size_t)Cd[c] * 2, c), RAD_F16, { nec[c], 1, Cd[c] });
                    pd[(size_t)q][c][2] = TAB(table_of(f16t((size_t)(Rd[c] * nec[c]), 1.0f, 2), nec[c], (size_t)Rd[c] * 2, c), RAD_F16, { nec[c], Rd[c], 1 });
                }
            for (int l = 0; l < o.layers; ++l) {
                const auto& a = pg[(size_t)(l % o.pools)];
                const auto& b = pd[(size_t)(l % o.pools)];
                const RadTensor st = T(rs_sorted[(size_t)l], RAD_I32, { Tr }), ot = T(rs_off[(size_t)l], RAD_I32, { o.E + 1 });
                const RadTensor xin = o.a_rot ? T(xr, RAD_F16, { Tr, 2 * o.hidden }) : T(x, RAD_F16, { o.M, o.hidden });
                G.t.push_back({ xin, a[0][0], a[0][1], a[0][2], a[0][3], a[0][4], st, ot,
                                a[1][0], a[1][1], a[1][2], a[1][3], a[1][4], T(h, RAD_F16, { Tr, wmax }) });
                if (o.a_rot) {
                    P.t.push_back({ T(x, RAD_F16, { o.M, o.hidden }), a[0][1], a[0][2], st, ot, a[1][1], a[1][2],
                                    T(xr, RAD_F16, { Tr, 2 * o.hidden }) });
                    P.bytes.push_back(0);
                }
                D.t.push_back({ T(h, RAD_F16, { Tr, wmax }), b[0][0], b[0][1], b[0][2], st, ot, b[1][0], b[1][1], b[1][2],
                                T(y, RAD_BF16, { Tr, o.hidden }) });
                G.bytes.push_back(layer_bytes(l, Rg, Cg, true, true));
                D.bytes.push_back(layer_bytes(l, Rd, Cd, true, false));
                E.bytes.push_back(G.bytes.back() + D.bytes.back());
            }
            if (o.a_rot) E.chain = { P, G, D };
            else E.chain = { G, D };
            libs.push_back(E);
            if (o.gd) {   // v5: the same tables, routing and x as one op, its own y
                if (o.a_rot || !o.h_rot) die("--gd needs --chain --h-rot 1 (a_rot 0)");
                Lib F{ "exl3r4d gd", row_of(o.exl3.c_str(), "exl3r4d_moe_gate_up_down_m16"), {}, {}, {} };
                F.p = { PI("M", o.M), PI("N", Rg[0]), PI("K", Cg[0]), PI("n_expert", o.E), PI("top_k", o.topk),
                        PS("dtype", kdt), PS("codebook", "mul1"), PI("N_odd", Rg[1]), PI("K_odd", Cg[1]) };
                g_gd_y = dalloc((size_t)(Tr * o.hidden * 2));
                for (int l = 0; l < o.layers; ++l) {
                    const auto& a = pg[(size_t)(l % o.pools)];
                    const auto& b = pd[(size_t)(l % o.pools)];
                    const RadTensor st = T(rs_sorted[(size_t)l], RAD_I32, { Tr }), ot = T(rs_off[(size_t)l], RAD_I32, { o.E + 1 });
                    F.t.push_back({ T(x, RAD_F16, { o.M, o.hidden }), a[0][0], a[0][1], a[0][2], a[0][3], a[0][4], b[0][0],
                                    b[0][2], st, ot, a[1][0], a[1][1], a[1][2], a[1][3], a[1][4], b[1][0], b[1][2],
                                    T(g_gd_y, RAD_BF16, { Tr, o.hidden }) });
                    F.bytes.push_back(E.bytes[(size_t)l]);
                }
                libs.push_back(F);
            }
        }
        if (!o.r4d.empty()) {   // libr4d, the served baseline's chain
            Lib E{ "libr4d chain", nullptr, {}, {}, {} };
            Lib G{ "libr4d gate_up", row_of(o.r4d.c_str(), "moe_gemm_w4a8"), {}, {}, {} };
            Lib Q{ "libr4d gq", row_of(o.r4d.c_str(), "gated_had_quant_fp8"), {}, {}, {} };
            Lib D{ "libr4d down", row_of(o.r4d.c_str(), "moe_gemm_w4a8"), {}, {}, {} };
            G.p = { PI("M", o.M), PI("N", Rg[0]), PI("K", Cg[0]), PI("N_odd", Rg[1]), PI("K_odd", Cg[1]), PI("parts", 2),
                    PI("n_expert", o.E), PI("top_k", o.topk), PI("group", 128), PS("a_order", "token"),
                    PS("dtype", "w4nl64a8h") };
            Q.p = { PI("M", Tr), PI("n", wmax), PI("group", 128), PS("dtype", "bf16"), PS("act", "silu") };
            D.p = { PI("M", o.M), PI("N", Rd[0]), PI("K", Cd[0]), PI("N_odd", Rd[1]), PI("K_odd", Cd[1]), PI("parts", 1),
                    PI("n_expert", o.E), PI("top_k", o.topk), PI("group", 128), PS("a_order", "sorted"),
                    PS("dtype", "w4nl64a8h") };
            void* a = random_buf((size_t)(o.M * o.hidden), 8, 111);
            std::vector<float> asc((size_t)(o.M * o.hidden / 128), 0.013f);
            void* as = dalloc(asc.size() * 4);
            put(as, asc.data(), asc.size() * 4);
            void* egu = dalloc((size_t)(Tr * 2 * wmax * 2));
            void* eq = dalloc((size_t)(Tr * wmax));
            void* es = dalloc((size_t)(Tr * (wmax / 128) * 4));
            void* y = dalloc((size_t)(Tr * o.hidden * 2));
            std::vector<std::array<std::array<RadTensor, 2>, 2>> pg((size_t)o.pools), pd((size_t)o.pools);
            for (int q = 0; q < o.pools; ++q)
                for (int c = 0; c < 2; ++c) {
                    const size_t cbg = (size_t)(Rg[c] * Cg[c] / 2), sbg = (size_t)(Rg[c] * 2 * Cg[c] / 128);
                    const size_t cbd = (size_t)(Rd[c] * Cd[c] / 2), sbd = (size_t)(Rd[c] * 2 * Cd[c] / 128);
                    pg[(size_t)q][c][0] = TAB(table_of(random_buf(cbg * (size_t)nec[c]), nec[c], cbg, c), kDtMoeW4, { nec[c], Rg[c], Cg[c] });
                    pg[(size_t)q][c][1] = TAB(table_of(random_buf(sbg * (size_t)nec[c], 0x30, 8), nec[c], sbg, c), RAD_F8E4M3, { nec[c], Rg[c], 2 * Cg[c] / 128 });
                    pd[(size_t)q][c][0] = TAB(table_of(random_buf(cbd * (size_t)nec[c]), nec[c], cbd, c), kDtMoeW4, { nec[c], Rd[c], Cd[c] });
                    pd[(size_t)q][c][1] = TAB(table_of(random_buf(sbd * (size_t)nec[c], 0x30, 8), nec[c], sbd, c), RAD_F8E4M3, { nec[c], Rd[c], 2 * Cd[c] / 128 });
                }
            for (int l = 0; l < o.layers; ++l) {
                const auto& g = pg[(size_t)(l % o.pools)];
                const auto& d = pd[(size_t)(l % o.pools)];
                const RadTensor st = T(rs_sorted[(size_t)l], RAD_I32, { Tr }), ot = T(rs_off[(size_t)l], RAD_I32, { o.E + 1 });
                G.t.push_back({ T(a, RAD_F8E4M3, { o.M, o.hidden }), T(as, RAD_F32, { o.M, o.hidden / 128 }), g[0][0], g[0][1],
                                st, ot, g[1][0], g[1][1], T(egu, RAD_BF16, { Tr, 2 * wmax }) });
                Q.t.push_back({ T(egu, RAD_BF16, { Tr, 2 * wmax }), T(eq, RAD_F8E4M3, { Tr, wmax }),
                                T(es, RAD_F32, { Tr, wmax / 128 }), RadTensor{} });
                D.t.push_back({ T(eq, RAD_F8E4M3, { Tr, wmax }), T(es, RAD_F32, { Tr, wmax / 128 }), d[0][0], d[0][1],
                                st, ot, d[1][0], d[1][1], T(y, RAD_BF16, { Tr, o.hidden }) });
                G.bytes.push_back(layer_bytes(l, Rg, Cg, false, true));
                D.bytes.push_back(layer_bytes(l, Rd, Cd, false, false));
                Q.bytes.push_back(0);
                E.bytes.push_back(G.bytes.back() + D.bytes.back());
            }
            E.chain = { G, Q, D };
            libs.push_back(E);
        }
    } else {
    // ---------------------------------------------------------------- exl3r4d
    {
        const std::string rn = !o.exl3_row.empty() ? o.exl3_row : gu ? "exl3r4d_moe_gate_up_m16" : "exl3r4d_moe_down_m16";
        Lib L{ "exl3r4d", row_of(o.exl3.c_str(), rn.c_str()), {}, {}, {} };
        if (std::strcmp(L.row->op, gu ? "moe_exl3_gate_up" : "moe_exl3_down") != 0)
            die("--exl3-row " + rn + " implements " + L.row->op + ", not the --op asked for");
        L.p = { PI("M", o.M), PI("N", R[0]), PI("K", C[0]), PI("n_expert", o.E), PI("top_k", o.topk),
                PS("dtype", kdt), PS("codebook", "mul1"), PI("N_odd", R[1]), PI("K_odd", C[1]),
                PI("h_rot", o.h_rot) };
        // --op gate_up --a-rot 1: the GEMM alone over an already rotated xr [T][2 K] (prep_gu's output; here constants),
        // without the prep kernel in front -- the decode band's gate_up dispatch as the arch issues it
        const bool arot = gu && o.a_rot;
        if (gu) L.p.push_back(PI("a_rot", arot ? 1 : 0));
        const int64_t xrows = arot ? Tr : Ma, xw = arot ? 2 * xcols : xcols;
        void* x = o.rand_data ? f16_rand((size_t)(xrows * xw), 1.0f, 0) : f16_buf((size_t)(xrows * xw), 0.5f);
        void* y = dalloc((size_t)(Tr * ycols * 2));
        L.y = y;
        std::vector<std::array<std::array<RadTensor, 5>, 2>> pool((size_t)o.pools);
        for (auto& pc : pool)
            for (int c = 0; c < 2; ++c) {
                const size_t cb = (size_t)(R[c] * C[c] * o.bits / 8);
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
                    if (rt.rows[(size_t)e]) bytes += (double)(R[c] * C[c] * o.bits / 8) + 2.0 * ((gu ? 2 : 1) * C[c] + R[c]);
            L.bytes.push_back(bytes);
            const RadTensor xt = T(x, RAD_F16, { xrows, xw }), yt = T(y, gu ? RAD_F16 : RAD_BF16, { Tr, ycols });
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
    }   // single op

    hipStream_t stream;
    if (o.cu_pct >= 100) ck(hipStreamCreate(&stream), "stream");
    else {
        int dev = 0, ncu = 0;
        ck(hipGetDevice(&dev), "device");
        ck(hipDeviceGetAttribute(&ncu, hipDeviceAttributeMultiprocessorCount, dev), "CU count");
        const int use = std::max(2, (ncu * std::max(o.cu_pct, 1) / 100) & ~1);   // whole WGPs (CU pairs)
        std::vector<uint32_t> mask((size_t)(ncu + 31) / 32, 0u);
        for (int c = 0; c < use; ++c) mask[(size_t)c / 32] |= 1u << (c % 32);
        ck(hipExtStreamCreateWithCUMask(&stream, (uint32_t)mask.size(), mask.data()), "stream (CU mask)");
        std::printf("perfmoe: timed stream limited to %d of %d CUs (--cu-pct %d)\n", use, ncu, o.cu_pct);
    }
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
    // Set up one row as the engine does: its init (once), its scratch (once, from its own hook).
    // --fused: the prototype entry and its counter block (zeroed once; the kernel's last workgroup re-zeroes it)
    typedef int (*FuProto)(const RadArgs*, const RadArgs*, unsigned*, int, RadStream);
    FuProto fu_proto = nullptr;
    unsigned* fu_ctr = nullptr;
    if (o.fused) {
        if (!o.chain || !o.h_rot) die("--fused needs --chain --h-rot 1");
        void* hl = dlopen(o.exl3.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
        fu_proto = hl ? (FuProto)dlsym(hl, "exl3r4d_fused_proto") : nullptr;
        if (!fu_proto) die("--fused needs an EXL3R4D_FUSED_PROTO build of exl3r4d");
        ck(hipMalloc((void**)&fu_ctr, (2 + 256 + 2 + 256 + 2 + 256) * sizeof(unsigned)), "hipMalloc (fused counters)");
        ck(hipMemset(fu_ctr, 0, (2 + 256 + 2 + 256 + 2 + 256) * sizeof(unsigned)), "memset");
    }
    {   // reading builds with EXL3R4D_PDN: the persistent down kernel's capacity, as the launcher computes it
        void* hl = dlopen(o.exl3.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
        auto dc = hl ? (int (*)(int*, int*, int*))dlsym(hl, "exl3r4d_debug_capacity") : nullptr;
        int regs = 0, shared = 0, nwgp = 0;
        if (dc) std::printf("perfmoe: pdn capacity %d (numRegs %d, sharedSizeBytes %d, multiprocessors %d)\n",
                            dc(&regs, &shared, &nwgp), regs, shared, nwgp);
    }
    hipStream_t ov_st[3];
    hipEvent_t ov_ev[4];
    if (o.overlap) {
        if (!o.chain) die("--overlap needs --chain");
        for (auto& x : ov_st) ck(hipStreamCreateWithFlags(&x, hipStreamNonBlocking), "stream");
        for (auto& x : ov_ev) ck(hipEventCreateWithFlags(&x, hipEventDisableTiming), "event");
    }
    auto setup = [&](Lib& L) {
        if (L.row->init) {
            const int rc = L.row->init(L.p.data(), (int)L.p.size(), 0, 1, &L.instance);
            if (rc != RAD_OK) { std::printf("perfmoe: %s init failed rc=%d\n", L.name, rc); std::exit(1); }
        }
        RadArgs a = args_of(L, 0);
        a.instance = L.instance;
        L.scratch_bytes = L.row->scratch ? L.row->scratch(&a) : 0;
        if (L.scratch_bytes > 0) L.scratch = dalloc((size_t)L.scratch_bytes);
    };
    // One layer of a library: its row, or its chain's rows in order, dependent launches on one stream.
    std::vector<double> host_ns(8, 0.0);   // --host-us: per library index, ns inside launch()
    size_t cur_lib = 0;
    auto timed = [&](const RadKernelInfo* row, const RadArgs* a) -> int {
        if (!o.host_us) return row->launch(a, (RadStream)stream);
        const auto t0 = std::chrono::steady_clock::now();
        const int rc = row->launch(a, (RadStream)stream);
        host_ns[cur_lib] += (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - t0).count();
        return rc;
    };
    auto launch_layer = [&](Lib& L, int l) -> int {
        if (L.chain.empty()) {
            RadArgs a = args_of(L, l);
            a.instance = L.instance;
            return timed(L.row, &a);
        }
        if (o.fused && L.chain.size() == 2 && fu_proto) {   // a_rot 0: prep + gate_up + down as one launch
            RadArgs ag = args_of(L.chain[0], l), ad = args_of(L.chain[1], l);
            ag.scratch = L.chain[0].scratch;
            ag.scratch_bytes = L.chain[0].scratch_bytes;
            const int rc = fu_proto(&ag, &ad, fu_ctr, o.fused, (RadStream)stream);
            if (rc != RAD_OK) std::printf("perfmoe: fused launch failed rc=%d\n", rc);
            return rc;
        }
        if (o.fused && L.chain.size() == 3 && fu_proto) {   // prep_gu, then gate_up + down as one launch
            RadArgs a0 = args_of(L.chain[0], l);
            a0.instance = L.chain[0].instance;
            int rc = o.skip_prep ? RAD_OK : L.chain[0].row->launch(&a0, (RadStream)stream);
            RadArgs ag = args_of(L.chain[1], l), ad = args_of(L.chain[2], l);
            if (rc == RAD_OK) rc = fu_proto(&ag, &ad, fu_ctr, o.fused, (RadStream)stream);
            if (rc != RAD_OK) std::printf("perfmoe: fused launch failed rc=%d\n", rc);
            return rc;
        }
        if (o.overlap) {   // every row of the chain on its own stream, all starting after the previous layer's join
            ck(hipEventRecord(ov_ev[0], stream), "record");
            for (size_t i = 0; i < L.chain.size(); ++i) {
                hipStream_t si = i == 0 ? stream : ov_st[i - 1];
                if (i) ck(hipStreamWaitEvent(si, ov_ev[0], 0), "wait");
                RadArgs a = args_of(L.chain[i], l);
                a.instance = L.chain[i].instance;
                const int rc = L.chain[i].row->launch(&a, (RadStream)si);
                if (rc != RAD_OK) { std::printf("perfmoe: %s launch failed rc=%d\n", L.chain[i].name, rc); return rc; }
                if (i) { ck(hipEventRecord(ov_ev[i], si), "record"); ck(hipStreamWaitEvent(stream, ov_ev[i], 0), "wait"); }
            }
            return RAD_OK;
        }
        for (Lib& S : L.chain) {
            if (o.skip_prep && L.chain.size() == 3 && &S == &L.chain[0]) continue;
            RadArgs a = args_of(S, l);
            a.instance = S.instance;
            const int rc = timed(S.row, &a);
            if (rc != RAD_OK) { std::printf("perfmoe: %s launch failed rc=%d\n", S.name, rc); return rc; }
        }
        return RAD_OK;
    };
    for (Lib& L : libs) {
        if (L.chain.empty()) setup(L);
        for (Lib& S : L.chain) setup(S);
        for (int l = 0; l < o.layers; ++l) {   // warm: every layer once, and every launch must succeed
            const int rc = launch_layer(L, l);
            if (rc != RAD_OK) { std::printf("perfmoe: %s launch failed rc=%d\n", L.name, rc); return 1; }
        }
        ck(hipStreamSynchronize(stream), "warm-up");
    }
    if (o.fused) {   // the check: layer 0 serial, then fused, h and y byte for byte
        Lib& L = libs[0];
        std::vector<char> h0(g_fu_hb), y0(g_fu_yb), h1(g_fu_hb), y1(g_fu_yb);
        const int keep = o.fused;
        o.fused = 0;
        (void)launch_layer(L, 0);
        ck(hipStreamSynchronize(stream), "check (serial)");
        ck(hipMemcpy(h0.data(), g_fu_h, g_fu_hb, hipMemcpyDeviceToHost), "copy");
        ck(hipMemcpy(y0.data(), g_fu_y, g_fu_yb, hipMemcpyDeviceToHost), "copy");
        ck(hipMemset(g_fu_h, 0x5a, g_fu_hb), "poison");
        ck(hipMemset(g_fu_y, 0x5a, g_fu_yb), "poison");
        o.fused = keep;
        (void)launch_layer(L, 0);
        ck(hipStreamSynchronize(stream), "check (fused)");
        ck(hipMemcpy(h1.data(), g_fu_h, g_fu_hb, hipMemcpyDeviceToHost), "copy");
        ck(hipMemcpy(y1.data(), g_fu_y, g_fu_yb, hipMemcpyDeviceToHost), "copy");
        size_t dh = 0, dy = 0;
        for (size_t i = 0; i < g_fu_hb; ++i) dh += h0[i] != h1[i];
        for (size_t i = 0; i < g_fu_yb; ++i) dy += y0[i] != y1[i];
        unsigned err = 0, left[2] = {};
        ck(hipMemcpy(&err, fu_ctr + 2 + 256, 4, hipMemcpyDeviceToHost), "copy");
        ck(hipMemcpy(left, fu_ctr, 8, hipMemcpyDeviceToHost), "copy");
        std::printf("FUSED-CHECK y %zu of %zu bytes differ; h %zu of %zu bytes differ (h past the classes' widths is "
                    "not written by the fused form); give-up flag %u; counters after %u %u\n", dy, g_fu_yb, dh,
                    g_fu_hb, err, left[0], left[1]);
        if (dy || err || left[0] || left[1]) { std::printf("perfmoe: FUSED-CHECK FAILED\n"); return 1; }
    }
    if (o.gd) {   // the check: layer 0 through the serial chain and through the v5 row, y byte for byte
        std::vector<char> y0(g_fu_yb), y1(g_fu_yb);
        ck(hipMemset(g_fu_y, 0x5a, g_fu_yb), "poison");
        ck(hipMemset(g_gd_y, 0xa5, g_fu_yb), "poison");
        (void)launch_layer(libs[0], 0);
        (void)launch_layer(libs[1], 0);
        ck(hipStreamSynchronize(stream), "check (gd)");
        ck(hipMemcpy(y0.data(), g_fu_y, g_fu_yb, hipMemcpyDeviceToHost), "copy");
        ck(hipMemcpy(y1.data(), g_gd_y, g_fu_yb, hipMemcpyDeviceToHost), "copy");
        size_t dy = 0;
        for (size_t i = 0; i < g_fu_yb; ++i) dy += y0[i] != y1[i];
        std::printf("GD-CHECK y %zu of %zu bytes differ (the v5 row against the serial chain)\n", dy, g_fu_yb);
        if (dy) { std::printf("perfmoe: GD-CHECK FAILED\n"); return 1; }
    }
    // WARM-CHECK (--chain): layer 0's y from the exl3r4d chain now -- its first launches -- and again after the timed
    // passes, byte for byte; with --gd the v5 row against the serial chain again too. A row that keeps state across
    // launches (exl3r4d's SUHC: the prep's resident suh, filled as it meets each expert) must give the bits it gave
    // cold.
    std::vector<char> wy0;
    if (o.chain && g_fu_y) {
        wy0.resize(g_fu_yb);
        (void)launch_layer(libs[0], 0);
        ck(hipStreamSynchronize(stream), "warm check (cold)");
        ck(hipMemcpy(wy0.data(), g_fu_y, g_fu_yb, hipMemcpyDeviceToHost), "copy");
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
                for (int l = 0; l < o.layers; ++l) (void)launch_layer(L, l);
            ck(hipEventRecord(e1, stream), "record");
            ck(hipEventSynchronize(e1), "sync");
            ck(hipEventElapsedTime(&wms, e0, e1), "elapsed");
        }
        std::printf("perfmoe: warmed %.0f ms\n", wms);
    }
    std::fill(host_ns.begin(), host_ns.end(), 0.0);
    for (int r = 0; r < o.reps; ++r)
        for (Lib& L : libs) {
            cur_lib = (size_t)(&L - libs.data());
            ck(hipEventRecord(e0, stream), "record");
            for (int it = 0; it < o.iters; ++it)
                for (int l = 0; l < o.layers; ++l) (void)launch_layer(L, l);
            ck(hipEventRecord(e1, stream), "record");
            ck(hipEventSynchronize(e1), "sync");
            float ms = 0;
            ck(hipEventElapsedTime(&ms, e0, e1), "elapsed");
            L.us.push_back((double)ms * 1000.0 / (o.iters * o.layers));
        }
    if (!o.wgstamps.empty()) {   // one more pass over the layers, the stamps cleared before the last layer's launches
        void* h = dlopen(o.exl3.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
        auto clr = h ? (int (*)(void*))dlsym(h, "exl3r4d_wgstamp_clear") : nullptr;
        auto rd = h ? (int (*)(int, unsigned long long*, int))dlsym(h, "exl3r4d_wgstamp_read") : nullptr;
        if (!clr || !rd) die("--wgstamps needs an EXL3R4D_WGSTAMP build of exl3r4d");
        if (o.wgstamps_lib < 0 || o.wgstamps_lib >= (int)libs.size()) die("--wgstamps-lib names no library");
        Lib& L = libs[(size_t)o.wgstamps_lib];
        for (int l = 0; l < o.layers; ++l) {
            if (l == o.layers - 1 && clr(stream) != RAD_OK) die("wgstamp clear");
            (void)launch_layer(L, l);
        }
        ck(hipStreamSynchronize(stream), "wgstamps");
        FILE* f = std::fopen(o.wgstamps.c_str(), "w");
        if (!f) die("cannot write " + o.wgstamps);
        std::fprintf(f, "op,wg,entry,exit,hw\n");
        constexpr int n = 8192;
        std::vector<unsigned long long> buf(3 * n);
        for (int op = 1; op >= 0; --op) {
            if (rd(op, buf.data(), n) != RAD_OK) die("wgstamp read");
            for (int i = 0; i < n; ++i)
                if (buf[3 * i]) std::fprintf(f, "%s,%d,%llu,%llu,%llu\n", op ? "gate_up" : "down", i, buf[3 * i],
                                             buf[3 * i + 1], buf[3 * i + 2]);
        }
        std::fclose(f);
    }
    for (Lib& L : libs) {
        double b = 0;
        for (double x : L.bytes) b += x;
        b /= (double)L.bytes.size();
        std::vector<double> u = L.us;
        std::sort(u.begin(), u.end());
        const double med = u[u.size() / 2];
        std::printf("RESULT %-17s %s tp%d M=%-3lld  %8.2f us [%.2f, %.2f]  %7.1f GB/s [%.1f, %.1f]  %.2f MB/dispatch",
                    L.name, o.chain ? "chain" : o.op.c_str(), o.tp, (long long)o.M, med, u.front(), u.back(), b / (med * 1e3),
                    b / (u.back() * 1e3), b / (u.front() * 1e3), b / 1e6);
        if (o.ceiling > 0) std::printf("  %.1f%% of %.1f", 100.0 * b / (med * 1e3) / o.ceiling, o.ceiling);
        if (o.host_us)
            std::printf("  host %.2f us/layer in launch()", host_ns[(size_t)(&L - libs.data())] / 1e3 / (o.reps * o.iters * o.layers));
        std::printf("\nREPS   %-17s %s tp%d M=%-3lld  GB/s", L.name, o.chain ? "chain" : o.op.c_str(), o.tp, (long long)o.M);
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
    if (!wy0.empty()) {
        std::vector<char> y1(g_fu_yb);
        ck(hipMemset(g_fu_y, 0x5a, g_fu_yb), "poison");
        (void)launch_layer(libs[0], 0);
        ck(hipStreamSynchronize(stream), "warm check (warm)");
        ck(hipMemcpy(y1.data(), g_fu_y, g_fu_yb, hipMemcpyDeviceToHost), "copy");
        size_t dy = 0;
        for (size_t i = 0; i < g_fu_yb; ++i) dy += wy0[i] != y1[i];
        if (!o.dump_y.empty()) {
            FILE* f = std::fopen(o.dump_y.c_str(), "wb");
            if (!f || std::fwrite(y1.data(), 1, y1.size(), f) != y1.size()) die("cannot write " + o.dump_y);
            std::fclose(f);
        }
        size_t dg = 0;
        if (o.gd && g_gd_y && libs.size() > 1) {   // the v5 row, warm, against the serial chain's (warm) y
            std::vector<char> g1(g_fu_yb);
            ck(hipMemset(g_gd_y, 0xa5, g_fu_yb), "poison");
            (void)launch_layer(libs[1], 0);
            ck(hipStreamSynchronize(stream), "warm check (gd)");
            ck(hipMemcpy(g1.data(), g_gd_y, g_fu_yb, hipMemcpyDeviceToHost), "copy");
            for (size_t i = 0; i < g_fu_yb; ++i) dg += y1[i] != g1[i];
        }
        std::printf("WARM-CHECK y %zu of %zu bytes differ (layer 0 after the timed passes vs its first launch)%s\n", dy,
                    g_fu_yb, o.gd ? (dg ? "; gd vs chain DIFFER" : "; gd vs chain 0 bytes differ") : "");
        if (dy || dg) { std::printf("perfmoe: WARM-CHECK FAILED\n"); return 1; }
    }
    return 0;
}
