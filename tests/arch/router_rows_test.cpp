/* router_rows_test.cpp -- router_rows.cpp's gate: each fused row equals libref's unfused chain, byte
 * for byte, called here one row at a time as the routed block issues them where the fusion does not
 * resolve (gemm_nt for the router and the shared gate, router_topk, moe_scatter over n_local).
 *
 * USAGE: router_rows_test <exl3test_router.so>   with RADIANCE_HOME naming a home that has
 *        kernels/libref.so (the test library finds libref the same way). EXIT 0 PASS, 1 FAIL,
 *        77 SKIP (no libref).
 *
 * Every output buffer is poisoned before each run, so a byte one side leaves untouched and the
 * other writes is a difference. Controls, each a wiring bug the comparison must see: (1) the
 * scatter run without the slice's expert_base on an expert-parallel slice; (2) the chain without
 * the shared gate, which leaves the gate buffer different from the fused op's.
 */
#include <rad_abi.h>
#include <rad_types.h>

#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failed = 0;

RadLaunchFn find_row(void* h, const char* name) {
    auto count = (int (*)(void))dlsym(h, "rad_kernel_count");
    auto at    = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    for (int i = 0; count && at && i < count(); ++i)
        if (!std::strcmp(at(i)->name, name)) return at(i)->launch;
    return nullptr;
}

struct Buf {
    std::vector<uint8_t> bytes;
    RadTensor t{};
    Buf(uint32_t dtype, std::vector<int64_t> shape) {
        int64_t n = 1;
        for (int64_t s : shape) n *= s;
        bytes.assign((size_t)(n * (dtype == RAD_I32 ? 4 : 2)), 0xA5);
        t.data = bytes.data(); t.dtype = dtype; t.rank = (uint32_t)shape.size();
        int64_t stride = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            t.shape[i] = shape[(size_t)i]; t.stride[i] = stride; stride *= shape[(size_t)i];
        }
    }
    void poison() { std::memset(bytes.data(), 0xA5, bytes.size()); }
};

uint64_t g_rng = 0x9E3779B97F4A7C15ull;
uint16_t rand_bf16(float scale) {
    g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
    const float f = ((float)(g_rng >> 40) / (float)(1u << 24) * 2.0f - 1.0f) * scale;
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);   /* round to nearest even */
}
void fill_bf16(Buf& b, float scale) {
    uint16_t* p = (uint16_t*)b.bytes.data();
    for (size_t i = 0; i < b.bytes.size() / 2; ++i) p[i] = rand_bf16(scale);
}

int launch(RadLaunchFn fn, std::vector<RadParam> p, std::vector<RadTensor> t) {
    RadArgs a{};
    a.t = t.data(); a.n_t = (int)t.size();
    a.p = p.data(); a.n_p = (int)p.size();
    a.world_size = 1;
    return fn(&a, nullptr);
}

struct Case {
    const char* name;
    int64_t M, E, n_local, base, top_k, K;
    int norm;
    const char* protect;   /* null: none */
    bool gemm, gate;
};

RadLaunchFn g_fused_ts, g_fused_gts, g_gemm, g_topk, g_scatter;

/* Run one case both ways; `mutate` breaks the chain the way a control asks. Returns whether every
 * output agreed byte for byte. */
bool agree(const Case& c, int mutate) {
    const int64_t T = c.M * c.top_k;
    Buf a(RAD_BF16, { c.M, c.K }), b(RAD_BF16, { c.E, c.K }), gb(RAD_BF16, { 1, c.K });
    fill_bf16(a, 1.0f); fill_bf16(b, 0.5f); fill_bf16(gb, 0.5f);
    Buf logits(RAD_BF16, { c.M, c.E });
    if (!c.gemm) fill_bf16(logits, 4.0f);
    const std::vector<uint8_t> logits_in = logits.bytes;
    struct Out { Buf ids, w, sorted, off, cnt, gate, lg; };
    auto make = [&]() {
        return Out{ Buf(RAD_I32, { c.M, c.top_k }), Buf(RAD_BF16, { c.M, c.top_k }),
                    Buf(RAD_I32, { T }), Buf(RAD_I32, { c.n_local + 1 }), Buf(RAD_I32, { c.n_local }),
                    Buf(RAD_BF16, { c.M, 1 }), Buf(RAD_BF16, { c.M, c.E }) };
    };
    Out f = make(), u = make();
    if (!c.gemm) { f.lg.bytes = logits_in; u.lg.bytes = logits_in; }

    std::vector<RadParam> fp = { RAD_INT("M", c.M), RAD_INT("n_expert", c.E),
                                 RAD_INT("n_local", c.n_local), RAD_INT("top_k", c.top_k),
                                 RAD_INT("norm", c.norm), RAD_STR("dtype", "bf16"),
                                 RAD_INT("expert_base", c.base) };
    if (c.gemm) fp.push_back(RAD_INT("K", c.K));
    if (c.protect) fp.push_back(RAD_STR("protect", c.protect));
    RadTensor none{};
    int s1 = c.gemm
        ? launch(g_fused_gts, fp, { a.t, b.t, c.gate ? gb.t : none, f.lg.t, f.ids.t, f.w.t, f.sorted.t,
                                    f.off.t, f.cnt.t, c.gate ? f.gate.t : none })
        : launch(g_fused_ts, fp, { f.lg.t, f.ids.t, f.w.t, f.sorted.t, f.off.t, f.cnt.t });

    /* the unfused chain, as the block issues it where the fusion does not resolve */
    int s2 = RAD_OK;
    if (c.gemm) {
        s2 = launch(g_gemm, { RAD_INT("M", c.M), RAD_INT("N", c.E), RAD_INT("K", c.K),
                              RAD_STR("dtype", "bf16") }, { a.t, b.t, u.lg.t });
        if (c.gate && mutate != 2 && s2 >= 0)
            s2 = launch(g_gemm, { RAD_INT("M", c.M), RAD_INT("N", 1), RAD_INT("K", c.K),
                                  RAD_STR("dtype", "bf16") }, { a.t, gb.t, u.gate.t });
    }
    if (s2 >= 0)
        s2 = launch(g_topk, { RAD_INT("M", c.M), RAD_INT("n_expert", c.E), RAD_INT("top_k", c.top_k),
                              RAD_INT("norm", c.norm), RAD_STR("dtype", "bf16") },
                    { u.lg.t, u.ids.t, u.w.t });
    std::vector<RadParam> sp = { RAD_INT("M", c.M), RAD_INT("n_expert", c.n_local),
                                 RAD_INT("top_k", c.top_k),
                                 RAD_INT("expert_base", mutate == 1 ? 0 : c.base) };
    if (c.protect) sp.push_back(RAD_STR("protect", c.protect));
    if (s2 >= 0) s2 = launch(g_scatter, sp, { u.ids.t, u.sorted.t, u.off.t, u.cnt.t });
    if (s1 < 0 || s2 < 0) {
        std::printf("  %s: launch failed (fused %d, chain %d)\n", c.name, s1, s2);
        return false;
    }
    bool same = true;
    const std::pair<const char*, std::pair<Buf*, Buf*>> outs[] = {
        { "logits", { &f.lg, &u.lg } }, { "expert_ids", { &f.ids, &u.ids } },
        { "expert_w", { &f.w, &u.w } }, { "sorted_tok", { &f.sorted, &u.sorted } },
        { "expert_offset", { &f.off, &u.off } }, { "expert_count", { &f.cnt, &u.cnt } },
        { "gate", { &f.gate, &u.gate } },
    };
    for (const auto& [nm, pr] : outs)
        if (pr.first->bytes != pr.second->bytes) {
            if (!mutate) std::printf("  %s: %s differs\n", c.name, nm);
            same = false;
        }
    return same;
}

}  /* namespace */

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s <exl3test_router.so>\n", argv[0]); return 2; }
    void* mine = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!mine) { std::printf("FAIL cannot load %s: %s\n", argv[1], dlerror()); return 1; }
    const char* home = std::getenv("RADIANCE_HOME");
    void* ref = nullptr;
    std::string list = home ? home : "";
    for (size_t at = 0; at <= list.size() && !ref;) {
        size_t colon = list.find(':', at);
        if (colon == std::string::npos) colon = list.size();
        ref = dlopen((list.substr(at, colon - at) + "/kernels/libref.so").c_str(), RTLD_NOW | RTLD_LOCAL);
        at = colon + 1;
    }
    if (!ref) { std::printf("SKIP: no kernels/libref.so on RADIANCE_HOME ('%s')\n", list.c_str()); return 77; }
    auto open = (int (*)(void))dlsym(mine, "rad_plugin_open");
    if (!open || open() != RAD_OK) { std::printf("FAIL the test library would not open\n"); return 1; }
    g_fused_ts  = find_row(mine, "exl3test_router_topk_scatter");
    g_fused_gts = find_row(mine, "exl3test_router_gemm_topk_scatter");
    g_gemm      = find_row(ref, "ref_gemm_nt");
    g_topk      = find_row(ref, "ref_router_topk");
    g_scatter   = find_row(ref, "ref_moe_scatter");
    if (!g_fused_ts || !g_fused_gts || !g_gemm || !g_topk || !g_scatter) {
        std::printf("FAIL a row is missing\n");
        return 1;
    }

    const Case cases[] = {
        { "decode, one token",              1,  16, 16, 0, 4,  64, 1, nullptr, false, false },
        { "decode, eight tokens, no norm",  8,  16, 16, 0, 4,  64, 0, nullptr, false, false },
        { "protected experts sorted last", 16,  16, 16, 0, 4,  64, 1, "3,5",   false, false },
        { "expert-parallel slice",         16,  16,  8, 8, 4,  64, 1, nullptr, false, false },
        { "the fused GEMM, no gate",       24,  32, 32, 0, 8, 256, 1, nullptr, true,  false },
        { "the fused GEMM and the gate",   64,  32, 32, 0, 8, 256, 1, "1,30",  true,  true  },
    };
    for (const Case& c : cases) {
        const bool ok = agree(c, 0);
        std::printf("%s %s\n", ok ? "PASS" : "FAIL", c.name);
        g_failed += !ok;
    }
    /* the controls: each must DIFFER */
    const bool c1 = agree(cases[3], 1), c2 = agree(cases[5], 2);
    std::printf("%s control: the scatter without the slice's expert_base differs\n", c1 ? "FAIL" : "PASS");
    std::printf("%s control: the chain without the gate differs\n", c2 ? "FAIL" : "PASS");
    g_failed += c1 + c2;
    std::printf("RESULT: %s\n", g_failed ? "FAIL" : "PASS");
    return g_failed ? 1 : 0;
}
