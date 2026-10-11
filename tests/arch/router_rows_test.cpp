/* router_rows_test.cpp -- router_rows.cpp's gate: each fused row equals libref's unfused chain, byte
 * for byte, called here one row at a time as the blocks issue them where the fusion does not
 * resolve: the routed block's gemm_nt (router, shared gate), router_topk and moe_scatter over
 * n_local; the gated attention's rmsnorm (q out of [q|gate], k in place), rope ("mrope", one
 * position component) on each, and kv_store -- the rope table built by libref's own rope_table.
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
    /* The M PARAMETER the fused row is handed, when not M: the engine collapses a band's range to
     * its top, so a 5-token pass in the M <= 64 band arrives with M = 64 and 5-row operands. */
    int64_t band = 0;
};

RadLaunchFn g_fused_ts, g_fused_gts, g_gemm, g_topk, g_scatter;
RadLaunchFn g_fused_qk, g_fused_qkkv, g_rmsnorm, g_rope, g_rope_table, g_kv_store;

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
    if (!c.gemm) {   /* into the storage each tensor already points at */
        std::memcpy(f.lg.bytes.data(), logits_in.data(), logits_in.size());
        std::memcpy(u.lg.bytes.data(), logits_in.data(), logits_in.size());
    }

    std::vector<RadParam> fp = { RAD_INT("M", c.band ? c.band : c.M), RAD_INT("n_expert", c.E),
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

/* ---- the attention prologue ---- */
struct QkCase {
    const char* name;
    int64_t T, nh, nkv, hd, rot;
    bool kv, gate_out;
    int64_t band = 0;   /* the M the fused row is handed, when not T (see Case::band) */
};

/* `mutate`: 1 the chain's rope at another theta than the table's; 2 the chain without the k norm. */
bool qk_agree(const QkCase& c, int mutate) {
    const double theta = 1.0e7, eps = 1e-6, wadd = 1.0;
    const int64_t table_rows = 128, block = 4, n_blocks = 64;
    Buf qg(RAD_BF16, { c.T, c.nh, 2 * c.hd }), k0(RAD_BF16, { c.T, c.nkv * c.hd });
    Buf v(RAD_BF16, { c.T, c.nkv * c.hd }), qw(RAD_BF16, { c.hd }), kw(RAD_BF16, { c.hd });
    fill_bf16(qg, 3.0f); fill_bf16(k0, 3.0f); fill_bf16(v, 1.0f); fill_bf16(qw, 0.3f); fill_bf16(kw, 0.3f);
    Buf pos(RAD_I32, { c.T }), slots(RAD_I32, { c.T });
    for (int64_t t = 0; t < c.T; ++t) {
        ((int32_t*)pos.bytes.data())[t] = (int32_t)((t * 37 + 5) % table_rows);
        ((int32_t*)slots.bytes.data())[t] = (int32_t)((t * 13 + 3) % (n_blocks * block));
    }
    /* the table, as the model builds it: libref's rope_table at every position this step names */
    std::vector<float> table((size_t)(table_rows * c.rot), 0.0f);
    RadTensor tt{};
    tt.data = table.data(); tt.dtype = RAD_F32; tt.rank = 2;
    tt.shape[0] = table_rows; tt.shape[1] = c.rot; tt.stride[0] = c.rot; tt.stride[1] = 1;
    if (launch(g_rope_table, { RAD_INT("M", c.T), RAD_INT("rot", c.rot), RAD_F64("theta", theta),
                               RAD_F64("scale", 1.0), RAD_STR("dtype", "f32") }, { pos.t, tt }) < 0) {
        std::printf("  %s: rope_table failed\n", c.name);
        return false;
    }
    struct Side { Buf k, qout, gout, cache; };
    auto make = [&]() {
        return Side{ k0, Buf(RAD_BF16, { c.T, c.nh * c.hd }), Buf(RAD_BF16, { c.T, c.nh * c.hd }),
                     Buf(RAD_BF16, { n_blocks, c.nkv, block, 2 * c.hd }) };
    };
    Side f = make(), u = make();
    f.k.t.data = f.k.bytes.data(); u.k.t.data = u.k.bytes.data();
    RadTensor none{};

    std::vector<RadParam> fp = { RAD_INT("M", c.band ? c.band : c.T), RAD_INT("head_dim", c.hd), RAD_INT("n_head", c.nh),
                                 RAD_INT("n_head_kv", c.nkv), RAD_INT("rot", c.rot),
                                 RAD_STR("q_dtype", "bf16"), RAD_F64("eps", eps), RAD_F64("wadd", wadd),
                                 RAD_INT("cs_f32", 1) };
    int s1;
    if (c.kv) {
        fp.push_back(RAD_INT("block_size", block));
        fp.push_back(RAD_STR("kv_dtype", "bf16"));
        s1 = launch(g_fused_qkkv, fp, { qg.t, f.k.t, tt, pos.t, qw.t, kw.t, f.qout.t, f.k.t,
                                        c.gate_out ? f.gout.t : none, v.t, slots.t, f.cache.t });
    } else {
        s1 = launch(g_fused_qk, fp, { qg.t, f.k.t, tt, pos.t, qw.t, kw.t, f.qout.t, f.k.t,
                                      c.gate_out ? f.gout.t : none });
    }

    /* the unfused chain, as the attention block issues it past the fused rows */
    RadTensor qview{};
    qview.data = qg.bytes.data(); qview.dtype = RAD_BF16; qview.rank = 2;
    qview.shape[0] = c.T * c.nh; qview.shape[1] = c.hd; qview.stride[0] = 2 * c.hd; qview.stride[1] = 1;
    int s2 = launch(g_rmsnorm, { RAD_INT("M", c.T * c.nh), RAD_INT("n", c.hd), RAD_F64("eps", eps),
                                 RAD_STR("dtype", "bf16"), RAD_F64("wadd", wadd) },
                    { qview, qw.t, u.qout.t });
    if (s2 >= 0 && mutate != 2)
        s2 = launch(g_rmsnorm, { RAD_INT("M", c.T * c.nkv), RAD_INT("n", c.hd), RAD_F64("eps", eps),
                                 RAD_STR("dtype", "bf16"), RAD_F64("wadd", wadd) },
                    { u.k.t, kw.t, u.k.t });
    const double th = mutate == 1 ? theta * 1.5 : theta;
    auto rope = [&](Buf& x, int64_t heads) {
        return launch(g_rope, { RAD_INT("M", c.T), RAD_INT("head_dim", c.hd), RAD_INT("n_head", heads),
                                RAD_INT("n_head_kv", 0), RAD_F64("theta", th), RAD_F64("scale", 1.0),
                                RAD_STR("mode", "mrope"), RAD_INT("rotary_dim", c.rot) },
                      { x.t, pos.t });
    };
    if (s2 >= 0) s2 = rope(u.qout, c.nh);
    if (s2 >= 0) s2 = rope(u.k, c.nkv);
    if (s2 >= 0 && c.kv)
        s2 = launch(g_kv_store, { RAD_INT("M", c.T), RAD_INT("head_dim", c.hd), RAD_INT("n_head_kv", c.nkv),
                                  RAD_INT("block_size", block), RAD_STR("kv_dtype", "bf16") },
                    { u.k.t, v.t, slots.t, u.cache.t });
    if (s2 >= 0 && c.gate_out)
        for (int64_t r = 0; r < c.T * c.nh; ++r)
            std::memcpy(u.gout.bytes.data() + r * c.hd * 2, qg.bytes.data() + (r * 2 * c.hd + c.hd) * 2,
                        (size_t)c.hd * 2);
    if (s1 < 0 || s2 < 0) {
        std::printf("  %s: launch failed (fused %d, chain %d)\n", c.name, s1, s2);
        return false;
    }
    bool same = true;
    const std::pair<const char*, std::pair<Buf*, Buf*>> outs[] = {
        { "q_out", { &f.qout, &u.qout } }, { "k_out", { &f.k, &u.k } },
        { "gate_out", { &f.gout, &u.gout } }, { "kv_cache", { &f.cache, &u.cache } },
    };
    for (const auto& [nm, pr] : outs) {
        if (pr.first->bytes == pr.second->bytes) continue;
        same = false;
        if (mutate) continue;
        /* how far: the largest difference in bf16 units of the last place, reported, not hidden */
        int64_t worst = 0, n_diff = 0;
        const uint16_t *x = (const uint16_t*)pr.first->bytes.data(), *y = (const uint16_t*)pr.second->bytes.data();
        for (size_t i = 0; i < pr.first->bytes.size() / 2; ++i)
            if (x[i] != y[i]) { ++n_diff; const int64_t d = std::llabs((int64_t)x[i] - (int64_t)y[i]); if (d > worst) worst = d; }
        std::printf("  %s: %s differs at %lld of %zu elements, at most %lld bf16 ULP\n", c.name, nm,
                    (long long)n_diff, pr.first->bytes.size() / 2, (long long)worst);
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
    g_fused_qk   = find_row(mine, "exl3test_qk_norm_rope_gate");
    g_fused_qkkv = find_row(mine, "exl3test_qk_norm_rope_gate_kv_store");
    g_rmsnorm    = find_row(ref, "ref_rmsnorm");
    g_rope       = find_row(ref, "ref_rope");
    g_rope_table = find_row(ref, "ref_rope_table");
    g_kv_store   = find_row(ref, "ref_kv_store");
    if (!g_fused_ts || !g_fused_gts || !g_gemm || !g_topk || !g_scatter || !g_fused_qk ||
        !g_fused_qkkv || !g_rmsnorm || !g_rope || !g_rope_table || !g_kv_store) {
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
        { "5 tokens in the M<=64 band",     5,  16, 16, 0, 4,  64, 1, nullptr, false, false, 64 },
        { "fused GEMM, 5 tokens, band 64",  5,  32, 32, 0, 8, 256, 1, nullptr, true,  true,  64 },
    };
    for (const Case& c : cases) {
        const bool ok = agree(c, 0);
        std::printf("%s %s\n", ok ? "PASS" : "FAIL", c.name);
        g_failed += !ok;
    }
    const QkCase qk[] = {
        { "prologue, decode, one token",          1, 24, 2, 256, 64, false, false },
        { "prologue, seven tokens, gate out",     7,  4, 2,  64, 16, false, true  },
        { "prologue + kv_store, 64 tokens",      64,  4, 2,  64, 16, true,  false },
        { "prologue + kv_store, full rotary",     5,  2, 1, 128, 128, true, true  },
        { "prologue, 5 tokens in the M<=64 band", 5, 24, 2, 256, 64, false, false, 64 },
        { "prologue + kv_store, 5 tokens, band 64", 5, 24, 2, 256, 64, true, false, 64 },
    };
    for (const QkCase& c : qk) {
        const bool ok = qk_agree(c, 0);
        std::printf("%s %s\n", ok ? "PASS" : "FAIL", c.name);
        g_failed += !ok;
    }
    /* the controls: each must DIFFER */
    const bool c1 = agree(cases[3], 1), c2 = agree(cases[5], 2);
    const bool c3 = qk_agree(qk[2], 1), c4 = qk_agree(qk[2], 2);
    std::printf("%s control: the scatter without the slice's expert_base differs\n", c1 ? "FAIL" : "PASS");
    std::printf("%s control: the chain without the gate differs\n", c2 ? "FAIL" : "PASS");
    std::printf("%s control: the chain's rope at another theta than the table's differs\n", c3 ? "FAIL" : "PASS");
    std::printf("%s control: the chain without the k norm differs\n", c4 ? "FAIL" : "PASS");
    g_failed += c1 + c2 + c3 + c4;
    std::printf("RESULT: %s\n", g_failed ? "FAIL" : "PASS");
    return g_failed ? 1 : 0;
}
