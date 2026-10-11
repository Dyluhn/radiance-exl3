/* router_rows.cpp -- TEST ONLY, never installed: host rows for the two router fusions a host-only
 * radiance 1.3.0 has none for, each running libref's own unfused rows in sequence.
 *
 * WHY. The routed block asks for `router_topk_scatter` on every layer and, where that resolves,
 * `router_gemm_topk_scatter`, and falls back to `gemm_nt` + `router_topk` + `moe_scatter` where they
 * do not. At 1.3.0 only libr4d -- a device library -- declares their schemas, and a host-only
 * engine fails twice over: an op no plugin declares a schema for is a declare ERROR
 * (core/plugin/schema.cpp:106; startup.cpp:403-407), and rad-convert refuses any recorded op that
 * resolves in no domain even when the block would never issue it (tools/rad_convert.cpp:604-650).
 * So a host engine cannot convert or serve qwen4exp -- the in-tree plugin or ours -- without rows
 * for both (P3 notes). These are those rows, for the host end-to-end test and nothing else.
 *
 * WHAT THEY COMPUTE IS libref's UNFUSED CHAIN, BY CONSTRUCTION: each launch calls libref's
 * `ref_gemm_nt` (the router, and the shared expert's one-column gate), `ref_router_topk` and
 * `ref_moe_scatter` rows on the operands the fused schema hands it, sliced as the unfused ops'
 * schemas want them, with `n_local` as the scatter's `n_expert`. That is the fused ops' definition
 * (docs/OPS.md, router_topk_scatter and router_gemm_topk_scatter), and router_rows_test.cpp checks
 * it byte for byte against libref's own rows called one by one. libref is found at open on
 * $RADIANCE_HOME (the test sets it); missing, the library declines the machine by name.
 *
 * THE SCHEMAS ARE libr4d's, slot for slot (libr4d/r4d_rows.cpp:817-860 at 1.3.0): keys, types,
 * requiredness, operand roles and optionality, which is what the loader compares
 * (core/plugin/loader.cpp, schema_same). Beside libr4d with a different one, one of the two would
 * be refused by name -- which is why this is never installed and only named in a test's --kernels.
 */
#include <rad_abi.h>
#include <rad_types.h>

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

#define P_INT(k)  { (k), RAD_P_INT, RAD_REQUIRED, 0 }
#define P_INTO(k) { (k), RAD_P_INT, RAD_OPTIONAL, 0 }
#define P_STR(k)  { (k), RAD_P_STR, RAD_REQUIRED, 0 }
#define P_STRO(k) { (k), RAD_P_STR, RAD_OPTIONAL, 0 }
#define N(a) (int)(sizeof(a) / sizeof((a)[0]))

const RadParamSpec kTopkScatterParams[] = {
    P_INT("M"), P_INT("n_expert"), P_INTO("n_local"), P_INT("top_k"), P_INT("norm"),
    P_STR("dtype"), P_INTO("expert_base"), P_STRO("protect"),
};
const RadOperandSpec kTopkScatterOperands[] = {
    { "logits",        RAD_OPD_IN,  0 },
    { "expert_ids",    RAD_OPD_OUT, 0 },
    { "expert_w",      RAD_OPD_OUT, 0 },
    { "sorted_tok",    RAD_OPD_OUT, 0 },
    { "expert_offset", RAD_OPD_OUT, 0 },
    { "expert_count",  RAD_OPD_OUT, 0 },
};
const RadParamSpec kGemmTopkScatterParams[] = {
    P_INT("M"), P_INT("K"), P_INT("n_expert"), P_INTO("n_local"), P_INT("top_k"), P_INT("norm"),
    P_STR("dtype"), P_INTO("expert_base"), P_STRO("protect"),
};
const RadOperandSpec kGemmTopkScatterOperands[] = {
    { "a",             RAD_OPD_IN,     0 },
    { "b",             RAD_OPD_WEIGHT, 0 },
    { "gate_b",        RAD_OPD_WEIGHT, 1 },
    { "logits",        RAD_OPD_OUT,    0 },
    { "expert_ids",    RAD_OPD_OUT,    0 },
    { "expert_w",      RAD_OPD_OUT,    0 },
    { "sorted_tok",    RAD_OPD_OUT,    0 },
    { "expert_offset", RAD_OPD_OUT,    0 },
    { "expert_count",  RAD_OPD_OUT,    0 },
    { "gate",          RAD_OPD_OUT,    1 },
};
const RadOpSchema kSchemas[] = {
    { "router_topk_scatter", kTopkScatterParams, N(kTopkScatterParams), kTopkScatterOperands,
      N(kTopkScatterOperands), "libr4d's schema; router_topk then moe_scatter, libref's rows" },
    { "router_gemm_topk_scatter", kGemmTopkScatterParams, N(kGemmTopkScatterParams),
      kGemmTopkScatterOperands, N(kGemmTopkScatterOperands),
      "libr4d's schema; gemm_nt, router_topk, moe_scatter (and the gate gemm_nt), libref's rows" },
};

/* libref's three rows, found at open. */
RadLaunchFn g_gemm_nt = nullptr, g_router_topk = nullptr, g_moe_scatter = nullptr;

/* The fused op's parameters with one INT replaced (or added) -- the scatter's n_expert is the
 * fused op's n_local -- and one more appended (gemm_nt's N). */
struct Params {
    std::vector<RadParam> p;
    explicit Params(const RadArgs* a) : p(a->p, a->p + a->n_p) {}
    void set_int(const char* key, long long v) {
        for (RadParam& q : p)
            if (q.kind == RAD_P_INT && std::strcmp(q.key, key) == 0) { q.ival = v; return; }
        RadParam q{};
        q.key = key; q.kind = RAD_P_INT; q.ival = v;
        p.push_back(q);
    }
};

long long geti(const RadArgs* a, const char* key, long long dflt) {
    long long v = dflt;
    return rad_args_geti(a, key, &v) ? v : dflt;
}

int call(RadLaunchFn fn, const RadArgs* a, const Params& p, std::initializer_list<RadTensor> t,
         RadStream s) {
    std::vector<RadTensor> tv(t);
    RadArgs sub = *a;
    sub.t = tv.data(); sub.n_t = (int)tv.size();
    sub.p = p.p.data(); sub.n_p = (int)p.p.size();
    return fn(&sub, s);
}

/* router_topk over `logits`, then moe_scatter over this rank's n_local experts. `at` is where the
 * fused op's [logits, ids, w, sorted, offset, count] start in a->t. */
int topk_scatter(const RadArgs* a, int at, RadStream s) {
    const RadTensor* t = a->t + at;
    int st = call(g_router_topk, a, Params(a), { t[0], t[1], t[2] }, s);
    if (st < 0) return st;
    Params ps(a);
    ps.set_int("n_expert", geti(a, "n_local", geti(a, "n_expert", 0)));
    return call(g_moe_scatter, a, ps, { t[1], t[3], t[4], t[5] }, s);
}

int launch_topk_scatter(const RadArgs* a, RadStream s) {
    if (!a || a->n_t != N(kTopkScatterOperands)) return RAD_E_INVAL;
    return topk_scatter(a, 0, s);
}

int launch_gemm_topk_scatter(const RadArgs* a, RadStream s) {
    if (!a || a->n_t != N(kGemmTopkScatterOperands)) return RAD_E_INVAL;
    const RadTensor* t = a->t;
    Params pg(a);
    pg.set_int("N", geti(a, "n_expert", 0));
    int st = call(g_gemm_nt, a, pg, { t[0], t[1], t[3] }, s);
    if (st < 0) return st;
    /* The shared expert's gate: both or neither (docs/OPS.md). */
    if ((t[2].data != nullptr) != (t[9].data != nullptr)) return RAD_E_INVAL;
    if (t[2].data) {
        Params pq(a);
        pq.set_int("N", 1);
        if ((st = call(g_gemm_nt, a, pq, { t[0], t[2], t[9] }, s)) < 0) return st;
    }
    return topk_scatter(a, 3, s);
}

const RadKernelInfo kRows[] = {
    { "exl3test_router_topk_scatter", "router_topk_scatter", "moe",
      "libref router_topk then moe_scatter", "any", "libref's", RAD_DOMAIN_HOST, 0,
      nullptr, 0, nullptr, 0, nullptr, nullptr, nullptr, launch_topk_scatter,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr },
    { "exl3test_router_gemm_topk_scatter", "router_gemm_topk_scatter", "moe",
      "libref gemm_nt, router_topk, moe_scatter", "any", "libref's", RAD_DOMAIN_HOST, 0,
      nullptr, 0, nullptr, 0, nullptr, nullptr, nullptr, launch_gemm_topk_scatter,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr },
};

}  /* namespace */

extern "C" {
uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
const RadPluginInfo* rad_plugin_info(void) {
    static const RadPluginInfo info = { RAD_PLUGIN_KERNEL, "exl3test_router", "0.1.0",
                                        "TEST ONLY: host rows for the router fusions, libref's chain",
                                        "host" };
    return &info;
}

/* libref, from the first $RADIANCE_HOME entry that has it. Missing: declined by name. */
int rad_plugin_open(void) {
    const char* home = std::getenv("RADIANCE_HOME");
    std::string list = home ? home : "";
    void* h = nullptr;
    for (size_t at = 0; at <= list.size() && !h;) {
        size_t colon = list.find(':', at);
        if (colon == std::string::npos) colon = list.size();
        const std::string path = list.substr(at, colon - at) + "/kernels/libref.so";
        h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
        if (!h) h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        at = colon + 1;
    }
    auto count = h ? (int (*)(void))dlsym(h, "rad_kernel_count") : nullptr;
    auto row   = h ? (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at") : nullptr;
    for (int i = 0; count && row && i < count(); ++i) {
        const RadKernelInfo* k = row(i);
        if (!std::strcmp(k->name, "ref_gemm_nt"))     g_gemm_nt = k->launch;
        if (!std::strcmp(k->name, "ref_router_topk")) g_router_topk = k->launch;
        if (!std::strcmp(k->name, "ref_moe_scatter")) g_moe_scatter = k->launch;
    }
    if (!g_gemm_nt || !g_router_topk || !g_moe_scatter) {
        std::fprintf(stderr, "exl3test_router: libref's ref_gemm_nt, ref_router_topk and "
                             "ref_moe_scatter rows were not found on RADIANCE_HOME ('%s'); this "
                             "test library runs them and declines without them\n",
                     home ? home : "");
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}
int rad_kernel_schema_count(void) { return N(kSchemas); }
const RadOpSchema* rad_kernel_schema_at(int i) { return i >= 0 && i < N(kSchemas) ? &kSchemas[i] : nullptr; }
int rad_kernel_count(void) { return N(kRows); }
const RadKernelInfo* rad_kernel_at(int i) { return i >= 0 && i < N(kRows) ? &kRows[i] : nullptr; }
}
