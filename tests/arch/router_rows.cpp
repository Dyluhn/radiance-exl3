/* router_rows.cpp -- TEST ONLY, never installed: host rows for the four fusions a host-only radiance
 * 1.3.0 has none for and qwen4exp always asks for, each running libref's own unfused rows.
 *
 * THE ATTENTION PROLOGUE TOO. The gated attention block always probes `qk_norm_rope_gate` (its
 * rope mode is fixed "mrope", which is NeoX at one position component) and, where that resolves,
 * `qk_norm_rope_gate_kv_store`, and takes them for every pass of at most 64 tokens. The unfused
 * chain they stand in for is two `rmsnorm`s (q heads read out of the [q|gate] projection, k heads
 * in place), two `rope`s and `kv_store` (rad_block_attn_gated_fp8.h, step). The norms and the
 * cache store here ARE libref's rows; the rotation reads the precomputed cos/sin table the schema
 * hands it (it carries no theta), which libref's `rope_table` fills with exactly the angles its
 * `rope` computes in line, so `x0 * c - x1 * s` here is libref's arithmetic, rounded the same way.
 * router_rows_test.cpp checks all four rows byte for byte against libref's chain.
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
#include <rad_plugin.h>
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
#define P_F64O(k) { (k), RAD_P_F64, RAD_OPTIONAL, 0 }
const RadParamSpec kQkParams[] = {
    P_INT("M"), P_INT("head_dim"), P_INT("n_head"), P_INT("n_head_kv"), P_INT("rot"),
    P_STR("q_dtype"), P_F64O("eps"), P_F64O("wadd"), P_INTO("cs_f32"), P_INTO("pos_i64"),
};
const RadOperandSpec kQkOperands[] = {
    { "q_gate",    RAD_OPD_IN,     1 },
    { "k",         RAD_OPD_IN,     0 },
    { "cos_sin",   RAD_OPD_IN,     0 },
    { "positions", RAD_OPD_IN,     0 },
    { "q_w",       RAD_OPD_WEIGHT, 1 },
    { "k_w",       RAD_OPD_WEIGHT, 0 },
    { "q_out",     RAD_OPD_OUT,    1 },
    { "k_out",     RAD_OPD_OUT,    0 },
    { "gate_out",  RAD_OPD_OUT,    1 },
};
const RadParamSpec kQkKvParams[] = {
    P_INT("M"), P_INT("head_dim"), P_INT("n_head"), P_INT("n_head_kv"), P_INT("rot"),
    P_STR("q_dtype"), P_F64O("eps"), P_F64O("wadd"), P_INTO("cs_f32"), P_INTO("pos_i64"),
    P_INT("block_size"), P_STR("kv_dtype"), P_F64O("k_scale"), P_F64O("v_scale"),
};
const RadOperandSpec kQkKvOperands[] = {
    { "q_gate",       RAD_OPD_IN,     1 },
    { "k",            RAD_OPD_IN,     0 },
    { "cos_sin",      RAD_OPD_IN,     0 },
    { "positions",    RAD_OPD_IN,     0 },
    { "q_w",          RAD_OPD_WEIGHT, 1 },
    { "k_w",          RAD_OPD_WEIGHT, 0 },
    { "q_out",        RAD_OPD_OUT,    1 },
    { "k_out",        RAD_OPD_OUT,    0 },
    { "gate_out",     RAD_OPD_OUT,    1 },
    { "v",            RAD_OPD_IN,     0 },
    { "slot_mapping", RAD_OPD_IN,     0 },
    { "kv_cache",     RAD_OPD_INOUT,  0 },
};
const RadOpSchema kSchemas[] = {
    { "qk_norm_rope_gate", kQkParams, N(kQkParams), kQkOperands, N(kQkOperands),
      "libr4d's schema; libref rmsnorm x2, the NeoX rope from the table" },
    { "qk_norm_rope_gate_kv_store", kQkKvParams, N(kQkKvParams), kQkKvOperands, N(kQkKvOperands),
      "libr4d's schema; the prologue above, then libref kv_store" },
    { "router_topk_scatter", kTopkScatterParams, N(kTopkScatterParams), kTopkScatterOperands,
      N(kTopkScatterOperands), "libr4d's schema; router_topk then moe_scatter, libref's rows" },
    { "router_gemm_topk_scatter", kGemmTopkScatterParams, N(kGemmTopkScatterParams),
      kGemmTopkScatterOperands, N(kGemmTopkScatterOperands),
      "libr4d's schema; gemm_nt, router_topk, moe_scatter (and the gate gemm_nt), libref's rows" },
};

/* libref's rows, found at open. */
RadLaunchFn g_gemm_nt = nullptr, g_router_topk = nullptr, g_moe_scatter = nullptr;
RadLaunchFn g_rmsnorm = nullptr, g_kv_store = nullptr;

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

/* ---- the attention prologue ---- */

/* `t` seen as [rows, cols] with `row_stride` elements between rows -- a dense tensor of
 * rows * row_stride elements, last stride 1 -- or a refusal. */
bool view2(const RadTensor& t, int64_t rows, int64_t cols, int64_t row_stride, RadTensor* out) {
    if (!t.data || t.rank < 1 || t.stride[t.rank - 1] != 1) return false;
    for (uint32_t i = t.rank - 1; i > 0; --i)
        if (t.stride[i - 1] != t.stride[i] * t.shape[i]) return false;   /* dense */
    if (rad_tensor_numel(&t) < (rows - 1) * row_stride + cols) return false;
    *out = RadTensor{};
    out->data = t.data; out->dtype = t.dtype; out->rank = 2;
    out->shape[0] = rows; out->shape[1] = cols;
    out->stride[0] = row_stride; out->stride[1] = 1;
    return true;
}

/* NeoX over the first `rot` of each head_dim-wide row of `x` ([rows, head_dim] bf16, dense),
 * rows / heads_per_token heads a token, at the token's position's row of the f32 table: libref's
 * rope_head (libref/ref_attn.cpp), the cos and sin read where it computes them. */
int rope_rows(const RadTensor& x, int64_t heads_per_token, int64_t head_dim, int64_t rot,
              const RadTensor& cs, const RadTensor& pos, bool pos_i64, int64_t T) {
    const int64_t half = rot / 2, table_rows = rad_tensor_numel(&cs) / rot;
    uint16_t* p = (uint16_t*)x.data;
    const float* c = (const float*)cs.data;
    for (int64_t t = 0; t < T; ++t) {
        const int64_t at = pos_i64 ? ((const int64_t*)pos.data)[t] : ((const int32_t*)pos.data)[t];
        if (at < 0 || at >= table_rows) return RAD_E_SHAPE;
        const float* row = c + at * rot;
        for (int64_t h = 0; h < heads_per_token; ++h) {
            uint16_t* v = p + (t * heads_per_token + h) * head_dim;
            for (int64_t i = 0; i < half; ++i) {
                const float cc = row[i], ss = row[half + i];
                const float x0 = rad_bf16_to_f32(v[i]), x1 = rad_bf16_to_f32(v[i + half]);
                v[i]        = rad_f32_to_bf16(x0 * cc - x1 * ss);
                v[i + half] = rad_f32_to_bf16(x0 * ss + x1 * cc);
            }
        }
    }
    return RAD_OK;
}

/* The prologue both rows run: q heads normed out of the [q|gate] projection into q_out, k heads
 * normed in place (k and k_out are one buffer at the block's issue), then both rotated, and the
 * gate copied out where gate_out is given. */
int qk_prologue(const RadArgs* a, RadStream s) {
    const RadTensor* t = a->t;
    const int64_t T = geti(a, "M", 0), hd = geti(a, "head_dim", 0), nh = geti(a, "n_head", 0);
    const int64_t nkv = geti(a, "n_head_kv", 0), rot = geti(a, "rot", 0);
    const char* qdt = rad_args_gets(a, "q_dtype");
    if (!qdt || std::strcmp(qdt, "bf16") || geti(a, "cs_f32", 0) != 1 || t[2].dtype != RAD_F32) {
        std::fprintf(stderr, "exl3test_router: qk_norm_rope_gate serves bf16 q with an f32 cos/sin "
                             "table only\n");
        return RAD_E_DTYPE;
    }
    if (T <= 0 || hd <= 0 || nkv <= 0 || rot <= 0 || rot > hd || (rot & 1)) return RAD_E_SHAPE;
    RadTensor qin, qout, kin, kout;
    if (nh > 0 && (!view2(t[0], T * nh, hd, 2 * hd, &qin) || !view2(t[6], T * nh, hd, hd, &qout)))
        return RAD_E_SHAPE;
    if (!view2(t[1], T * nkv, hd, hd, &kin) || !view2(t[7], T * nkv, hd, hd, &kout)) return RAD_E_SHAPE;
    Params pn(a);
    pn.set_int("n", hd);
    std::vector<RadParam> np;
    for (const RadParam& q : pn.p)   /* rmsnorm's own: M n eps dtype wadd */
        if (!std::strcmp(q.key, "n") || !std::strcmp(q.key, "eps") || !std::strcmp(q.key, "wadd"))
            np.push_back(q);
    np.push_back(RAD_STR("dtype", qdt));
    Params pq(a), pk(a);
    pq.p = np; pq.set_int("M", T * nh);
    pk.p = np; pk.set_int("M", T * nkv);
    int st;
    if (nh > 0 && (st = call(g_rmsnorm, a, pq, { qin, t[4], qout }, s)) < 0) return st;
    if ((st = call(g_rmsnorm, a, pk, { kin, t[5], kout }, s)) < 0) return st;
    const bool i64 = geti(a, "pos_i64", 0) != 0;
    if (nh > 0 && (st = rope_rows(qout, nh, hd, rot, t[2], t[3], i64, T)) < 0) return st;
    if ((st = rope_rows(kout, nkv, hd, rot, t[2], t[3], i64, T)) < 0) return st;
    if (nh > 0 && t[8].data) {   /* the gate halves, as written */
        RadTensor gout;
        if (!view2(t[8], T * nh, hd, hd, &gout)) return RAD_E_SHAPE;
        for (int64_t r = 0; r < T * nh; ++r)
            std::memcpy((uint16_t*)gout.data + r * hd, (const uint16_t*)qin.data + r * 2 * hd + hd,
                        (size_t)hd * 2);
    }
    return RAD_OK;
}

int launch_qk(const RadArgs* a, RadStream s) {
    if (!a || a->n_t != N(kQkOperands)) return RAD_E_INVAL;
    return qk_prologue(a, s);
}

int launch_qk_kv(const RadArgs* a, RadStream s) {
    if (!a || a->n_t != N(kQkKvOperands)) return RAD_E_INVAL;
    int st = qk_prologue(a, s);
    if (st < 0) return st;
    const RadTensor* t = a->t;
    return call(g_kv_store, a, Params(a), { t[7], t[9], t[10], t[11] }, s);
}

const RadKernelInfo kRows[] = {
    { "exl3test_qk_norm_rope_gate", "qk_norm_rope_gate", "attn",
      "libref rmsnorm x2, NeoX rope from the table", "any", "bf16", RAD_DOMAIN_HOST, 0,
      nullptr, 0, nullptr, 0, nullptr, nullptr, nullptr, launch_qk,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr },
    { "exl3test_qk_norm_rope_gate_kv_store", "qk_norm_rope_gate_kv_store", "attn",
      "the prologue, then libref kv_store", "any", "bf16", RAD_DOMAIN_HOST, 0,
      nullptr, 0, nullptr, 0, nullptr, nullptr, nullptr, launch_qk_kv,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr },
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
                                        "TEST ONLY: host rows for four fusions, libref's chain",
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
        if (!std::strcmp(k->name, "ref_rmsnorm"))     g_rmsnorm = k->launch;
        if (!std::strcmp(k->name, "ref_kv_store"))    g_kv_store = k->launch;
    }
    if (!g_gemm_nt || !g_router_topk || !g_moe_scatter || !g_rmsnorm || !g_kv_store) {
        std::fprintf(stderr, "exl3test_router: libref's ref_gemm_nt, ref_router_topk, "
                             "ref_moe_scatter, ref_rmsnorm and ref_kv_store rows were not found on "
                             "RADIANCE_HOME ('%s'); this test library runs them and declines "
                             "without them\n",
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
