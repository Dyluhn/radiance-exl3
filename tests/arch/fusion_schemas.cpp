/* fusion_schemas.cpp -- TEST ONLY, never installed: a kernel plugin that DECLARES the schemas of three
 * fused ops a host-only radiance 1.3.0 has none for, and implements none of them.
 *
 * WHY. qwen4exp's blocks ask for fused forms and expect a null handle where no kernel serves one,
 * then issue the parts: the routed block's `router_topk_scatter` (and, where that resolves,
 * `router_gemm_topk_scatter`) falls back to `router_topk` + `moe_scatter`, and the gated
 * attention's `qk_norm_rope_gate` to its norm and rope -- all of which libref serves. But only
 * libr4d, a device library, declares these SCHEMAS at 1.3.0, and an op no loaded plugin declares a
 * schema for is a declare ERROR, not a null handle (core/plugin/schema.cpp:106; the builder's
 * errors fail the declare, core/build/startup.cpp:403-407). So a host-only engine cannot declare
 * qwen4exp at all (Step 0, S0.1, and measured again in P3's notes) -- the in-tree plugin or this
 * repository's. Declaring the schemas with no row turns each miss back into what the block was
 * written for: the fusion resolves to nothing and the parts run. No numbers change: a fused form
 * is byte-identical to the parts it stands in for (docs/OPS.md).
 *
 * The QSA indexer's ops (`qsa_work`, `qsa_score`, `qsa_select`, `qsa_tail_store`) are device-only
 * too and have NO unfused form, so a host-only fixture carries no indexer (`indexer_n_heads` 0):
 * below the indexer's exactness bound QSA is dense causal attention bit for bit, which is what the
 * attention then runs (arch/qwen4exp_fp8/qwen4exp_fp8.cpp's header).
 *
 * THE SCHEMAS ARE libr4d's, slot for slot (libr4d/r4d_rows.cpp:210-237 and 817-860 at 1.3.0): keys, types,
 * requiredness, operand roles and optionality, which is what the loader compares (core/plugin/
 * loader.cpp, schema_same). Loaded beside libr4d with a different one, one of the two would be
 * refused by name -- which is why this is only ever named in a test's --kernels list, never
 * installed, and why tests/arch/CMakeLists.txt compares it against libr4d wherever libr4d exists.
 */
#include <rad_abi.h>

namespace {

#define P_INT(k)  { (k), RAD_P_INT, RAD_REQUIRED, 0 }
#define P_INTO(k) { (k), RAD_P_INT, RAD_OPTIONAL, 0 }
#define P_STR(k)  { (k), RAD_P_STR, RAD_REQUIRED, 0 }
#define P_STRO(k) { (k), RAD_P_STR, RAD_OPTIONAL, 0 }
#define P_F64O(k) { (k), RAD_P_F64, RAD_OPTIONAL, 0 }
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
const RadParamSpec kQkNormRopeGateParams[] = {
    P_INT("M"), P_INT("head_dim"), P_INT("n_head"), P_INT("n_head_kv"), P_INT("rot"),
    P_STR("q_dtype"), P_F64O("eps"), P_F64O("wadd"), P_INTO("cs_f32"), P_INTO("pos_i64"),
};
const RadOperandSpec kQkNormRopeGateOperands[] = {
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
const RadOpSchema kSchemas[] = {
    { "qk_norm_rope_gate", kQkNormRopeGateParams, N(kQkNormRopeGateParams), kQkNormRopeGateOperands,
      N(kQkNormRopeGateOperands), "libr4d's schema, declared so a host engine can resolve it to nothing" },
    { "router_topk_scatter", kTopkScatterParams, N(kTopkScatterParams), kTopkScatterOperands,
      N(kTopkScatterOperands), "libr4d's schema, declared so a host engine can resolve it to nothing" },
    { "router_gemm_topk_scatter", kGemmTopkScatterParams, N(kGemmTopkScatterParams),
      kGemmTopkScatterOperands, N(kGemmTopkScatterOperands),
      "libr4d's schema, declared so a host engine can resolve it to nothing" },
};

}  /* namespace */

extern "C" {
uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
const RadPluginInfo* rad_plugin_info(void) {
    static const RadPluginInfo info = { RAD_PLUGIN_KERNEL, "exl3test_fusion_schemas", "0.1.0",
                                        "TEST ONLY: three of libr4d's fusion schemas, no rows",
                                        "host" };
    return &info;
}
int rad_kernel_schema_count(void) { return N(kSchemas); }
const RadOpSchema* rad_kernel_schema_at(int i) { return i >= 0 && i < N(kSchemas) ? &kSchemas[i] : nullptr; }
int rad_kernel_count(void) { return 0; }
const RadKernelInfo* rad_kernel_at(int) { return nullptr; }
}
