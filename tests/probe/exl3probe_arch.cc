/* exl3probe_arch.cc -- Step 0 throwaway (S0.3, S0.4, S0.5, S0.10): a stub architecture plugin
 * claiming ("qwen4exp", "exl3"). NOT the architecture plugin (architectures/qwen4exp_exl3, owner A).
 *
 * It declares a handful of qwen4exp's weights under the in-tree plugin's names, each read by the
 * probe op `exl3probe_mm` (tests/probe/exl3probe_kern.cc), and nothing else -- enough for
 * rad-convert and a host-only declare at --tp 1 or 2 to run:
 *   - every routed expert, `blk.L.ffn_gate_up_exps.E.weight` / `blk.L.ffn_down_exps.E.weight`, as
 *     ONE operand selecting ALL planes (D6), grp_expert_slot(L, E, 0|1), gate_up ROW-sharded with
 *     two stacked row parts and down COL-sharded, at --tp 2 cut by the in-tree parity split
 *     (qwen4exp_fp8.cpp: nb = n_ff/128 blocks, contiguous per parity, the rank's place rotated
 *     by the parity) through rad_weight_shard_span (D7);
 *   - `blk.L.ffn_gate_inp_shexp.weight` (no recipe rule: the checkpoint's bf16), `blk.L.ffn_down_shexp.weight`
 *     (the served recipe's int8 rule) and `blk.1.ple_ngram.weight` from its 128 checkpoint shards
 *     (the served recipe's fp8 rule) -- the three kinds of non-expert weight --reuse must copy (S0.5).
 * The container key `exl3probe.cut_shift` (rad-convert --set) moves every TP cut right by that many
 * columns -- S0.3's negative control: an odd shift puts down's u4 codes cut off a byte, which the
 * core's plane slicer must refuse.
 * Its step refuses: nothing here runs.
 */
#include <arch/rad_arch.h>

#include <algorithm>
#include <cstdio>

namespace exl3probe_arch {

using namespace rad::arch;

const char kCkpt[] = "model.language_model.layers.";

int op_for(RadBuilder* b, const RadBuildCtx* ctx, rad_weight h, int64_t n, int64_t k) {
    if (!h) return RAD_E_INVAL;
    const rad_op op = RAD_OP(b, "exl3probe_mm",
                             RAD_PARAMS(RAD_RANGE("M", 1, ctx->max_tok), RAD_INT("N", n),
                                        RAD_INT("K", k)),
                             RAD_WEIGHTS(h));
    return op ? RAD_OK : RAD_E_INVAL;
}

/* The in-tree split (qwen4exp_fp8.cpp, `g.world <= nb`): expert parity q's 128-blocks are cut
 * contiguously across the ranks, this rank's place being (rank - q) mod world. */
void parity_span(int rank, int world, int64_t n_ff, int q, int64_t* lo, int64_t* hi) {
    const int64_t nb = n_ff / 128, base = nb / world, extra = nb % world;
    const int64_t p = (rank - q + world) % world;
    const int64_t b0 = p * base + std::min<int64_t>(p, extra);
    *lo = b0 * 128;
    *hi = (b0 + base + (p < extra ? 1 : 0)) * 128;
}

int declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* ctx) {
    std::fprintf(stderr, "exl3probe_arch: declare arch '%s' quant '%s' rank %d of %d\n", m->arch_id,
                 m->quant, ctx->rank, ctx->world_size);
    Names nm(ctx->scope);
    const int64_t E = m->n_expert, H = m->n_embd, F = m->n_ff, W = ctx->world_size;
    const bool split = W > 1;
    const int64_t shift = rad_meta_geti(m, "exl3probe.cut_shift", 0);
    if (F % 128 || (split && W > F / 128)) return RAD_E_SHAPE;

    for (int l = 0; l < (int)m->n_layers; ++l) {
        const char* gu_src = nm.ckpt("%s%d.mlp.experts.gate_up_proj", kCkpt, l);
        const char* dn_src = nm.ckpt("%s%d.mlp.experts.down_proj", kCkpt, l);
        for (int e = 0; e < (int)E; ++e) {
            const char* guw = nm.f("blk.%d.ffn_gate_up_exps.%d.weight", l, e);
            const char* dnw = nm.f("blk.%d.ffn_down_exps.%d.weight", l, e);
            RAD_ARCH_TRY(map_slice(b, guw, gu_src, e));
            RAD_ARCH_TRY(map_slice(b, dnw, dn_src, e));
            int64_t lo = 0, hi = F;
            if (split) {
                parity_span(ctx->rank, (int)W, F, e % 2, &lo, &hi);
                lo = lo > 0 ? lo + shift : lo;
                hi = hi < F ? hi + shift : hi;
            }
            const int64_t w = hi - lo, parts[2] = { w, w };
            const rad_weight hgu =
                decl_view(b, guw, nullptr, nullptr, weight_codes_dtype(b, guw, RAD_BF16),
                          {2 * w, H}, RAD_ACCESS_CONDITIONAL, RAD_SHARD_ROW,
                          grp_expert_slot(l, e, 0), parts, 2);
            const rad_weight hdn =
                decl_view(b, dnw, nullptr, nullptr, weight_codes_dtype(b, dnw, RAD_BF16), {H, w},
                          RAD_ACCESS_CONDITIONAL, RAD_SHARD_COL, grp_expert_slot(l, e, 1));
            if (!hgu || !hdn) return RAD_E_INVAL;
            if (split) {
                RAD_ARCH_TRY(rad_weight_shard_span(b, hgu, lo, hi));
                RAD_ARCH_TRY(rad_weight_shard_span(b, hdn, lo, hi));
            }
            RAD_ARCH_TRY(op_for(b, ctx, hgu, 2 * w, H));
            RAD_ARCH_TRY(op_for(b, ctx, hdn, H, w));
        }
        const char* shg = nm.f("blk.%d.ffn_gate_inp_shexp.weight", l);
        RAD_ARCH_TRY(map_copy(b, shg, nm.ckpt("%s%d.mlp.shared_expert_gate.weight", kCkpt, l)));
        RAD_ARCH_TRY(op_for(b, ctx,
                            decl_view(b, shg, nullptr, nullptr,
                                      weight_codes_dtype(b, shg, RAD_BF16), {1, H},
                                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l)),
                            1, H));
        const char* shdn = nm.f("blk.%d.ffn_down_shexp.weight", l);
        RAD_ARCH_TRY(map_copy(b, shdn, nm.ckpt("%s%d.mlp.shared_expert.down_proj.weight", kCkpt, l)));
        RAD_ARCH_TRY(op_for(b, ctx,
                            decl_view(b, shdn, nullptr, nullptr,
                                      weight_codes_dtype(b, shdn, RAD_BF16), {H, F},
                                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l)),
                            H, F));
    }

    /* The n-gram table: 128 checkpoint shards joined in order (rad_arch.h map_concat_shards). Its
     * extent is read back from what the model says it is. */
    const char* ng = "blk.1.ple_ngram.weight";
    RAD_ARCH_TRY(map_concat_shards(b, ng, nm.ckpt("%s1.ple.ple_embedding.ngram_embedding.shard_", kCkpt),
                                   ".weight", 128));
    int64_t shape[RAD_MAX_RANK] = {0};
    uint32_t rank = 0;
    RadEncoding e{};
    if (rad_weight_encoding(b, ng, &e, shape, &rank) != RAD_OK || rank != 2) return RAD_E_NOTFOUND;
    RAD_ARCH_TRY(op_for(b, ctx,
                        decl_view(b, ng, nullptr, nullptr, weight_codes_dtype(b, ng, RAD_BF16),
                                  {shape[0], shape[1]}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                                  grp_layer(1)),
                        shape[0], shape[1]));
    return RAD_OK;
}

void step(RadCtx*, const RadBatch*) {
    std::fprintf(stderr, "exl3probe_arch: step is not implemented (Step 0 probe, declare only)\n");
}

}  // namespace exl3probe_arch

RAD_ARCH_PLUGIN(exl3probe_arch, "qwen4exp", "exl3", "0.0.1",
                "Step 0 probe: claims (qwen4exp, exl3) and declares only what the probes ask")
