/* exl3_stage.h -- whether radiance's prefill stager runs for this serving setup: an AUTOMATIC per-system decision
 * (PLAN §0: the plugin ships to other machines; only the R9700 is certain; nothing may be fitted to one box's link).
 *
 * WHAT IS DECIDED. radiance 1.4.0's prefill stager, where its planner budgets a ring (at this model's tp2, 488 MiB:
 * two buffers of one routed layer's pooled experts), copies each routed layer's pooled experts onto the card ahead of
 * the layer on every pass of more than ~1 K tokens. With it off, the kernels read those experts zero-copy, as in
 * decode. The architecture's one switch is declarative (rad_block_moe_exl3.h, exl3::kPrefillStageForced): an appended,
 * never-issued expert reader makes the routed layers' spans overlap and the stager turns itself off.
 *
 * THE RULE, FROM MEASURED QUANTITIES. Staging pays only while a layer's copy hides behind a layer's prefill time:
 *
 *     stage  iff  P / (eta * B) <= C * (max_tok / C_tok)
 *
 *   B      the narrowest host path over the SERVING cards (exl3_link_serving: every hop to the root port, the
 *          minimum over the ranks -- every rank computes it from the same sysfs, so all ranks agree);
 *   P      one routed layer's pooled expert bytes a rank: 256 MB measured (the tp2 ring, 488 MiB, holds two);
 *   C      a routed layer's prefill time with staging off at C_tok tokens a step: 11 ms at 2048 (tp2, 24K prefill
 *          at 3,880 tok/s, 48 layers, P6 v13); a step of max_tok tokens takes proportionally longer;
 *   eta    the DMA efficiency of a PCIe path, 0.85 (an assumption, not measured here).
 *
 * That puts the path threshold at ~27 GB/s at 2048 tokens a step, ~14 GB/s at 4096: Gen4/Gen5 x16 and Gen5 x8
 * stage, Gen4 x4/x8 do not. The one measurement with the stager armed (tp2, a Gen4 x4 rank: 7.9 GB/s) agrees -- off
 * 2.2x faster at 24K (v13 3,880 vs v12 1,796 tok/s), the copy taking ~21 ms a layer against 11. The fast-link side is
 * the model's, not a measurement: on the development box the stager arms only at tp2, which always includes the x4
 * card (at tp1 the planner budgets no ring, so the choice changes nothing there). P and C are this model's at its
 * equal-VRAM pool; a machine that pools a larger share of each layer moves the threshold up, a smaller one down.
 *
 * OVERRIDE: EXL3_PREFILL_STAGE = auto (the default) | on | off, read once a process. A test build may force it at
 * compile time (QWEN4EXP_EXL3_PREFILL_STAGE 0 / 1). The choice and its inputs are printed once, at declare. */
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "../../common/exl3_link.h"

namespace rad {
namespace arch {
namespace exl3 {

struct StageRule {
    double layer_bytes = 256e6;   /* P */
    double layer_sec   = 0.011;   /* C at layer_tok tokens a step */
    double layer_tok   = 2048;    /* C_tok */
    double dma_eff     = 0.85;    /* eta */
    /* the path GB/s at and above which staging hides behind a step of `max_tok` tokens */
    double threshold_gbps(long long max_tok) const {
        const double tok = max_tok > 0 ? (double)max_tok : layer_tok;
        return layer_bytes / (dma_eff * layer_sec * (tok / layer_tok)) / 1e9;
    }
};

struct StageChoice {
    bool        stage = true;     /* radiance's own default when nothing can be read */
    const char* source = "auto";  /* "auto", "env", "build" */
    double      min_gbps = EXL3_LINK_UNKNOWN;
    double      threshold_gbps = 0;
    std::string line;             /* the startup line */
};

/* The decision from its inputs alone (testable without a machine). `env` is EXL3_PREFILL_STAGE's value or null;
 * `forced` the build's QWEN4EXP_EXL3_PREFILL_STAGE (-1 automatic). `cards` describes the serving cards for the line. */
inline StageChoice decide_stage(int forced, const char* env, double min_gbps, long long max_tok, const std::string& cards,
                                const StageRule& rule = StageRule{}) {
    StageChoice c;
    c.min_gbps = min_gbps;
    c.threshold_gbps = rule.threshold_gbps(max_tok);
    char buf[640], gb[48];
    if (min_gbps > 0.0) std::snprintf(gb, sizeof gb, "%.1f GB/s", min_gbps);
    else std::snprintf(gb, sizeof gb, "unknown");
    if (forced == 0 || forced == 1) {
        c.stage = forced == 1;
        c.source = "build";
    } else if (env && (!std::strcmp(env, "on") || !std::strcmp(env, "off"))) {
        c.stage = !std::strcmp(env, "on");
        c.source = "env";
    } else {
        c.stage = !(min_gbps > 0.0) || min_gbps >= c.threshold_gbps;   /* unknown link: the stager's own default */
        c.source = "auto";
    }
    const bool bad_env = env && *env && std::strcmp(env, "on") && std::strcmp(env, "off") && std::strcmp(env, "auto");
    if (!std::strcmp(c.source, "auto"))
        std::snprintf(buf, sizeof buf, "prefill staging %s (automatic): the serving cards' narrowest host path %s%s "
                      "against %.1f GB/s, the bandwidth at which a layer's pooled experts copy within a layer's "
                      "prefill at %lld tokens a step [%s]%s",
                      c.stage ? "ON" : "OFF", gb, min_gbps > 0.0 ? "" : " (radiance's default kept)",
                      c.threshold_gbps, max_tok, cards.c_str(),
                      bad_env ? "; EXL3_PREFILL_STAGE is not auto | on | off and was ignored" : "");
    else
        std::snprintf(buf, sizeof buf, "prefill staging %s (set by %s); the serving cards' narrowest host path %s [%s]",
                      c.stage ? "ON" : "OFF",
                      !std::strcmp(c.source, "env") ? "EXL3_PREFILL_STAGE" : "the build's QWEN4EXP_EXL3_PREFILL_STAGE", gb,
                      cards.c_str());
    c.line = buf;
    return c;
}

/* The serving cards' narrowest path and a description, from sysfs (EXL3_SYSFS overrides the root, for tests). */
inline double serving_min_gbps(int world_size, std::string* cards) {
    const char* root = std::getenv("EXL3_SYSFS");
    const std::string sys = root && *root ? root : "/sys";
    Exl3Link l[16];
    const int n = exl3_link_serving(sys.c_str(), std::getenv("ROCR_VISIBLE_DEVICES"), std::getenv("HIP_VISIBLE_DEVICES"),
                                    world_size, l, 16);
    double m = EXL3_LINK_UNKNOWN;
    std::string d;
    if (n >= world_size && n > 0) {
        for (int i = 0; i < n; ++i) {
            char b[96];
            std::snprintf(b, sizeof b, "%srank %d %s %.1f GB/s%s", i ? ", " : "", i, l[i].bdf, l[i].path_gbps,
                          l[i].link_capped ? " (capped)" : "");
            d += b;
            if (l[i].path_gbps > 0.0 && (m < 0.0 || l[i].path_gbps < m)) m = l[i].path_gbps;
        }
    } else {
        m = exl3_link_min_gbps(sys.c_str());
        d = n < 0 ? "the rank-to-card order could not be read; the narrowest of every discrete card"
                  : "fewer cards found than ranks; the narrowest of every discrete card";
    }
    *cards = d;
    return m;
}

/* Once a process: every rank declares on its own thread and gets the first caller's answer (the inputs are the
 * same for all). Printed once. */
inline bool prefill_stage(int forced, int world_size, long long max_tok) {
    static const StageChoice choice = [&] {
        std::string cards;
        const double m = forced == 0 || forced == 1 ? EXL3_LINK_UNKNOWN : serving_min_gbps(world_size, &cards);
        StageChoice c = decide_stage(forced, std::getenv("EXL3_PREFILL_STAGE"), m, max_tok, cards);
        std::fprintf(stderr, "I qwen4exp_exl3: %s\n", c.line.c_str());
        return c;
    }();
    return choice.stage;
}

}  // namespace exl3
}  // namespace arch
}  // namespace rad
