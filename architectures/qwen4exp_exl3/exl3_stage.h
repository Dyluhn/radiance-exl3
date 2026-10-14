/* exl3_stage.h -- whether radiance's prefill stager runs for this serving setup. THE DEFAULT IS RADIANCE'S OWN
 * (staging on), as the baseline container is served; the plugin only turns it off when asked.
 *
 * WHAT IS DECIDED. radiance's prefill stager, where its planner budgets a ring, copies each routed layer's pooled
 * experts onto the card ahead of the layer on every step of more than 1024 tokens -- ALL of them, once a step. With it
 * off, the prefill kernel reads them zero-copy: one workgroup a map entry of up to 64 rows of one expert
 * (kernels/exl3r4d/exl3r4d_prefill.h, 16 x MTX 4), each entry streaming its expert's weights over the host path. So a
 * step moves, per pooled expert, ceil(rows / 64) copies of it zero-copy (none if no row routes to it) against exactly
 * one staged. The architecture's one switch is declarative (rad_block_moe_exl3.h, exl3::kPrefillStageForced): an
 * appended, never-issued expert reader makes the routed layers' spans overlap and the stager turns itself off.
 *
 * WHY RADIANCE'S DEFAULT (2026-10-14). v1.1.0 shipped `auto` (below) as the default. Its evidence was two passes, both
 * on the development box's Gen4 x4 card, where the link bounds both arms. On Gen5 x16 links it is the wrong call:
 * outside testers measured stock 1.5.0 vs this plugin at tp2, prefill -7..-9 % with the rule's choice (off) and about
 * even with staging on; on the development box's x16 card at tp1, 8K prefill 2,478 tok/s staged vs 2,305 not.
 *
 * `auto`, OPT-IN: stage iff R = E[ceil(n / 64)] >= 1.5, n an expert's rows in a full step (Poisson, lambda = max_tok x
 * top_k / n_expert: routing taken as uniform). R is the zero-copy bytes over the staged bytes. Measured on this model
 * (512 experts, top-10) ON A GEN4 X4 PATH ONLY, staging on vs off in the same harness:
 *   tp2, 2048 a step (lambda 40, R 1.00): off 2.2x faster (24K 3,726 vs 1,705 tok/s; 8K 3,083 vs 1,661);
 *   tp1, 4096 a step (lambda 80, R 1.96): on 1.47x faster (8K 607 vs 413 tok/s; the baseline 580).
 * It is for hosts whose cards sit behind a slow link. The host path is read and printed (common/exl3_link.h, the
 * serving cards' narrowest) but decides nothing.
 *
 * OVERRIDE: EXL3_PREFILL_STAGE = on (the default: radiance's own) | off | auto, read once a process. A test build may
 * force it at compile time (QWEN4EXP_EXL3_PREFILL_STAGE 0 / 1). The choice and its inputs are printed once, at
 * declare. */
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "../../common/exl3_link.h"

namespace rad {
namespace arch {
namespace exl3 {

struct StageRule {
    double entry_rows = 64;    /* rows a prefill map entry at most (16 x EXL3R4D_PF_MTX 4) */
    double min_reads  = 1.5;   /* stage at and above this many zero-copy reads of a pooled expert a step */
    /* E[ceil(n / entry_rows)], n ~ Poisson(max_tok x top_k / n_expert): a pooled expert's zero-copy reads a full step */
    double reads(long long max_tok, long long n_expert, long long top_k) const {
        if (max_tok <= 0 || n_expert <= 0 || top_k <= 0) return -1.0;
        const double lam = (double)max_tok * (double)top_k / (double)n_expert;
        const long long hi = (long long)(lam + 12.0 * std::sqrt(lam) + 32.0);
        double pmf = std::exp(-lam), r = 0.0;   /* P(n = 0) */
        for (long long n = 1; n <= hi; ++n) {
            pmf *= lam / (double)n;
            r += pmf * std::ceil((double)n / entry_rows);
        }
        return r;
    }
};

struct StageChoice {
    bool        stage = true;     /* radiance's own default when nothing can be read */
    const char* source = "default";  /* "default", "auto", "env", "build" */
    double      min_gbps = EXL3_LINK_UNKNOWN;
    double      reads = -1.0;     /* R; < 0 when the model's expert counts are unknown */
    std::string line;             /* the startup line */
};

/* The decision from its inputs alone (testable without a machine). `env` is EXL3_PREFILL_STAGE's value or null;
 * `forced` the build's QWEN4EXP_EXL3_PREFILL_STAGE (-1 automatic). `cards` describes the serving cards for the line. */
inline StageChoice decide_stage(int forced, const char* env, double min_gbps, long long max_tok, long long n_expert,
                                long long top_k, const std::string& cards, const StageRule& rule = StageRule{}) {
    StageChoice c;
    c.min_gbps = min_gbps;
    c.reads = rule.reads(max_tok, n_expert, top_k);
    char buf[768], gb[48];
    if (min_gbps > 0.0) std::snprintf(gb, sizeof gb, "%.1f GB/s", min_gbps);
    else std::snprintf(gb, sizeof gb, "unknown");
    enum { BUILD, ENV, AUTO, DEFAULT } src;
    if (forced == 0 || forced == 1) {
        c.stage = forced == 1;
        c.source = "build";
        src = BUILD;
    } else if (env && (!std::strcmp(env, "on") || !std::strcmp(env, "off"))) {
        c.stage = !std::strcmp(env, "on");
        c.source = "env";
        src = ENV;
    } else if (env && !std::strcmp(env, "auto")) {
        c.stage = !(c.reads >= 0.0) || c.reads >= rule.min_reads;   /* unknown counts: the stager's own default */
        c.source = "auto";
        src = AUTO;
    } else {
        c.stage = true;
        src = DEFAULT;
    }
    const bool bad_env = env && *env && std::strcmp(env, "on") && std::strcmp(env, "off") && std::strcmp(env, "auto");
    if (src == AUTO)
        std::snprintf(buf, sizeof buf, "prefill staging %s (EXL3_PREFILL_STAGE=auto): a pooled expert's zero-copy reads "
                      "a step of %lld tokens %.2f (top-%lld of %lld, %g-row entries) against %.2f to stage%s; the serving "
                      "cards' narrowest host path %s [%s]",
                      c.stage ? "ON" : "OFF", max_tok, c.reads, top_k, n_expert, rule.entry_rows, rule.min_reads,
                      c.reads >= 0.0 ? "" : " (unknown: radiance's default kept)", gb, cards.c_str());
    else if (src == DEFAULT)
        std::snprintf(buf, sizeof buf, "prefill staging ON (radiance's default; EXL3_PREFILL_STAGE=off or auto changes "
                      "it)%s; the serving cards' narrowest host path %s [%s]",
                      bad_env ? "; EXL3_PREFILL_STAGE is not on | off | auto and was ignored" : "", gb, cards.c_str());
    else
        std::snprintf(buf, sizeof buf, "prefill staging %s (set by %s); the serving cards' narrowest host path %s [%s]",
                      c.stage ? "ON" : "OFF", src == ENV ? "EXL3_PREFILL_STAGE" : "the build's QWEN4EXP_EXL3_PREFILL_STAGE",
                      gb, cards.c_str());
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
            char gb[32];
            std::snprintf(gb, sizeof gb, "%.1f", l[i].path_gbps);
            d += std::string(i ? ", " : "") + "rank " + std::to_string(i) + " " + l[i].bdf + " " + gb + " GB/s" +
                 (l[i].link_capped ? " (capped)" : "");
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
inline bool prefill_stage(int forced, int world_size, long long max_tok, long long n_expert, long long top_k) {
    static const StageChoice choice = [&] {
        std::string cards;
        const double m = forced == 0 || forced == 1 ? EXL3_LINK_UNKNOWN : serving_min_gbps(world_size, &cards);
        StageChoice c = decide_stage(forced, std::getenv("EXL3_PREFILL_STAGE"), m, max_tok, n_expert, top_k, cards);
        std::fprintf(stderr, "I qwen4exp_exl3: %s\n", c.line.c_str());
        return c;
    }();
    return choice.stage;
}

}  // namespace exl3
}  // namespace arch
}  // namespace rad
