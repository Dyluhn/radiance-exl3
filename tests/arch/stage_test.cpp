// stage_test.cpp -- exl3_stage.h's prefill-staging decision from its inputs alone (no radiance, no device): the
// default (radiance's own: on), the opt-in `auto` rule's zero-copy reads and choices, the overrides, and the line. One line a case; exit 0 when every case holds.
#include "../../architectures/qwen4exp_exl3/exl3_stage.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

namespace exl3 = rad::arch::exl3;
static int g_fail = 0;
static void check(const char* what, bool ok) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_fail;
}

int main() {
    const double g5x16 = 32.0 * 16 * 128 / 130 / 8, g4x4 = 16.0 * 4 * 128 / 130 / 8;
    const long long E = 512, K = 10;   /* this model */
    const exl3::StageRule r;
    check("tp2's 2048 a step (lambda 40): R 1.00 -- each pooled expert read about once either way",
          std::fabs(r.reads(2048, E, K) - 1.0) < 0.01);
    check("tp1's 4096 a step (lambda 80): R ~1.96 -- zero-copy reads it about twice",
          std::fabs(r.reads(4096, E, K) - 1.96) < 0.02);
    check("R is the expected entries: lambda 64 exactly at an entry, ~1.5", std::fabs(r.reads(64 * E / K, E, K) - 1.47) < 0.03);
    check("unknown counts: no R", r.reads(2048, 0, K) < 0 && r.reads(0, E, K) < 0 && r.reads(2048, E, 0) < 0);
    auto unset_at = [&](double gbps, long long tok) { return exl3::decide_stage(-1, nullptr, gbps, tok, E, K, "x").stage; };
    check("unset is radiance's default (on) at every step size and link -- v1.1.0's rule-by-default was x4-only evidence",
          unset_at(g4x4, 2048) && unset_at(g5x16, 2048) && unset_at(g5x16, 4096) && unset_at(EXL3_LINK_UNKNOWN, 2048));
    check("unset reports the default as its source", !std::strcmp(exl3::decide_stage(-1, nullptr, g5x16, 2048, E, K, "x").source, "default"));
    auto auto_at = [&](double gbps, long long tok) { return exl3::decide_stage(-1, "auto", gbps, tok, E, K, "x").stage; };
    check("auto at tp2's 2048 a step: off -- as measured on the x4 card (off 2.2x)", !auto_at(g4x4, 2048));
    check("auto at tp1's 4096 a step on the x4 card: on -- as measured (on 1.47x)", auto_at(g4x4, 4096));
    check("auto: the link does not decide (a Gen5 x16 path the same)", !auto_at(g5x16, 2048) && auto_at(g5x16, 4096));
    check("auto with unknown expert counts keeps radiance's default (on)", exl3::decide_stage(-1, "auto", g4x4, 2048, 0, 0, "x").stage);
    check("EXL3_PREFILL_STAGE=on stages at 2048", exl3::decide_stage(-1, "on", g4x4, 2048, E, K, "x").stage);
    check("EXL3_PREFILL_STAGE=off does not at 4096", !exl3::decide_stage(-1, "off", g4x4, 4096, E, K, "x").stage);
    const exl3::StageChoice junk = exl3::decide_stage(-1, "maybe", g4x4, 2048, E, K, "x");
    check("an unreadable EXL3_PREFILL_STAGE is ignored (radiance's default) and said so",
          junk.stage && !std::strcmp(junk.source, "default") && junk.line.find("was ignored") != std::string::npos);
    check("a forced build wins over the environment", exl3::decide_stage(1, "off", g4x4, 2048, E, K, "x").stage &&
                                                          !exl3::decide_stage(0, "on", g4x4, 4096, E, K, "x").stage);
    const std::string cards = "rank 0 0000:03:00.0 63.0 GB/s, rank 1 ...";
    const exl3::StageChoice d = exl3::decide_stage(-1, nullptr, g4x4, 2048, E, K, cards);
    check("the default line names the choice, how to change it and the host path",
          d.line.find("ON (radiance's default") != std::string::npos && d.line.find("off or auto") != std::string::npos &&
              d.line.find("7.9 GB/s") != std::string::npos && d.line.find("rank 0 0000:03:00.0") != std::string::npos);
    const exl3::StageChoice c = exl3::decide_stage(-1, "auto", g4x4, 2048, E, K, cards);
    check("the auto line names the choice, the inputs and the threshold",
          c.line.find("OFF (EXL3_PREFILL_STAGE=auto)") != std::string::npos && c.line.find("1.00") != std::string::npos &&
              c.line.find("1.50") != std::string::npos && c.line.find("2048 tokens") != std::string::npos &&
              c.line.find("top-10 of 512") != std::string::npos && c.line.find("7.9 GB/s") != std::string::npos);
    std::printf("%s\n%s\n", d.line.c_str(), c.line.c_str());
    std::printf("stage_test: %d failed\n", g_fail);
    return g_fail ? 1 : 0;
}
