// stage_test.cpp -- exl3_stage.h's automatic prefill-staging decision from its inputs alone (no radiance, no device):
// the zero-copy reads the rule derives, which setups stage at the step sizes the composes use, the overrides, and
// the line. One line a case; exit 0 when every case holds.
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
    auto auto_at = [&](double gbps, long long tok) { return exl3::decide_stage(-1, nullptr, gbps, tok, E, K, "x").stage; };
    check("the development box at tp2 (2048 a step): off -- as measured (off 2.2x)", !auto_at(g4x4, 2048));
    check("the development box at tp1 on the x4 card (4096 a step): on -- as measured (on 1.47x)", auto_at(g4x4, 4096));
    check("the link does not decide: a Gen5 x16 path at 2048 is off too, at 4096 on too",
          !auto_at(g5x16, 2048) && auto_at(g5x16, 4096));
    check("an unknown link decides the same", !auto_at(EXL3_LINK_UNKNOWN, 2048) && auto_at(EXL3_LINK_UNKNOWN, 4096));
    check("unknown expert counts keep radiance's default (on)", exl3::decide_stage(-1, nullptr, g4x4, 2048, 0, 0, "x").stage);
    check("EXL3_PREFILL_STAGE=on stages at 2048", exl3::decide_stage(-1, "on", g4x4, 2048, E, K, "x").stage);
    check("EXL3_PREFILL_STAGE=off does not at 4096", !exl3::decide_stage(-1, "off", g4x4, 4096, E, K, "x").stage);
    check("EXL3_PREFILL_STAGE=auto is the rule", !exl3::decide_stage(-1, "auto", g4x4, 2048, E, K, "x").stage &&
                                                      exl3::decide_stage(-1, "auto", g4x4, 4096, E, K, "x").stage);
    const exl3::StageChoice junk = exl3::decide_stage(-1, "maybe", g4x4, 2048, E, K, "x");
    check("an unreadable EXL3_PREFILL_STAGE is ignored (the rule) and said so",
          !junk.stage && !std::strcmp(junk.source, "auto") && junk.line.find("was ignored") != std::string::npos);
    check("a forced build wins over the environment", exl3::decide_stage(1, "off", g4x4, 2048, E, K, "x").stage &&
                                                          !exl3::decide_stage(0, "on", g4x4, 4096, E, K, "x").stage);
    const exl3::StageChoice c = exl3::decide_stage(-1, nullptr, g4x4, 2048, E, K, "rank 0 0000:03:00.0 63.0 GB/s, rank 1 ...");
    check("the line names the choice, the inputs and the threshold",
          c.line.find("OFF (automatic)") != std::string::npos && c.line.find("1.00") != std::string::npos &&
              c.line.find("1.50") != std::string::npos && c.line.find("2048 tokens") != std::string::npos &&
              c.line.find("top-10 of 512") != std::string::npos && c.line.find("7.9 GB/s") != std::string::npos &&
              c.line.find("rank 0 0000:03:00.0") != std::string::npos);
    std::printf("%s\n", c.line.c_str());
    std::printf("%s\n", exl3::decide_stage(-1, nullptr, g4x4, 4096, E, K, "x").line.c_str());
    std::printf("stage_test: %d failed\n", g_fail);
    return g_fail ? 1 : 0;
}
