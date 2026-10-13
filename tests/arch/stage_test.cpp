// stage_test.cpp -- exl3_stage.h's automatic prefill-staging decision from its inputs alone (no radiance, no device):
// the threshold the rule derives, which links stage at the step sizes the composes use, the overrides, and the line.
// One line a case; exit 0 when every case holds.
#include "../../architectures/qwen4exp_exl3/exl3_stage.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace exl3 = rad::arch::exl3;
static int g_fail = 0;
static void check(const char* what, bool ok) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_fail;
}

int main() {
    const double g5x16 = 32.0 * 16 * 128 / 130 / 8, g4x16 = 16.0 * 16 * 128 / 130 / 8, g5x8 = g4x16,
                 g4x8 = 16.0 * 8 * 128 / 130 / 8, g4x4 = 16.0 * 4 * 128 / 130 / 8;
    const exl3::StageRule r;
    check("threshold at 2048 tokens a step: 256 MB / (0.85 x 11 ms) = 27.4 GB/s",
          std::fabs(r.threshold_gbps(2048) - 27.38) < 0.05);
    check("threshold halves at 4096 tokens a step (a layer's prefill takes twice as long)",
          std::fabs(r.threshold_gbps(4096) - r.threshold_gbps(2048) / 2) < 1e-9);
    auto auto_at = [&](double gbps, long long tok) { return exl3::decide_stage(-1, nullptr, gbps, tok, "x").stage; };
    check("2048 a step: Gen5 x16 and Gen4 x16 / Gen5 x8 stage", auto_at(g5x16, 2048) && auto_at(g4x16, 2048) && auto_at(g5x8, 2048));
    check("2048 a step: Gen4 x8 and Gen4 x4 do not", !auto_at(g4x8, 2048) && !auto_at(g4x4, 2048));
    check("the development box at tp2 (an x4 rank, 7.9 GB/s): off -- as measured (2.2x)", !auto_at(g4x4, 2048));
    check("4096 a step: Gen4 x8 stages (15.8 >= 13.7), Gen4 x4 still not", auto_at(g4x8, 4096) && !auto_at(g4x4, 4096));
    check("an unknown link keeps radiance's default (on)", auto_at(EXL3_LINK_UNKNOWN, 2048));
    check("EXL3_PREFILL_STAGE=on stages over a slow link", exl3::decide_stage(-1, "on", g4x4, 2048, "x").stage);
    check("EXL3_PREFILL_STAGE=off does not over a fast one", !exl3::decide_stage(-1, "off", g5x16, 2048, "x").stage);
    check("EXL3_PREFILL_STAGE=auto is the rule", !exl3::decide_stage(-1, "auto", g4x4, 2048, "x").stage);
    const exl3::StageChoice junk = exl3::decide_stage(-1, "maybe", g4x4, 2048, "x");
    check("an unreadable EXL3_PREFILL_STAGE is ignored and said so",
          !junk.stage && junk.line.find("was ignored") != std::string::npos);
    check("a forced build wins over the environment", exl3::decide_stage(1, "off", g4x4, 2048, "x").stage &&
                                                          !exl3::decide_stage(0, "on", g5x16, 2048, "x").stage);
    const exl3::StageChoice c = exl3::decide_stage(-1, nullptr, g4x4, 2048, "rank 0 0000:03:00.0 63.0 GB/s, rank 1 ...");
    check("the line names the choice, the inputs and the threshold",
          c.line.find("OFF (automatic)") != std::string::npos && c.line.find("7.9 GB/s") != std::string::npos &&
              c.line.find("27.4 GB/s") != std::string::npos && c.line.find("2048 tokens") != std::string::npos &&
              c.line.find("rank 0 0000:03:00.0") != std::string::npos);
    std::printf("%s\n", c.line.c_str());
    std::printf("stage_test: %d failed\n", g_fail);
    return g_fail ? 1 : 0;
}
