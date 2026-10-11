/* twin_plugin.cpp -- TEST ONLY, never installed: this repository's qwen4exp_exl3 plugin under a second
 * plugin name and file, claiming the same ("qwen4exp", "exl3") key. The engine's loader must refuse
 * whichever of the two it meets second, naming both (tests/arch/duplicate_key.sh), because two
 * claimants of one key are a coin toss over which graph a model gets built from. */
#define RAD_ARCH_NO_EXPORTS 1
#include "../../architectures/qwen4exp_exl3/qwen4exp_exl3.cpp"

namespace qwen4exp_exl3_twin = qwen4exp_exl3;
RAD_ARCH_PROBE(qwen4exp_exl3_twin)
RAD_ARCH_PLUGIN(qwen4exp_exl3_twin, "qwen4exp", "exl3", "0.1.0",
                "TEST ONLY: a second claimant of (qwen4exp, exl3), which the loader must refuse")
