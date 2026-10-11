#!/usr/bin/env python3
"""mutation_check.py -- arch_static_test's negative controls: each mutation below breaks one property
of the plugin's source, the test binary is rebuilt, and the case that guards the property must FAIL
(exit 1, not a crash). Then the file is restored from git.

    tests/arch/mutation_check.py BUILD_DIR      (tests/arch's build directory; run from anywhere)

EXIT 0 every mutation caught; 1 one was not (or did not apply, or did not build); 2 the plugin's files
have uncommitted changes -- the restore is `git checkout`, which would lose them, so it refuses.
Minutes: one rebuild a mutation. Run it after changing the plugin or the test; a mutation whose
pattern no longer matches is reported, never skipped silently.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
H = "architectures/qwen4exp_exl3/rad_block_moe_exl3.h"
C = "architectures/qwen4exp_exl3/qwen4exp_exl3.cpp"

# (what it breaks, file, exact text, replacement, the case that must go red)
MUTATIONS = [
    ("an issue names the other parity class's run", H,
     "                const int sq = exl3::kTestMutation == 1 ? 1 - q : q;\n                opd[",
     "                const int sq = 1 - q;\n                opd[",
     "decode_issues_gate_up_then_down_over_both_parity_tables"),
    ("down gathers the bf16 input instead of reading its sorted rows", H,
     "issue(op_dn, 0, brows(w.eff, rows * k), brows(w.edn, rows * k));",
     "issue(op_dn, 0, brow_slice(w.h.x, r0, rows, g.n_embd), brows(w.edn, rows * k));",
     "decode_issues_gate_up_then_down_over_both_parity_tables"),
    ("gate_up's N_odd from the even class", H, 'RAD_INT("N_odd", 2 * wq[1])', 'RAD_INT("N_odd", 2 * wq[0])',
     "tp2_splits_every_expert_384_256_by_parity"),
    ("a span from the other parity", H, "RAD_ARCH_TRY(rad_weight_shard_span(b, h, qlo[q], qhi[q]));",
     "RAD_ARCH_TRY(rad_weight_shard_span(b, h, qlo[1 - q], qhi[1 - q]));",
     "tp2_splits_every_expert_384_256_by_parity"),
    ("gate_up's suh cut with the rows", H, "            cut  = !per_col;", "            cut  = true;",
     "tp2_splits_every_expert_384_256_by_parity"),
    ("the declared planes in another order", H,
     "tab.push_back(table_run(gate_up, t)[(size_t)(mut3(gate_up, t) ? s ^ 1 : s)]);",
     "tab.push_back(table_run(gate_up, n_tab - 1 - t)[(size_t)(mut3(gate_up, t) ? s ^ 1 : s)]);",
     "tp1_declares_exl3_experts_and_no_activation_quant"),
    ("the intermediate at bf16", C, "RAD_F16,     {ER, m.moecfg.n_ff_exp}", "g.act_dtype, {ER, m.moecfg.n_ff_exp}",
     "tp1_declares_exl3_experts_and_no_activation_quant"),
    ("one table accepts two encodings", H, "if (have[i] && rad_enc_equal(&enc[i], &first[i])) continue;",
     "if (have[i]) continue;", "a_second_trellis_width_in_one_table_is_refused"),
    ("stacking unchecked", H, "if (k[i] == exl3::kExl3 && !have[i] && x[i].stacked == (i == 0)) {",
     "if (k[i] == exl3::kExl3 && !have[i]) {", "an_unstacked_gate_up_is_refused"),
    ("the four-class split accepted", H, "if (c.ff_hi4[0] > 0) {", "if (false && c.ff_hi4[0] > 0) {",
     "four_ranks_are_refused_by_name"),
    ("the in-tree codes_i8 rule", C, "m.hccfg.codes_i8 = m.trunk_i8 ? 1 : 0;",
     "m.hccfg.codes_i8 = (m.trunk_i8 && m.moecfg.expert_rot) ? 1 : 0;",
     "an_int8_trunk_has_its_connections_write_int8_codes"),
    ("w_bf16 not taken from the trunk", C, "            g.w_bf16 = plain;", "            g.w_bf16 = false;",
     "a_bf16_trunk_is_served_bf16_throughout"),
    ("an imported quantisation accepted", C,
     'if (const char* imp = rad_meta_gets(meta, "imported_quantization_config.quant_method", nullptr)) {',
     "if (const char* imp = (const char*)nullptr) {", "an_imported_quantisation_is_refused_by_name"),
    ("the down tap on the f16 plane", H, "{b_calib_eff, b_gram_dn}, {b_gram_dn});", "{w.eff, b_gram_dn}, {b_gram_dn});",
     "calibration_taps_the_inputs_the_ops_are_handed"),
    ("protected experts not split out", H,
     "if (k[0] == exl3::kPlainBf16 && k[1] == exl3::kPlainBf16) { prot.push_back(e); continue; }", "",
     "protected_bf16_experts_keep_their_own_pair"),
    ("a model-less probe accepted", H, '    if (!known) {\n        fprintf(stderr, "radiance: nothing is known',
     '    if (!known) return RAD_OK;\n    if (!known) {\n        fprintf(stderr, "radiance: nothing is known',
     "a_builder_with_no_model_is_refused"),
    # EPI 1 (interface v3): gate_up binds down's suh run as dn_suh, both ops carry h_rot = 1
    ("dn_suh bound to gate_up's own suh run", H, "w_dn[exl3::kDownSuh]) : w_dn[t];", "w_gu[1]) : w_dn[t];",
     "tp1_declares_exl3_experts_and_no_activation_quant"),
    ("dn_suh bound to down's svh run", H, "kDownSuh = 1;", "kDownSuh = 2;",
     "tp1_declares_exl3_experts_and_no_activation_quant"),
    ("gate_up declared without its dn_suh tables", H,
     "const int n_tab = gate_up ? exl3::kTablesGateUp : exl3::kPlanesDown;",
     "const int n_tab = gate_up ? exl3::kPlanesGateUp : exl3::kPlanesDown;",
     "tp1_declares_exl3_experts_and_no_activation_quant"),
    ("a mixed pair: down without h_rot", H, 'RAD_STR("codebook", eenc[1].cb), RAD_INT("h_rot", exl3::kHRot)',
     'RAD_STR("codebook", eenc[1].cb), RAD_INT("h_rot", 0)', "tp1_declares_exl3_experts_and_no_activation_quant"),
    ("a mixed pair: gate_up without h_rot", H, 'RAD_STR("codebook", eenc[0].cb), RAD_INT("h_rot", exl3::kHRot)',
     'RAD_STR("codebook", eenc[0].cb), RAD_INT("h_rot", 0)', "tp1_declares_exl3_experts_and_no_activation_quant"),
    ("the A/B build's h_rot 0 left as the default", H, "#define QWEN4EXP_EXL3_H_ROT 1",
     "#define QWEN4EXP_EXL3_H_ROT 0", "tp1_declares_exl3_experts_and_no_activation_quant"),
    ("dn_suh issued from the other parity class", H,
     "RAD_WTAB(table_run(gate_up, t)[(size_t)(mut3(gate_up, t) ? 1 - sq : sq)], nh);",
     "RAD_WTAB(table_run(gate_up, t)[(size_t)(t == exl3::kPlanesGateUp ? 1 - sq : sq)], nh);",
     "decode_issues_gate_up_then_down_over_both_parity_tables"),
    ("a test mutation left on in a real build", H, "#define QWEN4EXP_EXL3_TEST_MUTATION 0",
     "#define QWEN4EXP_EXL3_TEST_MUTATION 1", "tp1_declares_exl3_experts_and_no_activation_quant"),
]


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, **kw)


def main(argv):
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    build = Path(argv[0]).resolve()
    dirty = run(["git", "status", "--porcelain", H, C]).stdout.strip()
    if dirty:
        print(f"refusing: uncommitted changes in\n{dirty}\nthe restore is git checkout")
        return 2
    test = build / "arch_static_test"
    bad = 0
    for what, f, old, new, case in MUTATIONS:
        path = REPO / f
        src = path.read_text()
        if src.count(old) != 1:
            print(f"NOT APPLIED  {what}: its text occurs {src.count(old)} times -- update the mutation")
            bad += 1
            continue
        path.write_text(src.replace(old, new))
        try:
            b = run(["cmake", "--build", str(build), "--target", "arch_static_test"])
            if b.returncode != 0:
                print(f"NOT BUILT    {what}")
                bad += 1
                continue
            r = run([str(test), case])
            caught = r.returncode == 1
            bad += not caught
            print(f"{'caught' if caught else 'MISSED (exit %d)' % r.returncode:12} {what} -> {case}")
        finally:
            run(["git", "checkout", "--", f])
    run(["cmake", "--build", str(build), "--target", "arch_static_test"])
    print(f"RESULT: {len(MUTATIONS) - bad} of {len(MUTATIONS)} caught")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
