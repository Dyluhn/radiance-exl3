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

# (what it breaks, file, exact text, replacement, the case that must go red[, the binary: SPLIT for
# arch_static_test_split, the build at QWEN4EXP_EXL3_PREP_SPLIT 1; default arch_static_test, at 0])
SPLIT = "arch_static_test_split"
GD = "arch_static_test_gd"   # QWEN4EXP_EXL3_PREP_SPLIT 1 and QWEN4EXP_EXL3_GD 1 (interface v5)
LANE = "arch_static_test_lane"   # the GD build with QWEN4EXP_EXL3_SHARED_LANE 1 (the shared arm on the second lane)
MUTATIONS = [
    ("a deferred MoE pass still issuing its all-reduce", H,
     "if (own_ar && issues_ar(T, rows))", "if (issues_ar(T, rows))",
     "deferred_moe_reduction_keeps_the_pass_slice_and_wire"),
    ("an overlapping MoE reduction using the standalone row", H,
     "beside && op_ar_ov ? op_ar_ov : op_ar", "op_ar",
     "deferred_moe_reduction_keeps_the_pass_slice_and_wire"),
    ("the pipelined pass keeping its own collective", C,
     "l.mlp.pass(c, T, r0(p), rn(p), false);", "l.mlp.pass(c, T, r0(p), rn(p));",
     "prefill_pipeline_joins_each_piece_before_writing_the_residual"),
    ("the pipelined step dropping the routing report", C,
     "        l.mlp.route_report(c);", "        (void)0;",
     "prefill_pipeline_joins_each_piece_before_writing_the_residual"),
    ("the score plane staying step-tall on a spanned deployment", C,
     "const int64_t qsa_rows = (m.span_on && ctx->max_mixed_tok <= m.moecfg.rows)",
     "const int64_t qsa_rows = (false)",
     "prefill_pipeline_joins_each_piece_before_writing_the_residual"),
    ("the MoE requesting a fusion outside the loaded vocabulary", H,
     'topk_scatter_rows > 0 && rad_op_declared(b, "router_topk_scatter")',
     'topk_scatter_rows > 0', "missing_optional_schemas_use_the_unfused_chain"),
    ("the MTP gather failing to keep the shared rope table live", C,
     "if (m.op_rope_cs && rad_op_writes(b, m.mtp.op_gather, &m.b_rope_cs, 1) < 0)",
     "if (false)", "shape_probe_preserves_span_choices_and_mtp_rope_liveness"),
    ("the shared arm's lane never joined back before the gather", H,
     "    if (sh_lane) (void)rad_lane_join(c_, 1, 0);   /* the arm on lane 1 joins back before the gather reads it */\n",
     "\n", "decode_issues_gate_up_then_down_over_both_parity_tables", LANE),
    ("the shared arm on lane 1 not ordered behind the normed input", H,
     "        (void)rad_lane_join(c_, 0, 1);\n        (void)rad_lane(c_, 1);\n",
     "        (void)rad_lane(c_, 1);\n", "decode_issues_gate_up_then_down_over_both_parity_tables", LANE),
    ("the prefetch describes a plane two ops read (radiance 1.4.0 refuses such an expert)", H,
     "if (np < 4 && n_in[o[0]][o[1]] == 1)",
     "if (np < 4 && n_in[o[0]][o[1]] >= 1)",
     "the_next_layer_prefetch_is_described_and_called", SPLIT),
    ("the routed layers are not linked for the prefetch", C,
     "for (size_t l = 0; l + 1 < m.layers.size(); ++l) m.layers[l].mlp.xp_next = &m.layers[l + 1].mlp;",
     "(void)0;",
     "the_next_layer_prefetch_is_described_and_called"),
    ("the prefetch mark goes out before down is read", H,
     "    if (!gd) issue(op_dn, EXL3_DOWN, brows(w.eff, rows * k), brows(w.edn, rows * k));\n    /* This layer's streamed experts are read: the next one's copies cross the link from here. */\n    (void)rad_expert_prefetch_mark(c_);",
     "    (void)rad_expert_prefetch_mark(c_);\n    if (!gd) issue(op_dn, EXL3_DOWN, brows(w.eff, rows * k), brows(w.edn, rows * k));",
     "the_next_layer_prefetch_is_described_and_called"),

    ("an issue names the other parity class's run", H,
     "const int sq = exl3::kTestMutation == 1 ? q ^ 1 : q, tq = mut3(op, t) ? sq ^ 1 : sq;",
     "const int sq = q ^ 1, tq = mut3(op, t) ? sq ^ 1 : sq;",
     "decode_issues_gate_up_then_down_over_both_parity_tables"),
    ("down gathers the bf16 input instead of reading its sorted rows", H,
     "issue(op_dn, EXL3_DOWN, brows(w.eff, rows * k), brows(w.edn, rows * k));",
     "issue(op_dn, EXL3_DOWN, brow_slice(w.h.x, r0, rows, g.n_embd), brows(w.edn, rows * k));",
     "decode_issues_gate_up_then_down_over_both_parity_tables"),
    ("gate_up's N_odd from the even class", H, "        nq_gu[q] = 2 * wq[q];", "        nq_gu[q] = 2 * wq[0];",
     "tp2_splits_every_expert_384_256_by_parity"),
    ("gate_up_down's N_odd from the even class", H,
     'K", g.n_embd), RAD_INT("n_expert", n_reg),\n'
     '                                           RAD_INT("top_k", c.top_k), RAD_INT("N_odd", 2 * wq[1]),',
     'K", g.n_embd), RAD_INT("n_expert", n_reg),\n'
     '                                           RAD_INT("top_k", c.top_k), RAD_INT("N_odd", 2 * wq[0]),',
     "tp2_splits_every_expert_384_256_by_parity", GD),
    ("a span from the other parity", H, "RAD_ARCH_TRY(rad_weight_shard_span(b, h, qlo[q], qhi[q]));",
     "RAD_ARCH_TRY(rad_weight_shard_span(b, h, qlo[1 - q], qhi[1 - q]));",
     "tp2_splits_every_expert_384_256_by_parity"),
    ("gate_up's suh cut with the rows", H, "            cut  = !per_col;", "            cut  = true;",
     "tp2_splits_every_expert_384_256_by_parity"),
    ("the declared planes in another order", H,
     "tab.push_back(table_run(op, t)[(size_t)(mut3(op, t) ? s ^ 1 : s)]);",
     "tab.push_back(table_run(op, n_tab - 1 - t)[(size_t)(mut3(op, t) ? s ^ 1 : s)]);",
     "tp1_declares_exl3_experts_and_no_activation_quant"),
    ("the intermediate at bf16", C, "RAD_F16,     {ER, m.moecfg.n_ff_exp}", "g.act_dtype, {ER, m.moecfg.n_ff_exp}",
     "tp1_declares_exl3_experts_and_no_activation_quant"),
    ("one table accepts two encodings", H, "if (have[i] && rad_enc_equal(&enc[i], &first[i])) continue;",
     "if (have[i]) continue;", "a_second_trellis_width_in_one_table_is_refused"),
    ("stacking unchecked", H, "if (k[i] == exl3::kExl3 && !have[i] && x[i].stacked == (i == 0)) {",
     "if (k[i] == exl3::kExl3 && !have[i]) {", "an_unstacked_gate_up_is_refused"),
    # interface v6: four ranks rotate the extra block over four classes by number mod 4 (the _mod4 ops)
    ("four ranks back on the parity split", C, "if (g.world == 4 && exl3::kExtraRank < 0) {", "if (false) {",
     "tp4_rotates_the_extra_block_over_four_classes_by_number_mod_4"),
    ("four ranks' classes cut remainder-first (one rank takes every class's extra block)", C,
     "block_slice(nb, 4, (g.rank - q + 4) % 4,", "block_slice(nb, 4, g.rank,",
     "tp4_rotates_the_extra_block_over_four_classes_by_number_mod_4"),
    ("the block reads four classes as two", H, "    ncls = c.ff_hi4[0] > 0 ? 4 : 2;", "    ncls = 2;",
     "tp4_rotates_the_extra_block_over_four_classes_by_number_mod_4"),
    ("four classes declared under the two-class ops' names", H,
     "rad_decl_op(b, ncls == 4 ? exl3::kOpDown4 : exl3::kOpDown,", "rad_decl_op(b, exl3::kOpDown,",
     "tp4_rotates_the_extra_block_over_four_classes_by_number_mod_4"),
    ("a four-class class's K_q from class 0", H, "        kq_dn[q] = wq[q];", "        kq_dn[q] = wq[0];",
     "tp4_rotates_the_extra_block_over_four_classes_by_number_mod_4"),
    ("a four-class run strided by two", H,
     "        for (int64_t s = q; s < n_reg; s += ncls) {\n            const long long ge",
     "        for (int64_t s = q; s < n_reg; s += 2) {\n            const long long ge",
     "tp4_rotates_the_extra_block_over_four_classes_by_number_mod_4"),
    ("a four-class table issued at the even count", H,
     "RAD_WTAB(table_run(op, t)[(size_t)tq], class_n(tq));", "RAD_WTAB(table_run(op, t)[(size_t)tq], n_reg / ncls);",
     "tp4_takes_510_quantised_experts_as_uneven_classes"),
    ("the stage-off reader declared two-class at four classes", H,
     "op_stage_off = rad_decl_op(b, ncls == 4 ? exl3::kOpPrepGu4 : exl3::kOpPrepGu,",
     "op_stage_off = rad_decl_op(b, exl3::kOpPrepGu,",
     "tp4_rotates_the_extra_block_over_four_classes_by_number_mod_4", "arch_static_test_nostage"),
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
    ("dn_suh bound to gate_up's own suh run", H, "w_gu[t] : w_dn[exl3::kDownSuh])", "w_gu[t] : w_gu[1])",
     "tp1_declares_exl3_experts_and_no_activation_quant"),
    ("dn_suh bound to down's svh run", H, "kDownSuh = 1;", "kDownSuh = 2;",
     "tp1_declares_exl3_experts_and_no_activation_quant"),
    ("gate_up declared without its dn_suh tables", H,
     "return op == EXL3_GATE_UP ? exl3::kTablesGateUp", "return op == EXL3_GATE_UP ? exl3::kPlanesGateUp",
     "tp1_declares_exl3_experts_and_no_activation_quant"),
    ("a mixed pair: down without h_rot", H, 'dp.push_back(RAD_INT("h_rot", exl3::kHRot));',
     'dp.push_back(RAD_INT("h_rot", 0));', "tp1_declares_exl3_experts_and_no_activation_quant"),
    ("a mixed pair: gate_up without h_rot", H, 'gp.push_back(RAD_INT("h_rot", exl3::kHRot));',
     'gp.push_back(RAD_INT("h_rot", 0));', "tp1_declares_exl3_experts_and_no_activation_quant"),
    ("the A/B build's h_rot 0 left as the default", H, "#define QWEN4EXP_EXL3_H_ROT 1",
     "#define QWEN4EXP_EXL3_H_ROT 0", "tp1_declares_exl3_experts_and_no_activation_quant"),
    ("dn_suh issued from the other parity class", H,
     "tq = mut3(op, t) ? sq ^ 1 : sq;", "tq = t == exl3::kPlanesGateUp ? sq ^ 1 : sq;",
     "decode_issues_gate_up_then_down_over_both_parity_tables"),
    # v4 issue order (QWEN4EXP_EXL3_PREP_SPLIT 1): prep_gu, the shared arm, gate_up(a_rot = 1) over xr,
    # in the decode band only
    ("the split form issued above the band", H, "if (op_pg && rows <= prep_rows) {", "if (op_pg) {",
     "the_decode_band_ends_at_16_rows", SPLIT),
    ("the split build issuing the one gate_up at decode", H, "if (op_pg && rows <= prep_rows) {",
     "if (op_pg && rows < 0) {", "decode_issues_gate_up_then_down_over_both_parity_tables", SPLIT),
    ("the shared arm behind the routed GEMM in the split form", H,
     "        if (QX_SH_L0) shared_up(c_, T, r0, rows, rfused && router_fused_gate);\n"
     "        if (QX_SH_L0) gq_sh.step(c_, w.sgu, T, r0, rows);\n"
     "        issue(op_gu_ar, EXL3_GATE_UP, brows(b_xr, rows * k), brows(w.eff, rows * k));\n",
     "        issue(op_gu_ar, EXL3_GATE_UP, brows(b_xr, rows * k), brows(w.eff, rows * k));\n"
     "        if (QX_SH_L0) shared_up(c_, T, r0, rows, rfused && router_fused_gate);\n"
     "        if (QX_SH_L0) gq_sh.step(c_, w.sgu, T, r0, rows);\n",
     "decode_issues_gate_up_then_down_over_both_parity_tables", SPLIT),
    ("gate_up(a_rot = 1) reading the token input, not xr", H,
     "issue(op_gu_ar, EXL3_GATE_UP, brows(b_xr, rows * k),",
     "issue(op_gu_ar, EXL3_GATE_UP, brow_slice(w.h.x, r0, rows, g.n_embd),",
     "decode_issues_gate_up_then_down_over_both_parity_tables", SPLIT),
    ("prep_gu's tables shifted onto suh.1 and svh", H, ": op == EXL3_PREP_GU ? w_gu[1 + t] : w_dn[t];",
     ": op == EXL3_PREP_GU ? w_gu[2 + t] : w_dn[t];", "tp1_declares_exl3_experts_and_no_activation_quant", SPLIT),
    ("the a_rot = 1 gate_up ranged over every M", H, "op_gu_ar = decl_gu(prep_rows, 1, b_xr);",
     "op_gu_ar = decl_gu(c.rows, 1, b_xr);", "tp1_declares_exl3_experts_and_no_activation_quant", SPLIT),
    ("the decode-band gate_up declared at a_rot 0", H, "op_gu_ar = decl_gu(prep_rows, 1, b_xr);",
     "op_gu_ar = decl_gu(prep_rows, 0, b_xr);", "tp1_declares_exl3_experts_and_no_activation_quant", SPLIT),
    ("xr one rotation wide", H, "{prep_rows * c.top_k, 2 * g.n_embd});", "{prep_rows * c.top_k, g.n_embd});",
     "tp1_declares_exl3_experts_and_no_activation_quant", SPLIT),
    ("prep_gu ranged over every M", H, 'std::vector<RadParam> pp = { RAD_RANGE("M", 1, prep_rows), RAD_INT("K", g.n_embd),',
     'std::vector<RadParam> pp = { RAD_RANGE("M", 1, c.rows), RAD_INT("K", g.n_embd),',
     "tp1_declares_exl3_experts_and_no_activation_quant", SPLIT),
    # the engine's stagers key a routed layer on its first declared expert op (stager.cpp:23-31)
    ("the decode-band pair declared before the one gate_up (P6's order-1 prefill)", H,
     "        op_gu = decl_gu(c.rows, 0, w.h.x);\n        RAD_ARCH_TRY(declare_decode_band());\n",
     "        RAD_ARCH_TRY(declare_decode_band());\n        op_gu = decl_gu(c.rows, 0, w.h.x);\n",
     "the_one_gate_up_is_every_layers_first_declared_expert_reader", SPLIT),
    ("the file tier keeping the split decode form", H,
     "band_split = exl3::kPrepSplit && !file_tier;", "band_split = exl3::kPrepSplit;",
     "the_file_tier_takes_the_one_gate_up", SPLIT),
    ("the file tier keeping gate_up_down", H,
     "band_gd    = exl3::kGd && !file_tier;", "band_gd    = exl3::kGd;",
     "the_file_tier_takes_the_one_gate_up", GD),
    ("gate_up_down issued above the band", H, "const bool gd = op_gd && rows >= exl3::kGdMinRows && rows <= gd_rows;",
     "const bool gd = op_gd && rows >= exl3::kGdMinRows;", "the_decode_band_ends_at_16_rows", GD),
    ("gate_up_down issued below its rows", H, "const bool gd = op_gd && rows >= exl3::kGdMinRows && rows <= gd_rows;",
     "const bool gd = op_gd && rows <= gd_rows;", "decode_issues_gate_up_then_down_over_both_parity_tables", GD),
    ("gate_up_down never issued", H, "const bool gd = op_gd && rows >= exl3::kGdMinRows && rows <= gd_rows;",
     "const bool gd = false;", "decode_issues_gate_up_then_down_over_both_parity_tables", GD),
    ("gate_up_down ranged from 1 row", H, 'RAD_PARAMS(RAD_RANGE("M", exl3::kGdDeclMin, gd_rows)',
     'RAD_PARAMS(RAD_RANGE("M", 1, gd_rows)', "tp1_declares_exl3_experts_and_no_activation_quant", GD),
    ("gate_up_down's down tables swapped (svh as codes)", H,
     "return w_dn[t == exl3::kTablesGateUp ? exl3::kDownCodes : exl3::kDownSvh];",
     "return w_dn[t == exl3::kTablesGateUp ? exl3::kDownSvh : exl3::kDownCodes];",
     "tp1_declares_exl3_experts_and_no_activation_quant", GD),
    ("gate_up_down declared over halves of two encodings", H,
     "!std::strcmp(eenc[0].dtype, eenc[1].dtype) && !std::strcmp(eenc[0].cb, eenc[1].cb) &&\n"
     "        c.rows >= exl3::kGdDeclMin) {",
     "true &&\n        c.rows >= exl3::kGdDeclMin) {", "a_layer_may_carry_its_own_trellis_width", GD),
    ("the shared arm behind gate_up_down", H,
     "        if (QX_SH_L0) gq_sh.step(c_, w.sgu, T, r0, rows);\n"
     "        if (QX_SH_L0 && exl3::kGdOrder == 1) sh_down.step(c_, w.sff, w.sout, T, r0, rows);   /* beside the launch */\n"
     "        issue(op_gd, EXL3_GU_DN, brow_slice(w.h.x, r0, rows, g.n_embd), brows(w.edn, rows * k));\n",
     "        if (QX_SH_L0 && exl3::kGdOrder == 1) sh_down.step(c_, w.sff, w.sout, T, r0, rows);   /* beside the launch */\n"
     "        issue(op_gd, EXL3_GU_DN, brow_slice(w.h.x, r0, rows, g.n_embd), brows(w.edn, rows * k));\n"
     "        if (QX_SH_L0) gq_sh.step(c_, w.sgu, T, r0, rows);\n",
     "decode_issues_gate_up_then_down_over_both_parity_tables", GD),
    # interface v7: gate_up_down_mod4 at four ranks, over the same rows as the two-class op
    ("four ranks without gate_up_down_mod4", H, "if (band_gd && (ncls == 2 || ncls == 4) && !c.calib",
     "if (band_gd && ncls == 2 && !c.calib", "tp4_rotates_the_extra_block_over_four_classes_by_number_mod_4", GD),
    ("gate_up_down_mod4 declared under the two-class op's name", H,
     "op_gd = rad_decl_op(b, exl3::kOpGateUpDown4, gdp.data(),", "op_gd = rad_decl_op(b, exl3::kOpGateUpDown, gdp.data(),",
     "tp4_rotates_the_extra_block_over_four_classes_by_number_mod_4", GD),
    ("gate_up_down_mod4 without its classes' N_q / K_q", H, "            gdp.insert(gdp.end(), cls_p.begin(), cls_p.end());\n",
     "", "tp4_rotates_the_extra_block_over_four_classes_by_number_mod_4", GD),
    ("gate_up_down_mod4 ranged from 1 row", H, 'std::vector<RadParam> gdp = { RAD_RANGE("M", exl3::kGdDeclMin, gd_rows),',
     'std::vector<RadParam> gdp = { RAD_RANGE("M", 1, gd_rows),',
     "tp4_rotates_the_extra_block_over_four_classes_by_number_mod_4", GD),
    ("gate_up_down_mod4 issued below its rows", H,
     "const bool gd = op_gd && rows >= exl3::kGdMinRows && rows <= gd_rows;", "const bool gd = op_gd && rows <= gd_rows;",
     "tp4_rotates_the_extra_block_over_four_classes_by_number_mod_4", GD),
    ("gate_up_down_mod4 never issued", H, "const bool gd = op_gd && rows >= exl3::kGdMinRows && rows <= gd_rows;",
     "const bool gd = false;", "tp4_rotates_the_extra_block_over_four_classes_by_number_mod_4", GD),
    ("the shipped default flipped back to order 0", H, "#define QWEN4EXP_EXL3_PREP_SPLIT 1",
     "#define QWEN4EXP_EXL3_PREP_SPLIT 0", "the_shipped_default_is_the_split_order", "arch_static_test_default"),
    ("prefill staging a build constant again (the define back at 1, as before the automatic decision)", H,
     "#define QWEN4EXP_EXL3_PREFILL_STAGE -1", "#define QWEN4EXP_EXL3_PREFILL_STAGE 1",
     "the_shipped_default_is_the_split_order", "arch_static_test_default"),
    ("the GD issue order back at 0 by default", H, "#define QWEN4EXP_EXL3_GD_ORDER 1", "#define QWEN4EXP_EXL3_GD_ORDER 0",
     "the_shipped_default_is_the_split_order", "arch_static_test_default"),
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
    bad = 0
    for what, f, old, new, case, *binary in MUTATIONS:
        target = binary[0] if binary else "arch_static_test"
        test = build / target
        path = REPO / f
        src = path.read_text()
        if src.count(old) != 1:
            print(f"NOT APPLIED  {what}: its text occurs {src.count(old)} times -- update the mutation")
            bad += 1
            continue
        path.write_text(src.replace(old, new))
        try:
            b = run(["cmake", "--build", str(build), "--target", target])
            if b.returncode != 0:
                print(f"NOT BUILT    {what}")
                bad += 1
                continue
            r = run([str(test), case])
            caught = r.returncode == 1
            bad += not caught
            print(f"{'caught' if caught else 'MISSED (exit %d)' % r.returncode:12} {what} -> {case}"
                  f"{' [split]' if target == SPLIT else ' [gd]' if target == GD else ''}")
        finally:
            run(["git", "checkout", "--", f])
    for target in ("arch_static_test", SPLIT, GD, "arch_static_test_default", "arch_static_test_nostage"):
        run(["cmake", "--build", str(build), "--target", target])
    print(f"RESULT: {len(MUTATIONS) - bad} of {len(MUTATIONS)} caught")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
