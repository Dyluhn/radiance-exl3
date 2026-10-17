/* ABI 17 and prefill-span regressions, included by arch_static_test.cpp inside its case namespace. */
void missing_optional_schemas_use_the_unfused_chain() {
    RadModelMeta m = meta();
    RadBuildCtx c = ctx(0, 1);
    RadBuilder b;
    CHECK_EQ(rad_op_declared(nullptr, "router_topk_scatter"), 0);
    CHECK_EQ(rad_op_declared(&b, nullptr), 0);
    CHECK_EQ(rad_op_declared(&b, "unknown_op"), 0);
    CHECK_EQ(rad_op_declared(&b, "router_topk_scatter"), 1);  /* no model instance yet */
    b.schemas.clear();
    exl3_model(b);
    REQUIRE_EQ(qwen4exp_exl3::declare(&b, &m, &c), RAD_OK);
    const auto& mod = qwen4exp_exl3::g_model[0];
    CHECK_EQ(count_op(b, "router_topk_scatter"), 0);
    CHECK_EQ(count_op(b, "router_gemm_topk_scatter"), 0);
    CHECK_EQ(count_op(b, "qk_norm_rope_gate"), 0);
    CHECK_EQ(count_op(b, "qk_norm_rope_gate_kv_store"), 0);
    CHECK_EQ(count_op(b, "gate_quant_fp8"), 0);
    CHECK_EQ(count_op(b, "attn_paged_gate_quant"), 0);
    RadCtx cx;
    const auto& moe = mod.layers[0].mlp;
    moe.pass(&cx, 1, 0, 1);
    CHECK(std::any_of(cx.issues.begin(), cx.issues.end(), [&](const RecIssue& i) { return i.op == moe.op_router; }));
    CHECK(std::any_of(cx.issues.begin(), cx.issues.end(), [&](const RecIssue& i) { return i.op == moe.op_topk; }));
    CHECK(std::any_of(cx.issues.begin(), cx.issues.end(), [&](const RecIssue& i) { return i.op == moe.op_scatter; }));
    CHECK(cx.step_fail.empty());
}

void deferred_moe_reduction_keeps_the_pass_slice_and_wire() {
    RadModelMeta m = meta();
    RadBuildCtx c = ctx(0, 3);
    c.max_tok = 4096; c.max_mixed_tok = 64;
    RadBuilder b;
    exl3_model(b);
    REQUIRE_EQ(qwen4exp_exl3::declare(&b, &m, &c), RAD_OK);
    const auto& moe = qwen4exp_exl3::g_model[0].layers[0].mlp;
    CHECK(moe.issues_ar(4096, 17));
    REQUIRE_EQ(moe.op_ar_ov > 0, true);
    CHECK_EQ(param_i(b.ops[moe.op_ar_ov - 1], "overlap"), 1);
    CHECK_EQ(param_i(b.ops[moe.op_ar - 1], "overlap"), -1);
    RadCtx cx;
    moe.pass(&cx, 4096, 2048, 17, false);
    for (const RecIssue& i : cx.issues) CHECK(b.ops[i.op - 1].op != "all_reduce");
    rad_lane(&cx, 1);
    moe.issue_ar(&cx, 2048, 17);
    const RecIssue& ar = cx.issues.back();
    CHECK_EQ(ar.op, moe.op_ar_ov);
    CHECK_EQ(ar.lane, 1);
    CHECK_EQ(ar.n, 17 * kEmbd);
    CHECK_EQ(ar.opd[0].handle, moe.w.h.x);
    CHECK_EQ(ar.opd[0].offset, 2048 * kEmbd);
    CHECK_EQ(ar.opd[0].rows, 17);
    moe.issue_ar(&cx, 2048, 17, false);
    CHECK_EQ(cx.issues.back().op, moe.op_ar);
    RadCtx serial;
    moe.pass(&serial, 4096, 2048, 17);
    CHECK_EQ(serial.issues.back().op, moe.op_ar);
    CHECK_EQ(serial.issues.back().lane, 0);
    moe.route_report(&serial);
    CHECK_EQ(serial.route_layers.size(), 1);
    CHECK_EQ(serial.route_layers[0], 0);
}

void prefill_pipeline_joins_each_piece_before_writing_the_residual() {
    RadModelMeta m = meta();
    RadBuildCtx c = ctx(0, 4);
    c.max_tok = 4096; c.max_mixed_tok = 64;
    RadBuilder b;
    exl3_model(b);
    REQUIRE_EQ(qwen4exp_exl3::declare(&b, &m, &c), RAD_OK);
    auto& mod = qwen4exp_exl3::g_model[0];
    CHECK(mod.span_on && mod.pipe_on);
    CHECK(b.concurrent.count(mod.a_x.x));
    CHECK(qwen4exp_exl3::pipelined(mod, 4096));
    CHECK(!qwen4exp_exl3::pipelined(mod, 2048));
    CHECK_EQ(buf_named(b, "qsa_score")->shape[0], 2048);
    CHECK_EQ(param_hi(b.ops[mod.layers[0].hc_ffn.op_read - 1], "M"), 2048);
    /* A pinned bounds copy is already established; the fake device records the two blits.
     * No allocation or device execution is needed to inspect this step's issues. */
    int32_t bounds[] = {0, 2048};
    mod.pipe_cu_tried = true; mod.pipe_cu_h = bounds;
    int32_t cu[] = {0, 4096}, context[] = {4096}, state[] = {0, 1};
    std::vector<int32_t> tokens(4096), positions(4096), table(1024);
    RadKVGroupBatch kv{};
    kv.group = 1; kv.block_table = table.data(); kv.block_table_pitch = 1024;
    kv.state_index = state; kv.state_index_pitch = 2;
    RadBatch batch{};
    batch.phase = RAD_PHASE_PREFILL; batch.n_tok = 4096; batch.n_seq = 1;
    batch.max_q_len = 4096; batch.max_ctx_len = 8192;
    batch.token_ids = tokens.data(); batch.positions = positions.data();
    batch.cu_seqlens = cu; batch.ctx_lens = context;
    batch.n_kv_groups = 1; batch.kv = &kv;
    RadCtx cx;
    qwen4exp_exl3::step(&cx, &batch);
    CHECK(cx.step_fail.empty());
    CHECK_EQ(cx.lane, 0);
    CHECK_EQ(g_device_calls, 2);
    CHECK_EQ(cx.route_layers.size(), kLayers);
    int reductions = 0, scores = 0;
    for (const RecIssue& i : cx.issues) {
        const RecOp& o = b.ops[i.op - 1];
        if (o.op == "all_reduce") {
            ++reductions;
            CHECK_EQ(i.lane, 1);
            if (i.op != mod.op_embed_ar) {
                CHECK_EQ(param_i(o, "overlap"), 1);
                CHECK_EQ(i.n, 2048 * kEmbd);
            }
        }
        if (o.op == "qsa_score") { ++scores; CHECK_EQ(i.n, 2048); }
    }
    CHECK_EQ(reductions, 33);  /* embedding plus two pieces of mix and MoE at eight layers */
    CHECK_EQ(scores, 4);       /* two windows at each of two attention layers */
    for (const auto& layer : mod.layers) {
        const auto& moe = layer.mlp;
        int gathers = 0, reductions_moe = 0, writes = 0;
        size_t last_gather = 0;
        std::map<int64_t, size_t> reduced;
        for (size_t at = 0; at < cx.issues.size(); ++at) {
            const RecIssue& i = cx.issues[at];
            if (i.op == moe.op_gather) { ++gathers; last_gather = at; CHECK_EQ(i.lane, 0); }
            if (i.op == moe.op_ar_ov) {
                ++reductions_moe;
                CHECK(at > last_gather);
                reduced[i.opd[0].offset] = at;
            }
            if (i.op == layer.hc_ffn.op_write) {
                ++writes;
                const int64_t offset = i.opd[0].offset;  /* x is the narrow block delta */
                REQUIRE_EQ(reduced.count(offset), 1);
                CHECK(std::any_of(cx.joins.begin(), cx.joins.end(), [&](const RadCtx::LaneJoin& j) {
                    return j.from == 1 && j.to == 0 && j.at > reduced[offset] && j.at <= at;
                }));
            }
        }
        CHECK_EQ(gathers, 2); CHECK_EQ(reductions_moe, 2); CHECK_EQ(writes, 2);
    }
    mod.pipe_cu_h = nullptr;
}

void shape_probe_preserves_span_choices_and_mtp_rope_liveness() {
    RadModelMeta m = meta();
    RadBuildCtx c = ctx(0, 4);
    c.max_tok = 4096; c.max_mixed_tok = 64;
    RadBuilder b;
    exl3_model(b);
    REQUIRE_EQ(qwen4exp_exl3::declare(&b, &m, &c), RAD_OK);
    const auto& mod = qwen4exp_exl3::g_model[0];
    const auto& gather = b.ops[mod.mtp.op_gather - 1];
    CHECK(std::find(gather.writes.begin(), gather.writes.end(), mod.b_rope_cs) != gather.writes.end());
    c.shape_probe = 1; c.max_tok = 1024; c.max_mixed_tok = 0;
    RadBuilder small;
    exl3_model(small);
    REQUIRE_EQ(qwen4exp_exl3::declare(&small, &m, &c), RAD_OK);
    CHECK(mod.span_on && mod.pipe_on);
    CHECK_EQ(mod.g.max_tok, 4096);
    CHECK_EQ(buf_named(small, "qsa_score")->shape[0], 1024);
    rad_buf x = 0;
    for (size_t i = 0; i < small.bufs.size(); ++i) if (small.bufs[i].first == "x") x = (rad_buf)i + 1;
    CHECK(x && small.concurrent.count(x));
    CHECK_EQ(param_hi(*ops_named(small, "hc_read")[0], "M"), 1024);
}

void mixed_steps_keep_decode_rows_off_the_chunked_state_path() {
    RadModelMeta m = meta();
    RadBuildCtx c = ctx(0, 1);
    RadBuilder b;
    exl3_model(b);
    REQUIRE_EQ(qwen4exp_exl3::declare(&b, &m, &c), RAD_OK);
    const auto& gdn = qwen4exp_exl3::g_model[0].layers[0].gdn;
    int32_t cu[] = {0, 1, 9}, accepted[] = {1, 0}, context[] = {10, 0};
    RadBatch batch{};
    batch.phase = RAD_PHASE_MIXED; batch.n_tok = 9; batch.n_seq = 2;
    batch.n_seq_decode = 1; batch.n_tok_decode = 1;
    batch.max_q_len = 8; batch.max_q_len_decode = 1;
    batch.cu_seqlens = cu; batch.num_accepted = accepted; batch.ctx_lens = context;
    RadCtx cx;
    gdn.step(&cx, &batch);
    int decode = 0, prefill = 0;
    for (const RecIssue& i : cx.issues) {
        if (i.op == gdn.op_cr || i.op == gdn.op_recur) {
            ++decode;
            CHECK_EQ(i.n, 1);
            CHECK(std::any_of(i.opd.begin(), i.opd.end(), [&](const RadOperand& o) { return o.raw == accepted && o.rows == 1; }));
            CHECK(std::any_of(i.opd.begin(), i.opd.end(), [&](const RadOperand& o) { return o.raw == cu && o.rows == 2; }));
        }
        if (i.op == gdn.op_scan) {
            ++prefill;
            CHECK(std::any_of(i.opd.begin(), i.opd.end(), [&](const RadOperand& o) { return o.raw == cu + 1 && o.rows == 2; }));
            CHECK(std::none_of(i.opd.begin(), i.opd.end(), [&](const RadOperand& o) { return o.raw == accepted; }));
        }
    }
    CHECK_EQ(decode, 1); CHECK_EQ(prefill, 1);
    CHECK(cx.step_fail.empty());
    CHECK_EQ(g_device_calls, 0);
}
