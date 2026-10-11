/* arch_static_test.cpp -- the qwen4exp_exl3 plugin's declared graph and its MoE block's issues,
 * checked with no GPU, no model file and no radiance core.
 *
 * The plugin source is #included with its exports suppressed, against a RECORDING builder and
 * context of this file's own: every weight, buffer, op and issue is kept, and rad_weight_encoding
 * answers from a table the case sets. That is radiance's own tests/arch_test.cpp pattern (and
 * RidgeFill's tests/rad_fake.h), written again here because a plugin that only works against one
 * builder has an undeclared dependency.
 *
 * WHAT IT CAN AND CANNOT SEE. It sees what this plugin declares and issues: the ops, their
 * parameters and weight lists, the per-rank widths and spans, the refusals and their messages. It
 * cannot see whether a kernel serves the ops or computes them right -- that is the fixture run
 * through exl3ref (P3 notes) -- and the EXL3 encoding here is the plan's D4 shape, not P1's builder.
 *
 * Each case is its own process (ctest runs `arch_static_test <case>`): the calibration tap reads
 * RADIANCE_CALIB_DIR once per process, and a case must not inherit another's state.
 */
#define RAD_ARCH_NO_EXPORTS 1

#include <rad_builder.h>
#include <rad_device.h>
#include <rad_runtime.h>

#include <unistd.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <set>
#include <string>
#include <utility>
#include <vector>

/* ==================================================================== the recording fakes */
struct RecParam { std::string key; int kind = 0; long long ival = 0, ihi = 0; std::string sval; };
struct RecOp {
    std::string             op;
    std::vector<RecParam>   p;
    std::vector<rad_weight> w;
    std::vector<rad_buf>    reads, writes;
};
struct RecSpan  { rad_weight w = 0; int64_t lo = 0, hi = 0; };
struct RecIssue { rad_op op = 0; std::vector<RadOperand> opd; int64_t n = 0; };

struct RadBuilder {
    std::vector<std::pair<std::string, RadWeightDecl>> weights;
    std::vector<std::pair<std::string, RadBufDecl>>    bufs;
    std::vector<RecOp>                                 ops;
    std::vector<RecSpan>                               spans;
    std::vector<std::string>                           notes;
    /* What rad_weight_encoding answers: the FIRST entry whose key the source name contains. */
    std::vector<std::pair<std::string, RadEncoding>>   encs;
};
struct RadCtx {
    int                   rank = 0;
    std::vector<RecIssue> issues;
    std::string           step_fail;
};

extern "C" {
rad_weight rad_decl_weight(RadBuilder* b, const char* name, const RadWeightDecl* d) {
    b->weights.push_back({name, *d});
    return (rad_weight)b->weights.size();
}
int rad_weight_encoding(RadBuilder* b, const char* source, RadEncoding* out, int64_t*, uint32_t*) {
    for (const auto& [key, e] : b->encs)
        if (source && std::strstr(source, key.c_str())) { *out = e; return RAD_OK; }
    return RAD_E_NOTFOUND;
}
rad_buf rad_decl_buffer(RadBuilder* b, const char* name, const RadBufDecl* d) {
    b->bufs.push_back({name, *d});
    return (rad_buf)b->bufs.size();
}
rad_op rad_decl_op(RadBuilder* b, const char* op, const RadParam* p, int n_p, const rad_weight* w,
                   int n_w) {
    RecOp r;
    r.op = op;
    for (int i = 0; i < n_p; ++i)
        r.p.push_back({p[i].key ? p[i].key : "", p[i].kind, p[i].ival, p[i].ihi,
                       p[i].sval ? p[i].sval : ""});
    r.w.assign(w, w + n_w);
    b->ops.push_back(r);
    return (rad_op)b->ops.size();
}
int rad_op_reads(RadBuilder* b, rad_op h, const rad_buf* x, int n) {
    b->ops[h - 1].reads.insert(b->ops[h - 1].reads.end(), x, x + n);
    return RAD_OK;
}
int rad_op_writes(RadBuilder* b, rad_op h, const rad_buf* x, int n) {
    b->ops[h - 1].writes.insert(b->ops[h - 1].writes.end(), x, x + n);
    return RAD_OK;
}
rad_kvgroup rad_decl_kv_group(RadBuilder*, const char*, const RadKVGroupDecl*) { return 1; }
int64_t rad_kv_block_size(RadBuilder*, rad_kvgroup) { return 4; }
int rad_bind_layer_kv(RadBuilder*, int, rad_kvgroup) { return RAD_OK; }
int rad_decl_name_map(RadBuilder*, const RadNameMap*) { return RAD_OK; }
void rad_note(RadBuilder* b, const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    b->notes.push_back(buf);
}
int rad_declare_logits(RadBuilder*, rad_buf) { return RAD_OK; }
int rad_buf_concurrent(RadBuilder*, rad_buf) { return RAD_OK; }
int rad_weight_shard_span(RadBuilder* b, rad_weight w, int64_t lo, int64_t hi) {
    b->spans.push_back({w, lo, hi});
    return RAD_OK;
}
int rad_weight_shard_span_parts(RadBuilder*, rad_weight, const int64_t*, const int64_t*,
                                const int64_t*, int) { return RAD_OK; }
int rad_kv_group_zero(RadBuilder*, rad_kvgroup) { return RAD_OK; }
int rad_declare_drafter(RadBuilder*, const RadDrafterDecl*) { return RAD_OK; }
int rad_declare_encoder(RadBuilder*, const RadEncoderDecl*) { return RAD_OK; }
long long rad_meta_geti(const RadModelMeta* m, const char* key, long long dflt) {
    for (int i = 0; i < m->n_kv; ++i) if (!std::strcmp(m->kv_key[i], key)) return atoll(m->kv_val[i]);
    return dflt;
}
double rad_meta_getf(const RadModelMeta* m, const char* key, double dflt) {
    for (int i = 0; i < m->n_kv; ++i) if (!std::strcmp(m->kv_key[i], key)) return atof(m->kv_val[i]);
    return dflt;
}
const char* rad_meta_gets(const RadModelMeta* m, const char* key, const char* dflt) {
    for (int i = 0; i < m->n_kv; ++i) if (!std::strcmp(m->kv_key[i], key)) return m->kv_val[i];
    return dflt;
}
int rad_issue(RadCtx* c, rad_op op, const RadOperand* opd, int n_opd, int64_t n) {
    c->issues.push_back({op, std::vector<RadOperand>(opd, opd + n_opd), n});
    return RAD_OK;
}
int rad_lane(RadCtx*, int) { return RAD_OK; }
int rad_lane_join(RadCtx*, int, int) { return RAD_OK; }
int rad_step_fail(RadCtx* c, const char* what) { c->step_fail = what ? what : ""; return RAD_E_SHAPE; }
int rad_rank(RadCtx* c) { return c->rank; }
RadStream rad_stream(RadCtx*) { return nullptr; }
int rad_route_report(RadCtx*, int, const RadRouting*) { return RAD_OK; }
int32_t* rad_route_counts(RadCtx*, int, int64_t) { return nullptr; }
void* rad_buf_ptr(RadCtx*, rad_buf b) { return (void*)(uintptr_t)(b * 64 + 16); }
/* The device API: declare and the MoE block's step touch none of it (the calibration tap and the
 * debug paths would), so every call is a failure the case reports. */
static int g_device_calls = 0;
void* rad_dev_alloc(int64_t, int) { ++g_device_calls; return nullptr; }
void  rad_dev_free(void*, int) { ++g_device_calls; }
void* rad_dev_host_ptr(void* p) { return p; }
void* rad_dev_device_ptr(void* p) { return p; }
int   rad_stream_create(RadStream*, int) { ++g_device_calls; return RAD_E_UNSUPPORTED; }
void  rad_stream_destroy(RadStream) {}
const char* rad_dev_last_error(void) { return "(fake)"; }
int   rad_stream_sync(RadStream) { ++g_device_calls; return RAD_OK; }
int   rad_memcpy_async(void*, const void*, int64_t, RadStream) { ++g_device_calls; return RAD_OK; }
int   rad_memset_async(void*, int, int64_t, RadStream) { ++g_device_calls; return RAD_OK; }
}  /* extern "C" */

#include "../../architectures/qwen4exp_exl3/qwen4exp_exl3.cpp"

/* ==================================================================== the cases */
namespace {

using rad::arch::MoeEXL3;
using rad::arch::calib_dir;

int g_failed = 0;
#define CHECK(cond) \
    do { if (!(cond)) { ++g_failed; fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } } while (0)
/* A precondition: the case stops here rather than read a graph that was never declared. */
#define REQUIRE_EQ(a, b) \
    do { const long long a_ = (long long)(a), b_ = (long long)(b); \
         if (a_ != b_) { ++g_failed; fprintf(stderr, "FAIL %s:%d  %s == %s  (%lld vs %lld) -- stopping\n", \
                                             __FILE__, __LINE__, #a, #b, a_, b_); return; } } while (0)
#define CHECK_EQ(a, b) \
    do { const long long a_ = (long long)(a), b_ = (long long)(b); \
         if (a_ != b_) { ++g_failed; fprintf(stderr, "FAIL %s:%d  %s == %s  (%lld vs %lld)\n", \
                                             __FILE__, __LINE__, #a, #b, a_, b_); } } while (0)

/* Qwen3.8-Flash-Next's metadata cut to eight layers, as radiance's tests/arch_test.cpp has it. */
const char* kKeys[] = {
    "full_attention_interval", "hc_count", "hc_lowrank", "heads_per_ngram",
    "hidden_act", "indexer_budget", "indexer_compress_ratio", "indexer_head_dim",
    "indexer_n_heads", "layer_types", "linear_conv_kernel_dim", "linear_key_head_dim",
    "linear_num_key_heads", "linear_num_value_heads", "linear_value_head_dim",
    "make_ngram_vocab_size_divisible_by",
    "moe_intermediate_size", "mtp_num_hidden_layers", "ngram_size", "ngram_vocab_size_base",
    "num_experts", "num_experts_per_tok", "output_gate_type", "partial_rotary_factor",
    "ple_conv_kernel_size", "ple_embed_dim", "ple_layer_ids", "shared_expert_intermediate_size",
    "split_ngram_parts", "rope_parameters.partial_rotary_factor", "eos_token_id",
};
const char* kVals[] = {
    "4", "4", "320", "8",
    "silu", "2048", "4", "128",
    "4",
    "linear_attention linear_attention linear_attention full_attention "
    "linear_attention linear_attention linear_attention full_attention",
    "4", "128",
    "16", "48", "128", "128",
    "640", "1", "3", "20000000",
    "512", "10", "sigmoid", "0.25",
    "4", "2560", "2", "640",
    "128", "0.25", "248044",
};
constexpr int kLayers = 8, kExperts = 512, kTopK = 10, kWidth = 640, kEmbd = 2560;

RadModelMeta meta() {
    RadModelMeta m{};
    m.arch_id     = "qwen4exp";
    m.name        = "test-q38-flashnext-exl3";
    m.quant       = "exl3";
    m.n_layers    = kLayers;
    m.n_embd      = kEmbd;
    m.n_head      = 24;
    m.n_head_kv   = 2;
    m.head_dim    = 256;
    m.n_ff        = kWidth;
    m.n_vocab     = 248320;
    m.n_ctx_train = 262144;
    m.rms_eps     = 1e-6f;
    m.rope_theta  = 1e7f;
    m.rope_scale  = 1.0f;
    m.n_kv        = (int)(sizeof(kKeys) / sizeof(kKeys[0]));
    m.kv_key      = kKeys;
    m.kv_val      = kVals;
    return m;
}

RadBuildCtx ctx(int rank, int world) {
    RadBuildCtx c{};
    c.rank = rank; c.world_size = world;
    c.max_tok = 2048; c.max_seqs = 8; c.max_spec = 3; c.max_ctx = 200000;
    c.scope = "";
    return c;
}

/* PLAN D4's shape: u{K} codes TILED [1x1], f16 sign vectors -- `suh` [0x1] (and `suh.1` for
 * gate_up's up half), `svh` [1x0]. Only the scheme and the codes dtype are what the plugin reads. */
RadEncoding exl3_enc(uint32_t codes, bool gate_up) {
    RadEncoding e;
    rad_enc_clear(&e);
    rad_enc_copy_str(e.scheme, "exl3");
    rad_enc_copy_str(e.transform, "mul1/h128");
    rad_enc_add_plane(&e, "codes", codes, 1, 1);
    rad_enc_add_plane(&e, "suh", RAD_F16, 0, 1);
    if (gate_up) rad_enc_add_plane(&e, "suh.1", RAD_F16, 0, 1);
    rad_enc_add_plane(&e, "svh", RAD_F16, 1, 0);
    return e;
}

/* Every routed expert at EXL3 4-bit; `first` entries, matched first, override some. */
void exl3_model(RadBuilder& b, std::vector<std::pair<std::string, RadEncoding>> first = {}) {
    b.encs = std::move(first);
    b.encs.push_back({ "ffn_gate_up_exps", exl3_enc(RAD_U4, true) });
    b.encs.push_back({ "ffn_down_exps",    exl3_enc(RAD_U4, false) });
}

std::string param_s(const RecOp& o, const char* key) {
    for (const auto& p : o.p) if (p.key == key) return p.sval;
    return "(absent)";
}
long long param_i(const RecOp& o, const char* key) {
    for (const auto& p : o.p) if (p.key == key) return p.ival;
    return -1;
}
int count_op(const RadBuilder& b, const std::string& name) {
    int n = 0;
    for (const auto& o : b.ops) n += o.op == name;
    return n;
}
std::vector<const RecOp*> ops_named(const RadBuilder& b, const std::string& name) {
    std::vector<const RecOp*> v;
    for (const auto& o : b.ops) if (o.op == name) v.push_back(&o);
    return v;
}
const std::string& wname(const RadBuilder& b, rad_weight h) { return b.weights[h - 1].first; }
const RadWeightDecl& wdecl(const RadBuilder& b, rad_weight h) { return b.weights[h - 1].second; }
const std::string& bname(const RadBuilder& b, rad_buf h) { return b.bufs[h - 1].first; }
const RadBufDecl* buf_named(const RadBuilder& b, const std::string& name) {
    for (const auto& [n, d] : b.bufs) if (n == name) return &d;
    return nullptr;
}
const RecSpan* span_of(const RadBuilder& b, rad_weight w) {
    for (const auto& s : b.spans) if (s.w == w) return &s;
    return nullptr;
}
/* The expert number in "blk.L.ffn_*_exps.E.weight". */
long long expert_of(const std::string& name) {
    const size_t at = name.find("_exps.");
    return at == std::string::npos ? -1 : std::atoll(name.c_str() + at + 6);
}
/* What a call wrote to stderr -- the refusals are fprintf'd, as the in-tree plugin's are -- echoed to
 * stdout so a test log shows the message an operator would read. */
std::string stderr_of(const std::function<void()>& fn) {
    fflush(stderr);
    FILE* tmp = tmpfile();
    const int saved = dup(2);
    dup2(fileno(tmp), 2);
    fn();
    fflush(stderr);
    dup2(saved, 2);
    close(saved);
    std::string out;
    rewind(tmp);
    char buf[4096];
    for (size_t n; (n = fread(buf, 1, sizeof buf, tmp)) > 0;) out.append(buf, n);
    fclose(tmp);
    printf("  stderr: %s", out.empty() ? "(nothing)\n" : out.c_str());
    return out;
}
bool has(const std::string& s, const char* what) { return s.find(what) != std::string::npos; }
const char* const kFp8Ops[] = { "had_quant_act_fp8", "gated_had_quant_fp8", "moe_gemm_q",
                                "moe_gemm_q_mod4" };

/* One rank's EXL3 op geometry: every trunk layer and the MTP head the same. */
void check_exl3_ops(const RadBuilder& b, int64_t n_gu, int64_t n_gu_odd, int64_t k_dn,
                    int64_t k_dn_odd, int64_t n_expert, const char* dtype) {
    const auto gu = ops_named(b, "moe_exl3_gate_up"), dn = ops_named(b, "moe_exl3_down");
    CHECK_EQ(gu.size(), kLayers + 1);   /* every layer is routed, and the MTP head */
    CHECK_EQ(dn.size(), kLayers + 1);
    for (const RecOp* o : gu) {
        CHECK_EQ(param_i(*o, "N"), n_gu);     CHECK_EQ(param_i(*o, "N_odd"), n_gu_odd);
        CHECK_EQ(param_i(*o, "K"), kEmbd);    CHECK_EQ(param_i(*o, "K_odd"), kEmbd);
        CHECK_EQ(param_i(*o, "n_expert"), n_expert);
        CHECK_EQ(param_i(*o, "top_k"), kTopK);
        CHECK(param_s(*o, "dtype") == dtype);
        CHECK_EQ(o->w.size(), n_expert);
        CHECK(o->reads.size() == 3 && bname(b, o->reads[0]) == "x");
        CHECK(o->writes.size() == 1 && bname(b, o->writes[0]) == "moe_eff");
    }
    for (const RecOp* o : dn) {
        CHECK_EQ(param_i(*o, "N"), kEmbd);    CHECK_EQ(param_i(*o, "N_odd"), kEmbd);
        CHECK_EQ(param_i(*o, "K"), k_dn);     CHECK_EQ(param_i(*o, "K_odd"), k_dn_odd);
        CHECK_EQ(param_i(*o, "n_expert"), n_expert);
        CHECK(param_s(*o, "dtype") == dtype);
        CHECK_EQ(o->w.size(), n_expert);
        CHECK(o->reads.size() == 3 && bname(b, o->reads[0]) == "moe_eff");
        CHECK(o->writes.size() == 1 && bname(b, o->writes[0]) == "moe_edn");
    }
    for (const char* op : kFp8Ops) CHECK_EQ(count_op(b, op), 0);
}

/* ---- one rank, the shipped geometry. Each expert is ONE weight over every plane, at the codes'
 * dtype; the two tables are the even experts then the odd ones; the intermediate is one f16 plane
 * and no fp8 activation op exists anywhere. */
void tp1_declares_exl3_experts_and_no_activation_quant() {
    RadModelMeta m = meta();
    RadBuildCtx c = ctx(0, 1);
    RadBuilder b;
    exl3_model(b);
    REQUIRE_EQ(qwen4exp_exl3::declare(&b, &m, &c), RAD_OK);
    check_exl3_ops(b, 2 * kWidth, 2 * kWidth, kWidth, kWidth, kExperts, "exl3k4");
    REQUIRE_EQ(ops_named(b, "moe_exl3_gate_up").size(), kLayers + 1);
    REQUIRE_EQ(ops_named(b, "moe_exl3_down").size(), kLayers + 1);
    const RecOp* gu = ops_named(b, "moe_exl3_gate_up").front();
    for (size_t i = 0; i < gu->w.size(); ++i) {
        const RadWeightDecl& d = wdecl(b, gu->w[i]);
        CHECK(d.planes == nullptr && d.source == nullptr);   /* every plane: one operand */
        CHECK_EQ(d.dtype, RAD_U4);
        CHECK_EQ(d.shape[0], 2 * kWidth);
        CHECK_EQ(d.shape[1], kEmbd);
        CHECK_EQ(d.access, RAD_ACCESS_CONDITIONAL);
        /* parity tables: entries 0..255 are experts 0, 2, ..., 510; 256..511 are 1, 3, ..., 511 */
        const long long want = i < kExperts / 2 ? 2 * (long long)i : 2 * (long long)(i - kExperts / 2) + 1;
        CHECK_EQ(expert_of(wname(b, gu->w[i])), want);
        CHECK(i == 0 || gu->w[i] == gu->w[i - 1] + 1);   /* one contiguous declared run */
    }
    const RecOp* dn = ops_named(b, "moe_exl3_down").front();
    for (rad_weight h : dn->w) {
        CHECK(wdecl(b, h).planes == nullptr && wdecl(b, h).dtype == RAD_U4);
        CHECK_EQ(wdecl(b, h).shape[0], kEmbd);
        CHECK_EQ(wdecl(b, h).shape[1], kWidth);
    }
    const RadBufDecl* eff = buf_named(b, "moe_eff");
    CHECK(eff && eff->dtype == RAD_F16 && eff->shape[0] == 2048 * kTopK && eff->shape[1] == kWidth);
    for (const auto& [name, d] : b.bufs)   /* no gate/up plane, no rotated input pair */
        CHECK(name.find("moe_egu") == std::string::npos && name.find("moe_hr") == std::string::npos);
    CHECK(b.spans.empty());   /* nothing is sliced at one rank */
    /* the in-tree plugin's rotated twin and its codes are not written: int8 connections or none */
    CHECK(!qwen4exp_exl3::g_model[0].moecfg.expert_rot);
}

/* ---- two ranks: the parity split, 384 and 256 columns, the wide slice alternating with parity. */
void tp2_splits_every_expert_384_256_by_parity() {
    RadModelMeta m = meta();
    for (int r = 0; r < 2; ++r) {
        RadBuildCtx c = ctx(r, 2);
        RadBuilder b;
        exl3_model(b);
        REQUIRE_EQ(qwen4exp_exl3::declare(&b, &m, &c), RAD_OK);
        const int64_t even = r == 0 ? 384 : 256, odd = r == 0 ? 256 : 384;
        check_exl3_ops(b, 2 * even, 2 * odd, even, odd, kExperts, "exl3k4");
        const int64_t lo_even = r == 0 ? 0 : 384, lo_odd = r == 0 ? 384 : 0;
        for (const char* name : { "moe_exl3_gate_up", "moe_exl3_down" }) {
            REQUIRE_EQ(ops_named(b, name).size(), kLayers + 1);
            const RecOp* o = ops_named(b, name).front();
            const bool gu = std::string(name) == "moe_exl3_gate_up";
            for (rad_weight h : o->w) {
                const bool is_even = expert_of(wname(b, h)) % 2 == 0;
                const int64_t width = is_even ? even : odd, lo = is_even ? lo_even : lo_odd;
                const RadWeightDecl& d = wdecl(b, h);
                const RecSpan* s = span_of(b, h);
                CHECK(s && s->lo == lo && s->hi == lo + width);
                CHECK_EQ(d.shard, gu ? RAD_SHARD_ROW : RAD_SHARD_COL);
                CHECK_EQ(d.shape[gu ? 0 : 1], gu ? 2 * width : width);
                if (gu) CHECK(d.n_row_parts == 2 && d.row_parts[0] == width && d.row_parts[1] == width);
            }
        }
    }
}

/* ---- a layer that keeps two experts plain bf16 serves them with the in-tree bf16 pair, sorted
 * last, and the EXL3 tables take the other 510. */
void protected_bf16_experts_keep_their_own_pair() {
    RadModelMeta m = meta();
    RadBuildCtx c = ctx(0, 1);
    RadBuilder b;
    const RadEncoding bf16 = rad_enc_plain(RAD_BF16);
    exl3_model(b, { { "blk.0.ffn_gate_up_exps.3.weight", bf16 }, { "blk.0.ffn_down_exps.3.weight", bf16 },
                    { "blk.0.ffn_gate_up_exps.5.weight", bf16 }, { "blk.0.ffn_down_exps.5.weight", bf16 } });
    REQUIRE_EQ(qwen4exp_exl3::declare(&b, &m, &c), RAD_OK);
    const auto gu = ops_named(b, "moe_exl3_gate_up");
    REQUIRE_EQ(gu.size(), kLayers + 1);
    CHECK_EQ(param_i(*gu[0], "n_expert"), kExperts - 2);
    for (size_t i = 1; i < gu.size(); ++i) CHECK_EQ(param_i(*gu[i], "n_expert"), kExperts);
    const auto pair = ops_named(b, "moe_gemm");
    CHECK_EQ(pair.size(), 2);
    for (const RecOp* o : pair) {
        CHECK(param_s(*o, "dtype") == "bf16");
        CHECK_EQ(o->w.size(), 2);
    }
    int protect_lists = 0;
    for (const auto& o : b.ops)
        if (param_s(o, "protect") == "3,5") ++protect_lists;
    CHECK(protect_lists > 0);
}

/* ---- each layer and projection may carry its own trellis width; one table, one encoding. */
void a_layer_may_carry_its_own_trellis_width() {
    RadModelMeta m = meta();
    RadBuildCtx c = ctx(0, 1);
    RadBuilder b;
    exl3_model(b, { { "blk.1.ffn_gate_up_exps.", exl3_enc(RAD_U5, true) } });
    REQUIRE_EQ(qwen4exp_exl3::declare(&b, &m, &c), RAD_OK);
    const auto gu = ops_named(b, "moe_exl3_gate_up"), dn = ops_named(b, "moe_exl3_down");
    REQUIRE_EQ(gu.size(), kLayers + 1);
    REQUIRE_EQ(dn.size(), kLayers + 1);
    CHECK(param_s(*gu[1], "dtype") == "exl3k5");
    CHECK(param_s(*dn[1], "dtype") == "exl3k4");
    CHECK(param_s(*gu[0], "dtype") == "exl3k4");
    CHECK_EQ(wdecl(b, gu[1]->w[0]).dtype, RAD_U5);
}

/* ---- an int8 trunk: the connections write the int8 codes themselves, since nothing on the expert
 * side reads E4M3 codes of the block input (the in-tree plugin does the same for its rotated and
 * bf16 experts; left at E4M3, every routed layer would write codes nobody reads and have its int8
 * shared expert quantise again). An fp8 trunk keeps the E4M3 codes. */
void an_int8_trunk_has_its_connections_write_int8_codes() {
    RadModelMeta m = meta();
    RadBuildCtx c = ctx(0, 1);
    RadBuilder i8;
    /* the served recipe's int8 trunk: every delta-net and attention linear and the shared expert
     * (the connections' own mixing weights stay as they are, and the QSA indexer bf16 as served) */
    const RadEncoding w8 = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    exl3_model(i8, { { "_hc", rad_enc_plain(RAD_BF16) }, { "qsa_qk", rad_enc_plain(RAD_BF16) },
                     { "ssm_", w8 }, { "attn_", w8 }, { "_shexp", w8 } });
    REQUIRE_EQ(qwen4exp_exl3::declare(&i8, &m, &c), RAD_OK);
    CHECK(qwen4exp_exl3::g_model[0].trunk_i8);
    CHECK_EQ(qwen4exp_exl3::g_model[0].hccfg.codes_i8, 1);
    CHECK(qwen4exp_exl3::g_model[0].a_x.q8_fed);
    RadBuilder fp8;
    exl3_model(fp8);
    REQUIRE_EQ(qwen4exp_exl3::declare(&fp8, &m, &c), RAD_OK);
    CHECK(!qwen4exp_exl3::g_model[0].trunk_i8);
    CHECK_EQ(qwen4exp_exl3::g_model[0].hccfg.codes_i8, 0);
}

/* ---- refusals, each by name. */
void a_non_exl3_expert_is_refused_by_name() {
    RadModelMeta m = meta();
    RadBuildCtx c = ctx(0, 1);
    RadBuilder b;
    exl3_model(b, { { "blk.0.ffn_gate_up_exps.7.weight", rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128) } });
    int st = 0;
    const std::string err = stderr_of([&] { st = qwen4exp_exl3::declare(&b, &m, &c); });
    CHECK_EQ(st, RAD_E_DTYPE);
    CHECK(has(err, "blk.0.ffn_gate_up_exps.7.weight") && has(err, "one table of experts takes one encoding"));
}
void a_second_trellis_width_in_one_table_is_refused() {
    RadModelMeta m = meta();
    RadBuildCtx c = ctx(0, 1);
    RadBuilder b;
    exl3_model(b, { { "blk.2.ffn_down_exps.9.weight", exl3_enc(RAD_U5, false) } });
    int st = 0;
    const std::string err = stderr_of([&] { st = qwen4exp_exl3::declare(&b, &m, &c); });
    CHECK_EQ(st, RAD_E_DTYPE);
    CHECK(has(err, "blk.2.ffn_down_exps.9.weight"));
}
void a_trellis_width_with_no_op_is_refused() {
    RadModelMeta m = meta();
    RadBuildCtx c = ctx(0, 1);
    RadBuilder b;
    exl3_model(b, { { "ffn_gate_up_exps", exl3_enc(RAD_U3, true) } });
    int st = 0;
    const std::string err = stderr_of([&] { st = qwen4exp_exl3::declare(&b, &m, &c); });
    CHECK_EQ(st, RAD_E_DTYPE);
    CHECK(has(err, "serves the routed experts as EXL3"));
}
void plain_bf16_experts_are_the_in_tree_plugins() {
    RadModelMeta m = meta();
    RadBuildCtx c = ctx(0, 1);
    RadBuilder b;
    b.encs = { { "_exps.", rad_enc_plain(RAD_BF16) } };
    int st = 0;
    const std::string err = stderr_of([&] { st = qwen4exp_exl3::declare(&b, &m, &c); });
    CHECK_EQ(st, RAD_E_DTYPE);
    CHECK(has(err, "qwen4exp_fp8"));
}
void a_builder_with_no_model_is_refused() {
    RadModelMeta m = meta();
    RadBuildCtx c = ctx(0, 1);
    RadBuilder b;   /* rad_weight_encoding answers nothing */
    int st = 0;
    const std::string err = stderr_of([&] { st = qwen4exp_exl3::declare(&b, &m, &c); });
    CHECK_EQ(st, RAD_E_NOTFOUND);
    CHECK(has(err, "nothing is known about"));
}
void four_ranks_are_refused_by_name() {
    RadModelMeta m = meta();
    RadBuildCtx c = ctx(0, 4);
    RadBuilder b;
    exl3_model(b);
    int st = 0;
    const std::string err = stderr_of([&] { st = qwen4exp_exl3::declare(&b, &m, &c); });
    CHECK_EQ(st, RAD_E_UNSUPPORTED);
    CHECK(has(err, "moe_exl3_gate_up") && has(err, "two parity tables"));
}

/* ---- the MoE block's issues on a decode step: gate_up over the bf16 input, then down over the
 * f16 intermediate, each with the even and the odd table, and nothing quantised between. The
 * trunk's first layer and the MTP head's block are stepped; the rest of the model is stock code. */
void decode_issues_gate_up_then_down_over_both_parity_tables() {
    RadModelMeta m = meta();
    RadBuildCtx c = ctx(0, 1);
    RadBuilder b;
    exl3_model(b);
    REQUIRE_EQ(qwen4exp_exl3::declare(&b, &m, &c), RAD_OK);
    const qwen4exp_exl3::Model& mod = qwen4exp_exl3::g_model[0];
    CHECK(mod.have_mtp);
    static const int32_t out[] = { 0, 1, 2, 3 };
    RadBatch batch{};
    batch.phase = RAD_PHASE_DECODE; batch.n_tok = 4; batch.n_seq = 4; batch.max_q_len = 1;
    batch.max_ctx_len = 100; batch.n_out = 4; batch.out_ids = out;
    for (const MoeEXL3* blk : { &mod.layers[0].mlp, &mod.mtp.mlp }) {
        RadCtx cx;
        blk->step(&cx, &batch);
        CHECK(cx.step_fail.empty());
        int at_gu = -1, at_dn = -1;
        for (size_t i = 0; i < cx.issues.size(); ++i) {
            const RecIssue& is = cx.issues[i];
            const std::string& op = b.ops[is.op - 1].op;
            for (const char* f : kFp8Ops) CHECK(op != f);
            if (op != "moe_exl3_gate_up" && op != "moe_exl3_down") continue;
            const bool gu = op == "moe_exl3_gate_up";
            (gu ? at_gu : at_dn) = (int)i;
            CHECK_EQ(is.n, 4);
            CHECK_EQ(is.opd.size(), 6);
            if (is.opd.size() != 6) continue;
            CHECK(is.opd[0].kind == RAD_OPK_BUF && bname(b, is.opd[0].handle) == (gu ? "x" : "moe_eff"));
            const rad_weight first = gu ? blk->w_gu[0] : blk->w_dn[0];
            const rad_weight first_odd = gu ? blk->w_gu[1] : blk->w_dn[1];
            CHECK(is.opd[1].kind == RAD_OPK_WTAB && is.opd[1].handle == first && is.opd[1].rows == kExperts / 2);
            CHECK(is.opd[2].kind == RAD_OPK_WTAB && is.opd[2].handle == first_odd && is.opd[2].rows == kExperts / 2);
            CHECK(expert_of(wname(b, is.opd[1].handle)) == 0 && expert_of(wname(b, is.opd[2].handle)) == 1);
            CHECK(bname(b, is.opd[3].handle) == "moe_sorted" && is.opd[3].rows == 4 * kTopK);
            CHECK(bname(b, is.opd[4].handle) == "moe_eoff" && is.opd[4].rows == kExperts + 1);
            CHECK(bname(b, is.opd[5].handle) == (gu ? "moe_eff" : "moe_edn") && is.opd[5].rows == 4 * kTopK);
        }
        CHECK(at_gu >= 0 && at_dn > at_gu);
    }
    CHECK_EQ(g_device_calls, 0);
    (void)&qwen4exp_exl3::step;    /* the model's step is stock code; its MoE block's is checked */
    (void)&qwen4exp_exl3::probe;
}

/* ---- the calibration tap: the gate_up Hessian over the bf16 input as it stands, the down one
 * over a bf16 copy of the f16 intermediate, in the model's own basis. The tap reads
 * RADIANCE_CALIB_DIR once per process and this case is a process of its own, so it sets it first;
 * declare only checks that it is set (nothing is written until a step reaches the token target). */
void calibration_taps_the_inputs_the_ops_are_handed() {
    setenv("RADIANCE_CALIB_DIR", "/nonexistent-calibration-dir-declare-only", 1);
    CHECK(calib_dir() != nullptr);
    RadModelMeta m = meta();
    RadBuildCtx c = ctx(0, 1);
    RadBuilder b;
    exl3_model(b);
    REQUIRE_EQ(qwen4exp_exl3::declare(&b, &m, &c), RAD_OK);
    const auto casts = ops_named(b, "cast");
    int calib_casts = 0;
    for (const RecOp* o : casts)
        if (param_s(*o, "from") == "f16" && param_s(*o, "to") == "bf16") {
            ++calib_casts;
            CHECK(bname(b, o->reads[0]) == "moe_eff");
        }
    CHECK_EQ(calib_casts, kLayers + 1);
    const auto grams = ops_named(b, "gram_accum");
    CHECK_EQ(grams.size(), 3 * (kLayers + 1));
    for (const RecOp* o : grams) {
        CHECK(param_s(*o, "dtype") == "bf16");
        const std::string in = bname(b, o->reads[0]);
        CHECK(in == "x" || in.find("moe_calib_eff") != std::string::npos);
    }
}

struct Case { const char* name; void (*fn)(); };
const Case kCases[] = {
    { "tp1_declares_exl3_experts_and_no_activation_quant", tp1_declares_exl3_experts_and_no_activation_quant },
    { "tp2_splits_every_expert_384_256_by_parity", tp2_splits_every_expert_384_256_by_parity },
    { "protected_bf16_experts_keep_their_own_pair", protected_bf16_experts_keep_their_own_pair },
    { "a_layer_may_carry_its_own_trellis_width", a_layer_may_carry_its_own_trellis_width },
    { "an_int8_trunk_has_its_connections_write_int8_codes", an_int8_trunk_has_its_connections_write_int8_codes },
    { "a_non_exl3_expert_is_refused_by_name", a_non_exl3_expert_is_refused_by_name },
    { "a_second_trellis_width_in_one_table_is_refused", a_second_trellis_width_in_one_table_is_refused },
    { "a_trellis_width_with_no_op_is_refused", a_trellis_width_with_no_op_is_refused },
    { "plain_bf16_experts_are_the_in_tree_plugins", plain_bf16_experts_are_the_in_tree_plugins },
    { "a_builder_with_no_model_is_refused", a_builder_with_no_model_is_refused },
    { "four_ranks_are_refused_by_name", four_ranks_are_refused_by_name },
    { "decode_issues_gate_up_then_down_over_both_parity_tables", decode_issues_gate_up_then_down_over_both_parity_tables },
    { "calibration_taps_the_inputs_the_ops_are_handed", calibration_taps_the_inputs_the_ops_are_handed },
};

}  /* namespace */

/* `arch_static_test --list` prints the cases; `arch_static_test NAME` runs one. Exit 0 PASS,
 * 1 FAIL, 77 SKIP (ctest's SKIP_RETURN_CODE), 2 no such case. */
int main(int argc, char** argv) {
    if (argc == 2 && !std::strcmp(argv[1], "--list")) {
        for (const Case& k : kCases) printf("%s\n", k.name);
        return 0;
    }
    if (argc != 2) { fprintf(stderr, "usage: %s --list | CASE\n", argv[0]); return 2; }
    for (const Case& k : kCases)
        if (!std::strcmp(argv[1], k.name)) {
            k.fn();
            printf("%s %s\n", g_failed ? "FAIL" : "PASS", k.name);
            return g_failed ? 1 : 0;
        }
    fprintf(stderr, "no case '%s' (--list)\n", argv[1]);
    return 2;
}
