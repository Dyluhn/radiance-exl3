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
#include <sys/wait.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <algorithm>
#include <map>
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
    /* radiance 1.4.0's next-layer expert prefetch: each call, as (kind 'p' predict / 's' step / 'm' mark,
     * the issue count when it was made, the layer it names: the predicted one for 'p', the stepped
     * one for 's', -1 for 'm'). */
    struct XpCall { char kind; size_t at; int layer; };
    std::vector<XpCall>   xp;
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
int rad_expert_prefetch(RadCtx* c, const RadPrefetchNext* next, rad_buf, int64_t, int64_t, int64_t) {
    c->xp.push_back({'p', c->issues.size(), next ? next->layer : -2});
    return RAD_OK;
}
int rad_expert_prefetch_step(RadCtx* c, int layer) { c->xp.push_back({'s', c->issues.size(), layer}); return RAD_OK; }
int rad_expert_prefetch_mark(RadCtx* c) { c->xp.push_back({'m', c->issues.size(), -1}); return RAD_OK; }
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

/* The encoding as the exl3 quantiser writes it (include/exl3_encoding.h, P1's own builder): codes
 * uK [1x1], suh f16 [0x1], svh f16 [1x0], and suh.1 [0x1] on a stacked gate_up. */
RadEncoding exl3_enc(int bits, bool gate_up, int codebook = EXL3_CB_MUL1) {
    RadEncoding e{};
    if (exl3_encoding_make(bits, codebook, gate_up ? 1 : 0, &e) != RAD_OK) {
        fprintf(stderr, "exl3_encoding_make(%d) refused\n", bits);
        std::abort();
    }
    return e;
}

/* Every routed expert at EXL3 4-bit mul1; `first` entries, matched first, override some. */
void exl3_model(RadBuilder& b, std::vector<std::pair<std::string, RadEncoding>> first = {}) {
    b.encs = std::move(first);
    b.encs.push_back({ "ffn_gate_up_exps", exl3_enc(4, true) });
    b.encs.push_back({ "ffn_down_exps",    exl3_enc(4, false) });
}

std::string param_s(const RecOp& o, const char* key) {
    for (const auto& p : o.p) if (p.key == key) return p.sval;
    return "(absent)";
}
long long param_i(const RecOp& o, const char* key) {
    for (const auto& p : o.p) if (p.key == key) return p.ival;
    return -1;
}
/* A ranged parameter's high bound (RAD_RANGE("M", 1, hi)). */
long long param_hi(const RecOp& o, const char* key) {
    for (const auto& p : o.p) if (p.key == key) return p.ihi;
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
/* THE ISSUE ORDER UNDER TEST (interface v4): this binary is built at QWEN4EXP_EXL3_PREP_SPLIT 0 and
 * at 1 (tests/arch/CMakeLists.txt), and every case holds for both. gate_up is declared once a layer
 * with a_rot = 0, and in the split build also with a_rot = 1 over the decode band, beside prep_gu. */
constexpr bool    kSplit    = QWEN4EXP_EXL3_PREP_SPLIT != 0;
constexpr int64_t kPrepBand = 16;
/* And with QWEN4EXP_EXL3_GD (interface v5, arch_static_test_gd): moe_exl3_gate_up_down over the same
 * band, declared after each layer's pair, issued there in place of the pair. */
constexpr bool    kGd       = QWEN4EXP_EXL3_GD != 0;
constexpr int64_t kGdMin    = EXL3_GD_MIN_ROWS;   /* gate_up_down's fewest rows; below, the band's split form */
/* QWEN4EXP_EXL3_PREFILL_STAGE 0 (arch_static_test_nostage): one never-issued prep_gu over layer 0's tables, ranged
 * M 1..1, declared after layer 1's block -- told apart from the decode band's prep_gu by that range. */
constexpr bool    kStageOn  = QWEN4EXP_EXL3_PREFILL_STAGE != 0;
bool is_stage_off(const RecOp& o) { return o.op == "moe_exl3_prep_gu" && param_hi(o, "M") == 1; }
std::vector<const RecOp*> gate_ups(const RadBuilder& b, int a_rot) {
    std::vector<const RecOp*> v;
    for (const RecOp* o : ops_named(b, "moe_exl3_gate_up"))
        if (param_i(*o, "a_rot") == a_rot) v.push_back(o);
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

/* THE PER-PLANE TABLES (PLAN D15, interface v3), written out rather than computed: each op's weight
 * list is the even experts' tables then the odd experts', a table a plane, in this order. gate_up's
 * fifth table is dn_suh: DOWN's suh run, the very handles down's own suh table names (EPI 1). */
struct PlaneWant { const char* role; bool cut; };            /* cut: sliced by this rank's span */
const PlaneWant kGu[] = { { "codes", true }, { "suh", false }, { "suh.1", false }, { "svh", true },
                          { "suh", true } };
const PlaneWant kDn[] = { { "codes", true }, { "suh", true }, { "svh", false } };
constexpr int64_t kGuTables = 5, kDnTables = 3, kDnSuh = 4;   /* kDnSuh: gate_up's dn_suh table */

/* THE DECODE-BAND FORM's declarations (split build; none otherwise): per layer prep_gu and
 * gate_up(a_rot = 1), both ranged over M 1..16, beside that layer's gate_up(a_rot = 0) `gu0[i]`:
 * the a_rot = 1 op is the same op over the same weights reading xr; prep_gu's tables are gate_up's
 * suh and suh.1 runs, the very handles, and it writes xr -- f16, a sorted row of the band each,
 * gate half then up half. */
void check_decode_band_form(const RadBuilder& b, const std::vector<const RecOp*>& gu0,
                            int64_t n_expert, const char* dtype) {
    const auto gu1 = gate_ups(b, 1);
    std::vector<const RecOp*> pg;
    for (const RecOp* o : ops_named(b, "moe_exl3_prep_gu")) if (!is_stage_off(*o)) pg.push_back(o);
    CHECK_EQ(gu1.size(), kSplit ? kLayers + 1 : 0);
    CHECK_EQ(pg.size(), kSplit ? kLayers + 1 : 0);
    if (gu1.size() != gu0.size() || pg.size() != gu0.size()) return;
    const int64_t half = n_expert / 2;
    for (size_t i = 0; i < gu1.size(); ++i) {
        const RecOp& a = *gu1[i], &z = *gu0[i], &p = *pg[i];
        CHECK_EQ(param_hi(a, "M"), kPrepBand);
        for (const char* k : { "N", "K", "N_odd", "K_odd", "n_expert", "top_k", "h_rot" })
            CHECK_EQ(param_i(a, k), param_i(z, k));
        CHECK(param_s(a, "dtype") == dtype && param_s(a, "codebook") == "mul1");
        CHECK(a.w == z.w);
        REQUIRE_EQ(a.reads.size() == 3 && p.writes.size() == 1, 1);
        CHECK(a.reads[0] == p.writes[0] && has(bname(b, a.reads[0]), "moe_exl3_xr"));
        CHECK(a.writes.size() == 1 && bname(b, a.writes[0]) == "moe_eff");
        const RadBufDecl* xr = nullptr;
        for (const auto& [name, d] : b.bufs) if (name == bname(b, p.writes[0])) xr = &d;
        CHECK(xr && xr->dtype == RAD_F16 && xr->shape[0] == kPrepBand * kTopK && xr->shape[1] == 2 * kEmbd);
        CHECK_EQ(param_hi(p, "M"), kPrepBand);
        CHECK_EQ(param_i(p, "K"), kEmbd);       CHECK_EQ(param_i(p, "K_odd"), kEmbd);
        CHECK_EQ(param_i(p, "n_expert"), n_expert);
        CHECK_EQ(param_i(p, "top_k"), kTopK);
        CHECK(param_s(p, "dtype") == dtype);
        CHECK(param_i(p, "N") == -1 && param_s(p, "codebook") == "(absent)" && param_i(p, "a_rot") == -1);
        REQUIRE_EQ(p.w.size(), 2 * 2 * half);
        for (int64_t q = 0; q < 2; ++q)
            for (int64_t t = 0; t < 2; ++t)
                for (int64_t j = 0; j < half; ++j)
                    CHECK(p.w[(size_t)(q * 2 * half + t * half + j)] ==
                          z.w[(size_t)(q * kGuTables * half + (1 + t) * half + j)]);
        CHECK(p.reads.size() == 3 && bname(b, p.reads[0]) == "x" && bname(b, p.reads[1]) == "moe_sorted" &&
              bname(b, p.reads[2]) == "moe_eoff");
    }
}

/* THE DECODE CHAIN AS ONE OP's declarations (GD build; none otherwise): per layer one
 * gate_up_down ranged over M 3..16, with gate_up's geometry, dtype and codebook and no flags; its
 * tables gate_up(a_rot = 0)'s five, the very handles, then down's codes and svh runs; reading x and
 * the routing, writing moe_edn; declared after that layer's gate_up and BEFORE its down (the
 * stager lets a layer's buffer go after its last declared reader, which every pass must issue). */
void check_gd_form(const RadBuilder& b, const std::vector<const RecOp*>& gu0, const std::vector<const RecOp*>& dn,
                   int64_t n_expert, const char* dtype) {
    const auto gd = ops_named(b, "moe_exl3_gate_up_down");
    CHECK_EQ(gd.size(), kGd ? kLayers + 1 : 0);
    if (gd.size() != gu0.size() || dn.size() != gu0.size()) return;
    const int64_t half = n_expert / 2;
    for (size_t i = 0; i < gd.size(); ++i) {
        const RecOp& a = *gd[i], &z = *gu0[i], &d = *dn[i];
        CHECK(&a > &z && &a < &d);
        CHECK_EQ(param_i(a, "M"), kGdMin);
        CHECK_EQ(param_hi(a, "M"), kPrepBand);
        for (const char* k : { "N", "K", "N_odd", "K_odd", "n_expert", "top_k" })
            CHECK_EQ(param_i(a, k), param_i(z, k));
        CHECK(param_s(a, "dtype") == dtype && param_s(a, "codebook") == "mul1");
        CHECK(param_i(a, "h_rot") == -1 && param_i(a, "a_rot") == -1);
        REQUIRE_EQ(a.w.size(), 2 * (kGuTables + 2) * half);
        for (int64_t q = 0; q < 2; ++q)
            for (int64_t t = 0; t < kGuTables + 2; ++t)
                for (int64_t j = 0; j < half; ++j) {
                    const rad_weight want = t < kGuTables ? z.w[(size_t)(q * kGuTables * half + t * half + j)]
                                          : d.w[(size_t)(q * kDnTables * half + (t == kGuTables ? 0 : 2) * half + j)];
                    CHECK(a.w[(size_t)(q * (kGuTables + 2) * half + t * half + j)] == want);
                }
        CHECK(a.reads.size() == 3 && bname(b, a.reads[0]) == "x" && bname(b, a.reads[1]) == "moe_sorted" &&
              bname(b, a.reads[2]) == "moe_eoff");
        CHECK(a.writes.size() == 1 && bname(b, a.writes[0]) == "moe_edn");
    }
}

/* One rank's EXL3 op geometry: every trunk layer and the MTP head the same. */
void check_exl3_ops(const RadBuilder& b, int64_t n_gu, int64_t n_gu_odd, int64_t k_dn,
                    int64_t k_dn_odd, int64_t n_expert, const char* dtype) {
    const auto gu = gate_ups(b, 0), dn = ops_named(b, "moe_exl3_down");
    CHECK_EQ(gu.size(), kLayers + 1);   /* every layer is routed, and the MTP head */
    CHECK_EQ(count_op(b, "moe_exl3_gate_up"), (kSplit ? 2 : 1) * (kLayers + 1));
    CHECK_EQ(dn.size(), kLayers + 1);
    for (const RecOp* o : gu) {
        CHECK_EQ(param_i(*o, "N"), n_gu);     CHECK_EQ(param_i(*o, "N_odd"), n_gu_odd);
        CHECK_EQ(param_i(*o, "K"), kEmbd);    CHECK_EQ(param_i(*o, "K_odd"), kEmbd);
        CHECK_EQ(param_i(*o, "n_expert"), n_expert);
        CHECK_EQ(param_i(*o, "top_k"), kTopK);
        CHECK(param_s(*o, "dtype") == dtype);
        CHECK(param_s(*o, "codebook") == "mul1");
        CHECK_EQ(param_i(*o, "h_rot"), 1);
        CHECK_EQ(param_hi(*o, "M"), 2048);
        CHECK_EQ(o->w.size(), kGuTables * n_expert);
        CHECK(o->reads.size() == 3 && bname(b, o->reads[0]) == "x");
        CHECK(o->writes.size() == 1 && bname(b, o->writes[0]) == "moe_eff");
    }
    for (const RecOp* o : dn) {
        CHECK_EQ(param_i(*o, "N"), kEmbd);    CHECK_EQ(param_i(*o, "N_odd"), kEmbd);
        CHECK_EQ(param_i(*o, "K"), k_dn);     CHECK_EQ(param_i(*o, "K_odd"), k_dn_odd);
        CHECK_EQ(param_i(*o, "n_expert"), n_expert);
        CHECK(param_s(*o, "dtype") == dtype);
        CHECK(param_s(*o, "codebook") == "mul1");
        CHECK_EQ(param_i(*o, "h_rot"), 1);
        CHECK_EQ(o->w.size(), kDnTables * n_expert);
        CHECK(o->reads.size() == 3 && bname(b, o->reads[0]) == "moe_eff");
        CHECK(o->writes.size() == 1 && bname(b, o->writes[0]) == "moe_edn");
    }
    for (const char* op : kFp8Ops) CHECK_EQ(count_op(b, op), 0);
    check_decode_band_form(b, gu, n_expert, dtype);
    check_gd_form(b, gu, dn, n_expert, dtype);
}

/* Every view of the first layer's two ops against the table above: name, source, plane, dtype,
 * this rank's extents, shard and span, and the order of the list. `w_even`/`w_odd` are this rank's
 * widths of the two classes, `lo_*` their first column (spans only where `split`). */
void check_views(const RadBuilder& b, bool split, int64_t w_even, int64_t w_odd, int64_t lo_even,
                 int64_t lo_odd) {
    const auto dns = ops_named(b, "moe_exl3_down");
    REQUIRE_EQ(dns.size(), kLayers + 1);
    for (int proj = 0; proj < 2; ++proj) {
        const auto ops = proj == 0 ? gate_ups(b, 0) : ops_named(b, "moe_exl3_down");
        REQUIRE_EQ(ops.size(), kLayers + 1);
        const RecOp* o = ops.front();
        const PlaneWant* want = proj == 0 ? kGu : kDn;
        const int64_t n_planes = proj == 0 ? kGuTables : kDnTables, half = kExperts / 2;
        REQUIRE_EQ(o->w.size(), 2 * n_planes * half);
        REQUIRE_EQ(dns.front()->w.size(), 2 * kDnTables * half);
        for (size_t i = 0; i < o->w.size(); ++i) {
            const int64_t q = (int64_t)i / (n_planes * half), pl = ((int64_t)i / half) % n_planes;
            if (proj == 0 && pl == kDnSuh) {   /* dn_suh: down's suh handle of the same expert */
                CHECK(o->w[i] == dns.front()->w[(size_t)(q * kDnTables * half + 1 * half + (int64_t)i % half)]);
                CHECK(wname(b, o->w[i]) == "blk.0.ffn_down_exps." + std::to_string(2 * ((int64_t)i % half) + q) + ".suh");
                continue;
            }
            const int64_t expert = 2 * ((int64_t)i % half) + q;
            const PlaneWant& pw = want[pl];
            const bool codes = pl == 0;
            const int64_t w = q == 0 ? w_even : w_odd, lo = q == 0 ? lo_even : lo_odd;
            const std::string& name = wname(b, o->w[i]);
            const RadWeightDecl& d = wdecl(b, o->w[i]);
            const std::string stem = proj == 0 ? "blk.0.ffn_gate_up_exps." : "blk.0.ffn_down_exps.";
            const std::string weight = stem + std::to_string(expert) + ".weight";
            CHECK(name == (codes ? weight : stem + std::to_string(expert) + "." + pw.role));
            CHECK(codes ? d.source == nullptr : (d.source && weight == d.source));
            CHECK(d.planes && std::string(d.planes) == pw.role);
            CHECK_EQ(d.dtype, codes ? RAD_U4 : RAD_F16);
            CHECK_EQ(d.access, RAD_ACCESS_CONDITIONAL);
            const bool per_row = std::string(pw.role) == "svh", per_col = !codes && !per_row;
            const int64_t rows = proj == 0 ? (per_col ? 1 : 2 * w) : (per_col ? 1 : kEmbd);
            const int64_t cols = proj == 0 ? (per_row ? 1 : kEmbd) : (per_row ? 1 : w);
            CHECK_EQ(d.shape[0], rows);
            CHECK_EQ(d.shape[1], cols);
            CHECK_EQ(d.shard, !pw.cut ? RAD_SHARD_NONE : proj == 0 ? RAD_SHARD_ROW : RAD_SHARD_COL);
            if (proj == 0 && pw.cut) CHECK(d.n_row_parts == 2 && d.row_parts[0] == w && d.row_parts[1] == w);
            const RecSpan* s = span_of(b, o->w[i]);
            if (split && pw.cut) CHECK(s && s->lo == lo && s->hi == lo + w);
            else                 CHECK(s == nullptr);
            if (i % half) CHECK(o->w[i] == o->w[i - 1] + 1);   /* each (plane, class) one run */
        }
    }
}

/* ---- one rank, the shipped geometry. Each expert's projection is one view a plane (D15); the
 * intermediate is one f16 plane; no fp8 activation op exists anywhere. */
void tp1_declares_exl3_experts_and_no_activation_quant() {
    RadModelMeta m = meta();
    RadBuildCtx c = ctx(0, 1);
    RadBuilder b;
    exl3_model(b);
    REQUIRE_EQ(qwen4exp_exl3::declare(&b, &m, &c), RAD_OK);
    check_exl3_ops(b, 2 * kWidth, 2 * kWidth, kWidth, kWidth, kExperts, "exl3k4");
    check_views(b, false, kWidth, kWidth, 0, 0);
    const RadBufDecl* eff = buf_named(b, "moe_eff");
    CHECK(eff && eff->dtype == RAD_F16 && eff->shape[0] == 2048 * kTopK && eff->shape[1] == kWidth);
    for (const auto& [name, d] : b.bufs)   /* no gate/up plane, no rotated input pair */
        CHECK(name.find("moe_egu") == std::string::npos && name.find("moe_hr") == std::string::npos);
    CHECK(b.spans.empty());   /* nothing is sliced at one rank */
    CHECK(!qwen4exp_exl3::g_model[0].moecfg.expert_rot);
}

/* ---- two ranks: the parity split, 384 and 256 columns, the wide slice alternating with parity;
 * each plane cut only along the dimension its block follows. */
void tp2_splits_every_expert_384_256_by_parity() {
    RadModelMeta m = meta();
    for (int r = 0; r < 2; ++r) {
        RadBuildCtx c = ctx(r, 2);
        RadBuilder b;
        exl3_model(b);
        REQUIRE_EQ(qwen4exp_exl3::declare(&b, &m, &c), RAD_OK);
        const int64_t even = r == 0 ? 384 : 256, odd = r == 0 ? 256 : 384;
        check_exl3_ops(b, 2 * even, 2 * odd, even, odd, kExperts, "exl3k4");
        check_views(b, true, even, odd, r == 0 ? 0 : 384, r == 0 ? 384 : 0);
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
    const auto gu = gate_ups(b, 0);
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
    exl3_model(b, { { "blk.1.ffn_gate_up_exps.", exl3_enc(5, true) } });
    REQUIRE_EQ(qwen4exp_exl3::declare(&b, &m, &c), RAD_OK);
    const auto gu = gate_ups(b, 0), dn = ops_named(b, "moe_exl3_down");
    REQUIRE_EQ(gu.size(), kLayers + 1);
    REQUIRE_EQ(dn.size(), kLayers + 1);
    CHECK(param_s(*gu[1], "dtype") == "exl3k5");
    CHECK(param_s(*dn[1], "dtype") == "exl3k4");
    CHECK(param_s(*gu[0], "dtype") == "exl3k4");
    CHECK_EQ(wdecl(b, gu[1]->w[0]).dtype, RAD_U5);
    /* gate_up_down has one dtype: not declared for the layer whose halves differ, which issues its pair */
    const auto gd = ops_named(b, "moe_exl3_gate_up_down");
    CHECK_EQ(gd.size(), kGd ? kLayers : 0);
    for (const RecOp* o : gd) CHECK(param_s(*o, "dtype") == "exl3k4");
    CHECK_EQ(qwen4exp_exl3::g_model[0].layers[1].mlp.op_gd, 0);
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

/* ---- a trunk the recipe leaves plain bf16 is served bf16 throughout (Geom::w_bf16): the
 * connections write no codes and the attention's gate is sigmoid + mul, not gate_quant_fp8. The
 * in-tree plugin decides this from the experts, which are never plain here; this one from the trunk. */
void a_bf16_trunk_is_served_bf16_throughout() {
    RadModelMeta m = meta();
    RadBuildCtx c = ctx(0, 1);
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    RadBuilder b;
    exl3_model(b, { { "ssm_", plain }, { "attn_", plain }, { "_shexp", plain }, { "qsa_qk", plain } });
    REQUIRE_EQ(qwen4exp_exl3::declare(&b, &m, &c), RAD_OK);
    CHECK(qwen4exp_exl3::g_model[0].g.w_bf16);
    CHECK_EQ(qwen4exp_exl3::g_model[0].hccfg.quant, 0);
    CHECK_EQ(count_op(b, "gate_quant_fp8"), 0);
    CHECK(count_op(b, "sigmoid") > 0);
    check_exl3_ops(b, 2 * kWidth, 2 * kWidth, kWidth, kWidth, kExperts, "exl3k4");
    RadBuilder q;   /* nothing known about the trunk: not bf16, as in-tree (block fp8) */
    exl3_model(q);
    REQUIRE_EQ(qwen4exp_exl3::declare(&q, &m, &c), RAD_OK);
    CHECK(!qwen4exp_exl3::g_model[0].g.w_bf16);
    CHECK_EQ(qwen4exp_exl3::g_model[0].hccfg.quant, 1);
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
    exl3_model(b, { { "blk.2.ffn_down_exps.9.weight", exl3_enc(5, false) } });
    int st = 0;
    const std::string err = stderr_of([&] { st = qwen4exp_exl3::declare(&b, &m, &c); });
    CHECK_EQ(st, RAD_E_DTYPE);
    CHECK(has(err, "blk.2.ffn_down_exps.9.weight"));
}
void an_exl3_encoding_the_quantiser_would_not_write_is_refused() {
    RadModelMeta m = meta();
    RadBuildCtx c = ctx(0, 1);
    RadBuilder b;
    RadEncoding odd_order = exl3_enc(4, true);             /* suh.1 before svh: not P1's spelling */
    std::swap(odd_order.plane[2], odd_order.plane[3]);
    exl3_model(b, { { "ffn_gate_up_exps", odd_order } });
    int st = 0;
    const std::string err = stderr_of([&] { st = qwen4exp_exl3::declare(&b, &m, &c); });
    CHECK_EQ(st, RAD_E_DTYPE);
    CHECK(has(err, "serves the routed experts as EXL3"));
}
void an_unstacked_gate_up_is_refused() {
    RadModelMeta m = meta();
    RadBuildCtx c = ctx(0, 1);
    RadBuilder b;
    exl3_model(b, { { "blk.3.ffn_gate_up_exps.", exl3_enc(4, false) } });   /* no suh.1 */
    int st = 0;
    const std::string err = stderr_of([&] { st = qwen4exp_exl3::declare(&b, &m, &c); });
    CHECK_EQ(st, RAD_E_DTYPE);
    CHECK(has(err, "blk.3.ffn_gate_up_exps.0.weight") && has(err, "stacked"));
}
void a_trellis_width_with_no_op_is_refused() {
    RadModelMeta m = meta();
    RadBuildCtx c = ctx(0, 1);
    RadBuilder b;
    exl3_model(b, { { "ffn_gate_up_exps", exl3_enc(1, true) } });   /* K=1: no op dtype exists */
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
/* With this plugin installed, the core routes an imported GPTQ / AutoRound / Quark qwen4exp to it
 * (the family's one quantised plugin); it refuses by name before declaring anything. */
void an_imported_quantisation_is_refused_by_name() {
    RadModelMeta m = meta();
    std::vector<const char*> keys(m.kv_key, m.kv_key + m.n_kv), vals(m.kv_val, m.kv_val + m.n_kv);
    keys.push_back("imported_quantization_config.quant_method");
    vals.push_back("gptq");
    m.n_kv = (int)keys.size(); m.kv_key = keys.data(); m.kv_val = vals.data();
    RadBuildCtx c = ctx(0, 1);
    RadBuilder b;
    exl3_model(b);
    int st = 0;
    const std::string err = stderr_of([&] { st = qwen4exp_exl3::declare(&b, &m, &c); });
    CHECK_EQ(st, RAD_E_UNSUPPORTED);
    CHECK(has(err, "('gptq') is imported") && has(err, "qwen4exp_fp8"));
    CHECK(b.ops.empty() && b.weights.empty());
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

/* ---- the MoE block's issues on one pass of `n` decode rows: the routed ops over the f16
 * intermediate, each with the even and the odd table, in interface v4's operand order written out,
 * nothing quantised between; and THE ISSUE ORDER -- in the split build at n <= 16, prep_gu, then
 * the shared arm's gate_up and quantiser, then gate_up(a_rot = 1) over xr; otherwise the one
 * gate_up (a_rot = 0) over the bf16 input with the shared arm behind it. The trunk's first layer
 * and the MTP head's block are stepped; the rest of the model is stock code. */
void check_moe_pass(const RadBuilder& b, const qwen4exp_exl3::Model& mod, int n) {
    std::vector<int32_t> out((size_t)n);
    for (int i = 0; i < n; ++i) out[(size_t)i] = i;
    RadBatch batch{};
    batch.phase = RAD_PHASE_DECODE; batch.n_tok = n; batch.n_seq = n; batch.max_q_len = 1;
    batch.max_ctx_len = 100; batch.n_out = n; batch.out_ids = out.data();
    const bool gdp = kGd && n >= kGdMin && n <= kPrepBand;
    const bool split = kSplit && n <= kPrepBand && !gdp;
    const MoeEXL3* trunk = &mod.layers[0].mlp;
    for (const MoeEXL3* blk : { trunk, &mod.mtp.mlp }) {
        RadCtx cx;
        blk->step(&cx, &batch);
        CHECK(cx.step_fail.empty());
        int at_pg = -1, at_gu = -1, at_dn = -1, at_shgu = -1, at_shq = -1, n_gu = 0, n_dn = 0, at_gd = -1, n_gd = 0;
        rad_buf xr = 0;
        for (size_t i = 0; i < cx.issues.size(); ++i) {
            const RecIssue& is = cx.issues[i];
            const RecOp& ro = b.ops[is.op - 1];
            const std::string& op = ro.op;
            for (const char* f : kFp8Ops) CHECK(op != f);
            /* the shared arm's gate_up writes moe_sgu, its quantiser reads it */
            for (rad_buf w : ro.writes) if (bname(b, w) == "moe_sgu") at_shgu = (int)i;
            for (rad_buf r : ro.reads)  if (bname(b, r) == "moe_sgu") at_shq = (int)i;
            if (op == "moe_exl3_prep_gu") {
                at_pg = (int)i;
                /* a, suh, suh1, sorted, offsets, suh_odd, suh1_odd -> xr; the tables gate_up's */
                CHECK_EQ(is.n, n);
                REQUIRE_EQ(is.opd.size(), 8);
                CHECK(is.opd[0].kind == RAD_OPK_BUF && bname(b, is.opd[0].handle) == "x");
                for (size_t q = 0; q < 2; ++q)
                    for (size_t t = 0; t < 2; ++t) {
                        const RadOperand& o = is.opd[(q == 0 ? 1 : 5) + t];
                        CHECK(o.kind == RAD_OPK_WTAB && o.handle == blk->w_gu[1 + t][q] && o.rows == kExperts / 2);
                    }
                CHECK(bname(b, is.opd[3].handle) == "moe_sorted" && is.opd[3].rows == n * kTopK);
                CHECK(bname(b, is.opd[4].handle) == "moe_eoff" && is.opd[4].rows == kExperts + 1);
                xr = is.opd[7].handle;
                CHECK(has(bname(b, xr), "moe_exl3_xr") && is.opd[7].rows == n * kTopK);
                continue;
            }
            if (op == "moe_exl3_gate_up_down") {
                /* x, gate_up's five tables then down's codes and svh (even), routing, the same
                 * (odd), moe_edn */
                at_gd = (int)i;
                ++n_gd;
                const size_t np = kGuTables + 2;
                CHECK_EQ(is.n, n);
                REQUIRE_EQ(is.opd.size(), 1 + 2 * np + 3);
                CHECK(is.opd[0].kind == RAD_OPK_BUF && bname(b, is.opd[0].handle) == "x");
                for (size_t q = 0; q < 2; ++q)
                    for (size_t pl = 0; pl < np; ++pl) {
                        const RadOperand& o = is.opd[(q == 0 ? 1 : 3 + np) + pl];
                        const std::vector<rad_weight>& run = pl == (size_t)kDnSuh ? blk->w_dn[1]
                                                           : pl < (size_t)kGuTables ? blk->w_gu[pl]
                                                           : blk->w_dn[pl == (size_t)kGuTables ? 0 : 2];
                        CHECK(o.kind == RAD_OPK_WTAB && o.handle == run[q] && o.rows == kExperts / 2);
                        CHECK_EQ(expert_of(wname(b, o.handle)), (long long)q);
                    }
                CHECK(bname(b, is.opd[1 + np].handle) == "moe_sorted" && is.opd[1 + np].rows == n * kTopK);
                CHECK(bname(b, is.opd[2 + np].handle) == "moe_eoff" && is.opd[2 + np].rows == kExperts + 1);
                CHECK(bname(b, is.opd[3 + 2 * np].handle) == "moe_edn" && is.opd[3 + 2 * np].rows == n * kTopK);
                continue;
            }
            if (op != "moe_exl3_gate_up" && op != "moe_exl3_down") continue;
            const bool gu = op == "moe_exl3_gate_up";
            (gu ? at_gu : at_dn) = (int)i;
            ++(gu ? n_gu : n_dn);
            if (gu) CHECK_EQ(param_i(ro, "a_rot"), split ? 1 : 0);
            /* the input, the even experts' tables, the routing, the odd experts' tables, the
             * output; gate_up's tables end with dn_suh, down's suh run */
            const size_t np = gu ? kGuTables : kDnTables;
            CHECK_EQ(is.n, n);
            REQUIRE_EQ(is.opd.size(), 1 + 2 * np + 3);
            CHECK(is.opd[0].kind == RAD_OPK_BUF);
            if (gu && split) CHECK(xr && is.opd[0].handle == xr && is.opd[0].rows == n * kTopK);
            else             CHECK(bname(b, is.opd[0].handle) == (gu ? "x" : "moe_eff"));
            const size_t at_sorted = 1 + np, at_odd = 3 + np, at_out = 3 + 2 * np;
            for (size_t q = 0; q < 2; ++q)
                for (size_t pl = 0; pl < np; ++pl) {
                    const RadOperand& o = is.opd[(q == 0 ? 1 : at_odd) + pl];
                    const std::vector<rad_weight>& run =
                        gu ? (pl == (size_t)kDnSuh ? blk->w_dn[1] : blk->w_gu[pl]) : blk->w_dn[pl];
                    CHECK(o.kind == RAD_OPK_WTAB && o.handle == run[q] && o.rows == kExperts / 2);
                    CHECK_EQ(expert_of(wname(b, o.handle)), (long long)q);
                    const RadWeightDecl& d = wdecl(b, o.handle);
                    CHECK(d.planes && std::string(d.planes) == (gu ? kGu : kDn)[pl].role);
                }
            CHECK(bname(b, is.opd[at_sorted].handle) == "moe_sorted" && is.opd[at_sorted].rows == n * kTopK);
            CHECK(bname(b, is.opd[at_sorted + 1].handle) == "moe_eoff" && is.opd[at_sorted + 1].rows == kExperts + 1);
            CHECK(bname(b, is.opd[at_out].handle) == (gu ? "moe_eff" : "moe_edn") && is.opd[at_out].rows == n * kTopK);
        }
        if (gdp) CHECK(n_gd == 1 && n_gu == 0 && n_dn == 0);
        else     CHECK(n_gd == 0 && n_gu == 1 && n_dn == 1 && at_dn > at_gu);
        CHECK_EQ(at_pg >= 0, split);
        const bool shared = blk->op_sgate != 0;
        if (blk == trunk) CHECK(shared);        /* else the order below would check nothing */
        CHECK_EQ(at_shgu >= 0, shared);
        if (shared) {
            CHECK(at_shq > at_shgu);
            if (gdp)        CHECK(at_shq < at_gd);   /* the shared arm ahead of the persistent launch */
            else if (split) CHECK(at_pg < at_shgu && at_shq < at_gu);
            else            CHECK(at_gu < at_shgu && at_shq < at_dn);
        }
    }
}
void decode_issues_gate_up_then_down_over_both_parity_tables() {
    RadModelMeta m = meta();
    RadBuildCtx c = ctx(0, 1);
    RadBuilder b;
    exl3_model(b);
    REQUIRE_EQ(qwen4exp_exl3::declare(&b, &m, &c), RAD_OK);
    CHECK(qwen4exp_exl3::g_model[0].have_mtp);
    check_moe_pass(b, qwen4exp_exl3::g_model[0], 4);
    check_moe_pass(b, qwen4exp_exl3::g_model[0], 2);   /* below gate_up_down's rows: the band's own form */
    check_moe_pass(b, qwen4exp_exl3::g_model[0], (int)kGdMin);       /* its first row count */
    check_moe_pass(b, qwen4exp_exl3::g_model[0], (int)kGdMin - 1);   /* the last below it */
    CHECK_EQ(g_device_calls, 0);
    (void)&qwen4exp_exl3::step;    /* the model's step is stock code; its MoE block's is checked */
    (void)&qwen4exp_exl3::probe;
}
/* ---- radiance 1.4.0's NEXT-LAYER EXPERT PREFETCH (rad_runtime.h RadPrefetchNext): every routed
 * layer links the next; a layer's description names, for each model expert, the planes the core may
 * point at a copy -- each of them in EXACTLY ONE declared op's weight list (the core refuses an expert
 * otherwise, core/runtime/xprefetch.cpp xp_bind), codes first; and a decode pass predicts at the
 * router input, steps ahead of the routed GEMMs, marks after down. */
void the_next_layer_prefetch_is_described_and_called() {
    RadModelMeta m = meta();
    RadBuildCtx c = ctx(0, 1);
    RadBuilder b;
    exl3_model(b);
    REQUIRE_EQ(qwen4exp_exl3::declare(&b, &m, &c), RAD_OK);
    const auto& mod = qwen4exp_exl3::g_model[0];
    const size_t nl = mod.layers.size();
    REQUIRE_EQ(nl > 1, true);
    const auto tables_holding = [&](rad_weight h) {
        int n = 0;
        for (const RecOp& o : b.ops)
            for (rad_weight w : o.w) if (w == h) { ++n; break; }
        return n;
    };
    for (size_t l = 0; l < nl; ++l) {
        const MoeEXL3& blk = mod.layers[l].mlp;
        CHECK(blk.xp_next == (l + 1 < nl ? &mod.layers[l + 1].mlp : nullptr));
        if (blk.op_gd) {   /* gate_up_down also reads down's codes and svh: no plane is in one table */
            CHECK_EQ(blk.xp.n_parts, 0);
            CHECK(blk.xp_parts.empty());
            continue;
        }
        CHECK_EQ(blk.xp.n_parts, 4);
        CHECK_EQ(blk.xp.layer, blk.layer);
        CHECK(blk.xp.router == blk.w_router && blk.xp.n_expert == kExperts && blk.xp.top_k == kTopK);
        CHECK(blk.xp.parts == blk.xp_parts.data() && blk.xp.count_at == blk.xp_count_at.data());
        for (int64_t e = 0; e < kExperts; ++e) {
            const int32_t s = blk.xp_count_at[(size_t)e];
            CHECK_EQ(s, (int32_t)e);   /* no protected experts here: the place is the number */
            if (s < 0) continue;
            const rad_weight* pt = &blk.xp_parts[(size_t)(e * 4)];
            /* the build's parts: the split decode form (default) and gate_up_down put gate_up's planes in
             * two ops, so down's codes and svh only; the plain build gate_up's codes and svh as well */
            const bool gu_in_one = !kSplit;
            const rad_weight want[4] = {
                gu_in_one ? blk.w_gu[0][(size_t)s] : blk.w_dn[0][(size_t)s],
                gu_in_one ? blk.w_dn[0][(size_t)s] : blk.w_dn[2][(size_t)s],
                gu_in_one ? blk.w_gu[3][(size_t)s] : 0,
                gu_in_one ? blk.w_dn[2][(size_t)s] : 0 };
            for (int k = 0; k < 4; ++k) {
                CHECK(pt[k] == want[k]);
                if (pt[k]) CHECK_EQ(tables_holding(pt[k]), 1);
            }
        }
    }
    /* a decode pass of the first layer: predict (for layer 1) before anything of the routed arm is
     * issued past the router, step (layer 0) before the first routed GEMM, mark after the last */
    std::vector<int32_t> out(4);
    for (int i = 0; i < 4; ++i) out[(size_t)i] = i;
    RadBatch batch{};
    batch.phase = RAD_PHASE_DECODE; batch.n_tok = 4; batch.n_seq = 4; batch.max_q_len = 1;
    batch.max_ctx_len = 100; batch.n_out = 4; batch.out_ids = out.data();
    RadCtx cx;
    mod.layers[0].mlp.step(&cx, &batch);
    CHECK(cx.step_fail.empty());
    const bool predicts = mod.layers[1].mlp.xp.n_parts > 0;   /* a layer with no description: nobody */
    REQUIRE_EQ(cx.xp.size(), predicts ? 3 : 2);
    if (!predicts) cx.xp.insert(cx.xp.begin(), RadCtx::XpCall{'p', 0, mod.layers[1].mlp.layer});
    CHECK(cx.xp[0].kind == 'p' && cx.xp[0].layer == mod.layers[1].mlp.layer);
    CHECK(cx.xp[1].kind == 's' && cx.xp[1].layer == mod.layers[0].mlp.layer);
    CHECK(cx.xp[2].kind == 'm');
    size_t first_routed = cx.issues.size(), last_routed = 0;
    for (size_t i = 0; i < cx.issues.size(); ++i) {
        const std::string& op = b.ops[cx.issues[i].op - 1].op;
        if (op.rfind("moe_exl3_", 0) == 0) { first_routed = std::min(first_routed, i); last_routed = i; }
    }
    CHECK(cx.xp[0].at <= cx.xp[1].at && cx.xp[1].at <= first_routed && cx.xp[2].at == last_routed + 1);
    /* the last layer predicts for nobody */
    RadCtx cl;
    mod.layers[nl - 1].mlp.step(&cl, &batch);
    CHECK(cl.xp.size() == 2 && cl.xp[0].kind == 's' && cl.xp[1].kind == 'm');
    CHECK_EQ(g_device_calls, 0);
}
/* ---- the band's edge: 16 rows take the split form (in the split build), 17 the one gate_up. */
void the_decode_band_ends_at_16_rows() {
    RadModelMeta m = meta();
    RadBuildCtx c = ctx(0, 1);
    RadBuilder b;
    exl3_model(b);
    REQUIRE_EQ(qwen4exp_exl3::declare(&b, &m, &c), RAD_OK);
    check_moe_pass(b, qwen4exp_exl3::g_model[0], 16);
    check_moe_pass(b, qwen4exp_exl3::g_model[0], 17);
}

/* ---- THE ENGINE KEYS A ROUTED LAYER'S STAGING ON ITS FIRST DECLARED EXPERT OP (core/place/
 * stager.cpp:23-31, first_use_op): the prefill stager copies the next layer's pooled experts and
 * orders the layer behind its own copies at that op. Every pass the stager arms for (above the top
 * arena level) issues the one gate_up, so it has to be that op, in both builds: no op declared
 * before a layer's gate_up(a_rot = 0) reads any weight it reads (measured when it was not: P6's
 * order-1 prefill staged one layer a pass). */
void the_one_gate_up_is_every_layers_first_declared_expert_reader() {
    RadModelMeta m = meta();
    for (int r = 0; r < 2; ++r) {
        RadBuildCtx c = ctx(r, 2);
        RadBuilder b;
        exl3_model(b);
        REQUIRE_EQ(qwen4exp_exl3::declare(&b, &m, &c), RAD_OK);
        const auto gu0 = gate_ups(b, 0);
        REQUIRE_EQ(gu0.size(), kLayers + 1);
        for (const RecOp* z : gu0) {
            const size_t zi = (size_t)(z - b.ops.data());
            const std::set<rad_weight> zw(z->w.begin(), z->w.end());
            for (size_t i = 0; i < zi; ++i)
                for (rad_weight x : b.ops[i].w)
                    if (zw.count(x)) {
                        fprintf(stderr, "  op %zu '%s' reads '%s' before gate_up(a_rot = 0), op %zu\n", i,
                                b.ops[i].op.c_str(), wname(b, x).c_str(), zi);
                        CHECK(!"an op declared before the one gate_up reads its experts");
                    }
        }
    }
}

/* ---- ...AND ITS LAST: the prefill stager lets a layer's buffer go after the last declared op that
 * reads its experts (stager.cpp after_op), so that op must be issued in every pass the stager arms
 * for -- down. A decode-band op declared after it (gate_up_down at 58905d9) held the buffers for the
 * whole pass: 32 layers staged in an 8K prefill against 768 (the engine's stats). */
void down_is_every_layers_last_declared_expert_reader() {
    RadModelMeta m = meta();
    for (int r = 0; r < 2; ++r) {
        RadBuildCtx c = ctx(r, 2);
        RadBuilder b;
        exl3_model(b);
        REQUIRE_EQ(qwen4exp_exl3::declare(&b, &m, &c), RAD_OK);
        const auto dn = ops_named(b, "moe_exl3_down");
        REQUIRE_EQ(dn.size(), kLayers + 1);
        for (const RecOp* d : dn) {
            const size_t di = (size_t)(d - b.ops.data());
            const std::set<rad_weight> dw(d->w.begin(), d->w.end());
            for (size_t i = di + 1; i < b.ops.size(); ++i)
                for (rad_weight x : is_stage_off(b.ops[i]) ? std::vector<rad_weight>{} : b.ops[i].w)
                    if (dw.count(x)) {
                        fprintf(stderr, "  op %zu '%s' reads '%s' after down, op %zu\n", i, b.ops[i].op.c_str(),
                                wname(b, x).c_str(), di);
                        CHECK(!"an op declared after down reads its experts");
                    }
        }
    }
}

/* ---- THE ROUTED LAYERS' SPANS of declared expert readers: the prefill stager turns itself off when two overlap
 * (stager.cpp init). With staging on (every real build) no two may; the PREFILL_STAGE 0 build makes layers 0 and 1
 * overlap on purpose, with its one never-issued reader. */
void routed_layer_spans_interleave_only_with_staging_off() {
    RadModelMeta m = meta();
    RadBuildCtx c = ctx(0, 1);
    RadBuilder b;
    exl3_model(b);
    REQUIRE_EQ(qwen4exp_exl3::declare(&b, &m, &c), RAD_OK);
    std::map<long long, std::pair<size_t, size_t>> span;   /* layer -> [first, last] op index reading its experts */
    for (size_t i = 0; i < b.ops.size(); ++i)
        for (rad_weight x : b.ops[i].w) {
            const std::string& n = wname(b, x);
            if (n.compare(0, 4, "blk.") || n.find("_exps.") == std::string::npos) continue;
            const long long l = std::atoll(n.c_str() + 4);
            auto it = span.find(l);
            if (it == span.end()) span[l] = { i, i };
            else it->second.second = i;
        }
    REQUIRE_EQ(span.size() >= 2, true);
    std::vector<std::pair<size_t, size_t>> v;
    for (const auto& kv : span) v.push_back(kv.second);
    std::sort(v.begin(), v.end());
    int overlaps = 0;
    for (size_t i = 1; i < v.size(); ++i) overlaps += v[i].first <= v[i - 1].second;
    if (kStageOn) CHECK_EQ(overlaps, 0);
    else {
        CHECK(overlaps >= 1);
        CHECK(span[0].second > span[1].first);
        int offs = 0;
        size_t last_off = 0, last_other = 0;   /* appended: after every other op of the language model */
        for (size_t i = 0; i < b.ops.size(); ++i) {
            if (is_stage_off(b.ops[i])) { ++offs; last_off = i; }
            else last_other = i;
        }
        CHECK_EQ(offs, 1);
        CHECK(last_off > last_other);
    }
}

/* ---- the split build refuses --weights-disk-tier by name: the file stager reads a layer's
 * file-tier experts at the layer's first declared expert op on every pass, which the decode band
 * does not issue there. Checked by running this binary again with the flag on its command line,
 * where the block reads it (/proc/self/cmdline); the build at 0 declares as always. */
int    g_argc = 0;
char** g_argv = nullptr;
void the_split_build_refuses_the_disk_tier_by_name() {
    bool flagged = false;
    for (int i = 2; i < g_argc; ++i) flagged = flagged || !std::strcmp(g_argv[i], "--weights-disk-tier");
    if (!flagged) {
        char self[4096];
        const ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
        if (n <= 0) { printf("  no /proc/self/exe: nothing to check\n"); std::exit(77); }
        self[n] = 0;
        const std::string cmd = std::string("'") + self + "' the_split_build_refuses_the_disk_tier_by_name "
                                "--weights-disk-tier";
        const int st = std::system(cmd.c_str());
        CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0);
        return;
    }
    RadModelMeta m = meta();
    RadBuildCtx c = ctx(0, 1);
    RadBuilder b;
    exl3_model(b);
    int st = 0;
    const std::string err = stderr_of([&] { st = qwen4exp_exl3::declare(&b, &m, &c); });
    if (kSplit || kGd) {
        CHECK_EQ(st, RAD_E_UNSUPPORTED);
        CHECK(has(err, "--weights-disk-tier") &&
              has(err, kGd ? "QWEN4EXP_EXL3_GD=1" : "QWEN4EXP_EXL3_PREP_SPLIT=1"));
    } else {
        CHECK_EQ(st, RAD_OK);
    }
}

/* ---- THE SHIPPED DEFAULT is the split order (QWEN4EXP_EXL3_PREP_SPLIT 1, picked by engine step
 * time). Meaningful only in the binary built with no define (arch_static_test_default); the two
 * built at an explicit order skip it. */
void the_shipped_default_is_the_split_order() {
#if defined(ARCH_STATIC_DEFAULT_BUILD)
    /* and the decode chain as one op from 6 tokens, with radiance's prefill stager left on */
    CHECK(kSplit);
    CHECK(kGd);
    CHECK_EQ(kGdMin, (int64_t)6);
    CHECK(kStageOn);
#else
    printf("  built at an explicit order: the default is not visible here\n");
    std::exit(77);
#endif
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
    { "a_bf16_trunk_is_served_bf16_throughout", a_bf16_trunk_is_served_bf16_throughout },
    { "a_non_exl3_expert_is_refused_by_name", a_non_exl3_expert_is_refused_by_name },
    { "a_second_trellis_width_in_one_table_is_refused", a_second_trellis_width_in_one_table_is_refused },
    { "a_trellis_width_with_no_op_is_refused", a_trellis_width_with_no_op_is_refused },
    { "an_exl3_encoding_the_quantiser_would_not_write_is_refused", an_exl3_encoding_the_quantiser_would_not_write_is_refused },
    { "an_unstacked_gate_up_is_refused", an_unstacked_gate_up_is_refused },
    { "plain_bf16_experts_are_the_in_tree_plugins", plain_bf16_experts_are_the_in_tree_plugins },
    { "a_builder_with_no_model_is_refused", a_builder_with_no_model_is_refused },
    { "an_imported_quantisation_is_refused_by_name", an_imported_quantisation_is_refused_by_name },
    { "four_ranks_are_refused_by_name", four_ranks_are_refused_by_name },
    { "decode_issues_gate_up_then_down_over_both_parity_tables", decode_issues_gate_up_then_down_over_both_parity_tables },
    { "the_decode_band_ends_at_16_rows", the_decode_band_ends_at_16_rows },
    { "the_next_layer_prefetch_is_described_and_called", the_next_layer_prefetch_is_described_and_called },
    { "the_one_gate_up_is_every_layers_first_declared_expert_reader", the_one_gate_up_is_every_layers_first_declared_expert_reader },
    { "down_is_every_layers_last_declared_expert_reader", down_is_every_layers_last_declared_expert_reader },
    { "routed_layer_spans_interleave_only_with_staging_off", routed_layer_spans_interleave_only_with_staging_off },
    { "the_split_build_refuses_the_disk_tier_by_name", the_split_build_refuses_the_disk_tier_by_name },
    { "the_shipped_default_is_the_split_order", the_shipped_default_is_the_split_order },
    { "calibration_taps_the_inputs_the_ops_are_handed", calibration_taps_the_inputs_the_ops_are_handed },
};

}  /* namespace */

/* `arch_static_test --list` prints the cases; `arch_static_test NAME` runs one. Exit 0 PASS,
 * 1 FAIL, 77 SKIP (ctest's SKIP_RETURN_CODE), 2 no such case. */
int main(int argc, char** argv) {
    g_argc = argc;
    g_argv = argv;
    if (argc == 2 && !std::strcmp(argv[1], "--list")) {
        for (const Case& k : kCases) printf("%s\n", k.name);
        return 0;
    }
    /* Arguments after CASE are ignored here: a case may run this binary again with a serve flag on
     * its command line (the_split_build_refuses_the_disk_tier_by_name). */
    if (argc < 2) { fprintf(stderr, "usage: %s --list | CASE [ARGS...]\n", argv[0]); return 2; }
    for (const Case& k : kCases)
        if (!std::strcmp(argv[1], k.name)) {
            k.fn();
            printf("%s %s\n", g_failed ? "FAIL" : "PASS", k.name);
            return g_failed ? 1 : 0;
        }
    fprintf(stderr, "no case '%s' (--list)\n", argv[1]);
    return 2;
}
