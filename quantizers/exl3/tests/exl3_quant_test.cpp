/* exl3_quant_test.cpp -- the `exl3` quantiser through its plugin ABI, the way rad-convert drives it: dlopen the
 * built .so, take rad_quant_at(0), call encoding / row_block / quantize / decode.
 *
 *   exl3_quant_test <exl3_quant.so> <fixture dir from tools/exl3_fixture.py --defects>
 *
 * What it holds the plugin to: the encoding comes from name + shape + options alone; the import is EXACT (the
 * codes plane re-tiled is the source trellis byte for byte, suh/svh are the source's); the decode equals the
 * fixture's dense experts -- computed independently, by common/exl3_ref.py -- to bf16 rounding; every R12
 * refusal returns its status AND says what it refused, each beside a positive control that succeeds. */
#include "rad_quant.h"

#include "exl3_check.h"
#include "exl3_encoding.h"
#include "exl3_layout.h"
#include "exl3_src.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

const RadQuantizerInfo* Q = nullptr;
std::string FIX;   /* the fixture directory */
enum { H = 256, W = 128, E = 8, LAYERS = 2, K = 4 };

struct Opts {
    std::vector<RadParam> p;
    std::vector<std::string> keep;
    Opts& s(const char* k, const std::string& v) {
        keep.push_back(v);
        RadParam r{};
        r.key = k;
        r.kind = RAD_P_STR;
        r.sval = nullptr;
        p.push_back(r);
        return *this;
    }
    Opts& i(const char* k, long long v) {
        RadParam r{};
        r.key = k;
        r.kind = RAD_P_INT;
        r.ival = v;
        p.push_back(r);
        return *this;
    }
    const RadParam* data() {   /* string pointers are fixed up last: `keep` may have reallocated */
        size_t j = 0;
        for (auto& r : p)
            if (r.kind == RAD_P_STR) r.sval = keep[j++].c_str();
        return p.data();
    }
    int n() const { return (int)p.size(); }
};

Opts std_opts(const std::string& src, int bits = K) {
    Opts o;
    o.s("src", src).i("bits", bits).s("codebook", "mul1");
    return o;
}

RadQuantWeight weight(const std::string& name, int64_t rows, int64_t cols) {
    RadQuantWeight w{};
    w.name = strdup(name.c_str());   /* tests only; lives to exit */
    w.rank = 2;
    w.shape[0] = rows;
    w.shape[1] = cols;
    w.rows = rows;
    w.cols = cols;
    w.layer = w.expert = -1;
    return w;
}

/* Run fn with stderr captured. Single-threaded use only. */
template <class F> std::string captured(F fn) {
    char path[] = "/tmp/exl3_quant_test_XXXXXX";
    const int fd = mkstemp(path);
    std::fflush(stderr);
    const int saved = dup(2);
    dup2(fd, 2);
    fn();
    std::fflush(stderr);
    dup2(saved, 2);
    close(saved);
    std::string out;
    char buf[4096];
    lseek(fd, 0, SEEK_SET);
    for (ssize_t n; (n = read(fd, buf, sizeof buf)) > 0;) out.append(buf, (size_t)n);
    close(fd);
    unlink(path);
    std::fputs(out.c_str(), stderr);   /* still shown in the test log */
    return out;
}

struct Result {
    int status = RAD_OK;
    RadEncoding enc{};
    std::vector<std::vector<uint8_t>> planes;
    std::string said;
};

Result quantise(Opts o, const std::string& name, int64_t rows, int64_t cols, bool capture = true) {
    Result r;
    RadQuantWeight w = weight(name, rows, cols);
    r.status = Q->encoding(o.data(), o.n(), &w, &r.enc);
    if (r.status != RAD_OK) return r;
    std::vector<void*> ptr;
    for (int i = 0; i < r.enc.n_planes; ++i) {
        r.planes.emplace_back((size_t)rad_enc_plane_bytes(&r.enc.plane[i], rows, cols), 0);
        ptr.push_back(r.planes.back().data());
    }
    std::vector<float> src((size_t)(rows * cols), NAN);   /* ignored (D2): NaN shows it */
    auto call = [&] { r.status = Q->quantize(o.data(), o.n(), &w, src.data(), 0, rows, ptr.data()); };
    if (capture) r.said = captured(call);
    else call();
    return r;
}

std::vector<float> decoded(const Result& r, const std::string& name, int64_t rows, int64_t cols, int64_t row0,
                           int64_t n, int* status = nullptr) {
    RadQuantWeight w = weight(name, rows, cols);
    std::vector<const void*> ptr;
    for (const auto& p : r.planes) ptr.push_back(p.data());
    std::vector<float> out((size_t)(n * cols));
    const int st = Q->decode(&r.enc, &w, ptr.data(), row0, n, out.data());
    if (status) *status = st;
    return out;
}

/* One tensor of a fixture directory, as bytes. */
std::vector<uint8_t> tensor(const std::string& dir, const std::string& name) {
    std::string why;
    exl3::Source* s = exl3::Source::open(dir, &why);
    exl3::Tensor t;
    if (!s || s->find(name, &t, &why) != 1) {
        std::fprintf(stderr, "fixture: %s: %s\n", name.c_str(), why.c_str());
        return {};
    }
    std::vector<uint8_t> b((size_t)t.bytes());
    s->read(t, b.data(), &why);
    return b;
}

std::string exl3_dir() { return FIX + "/exl3"; }
std::string defects_dir() { return FIX + "/exl3-defects"; }
std::string gu_name(int l, int e) { return "blk." + std::to_string(l) + ".ffn_gate_up_exps." + std::to_string(e) + ".weight"; }
std::string dn_name(int l, int e) { return "blk." + std::to_string(l) + ".ffn_down_exps." + std::to_string(e) + ".weight"; }
std::string src_prefix(int l, int e) {
    return (l < LAYERS ? "model.language_model.layers." + std::to_string(l) + ".mlp.experts."
                       : std::string("mtp.layers.0.mlp.experts.")) + std::to_string(e);
}
std::string skel_prefix(int l) {
    return l < LAYERS ? "model.language_model.layers." + std::to_string(l) + ".mlp.experts" : "mtp.layers.0.mlp.experts";
}
bool has(const std::string& s, const char* sub) { return s.find(sub) != std::string::npos; }

/* ================================================================== tests */

TEST(plugin_names_itself_and_its_options) {
    CHECK(Q && std::strcmp(Q->name, "exl3") == 0);
    CHECK(Q->n_options == 3);
    CHECK(Q->decode != nullptr && Q->encoding && Q->quantize && Q->row_block);
    CHECK(std::strcmp(Q->options[0].key, "src") == 0 && Q->options[0].type == RAD_P_STR && !Q->options[0].deflt);
    CHECK(std::strcmp(Q->options[1].key, "bits") == 0 && Q->options[1].type == RAD_P_INT && !Q->options[1].deflt);
    CHECK(std::strcmp(Q->options[2].key, "codebook") == 0 && Q->options[2].deflt &&
          std::strcmp(Q->options[2].deflt, "mul1") == 0);
}

TEST(encoding_is_the_d4_encoding_from_name_shape_and_options_alone) {
    RadEncoding e{}, want{};
    char spell[256];
    Opts o = std_opts("/nonexistent");    /* encoding never reads the source */
    RadQuantWeight gu = weight("blk.7.ffn_gate_up_exps.3.weight", 1280, 2560);
    CHECK(Q->encoding(o.data(), o.n(), &gu, &e) == RAD_OK);
    exl3_encoding_make(K, EXL3_CB_MUL1, 1, &want);
    CHECK(rad_enc_equal(&e, &want));
    CHECK(rad_enc_valid(&e) == 1);
    rad_enc_format(&e, spell, sizeof spell);
    CHECK(std::string(spell) == "exl3:codes=u4[1x1],suh=f16[*x1],svh=f16[1x*],suh.1=f16[*x1]/mul1/h128");
    gu.layer = 7; gu.expert = 3;          /* told the movement group: the answer may not move */
    CHECK(Q->encoding(o.data(), o.n(), &gu, &e) == RAD_OK && rad_enc_equal(&e, &want));
    RadQuantWeight dn = weight("blk.48.ffn_down_exps.511.weight", 2560, 640);
    exl3_encoding_make(K, EXL3_CB_MUL1, 0, &want);
    CHECK(Q->encoding(o.data(), o.n(), &dn, &e) == RAD_OK && rad_enc_equal(&e, &want));
    Opts mcg;
    mcg.s("src", "/x").i("bits", 6).s("codebook", "mcg");
    CHECK(Q->encoding(mcg.data(), mcg.n(), &dn, &e) == RAD_OK);
    CHECK(std::string(e.transform) == "mcg/h128" && e.plane[0].dtype == RAD_U6);
}

TEST(encoding_declines_what_is_not_a_routed_expert) {
    RadEncoding e{};
    Opts o = std_opts("/x");
    for (const char* n : {"blk.0.ffn_gate_up_shexp.weight", "blk.0.attn_qg.weight", "output.weight",
                          "blk.0.ffn_down_exps.3.scale", "xblk.0.ffn_down_exps.3.weight",
                          "blk.0.ffn_down_exps.3.weight.1", "blk.0.ffn_down_exps.-1.weight"}) {
        RadQuantWeight w = weight(n, 2560, 640);
        CHECK(Q->encoding(o.data(), o.n(), &w, &e) == RAD_E_UNSUPPORTED);
    }
}

TEST(encoding_refuses_bad_options_and_shapes_by_name) {
    RadEncoding e{};
    RadQuantWeight gu = weight("blk.0.ffn_gate_up_exps.0.weight", 1280, 2560);
    std::string said;
    for (int bits : {1, 9, 0}) {
        Opts o = std_opts("/x", bits);
        said = captured([&] { CHECK(Q->encoding(o.data(), o.n(), &gu, &e) == RAD_E_INVAL); });
        CHECK(has(said, "bits=") && has(said, "blk.0.ffn_gate_up_exps.0.weight"));
    }
    Opts nobits;
    nobits.s("src", "/x");
    said = captured([&] { CHECK(Q->encoding(nobits.data(), nobits.n(), &gu, &e) == RAD_E_INVAL); });
    CHECK(has(said, "bits=K is required"));
    Opts nosrc;
    nosrc.i("bits", 4);
    said = captured([&] { CHECK(Q->encoding(nosrc.data(), nosrc.n(), &gu, &e) == RAD_E_INVAL); });
    CHECK(has(said, "src=DIR"));
    Opts badcb;
    badcb.s("src", "/x").i("bits", 4).s("codebook", "qtip");
    said = captured([&] { CHECK(Q->encoding(badcb.data(), badcb.n(), &gu, &e) == RAD_E_INVAL); });
    CHECK(has(said, "codebook=qtip"));
    Opts o = std_opts("/x");
    RadQuantWeight odd = weight("blk.0.ffn_gate_up_exps.0.weight", 640, 2560);   /* halves of 320 */
    said = captured([&] { CHECK(Q->encoding(o.data(), o.n(), &odd, &e) == RAD_E_SHAPE); });
    CHECK(has(said, "[640, 2560]"));
    RadQuantWeight narrow = weight("blk.0.ffn_down_exps.0.weight", 2560, 600);
    CHECK(Q->encoding(o.data(), o.n(), &narrow, &e) == RAD_E_SHAPE);
    RadQuantWeight vec = weight("blk.0.ffn_down_exps.0.weight", 2560, 640);
    vec.rank = 1;
    CHECK(Q->encoding(o.data(), o.n(), &vec, &e) == RAD_E_SHAPE);
}

TEST(row_block_is_the_whole_weight) {
    Opts o = std_opts(exl3_dir());
    RadQuantWeight w = weight(gu_name(0, 0), 2 * W, H);
    CHECK(Q->row_block(o.data(), o.n(), &w) == 0);
}

/* The import is a permutation of the source's bits: re-tiled, the codes plane IS the source trellis. */
TEST(import_is_exact_for_trunk_and_mtp_experts) {
    for (int l = 0; l <= LAYERS; ++l)
        for (int e : {0, 5, 7}) {
            const std::string p = src_prefix(l, e);
            Result g = quantise(std_opts(exl3_dir()), gu_name(l, e), 2 * W, H);
            Result d = quantise(std_opts(exl3_dir()), dn_name(l, e), H, W);
            CHECK(g.status == RAD_OK && d.status == RAD_OK);
            if (g.status != RAD_OK || d.status != RAD_OK) continue;
            const int64_t rb_g = exl3_codes_row_bytes(H, K), rb_d = exl3_codes_row_bytes(W, K);
            std::vector<uint16_t> rec((size_t)exl3_record_elems(W, H, K));
            exl3_codes_to_record(g.planes[0].data(), rb_g, W, H, K, rec.data());
            CHECK(std::memcmp(rec.data(), tensor(exl3_dir(), p + ".gate_proj.trellis").data(), rec.size() * 2) == 0);
            exl3_codes_to_record(g.planes[0].data() + W * rb_g, rb_g, W, H, K, rec.data());
            CHECK(std::memcmp(rec.data(), tensor(exl3_dir(), p + ".up_proj.trellis").data(), rec.size() * 2) == 0);
            exl3_codes_to_record(d.planes[0].data(), rb_d, H, W, K, rec.data());
            CHECK(std::memcmp(rec.data(), tensor(exl3_dir(), p + ".down_proj.trellis").data(), rec.size() * 2) == 0);
            CHECK(g.planes[EXL3_PLANE_SUH] == tensor(exl3_dir(), p + ".gate_proj.suh"));
            CHECK(g.planes[EXL3_PLANE_SUH1] == tensor(exl3_dir(), p + ".up_proj.suh"));
            std::vector<uint8_t> svh = tensor(exl3_dir(), p + ".gate_proj.svh"), up = tensor(exl3_dir(), p + ".up_proj.svh");
            svh.insert(svh.end(), up.begin(), up.end());
            CHECK(g.planes[EXL3_PLANE_SVH] == svh);
            CHECK(d.planes[EXL3_PLANE_SUH] == tensor(exl3_dir(), p + ".down_proj.suh"));
            CHECK(d.planes[EXL3_PLANE_SVH] == tensor(exl3_dir(), p + ".down_proj.svh"));
        }
}

/* The fixture's skeleton holds bf16 of the dense experts exl3_ref.py decoded: our decode must agree to bf16
 * rounding of each element (RNE, 8-bit significand: <= 2^-8 |x|) plus the f32 disagreement of two rotation orders, bounded at
 * 1e-5 of the weight's largest |x| (common/tests measures ~6e-7). Returned: the worst excess over the bf16
 * part, in units of max|x| -- <= 1e-5 passes. Negative control: another expert's planes do not. */
double worst_excess(const std::vector<float>& got, const uint16_t* bf, size_t n) {
    double worst = 0, scale = 0;
    for (size_t i = 0; i < n; ++i) scale = std::fmax(scale, std::fabs((double)got[i]));
    for (size_t i = 0; i < n; ++i) {
        const uint32_t u = (uint32_t)bf[i] << 16;
        float ref;
        std::memcpy(&ref, &u, 4);
        const double excess = std::fabs((double)got[i] - ref) - std::fabs((double)got[i]) / 256.0;
        worst = std::fmax(worst, excess / scale);
    }
    return worst;
}

TEST(decode_equals_the_independent_dense_experts_to_bf16_rounding) {
    double worst = 0, other = 0;
    for (int l = 0; l <= LAYERS; ++l) {
        const std::vector<uint8_t> gu = tensor(FIX + "/bf16", skel_prefix(l) + ".gate_up_proj");
        const std::vector<uint8_t> dn = tensor(FIX + "/bf16", skel_prefix(l) + ".down_proj");
        CHECK(gu.size() == (size_t)E * 2 * W * H * 2 && dn.size() == (size_t)E * H * W * 2);
        if (gu.empty() || dn.empty()) continue;
        for (int e = 0; e < E; ++e) {
            Result g = quantise(std_opts(exl3_dir()), gu_name(l, e), 2 * W, H, false);
            Result d = quantise(std_opts(exl3_dir()), dn_name(l, e), H, W, false);
            const auto* gb = reinterpret_cast<const uint16_t*>(gu.data()) + (size_t)e * 2 * W * H;
            const auto* db = reinterpret_cast<const uint16_t*>(dn.data()) + (size_t)e * H * W;
            worst = std::fmax(worst, worst_excess(decoded(g, gu_name(l, e), 2 * W, H, 0, 2 * W), gb, 2 * W * H));
            worst = std::fmax(worst, worst_excess(decoded(d, dn_name(l, e), H, W, 0, H), db, H * W));
            if (e == 1) {
                const auto* wrong = reinterpret_cast<const uint16_t*>(gu.data());   /* expert 0's */
                other = worst_excess(decoded(g, gu_name(l, e), 2 * W, H, 0, 2 * W), wrong, 2 * W * H);
            }
        }
    }
    std::printf("  decode vs exl3_ref.py dense experts: worst excess over bf16 rounding %.3g of max|W| (bound 1e-5); "
                "another expert's planes: %.3g\n", worst, other);
    CHECK(worst <= 1e-5);
    CHECK(other > 0.1);
}

TEST(decode_reads_any_rows_and_only_its_own_scheme) {
    Result g = quantise(std_opts(exl3_dir()), gu_name(1, 2), 2 * W, H);
    int st = 0;
    const std::vector<float> all = decoded(g, gu_name(1, 2), 2 * W, H, 0, 2 * W);
    const std::vector<float> mid = decoded(g, gu_name(1, 2), 2 * W, H, 100, 37, &st);
    CHECK(st == RAD_OK);
    CHECK(std::memcmp(mid.data(), all.data() + 100 * H, mid.size() * 4) == 0);
    Result plain = g;
    plain.enc = rad_enc_plain(RAD_BF16);
    decoded(plain, gu_name(1, 2), 2 * W, H, 0, 2 * W, &st);
    CHECK(st == RAD_E_UNSUPPORTED);                /* not ours: the core asks the next decoder */
    Result bad = g;
    bad.enc.n_planes = 2;                          /* says exl3, is not what we write */
    decoded(bad, gu_name(1, 2), 2 * W, H, 0, 2 * W, &st);
    CHECK(st == RAD_E_FORMAT);
    decoded(g, gu_name(1, 2), 2 * W, H, 250, 10, &st);
    CHECK(st == RAD_E_SHAPE);                      /* past the end */
}

/* ------------------------------------------------------------------ R12: refused by name, each */

TEST(refuses_a_k_other_than_the_recipes) {
    Result r = quantise(std_opts(exl3_dir(), 3), gu_name(0, 0), 2 * W, H);
    CHECK(r.status == RAD_E_DTYPE && has(r.said, "is K=4") && has(r.said, "bits=3"));
    CHECK(has(r.said, gu_name(0, 0).c_str()));
    CHECK(quantise(std_opts(exl3_dir(), 4), gu_name(0, 0), 2 * W, H).status == RAD_OK);   /* control */
}

TEST(refuses_a_wrong_marker_value) {
    Result r = quantise(std_opts(defects_dir()), gu_name(0, 1), 2 * W, H);
    CHECK(r.status == RAD_E_FORMAT && has(r.said, "0x12345678") && has(r.said, "gate_proj.mul1"));
}

TEST(refuses_another_codebook_by_its_marker) {
    Result mcg = quantise(std_opts(defects_dir()), gu_name(0, 2), 2 * W, H);
    CHECK(mcg.status == RAD_E_DTYPE && has(mcg.said, "codebook mcg") && has(mcg.said, "codebook=mul1"));
    Result none = quantise(std_opts(defects_dir()), gu_name(0, 3), 2 * W, H);
    CHECK(none.status == RAD_E_DTYPE && has(none.said, "codebook 3inst"));
    Result both = quantise(std_opts(defects_dir()), gu_name(0, 6), 2 * W, H);
    CHECK(both.status == RAD_E_FORMAT && has(both.said, "both the mul1 and the mcg marker"));
}

TEST(refuses_half_integer_k) {
    Result r = quantise(std_opts(defects_dir()), gu_name(0, 4), 2 * W, H);
    CHECK(r.status == RAD_E_UNSUPPORTED && has(r.said, "half-integer K = 4.5"));
}

TEST(refuses_a_bias) {
    Result r = quantise(std_opts(defects_dir()), gu_name(0, 5), 2 * W, H);
    CHECK(r.status == RAD_E_UNSUPPORTED && has(r.said, "gate_proj has a .bias"));
}

TEST(refuses_what_the_source_does_not_have) {
    Result svh = quantise(std_opts(defects_dir()), dn_name(0, 3), H, W);
    CHECK(svh.status == RAD_E_NOTFOUND && has(svh.said, "down_proj.svh is not in the source"));
    Result e8 = quantise(std_opts(exl3_dir()), gu_name(0, E), 2 * W, H);
    CHECK(e8.status == RAD_E_NOTFOUND && has(e8.said, "experts.8.gate_proj"));
    Result past = quantise(std_opts(exl3_dir()), gu_name(LAYERS + 1, 0), 2 * W, H);
    CHECK(past.status == RAD_E_NOTFOUND && has(past.said, "past the source's 2 layers"));
    /* positive controls in the same defective directory: its clean experts import */
    CHECK(quantise(std_opts(defects_dir()), gu_name(0, 0), 2 * W, H).status == RAD_OK);
    CHECK(quantise(std_opts(defects_dir()), gu_name(0, 7), 2 * W, H).status == RAD_OK);
    CHECK(quantise(std_opts(defects_dir()), dn_name(0, 2), H, W).status == RAD_OK);
}

TEST(refuses_a_shape_that_disagrees_with_the_source) {
    /* the fixture's experts are [256, 256] gate_up; asked as [512, 256] the halves are 256 rows, not 128 */
    Result r = quantise(std_opts(exl3_dir()), gu_name(0, 0), 4 * W, H);
    CHECK(r.status == RAD_E_SHAPE && has(r.said, "n=256 out"));
}

TEST(refuses_a_source_that_is_not_exl3) {
    char dir[] = "/tmp/exl3_notexl3_XXXXXX";
    CHECK(mkdtemp(dir) != nullptr);
    const std::string cfg = std::string(dir) + "/config.json";
    FILE* f = std::fopen(cfg.c_str(), "w");
    std::fputs("{\"text_config\": {\"num_hidden_layers\": 2}, \"quantization_config\": {\"quant_method\": \"fp8\"}}", f);
    std::fclose(f);
    Result r = quantise(std_opts(dir), gu_name(0, 0), 2 * W, H);
    CHECK(r.status == RAD_E_FORMAT && has(r.said, "is not an EXL3 checkpoint") && has(r.said, "'fp8'"));
    unlink(cfg.c_str());
    rmdir(dir);
    Result none = quantise(std_opts("/nonexistent-exl3-src"), gu_name(0, 0), 2 * W, H);
    CHECK(none.status == RAD_E_FORMAT && has(none.said, "config.json"));
}

TEST(refuses_a_partial_row_block) {
    Opts o = std_opts(exl3_dir());
    RadQuantWeight w = weight(gu_name(0, 0), 2 * W, H);
    RadEncoding e{};
    Q->encoding(o.data(), o.n(), &w, &e);
    std::vector<std::vector<uint8_t>> planes;
    std::vector<void*> ptr;
    for (int i = 0; i < e.n_planes; ++i) {
        planes.emplace_back((size_t)rad_enc_plane_bytes(&e.plane[i], 2 * W, H), 0);
        ptr.push_back(planes.back().data());
    }
    std::vector<float> src(16 * H);
    const std::string said = captured([&] {
        CHECK(Q->quantize(o.data(), o.n(), &w, src.data(), 16, 16, ptr.data()) == RAD_E_STATE);
    });
    CHECK(has(said, "row_block 0"));
}

/* rad-convert quantises small weights on every thread at once. */
TEST(concurrent_imports_equal_serial_ones) {
    std::vector<Result> serial, par(16);
    for (int i = 0; i < 16; ++i) serial.push_back(quantise(std_opts(exl3_dir()), gu_name(i % 3, i % E), 2 * W, H, false));
    std::vector<std::thread> th;
    for (int i = 0; i < 16; ++i)
        th.emplace_back([&, i] { par[i] = quantise(std_opts(exl3_dir()), gu_name(i % 3, i % E), 2 * W, H, false); });
    for (auto& t : th) t.join();
    int same = 0;
    for (int i = 0; i < 16; ++i) same += par[i].status == RAD_OK && par[i].planes == serial[i].planes;
    CHECK(same == 16);
}

}  // namespace

int main(int argc, char** argv) {
    struct stat st {};
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <exl3_quant.so> <fixture dir>\n", argv[0]);
        return 2;
    }
    FIX = argv[2];
    if (stat((FIX + "/exl3/config.json").c_str(), &st) != 0 || stat((FIX + "/exl3-defects").c_str(), &st) != 0) {
        std::printf("SKIPPED: no fixture at %s (tools/exl3_fixture.py --out %s --defects makes it)\n", argv[2], argv[2]);
        return 77;
    }
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        std::fprintf(stderr, "dlopen %s: %s\n", argv[1], dlerror());
        return 1;
    }
    auto abi = (uint32_t(*)(void))dlsym(h, "rad_plugin_abi_version");
    auto info = (const RadPluginInfo* (*)(void))dlsym(h, "rad_plugin_info");
    auto count = (int (*)(void))dlsym(h, "rad_quant_count");
    auto at = (const RadQuantizerInfo* (*)(int))dlsym(h, "rad_quant_at");
    CHECK(abi && info && count && at);
    if (!abi || !info || !count || !at) return exl3_check_done();
    CHECK(abi() == RAD_ABI_VERSION);
    CHECK(info()->kind == RAD_PLUGIN_QUANT);
    CHECK(count() == 1 && at(1) == nullptr);
    Q = at(0);
    RUN(plugin_names_itself_and_its_options);
    RUN(encoding_is_the_d4_encoding_from_name_shape_and_options_alone);
    RUN(encoding_declines_what_is_not_a_routed_expert);
    RUN(encoding_refuses_bad_options_and_shapes_by_name);
    RUN(row_block_is_the_whole_weight);
    RUN(import_is_exact_for_trunk_and_mtp_experts);
    RUN(decode_equals_the_independent_dense_experts_to_bf16_rounding);
    RUN(decode_reads_any_rows_and_only_its_own_scheme);
    RUN(refuses_a_k_other_than_the_recipes);
    RUN(refuses_a_wrong_marker_value);
    RUN(refuses_another_codebook_by_its_marker);
    RUN(refuses_half_integer_k);
    RUN(refuses_a_bias);
    RUN(refuses_what_the_source_does_not_have);
    RUN(refuses_a_shape_that_disagrees_with_the_source);
    RUN(refuses_a_source_that_is_not_exl3);
    RUN(refuses_a_partial_row_block);
    RUN(concurrent_imports_equal_serial_ones);
    return exl3_check_done();
}
