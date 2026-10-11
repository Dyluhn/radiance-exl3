// exl3ref_run -- load a kernel library the way the engine does and run ONE case of an EXL3 op on it.
//
//   exl3ref_run <plugin.so> <case.txt>     run the case; outputs land beside it as out_<name>.bin
//   exl3ref_run <plugin.so> --schemas      the plugin's schemas against include/exl3_abi.h, field by field
//
// The case file (written by test_exl3ref.py; paths relative to it):
//   op <name>
//   param <key> int <value> | param <key> str <value>
//   opd <name> absent
//   opd <name> <in|out|wtab> <dtype> <rank> <dims...> <file|-> [table|stacked] [pitch=<elements>]
// An `in`/`wtab` file holds the operand dense; `pitch` stores a 2-D operand's rows that many elements
// apart (the gap filled with garbage); `table` passes a weight table as the engine does (a pointer per
// entry, stride[0] = 0) and `stacked` as one rank-2 block; an `out` is allocated, prefilled with 0xFF
// (NaN in f16 and bf16) and written back dense.
//
// It prints, one line each: `shape <i> <rc> <dtype> <rank> <dims,> <flags> <fill> <idx_max>` for every
// operand (the row's opd_shape), `rc <launch status>`, `redzone ok|FAIL <operand>` and
// `inputs ok|CHANGED <operand>`. Every buffer it hands the library has a 256-byte poisoned margin on
// both sides, checked after the launch: an out-of-bounds write is the one fault a comparison of the
// outputs cannot see. The verdict is the caller's; this exits non-zero only when it could not run.
#include "exl3_abi.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr size_t kRed = 256;
constexpr unsigned char kRedFill = 0xA5;

struct Buf {                       // a payload between two poisoned margins
    std::vector<unsigned char> b;
    void alloc(size_t n, unsigned char fill) {
        b.assign(n + 2 * kRed, kRedFill);
        std::memset(b.data() + kRed, fill, n);
    }
    unsigned char* p() { return b.data() + kRed; }
    size_t n() const { return b.size() - 2 * kRed; }
    bool red_ok() const {
        for (size_t i = 0; i < kRed; ++i)
            if (b[i] != kRedFill || b[b.size() - 1 - i] != kRedFill) return false;
        return true;
    }
};

struct Opd {
    std::string name, role = "absent", file, form;
    uint32_t dt = 0;
    std::vector<int64_t> dims;
    int64_t pitch = 0;
    Buf buf, table;                // the payload, and a pointer table for `table` form
    std::vector<unsigned char> before;
};

[[noreturn]] void die(const std::string& m) {
    std::fprintf(stderr, "exl3ref_run: %s\n", m.c_str());
    std::exit(2);
}

template <class F> F sym(void* h, const char* name) {
    void* s = dlsym(h, name);
    if (!s) die(std::string("the plugin has no ") + name);
    return reinterpret_cast<F>(s);
}

std::string dir_of(const std::string& p) {
    const size_t s = p.rfind('/');
    return s == std::string::npos ? "." : p.substr(0, s);
}

int schemas(void* h) {
    auto count = sym<int (*)()>(h, "rad_kernel_schema_count");
    auto at = sym<const RadOpSchema* (*)(int)>(h, "rad_kernel_schema_at");
    int bad = 0;
    auto same = [](const char* a, const char* b) { return a && b && std::strcmp(a, b) == 0; };
    if (count() != exl3_schema_count()) {
        std::printf("FAIL schema count %d, include/exl3_abi.h has %d\n", count(), exl3_schema_count());
        return 1;
    }
    for (int i = 0; i < count(); ++i) {
        const RadOpSchema *s = at(i), *w = exl3_schema_at(i);
        bool ok = s && same(s->op, w->op) && s->n_params == w->n_params &&
                  s->n_operands == w->n_operands && same(s->doc, w->doc);
        for (int k = 0; ok && k < w->n_params; ++k)
            ok = same(s->params[k].key, w->params[k].key) && s->params[k].type == w->params[k].type &&
                 s->params[k].required == w->params[k].required && s->params[k].role == w->params[k].role;
        for (int k = 0; ok && k < w->n_operands; ++k)
            ok = same(s->operands[k].name, w->operands[k].name) &&
                 s->operands[k].role == w->operands[k].role &&
                 s->operands[k].optional == w->operands[k].optional;
        std::printf("%s schema %s\n", ok ? "ok" : "FAIL", w->op);
        bad += !ok;
    }
    return bad ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) die("usage: exl3ref_run <plugin.so> <case.txt> | --schemas");
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) die(std::string("dlopen: ") + dlerror());
    if (sym<uint32_t (*)()>(h, "rad_plugin_abi_version")() != RAD_ABI_VERSION)
        die("the plugin reports another ABI version");
    if (!std::strcmp(argv[2], "--schemas")) return schemas(h);

    // ---- the case
    std::ifstream in(argv[2]);
    if (!in) die(std::string("cannot read ") + argv[2]);
    const std::string dir = dir_of(argv[2]);
    std::string op, line;
    std::vector<std::string> keys, svals;
    std::vector<RadParam> params;
    std::vector<Opd> opd;
    std::vector<std::pair<std::string, std::string>> raw_params;   // key, "int v" | "str v"
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string kind;
        if (!(ls >> kind)) continue;
        if (kind == "op") { ls >> op; continue; }
        if (kind == "param") {
            std::string k, t, v;
            ls >> k >> t >> v;
            raw_params.push_back({ k, t + " " + v });
            continue;
        }
        if (kind != "opd") die("bad line: " + line);
        Opd o;
        ls >> o.name >> o.role;
        if (o.role != "absent") {
            std::string dts;
            int rank = 0;
            ls >> dts >> rank;
            o.dt = rad_dtype_parse(dts.c_str());
            if (!o.dt) die("unknown dtype " + dts);
            o.dims.resize((size_t)rank);
            for (auto& d : o.dims) ls >> d;
            ls >> o.file;
            std::string extra;
            while (ls >> extra) {
                if (extra == "table" || extra == "stacked") o.form = extra;
                else if (extra.rfind("pitch=", 0) == 0) o.pitch = std::stoll(extra.substr(6));
                else die("bad operand option " + extra);
            }
        }
        opd.push_back(std::move(o));
    }
    // Strings outlive the RadParam that points at them.
    keys.reserve(raw_params.size());
    svals.reserve(raw_params.size());
    for (auto& kv : raw_params) {
        keys.push_back(kv.first);
        const std::string t = kv.second.substr(0, 3), v = kv.second.substr(4);
        if (t == "int") params.push_back(RAD_INT(nullptr, std::stoll(v)));
        else { svals.push_back(v); params.push_back(RAD_STR(nullptr, nullptr)); }
    }
    for (size_t i = 0, s = 0; i < params.size(); ++i) {
        params[i].key = keys[i].c_str();
        if (params[i].kind == RAD_P_STR) params[i].sval = svals[s++].c_str();
    }

    // ---- the row
    auto count = sym<int (*)()>(h, "rad_kernel_count");
    auto at = sym<const RadKernelInfo* (*)(int)>(h, "rad_kernel_at");
    const RadKernelInfo* row = nullptr;
    for (int i = 0; i < count() && !row; ++i)
        if (at(i) && at(i)->op && op == at(i)->op && at(i)->domain == RAD_DOMAIN_HOST) row = at(i);
    if (!row) die("no host row for op " + op);

    for (int i = 0; i < (int)opd.size(); ++i) {
        RadOpdDesc d;
        std::memset(&d, 0, sizeof d);
        const int rc = row->opd_shape ? row->opd_shape(params.data(), (int)params.size(), i, &d) : -999;
        std::printf("shape %d %d %u %u ", i, rc, d.dtype, d.rank);
        for (uint32_t r = 0; r < d.rank; ++r) std::printf("%s%lld", r ? "," : "", (long long)d.shape[r]);
        std::printf("%s %u %u %lld\n", d.rank ? "" : "-", d.flags, d.fill, (long long)d.idx_max);
    }

    // ---- the operands
    std::vector<RadTensor> t(opd.size());
    for (size_t i = 0; i < opd.size(); ++i) {
        Opd& o = opd[i];
        RadTensor& x = t[i];
        std::memset(&x, 0, sizeof x);
        if (o.role == "absent") continue;
        x.dtype = o.dt;
        x.rank = (uint32_t)o.dims.size();
        for (size_t r = 0; r < o.dims.size(); ++r) x.shape[r] = o.dims[r];
        rad_tensor_pack(&x);
        const int64_t rows = x.rank >= 2 ? rad_tensor_rows(&x) : 1;
        const int64_t width = x.rank >= 2 ? x.shape[x.rank - 1] : rad_tensor_numel(&x);
        const int64_t pitch = o.pitch > 0 ? o.pitch : width;
        if (pitch < width) die(o.name + ": pitch narrower than the row");
        if (x.rank >= 2) x.stride[x.rank - 2] = pitch;
        const int64_t row_bytes = rad_dtype_bytes(o.dt, width), pitch_bytes = rad_dtype_bytes(o.dt, pitch);
        o.buf.alloc((size_t)(rows * pitch_bytes), o.role == "out" ? 0xFF : 0x5A);
        if (o.role != "out") {
            std::ifstream f(dir + "/" + o.file, std::ios::binary);
            std::vector<char> dense((size_t)(rows * row_bytes));
            if (!f.read(dense.data(), (std::streamsize)dense.size())) die("short file " + o.file);
            for (int64_t r = 0; r < rows; ++r)
                std::memcpy(o.buf.p() + r * pitch_bytes, dense.data() + r * row_bytes, (size_t)row_bytes);
        }
        x.data = o.buf.p();
        if (o.role == "wtab" && o.form == "table") {       // the engine's form: a pointer per entry
            if (x.rank != 2) die(o.name + ": a table is [entries, bytes]");
            o.table.alloc((size_t)x.shape[0] * sizeof(void*), 0);
            for (int64_t e = 0; e < x.shape[0]; ++e) {
                void* p = o.buf.p() + e * pitch_bytes;
                std::memcpy(o.table.p() + e * sizeof(void*), &p, sizeof p);
            }
            x.data = o.table.p();
            x.stride[0] = 0;
        }
        if (o.role != "out") o.before.assign(o.buf.p(), o.buf.p() + o.buf.n());
    }

    RadArgs a;
    std::memset(&a, 0, sizeof a);
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = params.data();
    a.n_p = (int)params.size();
    a.world_size = 1;
    std::printf("rc %d\n", row->launch(&a, nullptr));

    // ---- what it touched
    std::string red_bad, changed;
    for (Opd& o : opd) {
        if (o.role == "absent") continue;
        if (!o.buf.red_ok() || (o.form == "table" && !o.table.red_ok())) red_bad += " " + o.name;
        if (o.role != "out" && std::memcmp(o.before.data(), o.buf.p(), o.buf.n()) != 0) changed += " " + o.name;
    }
    std::printf("redzone %s%s\n", red_bad.empty() ? "ok" : "FAIL", red_bad.c_str());
    std::printf("inputs %s%s\n", changed.empty() ? "ok" : "CHANGED", changed.c_str());

    for (size_t i = 0; i < opd.size(); ++i) {
        Opd& o = opd[i];
        if (o.role != "out") continue;
        const RadTensor& x = t[i];
        const int64_t rows = rad_tensor_rows(&x), width = x.shape[x.rank - 1];
        const int64_t row_bytes = rad_dtype_bytes(o.dt, width), pitch_bytes = rad_dtype_bytes(o.dt, x.stride[x.rank - 2]);
        std::ofstream f(dir + "/out_" + o.name + ".bin", std::ios::binary);
        for (int64_t r = 0; r < rows; ++r)
            f.write((const char*)o.buf.p() + r * pitch_bytes, (std::streamsize)row_bytes);
    }
    return 0;
}
