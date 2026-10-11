// exl3ref_run -- load a kernel library the way the engine does and run ONE case of an EXL3 op on it.
//
//   exl3ref_run <library.so> <case.txt> [--domain host|device]   run the case; outputs land beside it
//   exl3ref_run <library.so> --schemas                           its schemas vs include/exl3_abi.h
//
// <library.so> is any library with rows for moe_exl3_gate_up / moe_exl3_down: the host reference
// (exl3ref, --domain host, the default) or the device library (exl3r4d, --domain device, which needs
// this binary built with HIP: EXL3_RUN_HIP). The same case run through both and compared is the
// device check of PLAN.md D15/Q-R2 -- rad-kbench cannot check a weight-table op at radiance 1.3.0.
//
// The case file (written by test_exl3ref.py; paths relative to it):
//   op <name>
//   param <key> int <value> | param <key> str <value>
//   opd <name> absent
//   opd <name> <in|out|wtab> <dtype> <rank> <dims...> <file|-> [table|stacked] [pitch=<elements>]
// An `in`/`wtab` file holds the operand dense (a table: its entries back to back); `pitch` stores a
// 2-D operand's rows that many elements apart, the gap garbage; `table` passes a weight table as the
// engine does (a pointer per entry, stride[0] = 0) and `stacked` as one tensor with a real leading
// stride; an `out` is allocated, prefilled with 0xFF (NaN in f16 and bf16) and written back dense.
//
// A WEIGHT TABLE GOES THROUGH THE ROW'S OWN LAYOUT HOOKS, as the engine's loader does: `layout` is
// asked with the table's canonical plane (the encoding made from the case's dtype and codebook);
// RAD_E_UNSUPPORTED keeps the plane as it is, RAD_OK re-lays every entry with `relayout`, described
// with the STORED dtype and the declared shape (rad-kbench's kf_describe_laid), and -- when the row
// has `unrelayout` -- inverts it again and compares with the plane, byte for byte.
//
// It prints, one line each: `shape <i> <rc> <dtype> <rank> <dims,> <flags> <fill> <idx_max>` for every
// operand (the row's opd_shape), `constraint <key> refused` for every constraint of the row the case's
// parameters do not meet, `layout <i> <as-is|<tag>> unrelayout <ok|FAIL|absent>` (or `layout <i> refused <rc>`,
// then `rc none`: not launched) for every table, `rc <launch status>`, `redzone ok|FAIL <operands>` and `inputs ok|CHANGED <operands>`. Every
// buffer it hands the library has a 256-byte poisoned margin on both sides, checked after the launch:
// an out-of-bounds write is the one fault a comparison of the outputs cannot see. The verdict is the
// caller's; this exits 2 when it could not run and 77 when asked for a domain this build has not.
#include "exl3_abi.h"

#include <dlfcn.h>

#ifdef EXL3_RUN_HIP
#include <hip/hip_runtime_api.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr size_t kRed = 256;
constexpr unsigned char kRedFill = 0xA5;
bool g_device = false;

[[noreturn]] void die(const std::string& m) {
    std::fprintf(stderr, "exl3ref_run: %s\n", m.c_str());
    std::exit(2);
}

#ifdef EXL3_RUN_HIP
void hip_check(hipError_t e, const char* what) {
    if (e != hipSuccess) die(std::string(what) + ": " + hipGetErrorString(e));
}
#endif

// A payload between two poisoned margins, on the host or the device.
struct Mem {
    std::vector<unsigned char> host;
    unsigned char* dev = nullptr;
    size_t n = 0;
    void alloc(size_t bytes, unsigned char fill) {
        if (bytes > ((size_t)1 << 40)) die("an operand of more than a TiB");   // and GCC can see n + 2*kRed fits
        n = bytes;
        std::vector<unsigned char> img(n + 2 * kRed, kRedFill);
        std::fill(img.begin() + kRed, img.end() - kRed, fill);
        if (!g_device) { host.swap(img); return; }
#ifdef EXL3_RUN_HIP
        hip_check(hipMalloc((void**)&dev, img.size()), "hipMalloc");
        hip_check(hipMemcpy(dev, img.data(), img.size(), hipMemcpyHostToDevice), "copy in");
#endif
    }
    unsigned char* p() { return (g_device ? dev : host.data()) + kRed; }
    void put(const void* src, size_t bytes, size_t off = 0) {
        if (!g_device) { std::memcpy(host.data() + kRed + off, src, bytes); return; }
#ifdef EXL3_RUN_HIP
        hip_check(hipMemcpy(dev + kRed + off, src, bytes, hipMemcpyHostToDevice), "copy in");
#endif
    }
    std::vector<unsigned char> image() const {          // the whole buffer, margins included
        if (!g_device) return host;
        std::vector<unsigned char> img(n + 2 * kRed);
#ifdef EXL3_RUN_HIP
        hip_check(hipMemcpy(img.data(), dev, img.size(), hipMemcpyDeviceToHost), "copy out");
#endif
        return img;
    }
};

bool red_ok(const std::vector<unsigned char>& img) {
    for (size_t i = 0; i < kRed; ++i)
        if (img[i] != kRedFill || img[img.size() - 1 - i] != kRedFill) return false;
    return true;
}

struct Opd {
    std::string name, role = "absent", file, form;
    uint32_t dt = 0;
    std::vector<int64_t> dims;
    int64_t pitch = 0;
    Mem buf, table;                 // the payload; a pointer table for the `table` form
    std::vector<unsigned char> written;
    std::string laid = "as-is", unlaid = "absent";
};

template <class F> F sym(void* h, const char* name) {
    void* s = dlsym(h, name);
    if (!s) die(std::string("the library has no ") + name);
    return reinterpret_cast<F>(s);
}

int schemas(void* h) {
    auto count = sym<int (*)()>(h, "rad_kernel_schema_count");
    auto at = sym<const RadOpSchema* (*)(int)>(h, "rad_kernel_schema_at");
    auto same = [](const char* a, const char* b) { return a && b && std::strcmp(a, b) == 0; };
    if (count() != exl3_schema_count()) {
        std::printf("FAIL schema count %d, include/exl3_abi.h has %d\n", count(), exl3_schema_count());
        return 1;
    }
    int bad = 0;
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

// Does the row's constraint hold for these parameters? What selection asks (core/util/rad_util.cpp:248-272): a key the
// geometry lacks does not hold; IN matches a string, or an integer's decimal form. A machine fact ("device.") has
// nothing measured here and is taken to hold -- this harness runs one rank on one card.
bool holds(const RadConstraint& c, const RadParam* P, int nP) {
    if (!c.key) return false;
    if (!std::strncmp(c.key, RAD_DEVICE_FACT_PREFIX, std::strlen(RAD_DEVICE_FACT_PREFIX))) return true;
    const RadParam* f = rad_param_find(P, nP, c.key);
    if (!f) return false;
    if (c.op == RAD_C_IN) {
        const std::string v = f->kind == RAD_P_STR ? std::string(f->sval ? f->sval : "")
                            : f->kind == RAD_P_INT ? std::to_string(f->ival) : std::string("\x01");
        std::istringstream set(c.sval ? c.sval : "");
        for (std::string tok; set >> tok;) if (tok == v) return true;
        return false;
    }
    if (f->kind != RAD_P_INT) return false;
    switch (c.op) {
        case RAD_C_EQ:  return f->ival == c.ival;
        case RAD_C_LE:  return f->ival <= c.ival;
        case RAD_C_GE:  return f->ival >= c.ival;
        case RAD_C_DIV: return c.ival != 0 && f->ival % c.ival == 0;
        default:        return false;
    }
}

std::vector<unsigned char> read_file(const std::string& path, size_t bytes) {
    std::ifstream f(path, std::ios::binary);
    std::vector<unsigned char> v(bytes);
    if (!f.read((char*)v.data(), (std::streamsize)bytes)) die("short file " + path);
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) die("usage: exl3ref_run <library.so> <case.txt> [--domain host|device] | --schemas");
    std::string domain = "host";
    for (int i = 3; i + 1 < argc; i += 2)
        if (!std::strcmp(argv[i], "--domain")) domain = argv[i + 1];
        else die(std::string("unknown option ") + argv[i]);
    if (domain != "host" && domain != "device") die("--domain is host or device");
    g_device = domain == "device";
#ifndef EXL3_RUN_HIP
    if (g_device) { std::printf("SKIP device domain: this exl3ref_run was built without HIP\n"); return 77; }
#endif
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) die(std::string("dlopen: ") + dlerror());
    if (sym<uint32_t (*)()>(h, "rad_plugin_abi_version")() != RAD_ABI_VERSION)
        die("the library reports another ABI version");
    if (!std::strcmp(argv[2], "--schemas")) return schemas(h);

    // ---- the case
    std::ifstream in(argv[2]);
    if (!in) die(std::string("cannot read ") + argv[2]);
    std::string dir = argv[2];
    dir = dir.rfind('/') == std::string::npos ? "." : dir.substr(0, dir.rfind('/'));
    std::string op, line;
    std::vector<std::pair<std::string, std::string>> raw;   // key, "int v" | "str v"
    std::vector<Opd> opd;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string kind;
        if (!(ls >> kind)) continue;
        if (kind == "op") { ls >> op; continue; }
        if (kind == "param") {
            std::string k, t, v;
            ls >> k >> t >> v;
            raw.push_back({ k, t + " " + v });
            continue;
        }
        if (kind != "opd") die("bad line: " + line);
        Opd o;
        ls >> o.name >> o.role;
        if (o.role != "absent") {
            std::string dts;
            int rank = 0;
            ls >> dts >> rank;
            if (!(o.dt = rad_dtype_parse(dts.c_str()))) die("unknown dtype " + dts);
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
    std::vector<std::string> keys, svals;                   // the strings outlive the params
    keys.reserve(raw.size());
    svals.reserve(raw.size());
    std::vector<RadParam> params;
    for (auto& kv : raw) {
        keys.push_back(kv.first);
        if (kv.second.compare(0, 3, "int") == 0) params.push_back(RAD_INT(nullptr, std::stoll(kv.second.substr(4))));
        else { svals.push_back(kv.second.substr(4)); params.push_back(RAD_STR(nullptr, nullptr)); }
    }
    for (size_t i = 0, s = 0; i < params.size(); ++i) {
        params[i].key = keys[i].c_str();
        if (params[i].kind == RAD_P_STR) params[i].sval = svals[s++].c_str();
    }
    const RadParam* P = params.data();
    const int nP = (int)params.size();

    // ---- the row
    auto count = sym<int (*)()>(h, "rad_kernel_count");
    auto at = sym<const RadKernelInfo* (*)(int)>(h, "rad_kernel_at");
    const int want_dom = g_device ? RAD_DOMAIN_DEVICE : RAD_DOMAIN_HOST;
    const RadKernelInfo* row = nullptr;
    for (int i = 0; i < count() && !row; ++i)
        if (at(i) && at(i)->op && op == at(i)->op && at(i)->domain == want_dom) row = at(i);
    if (!row) die("no " + domain + " row for op " + op);
    // The row's own constraints, as selection would apply them: a refusal is printed by key and the case still runs,
    // so the launch's own checks are exercised too. A row the engine would never select for this geometry is the
    // caller's to report (a skip on a device library, a failure on the reference).
    for (int k = 0; k < row->n_constraints; ++k)
        if (!holds(row->constraints[k], P, nP))
            std::printf("constraint %s refused\n", row->constraints[k].key ? row->constraints[k].key : "?");
    const int gate_up = op == EXL3_OP_GATE_UP;
    const Exl3Positions* pos = exl3_positions(gate_up);

    for (int i = 0; i < (int)opd.size(); ++i) {
        RadOpdDesc d;
        std::memset(&d, 0, sizeof d);
        const int rc = row->opd_shape ? row->opd_shape(P, nP, i, &d) : -999;
        std::printf("shape %d %d %u %u ", i, rc, d.dtype, d.rank);
        for (uint32_t r = 0; r < d.rank; ++r) std::printf("%s%lld", r ? "," : "", (long long)d.shape[r]);
        std::printf("%s %u %u %lld\n", d.rank ? "" : "-", d.flags, d.fill, (long long)d.idx_max);
    }

    // The encoding the planes come from, for a row with layout hooks (include/exl3_encoding.h).
    // gate_up's dn_suh tables are views of DOWN's encoding (unstacked), the rest of the op's own.
    RadEncoding enc, enc_down;
    const char* dts = rad_param_gets(P, nP, "dtype", "");
    const char* cbs = rad_param_gets(P, nP, "codebook", "");
    const bool have_enc =
        exl3_encoding_make(exl3_gemm_bits(dts), exl3_codebook_from_name(cbs), gate_up, &enc) == RAD_OK &&
        exl3_encoding_make(exl3_gemm_bits(dts), exl3_codebook_from_name(cbs), 0, &enc_down) == RAD_OK;
    auto plane_of = [&](int k) -> int {                     // EXL3_P_* of a table operand, or -1
        for (int c = 0; c < 2; ++c)
            for (int p = 0; p < EXL3_N_P; ++p)
                if (pos->plane[c][p] == k) return p;
        return -1;
    };
    static const int enc_plane[EXL3_N_P] = { EXL3_PLANE_CODES, EXL3_PLANE_SUH, EXL3_PLANE_SUH1,
                                             EXL3_PLANE_SVH, EXL3_PLANE_SUH };

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
        const int64_t numel = rad_tensor_numel(&x);

        if (o.role == "wtab") {
            const int64_t n_ent = x.shape[0], ent_n = n_ent ? numel / n_ent : 0;
            const int64_t ent_b = rad_dtype_bytes(o.dt, ent_n);
            std::vector<unsigned char> blob = read_file(dir + "/" + o.file, (size_t)(n_ent * ent_b));
            // The row's layout hook, with this table's canonical plane: [rows, cols] of one entry.
            RadLayout L{};
            int lrc = RAD_E_UNSUPPORTED, sel = -1;
            RadTensor pt{};
            const int pl = plane_of((int)i);
            const RadEncoding* ep = pl == EXL3_P_DN_SUH ? &enc_down : &enc;
            if (row->layout && have_enc && pl >= 0) {
                sel = enc_plane[pl];
                pt.dtype = o.dt;
                pt.rank = 2;
                pt.shape[0] = x.rank == 3 ? x.shape[1] : 1;
                pt.shape[1] = x.rank == 3 ? x.shape[2] : ent_n;
                rad_tensor_pack(&pt);
                lrc = row->layout(P, nP, (int)i, ep, &sel, &pt, 1, &L);
            }
            if (lrc != RAD_OK && lrc != RAD_E_UNSUPPORTED) {
                std::printf("layout %zu refused %d\nrc none\n", i, lrc);
                return 0;
            }
            int64_t stored_b = ent_b;
            std::vector<unsigned char> stored = blob;
            if (lrc == RAD_OK) {                             // re-laid: every entry, and back again
                if (!row->relayout || L.bytes <= 0) die(o.name + ": a layout with no relayout");
                const int64_t al = L.align > 16 ? L.align : 16;
                stored_b = (L.bytes + al - 1) / al * al;
                stored.assign((size_t)(n_ent * stored_b), 0);
                bool inv_ok = true;
                for (int64_t e = 0; e < n_ent; ++e) {
                    pt.data = blob.data() + e * ent_b;
                    if (row->relayout(P, nP, (int)i, ep, &sel, &pt, 1, stored.data() + e * stored_b, L.bytes) != RAD_OK)
                        die(o.name + ": relayout failed");
                    if (row->unrelayout) {
                        std::vector<unsigned char> back((size_t)ent_b, 0);
                        RadTensor bt = pt;
                        bt.data = back.data();
                        inv_ok &= row->unrelayout(P, nP, (int)i, ep, &sel, stored.data() + e * stored_b,
                                                  L.bytes, &bt, 1) == RAD_OK &&
                                  std::memcmp(back.data(), blob.data() + e * ent_b, (size_t)ent_b) == 0;
                    }
                }
                o.laid = L.tag ? L.tag : row->name;
                o.unlaid = row->unrelayout ? (inv_ok ? "ok" : "FAIL") : "absent";
                o.form = "table";                            // a stored form has no stacked reading
                x.dtype = L.dtype;
            } else if (o.form == "stacked") {
                x.stride[0] = ent_n;
            }
            std::printf("layout %zu %s unrelayout %s\n", i, o.laid.c_str(), o.unlaid.c_str());
            o.buf.alloc(stored.size(), 0);
            o.buf.put(stored.data(), stored.size());
            o.written = stored;
            x.data = o.buf.p();
            if (o.form == "table") {
                std::vector<void*> ptrs((size_t)n_ent);
                for (int64_t e = 0; e < n_ent; ++e) ptrs[(size_t)e] = o.buf.p() + e * stored_b;
                o.table.alloc(ptrs.size() * sizeof(void*), 0);
                if (!ptrs.empty()) o.table.put(ptrs.data(), ptrs.size() * sizeof(void*));
                x.data = o.table.p();
                x.stride[0] = 0;
            }
            continue;
        }

        const int64_t rows = x.rank >= 2 ? rad_tensor_rows(&x) : 1;
        const int64_t width = x.rank >= 2 ? x.shape[x.rank - 1] : numel;
        const int64_t pitch = o.pitch > 0 ? o.pitch : width;
        if (pitch < width) die(o.name + ": pitch narrower than the row");
        if (x.rank >= 2) x.stride[x.rank - 2] = pitch;
        const int64_t row_b = rad_dtype_bytes(o.dt, width), pitch_b = rad_dtype_bytes(o.dt, pitch);
        o.buf.alloc((size_t)(rows * pitch_b), o.role == "out" ? 0xFF : 0x5A);
        if (o.role == "in") {
            std::vector<unsigned char> dense = read_file(dir + "/" + o.file, (size_t)(rows * row_b));
            for (int64_t r = 0; r < rows; ++r) o.buf.put(dense.data() + r * row_b, (size_t)row_b, (size_t)(r * pitch_b));
            o.written = o.buf.image();
        }
        x.data = o.buf.p();
    }

    RadArgs a;
    std::memset(&a, 0, sizeof a);
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = P;
    a.n_p = nP;
    a.world_size = 1;
    Mem scratch;
    if (row->scratch) {
        const int64_t need = row->scratch(&a);
        if (need > 0) {
            scratch.alloc((size_t)need, 0);
            a.scratch = scratch.p();
            a.scratch_bytes = need;
        }
    }
    void* stream = nullptr;
#ifdef EXL3_RUN_HIP
    if (g_device) hip_check(hipStreamCreate((hipStream_t*)&stream), "hipStreamCreate");
#endif
    const int rc = row->launch(&a, (RadStream)stream);
#ifdef EXL3_RUN_HIP
    if (g_device) hip_check(hipStreamSynchronize((hipStream_t)stream), "the launch");
#endif
    std::printf("rc %d\n", rc);

    // ---- what it touched
    std::string red_bad, changed;
    for (Opd& o : opd) {
        if (o.role == "absent") continue;
        const std::vector<unsigned char> img = o.buf.image();
        if (!red_ok(img) || (o.form == "table" && !red_ok(o.table.image()))) red_bad += " " + o.name;
        if (o.role == "in" && img != o.written) changed += " " + o.name;
        if (o.role == "wtab" && std::memcmp(img.data() + kRed, o.written.data(), o.written.size()) != 0)
            changed += " " + o.name;
    }
    if (scratch.n && !red_ok(scratch.image())) red_bad += " scratch";
    std::printf("redzone %s%s\n", red_bad.empty() ? "ok" : "FAIL", red_bad.c_str());
    std::printf("inputs %s%s\n", changed.empty() ? "ok" : "CHANGED", changed.c_str());

    for (size_t i = 0; i < opd.size(); ++i) {
        Opd& o = opd[i];
        if (o.role != "out") continue;
        const RadTensor& x = t[i];
        const int64_t rows = rad_tensor_rows(&x), width = x.shape[x.rank - 1];
        const int64_t row_b = rad_dtype_bytes(o.dt, width), pitch_b = rad_dtype_bytes(o.dt, x.stride[x.rank - 2]);
        const std::vector<unsigned char> img = o.buf.image();
        std::ofstream f(dir + "/out_" + o.name + ".bin", std::ios::binary);
        for (int64_t r = 0; r < rows; ++r) f.write((const char*)img.data() + kRed + r * pitch_b, (std::streamsize)row_b);
    }
    return 0;
}
