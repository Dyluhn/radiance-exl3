/* exl3probe_kern.cc -- Step 0 throwaway (S0.3, S0.10): a kernel library whose only job is to be
 * ASKED. It computes nothing: every launch, relayout and unrelayout refuses by name.
 *
 * One op, `exl3probe_mm` (x, w -> y; M ranged, N, K), with two DEVICE rows that split the M range
 * at 16 -- the shape of a decode row and a prefill row reading one weight. Both have a layout hook
 * that accepts ANY encoding as one operand of every plane it selects, and logs each call to
 * stderr: the band (M), N, K, the encoding, how many planes are selected and the slice of each it
 * is handed. That log is the evidence for:
 *   S0.10  one operand may select all planes of a custom scheme; how often layout is asked
 *   S0.3   at --tp 2, the per-rank slice of every plane (codes [1x1], suh [0x1], svh [1x0])
 *
 * Built twice (tests/probe/CMakeLists.txt): `exl3probe_k` returns one tag from both rows;
 * `exl3probe_k2` (-DEXL3PROBE_TWO_TAGS) returns a different tag from the prefill row, which the
 * core must refuse ("wanted in two stored forms") -- the negative control for S0.10.
 *
 * Device-domain rows in a library with no code objects stay selectable on the host backend
 * (core/plugin/loader.cpp leaves device rows out only when the library HAS fat binaries and none
 * runs here), so this runs in a host-only build.
 */
#include "rad_abi.h"
#include "rad_encoding.h"

#include <cstdio>
#include <cstring>

#define NELEM(a) (int)(sizeof(a) / sizeof((a)[0]))

namespace {

#ifdef EXL3PROBE_TWO_TAGS
const char kName[] = "exl3probe_k2";
#else
const char kName[] = "exl3probe_k";
#endif

int refuse_launch(const RadArgs*, RadStream) { return RAD_E_UNSUPPORTED; }

long long param(const RadParam* p, int n_p, const char* key) {
    for (int i = 0; i < n_p; ++i)
        if (std::strcmp(p[i].key, key) == 0) return p[i].ival;
    return -1;
}

int layout_common(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                  const RadTensor* planes, int n_planes, RadLayout* out, const char* tag) {
    /* One write a line: the ranks of a --tp 2 declare run on threads of their own. */
    char name[256], line[1024];
    rad_enc_format(enc, name, sizeof name);
    int at = std::snprintf(line, sizeof line, "exl3probe layout: lib %s tag %s operand %d M=%lld "
                           "N=%lld K=%lld enc %s selected %d of %d:", kName, tag, operand,
                           param(p, n_p, "M"), param(p, n_p, "N"), param(p, n_p, "K"), name,
                           n_planes, enc->n_planes);
    int64_t bytes = 0;
    for (int i = 0; i < n_planes; ++i) {
        const RadEncPlane& pl = enc->plane[sel[i]];
        const int64_t r = planes[i].shape[0], c = planes[i].rank > 1 ? planes[i].shape[1] : 1;
        if (at > 0 && at < (int)sizeof line)
            at += std::snprintf(line + at, sizeof line - at, " %s[%lldx%lld]", pl.role,
                                (long long)r, (long long)c);
        bytes += r * rad_enc_row_bytes(pl.dtype, c);
    }
    std::fprintf(stderr, "%s\n", line);
    out->tag = tag;
    out->dtype = RAD_U8;
    out->rank = 1;
    out->shape[0] = bytes;
    out->align = 256;
    out->bytes = bytes;
    return RAD_OK;
}

int layout_decode(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                  const RadTensor* planes, int n_planes, RadLayout* out) {
    return layout_common(p, n_p, operand, enc, sel, planes, n_planes, out, "exl3probe.joined");
}

int layout_prefill(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                   const int* sel, const RadTensor* planes, int n_planes, RadLayout* out) {
#ifdef EXL3PROBE_TWO_TAGS
    return layout_common(p, n_p, operand, enc, sel, planes, n_planes, out, "exl3probe.other");
#else
    return layout_common(p, n_p, operand, enc, sel, planes, n_planes, out, "exl3probe.joined");
#endif
}

/* The stored form: the selected planes' dense bytes one after another, in selection order.
 * Lossless and derives nothing (rad_abi.h); it exists so a host engine load at --tp 2 runs its
 * per-rank plane cut (core/format/share.cpp) over every weight (S0.3). */
int concat_relayout(const RadParam*, int, int, const RadEncoding* enc, const int* sel,
                    const RadTensor* planes, int n_planes, void* dst, int64_t dst_bytes) {
    int64_t at = 0;
    for (int i = 0; i < n_planes; ++i) {
        const int64_t r = planes[i].shape[0], c = planes[i].rank > 1 ? planes[i].shape[1] : 1;
        const int64_t n = r * rad_enc_row_bytes(enc->plane[sel[i]].dtype, c);
        if (at + n > dst_bytes) return RAD_E_SHAPE;
        std::memcpy(static_cast<char*>(dst) + at, planes[i].data, (size_t)n);
        at += n;
    }
    return at == dst_bytes ? RAD_OK : RAD_E_SHAPE;
}

const RadParamSpec kParams[] = {
    { "M", RAD_P_INT, RAD_REQUIRED, 0 },
    { "N", RAD_P_INT, RAD_REQUIRED, 0 },
    { "K", RAD_P_INT, RAD_REQUIRED, 0 },
};
const RadOperandSpec kOperands[] = {
    { "x", RAD_OPD_IN, 0 },
    { "w", RAD_OPD_WEIGHT, 0 },
    { "y", RAD_OPD_OUT, 0 },
};
const RadOpSchema kSchemas[] = {
    { "exl3probe_mm", kParams, NELEM(kParams), kOperands, NELEM(kOperands),
      "Step 0 probe: y = x w^T over a weight of any encoding (declare only; never launches)" },
};

const RadConstraint kDecode[] = { RAD_CLE("M", 16) };

const RadKernelInfo kKernels[] = {
    { .name = "exl3probe_mm_m16", .op = "exl3probe_mm", .family = "probe",
      .computes = "nothing; declare-time probe", .shape = "M <= 16", .dtypes = "any",
      .domain = RAD_DOMAIN_DEVICE, .priority = 20,
      .constraints = kDecode, .n_constraints = NELEM(kDecode),
      .launch = refuse_launch, .layout = layout_decode, .relayout = concat_relayout },
    { .name = "exl3probe_mm_any", .op = "exl3probe_mm", .family = "probe",
      .computes = "nothing; declare-time probe", .shape = "any M", .dtypes = "any",
      .domain = RAD_DOMAIN_DEVICE, .priority = 10,
      .launch = refuse_launch, .layout = layout_prefill, .relayout = concat_relayout },
};

const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL, kName, "0.0.1", "Step 0 probe kernel library (throwaway; computes nothing)",
    "host",
};

}  // namespace

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }
extern "C" int rad_kernel_schema_count(void) { return NELEM(kSchemas); }
extern "C" const RadOpSchema* rad_kernel_schema_at(int i) {
    return (i < 0 || i >= NELEM(kSchemas)) ? nullptr : &kSchemas[i];
}
extern "C" int rad_kernel_count(void) { return NELEM(kKernels); }
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    return (i < 0 || i >= NELEM(kKernels)) ? nullptr : &kKernels[i];
}
