/* exl3probe_quant.cc -- Step 0 throwaway (S0.2): a quantiser plugin that writes a scheme of its own
 * shaped like PLAN.md D4, so the core's handling of such a scheme can be observed at 1.3.0.
 *
 * NOT the EXL3 quantiser (that is quantizers/exl3, owner Q). The numbers are a plain 4-bit
 * grid, chosen so the decode is exact enough to read off rad-convert's error report:
 *
 *   scheme "exl3", transform "mul1/h128" (a string the core does not define)
 *   codes  u4  TILED [1x1]   code = clamp(round(x / (svh[i] * s[j])) + 8, 0, 15)
 *   suh    f16 TILED [0x1]   s[j] for the gate half (rows < R/2) or the whole weight: +1/-1 by column
 *   suh.1  f16 TILED [0x1]   s[j] for the up half (rows >= R/2) -- gate_up only, so 4 planes
 *   svh    f16 TILED [1x0]   per row: max|x| / 7 (1 for an all-zero row)
 *
 * The suh/suh.1 signs depend on the column only, so every row_block call writes the same values
 * into those whole-extent planes. The decode needs every plane: a decoder that skipped suh.1 or
 * misread svh would show in the error report as a relative error near 1.
 *
 * Built twice: `exl3probe_quant` and, with -DEXL3PROBE_BROKEN_DECODE, `exl3probe_quant_bad`, whose
 * decode reads suh for every row (suh.1 never) -- the negative control: rad-convert's error report
 * for gate_up must move from the grid's error to about 1 on the up half.
 *
 * The decoder sees only the view it is handed (core/format/encoding.cpp enc_decode_rows: rows =
 * the caller's chunk or slice, no absolute row), so "up half" means rows >= w->rows / 2 of the
 * VIEW. The first EXL3PROBE_LOG_DECODES calls are logged to stderr to show what it is handed.
 */
#include "rad_plugin.h"
#include "rad_quant.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace {

const char kScheme[] = "exl3";
const char kTransform[] = "mul1/h128";
enum { EXL3PROBE_LOG_DECODES = 4 };

bool is_gate_up(const char* name) { return name && std::strstr(name, "gate_up") != nullptr; }

/* +1/-1 by column: two different patterns, so reading suh for the up half is visible. */
float sign_gate(int64_t j) { return (j % 2) ? -1.0f : 1.0f; }
float sign_up(int64_t j) { return (j % 3) ? 1.0f : -1.0f; }

int enc(const RadParam*, int, const RadQuantWeight* w, RadEncoding* out) {
    if (w->rank != 2 || w->cols % 2) return RAD_E_UNSUPPORTED;
    const bool gu = is_gate_up(w->name);
    if (gu && w->rows % 2) return RAD_E_SHAPE;
    rad_enc_clear(out);
    rad_enc_copy_str(out->scheme, kScheme);
    rad_enc_copy_str(out->transform, kTransform);
    rad_enc_add_plane(out, "codes", RAD_U4, 1, 1);
    rad_enc_add_plane(out, "suh", RAD_F16, 0, 1);
    if (gu) rad_enc_add_plane(out, "suh.1", RAD_F16, 0, 1);
    rad_enc_add_plane(out, "svh", RAD_F16, 1, 0);
    return RAD_OK;
}

/* 16 rows a call: the P1 brief's row_block, legal beside [0x1] planes (rad_convert.cpp holds a
 * plane with block rows 0 whole across calls). */
int64_t row_block(const RadParam*, int, const RadQuantWeight*) { return 16; }

int quantize(const RadParam*, int, const RadQuantWeight* w, float* src, int64_t row0, int64_t rows,
             void* const* planes) {
    const bool gu = is_gate_up(w->name);
    const int64_t C = w->cols, half = w->rows / 2;
    uint8_t* codes = static_cast<uint8_t*>(planes[0]);
    uint16_t* suh = static_cast<uint16_t*>(planes[1]);
    uint16_t* suh1 = gu ? static_cast<uint16_t*>(planes[2]) : nullptr;
    uint16_t* svh = static_cast<uint16_t*>(planes[gu ? 3 : 2]);
    for (int64_t j = 0; j < C; ++j) {
        suh[j] = rad_f32_to_f16(sign_gate(j));
        if (suh1) suh1[j] = rad_f32_to_f16(sign_up(j));
    }
    for (int64_t i = 0; i < rows; ++i) {
        const float* x = src + i * C;
        float m = 0.0f;
        for (int64_t j = 0; j < C; ++j) m = std::fmax(m, std::fabs(x[j]));
        const uint16_t sh = rad_f32_to_f16(m > 0.0f ? m / 7.0f : 1.0f);
        svh[i] = sh;
        const float s_row = rad_f16_to_f32(sh);
        const bool up = gu && row0 + i >= half;
        uint8_t* q = codes + i * (C / 2);
        for (int64_t j = 0; j < C; ++j) {
            const float s = s_row * (up ? sign_up(j) : sign_gate(j));
            float c = std::nearbyint(x[j] / s) + 8.0f;
            c = c < 0.0f ? 0.0f : c > 15.0f ? 15.0f : c;
            const uint8_t v = (uint8_t)c;
            q[j / 2] |= (j % 2) ? (uint8_t)(v << 4) : v;   /* LSB-first (rad_encoding.h) */
        }
    }
    return RAD_OK;
}

int decode(const RadEncoding* e, const RadQuantWeight* w, const void* const* planes, int64_t row0,
           int64_t rows, float* out) {
    if (std::strcmp(e->scheme, kScheme) != 0) return RAD_E_UNSUPPORTED;
    static int logged = 0;
    const bool gu = e->n_planes == 4;
    if (logged < EXL3PROBE_LOG_DECODES) {
        ++logged;
        std::fprintf(stderr, "exl3probe decode: view %lldx%lld, rows [%lld, %lld), %d planes, "
                     "transform '%s'\n", (long long)w->rows, (long long)w->cols, (long long)row0,
                     (long long)(row0 + rows), e->n_planes, e->transform);
    }
    const int64_t C = w->cols, half = w->rows / 2;
    const uint8_t* codes = static_cast<const uint8_t*>(planes[0]);
    const uint16_t* suh = static_cast<const uint16_t*>(planes[1]);
    const uint16_t* suh1 = gu ? static_cast<const uint16_t*>(planes[2]) : nullptr;
    const uint16_t* svh = static_cast<const uint16_t*>(planes[gu ? 3 : 2]);
    for (int64_t i = 0; i < rows; ++i) {
        const int64_t r = row0 + i;
#ifdef EXL3PROBE_BROKEN_DECODE
        const uint16_t* s = suh;
        (void)suh1;
        (void)half;
#else
        const uint16_t* s = gu && r >= half ? suh1 : suh;
#endif
        const float s_row = rad_f16_to_f32(svh[r]);
        const uint8_t* q = codes + r * (C / 2);
        for (int64_t j = 0; j < C; ++j) {
            const int v = (j % 2) ? (q[j / 2] >> 4) : (q[j / 2] & 15);
            out[i * C + j] = (float)(v - 8) * s_row * rad_f16_to_f32(s[j]);
        }
    }
    return RAD_OK;
}

const RadQuantizerInfo kQuant[] = {{
    "exl3probe", "Step 0 probe: a 4-bit grid in a D4-shaped custom scheme, decoded by its own hook",
    nullptr, 0, enc, row_block, quantize, decode,
}};

#ifdef EXL3PROBE_BROKEN_DECODE
const RadPluginInfo kInfo = {
    RAD_PLUGIN_QUANT, "exl3probe_quant_bad", "0.0.1",
    "Step 0 probe quantiser with a decode that ignores suh.1 (negative control)", "host",
};
#else
const RadPluginInfo kInfo = {
    RAD_PLUGIN_QUANT, "exl3probe_quant", "0.0.1", "Step 0 probe quantiser (throwaway)", "host",
};
#endif

}  // namespace

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }
extern "C" int rad_quant_count(void) { return 1; }
extern "C" const RadQuantizerInfo* rad_quant_at(int i) { return i == 0 ? &kQuant[0] : nullptr; }
