/* exl3r4d.h -- what the host files of the gfx1201 EXL3 library share. */
#ifndef EXL3R4D_H
#define EXL3R4D_H

#include "exl3_abi.h"
#include "exl3_decode.h"
#include "exl3_layout.h"

#if EXL3_IFACE_VERSION != 3
#error "exl3r4d is written against EXL3_IFACE_VERSION 3 (include/exl3_abi.h, PLAN.md D15; v3 = EPI 1, h_rot)"
#endif
#if EXL3_DECODE_API_VERSION != 1 || EXL3_LAYOUT_API_VERSION != 1
#error "exl3r4d is written against EXL3_DECODE_API_VERSION 1 and EXL3_LAYOUT_API_VERSION 1 (common/)"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* The stored codes plane: exllamav3's tile-stream record (common/exl3_layout.h), gate's record then
 * up's for a stacked gate_up. Plugin-private (rad_types.h: >= RAD_DT_PLUGIN_BASE is sized by the
 * layout hook); the logical [rows, cols] stays the shape. */
enum { EXL3R4D_DT_TRELLIS = RAD_DT_PLUGIN_BASE + 0 };

/* The one K the device code is built for (exl3k4). Every other is refused by name (R12). */
enum { EXL3R4D_BITS = 4 };

/* layout.c: the hooks of both rows (gate_up 1 / down 0 is the row's own op). */
int exl3r4d_layout(int gate_up, const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                   const int* sel, const RadTensor* planes, int n_planes, RadLayout* out);
int exl3r4d_relayout(int gate_up, const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                     const int* sel, const RadTensor* planes, int n_planes, void* dst,
                     int64_t dst_bytes);
int exl3r4d_unrelayout(int gate_up, const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                       const int* sel, const void* src, int64_t src_bytes, const RadTensor* planes,
                       int n_planes);

/* The fast K=4 conversions the hooks use (exposed for the property test against common/exl3_layout). */
void exl3r4d_k4_to_record(const uint8_t* codes, int64_t row_bytes, int64_t rows, int64_t cols,
                          uint32_t* record);
void exl3r4d_k4_from_record(const uint32_t* record, int64_t rows, int64_t cols, uint8_t* codes,
                            int64_t row_bytes);

/* launch.c */
int     exl3r4d_launch(const RadArgs* a, RadStream s, int gate_up);
int64_t exl3r4d_scratch(const RadArgs* a, int gate_up);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* EXL3R4D_H */
