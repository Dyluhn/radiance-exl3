/* exl3r4d.h -- what the host files of the gfx1201 EXL3 library share. */
#ifndef EXL3R4D_H
#define EXL3R4D_H

#include "exl3_abi.h"
#include "exl3_decode.h"
#include "exl3_layout.h"

#if EXL3_IFACE_VERSION != 5
#error "exl3r4d is written against EXL3_IFACE_VERSION 5 (include/exl3_abi.h, PLAN.md D15; v3 h_rot, v4 a_rot + prep_gu, v5 gate_up_down)"
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

/* The K the decode-band device code is built for (R23: exl3k2 .. exl3k6, codebook mul1). Every other is refused by
 * name (R12). The M > 16 band (P5) is K = 4 only for now: exl3r4d_fill refuses another K for it. */
enum { EXL3R4D_BITS_MIN = 2, EXL3R4D_BITS_MAX = 6 };
static inline int exl3r4d_bits_ok(int bits) { return bits >= EXL3R4D_BITS_MIN && bits <= EXL3R4D_BITS_MAX; }

/* layout.c: the hooks of both rows (gate_up 1 / down 0 is the row's own op). */
int exl3r4d_layout(int gate_up, const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                   const int* sel, const RadTensor* planes, int n_planes, RadLayout* out);
int exl3r4d_relayout(int gate_up, const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                     const int* sel, const RadTensor* planes, int n_planes, void* dst,
                     int64_t dst_bytes);
int exl3r4d_unrelayout(int gate_up, const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                       const int* sel, const void* src, int64_t src_bytes, const RadTensor* planes,
                       int n_planes);

/* The fast conversions the hooks use (exposed for the property test against common/exl3_layout): K = 4's byte
 * shuffle, and any K's symbol-at-a-time form (a tile's 16 rows read once, packed through a 64-bit accumulator). */
void exl3r4d_k4_to_record(const uint8_t* codes, int64_t row_bytes, int64_t rows, int64_t cols,
                          uint32_t* record);
void exl3r4d_k4_from_record(const uint32_t* record, int64_t rows, int64_t cols, uint8_t* codes,
                            int64_t row_bytes);
void exl3r4d_kx_to_record(const uint8_t* codes, int64_t row_bytes, int64_t rows, int64_t cols, int bits,
                          uint32_t* record);
void exl3r4d_kx_from_record(const uint32_t* record, int64_t rows, int64_t cols, int bits, uint8_t* codes,
                            int64_t row_bytes);

/* launch.c */
int     exl3r4d_launch(const RadArgs* a, RadStream s, int gate_up);
int64_t exl3r4d_scratch(const RadArgs* a, int gate_up);
/* v5 moe_exl3_gate_up_down: the fused decode launch, its scratch (xr + h) and its instance (counter pool) */
int     exl3r4d_launch_gd(const RadArgs* a, RadStream s);
int64_t exl3r4d_scratch_gd(const RadArgs* a);
int     exl3r4d_ctr_init_gd(const RadParam* p, int n_p, void** out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* EXL3R4D_H */
