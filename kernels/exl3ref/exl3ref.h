/* exl3ref.h -- what the two files of the EXL3 host reference share. */
#ifndef EXL3REF_H
#define EXL3REF_H

#include "exl3_abi.h"
#include "exl3_decode.h"

#if EXL3_IFACE_VERSION != 3
#error "exl3ref is written against EXL3_IFACE_VERSION 3 (include/exl3_abi.h)"
#endif
#if EXL3_DECODE_API_VERSION != 1
#error "exl3ref is written against EXL3_DECODE_API_VERSION 1 (common/exl3_decode.h)"
#endif

/* ref.c: the launch of either op (gate_up 1, down 0). */
int exl3ref_launch(const RadArgs* a, int gate_up);

#endif /* EXL3REF_H */
