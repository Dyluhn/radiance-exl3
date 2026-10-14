/* exl3ref.h -- what the two files of the EXL3 host reference share. */
#ifndef EXL3REF_H
#define EXL3REF_H

#include "exl3_abi.h"
#include "exl3_decode.h"

#if EXL3_IFACE_VERSION != 6
#error "exl3ref is written against EXL3_IFACE_VERSION 6 (include/exl3_abi.h)"
#endif
#if EXL3_DECODE_API_VERSION != 1
#error "exl3ref is written against EXL3_DECODE_API_VERSION 1 (common/exl3_decode.h)"
#endif

/* ref.c: the launch of any of the four ops (EXL3_DOWN, EXL3_GATE_UP, EXL3_PREP_GU, EXL3_GU_DN). */
int exl3ref_launch(const RadArgs* a, int op);

#endif /* EXL3REF_H */
