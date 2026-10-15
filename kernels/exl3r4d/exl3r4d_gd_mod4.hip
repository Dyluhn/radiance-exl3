// exl3r4d_gd_mod4.hip -- P10 READING BUILDS ONLY (CMake EXL3R4D_GD4_PROTO): the fused gate_up_down kernel at four
// classes -- exl3r4d.hip a third time, its fused section alone (EXL3R4D_GD4), for launch.c's exl3r4d_gd4_proto. A
// kernel that reads no implicit argument gets no hidden block, so its argument segment is the explicit 704 bytes (the
// four class records, down's four, the tail) inside the engine's 768-byte slot; blockDim / gridDim come from constants
// and the block here (exl3r4d_common.hip.h, EXL3R4D_BDIM).
#define EXL3R4D_NCLS 4
#define EXL3R4D_GD4 1
#include "exl3r4d.hip"
