// exl3r4d_gd_mod4.hip -- the fused gate_up_down kernel and its launch for moe_exl3_gate_up_down_mod4 (interface v7):
// exl3r4d.hip a third time at four classes, its fused section alone (EXL3R4D_GD4). The kernel reads no implicit
// argument -- its block size is the constant 256 (EXL3R4D_BDIM), its grid a field of the block -- so it gets no hidden
// block and its argument segment is the explicit 704 bytes (gate_up's four class records, down's four, the tail) inside
// the engine's 768-byte slot (tests/gpu/kernarg_check.sh). The two-class build is the same source at 2, unchanged.
#define EXL3R4D_NCLS 4
#undef EXL3R4D_WGSTAMP   // a reading build's stamps are the two-class build's (the four-class build takes no reading arm)
#define EXL3R4D_GD4 1
#include "exl3r4d.hip"
