// exl3r4d_mod4.hip -- the decode band's kernels for the _mod4 ops (interface v6: the experts in four classes by number
// mod 4): exl3r4d.hip compiled a second time with four class records in its argument block. The two-class build is the
// same file at EXL3R4D_NCLS 2, so its code objects are what they were; exl3r4d_common.hip.h names the types and the
// launches this build takes.
#define EXL3R4D_NCLS 4
#undef EXL3R4D_WGSTAMP   // a reading build's stamps are the two-class build's (the four-class build takes no reading arm)
#include "exl3r4d.hip"
