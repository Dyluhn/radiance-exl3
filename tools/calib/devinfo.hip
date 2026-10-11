// devinfo.hip -- every device HIP can see, one line each, by HIP index. run.sh reads this (with no
// HIP_VISIBLE_DEVICES set) to find the index of the card it was asked for. radiance-exl3 S0.11. Apache-2.0.
#include "devid.h"

int main() {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess) { std::printf("hipGetDeviceCount failed\n"); return 1; }
    for (int i = 0; i < n; ++i)
        if (calib_print_device(i)) return 1;
    return 0;
}
