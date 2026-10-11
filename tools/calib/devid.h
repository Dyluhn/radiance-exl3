// devid.h -- one line naming the device an instrument is about to measure, so every log proves which
// card it ran on. run.sh compares the printed PCI address with the one it was asked to measure and
// throws the log away on a mismatch; the HIP index alone is not evidence (its order is the runtime's).
// Part of tools/calib (radiance-exl3 S0.11). Apache-2.0.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdio>

static inline int calib_print_device(int index) {
    hipDeviceProp_t p;
    if (hipGetDeviceProperties(&p, index) != hipSuccess) {
        std::printf("device %d: hipGetDeviceProperties failed\n", index);
        return 1;
    }
    std::printf("device %d pci=%04x:%02x:%02x.0 arch=%s cus=%d lds_per_cu=%zu lds_per_block=%zu clock_khz=%d\n",
                index, p.pciDomainID, p.pciBusID, p.pciDeviceID, p.gcnArchName, p.multiProcessorCount,
                (size_t)p.maxSharedMemoryPerMultiProcessor, (size_t)p.sharedMemPerBlock, p.clockRate);
    return 0;
}
