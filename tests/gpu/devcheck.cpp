// devcheck -- print what HIP sees, so a GPU session can prove which card it ran on.
//
//   devcheck     prints "visible <n>" and one "device <i> pci <domain:bus:dev.fn> <gcnArchName>" line a device
//
// run_gpu.sh runs it in the same container, with the same visibility variable, as the job it guards, and refuses
// the job unless exactly the asked-for card is device 0. Host API only; no kernel runs.
#include <hip/hip_runtime_api.h>

#include <cstdio>

int main() {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess) {
        std::printf("visible 0 (hipGetDeviceCount failed)\n");
        return 1;
    }
    std::printf("visible %d\n", n);
    for (int i = 0; i < n; ++i) {
        char bus[32] = {0};
        hipDeviceProp_t p;
        if (hipDeviceGetPCIBusId(bus, (int)sizeof bus, i) != hipSuccess || hipGetDeviceProperties(&p, i) != hipSuccess)
            return 1;
        for (char* c = bus; *c; ++c)
            if (*c >= 'A' && *c <= 'F') *c = (char)(*c - 'A' + 'a');
        std::printf("device %d pci %s %s\n", i, bus, p.gcnArchName);
    }
    return 0;
}
