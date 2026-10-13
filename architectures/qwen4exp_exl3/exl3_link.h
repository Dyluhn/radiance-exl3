/* exl3_link.h -- the host link each discrete AMD GPU of this machine reaches the host through, read from sysfs, for
 * decisions the architecture must make at declare (where no ABI field names the device or its link): the prefill
 * stager's on/off (QWEN4EXP_EXL3_PREFILL_STAGE) is the first.
 *
 * A CARD'S LINK IS ITS PATH'S NARROWEST HOP, NOT ITS ENDPOINT. An R9700 behind an x4 slot reports Gen5 x16 at its
 * endpoint and at its board switch; the x4 shows only at the upstream port and the root port (measured on the
 * development box: 0000:13:00.0 -> 0000:11:00.0 Gen4 x4 -> 0000:00:02.2 Gen4 x4). So every hop from the device up to
 * the root complex is read and the minimum taken: speed (GT/s) x width x line code (128/130 from Gen3, 8/10 below)
 * / 8 = GB/s each way.
 *
 * DISCRETE ONLY. An APU's graphics is an amdgpu PCI device too, with an internal "link" that is not one (Gen4 x16 on
 * the development box's gfx1036); the engine drops it by HIP's integrated attribute, which the architecture cannot
 * ask at declare. amdgpu's sysfs tells them apart: a discrete card names its VRAM's vendor (mem_info_vram_vendor)
 * and carries gigabytes of it; an APU's carve-out has no vendor and is small. Both are checked.
 *
 * Every function takes the sysfs root ("/sys" when serving) so a test can hand it a fake tree. Standard library
 * and POSIX only; nothing here touches a device. */
#pragma once

#include <dirent.h>
#include <limits.h>
#include <stdlib.h>

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

namespace exl3link {

inline std::string read_line(const std::string& path) {
    std::ifstream f(path);
    std::string s;
    if (f) std::getline(f, s);
    return s;
}

/* One hop: GB/s each way from its current speed and width; 0 when the hop reports no link. */
inline double hop_gbps(const std::string& dev) {
    const double gts = std::atof(read_line(dev + "/current_link_speed").c_str());   /* "16.0 GT/s PCIe" */
    const int width = std::atoi(read_line(dev + "/current_link_width").c_str());
    if (gts <= 0.0 || width <= 0) return 0.0;
    return gts * width * (gts >= 8.0 ? 128.0 / 130.0 : 0.8) / 8.0;
}

/* The device's path to the root complex, narrowest hop; 0 when no hop reports a link (or no such device). */
inline double path_gbps(const std::string& sys, const std::string& bdf) {
    char buf[PATH_MAX];
    if (!realpath((sys + "/bus/pci/devices/" + bdf).c_str(), buf)) return 0.0;
    std::string p = buf;
    double m = 0.0;
    /* up through the PCI hierarchy: stop at the host bridge directory ("pci0000:00") */
    while (p.size() > 1) {
        const std::string leaf = p.substr(p.find_last_of('/') + 1);
        if (leaf.rfind("pci", 0) == 0 || leaf.find(':') == std::string::npos) break;
        const double g = hop_gbps(p);
        if (g > 0.0) m = m > 0.0 ? std::min(m, g) : g;
        p = p.substr(0, p.find_last_of('/'));
    }
    return m;
}

/* An AMD display-class device with VRAM of its own (see the header comment). */
inline bool discrete_amd_gpu(const std::string& sys, const std::string& bdf) {
    const std::string d = sys + "/bus/pci/devices/" + bdf;
    if (read_line(d + "/vendor") != "0x1002") return false;
    const std::string cls = read_line(d + "/class");
    if (cls.rfind("0x03", 0) != 0) return false;
    const long long vram = std::atoll(read_line(d + "/mem_info_vram_total").c_str());
    return !read_line(d + "/mem_info_vram_vendor").empty() || vram >= (4LL << 30);
}

/* The machine's discrete AMD GPUs, by PCI address. */
inline std::vector<std::string> discrete_amd_gpus(const std::string& sys) {
    std::vector<std::string> out;
    DIR* dir = opendir((sys + "/bus/pci/devices").c_str());
    if (!dir) return out;
    while (const dirent* e = readdir(dir)) {
        const std::string n = e->d_name;
        if (n.empty() || n[0] == '.') continue;
        if (discrete_amd_gpu(sys, n)) out.push_back(n);
    }
    closedir(dir);
    std::sort(out.begin(), out.end());
    return out;
}

/* The narrowest host path over every discrete AMD GPU (GB/s); 0 when none is found or none reports a link -- the
 * caller then keeps its default. Every rank computes the same value from the same files, so no rank has to tell
 * another what it decided. */
inline double min_discrete_path_gbps(const std::string& sys) {
    double m = 0.0;
    for (const std::string& b : discrete_amd_gpus(sys)) {
        const double g = path_gbps(sys, b);
        if (g > 0.0) m = m > 0.0 ? std::min(m, g) : g;
    }
    return m;
}

}  // namespace exl3link
