// link_test.cpp -- exl3_link.h against a fake sysfs tree shaped like the development box: 0000:03:00.0 Gen5 x16 at
// every hop, 0000:13:00.0 Gen5 x16 at its endpoint and board switch but Gen4 x4 upstream, an APU's graphics at
// 0000:14:00.0 (internal Gen4 x16, no VRAM vendor, 2 GiB carve-out) and a non-AMD device. Prints one line a case;
// exit 0 when every case holds.
#include "../../architectures/qwen4exp_exl3/exl3_link.h"

#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <string>

static std::string g_root;
static int g_fail = 0;   // a setup failure counts too

static void put(const std::string& path, const std::string& text) {
    std::ofstream(path) << text << "\n";
}
static void mkdirs(const std::string& p) {
    for (size_t i = 1; i <= p.size(); ++i)
        if (i == p.size() || p[i] == '/') mkdir(p.substr(0, i).c_str(), 0755);
}
// a hop directory with a link
static std::string hop(const std::string& dir, const char* gts, int width) {
    mkdirs(dir);
    put(dir + "/current_link_speed", std::string(gts) + " GT/s PCIe");
    put(dir + "/current_link_width", std::to_string(width));
    return dir;
}
static void device(const std::string& dir, const std::string& bdf, const char* vendor, const char* cls,
                   long long vram, const char* vram_vendor) {
    put(dir + "/vendor", vendor);
    put(dir + "/class", cls);
    put(dir + "/mem_info_vram_total", std::to_string(vram));
    put(dir + "/mem_info_vram_vendor", vram_vendor);
    mkdirs(g_root + "/bus/pci/devices");
    if (symlink(dir.c_str(), (g_root + "/bus/pci/devices/" + bdf).c_str()) != 0) ++g_fail;
}
static void check(const char* what, bool ok) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_fail;
}
static bool near(double a, double b) { return std::fabs(a - b) < 1e-6 * std::max(1.0, std::fabs(b)); }

int main() {
    char tmpl[] = "/tmp/exl3_link_test.XXXXXX";
    if (!mkdtemp(tmpl)) { std::perror("mkdtemp"); return 2; }
    g_root = tmpl;
    const std::string pci = g_root + "/devices/pci0000:00";
    const long long vram = 32624LL << 20;
    // the fast card: four hops at Gen5 x16
    std::string a = hop(pci + "/0000:00:01.1", "32.0", 16);
    a = hop(a + "/0000:01:00.0", "32.0", 16);
    a = hop(a + "/0000:02:00.0", "32.0", 16);
    a = hop(a + "/0000:03:00.0", "32.0", 16);
    device(a, "0000:03:00.0", "0x1002", "0x030000", vram, "samsung");
    // the card behind an x4 slot: Gen4 x4 at the root port and the upstream port, Gen5 x16 below
    std::string b = hop(pci + "/0000:00:02.2", "16.0", 4);
    b = hop(b + "/0000:11:00.0", "16.0", 4);
    b = hop(b + "/0000:12:00.0", "32.0", 16);
    b = hop(b + "/0000:13:00.0", "32.0", 16);
    device(b, "0000:13:00.0", "0x1002", "0x030000", vram, "samsung");
    // the APU's graphics
    std::string c = hop(pci + "/0000:00:08.1", "16.0", 16);
    c = hop(c + "/0000:14:00.0", "16.0", 16);
    device(c, "0000:14:00.0", "0x1002", "0x030000", 2048LL << 20, "");
    // someone else's card
    std::string d = hop(pci + "/0000:00:03.1", "8.0", 8);
    d = hop(d + "/0000:21:00.0", "8.0", 8);
    device(d, "0000:21:00.0", "0x10de", "0x030000", vram, "micron");

    const double gen5x16 = 32.0 * 16 * 128.0 / 130.0 / 8.0, gen4x4 = 16.0 * 4 * 128.0 / 130.0 / 8.0;
    check("hop: Gen5 x16 = 63.0 GB/s", near(exl3link::hop_gbps(a), gen5x16));
    check("path: the fast card is its links' minimum (Gen5 x16)", near(exl3link::path_gbps(g_root, "0000:03:00.0"), gen5x16));
    check("path: the x4-slot card is Gen4 x4 although its endpoint says Gen5 x16",
          near(exl3link::path_gbps(g_root, "0000:13:00.0"), gen4x4));
    check("path: an unknown device is 0", exl3link::path_gbps(g_root, "0000:99:00.0") == 0.0);
    const std::vector<std::string> gpus = exl3link::discrete_amd_gpus(g_root);
    check("discrete: the two cards, not the APU, not the other vendor",
          gpus.size() == 2 && gpus[0] == "0000:03:00.0" && gpus[1] == "0000:13:00.0");
    check("machine: the narrowest discrete path is the x4 card's", near(exl3link::min_discrete_path_gbps(g_root), gen4x4));
    check("machine: no sysfs tree -> 0 (the caller keeps its default)",
          exl3link::min_discrete_path_gbps(g_root + "/nothing") == 0.0);
    std::printf("link_test: %d failed\n", g_fail);
    std::string rm = "rm -rf '" + g_root + "'";
    if (std::system(rm.c_str()) != 0) std::printf("note: could not remove %s\n", g_root.c_str());
    return g_fail ? 1 : 0;
}
