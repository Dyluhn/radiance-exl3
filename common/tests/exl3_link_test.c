/* exl3_link_test.c -- exl3_link.c against a fake sysfs tree shaped like the development box: 0000:03:00.0 Gen5 x16 at
 * every hop, 0000:13:00.0 Gen5 x16 at its endpoint and board switch but Gen4 x4 upstream, an APU's graphics at
 * 0000:14:00.0 (internal Gen4 x16, no VRAM vendor, 2 GiB carve-out) and another vendor's card. One line a case. */
#define _XOPEN_SOURCE 700
#include "exl3_link.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static char g_root[64];
static int g_fail = 0;   /* a setup failure counts too */

static void put(const char* dir, const char* name, const char* text) {
    char p[1024];
    if (snprintf(p, sizeof p, "%s/%s", dir, name) >= (int)sizeof p) { ++g_fail; return; }
    FILE* f = fopen(p, "w");
    if (!f) { ++g_fail; return; }
    fprintf(f, "%s\n", text);
    fclose(f);
}
static void mkdirs(const char* path) {
    char p[1024];
    if (snprintf(p, sizeof p, "%s", path) >= (int)sizeof p) { ++g_fail; return; }
    for (char* s = p + 1; *s; ++s)
        if (*s == '/') { *s = '\0'; mkdir(p, 0755); *s = '/'; }
    mkdir(p, 0755);
}
/* hop: dir = parent/name with a link of `gts` GT/s x `width`; returns it in `out` */
static void hop(char* out, const char* parent, const char* name, const char* gts, int width) {
    char s[64], w[16];
    sprintf(out, "%s/%s", parent, name);
    mkdirs(out);
    snprintf(s, sizeof s, "%s GT/s PCIe", gts);
    snprintf(w, sizeof w, "%d", width);
    put(out, "current_link_speed", s);
    put(out, "current_link_width", w);
}
static void device(const char* dir, const char* bdf, const char* vendor, long long vram, const char* vram_vendor) {
    char v[32], p[512];
    put(dir, "vendor", vendor);
    put(dir, "class", "0x030000");
    snprintf(v, sizeof v, "%lld", vram);
    put(dir, "mem_info_vram_total", v);
    put(dir, "mem_info_vram_vendor", vram_vendor);
    snprintf(p, sizeof p, "%s/bus/pci/devices", g_root);
    mkdirs(p);
    snprintf(p, sizeof p, "%s/bus/pci/devices/%s", g_root, bdf);
    if (symlink(dir, p) != 0) ++g_fail;
}
static void check(const char* what, int ok) {
    printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_fail;
}
static int near(double a, double b) { return fabs(a - b) < 1e-9 * (fabs(b) > 1 ? fabs(b) : 1); }

int main(void) {
    strcpy(g_root, "/tmp/exl3_link_test.XXXXXX");
    if (!mkdtemp(g_root)) { perror("mkdtemp"); return 2; }
    char pci[512], a[512], b[512], c[512], d[512], t[512];
    const long long vram = 32624LL << 20;
    snprintf(pci, sizeof pci, "%s/devices/pci0000:00", g_root);
    hop(a, pci, "0000:00:01.1", "32.0", 16); hop(t, a, "0000:01:00.0", "32.0", 16);
    hop(a, t, "0000:02:00.0", "32.0", 16);   hop(t, a, "0000:03:00.0", "32.0", 16);
    device(t, "0000:03:00.0", "0x1002", vram, "samsung");
    hop(b, pci, "0000:00:02.2", "16.0", 4);  hop(t, b, "0000:11:00.0", "16.0", 4);
    hop(b, t, "0000:12:00.0", "32.0", 16);   hop(t, b, "0000:13:00.0", "32.0", 16);
    device(t, "0000:13:00.0", "0x1002", vram, "samsung");
    hop(c, pci, "0000:00:08.1", "16.0", 16); hop(t, c, "0000:14:00.0", "16.0", 16);
    device(t, "0000:14:00.0", "0x1002", 2048LL << 20, "");
    hop(d, pci, "0000:00:03.1", "8.0", 8);   hop(t, d, "0000:21:00.0", "8.0", 8);
    device(t, "0000:21:00.0", "0x10de", vram, "micron");

    const double g5x16 = 32.0 * 16 * 128.0 / 130.0 / 8.0, g4x4 = 16.0 * 4 * 128.0 / 130.0 / 8.0;
    Exl3Link l[8];
    const int n = exl3_link_gpus(g_root, l, 8);
    check("gpus: the two cards, in PCI order -- not the APU, not the other vendor",
          n == 2 && !strcmp(l[0].bdf, "0000:03:00.0") && !strcmp(l[1].bdf, "0000:13:00.0"));
    check("fast card: path = endpoint = Gen5 x16 (63.0 GB/s), not capped",
          n == 2 && near(l[0].path_gbps, g5x16) && near(l[0].endpoint_gbps, g5x16) && !l[0].link_capped);
    check("x4-slot card: endpoint Gen5 x16 but path Gen4 x4 (7.9 GB/s), capped, set at the upstream port 0000:11:00.0",
          n == 2 && near(l[1].endpoint_gbps, g5x16) && near(l[1].path_gbps, g4x4) && l[1].link_capped &&
              !strcmp(l[1].narrow_bdf, "0000:11:00.0"));
    check("path of a named device", near(exl3_link_path_gbps(g_root, "0000:13:00.0"), g4x4));
    check("an unknown device -> EXL3_LINK_UNKNOWN", exl3_link_path_gbps(g_root, "0000:99:00.0") == EXL3_LINK_UNKNOWN);
    check("machine minimum = the x4 card's", near(exl3_link_min_gbps(g_root), g4x4));
    char none[600];
    snprintf(none, sizeof none, "%s/nothing", g_root);
    check("no sysfs tree -> EXL3_LINK_UNKNOWN, and no GPU list",
          exl3_link_min_gbps(none) == EXL3_LINK_UNKNOWN && exl3_link_gpus(none, l, 8) < 0);
    /* KFD topology: node 0 the CPU, then the cards in the runtime's order, the APU last (the box's own order) */
    struct { const char* props; } nodes[4] = {
        {"cpu_cores_count 16\nsimd_count 0\nlocation_id 0\ndomain 0\ngfx_target_version 0"},
        {"simd_count 128\nlocation_id 768\ndomain 0\ngfx_target_version 120001"},
        {"simd_count 128\nlocation_id 4864\ndomain 0\ngfx_target_version 120001"},
        {"simd_count 4\nlocation_id 5120\ndomain 0\ngfx_target_version 100306"} };
    for (int i = 0; i < 4; ++i) {
        char dir[600];
        snprintf(dir, sizeof dir, "%s/class/kfd/kfd/topology/nodes/%d", g_root, i);
        mkdirs(dir);
        put(dir, "properties", nodes[i].props);
    }
    Exl3Link sv[8];
    int m = exl3_link_serving(g_root, NULL, NULL, 2, sv, 8);
    check("serving tp2: rank 0 on 0000:03:00.0, rank 1 on 0000:13:00.0 (the APU never)",
          m == 2 && !strcmp(sv[0].bdf, "0000:03:00.0") && !strcmp(sv[1].bdf, "0000:13:00.0") &&
              near(sv[1].path_gbps, g4x4));
    m = exl3_link_serving(g_root, NULL, NULL, 1, sv, 8);
    check("serving tp1: the first card only (Gen5 x16)", m == 1 && near(sv[0].path_gbps, g5x16));
    m = exl3_link_serving(g_root, NULL, "1", 1, sv, 8);
    check("HIP_VISIBLE_DEVICES=1: tp1 on the x4-slot card", m == 1 && !strcmp(sv[0].bdf, "0000:13:00.0"));
    m = exl3_link_serving(g_root, "2,0", NULL, 2, sv, 8);
    check("ROCR_VISIBLE_DEVICES=2,0: the APU dropped beside a discrete card -> one serving card",
          m == 1 && !strcmp(sv[0].bdf, "0000:03:00.0"));
    check("a UUID visibility list -> -2 (the caller falls back)", exl3_link_serving(g_root, "GPU-1234", NULL, 2, sv, 8) == -2);
    check("an index past the agents -> -2", exl3_link_serving(g_root, NULL, "5", 1, sv, 8) == -2);
    check("no KFD topology -> -1", exl3_link_serving(none, NULL, NULL, 2, sv, 8) == -1);
    printf("exl3_link_test: %d failed\n", g_fail);
    char rm[600];
    snprintf(rm, sizeof rm, "rm -rf '%s'", g_root);
    if (system(rm) != 0) printf("note: could not remove %s\n", g_root);
    return g_fail ? 1 : 0;
}
