/* exl3_link.c -- see exl3_link.h. */
#define _XOPEN_SOURCE 700
#include "exl3_link.h"

#include <dirent.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The first line of a small sysfs file, newline cut; "" when absent. */
static void read_line(const char* path, char* buf, size_t cap) {
    buf[0] = '\0';
    FILE* f = fopen(path, "r");
    if (!f) return;
    if (!fgets(buf, (int)cap, f)) buf[0] = '\0';
    fclose(f);
    buf[strcspn(buf, "\n")] = '\0';
}

static void join(char* out, size_t cap, const char* a, const char* b) { snprintf(out, cap, "%s/%s", a, b); }

double exl3_link_hop_gbps(const char* dev_dir) {
    char p[PATH_MAX], s[64], w[64];
    join(p, sizeof p, dev_dir, "current_link_speed");
    read_line(p, s, sizeof s);   /* "16.0 GT/s PCIe" */
    join(p, sizeof p, dev_dir, "current_link_width");
    read_line(p, w, sizeof w);
    const double gts = atof(s);
    const int width = atoi(w);
    if (gts <= 0.0 || width <= 0) return EXL3_LINK_UNKNOWN;
    return gts * width * (gts >= 8.0 ? 128.0 / 130.0 : 0.8) / 8.0;
}

/* The path of `bdf`, filled into `l` (bdf, path, endpoint, capped, narrow hop). */
static void path_of(const char* sysfs, const char* bdf, Exl3Link* l) {
    memset(l, 0, sizeof *l);
    snprintf(l->bdf, sizeof l->bdf, "%s", bdf);
    l->path_gbps = l->endpoint_gbps = EXL3_LINK_UNKNOWN;
    char link[PATH_MAX], p[PATH_MAX];
    snprintf(link, sizeof link, "%s/bus/pci/devices/%s", sysfs, bdf);
    if (!realpath(link, p)) return;
    int first = 1;
    for (;;) {   /* up through the PCI hierarchy; stop at the host bridge ("pci0000:00") or a non-PCI directory */
        char* slash = strrchr(p, '/');
        if (!slash || slash == p) break;
        const char* leaf = slash + 1;
        if (strncmp(leaf, "pci", 3) == 0 || !strchr(leaf, ':')) break;
        const double g = exl3_link_hop_gbps(p);
        if (first) l->endpoint_gbps = g;
        first = 0;
        if (g > 0.0 && (l->path_gbps < 0.0 || g < l->path_gbps)) {
            l->path_gbps = g;
            snprintf(l->narrow_bdf, sizeof l->narrow_bdf, "%s", leaf);
        }
        *slash = '\0';
    }
    l->link_capped = l->endpoint_gbps > 0.0 && l->path_gbps > 0.0 && l->path_gbps < l->endpoint_gbps;
}

double exl3_link_path_gbps(const char* sysfs, const char* bdf) {
    Exl3Link l;
    path_of(sysfs, bdf, &l);
    return l.path_gbps;
}

int exl3_link_discrete(const char* sysfs, const char* bdf) {
    char d[PATH_MAX], p[PATH_MAX], v[64];
    snprintf(d, sizeof d, "%s/bus/pci/devices/%s", sysfs, bdf);
    join(p, sizeof p, d, "vendor");
    read_line(p, v, sizeof v);
    if (strcmp(v, "0x1002") != 0) return 0;
    join(p, sizeof p, d, "class");
    read_line(p, v, sizeof v);
    if (strncmp(v, "0x03", 4) != 0) return 0;
    join(p, sizeof p, d, "mem_info_vram_vendor");
    read_line(p, v, sizeof v);
    if (v[0]) return 1;
    join(p, sizeof p, d, "mem_info_vram_total");
    read_line(p, v, sizeof v);
    return atoll(v) >= (4LL << 30);
}

static int by_bdf(const void* a, const void* b) { return strcmp(((const Exl3Link*)a)->bdf, ((const Exl3Link*)b)->bdf); }

int exl3_link_gpus(const char* sysfs, Exl3Link* out, int cap) {
    char d[PATH_MAX];
    snprintf(d, sizeof d, "%s/bus/pci/devices", sysfs);
    DIR* dir = opendir(d);
    if (!dir) return -1;
    int n = 0, kept = 0;
    Exl3Link all[64];
    for (struct dirent* e; (e = readdir(dir)) != NULL;) {
        if (e->d_name[0] == '.' || strlen(e->d_name) >= sizeof all[0].bdf) continue;
        if (!exl3_link_discrete(sysfs, e->d_name)) continue;
        if (kept < 64) path_of(sysfs, e->d_name, &all[kept++]);
        ++n;
    }
    closedir(dir);
    qsort(all, (size_t)kept, sizeof all[0], by_bdf);
    for (int i = 0; i < kept && i < cap; ++i) out[i] = all[i];
    return n;
}

double exl3_link_min_gbps(const char* sysfs) {
    Exl3Link l[64];
    const int n = exl3_link_gpus(sysfs, l, 64);
    double m = EXL3_LINK_UNKNOWN;
    for (int i = 0; i < n && i < 64; ++i)
        if (l[i].path_gbps > 0.0 && (m < 0.0 || l[i].path_gbps < m)) m = l[i].path_gbps;
    return m;
}
