/* exl3_link.h -- the host link each discrete AMD GPU of this machine reaches the host through, read from sysfs.
 * Plain C99 + POSIX, header-only (static inline: a C or C++ translation unit includes it), no
 * radiance, nothing touches a device. A C99 unit defines _XOPEN_SOURCE 700 before its first include (realpath).
 *
 * WHAT IT IS FOR. Decisions the plugin must make where no ABI field names the device or its link, and that every rank
 * must make identically (each computes the same answer from the same files, so no rank tells another): the prefill
 * stager's on/off (the architecture, at declare) and the per-card expert split (decode). A shipped plugin may not
 * bake a machine's link into a build define (PLAN §0: TP size, PCIe links, RAM vary; users do not rebuild).
 *
 * A CARD'S LINK IS ITS PATH'S NARROWEST HOP, NOT ITS ENDPOINT. An R9700 behind an x4 slot reports Gen5 x16 at its
 * endpoint and at its board switch; the x4 shows only at the upstream and root ports (the development box:
 * 0000:13:00.0 -> 0000:11:00.0 Gen4 x4 -> 0000:00:02.2 Gen4 x4). So every hop from the device up to the host bridge
 * is read and the minimum taken: speed (GT/s) x width x line code (128/130 from Gen3, 8/10 below) / 8 = GB/s each
 * way. `link_capped` says a hop above the device is narrower than the device's own link.
 *
 * DISCRETE ONLY. An APU's graphics is an amdgpu PCI device too, with an internal "link" that is not one (Gen4 x16
 * on the development box's gfx1036); the engine drops it by HIP's integrated attribute, which cannot be asked at
 * declare. amdgpu's sysfs tells them apart: a discrete card names its VRAM's vendor (mem_info_vram_vendor) or holds
 * at least 4 GiB of VRAM; an APU's carve-out has no vendor and is small.
 *
 * WHICH CARD A RANK IS ON is the caller's: the list is in PCI-address order, which is the device order HIP gives
 * when no *_VISIBLE_DEVICES mask reorders it. A caller that cannot tell should take exl3_link_min_gbps (the
 * narrowest card), which is right for any mapping when every card serves.
 *
 * Every function takes the sysfs root ("/sys" when serving) so a test can hand it a fake tree. */
#ifndef EXL3_LINK_H
#define EXL3_LINK_H

#include <dirent.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EXL3_LINK_UNKNOWN (-1.0)   /* no hop reports a link, no such device, or no sysfs tree */

typedef struct Exl3Link {
    char   bdf[16];          /* "0000:13:00.0" */
    double path_gbps;        /* the narrowest hop to the host bridge, GB/s each way; EXL3_LINK_UNKNOWN if none */
    double endpoint_gbps;    /* the device's own link; EXL3_LINK_UNKNOWN if it reports none */
    int    link_capped;      /* 1 when a hop above the device is narrower than the device's own link */
    char   narrow_bdf[16];   /* the hop nearest the card that sets path_gbps ("" when unknown) */
} Exl3Link;

/* (definitions below, header-only: static inline, C99 and C++) */

/* One hop's GB/s from its sysfs directory; EXL3_LINK_UNKNOWN when it reports no link. */
static inline double exl3_link_hop_gbps(const char* dev_dir);

/* Device `bdf`'s path bandwidth (see above), or EXL3_LINK_UNKNOWN. */
static inline double exl3_link_path_gbps(const char* sysfs, const char* bdf);

/* 1 when `bdf` is a discrete AMD display-class GPU (see above). */
static inline int exl3_link_discrete(const char* sysfs, const char* bdf);

/* The machine's discrete AMD GPUs in PCI-address order, at most `cap` of them into `out`; returns how many there
 * are (may exceed cap), 0 for none, negative when the PCI directory cannot be read. */
static inline int exl3_link_gpus(const char* sysfs, Exl3Link* out, int cap);

/* The narrowest path over every discrete AMD GPU, or EXL3_LINK_UNKNOWN. */
static inline double exl3_link_min_gbps(const char* sysfs);

/* THE CARDS AN ENGINE OF `world_size` RANKS SERVES ON, in rank order, as radiance 1.4.0 picks them
 * (core/engine_bringup.cpp usable_devices): the runtime's GPU agents -- KFD topology nodes with SIMDs, in node order --
 * narrowed by ROCR_VISIBLE_DEVICES, then HIP_VISIBLE_DEVICES (each an index list; the caller passes the variables'
 * values, NULL when unset); integrated graphics dropped when a discrete card is there; then only the cards of the
 * first one's gfx target; rank r on the r-th. Returns how many serving cards were written (at most world_size and
 * cap; fewer than world_size: the machine shows fewer), or negative when the order cannot be reproduced: -1 no KFD
 * topology, -2 a visibility list this does not read (a UUID form, a malformed entry). The caller then falls back
 * to exl3_link_min_gbps, which is right for any mapping when every card serves. */
static inline int exl3_link_serving(const char* sysfs, const char* rocr_visible, const char* hip_visible, int world_size,
                      Exl3Link* out, int cap);


/* The first line of a small sysfs file, newline cut; "" when absent. */
static inline void exl3l_read_line(const char* path, char* buf, size_t cap) {
    buf[0] = '\0';
    FILE* f = fopen(path, "r");
    if (!f) return;
    if (!fgets(buf, (int)cap, f)) buf[0] = '\0';
    fclose(f);
    buf[strcspn(buf, "\n")] = '\0';
}

/* a/b into out; "" when it does not fit (the read then finds nothing: an unknown) */
static inline void exl3l_join(char* out, size_t cap, const char* a, const char* b) {
    const int n = snprintf(out, cap, "%s/%s", a, b);
    if (n < 0 || (size_t)n >= cap) out[0] = '\0';
}

static inline double exl3_link_hop_gbps(const char* dev_dir) {
    char p[PATH_MAX], s[64], w[64];
    exl3l_join(p, sizeof p, dev_dir, "current_link_speed");
    exl3l_read_line(p, s, sizeof s);   /* "16.0 GT/s PCIe" */
    exl3l_join(p, sizeof p, dev_dir, "current_link_width");
    exl3l_read_line(p, w, sizeof w);
    const double gts = atof(s);
    const int width = atoi(w);
    if (gts <= 0.0 || width <= 0) return EXL3_LINK_UNKNOWN;
    return gts * width * (gts >= 8.0 ? 128.0 / 130.0 : 0.8) / 8.0;
}

/* The path of `bdf`, filled into `l` (bdf, path, endpoint, capped, narrow hop). */
static inline void exl3l_path_of(const char* sysfs, const char* bdf, Exl3Link* l) {
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

static inline double exl3_link_path_gbps(const char* sysfs, const char* bdf) {
    Exl3Link l;
    exl3l_path_of(sysfs, bdf, &l);
    return l.path_gbps;
}

static inline int exl3_link_discrete(const char* sysfs, const char* bdf) {
    char d[PATH_MAX], p[PATH_MAX], v[64];
    snprintf(d, sizeof d, "%s/bus/pci/devices/%s", sysfs, bdf);
    exl3l_join(p, sizeof p, d, "vendor");
    exl3l_read_line(p, v, sizeof v);
    if (strcmp(v, "0x1002") != 0) return 0;
    exl3l_join(p, sizeof p, d, "class");
    exl3l_read_line(p, v, sizeof v);
    if (strncmp(v, "0x03", 4) != 0) return 0;
    exl3l_join(p, sizeof p, d, "mem_info_vram_vendor");
    exl3l_read_line(p, v, sizeof v);
    if (v[0]) return 1;
    exl3l_join(p, sizeof p, d, "mem_info_vram_total");
    exl3l_read_line(p, v, sizeof v);
    return atoll(v) >= (4LL << 30);
}

static inline int exl3l_by_bdf(const void* a, const void* b) { return strcmp(((const Exl3Link*)a)->bdf, ((const Exl3Link*)b)->bdf); }

static inline int exl3_link_gpus(const char* sysfs, Exl3Link* out, int cap) {
    char d[PATH_MAX];
    snprintf(d, sizeof d, "%s/bus/pci/devices", sysfs);
    DIR* dir = opendir(d);
    if (!dir) return -1;
    int n = 0, kept = 0;
    Exl3Link all[64];
    for (struct dirent* e; (e = readdir(dir)) != NULL;) {
        if (e->d_name[0] == '.' || strlen(e->d_name) >= sizeof all[0].bdf) continue;
        if (!exl3_link_discrete(sysfs, e->d_name)) continue;
        if (kept < 64) exl3l_path_of(sysfs, e->d_name, &all[kept++]);
        ++n;
    }
    closedir(dir);
    qsort(all, (size_t)kept, sizeof all[0], exl3l_by_bdf);
    for (int i = 0; i < kept && i < cap; ++i) out[i] = all[i];
    return n;
}

static inline double exl3_link_min_gbps(const char* sysfs) {
    Exl3Link l[64];
    const int n = exl3_link_gpus(sysfs, l, 64);
    double m = EXL3_LINK_UNKNOWN;
    for (int i = 0; i < n && i < 64; ++i)
        if (l[i].path_gbps > 0.0 && (m < 0.0 || l[i].path_gbps < m)) m = l[i].path_gbps;
    return m;
}

/* ------------------------------------------------------------------ serving cards */

typedef struct Exl3lAgent { char bdf[16]; long long gfx; } Exl3lAgent;

static inline long long exl3l_prop_of(const char* props, const char* key) {   /* "key value" lines of a KFD properties file */
    const size_t k = strlen(key);
    for (const char* l = props; l && *l; l = strchr(l, '\n') ? strchr(l, '\n') + 1 : NULL)
        if (strncmp(l, key, k) == 0 && l[k] == ' ') return atoll(l + k + 1);
    return -1;
}

/* KFD's GPU nodes in node order (the runtime's agent order); -1 without a topology. */
static inline int exl3l_kfd_agents(const char* sysfs, Exl3lAgent* a, int cap) {
    int n = 0;
    for (int node = 0; node < 256; ++node) {
        char p[PATH_MAX], buf[8192];
        snprintf(p, sizeof p, "%s/class/kfd/kfd/topology/nodes/%d/properties", sysfs, node);
        FILE* f = fopen(p, "r");
        if (!f) { if (node == 0) return -1; break; }
        const size_t got = fread(buf, 1, sizeof buf - 1, f);
        fclose(f);
        buf[got] = '\0';
        if (exl3l_prop_of(buf, "simd_count") <= 0) continue;   /* a CPU node */
        const long long loc = exl3l_prop_of(buf, "location_id"), dom = exl3l_prop_of(buf, "domain");
        if (loc < 0 || n >= cap) continue;
        snprintf(a[n].bdf, sizeof a[n].bdf, "%04x:%02x:%02x.%x", (unsigned)(dom < 0 ? 0 : dom & 0xffff),
                 (unsigned)((loc >> 8) & 0xff), (unsigned)((loc >> 3) & 0x1f), (unsigned)(loc & 7));
        a[n].gfx = exl3l_prop_of(buf, "gfx_target_version");
        ++n;
    }
    return n;
}

/* Keep the entries a visibility list names, in its order; 0 ok (an unset or empty list keeps all), -2 unreadable. */
static inline int exl3l_narrow(Exl3lAgent* a, int* n, const char* list) {
    if (!list || !*list) return 0;
    Exl3lAgent keep[64];
    int k = 0;
    for (const char* s = list; *s;) {
        char* end;
        const long v = strtol(s, &end, 10);
        if (end == s || v < 0 || v >= *n || (*end && *end != ',')) return -2;   /* a UUID, a range, junk */
        if (k < 64) keep[k++] = a[v];
        s = *end ? end + 1 : end;
    }
    memcpy(a, keep, sizeof keep[0] * (size_t)k);
    *n = k;
    return 0;
}

static inline int exl3_link_serving(const char* sysfs, const char* rocr_visible, const char* hip_visible, int world_size,
                      Exl3Link* out, int cap) {
    Exl3lAgent a[64];
    int n = exl3l_kfd_agents(sysfs, a, 64);
    if (n < 0) return -1;
    if (exl3l_narrow(a, &n, rocr_visible) < 0 || exl3l_narrow(a, &n, hip_visible) < 0) return -2;
    int any_discrete = 0;
    for (int i = 0; i < n; ++i) any_discrete |= exl3_link_discrete(sysfs, a[i].bdf);
    long long want = -1;
    int w = 0;
    for (int i = 0; i < n && w < world_size && w < cap; ++i) {
        if (any_discrete && !exl3_link_discrete(sysfs, a[i].bdf)) continue;
        if (want >= 0 && a[i].gfx != want) continue;
        want = a[i].gfx;
        exl3l_path_of(sysfs, a[i].bdf, &out[w++]);
    }
    return w;
}

#endif
