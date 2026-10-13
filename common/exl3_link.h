/* exl3_link.h -- the host link each discrete AMD GPU of this machine reaches the host through, read from sysfs.
 * Plain C99 + POSIX, no radiance, nothing touches a device.
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

#ifdef __cplusplus
extern "C" {
#endif

#define EXL3_LINK_UNKNOWN (-1.0)   /* no hop reports a link, no such device, or no sysfs tree */

typedef struct Exl3Link {
    char   bdf[16];          /* "0000:13:00.0" */
    double path_gbps;        /* the narrowest hop to the host bridge, GB/s each way; EXL3_LINK_UNKNOWN if none */
    double endpoint_gbps;    /* the device's own link; EXL3_LINK_UNKNOWN if it reports none */
    int    link_capped;      /* 1 when a hop above the device is narrower than the device's own link */
    char   narrow_bdf[16];   /* the hop nearest the card that sets path_gbps ("" when unknown) */
} Exl3Link;

/* One hop's GB/s from its sysfs directory; EXL3_LINK_UNKNOWN when it reports no link. */
double exl3_link_hop_gbps(const char* dev_dir);

/* Device `bdf`'s path bandwidth (see above), or EXL3_LINK_UNKNOWN. */
double exl3_link_path_gbps(const char* sysfs, const char* bdf);

/* 1 when `bdf` is a discrete AMD display-class GPU (see above). */
int exl3_link_discrete(const char* sysfs, const char* bdf);

/* The machine's discrete AMD GPUs in PCI-address order, at most `cap` of them into `out`; returns how many there
 * are (may exceed cap), 0 for none, negative when the PCI directory cannot be read. */
int exl3_link_gpus(const char* sysfs, Exl3Link* out, int cap);

/* The narrowest path over every discrete AMD GPU, or EXL3_LINK_UNKNOWN. */
double exl3_link_min_gbps(const char* sysfs);

#ifdef __cplusplus
}
#endif
#endif
