/* rows_static_test.c -- static checks over exl3r4d's row table, read from the built plugin (host, no device call).
 *
 *   rows_static_test <exl3r4d.so>
 *
 * 1. A row with a layout hook claims a stored form for some operand, and the loader then has to PRODUCE that form
 *    (relayout) and the oracle / a domain move has to give the planes back (unrelayout): every row that registers
 *    .layout must register .relayout and .unrelayout too. (Caught late once: 65b33db claimed "exl3r4d.f16.vec" for
 *    the scale planes and the prep_gu row had only .layout -- the engine refused at load, P3's e2e, integrate-v4.)
 * 2. Rows of one op that read the same weights register the SAME hooks, so no two of them can ask two stored forms of
 *    one weight (a weight is stored once): checked as equal function pointers for gate_up's rows and for down's, and
 *    prep_gu's tables are gate_up's, checked by asking both hooks the same question in tests/gpu/layout_test.c.
 * Exit 0 on pass, 1 on a failure, 77 (skipped) when the plugin cannot be opened. */
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#include "exl3r4d.h"

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: rows_static_test <exl3r4d.so>\n"); return 2; }
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { printf("SKIPPED: cannot open %s: %s\n", argv[1], dlerror()); return 77; }
    int (*count)(void) = (int (*)(void))dlsym(h, "rad_kernel_count");
    const RadKernelInfo* (*at)(int) = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    if (!count || !at) { printf("FAIL the plugin exports no row table\n"); return 1; }
    int pass = 0, fail = 0;
    const RadKernelInfo *gu = NULL, *dn = NULL;
    for (int i = 0; i < count(); ++i) {
        const RadKernelInfo* r = at(i);
        if (!r || !r->name || !r->op) { printf("FAIL row %d is null or unnamed\n", i); ++fail; continue; }
        if (r->layout && (!r->relayout || !r->unrelayout)) {
            printf("FAIL %s registers a layout hook without %s%s%s\n", r->name, r->relayout ? "" : "relayout",
                   !r->relayout && !r->unrelayout ? " and " : "", r->unrelayout ? "" : "unrelayout");
            ++fail;
        } else {
            printf("PASS %s: layout%s\n", r->name, r->layout ? ", relayout and unrelayout all registered" : " none");
            ++pass;
        }
        const RadKernelInfo** first = strcmp(r->op, EXL3_OP_GATE_UP) == 0 ? &gu
                                    : strcmp(r->op, EXL3_OP_DOWN) == 0    ? &dn : NULL;
        if (!first) continue;
        if (!*first) { *first = r; continue; }
        if (r->layout != (*first)->layout || r->relayout != (*first)->relayout ||
            r->unrelayout != (*first)->unrelayout) {
            printf("FAIL %s and %s (one op, the same weights) register different layout hooks\n", r->name,
                   (*first)->name);
            ++fail;
        } else {
            printf("PASS %s: the same layout hooks as %s\n", r->name, (*first)->name);
            ++pass;
        }
    }
    printf("rows_static: %d passed, %d failed\n", pass, fail);
    return fail ? 1 : 0;
}
