/* rowpick.c -- a TEST-ONLY kernel-plugin shim that exposes ONE row of another kernel library, so a harness that
 * takes the first row of an op (tests/ref/exl3ref_run.cpp: first row with the op and domain, constraints not
 * consulted) runs the row under test. Never installed.
 *
 *   EXL3_ROWPICK_LIB  the real library (.so) whose rows and schemas are forwarded
 *   EXL3_ROWPICK_ROW  substring of the row names to keep (every op's row that matches; e.g. "_pf")
 *
 * Everything the harness calls is forwarded unchanged: the ABI version, the schemas, and the kept rows themselves
 * (the same RadKernelInfo pointers, so the same hooks run). Exits the process (status 2) when either variable is
 * missing or the library cannot be opened -- a test that silently ran another row would check nothing. */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rad_abi.h"

static void* g_lib;
static const RadKernelInfo* g_rows[64];
static int g_n = -1;

static void* need(const char* sym) {
    void* f = dlsym(g_lib, sym);
    if (!f) { fprintf(stderr, "rowpick: %s has no %s\n", getenv("EXL3_ROWPICK_LIB"), sym); exit(2); }
    return f;
}

static void load(void) {
    if (g_lib) return;
    const char* lib = getenv("EXL3_ROWPICK_LIB");
    const char* row = getenv("EXL3_ROWPICK_ROW");
    if (!lib || !row) { fprintf(stderr, "rowpick: set EXL3_ROWPICK_LIB and EXL3_ROWPICK_ROW\n"); exit(2); }
    g_lib = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!g_lib) { fprintf(stderr, "rowpick: %s\n", dlerror()); exit(2); }
    int (*count)(void) = (int (*)(void))need("rad_kernel_count");
    const RadKernelInfo* (*at)(int) = (const RadKernelInfo* (*)(int))need("rad_kernel_at");
    g_n = 0;
    for (int i = 0; i < count() && g_n < 64; ++i)
        if (at(i) && at(i)->name && strstr(at(i)->name, row)) g_rows[g_n++] = at(i);
    if (g_n == 0) { fprintf(stderr, "rowpick: no row of %s matches '%s'\n", lib, row); exit(2); }
    fprintf(stderr, "rowpick: %d row(s) of %s matching '%s'\n", g_n, lib, row);
}

uint32_t rad_plugin_abi_version(void) { load(); return ((uint32_t (*)(void))need("rad_plugin_abi_version"))(); }
const RadPluginInfo* rad_plugin_info(void) { load(); return ((const RadPluginInfo* (*)(void))need("rad_plugin_info"))(); }
int rad_kernel_schema_count(void) { load(); return ((int (*)(void))need("rad_kernel_schema_count"))(); }
const RadOpSchema* rad_kernel_schema_at(int i) {
    load();
    return ((const RadOpSchema* (*)(int))need("rad_kernel_schema_at"))(i);
}
int rad_kernel_count(void) { load(); return g_n; }
const RadKernelInfo* rad_kernel_at(int i) { load(); return i >= 0 && i < g_n ? g_rows[i] : 0; }
