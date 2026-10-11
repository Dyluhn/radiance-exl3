/* exl3_check.h -- the smallest test harness that can fail: CHECK counts, TEST names, main returns the
 * count. A test file whose checks never ran says so (exit 77 = SKIPPED to ctest), never "passed". */
#ifndef EXL3_CHECK_H
#define EXL3_CHECK_H

#include <stdio.h>

static int exl3_check_failed = 0, exl3_check_ran = 0;

#define CHECK(cond)                                                                   \
    do {                                                                              \
        ++exl3_check_ran;                                                             \
        if (!(cond)) {                                                                \
            ++exl3_check_failed;                                                      \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);  \
        }                                                                             \
    } while (0)

#define CHECK_EQ_U(a, b)                                                              \
    do {                                                                              \
        const unsigned long long a_ = (unsigned long long)(a), b_ = (unsigned long long)(b); \
        ++exl3_check_ran;                                                             \
        if (a_ != b_) {                                                               \
            ++exl3_check_failed;                                                      \
            fprintf(stderr, "%s:%d: CHECK_EQ failed: %s = 0x%llx, %s = 0x%llx\n",     \
                    __FILE__, __LINE__, #a, a_, #b, b_);                              \
        }                                                                             \
    } while (0)

#define TEST(name) static void name(void)
#define RUN(name)                                                       \
    do {                                                                \
        const int before_ = exl3_check_failed;                          \
        name();                                                         \
        printf("%s %s\n", exl3_check_failed == before_ ? "ok  " : "FAIL", #name); \
    } while (0)

static int exl3_check_done(void) {
    if (exl3_check_ran == 0) {
        printf("SKIPPED: no check ran\n");
        return 77;
    }
    printf("%d check(s), %d failed\n", exl3_check_ran, exl3_check_failed);
    return exl3_check_failed ? 1 : 0;
}

/* splitmix64: a seeded stream every test can name, the same on every machine */
static unsigned long long exl3_rng_state = 1;
static unsigned long long exl3_rng(void) {
    unsigned long long z = (exl3_rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

#endif
