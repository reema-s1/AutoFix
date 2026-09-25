/*
 * minitest.h - a single-header test harness for C and C++.
 *
 * Emits the line protocol parsed by AutoFix:
 *
 *   RUN  <name>
 *   PASS <name>
 *   FAIL <name> (<file>:<line>): <message>
 *   SUMMARY passed=<n> failed=<n>
 *
 * Output is flushed after every line so that a crash is attributed to the
 * test that was running. An optional argv[1] runs only tests whose name
 * contains that substring.
 */
#ifndef MINITEST_H
#define MINITEST_H

#include <stdio.h>
#include <string.h>

static int mt_passed_;
static int mt_failed_;
static int mt_current_failed_;
static const char *mt_current_;
static const char *mt_filter_;

/* Formats v in decimal without relying on %lld, which some C runtimes lack. */
static const char *mt_fmt_ll_(long long v, char *buf)
{
    char digits[24];
    int n = 0;
    int i = 0;
    unsigned long long u = v < 0 ? 0ull - (unsigned long long)v : (unsigned long long)v;

    do {
        digits[n++] = (char)('0' + (int)(u % 10u));
        u /= 10u;
    } while (u != 0u);
    if (v < 0) {
        buf[i++] = '-';
    }
    while (n > 0) {
        buf[i++] = digits[--n];
    }
    buf[i] = '\0';
    return buf;
}

#define MT_FAIL_(file, line, ...)                               \
    do {                                                        \
        printf("FAIL %s (%s:%d): ", mt_current_, file, line);   \
        printf(__VA_ARGS__);                                    \
        printf("\n");                                           \
        fflush(stdout);                                         \
        mt_current_failed_ = 1;                                 \
    } while (0)

#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            MT_FAIL_(__FILE__, __LINE__, "check failed: %s", #cond);    \
            return;                                                     \
        }                                                               \
    } while (0)

#define CHECK_EQ_INT(expected, actual)                                          \
    do {                                                                        \
        long long mt_e_ = (long long)(expected);                                \
        long long mt_a_ = (long long)(actual);                                  \
        if (mt_e_ != mt_a_) {                                                   \
            char mt_eb_[24];                                                    \
            char mt_ab_[24];                                                    \
            MT_FAIL_(__FILE__, __LINE__, "expected %s == %s, got %s", #actual,  \
                     mt_fmt_ll_(mt_e_, mt_eb_), mt_fmt_ll_(mt_a_, mt_ab_));     \
            return;                                                             \
        }                                                                       \
    } while (0)

#define CHECK_EQ_STR(expected, actual)                                          \
    do {                                                                        \
        const char *mt_e_ = (expected);                                         \
        const char *mt_a_ = (actual);                                           \
        if (mt_a_ == NULL || strcmp(mt_e_, mt_a_) != 0) {                       \
            MT_FAIL_(__FILE__, __LINE__, "expected %s == \"%s\", got \"%s\"",   \
                     #actual, mt_e_, mt_a_ ? mt_a_ : "(null)");                 \
            return;                                                             \
        }                                                                       \
    } while (0)

#define TEST(name) static void name(void)

static void mt_run_(const char *name, void (*fn)(void))
{
    if (mt_filter_ != NULL && strstr(name, mt_filter_) == NULL) {
        return;
    }
    mt_current_ = name;
    mt_current_failed_ = 0;
    printf("RUN  %s\n", name);
    fflush(stdout);
    fn();
    if (mt_current_failed_) {
        mt_failed_++;
    } else {
        printf("PASS %s\n", name);
        mt_passed_++;
    }
    fflush(stdout);
}

#define RUN_TEST(name) mt_run_(#name, name)

#define MT_MAIN_BEGIN                           \
    int main(int argc, char **argv)             \
    {                                           \
        mt_filter_ = argc > 1 ? argv[1] : NULL;

#define MT_MAIN_END                                                         \
        printf("SUMMARY passed=%d failed=%d\n", mt_passed_, mt_failed_);    \
        return mt_failed_ ? 1 : 0;                                          \
    }

#endif /* MINITEST_H */
