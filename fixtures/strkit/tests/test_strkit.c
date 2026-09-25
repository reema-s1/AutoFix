#include <limits.h>

#include "minitest.h"
#include "strkit.h"

TEST(trim_removes_surrounding_whitespace)
{
    char s[] = "  \t hello world \n ";

    CHECK_EQ_INT(11, str_trim(s));
    CHECK_EQ_STR("hello world", s);
}

TEST(trim_handles_blank_and_empty)
{
    char blank[] = "   \t\n";
    char empty[] = "";

    CHECK_EQ_INT(0, str_trim(blank));
    CHECK_EQ_STR("", blank);
    CHECK_EQ_INT(0, str_trim(empty));
}

TEST(to_int_parses_signed_values)
{
    int v = 0;

    CHECK_EQ_INT(0, str_to_int("42", &v));
    CHECK_EQ_INT(42, v);
    CHECK_EQ_INT(0, str_to_int("-17", &v));
    CHECK_EQ_INT(-17, v);
    CHECK_EQ_INT(0, str_to_int("+0", &v));
    CHECK_EQ_INT(0, v);
}

TEST(to_int_accepts_int_limits)
{
    int v = 0;

    CHECK_EQ_INT(0, str_to_int("2147483647", &v));
    CHECK_EQ_INT(INT_MAX, v);
    CHECK_EQ_INT(0, str_to_int("-2147483648", &v));
    CHECK_EQ_INT(INT_MIN, v);
}

TEST(to_int_rejects_overflow_and_garbage)
{
    int v = 123;

    CHECK_EQ_INT(-1, str_to_int("2147483648", &v));
    CHECK_EQ_INT(-1, str_to_int("-2147483649", &v));
    CHECK_EQ_INT(-1, str_to_int("99999999999", &v));
    CHECK_EQ_INT(-1, str_to_int("12a", &v));
    CHECK_EQ_INT(-1, str_to_int("", &v));
    CHECK_EQ_INT(-1, str_to_int("-", &v));
    CHECK_EQ_INT(123, v);
}

TEST(split_keeps_empty_fields)
{
    char out[4][STR_TOKEN_MAX];

    CHECK_EQ_INT(4, str_split("a,,b,", ',', out, 4));
    CHECK_EQ_STR("a", out[0]);
    CHECK_EQ_STR("", out[1]);
    CHECK_EQ_STR("b", out[2]);
    CHECK_EQ_STR("", out[3]);
}

TEST(split_reports_total_beyond_max)
{
    char out[2][STR_TOKEN_MAX];

    CHECK_EQ_INT(3, str_split("x y z", ' ', out, 2));
    CHECK_EQ_STR("x", out[0]);
    CHECK_EQ_STR("y", out[1]);
}

TEST(join_fits)
{
    const char *parts[] = {"usr", "local", "bin"};
    char buf[32];

    CHECK_EQ_INT(13, str_join(buf, sizeof buf, parts, 3, "/"));
    CHECK_EQ_STR("usr/local/bin", buf);
}

TEST(join_truncates_without_overflow)
{
    const char *parts[] = {"alpha", "beta"};
    struct {
        char buf[6];
        char guard;
    } mem;

    memset(&mem, 'X', sizeof mem);
    CHECK_EQ_INT(11, str_join(mem.buf, sizeof mem.buf, parts, 2, ", "));
    CHECK_EQ_STR("alpha", mem.buf);
    CHECK_EQ_INT('X', mem.guard);
}

TEST(count_is_non_overlapping)
{
    CHECK_EQ_INT(2, str_count("aaaa", "aa"));
    CHECK_EQ_INT(3, str_count("abcabcabc", "abc"));
    CHECK_EQ_INT(0, str_count("abc", "zz"));
    CHECK_EQ_INT(0, str_count("abc", ""));
}

MT_MAIN_BEGIN
    RUN_TEST(trim_removes_surrounding_whitespace);
    RUN_TEST(trim_handles_blank_and_empty);
    RUN_TEST(to_int_parses_signed_values);
    RUN_TEST(to_int_accepts_int_limits);
    RUN_TEST(to_int_rejects_overflow_and_garbage);
    RUN_TEST(split_keeps_empty_fields);
    RUN_TEST(split_reports_total_beyond_max);
    RUN_TEST(join_fits);
    RUN_TEST(join_truncates_without_overflow);
    RUN_TEST(count_is_non_overlapping);
MT_MAIN_END
