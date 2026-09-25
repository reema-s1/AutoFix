#include "strkit.h"

#include <ctype.h>
#include <limits.h>
#include <string.h>

size_t str_trim(char *s)
{
    size_t start = 0;
    size_t end = strlen(s);

    while (s[start] != '\0' && isspace((unsigned char)s[start])) {
        start++;
    }
    while (end > start && isspace((unsigned char)s[end - 1])) {
        end--;
    }
    memmove(s, s + start, end - start);
    s[end - start] = '\0';
    return end - start;
}

int str_to_int(const char *s, int *out)
{
    unsigned long long limit;
    unsigned long long value = 0;
    int negative = 0;

    if (s == NULL || *s == '\0') {
        return -1;
    }
    if (*s == '+' || *s == '-') {
        negative = (*s == '-');
        s++;
    }
    if (!isdigit((unsigned char)*s)) {
        return -1;
    }
    limit = negative ? (unsigned long long)INT_MAX + 1ull : (unsigned long long)INT_MAX;
    while (isdigit((unsigned char)*s)) {
        unsigned long long digit = (unsigned long long)(*s - '0');
        if (value > (limit - digit) / 10) {
            return -1;
        }
        value = value * 10 + digit;
        s++;
    }
    if (*s != '\0') {
        return -1;
    }
    *out = negative ? (int)(-(long long)value) : (int)value;
    return 0;
}

size_t str_split(const char *s, char delim, char out[][STR_TOKEN_MAX], size_t max)
{
    size_t count = 0;
    const char *start = s;
    const char *p;

    for (p = s;; p++) {
        if (*p == delim || *p == '\0') {
            if (count < max) {
                size_t len = (size_t)(p - start);
                if (len >= STR_TOKEN_MAX) {
                    len = STR_TOKEN_MAX - 1;
                }
                memcpy(out[count], start, len);
                out[count][len] = '\0';
            }
            count++;
            if (*p == '\0') {
                break;
            }
            start = p + 1;
        }
    }
    return count;
}

static void append(char *dst, size_t dst_size, size_t *len, const char *src, size_t n)
{
    size_t i;

    for (i = 0; i < n; i++, (*len)++) {
        if (*len + 1 < dst_size) {
            dst[*len] = src[i];
        }
    }
}

size_t str_join(char *dst, size_t dst_size, const char *const *parts, size_t n, const char *sep)
{
    size_t len = 0;
    size_t sep_len = strlen(sep);
    size_t i;

    for (i = 0; i < n; i++) {
        if (i > 0) {
            append(dst, dst_size, &len, sep, sep_len);
        }
        append(dst, dst_size, &len, parts[i], strlen(parts[i]));
    }
    if (dst_size > 0) {
        dst[len < dst_size ? len : dst_size - 1] = '\0';
    }
    return len;
}

size_t str_count(const char *s, const char *needle)
{
    size_t count = 0;
    size_t needle_len = strlen(needle);
    const char *p = s;

    if (needle_len == 0) {
        return 0;
    }
    while ((p = strstr(p, needle)) != NULL) {
        count++;
        p += needle_len;
    }
    return count;
}
