#ifndef STRKIT_H
#define STRKIT_H

#include <stddef.h>

#define STR_TOKEN_MAX 32

/* Removes leading and trailing whitespace in place. Returns the new length. */
size_t str_trim(char *s);

/*
 * Parses a base-10 int with an optional sign. The whole string must be
 * consumed. Returns 0 and stores the value on success; returns -1 (and leaves
 * *out untouched) on empty input, trailing characters or overflow.
 */
int str_to_int(const char *s, int *out);

/*
 * Splits s on delim, keeping empty fields ("a,,b" has three fields). Writes
 * at most max fields into out, each truncated to STR_TOKEN_MAX - 1 characters.
 * Returns the total number of fields, which may exceed max.
 */
size_t str_split(const char *s, char delim, char out[][STR_TOKEN_MAX], size_t max);

/*
 * Joins parts with sep into dst, snprintf-style: the result is truncated to
 * fit and always NUL-terminated when dst_size > 0. Returns the length the
 * full result would have (excluding the terminator).
 */
size_t str_join(char *dst, size_t dst_size, const char *const *parts, size_t n, const char *sep);

/* Counts non-overlapping occurrences of needle in s. An empty needle counts 0. */
size_t str_count(const char *s, const char *needle);

#endif /* STRKIT_H */
