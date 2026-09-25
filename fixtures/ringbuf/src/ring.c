#include "ring.h"

#include <stdlib.h>

ring_t *ring_create(size_t cap)
{
    ring_t *r;

    if (cap == 0) {
        return NULL;
    }
    r = malloc(sizeof *r);
    if (r == NULL) {
        return NULL;
    }
    r->buf = malloc(cap * sizeof *r->buf);
    if (r->buf == NULL) {
        free(r);
        return NULL;
    }
    r->cap = cap;
    r->head = 0;
    r->tail = 0;
    r->count = 0;
    return r;
}

void ring_destroy(ring_t *r)
{
    if (r != NULL) {
        free(r->buf);
        free(r);
    }
}

int ring_push(ring_t *r, int value)
{
    if (r->count == r->cap) {
        return -1;
    }
    r->buf[r->head] = value;
    r->head = (r->head + 1) % r->cap;
    r->count++;
    return 0;
}

int ring_pop(ring_t *r, int *out)
{
    if (r->count == 0) {
        return -1;
    }
    *out = r->buf[r->tail];
    r->tail = (r->tail + 1) % r->cap;
    r->count--;
    return 0;
}

int ring_peek(const ring_t *r, int *out)
{
    if (r->count == 0) {
        return -1;
    }
    *out = r->buf[r->tail];
    return 0;
}

size_t ring_size(const ring_t *r)
{
    return r->count;
}

int ring_is_full(const ring_t *r)
{
    return r->count == r->cap;
}

size_t ring_snapshot(const ring_t *r, int *out, size_t max)
{
    size_t i;
    size_t n = r->count < max ? r->count : max;

    for (i = 0; i < n; i++) {
        out[i] = r->buf[(r->tail + i) % r->cap];
    }
    return n;
}
