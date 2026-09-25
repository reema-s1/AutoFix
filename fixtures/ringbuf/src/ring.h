#ifndef RING_H
#define RING_H

#include <stddef.h>

/* A fixed-capacity FIFO queue of ints backed by a circular buffer. */
typedef struct {
    int *buf;
    size_t cap;
    size_t head;  /* next slot to write */
    size_t tail;  /* next slot to read */
    size_t count;
} ring_t;

/* Returns NULL if cap is 0 or allocation fails. */
ring_t *ring_create(size_t cap);
void ring_destroy(ring_t *r);

/* Returns 0 on success, -1 if the ring is full. */
int ring_push(ring_t *r, int value);

/* Removes the oldest value into *out. Returns 0 on success, -1 if empty. */
int ring_pop(ring_t *r, int *out);

/* Reads the oldest value without removing it. Returns 0 on success, -1 if empty. */
int ring_peek(const ring_t *r, int *out);

size_t ring_size(const ring_t *r);
int ring_is_full(const ring_t *r);

/* Copies up to max values, oldest first, into out. Returns the number copied. */
size_t ring_snapshot(const ring_t *r, int *out, size_t max);

#endif /* RING_H */
