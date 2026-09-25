#ifndef WINDOW_H
#define WINDOW_H

#include <stddef.h>
#include <stdint.h>

/* Sliding-window ARQ primitives with 8-bit wrapping sequence numbers. */

#define SEQ_MOD 256u

typedef uint8_t seq_t;

/* 1 if seq lies in the window [base, base + size) modulo SEQ_MOD. */
int seq_in_window(seq_t base, seq_t seq, unsigned size);

/* 16-bit ones' complement checksum (RFC 1071); an odd trailing byte is padded with zero. */
uint16_t checksum16(const uint8_t *data, size_t len);

/* ---- sender ------------------------------------------------------------ */

typedef struct {
    seq_t base;     /* oldest unacknowledged sequence number */
    seq_t next;     /* next sequence number to assign */
    unsigned size;  /* window size, < SEQ_MOD / 2 */
} sender_t;

void sender_init(sender_t *s, unsigned size);
int sender_can_send(const sender_t *s);

/* Assigns and returns the next sequence number. Callers check sender_can_send first. */
seq_t sender_send(sender_t *s);

/*
 * Cumulative acknowledgement: ack is the next sequence number the receiver
 * expects, so every frame before it has arrived. Returns the number of frames
 * released from the window; duplicate or out-of-range acks release nothing.
 */
int sender_on_ack(sender_t *s, seq_t ack);

/* Number of frames sent but not yet acknowledged. */
unsigned sender_in_flight(const sender_t *s);

/* ---- receiver (selective repeat) --------------------------------------- */

typedef struct {
    int present;
    int payload;
} slot_t;

typedef struct {
    seq_t expected;  /* next in-order sequence number to deliver */
    unsigned size;
    slot_t slots[SEQ_MOD];
} receiver_t;

void receiver_init(receiver_t *r, unsigned size);

/*
 * Accepts a frame. Buffers it if it is inside the receive window, then
 * delivers every in-order payload into out (up to max). Returns the number
 * delivered, or -1 if seq is outside the window (the frame is dropped).
 */
int receiver_on_frame(receiver_t *r, seq_t seq, int payload, int *out, size_t max);

#endif /* WINDOW_H */
