#include "window.h"

#include <string.h>

int seq_in_window(seq_t base, seq_t seq, unsigned size)
{
    return (seq_t)(seq - base) < size;
}

uint16_t checksum16(const uint8_t *data, size_t len)
{
    uint32_t sum = 0;
    size_t i;

    for (i = 0; i + 1 < len; i += 2) {
        sum += (uint32_t)((data[i] << 8) | data[i + 1]);
    }
    if (len & 1u) {
        sum += (uint32_t)(data[len - 1] << 8);
    }
    while (sum >> 16) {
        sum = (sum & 0xFFFFu) + (sum >> 16);
    }
    return (uint16_t)~sum;
}

void sender_init(sender_t *s, unsigned size)
{
    s->base = 0;
    s->next = 0;
    s->size = size;
}

unsigned sender_in_flight(const sender_t *s)
{
    return (seq_t)(s->next - s->base);
}

int sender_can_send(const sender_t *s)
{
    return sender_in_flight(s) < s->size;
}

seq_t sender_send(sender_t *s)
{
    return s->next++;
}

int sender_on_ack(sender_t *s, seq_t ack)
{
    unsigned in_flight = sender_in_flight(s);
    unsigned advance = (seq_t)(ack - s->base);

    if (advance == 0 || advance > in_flight) {
        return 0;
    }
    s->base = ack;
    return (int)advance;
}

void receiver_init(receiver_t *r, unsigned size)
{
    memset(r, 0, sizeof *r);
    r->size = size;
}

static slot_t *slot_for(receiver_t *r, seq_t seq)
{
    if (!seq_in_window(r->expected, seq, r->size)) {
        return NULL;
    }
    return &r->slots[seq];
}

int receiver_on_frame(receiver_t *r, seq_t seq, int payload, int *out, size_t max)
{
    slot_t *slot = slot_for(r, seq);
    int delivered = 0;

    if (slot == NULL) {
        return -1;
    }
    slot->present = 1;
    slot->payload = payload;

    while (r->slots[r->expected].present && (size_t)delivered < max) {
        out[delivered++] = r->slots[r->expected].payload;
        r->slots[r->expected].present = 0;
        r->expected++;
    }
    return delivered;
}
