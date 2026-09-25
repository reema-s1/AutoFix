#include "minitest.h"
#include "window.h"

TEST(window_contains_simple_range)
{
    CHECK(seq_in_window(10, 10, 4));
    CHECK(seq_in_window(10, 13, 4));
    CHECK(!seq_in_window(10, 14, 4));
    CHECK(!seq_in_window(10, 9, 4));
}

TEST(window_wraps_past_255)
{
    CHECK(seq_in_window(250, 250, 8));
    CHECK(seq_in_window(250, 255, 8));
    CHECK(seq_in_window(250, 0, 8));
    CHECK(seq_in_window(250, 1, 8));
    CHECK(!seq_in_window(250, 2, 8));
    CHECK(!seq_in_window(250, 249, 8));
}

TEST(checksum_matches_rfc1071_example)
{
    const uint8_t data[] = {0x00, 0x01, 0xf2, 0x03, 0xf4, 0xf5, 0xf6, 0xf7};
    CHECK_EQ_INT(0x220d, checksum16(data, sizeof data));
}

TEST(checksum_pads_odd_trailing_byte)
{
    const uint8_t data[] = {0x01, 0x02, 0x03};
    CHECK_EQ_INT(0xfbfd, checksum16(data, sizeof data));
}

TEST(checksum_verifies_to_zero)
{
    uint8_t frame[6] = {0xde, 0xad, 0xbe, 0xef, 0, 0};
    uint16_t c = checksum16(frame, 4);

    frame[4] = (uint8_t)(c >> 8);
    frame[5] = (uint8_t)(c & 0xff);
    CHECK_EQ_INT(0, checksum16(frame, sizeof frame));
}

TEST(sender_limits_frames_to_window)
{
    sender_t s;
    int sent = 0;

    sender_init(&s, 4);
    while (sender_can_send(&s)) {
        sender_send(&s);
        sent++;
    }
    CHECK_EQ_INT(4, sent);
    CHECK_EQ_INT(4, sender_in_flight(&s));
}

TEST(cumulative_ack_releases_frames)
{
    sender_t s;

    sender_init(&s, 4);
    sender_send(&s);
    sender_send(&s);
    sender_send(&s);
    CHECK_EQ_INT(2, sender_on_ack(&s, 2));
    CHECK_EQ_INT(1, sender_in_flight(&s));
    CHECK_EQ_INT(1, sender_on_ack(&s, 3));
    CHECK_EQ_INT(0, sender_in_flight(&s));
    CHECK(sender_can_send(&s));
}

TEST(duplicate_and_bogus_acks_are_ignored)
{
    sender_t s;

    sender_init(&s, 4);
    sender_send(&s);
    sender_send(&s);
    CHECK_EQ_INT(1, sender_on_ack(&s, 1));
    CHECK_EQ_INT(0, sender_on_ack(&s, 1));
    CHECK_EQ_INT(0, sender_on_ack(&s, 9));
    CHECK_EQ_INT(1, sender_in_flight(&s));
}

TEST(sender_acks_across_wraparound)
{
    sender_t s;
    int i;

    sender_init(&s, 8);
    for (i = 0; i < 300; i++) {
        seq_t seq = sender_send(&s);
        CHECK_EQ_INT(1, sender_on_ack(&s, (seq_t)(seq + 1)));
    }
    CHECK_EQ_INT(300 % 256, s.base);
}

TEST(receiver_delivers_in_order)
{
    receiver_t r;
    int out[8];

    receiver_init(&r, 4);
    CHECK_EQ_INT(1, receiver_on_frame(&r, 0, 100, out, 8));
    CHECK_EQ_INT(100, out[0]);
    CHECK_EQ_INT(1, receiver_on_frame(&r, 1, 101, out, 8));
    CHECK_EQ_INT(101, out[0]);
}

TEST(receiver_buffers_out_of_order_frames)
{
    receiver_t r;
    int out[8];

    receiver_init(&r, 4);
    CHECK_EQ_INT(0, receiver_on_frame(&r, 2, 202, out, 8));
    CHECK_EQ_INT(0, receiver_on_frame(&r, 1, 201, out, 8));
    CHECK_EQ_INT(3, receiver_on_frame(&r, 0, 200, out, 8));
    CHECK_EQ_INT(200, out[0]);
    CHECK_EQ_INT(201, out[1]);
    CHECK_EQ_INT(202, out[2]);
    CHECK_EQ_INT(3, r.expected);
}

TEST(receiver_drops_frames_outside_window)
{
    receiver_t r;
    int out[8];

    receiver_init(&r, 4);
    CHECK_EQ_INT(-1, receiver_on_frame(&r, 4, 1, out, 8));
    CHECK_EQ_INT(-1, receiver_on_frame(&r, 200, 1, out, 8));
    CHECK_EQ_INT(1, receiver_on_frame(&r, 0, 7, out, 8));
}

MT_MAIN_BEGIN
    RUN_TEST(window_contains_simple_range);
    RUN_TEST(window_wraps_past_255);
    RUN_TEST(checksum_matches_rfc1071_example);
    RUN_TEST(checksum_pads_odd_trailing_byte);
    RUN_TEST(checksum_verifies_to_zero);
    RUN_TEST(sender_limits_frames_to_window);
    RUN_TEST(cumulative_ack_releases_frames);
    RUN_TEST(duplicate_and_bogus_acks_are_ignored);
    RUN_TEST(sender_acks_across_wraparound);
    RUN_TEST(receiver_delivers_in_order);
    RUN_TEST(receiver_buffers_out_of_order_frames);
    RUN_TEST(receiver_drops_frames_outside_window);
MT_MAIN_END
