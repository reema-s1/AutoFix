#include "minitest.h"
#include "ring.h"

TEST(create_rejects_zero_capacity)
{
    CHECK(ring_create(0) == NULL);
}

TEST(push_then_pop_is_fifo)
{
    ring_t *r = ring_create(4);
    int v = 0;

    CHECK(r != NULL);
    CHECK_EQ_INT(0, ring_push(r, 1));
    CHECK_EQ_INT(0, ring_push(r, 2));
    CHECK_EQ_INT(0, ring_push(r, 3));
    CHECK_EQ_INT(0, ring_pop(r, &v));
    CHECK_EQ_INT(1, v);
    CHECK_EQ_INT(0, ring_pop(r, &v));
    CHECK_EQ_INT(2, v);
    CHECK_EQ_INT(1, ring_size(r));
    ring_destroy(r);
}

TEST(push_fails_when_full)
{
    ring_t *r = ring_create(3);
    int v = 0;

    CHECK_EQ_INT(0, ring_push(r, 10));
    CHECK_EQ_INT(0, ring_push(r, 20));
    CHECK_EQ_INT(0, ring_push(r, 30));
    CHECK(ring_is_full(r));
    CHECK_EQ_INT(-1, ring_push(r, 40));
    CHECK_EQ_INT(3, ring_size(r));
    CHECK_EQ_INT(0, ring_pop(r, &v));
    CHECK_EQ_INT(10, v);
    ring_destroy(r);
}

TEST(pop_and_peek_fail_when_empty)
{
    ring_t *r = ring_create(2);
    int v = 99;

    CHECK_EQ_INT(-1, ring_pop(r, &v));
    CHECK_EQ_INT(-1, ring_peek(r, &v));
    CHECK_EQ_INT(99, v);
    ring_destroy(r);
}

TEST(peek_returns_oldest_without_removing)
{
    ring_t *r = ring_create(3);
    int v = 0;

    ring_push(r, 7);
    ring_push(r, 8);
    CHECK_EQ_INT(0, ring_peek(r, &v));
    CHECK_EQ_INT(7, v);
    CHECK_EQ_INT(2, ring_size(r));
    ring_destroy(r);
}

TEST(wraps_around_many_times)
{
    ring_t *r = ring_create(3);
    int i;
    int v = 0;

    for (i = 0; i < 100; i++) {
        CHECK_EQ_INT(0, ring_push(r, i));
        CHECK_EQ_INT(0, ring_peek(r, &v));
        CHECK_EQ_INT(i, v);
        CHECK_EQ_INT(0, ring_pop(r, &v));
        CHECK_EQ_INT(i, v);
    }
    CHECK_EQ_INT(0, ring_size(r));
    ring_destroy(r);
}

TEST(snapshot_is_oldest_first_after_wrap)
{
    ring_t *r = ring_create(4);
    int out[4] = {0};
    int v;

    ring_push(r, 1);
    ring_push(r, 2);
    ring_push(r, 3);
    ring_pop(r, &v);
    ring_pop(r, &v);
    ring_push(r, 4);
    ring_push(r, 5);
    ring_push(r, 6);
    CHECK_EQ_INT(4, ring_snapshot(r, out, 4));
    CHECK_EQ_INT(3, out[0]);
    CHECK_EQ_INT(4, out[1]);
    CHECK_EQ_INT(5, out[2]);
    CHECK_EQ_INT(6, out[3]);
    CHECK_EQ_INT(2, ring_snapshot(r, out, 2));
    ring_destroy(r);
}

MT_MAIN_BEGIN
    RUN_TEST(create_rejects_zero_capacity);
    RUN_TEST(push_then_pop_is_fifo);
    RUN_TEST(push_fails_when_full);
    RUN_TEST(pop_and_peek_fail_when_empty);
    RUN_TEST(peek_returns_oldest_without_removing);
    RUN_TEST(wraps_around_many_times);
    RUN_TEST(snapshot_is_oldest_first_after_wrap);
MT_MAIN_END
