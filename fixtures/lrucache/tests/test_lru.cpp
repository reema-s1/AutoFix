#include <string>
#include <vector>

#include "lru.hpp"
#include "minitest.h"

static bool keys_are(const lru::Cache &cache, const std::vector<int> &expected)
{
    return cache.keys() == expected;
}

TEST(put_and_get)
{
    lru::Cache cache(2);
    std::string v;

    cache.put(1, "one");
    CHECK(cache.get(1, v));
    CHECK_EQ_STR("one", v.c_str());
    CHECK(!cache.get(2, v));
}

TEST(update_overwrites_value)
{
    lru::Cache cache(2);
    std::string v;

    cache.put(1, "one");
    cache.put(1, "uno");
    CHECK_EQ_INT(1, cache.size());
    CHECK(cache.get(1, v));
    CHECK_EQ_STR("uno", v.c_str());
}

TEST(evicts_least_recently_used)
{
    lru::Cache cache(2);

    cache.put(1, "one");
    cache.put(2, "two");
    cache.put(3, "three");
    CHECK(!cache.contains(1));
    CHECK(cache.contains(2));
    CHECK(cache.contains(3));
    CHECK_EQ_INT(1, cache.evictions());
    CHECK_EQ_INT(2, cache.size());
}

TEST(get_refreshes_recency)
{
    lru::Cache cache(2);
    std::string v;

    cache.put(1, "one");
    cache.put(2, "two");
    CHECK(cache.get(1, v));
    cache.put(3, "three");
    CHECK(cache.contains(1));
    CHECK(!cache.contains(2));
    CHECK(keys_are(cache, {3, 1}));
}

TEST(put_existing_refreshes_recency)
{
    lru::Cache cache(2);

    cache.put(1, "one");
    cache.put(2, "two");
    cache.put(1, "uno");
    cache.put(3, "three");
    CHECK(keys_are(cache, {3, 1}));
}

TEST(contains_does_not_refresh)
{
    lru::Cache cache(2);

    cache.put(1, "one");
    cache.put(2, "two");
    CHECK(cache.contains(1));
    cache.put(3, "three");
    CHECK(!cache.contains(1));
}

TEST(many_evictions_keep_index_consistent)
{
    lru::Cache cache(3);
    std::string v;
    int i;

    for (i = 0; i < 50; i++) {
        cache.put(i, std::to_string(i));
    }
    CHECK_EQ_INT(47, cache.evictions());
    CHECK(keys_are(cache, {49, 48, 47}));
    for (i = 0; i < 47; i++) {
        CHECK(!cache.contains(i));
    }
    CHECK(cache.get(47, v));
    CHECK_EQ_STR("47", v.c_str());
}

TEST(zero_capacity_stores_nothing)
{
    lru::Cache cache(0);
    std::string v;

    cache.put(1, "one");
    CHECK_EQ_INT(0, cache.size());
    CHECK(!cache.get(1, v));
}

MT_MAIN_BEGIN
    RUN_TEST(put_and_get);
    RUN_TEST(update_overwrites_value);
    RUN_TEST(evicts_least_recently_used);
    RUN_TEST(get_refreshes_recency);
    RUN_TEST(put_existing_refreshes_recency);
    RUN_TEST(contains_does_not_refresh);
    RUN_TEST(many_evictions_keep_index_consistent);
    RUN_TEST(zero_capacity_stores_nothing);
MT_MAIN_END
