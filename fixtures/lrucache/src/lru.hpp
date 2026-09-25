#ifndef LRU_HPP
#define LRU_HPP

#include <cstddef>
#include <list>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lru {

// A least-recently-used cache from int keys to string values.
class Cache {
public:
    explicit Cache(std::size_t capacity);

    // Returns true and copies the value if key is present, marking it most recently used.
    bool get(int key, std::string &value);

    // Inserts or updates key and marks it most recently used. When the cache is
    // full, the least recently used entry is evicted first.
    void put(int key, const std::string &value);

    // Presence check that does not affect recency.
    bool contains(int key) const;

    std::size_t size() const;
    std::size_t capacity() const;
    std::size_t evictions() const;

    // Keys ordered from most to least recently used.
    std::vector<int> keys() const;

private:
    using Entry = std::pair<int, std::string>;

    std::size_t capacity_;
    std::size_t evictions_ = 0;
    std::list<Entry> order_;  // front is the most recently used entry
    std::unordered_map<int, std::list<Entry>::iterator> index_;
};

}  // namespace lru

#endif  // LRU_HPP
