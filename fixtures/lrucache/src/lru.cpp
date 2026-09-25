#include "lru.hpp"

namespace lru {

Cache::Cache(std::size_t capacity) : capacity_(capacity) {}

bool Cache::get(int key, std::string &value)
{
    auto it = index_.find(key);
    if (it == index_.end()) {
        return false;
    }
    order_.splice(order_.begin(), order_, it->second);
    value = it->second->second;
    return true;
}

void Cache::put(int key, const std::string &value)
{
    if (capacity_ == 0) {
        return;
    }
    auto it = index_.find(key);
    if (it != index_.end()) {
        it->second->second = value;
        order_.splice(order_.begin(), order_, it->second);
        return;
    }
    if (order_.size() == capacity_) {
        index_.erase(order_.back().first);
        order_.pop_back();
        ++evictions_;
    }
    order_.emplace_front(key, value);
    index_[key] = order_.begin();
}

bool Cache::contains(int key) const
{
    return index_.count(key) != 0;
}

std::size_t Cache::size() const
{
    return order_.size();
}

std::size_t Cache::capacity() const
{
    return capacity_;
}

std::size_t Cache::evictions() const
{
    return evictions_;
}

std::vector<int> Cache::keys() const
{
    std::vector<int> out;
    out.reserve(order_.size());
    for (const auto &entry : order_) {
        out.push_back(entry.first);
    }
    return out;
}

}  // namespace lru
