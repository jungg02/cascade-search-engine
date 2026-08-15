// A fixed-capacity, string-keyed, thread-safe LRU cache: O(1) get/put,
// evicts the least recently *used* (not inserted) entry when full.
#pragma once

#include <list>
#include <mutex>
#include <string>
#include <unordered_map>

namespace cascade {

template <typename Value>
class LruCache {
 public:
  explicit LruCache(size_t capacity) : capacity_(capacity) {}

  // Returns true and copies the cached value into *out on a hit, moving the
  // entry to the front (most recently used). Returns false on a miss.
  bool get(const std::string& key, Value* out) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = index_.find(key);
    if (it == index_.end()) return false;
    entries_.splice(entries_.begin(), entries_, it->second);
    *out = it->second->value;
    return true;
  }

  // Inserts or overwrites key, moving it to the front. Evicts the
  // least-recently-used entry first if capacity_ is exceeded.
  void put(const std::string& key, Value value) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = index_.find(key);
    if (it != index_.end()) {
      it->second->value = std::move(value);
      entries_.splice(entries_.begin(), entries_, it->second);
      return;
    }
    entries_.push_front(Entry{key, std::move(value)});
    index_[key] = entries_.begin();
    if (index_.size() > capacity_) {
      index_.erase(entries_.back().key);
      entries_.pop_back();
    }
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return index_.size();
  }

 private:
  struct Entry {
    std::string key;
    Value value;
  };

  mutable std::mutex mu_;
  size_t capacity_;
  std::list<Entry> entries_;  // front = most recently used
  std::unordered_map<std::string, typename std::list<Entry>::iterator> index_;
};

}  // namespace cascade
