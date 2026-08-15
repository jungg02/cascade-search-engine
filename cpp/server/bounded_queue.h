// A blocking, bounded FIFO queue. try_push never blocks: it fails immediately
// when the queue is already at capacity, which is what lets the server's RPC
// handler shed load instead of queueing without bound.
#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>

namespace cascade {

template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(size_t capacity) : capacity_(capacity) {}

  // Returns false without blocking or modifying the queue if it is already
  // at capacity.
  bool try_push(T item) {
    std::lock_guard<std::mutex> lock(mu_);
    if (items_.size() >= capacity_) return false;
    items_.push_back(std::move(item));
    not_empty_.notify_one();
    return true;
  }

  // Blocks until an item is available or shutdown() is called, in which case
  // it returns std::nullopt.
  std::optional<T> pop() {
    std::unique_lock<std::mutex> lock(mu_);
    not_empty_.wait(lock, [this] { return !items_.empty() || shutting_down_; });
    if (items_.empty()) return std::nullopt;  // woken by shutdown()
    T item = std::move(items_.front());
    items_.pop_front();
    return item;
  }

  void shutdown() {
    std::lock_guard<std::mutex> lock(mu_);
    shutting_down_ = true;
    not_empty_.notify_all();
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return items_.size();
  }

 private:
  mutable std::mutex mu_;
  std::condition_variable not_empty_;
  std::deque<T> items_;
  size_t capacity_;
  bool shutting_down_ = false;
};

}  // namespace cascade
