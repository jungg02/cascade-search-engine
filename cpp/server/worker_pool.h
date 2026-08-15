// A fixed-size pool of worker threads pulling closures off a BoundedQueue.
// try_submit is the load-shedding boundary: it returns false immediately,
// without blocking, when the queue is already at capacity.
#pragma once

#include <functional>
#include <thread>
#include <vector>

#include "bounded_queue.h"

namespace cascade {

class WorkerPool {
 public:
  WorkerPool(size_t num_workers, size_t queue_depth) : queue_(queue_depth) {
    workers_.reserve(num_workers);
    for (size_t i = 0; i < num_workers; i++) {
      workers_.emplace_back([this] {
        while (true) {
          auto task = queue_.pop();
          if (!task.has_value()) return;  // shutdown
          (*task)();
        }
      });
    }
  }

  ~WorkerPool() { shutdown(); }

  WorkerPool(const WorkerPool&) = delete;
  WorkerPool& operator=(const WorkerPool&) = delete;

  // Returns false without blocking if the queue is already at capacity.
  bool try_submit(std::function<void()> task) {
    return queue_.try_push(std::move(task));
  }

  size_t queue_size() const { return queue_.size(); }

  void shutdown() {
    queue_.shutdown();
    for (auto& t : workers_) {
      if (t.joinable()) t.join();
    }
  }

 private:
  BoundedQueue<std::function<void()>> queue_;
  std::vector<std::thread> workers_;
};

}  // namespace cascade
