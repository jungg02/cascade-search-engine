// Correctness tests for BoundedQueue: FIFO order, capacity rejection without
// blocking, safe concurrent use, and shutdown waking blocked pop() calls.

#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "bounded_queue.h"

using namespace cascade;

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
  if (condition) {
    std::cout << "  ok   " << what << "\n";
  } else {
    std::cout << "  FAIL " << what << "\n";
    failures++;
  }
  std::cout.flush();
}

void test_fifo_order() {
  std::cout << "fifo order\n";
  BoundedQueue<int> q(10);
  for (int i = 0; i < 5; i++) check(q.try_push(i), "push " + std::to_string(i));
  bool in_order = true;
  for (int i = 0; i < 5; i++) {
    auto item = q.pop();
    if (!item.has_value() || *item != i) in_order = false;
  }
  check(in_order, "pop returns items in FIFO order");
}

void test_rejects_when_full() {
  std::cout << "capacity\n";
  BoundedQueue<int> q(2);
  check(q.try_push(1), "push into empty slot 1");
  check(q.try_push(2), "push into empty slot 2");
  check(!q.try_push(3), "push rejected when at capacity");
  check(q.size() == 2, "size stays at capacity after rejected push");
  auto item = q.pop();
  check(item.has_value() && *item == 1, "first item still 1 after rejected push");
}

void test_concurrent_producers_consumers() {
  std::cout << "concurrent stress\n";
  constexpr int kProducers = 4;
  constexpr int kItemsPerProducer = 2000;
  constexpr int kConsumers = 4;
  constexpr int kTotal = kProducers * kItemsPerProducer;
  BoundedQueue<int> q(64);

  std::atomic<long long> sum_pushed{0};
  std::atomic<long long> sum_popped{0};
  std::atomic<int> popped_count{0};

  std::vector<std::thread> producers;
  for (int p = 0; p < kProducers; p++) {
    producers.emplace_back([&, p] {
      for (int i = 0; i < kItemsPerProducer; i++) {
        int value = p * kItemsPerProducer + i;
        sum_pushed += value;
        while (!q.try_push(value)) {
          std::this_thread::yield();
        }
      }
    });
  }

  std::vector<std::thread> consumers;
  for (int c = 0; c < kConsumers; c++) {
    consumers.emplace_back([&] {
      while (popped_count.load() < kTotal) {
        auto item = q.pop();
        if (!item.has_value()) return;  // shutdown
        sum_popped += *item;
        popped_count++;
      }
    });
  }

  for (auto& t : producers) t.join();
  while (popped_count.load() < kTotal) std::this_thread::yield();
  q.shutdown();
  for (auto& t : consumers) t.join();

  check(popped_count.load() == kTotal, "every pushed item was popped exactly once (count)");
  check(sum_pushed.load() == sum_popped.load(),
        "every pushed item was popped exactly once (sum, catches duplicates/substitutions)");
}

void test_shutdown_wakes_blocked_pop() {
  std::cout << "shutdown\n";
  BoundedQueue<int> q(4);
  std::atomic<bool> returned{false};
  std::thread t([&] {
    auto item = q.pop();  // blocks: queue is empty
    returned = !item.has_value();
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  check(!returned.load(), "pop() is still blocked before shutdown");
  q.shutdown();
  t.join();
  check(returned.load(), "shutdown() wakes pop() with nullopt");
}

}  // namespace

int main() {
  test_fifo_order();
  test_rejects_when_full();
  test_concurrent_producers_consumers();
  test_shutdown_wakes_blocked_pop();
  std::cout << (failures == 0 ? "\nall checks passed\n"
                              : "\n" + std::to_string(failures) + " check(s) failed\n");
  return failures == 0 ? 0 : 1;
}
