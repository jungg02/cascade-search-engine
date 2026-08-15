// Correctness tests for WorkerPool: submitted tasks run, try_submit sheds
// when the queue is full and all workers are busy, and shutdown joins
// cleanly.

#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include "worker_pool.h"

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

void test_submitted_tasks_run() {
  std::cout << "tasks run\n";
  WorkerPool pool(4, 16);
  std::atomic<int> counter{0};
  constexpr int kTasks = 100;
  for (int i = 0; i < kTasks; i++) {
    while (!pool.try_submit([&counter] { counter++; })) {
      std::this_thread::yield();
    }
  }
  for (int i = 0; i < 2000 && counter.load() < kTasks; i++) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  check(counter.load() == kTasks, "all 100 submitted tasks ran exactly once");
}

void test_sheds_when_saturated() {
  std::cout << "load shedding\n";
  // One worker, blocked on a task that won't finish until we let it; queue
  // depth 1, so total in-flight capacity (1 running + 1 queued) is 2.
  WorkerPool pool(1, 1);
  std::atomic<bool> release{false};
  std::atomic<bool> first_task_running{false};

  check(pool.try_submit([&] {
          first_task_running = true;
          while (!release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }),
        "first task accepted (occupies the one worker)");
  for (int i = 0; i < 2000 && !first_task_running.load(); i++) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  check(first_task_running.load(), "first task is actually running before we continue");

  check(pool.try_submit([] {}), "second task accepted (fills the depth-1 queue)");
  check(!pool.try_submit([] {}), "third task rejected: worker busy and queue full");

  release = true;
}

}  // namespace

int main() {
  test_submitted_tasks_run();
  test_sheds_when_saturated();
  std::cout << (failures == 0 ? "\nall checks passed\n"
                              : "\n" + std::to_string(failures) + " check(s) failed\n");
  return failures == 0 ? 0 : 1;
}
