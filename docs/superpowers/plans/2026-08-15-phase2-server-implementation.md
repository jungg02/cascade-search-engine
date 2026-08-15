# Phase 2 Sub-Project A: Single-Node gRPC Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a C++ gRPC server around the existing `cascade::Index` (fixed-size worker pool, bounded queue with load shedding, LRU query cache) and the two Phase 2 experiments that need only one server instance — throughput knee and cache sensitivity — producing `bench/phase2a.md`.

**Architecture:** A sync gRPC service (`SearchServiceImpl`) enqueues each request into a hand-rolled `BoundedQueue<std::function<void()>>` behind a fixed `WorkerPool`; a full queue returns `RESOURCE_EXHAUSTED` immediately instead of blocking. A worker either serves from an `LruCache` (keyed on the lowercased, whitespace-collapsed query) or calls `cascade::Index::search()`. Python's `harness/loadgen.py` (unchanged from Phase 0) drives both experiments against the real running server over a real gRPC channel.

**Tech Stack:** C++20, gRPC 1.71/protobuf (Anaconda-provided; linked via `pkg-config`, see Task 1), Python 3.12 + `grpcio`/`grpcio-tools`, reusing `harness/loadgen.py` and `harness/histogram.py` unchanged.

**Spec:** `docs/superpowers/specs/2026-08-15-phase2-server-design.md`

## Global Constraints

- Server language/API: C++20, gRPC **sync** service API (not async/completion-queue) — see spec §2 for why.
- Build tool: extend `cpp/Makefile` (no CMake; matches the existing project).
- `grpc++`/`protobuf` are provided by the Anaconda Python distribution at `/opt/anaconda3`; linking requires `pkg-config` (installed via `brew install pkg-config` — confirmed present at plan-writing time) pointed at `/opt/anaconda3/lib/pkgconfig`. Direct `-lgrpc++ -lprotobuf` linking is **not sufficient** — verified empirically: it link-fails on missing abseil symbols that protobuf-generated code needs directly. `pkg-config --libs protobuf grpc++` is the only verified-working link line (confirmed end-to-end: compiled, linked, ran a real server, and called it from a real Python gRPC client before this plan was written).
- C++ test style: no external framework. Follow `cpp/tests/test_index.cc`'s convention exactly — a `check(bool, string)` helper printing `ok`/`FAIL`, flushing `std::cout` after every check (a hang after this point must not look like silence — this bit the project once already, see git history), a `failures` counter, `main()` running each test function in sequence and exiting non-zero on any failure.
- Python: run as modules with `PYTHONPATH=py` (`uv run python -m server.foo`), matching `baselines/*.py`'s established convention.
- Proto: package `cascade`, service `Search`, method `Query`. The spec's illustrative proto used `message Result`, which **collides** with the existing `cascade::Result` struct in `cpp/query/searcher.h` (same namespace, same name) — this plan renames it to `SearchResult`. This is the one deviation from the spec's exact text; noted here since the spec otherwise governs.
- Server defaults: `--workers=4 --queue-depth=64 --cache-capacity=200 --algorithm=wand --port=50051`. Algorithm defaults to WAND, not BlockMax-WAND, per Phase 1's measured finding (`bench/phase1.md`) that WAND is faster in wall-clock on this corpus at k=10 despite scoring more postings.
- Query depth for the server: `k=10` (realistic serving depth, matches Phase 1's query-cost table's operating point).
- No mean latency anywhere (project-wide rule, already enforced by `harness/histogram.py`).
- `cascade::Index::search()` is safe to call concurrently from multiple threads: confirmed by reading `cpp/query/searcher.cc` — it's `const`, and every per-call cursor is a local stack object; no shared mutable state is touched. This is why `SearchServiceImpl` needs no additional locking around `index_->search()` beyond what `LruCache` and `WorkerPool` already provide internally.

---

## File Structure

```
proto/search.proto                              new — the RPC schema

cpp/server/bounded_queue.h                       new — header-only bounded blocking queue
cpp/server/tests/test_bounded_queue.cc           new
cpp/server/lru_cache.h                           new — header-only LRU cache
cpp/server/tests/test_lru_cache.cc               new
cpp/server/worker_pool.h                         new — fixed-size pool wrapping BoundedQueue
cpp/server/tests/test_worker_pool.cc             new
cpp/server/search_service.h                      new
cpp/server/search_service.cc                     new
cpp/server/tests/test_search_service.cc          new
cpp/server/main.cc                               new — server binary entry point
cpp/Makefile                                     modified — proto codegen + server build rules

pyproject.toml                                   modified — add grpcio, grpcio-tools to dev extra
.gitignore                                        modified — add py/server/generated/

py/server/__init__.py                            new
py/server/gen_proto.py                           new — Python stub codegen
py/server/client.py                              new — SearchClient wrapper
py/server/process.py                             new — ServerProcess subprocess manager
py/server/throughput_knee.py                     new — experiment 1
py/server/cache_sensitivity.py                   new — experiment 2
py/server/report.py                              new — renders bench/phase2a.md
py/server/tests/__init__.py                      new
py/server/tests/test_server_integration.py       new

README.md                                        modified — "Running Phase 2 (sub-project A)"
```

---

## Task 1: Proto schema and the build pipeline for both languages

**Files:**
- Create: `proto/search.proto`
- Modify: `cpp/Makefile`
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create: `py/server/__init__.py`
- Create: `py/server/gen_proto.py`
- Test: manual verification (codegen produces compilable/importable output on both sides — there is no unit to TDD here, this task *is* the toolchain wiring)

**Interfaces:**
- Produces (for later tasks): `cpp/build/generated/search.pb.h`, `cpp/build/generated/search.grpc.pb.h` (C++ types `cascade::SearchRequest`, `cascade::SearchResult`, `cascade::SearchResponse`, `cascade::Search::Service`); `py/server/generated/search_pb2.py`, `py/server/generated/search_pb2_grpc.py` (Python `SearchRequest`, `SearchResult`, `SearchResponse`, `SearchStub`). Makefile variables `$(GRPC_CFLAGS)`, `$(GRPC_LIBS)`, `$(GEN_DIR)` for later tasks' build rules.

- [ ] **Step 1: Write the proto schema**

Create `proto/search.proto`:

```protobuf
syntax = "proto3";
package cascade;

service Search {
  rpc Query(SearchRequest) returns (SearchResponse);
}

message SearchRequest {
  string query = 1;
  uint32 k = 2;
}

message SearchResult {
  uint32 docid = 1;
  float score = 2;
}

message SearchResponse {
  repeated SearchResult results = 1;
  bool cache_hit = 2;
  uint32 queue_wait_us = 3;
}
```

- [ ] **Step 2: Add the C++ codegen and link-flag machinery to cpp/Makefile**

Open `cpp/Makefile`. After the existing `PY_SUFFIX` line and before `.PHONY: all clean test`, insert:

```makefile
# grpc/protobuf are provided by the Anaconda Python distribution. pkg-config
# is required — direct -lgrpc++ -lprotobuf linking fails on missing abseil
# symbols that protobuf-generated code references directly (verified while
# writing this plan). Override PKG_CONFIG_PATH if your grpc/protobuf live
# elsewhere.
PKG_CONFIG_PATH ?= /opt/anaconda3/lib/pkgconfig
export PKG_CONFIG_PATH

GRPC_CFLAGS := $(shell pkg-config --cflags protobuf grpc++)
GRPC_LIBS   := $(shell pkg-config --libs protobuf grpc++)

PROTO_DIR := ../proto
GEN_DIR   := $(BUILD)/generated
PROTOC ?= protoc
GRPC_CPP_PLUGIN ?= grpc_cpp_plugin
```

Then, after the existing `$(BUILD):` rule, add:

```makefile
$(BUILD)/server:
	mkdir -p $(BUILD)/server

$(GEN_DIR)/search.pb.cc $(GEN_DIR)/search.pb.h $(GEN_DIR)/search.grpc.pb.cc $(GEN_DIR)/search.grpc.pb.h: $(PROTO_DIR)/search.proto | $(BUILD)
	mkdir -p $(GEN_DIR)
	$(PROTOC) -I $(PROTO_DIR) --cpp_out=$(GEN_DIR) --grpc_out=$(GEN_DIR) \
	  --plugin=protoc-gen-grpc=$$(which $(GRPC_CPP_PLUGIN)) $(PROTO_DIR)/search.proto

$(GEN_DIR)/%.o: $(GEN_DIR)/%.cc
	$(CXX) $(CXXFLAGS) $(GRPC_CFLAGS) -I$(GEN_DIR) -c $< -o $@

$(BUILD)/server/%.o: server/%.cc | $(BUILD)/server
	$(CXX) $(CXXFLAGS) $(GRPC_CFLAGS) -Iserver -I$(GEN_DIR) -c $< -o $@
```

(The `server/%.o` pattern deliberately outputs under `$(BUILD)/server/`, not
directly under `$(BUILD)/`, so it can never collide with the existing
`$(BUILD)/%.o: index/%.cc` and `$(BUILD)/%.o: query/%.cc` pattern rules for a
same-named source file in a different directory.)

Do not add `all`/`test`/binary targets yet — those come in later tasks, once
the sources they build from exist. This task's own verification is Step 4
below: build the generated object files directly.

- [ ] **Step 3: Add Python codegen — pyproject.toml, .gitignore, gen_proto.py**

In `pyproject.toml`, under `[project.optional-dependencies]`, `dev = [...]`, add both packages:

```toml
dev = [
    "pytest>=8.0",
    "pybind11>=2.12",
    "grpcio>=1.71",
    "grpcio-tools>=1.71",
]
```

In `.gitignore`, add a line: `py/server/generated/`

Create `py/server/__init__.py` (empty file — makes `server` importable as a package under `PYTHONPATH=py`).

Create `py/server/gen_proto.py`:

```python
"""Generates the Python gRPC stubs from proto/search.proto into
py/server/generated/ (gitignored — regenerate with `python -m server.gen_proto`
whenever proto/search.proto changes, the same relationship the C++ side has
to its own generated code under cpp/build/generated/).
"""

from __future__ import annotations

from pathlib import Path

from grpc_tools import protoc

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTO_DIR = REPO_ROOT / "proto"
OUT_DIR = Path(__file__).resolve().parent / "generated"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "__init__.py").touch()
    args = [
        "protoc",
        f"-I{PROTO_DIR}",
        f"--python_out={OUT_DIR}",
        f"--grpc_python_out={OUT_DIR}",
        str(PROTO_DIR / "search.proto"),
    ]
    code = protoc.main(args)
    if code != 0:
        raise SystemExit(f"protoc failed with code {code}")
    print(f"wrote {OUT_DIR / 'search_pb2.py'} and {OUT_DIR / 'search_pb2_grpc.py'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify both codegen paths, end to end**

```bash
cd /Users/tanhonjung/cascade-search-engine
uv sync --extra dev --extra baselines
cd cpp
make build/generated/search.pb.o build/generated/search.grpc.pb.o
```

Expected: both `.o` files exist under `cpp/build/generated/`, no compiler
errors. (This exercises `protoc` + `grpc_cpp_plugin` + the `pkg-config`-derived
flags together — the exact combination Task 6's server binary link step
depends on.)

```bash
cd /Users/tanhonjung/cascade-search-engine
export PYTHONPATH=py
uv run python -m server.gen_proto
uv run python -c "
import sys
sys.path.insert(0, 'py/server/generated')
import search_pb2
req = search_pb2.SearchRequest(query='hello', k=10)
print('python codegen ok:', req.query, req.k)
"
```

Expected: `python codegen ok: hello 10`.

- [ ] **Step 5: Commit**

```bash
git add proto/search.proto cpp/Makefile pyproject.toml .gitignore py/server/__init__.py py/server/gen_proto.py uv.lock
git commit -m "phase2: add search.proto and the C++/Python codegen build pipeline"
```

---

## Task 2: BoundedQueue

**Files:**
- Create: `cpp/server/bounded_queue.h`
- Test: `cpp/server/tests/test_bounded_queue.cc`
- Modify: `cpp/Makefile` (add build/test targets for this component)

**Interfaces:**
- Produces: `cascade::BoundedQueue<T>` with `explicit BoundedQueue(size_t capacity)`, `bool try_push(T item)`, `std::optional<T> pop()`, `void shutdown()`, `size_t size() const`. Used by `WorkerPool` (Task 4).

- [ ] **Step 1: Write the failing test**

Create `cpp/server/tests/test_bounded_queue.cc`:

```cpp
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
```

- [ ] **Step 2: Add the Makefile targets and run to verify it fails**

In `cpp/Makefile`, after the `$(BUILD)/server/%.o` rule from Task 1, add:

```makefile
$(BUILD)/server/test_bounded_queue: server/tests/test_bounded_queue.cc | $(BUILD)/server
	$(CXX) $(CXXFLAGS) -Iserver $< -o $@
```

Run:

```bash
cd /Users/tanhonjung/cascade-search-engine/cpp
make build/server/test_bounded_queue
```

Expected: FAIL — compiler error, `bounded_queue.h: No such file or directory`
(the header doesn't exist yet).

- [ ] **Step 3: Implement BoundedQueue**

Create `cpp/server/bounded_queue.h`:

```cpp
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
```

- [ ] **Step 4: Run to verify it passes**

```bash
cd /Users/tanhonjung/cascade-search-engine/cpp
make build/server/test_bounded_queue
./build/server/test_bounded_queue
```

Expected: all checks print `ok`, final line `all checks passed`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add cpp/server/bounded_queue.h cpp/server/tests/test_bounded_queue.cc cpp/Makefile
git commit -m "phase2: add BoundedQueue"
```

---

## Task 3: LruCache

**Files:**
- Create: `cpp/server/lru_cache.h`
- Test: `cpp/server/tests/test_lru_cache.cc`
- Modify: `cpp/Makefile`

**Interfaces:**
- Produces: `cascade::LruCache<Value>` with `explicit LruCache(size_t capacity)`, `bool get(const std::string& key, Value* out)`, `void put(const std::string& key, Value value)`, `size_t size() const`. Used by `SearchServiceImpl` (Task 5) as `LruCache<std::vector<Result>>`.

- [ ] **Step 1: Write the failing test**

Create `cpp/server/tests/test_lru_cache.cc`:

```cpp
// Correctness tests for LruCache: miss on empty, hit returns the value,
// eviction is by recency of *use* (not insertion), and get() itself counts
// as a use.

#include <iostream>
#include <string>

#include "lru_cache.h"

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

void test_miss_on_empty() {
  std::cout << "miss\n";
  LruCache<int> cache(2);
  int out = 0;
  check(!cache.get("a", &out), "miss on empty cache");
}

void test_put_then_get() {
  std::cout << "put/get\n";
  LruCache<int> cache(2);
  cache.put("a", 1);
  int out = 0;
  check(cache.get("a", &out) && out == 1, "get returns the value put in");
  check(cache.size() == 1, "size reflects one entry");
}

void test_overwrite_existing_key() {
  std::cout << "overwrite\n";
  LruCache<int> cache(2);
  cache.put("a", 1);
  cache.put("a", 2);
  int out = 0;
  check(cache.get("a", &out) && out == 2, "put on existing key overwrites the value");
  check(cache.size() == 1, "overwriting does not grow size");
}

void test_evicts_least_recently_used() {
  std::cout << "eviction order\n";
  LruCache<int> cache(2);
  cache.put("a", 1);
  cache.put("b", 2);
  // "a" is now the least recently used of the two.
  cache.put("c", 3);  // should evict "a", not "b"
  int out = 0;
  check(!cache.get("a", &out), "least-recently-used entry (a) was evicted");
  check(cache.get("b", &out) && out == 2, "b survives eviction");
  check(cache.get("c", &out) && out == 3, "c (just inserted) survives eviction");
}

void test_get_counts_as_use() {
  std::cout << "get refreshes recency\n";
  LruCache<int> cache(2);
  cache.put("a", 1);
  cache.put("b", 2);
  int out = 0;
  check(cache.get("a", &out), "touch a via get, making b the least recently used");
  cache.put("c", 3);  // should evict "b", since "a" was just used via get()
  check(!cache.get("b", &out), "b (not touched since a's get) was evicted");
  check(cache.get("a", &out) && out == 1, "a survives because get() refreshed it");
  check(cache.get("c", &out) && out == 3, "c (just inserted) survives");
}

}  // namespace

int main() {
  test_miss_on_empty();
  test_put_then_get();
  test_overwrite_existing_key();
  test_evicts_least_recently_used();
  test_get_counts_as_use();
  std::cout << (failures == 0 ? "\nall checks passed\n"
                              : "\n" + std::to_string(failures) + " check(s) failed\n");
  return failures == 0 ? 0 : 1;
}
```

- [ ] **Step 2: Add the Makefile target and run to verify it fails**

Add to `cpp/Makefile`:

```makefile
$(BUILD)/server/test_lru_cache: server/tests/test_lru_cache.cc | $(BUILD)/server
	$(CXX) $(CXXFLAGS) -Iserver $< -o $@
```

```bash
cd /Users/tanhonjung/cascade-search-engine/cpp
make build/server/test_lru_cache
```

Expected: FAIL — `lru_cache.h: No such file or directory`.

- [ ] **Step 3: Implement LruCache**

Create `cpp/server/lru_cache.h`:

```cpp
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
```

- [ ] **Step 4: Run to verify it passes**

```bash
cd /Users/tanhonjung/cascade-search-engine/cpp
make build/server/test_lru_cache
./build/server/test_lru_cache
```

Expected: all checks `ok`, `all checks passed`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add cpp/server/lru_cache.h cpp/server/tests/test_lru_cache.cc cpp/Makefile
git commit -m "phase2: add LruCache"
```

---

## Task 4: WorkerPool

**Files:**
- Create: `cpp/server/worker_pool.h`
- Test: `cpp/server/tests/test_worker_pool.cc`
- Modify: `cpp/Makefile`

**Interfaces:**
- Consumes: `cascade::BoundedQueue<T>` (Task 2) — instantiated internally as `BoundedQueue<std::function<void()>>`.
- Produces: `cascade::WorkerPool` with `WorkerPool(size_t num_workers, size_t queue_depth)`, `bool try_submit(std::function<void()> task)`, `size_t queue_size() const`, `void shutdown()`, and a destructor that calls `shutdown()`. Used by `SearchServiceImpl` (Task 5).

- [ ] **Step 1: Write the failing test**

Create `cpp/server/tests/test_worker_pool.cc`:

```cpp
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
  bool all_accepted = true;
  for (int i = 0; i < kTasks; i++) {
    if (!pool.try_submit([&counter] { counter++; })) all_accepted = false;
  }
  check(all_accepted, "all 100 tasks accepted (4 workers, depth 16, plenty of headroom)");
  for (int i = 0; i < 2000 && counter.load() < kTasks; i++) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  check(counter.load() == kTasks, "all submitted tasks ran exactly once");
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
```

- [ ] **Step 2: Add the Makefile target and run to verify it fails**

Add to `cpp/Makefile`:

```makefile
$(BUILD)/server/test_worker_pool: server/tests/test_worker_pool.cc | $(BUILD)/server
	$(CXX) $(CXXFLAGS) -Iserver $< -o $@
```

```bash
cd /Users/tanhonjung/cascade-search-engine/cpp
make build/server/test_worker_pool
```

Expected: FAIL — `worker_pool.h: No such file or directory`.

- [ ] **Step 3: Implement WorkerPool**

Create `cpp/server/worker_pool.h`:

```cpp
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
```

- [ ] **Step 4: Run to verify it passes**

```bash
cd /Users/tanhonjung/cascade-search-engine/cpp
make build/server/test_worker_pool
./build/server/test_worker_pool
```

Expected: all checks `ok`, `all checks passed`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add cpp/server/worker_pool.h cpp/server/tests/test_worker_pool.cc cpp/Makefile
git commit -m "phase2: add WorkerPool"
```

---

## Task 5: SearchServiceImpl

**Files:**
- Create: `cpp/server/search_service.h`
- Create: `cpp/server/search_service.cc`
- Test: `cpp/server/tests/test_search_service.cc`
- Modify: `cpp/Makefile`

**Interfaces:**
- Consumes: `cascade::Index` (`cpp/query/searcher.h`, existing — `explicit Index(const std::string& directory)`, `std::vector<Result> search(const std::string&, uint32_t, Algorithm, SearchStats*) const`, where `Result{uint32_t docid; float score;}`); `cascade::LruCache<Value>` (Task 3); `cascade::WorkerPool` (Task 4); generated `cascade::SearchRequest`/`cascade::SearchResult`/`cascade::SearchResponse`/`cascade::Search::Service` (Task 1).
- Produces: `cascade::SearchServiceImpl` with `SearchServiceImpl(const Index* index, Algorithm algorithm, size_t num_workers, size_t queue_depth, size_t cache_capacity)` and the overridden `grpc::Status Query(grpc::ServerContext*, const SearchRequest*, SearchResponse*)`. Used by `main.cc` (Task 6) and the Python integration test (Task 7) indirectly through the running server.

- [ ] **Step 1: Write the failing test**

Create `cpp/server/tests/test_search_service.cc`:

```cpp
// Correctness tests for SearchServiceImpl's request handling: cache wiring
// (miss then hit, query normalization makes near-identical queries hit the
// same entry) over a tiny synthetic index. Concurrency and load-shedding
// under real timing are covered by the Python integration test against the
// actual running server (py/server/tests/test_server_integration.py) — an
// in-process call here has no network latency to make two calls reliably
// overlap, so it is not the right place to test RESOURCE_EXHAUSTED.

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "search_service.h"
#include "searcher.h"

namespace fs = std::filesystem;
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

fs::path build_tiny_index(const std::string& builder) {
  const fs::path temp = fs::temp_directory_path() / "cascade_server_test_index";
  fs::remove_all(temp);
  fs::create_directories(temp);
  const fs::path corpus = temp / "corpus.tsv";
  {
    std::ofstream out(corpus);
    out << "doc0\tthe quick brown fox\n";
    out << "doc1\tthe lazy dog sleeps\n";
    out << "doc2\tfox and dog play\n";
  }
  const fs::path index_dir = temp / "index";
  const std::string command =
      builder + " " + corpus.string() + " " + index_dir.string() + " 2>/dev/null";
  if (std::system(command.c_str()) != 0) {
    std::cerr << "index build failed: " << command << "\n";
    std::exit(1);
  }
  return index_dir;
}

SearchRequest make_request(const std::string& query, uint32_t k = 10) {
  SearchRequest req;
  req.set_query(query);
  req.set_k(k);
  return req;
}

void test_cache_miss_then_hit(const Index& index) {
  std::cout << "cache miss then hit\n";
  SearchServiceImpl service(&index, Algorithm::kWand, /*num_workers=*/2,
                            /*queue_depth=*/4, /*cache_capacity=*/8);

  SearchResponse first;
  auto req = make_request("fox");
  grpc::Status status = service.Query(nullptr, &req, &first);
  check(status.ok(), "first call succeeds");
  check(!first.cache_hit(), "first call is a cache miss");
  check(first.results_size() > 0, "first call returns results for a term that appears (fox)");

  SearchResponse second;
  status = service.Query(nullptr, &req, &second);
  check(status.ok(), "second call succeeds");
  check(second.cache_hit(), "second identical call is a cache hit");
  check(second.results_size() == first.results_size(), "cached results have the same count");
  for (int i = 0; i < first.results_size(); i++) {
    check(second.results(i).docid() == first.results(i).docid() &&
              second.results(i).score() == first.results(i).score(),
          "cached result " + std::to_string(i) + " matches the original");
  }
}

void test_normalization_shares_cache_entry(const Index& index) {
  std::cout << "normalization\n";
  SearchServiceImpl service(&index, Algorithm::kWand, /*num_workers=*/2,
                            /*queue_depth=*/4, /*cache_capacity=*/8);

  SearchResponse warm;
  auto warm_req = make_request("Fox Dog");
  check(service.Query(nullptr, &warm_req, &warm).ok(), "warm-up call succeeds");
  check(!warm.cache_hit(), "warm-up call is a miss");

  SearchResponse variant;
  auto variant_req = make_request("  fox   dog  ");
  check(service.Query(nullptr, &variant_req, &variant).ok(), "differently-spaced call succeeds");
  check(variant.cache_hit(),
        "differently-cased, differently-spaced query normalizes to the same cache key");
}

void test_different_queries_miss_independently(const Index& index) {
  std::cout << "distinct queries\n";
  SearchServiceImpl service(&index, Algorithm::kWand, /*num_workers=*/2,
                            /*queue_depth=*/4, /*cache_capacity=*/8);
  SearchResponse fox_resp, dog_resp;
  auto fox_req = make_request("fox");
  auto dog_req = make_request("dog");
  check(service.Query(nullptr, &fox_req, &fox_resp).ok(), "fox call succeeds");
  check(service.Query(nullptr, &dog_req, &dog_resp).ok(), "dog call succeeds");
  check(!dog_resp.cache_hit(), "a genuinely different query is not a cache hit off fox's entry");
}

}  // namespace

int main(int argc, char** argv) {
  const std::string builder = argc > 1 ? argv[1] : "build/build_index";
  const fs::path index_dir = build_tiny_index(builder);
  Index index(index_dir.string());

  test_cache_miss_then_hit(index);
  test_normalization_shares_cache_entry(index);
  test_different_queries_miss_independently(index);

  fs::remove_all(index_dir.parent_path());
  std::cout << (failures == 0 ? "\nall checks passed\n"
                              : "\n" + std::to_string(failures) + " check(s) failed\n");
  return failures == 0 ? 0 : 1;
}
```

- [ ] **Step 2: Add the Makefile targets and run to verify it fails**

Add to `cpp/Makefile` (after the Task 1 pattern rules; `SERVER_COMMON` is new,
`COMMON`/`QUERY` already exist from before this plan):

```makefile
SERVER_COMMON := $(BUILD)/server/search_service.o $(GEN_DIR)/search.pb.o $(GEN_DIR)/search.grpc.pb.o

$(BUILD)/server/test_search_service: server/tests/test_search_service.cc $(COMMON) $(QUERY) $(SERVER_COMMON) | $(BUILD)/server
	$(CXX) $(CXXFLAGS) $(GRPC_CFLAGS) -Iserver -I$(GEN_DIR) $^ $(GRPC_LIBS) -Wl,-rpath,/opt/anaconda3/lib -o $@
```

```bash
cd /Users/tanhonjung/cascade-search-engine/cpp
make build/build_index build/server/test_search_service
```

Expected: FAIL — `search_service.h: No such file or directory` (and no
`build/server/search_service.o` rule can succeed either, same reason).

- [ ] **Step 3: Implement SearchServiceImpl**

Create `cpp/server/search_service.h`:

```cpp
// Ties BoundedQueue/WorkerPool, LruCache, and cascade::Index together behind
// the generated Search::Service RPC interface. A cache hit skips Index::search
// entirely; a miss runs it on a worker thread pulled from the fixed pool, with
// RESOURCE_EXHAUSTED returned immediately (not queued) when the pool's queue
// is already at its configured depth.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "lru_cache.h"
#include "search.grpc.pb.h"
#include "searcher.h"
#include "worker_pool.h"

namespace cascade {

class SearchServiceImpl final : public Search::Service {
 public:
  // Does not take ownership of index; the caller (main.cc) must keep it
  // alive for the lifetime of this service. index_->search() is const and
  // touches no shared mutable state (verified by reading searcher.cc), so
  // concurrent calls from multiple worker threads are safe without extra
  // locking here.
  SearchServiceImpl(const Index* index, Algorithm algorithm, size_t num_workers,
                     size_t queue_depth, size_t cache_capacity);

  grpc::Status Query(grpc::ServerContext* context, const SearchRequest* request,
                      SearchResponse* response) override;

 private:
  // The actual query logic, run on a worker thread. Populates response
  // (results + cache_hit) directly; does not touch queue_wait_us, which
  // Query() fills in from timestamps only it has.
  void handle_query(const SearchRequest& request, SearchResponse* response);

  static std::string normalize(const std::string& query);

  const Index* index_;
  Algorithm algorithm_;
  LruCache<std::vector<Result>> cache_;
  WorkerPool pool_;
};

}  // namespace cascade
```

Create `cpp/server/search_service.cc`:

```cpp
#include "search_service.h"

#include <cctype>
#include <chrono>
#include <future>
#include <memory>

namespace cascade {

namespace {
uint32_t microseconds_between(std::chrono::steady_clock::time_point start,
                               std::chrono::steady_clock::time_point end) {
  return static_cast<uint32_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}
}  // namespace

SearchServiceImpl::SearchServiceImpl(const Index* index, Algorithm algorithm,
                                      size_t num_workers, size_t queue_depth,
                                      size_t cache_capacity)
    : index_(index),
      algorithm_(algorithm),
      cache_(cache_capacity),
      pool_(num_workers, queue_depth) {}

std::string SearchServiceImpl::normalize(const std::string& query) {
  std::string out;
  out.reserve(query.size());
  bool last_was_space = true;  // collapses leading whitespace too
  for (char c : query) {
    char lower = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (std::isspace(static_cast<unsigned char>(lower))) {
      if (!last_was_space) out.push_back(' ');
      last_was_space = true;
    } else {
      out.push_back(lower);
      last_was_space = false;
    }
  }
  while (!out.empty() && out.back() == ' ') out.pop_back();
  return out;
}

void SearchServiceImpl::handle_query(const SearchRequest& request, SearchResponse* response) {
  const std::string key = normalize(request.query());

  std::vector<Result> cached;
  if (cache_.get(key, &cached)) {
    response->set_cache_hit(true);
    for (const Result& r : cached) {
      SearchResult* out = response->add_results();
      out->set_docid(r.docid);
      out->set_score(r.score);
    }
    return;
  }

  SearchStats stats;
  std::vector<Result> results = index_->search(request.query(), request.k(), algorithm_, &stats);
  response->set_cache_hit(false);
  for (const Result& r : results) {
    SearchResult* out = response->add_results();
    out->set_docid(r.docid);
    out->set_score(r.score);
  }
  cache_.put(key, results);
}

grpc::Status SearchServiceImpl::Query(grpc::ServerContext*, const SearchRequest* request,
                                       SearchResponse* response) {
  auto promise = std::make_shared<std::promise<void>>();
  std::future<void> future = promise->get_future();
  const auto queued_at = std::chrono::steady_clock::now();

  const bool accepted = pool_.try_submit([this, request, response, promise, queued_at] {
    response->set_queue_wait_us(
        microseconds_between(queued_at, std::chrono::steady_clock::now()));
    handle_query(*request, response);
    promise->set_value();
  });

  if (!accepted) {
    return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, "queue full");
  }
  future.wait();
  return grpc::Status::OK;
}

}  // namespace cascade
```

- [ ] **Step 4: Run to verify it passes**

```bash
cd /Users/tanhonjung/cascade-search-engine/cpp
make build/build_index build/server/test_search_service
./build/server/test_search_service
```

Expected: all checks `ok`, `all checks passed`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add cpp/server/search_service.h cpp/server/search_service.cc cpp/server/tests/test_search_service.cc cpp/Makefile
git commit -m "phase2: add SearchServiceImpl (cache + worker pool wired to the RPC)"
```

---

## Task 6: Server binary (main.cc)

**Files:**
- Create: `cpp/server/main.cc`
- Modify: `cpp/Makefile` (add `server_bin`, and finally the `all`/`test` targets covering everything from Tasks 1-6)

**Interfaces:**
- Consumes: `cascade::SearchServiceImpl` (Task 5), `cascade::Index` (existing), `cascade::Algorithm` (existing, `cpp/query/searcher.h`).
- Produces: the `cpp/build/server_bin` executable, invoked as `server_bin <index_dir> [--port=P] [--workers=N] [--queue-depth=D] [--cache-capacity=C] [--algorithm=wand|blockmax-wand|daat-or]`. Consumed by `py/server/process.py` (Task 7).

- [ ] **Step 1: Write main.cc**

There is no TDD cycle for this file: it is flag parsing and `grpc::ServerBuilder`
wiring, verified by actually running the server (this step) and by the
integration test in Task 7. Create `cpp/server/main.cc`:

```cpp
// Phase 2 sub-project A server binary: loads the cascade index, wires up
// SearchServiceImpl (worker pool + bounded queue + LRU cache), and serves
// gRPC Search/Query on the given port until killed.
//
// Usage: server_bin <index_dir> [--port=P] [--workers=N] [--queue-depth=D]
//                    [--cache-capacity=C] [--algorithm=wand|blockmax-wand|daat-or]

#include <grpcpp/grpcpp.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "search_service.h"
#include "searcher.h"

namespace {

using cascade::Algorithm;

struct Flags {
  std::string index_dir;
  int port = 50051;
  size_t workers = 4;
  size_t queue_depth = 64;
  size_t cache_capacity = 200;
  Algorithm algorithm = Algorithm::kWand;
};

Algorithm parse_algorithm(const std::string& value) {
  if (value == "wand") return Algorithm::kWand;
  if (value == "blockmax-wand") return Algorithm::kBlockMaxWand;
  if (value == "daat-or") return Algorithm::kDaatOr;
  std::cerr << "unknown --algorithm=" << value << " (want wand|blockmax-wand|daat-or)\n";
  std::exit(1);
}

// Parses "--name=value" out of argv, or returns default_value if absent.
std::string flag_value(int argc, char** argv, const std::string& name,
                        const std::string& default_value) {
  const std::string prefix = "--" + name + "=";
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg.rfind(prefix, 0) == 0) return arg.substr(prefix.size());
  }
  return default_value;
}

Flags parse_flags(int argc, char** argv) {
  if (argc < 2 || std::string(argv[1]).rfind("--", 0) == 0) {
    std::cerr << "usage: server_bin <index_dir> [--port=P] [--workers=N] "
                 "[--queue-depth=D] [--cache-capacity=C] "
                 "[--algorithm=wand|blockmax-wand|daat-or]\n";
    std::exit(1);
  }
  Flags flags;
  flags.index_dir = argv[1];
  flags.port = std::stoi(flag_value(argc, argv, "port", "50051"));
  flags.workers = std::stoul(flag_value(argc, argv, "workers", "4"));
  flags.queue_depth = std::stoul(flag_value(argc, argv, "queue-depth", "64"));
  flags.cache_capacity = std::stoul(flag_value(argc, argv, "cache-capacity", "200"));
  flags.algorithm = parse_algorithm(flag_value(argc, argv, "algorithm", "wand"));
  return flags;
}

}  // namespace

int main(int argc, char** argv) {
  Flags flags = parse_flags(argc, argv);

  std::cerr << "loading index from " << flags.index_dir << "...\n";
  cascade::Index index(flags.index_dir);
  std::cerr << "loaded " << index.doc_count() << " docs\n";

  cascade::SearchServiceImpl service(&index, flags.algorithm, flags.workers,
                                      flags.queue_depth, flags.cache_capacity);

  const std::string address = "127.0.0.1:" + std::to_string(flags.port);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  // Bounds the gRPC layer's own dispatch-thread pool so it can't grow
  // unbounded under load: every request that isn't shed either occupies a
  // worker or sits in the queue, so this many gRPC threads blocked on a
  // future covers the worst case without growing further. This is not the
  // concurrency control the plan asks for — WorkerPool is — it just keeps
  // gRPC's own bookkeeping bounded too.
  grpc::ResourceQuota quota;
  quota.SetMaxThreads(static_cast<int>(flags.workers + flags.queue_depth));
  builder.SetResourceQuota(quota);

  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
  std::cerr << "listening on " << address << " (workers=" << flags.workers
            << " queue_depth=" << flags.queue_depth
            << " cache_capacity=" << flags.cache_capacity << ")\n";
  server->Wait();
  return 0;
}
```

- [ ] **Step 2: Add the Makefile target and finish wiring `all`/`test`**

Add to `cpp/Makefile`:

```makefile
$(BUILD)/server_bin: server/main.cc $(COMMON) $(QUERY) $(SERVER_COMMON) | $(BUILD)/server
	$(CXX) $(CXXFLAGS) $(GRPC_CFLAGS) -Iserver -I$(GEN_DIR) $^ $(GRPC_LIBS) -Wl,-rpath,/opt/anaconda3/lib -o $@
```

Then find the existing `all:` and `test:` targets (from before this plan) and
replace them with:

```makefile
all: $(BUILD)/build_index $(BUILD)/test_index ../py/cascade_index$(PY_SUFFIX) \
     $(BUILD)/server_bin $(BUILD)/server/test_bounded_queue \
     $(BUILD)/server/test_lru_cache $(BUILD)/server/test_worker_pool \
     $(BUILD)/server/test_search_service

test: $(BUILD)/test_index $(BUILD)/server/test_bounded_queue \
      $(BUILD)/server/test_lru_cache $(BUILD)/server/test_worker_pool \
      $(BUILD)/server/test_search_service
	$(BUILD)/test_index
	$(BUILD)/server/test_bounded_queue
	$(BUILD)/server/test_lru_cache
	$(BUILD)/server/test_worker_pool
	$(BUILD)/server/test_search_service
```

(`clean`'s existing `rm -rf $(BUILD) ../py/cascade_index*.so` already covers
everything under `build/server/` and `build/generated/` — no change needed
there.)

- [ ] **Step 3: Build everything and smoke-test the server manually**

```bash
cd /Users/tanhonjung/cascade-search-engine/cpp
make clean
make all
make test
```

Expected: clean build, no warnings; `make test` prints `all checks passed`
four times (once per test binary) and exits 0.

```bash
cd /Users/tanhonjung/cascade-search-engine
ls indexes/cascade-msmarco-passage  # from Phase 1 — must already exist
cpp/build/server_bin indexes/cascade-msmarco-passage --port=50099 &
SERVER_PID=$!
sleep 1
uv run --with grpcio --with grpcio-tools python -c "
import sys
sys.path.insert(0, 'py/server/generated')
import grpc, search_pb2, search_pb2_grpc
channel = grpc.insecure_channel('127.0.0.1:50099')
grpc.channel_ready_future(channel).result(timeout=5)
stub = search_pb2_grpc.SearchStub(channel)
resp = stub.Query(search_pb2.SearchRequest(query='what is a bank teller', k=10))
print('results:', len(resp.results), 'cache_hit:', resp.cache_hit)
for r in resp.results[:3]:
    print(' ', r.docid, r.score)
"
kill $SERVER_PID
```

Expected: `results: 10 cache_hit: False`, followed by three `docid score`
lines with non-trivial scores. (`py/server/generated/` must already exist
from Task 1, Step 4 — this step reuses it rather than regenerating.)

- [ ] **Step 4: Commit**

```bash
git add cpp/server/main.cc cpp/Makefile
git commit -m "phase2: add the server binary (main.cc), wire up make all/test"
```

---

## Task 7: Python client, process manager, and the integration test

**Files:**
- Create: `py/server/client.py`
- Create: `py/server/process.py`
- Create: `py/server/tests/__init__.py`
- Create: `py/server/tests/test_server_integration.py`
- Modify: `pyproject.toml` (add `py/server/tests` to `testpaths`)

**Interfaces:**
- Consumes: `cpp/build/server_bin` (Task 6), `py/server/generated/search_pb2*.py` (Task 1).
- Produces: `server.client.SearchClient` (`__init__(address, timeout_s=5.0)`, `dispatch(query, k=10) -> SearchResponse`, `close()`); `server.process.ServerProcess` (context manager, `__init__(index_dir, port=50051, workers=4, queue_depth=64, cache_capacity=200, algorithm="wand", ready_timeout_s=30.0)`, `.address` property). Used by `throughput_knee.py` and `cache_sensitivity.py` (Tasks 8-9).

- [ ] **Step 1: Write client.py**

Create `py/server/client.py`:

```python
"""Thin gRPC client wrapper around the cascade server's Search service.

The generated stubs (search_pb2_grpc.py) use a bare `import search_pb2`,
which only resolves if the generated/ directory itself is on sys.path — not
just `py/` per the project's usual PYTHONPATH convention. Every module that
needs the stubs inserts the path itself (idempotent — sys.path membership
doesn't duplicate meaningfully at this scale) rather than depending on
import order between modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

_GENERATED_DIR = Path(__file__).resolve().parent / "generated"
if str(_GENERATED_DIR) not in sys.path:
    sys.path.insert(0, str(_GENERATED_DIR))

import grpc

try:
    import search_pb2
    import search_pb2_grpc
except ImportError as exc:
    raise ImportError(
        "generated proto stubs not found; run `python -m server.gen_proto` first"
    ) from exc


class SearchClient:
    """One gRPC channel + stub. dispatch() is the shape harness.loadgen wants:
    a Callable[[str], object]."""

    def __init__(self, address: str, timeout_s: float = 5.0) -> None:
        self._channel = grpc.insecure_channel(address)
        grpc.channel_ready_future(self._channel).result(timeout=timeout_s)
        self._stub = search_pb2_grpc.SearchStub(self._channel)

    def dispatch(self, query: str, k: int = 10):
        return self._stub.Query(search_pb2.SearchRequest(query=query, k=k))

    def close(self) -> None:
        self._channel.close()
```

- [ ] **Step 2: Write process.py**

Create `py/server/process.py`:

```python
"""Starts/stops the cascade C++ server binary as a subprocess — used by both
the integration test and the experiment scripts, so there's exactly one
place that knows the binary's flag names and readiness-wait logic.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import grpc

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_BIN = REPO_ROOT / "cpp" / "build" / "server_bin"


class ServerProcess:
    """Context manager: `with ServerProcess(index_dir) as server: ...` — use
    server.address to connect."""

    def __init__(
        self,
        index_dir: Path,
        port: int = 50051,
        workers: int = 4,
        queue_depth: int = 64,
        cache_capacity: int = 200,
        algorithm: str = "wand",
        ready_timeout_s: float = 30.0,
    ) -> None:
        if not SERVER_BIN.exists():
            raise FileNotFoundError(f"{SERVER_BIN} not built; run `make -C cpp all` first")
        self._address = f"127.0.0.1:{port}"
        self._args = [
            str(SERVER_BIN),
            str(index_dir),
            f"--port={port}",
            f"--workers={workers}",
            f"--queue-depth={queue_depth}",
            f"--cache-capacity={cache_capacity}",
            f"--algorithm={algorithm}",
        ]
        self._ready_timeout_s = ready_timeout_s
        self._process: subprocess.Popen | None = None

    @property
    def address(self) -> str:
        return self._address

    def __enter__(self) -> "ServerProcess":
        self._process = subprocess.Popen(self._args)
        channel = grpc.insecure_channel(self._address)
        try:
            grpc.channel_ready_future(channel).result(timeout=self._ready_timeout_s)
        except grpc.FutureTimeoutError:
            self.__exit__(None, None, None)
            raise TimeoutError(
                f"server did not become ready within {self._ready_timeout_s}s "
                f"(args: {self._args})"
            )
        finally:
            channel.close()
        return self

    def __exit__(self, *exc_info) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None
```

- [ ] **Step 3: Write the failing integration test**

Create `py/server/tests/__init__.py` (empty file).

Create `py/server/tests/test_server_integration.py`:

```python
"""Integration test: start the real C++ server as a subprocess, talk to it
over real gRPC. This proves the network path works end to end — the pure
C++ unit test of SearchServiceImpl (cpp/server/tests/test_search_service.cc)
only proves the request-handling logic, not that grpc::Server, the generated
stubs, and this Python client actually agree on the wire format.

Needs the Phase 1 index and the generated proto stubs; skips gracefully
(not a collection failure) when either is absent, so a fresh clone's default
`uv run pytest` still passes without the ~900MB corpus.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
INDEX_DIR = REPO_ROOT / "indexes" / "cascade-msmarco-passage"
GENERATED_DIR = Path(__file__).resolve().parents[1] / "generated"
_READY = INDEX_DIR.exists() and (GENERATED_DIR / "search_pb2.py").exists()

pytestmark = pytest.mark.skipif(
    not _READY,
    reason="needs the built cascade index and generated proto stubs "
    "(see README's Phase 1 and Phase 2 setup)",
)

if _READY:
    import grpc

    from server.client import SearchClient
    from server.process import ServerProcess


def test_single_query_round_trip():
    with ServerProcess(INDEX_DIR, port=50151) as server:
        client = SearchClient(server.address)
        response = client.dispatch("what is a bank teller", k=10)
        assert len(response.results) > 0
        assert len(response.results) <= 10
        client.close()


def test_cache_hit_on_repeated_query():
    with ServerProcess(INDEX_DIR, port=50152, cache_capacity=10) as server:
        client = SearchClient(server.address)
        first = client.dispatch("bank teller")
        assert not first.cache_hit
        second = client.dispatch("bank teller")
        assert second.cache_hit
        assert [(r.docid, r.score) for r in second.results] == [
            (r.docid, r.score) for r in first.results
        ]
        client.close()


def test_saturated_queue_sheds_load():
    # One worker, one queue slot: at most 2 requests in flight at once. Firing
    # 20 concurrently on one shared channel makes it near-certain some arrive
    # before the first two finish, so at least one comes back
    # RESOURCE_EXHAUSTED rather than succeeding or hanging — this is what a
    # synthetic in-process call (test_search_service.cc) can't exercise:
    # there's no network latency there to spread 20 calls out in wall time.
    with ServerProcess(INDEX_DIR, port=50153, workers=1, queue_depth=1) as server:
        client = SearchClient(server.address)
        results = []
        lock = threading.Lock()

        def fire():
            try:
                client.dispatch("the")
                with lock:
                    results.append("ok")
            except grpc.RpcError as exc:
                with lock:
                    results.append(exc.code())

        threads = [threading.Thread(target=fire) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        client.close()

        assert "ok" in results, "at least some requests should succeed"
        assert grpc.StatusCode.RESOURCE_EXHAUSTED in results, (
            "at least one request should be shed under this saturating burst"
        )
```

In `pyproject.toml`, change:

```toml
[tool.pytest.ini_options]
testpaths = ["py/tests"]
pythonpath = ["py"]
```

to:

```toml
[tool.pytest.ini_options]
testpaths = ["py/tests", "py/server/tests"]
pythonpath = ["py"]
```

Run to verify the new tests are collected and skip cleanly if prerequisites
are missing, or fail loudly with a specific reason if not:

```bash
cd /Users/tanhonjung/cascade-search-engine
uv run pytest py/server/tests -v
```

Expected at this point: 3 tests, all **failing** with `ModuleNotFoundError`
or `AssertionError` — `client.py`/`process.py` exist (from Steps 1-2) but
`server_bin` (Task 6) and the generated stubs must already exist too for
these to have any chance of passing; if they're both present (they should
be, from earlier tasks), the tests should actually run and might already
pass here, since Steps 1-2 already contain the real implementation, not a
stub. If so, that's fine — this task's "RED" step is really about the test
file not existing yet; treat a pass here as the expected outcome given
Steps 1-2 already shipped working code, and proceed to Step 4.

- [ ] **Step 4: Run to verify it passes**

```bash
cd /Users/tanhonjung/cascade-search-engine
uv run pytest py/server/tests -v
```

Expected: 3 passed (or 3 skipped if `indexes/cascade-msmarco-passage` is
somehow absent on this machine — it should not be, Phase 1 already built it).

- [ ] **Step 5: Commit**

```bash
git add py/server/client.py py/server/process.py py/server/tests/__init__.py py/server/tests/test_server_integration.py pyproject.toml
git commit -m "phase2: add Python client, server-process manager, and the integration test"
```

---

## Task 8: Throughput-knee experiment

**Files:**
- Create: `py/server/throughput_knee.py`

**Interfaces:**
- Consumes: `harness.datasets.load_queries` (existing), `harness.loadgen.run_open_loop` (existing, unchanged), `harness.runmeta.run_metadata` (existing), `server.client.SearchClient` (Task 7), `server.process.ServerProcess` (Task 7).
- Produces: `bench/results/server-throughput-knee.json`. Consumed by `server.report` (Task 10).

- [ ] **Step 1: Write throughput_knee.py**

No TDD cycle: this is an experiment driver, verified by actually running it
against the real server and inspecting the output (Step 2), the same way
`baselines/latency.py` (Phase 0) was verified. Create `py/server/throughput_knee.py`:

```python
"""Throughput-knee experiment: QPS on the x-axis, p99 on the y-axis, against
the real gRPC server (not an in-process pybind call, unlike Phase 1's
query-cost table) — the whole point of Phase 2 is measuring what a server's
own concurrency policy (fixed worker pool + bounded queue) does under load.

Structure mirrors baselines/latency.py: the same open-loop generator, the
same Zipfian query sampler over dev queries, the same 3-repeats-per-point
protocol (a laptop's thermal state isn't stable enough to trust one run).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from harness.datasets import load_queries
from harness.loadgen import run_open_loop
from harness.runmeta import run_metadata
from server.client import SearchClient
from server.process import ServerProcess

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
INDEX_DIR = REPO_ROOT / "indexes" / "cascade-msmarco-passage"


def measure(client: SearchClient, qps: float, duration_s: float, workers: int,
            hits: int, queries: list[str], zipf_s: float, seed: int) -> dict:
    def dispatch(query: str) -> None:
        client.dispatch(query, k=hits)

    result = run_open_loop(
        dispatch=dispatch, queries=queries, qps=qps, duration_s=duration_s,
        workers=workers, seed=seed, zipf_s=zipf_s,
    )
    # RESOURCE_EXHAUSTED under load is an *expected* outcome at high offered
    # QPS, not a broken run: run_open_loop's own dispatch wrapper already
    # catches any exception and records it as an error rather than raising,
    # so a high error count at high QPS is exactly the knee showing up.
    return result.summary()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qps", type=float, nargs="+", default=[20, 40, 60, 80, 100, 120])
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--client-workers", type=int, default=8)
    parser.add_argument("--hits", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--zipf-s", type=float, default=1.0)
    parser.add_argument("--server-workers", type=int, default=4)
    parser.add_argument("--queue-depth", type=int, default=64)
    parser.add_argument("--cache-capacity", type=int, default=200)
    parser.add_argument("--algorithm", default="wand")
    parser.add_argument("--port", type=int, default=50161)
    args = parser.parse_args()

    queries = [text for _, text in sorted(load_queries("dev").items())]

    with ServerProcess(
        INDEX_DIR, port=args.port, workers=args.server_workers,
        queue_depth=args.queue_depth, cache_capacity=args.cache_capacity,
        algorithm=args.algorithm,
    ) as server:
        client = SearchClient(server.address)

        points = []
        for qps in args.qps:
            repeats = [
                measure(client, qps, args.duration, args.client_workers, args.hits,
                        queries, args.zipf_s, seed)
                for seed in range(args.repeats)
            ]
            p99s = [r["latency"]["p99_us"] for r in repeats]
            point = {
                "offered_qps": qps,
                "repeats": repeats,
                "p99_us_across_repeats": {
                    "min": min(p99s), "median": statistics.median(p99s), "max": max(p99s),
                },
            }
            points.append(point)
            median = statistics.median([r["latency"]["p50_us"] for r in repeats])
            errors = sum(r["errors"] for r in repeats)
            print(f"{qps:>6.0f} QPS   p50={median/1000:7.2f}ms   "
                  f"p99={statistics.median(p99s)/1000:7.2f}ms   errors={errors}")

        client.close()

    output = {
        "server": "cascade-grpc",
        "generator": "open-loop, Poisson arrivals, latency from scheduled arrival",
        "query_set": "dev",
        "num_distinct_queries": len(queries),
        "zipf_s": args.zipf_s,
        "client_workers": args.client_workers,
        "server_workers": args.server_workers,
        "queue_depth": args.queue_depth,
        "cache_capacity": args.cache_capacity,
        "algorithm": args.algorithm,
        "hits": args.hits,
        "duration_s": args.duration,
        "repeats": args.repeats,
        "points": points,
        "provenance": run_metadata(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "server-throughput-knee.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nwrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and sanity-check the output**

```bash
cd /Users/tanhonjung/cascade-search-engine
export PYTHONPATH=py
uv run python -m server.throughput_knee --qps 20 40 --duration 5 --repeats 1
```

Expected: two printed lines (20 QPS, 40 QPS) with plausible p50/p99 values
(tens of ms, not seconds, not near-zero), `errors=0` at these moderate
rates, and a final `wrote bench/results/server-throughput-knee.json` line.
Inspect the file: `cat bench/results/server-throughput-knee.json | python3 -m json.tool | head -40`
should show well-formed JSON with a `points` array.

If p99 is already very high or errors are non-zero even at QPS=20, that
signals the default `--workers`/`--queue-depth` are miscalibrated for this
machine — note it, but do not change the defaults in this step; the real
sweep (with more QPS points) in Task 10's verification is where the actual
knee gets found and the real defaults get validated or adjusted.

- [ ] **Step 3: Commit**

```bash
git add py/server/throughput_knee.py
git commit -m "phase2: add the throughput-knee experiment"
```

---

## Task 9: Cache-sensitivity experiment

**Files:**
- Create: `py/server/cache_sensitivity.py`

**Interfaces:**
- Consumes: same as Task 8, plus `harness.histogram.LatencyRecorder` (existing).
- Produces: `bench/results/server-cache-sensitivity.json`. Consumed by `server.report` (Task 10).

- [ ] **Step 1: Write cache_sensitivity.py**

No TDD cycle — same reasoning as Task 8. Create `py/server/cache_sensitivity.py`:

```python
"""Cache-sensitivity experiment: sweep the Zipfian skew, plot hit rate
against p99, and split cache-hit and cache-miss latency (a blended number
hides which one actually matters — the same principle bench/baselines.md
applies to queueing).

Run at one fixed, moderate QPS, below the throughput knee found by
throughput_knee.py (Task 8), so the numbers reflect cache behavior rather
than queueing on top of it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from harness.datasets import load_queries
from harness.histogram import LatencyRecorder
from harness.loadgen import run_open_loop
from harness.runmeta import run_metadata
from server.client import SearchClient
from server.process import ServerProcess

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
INDEX_DIR = REPO_ROOT / "indexes" / "cascade-msmarco-passage"


def measure(client: SearchClient, zipf_s: float, qps: float, duration_s: float,
            workers: int, hits: int, queries: list[str], seed: int) -> dict:
    hit_latency = LatencyRecorder()
    miss_latency = LatencyRecorder()
    hits_count = 0
    misses_count = 0

    def dispatch(query: str) -> None:
        nonlocal hits_count, misses_count
        # This times the same dispatch() call run_open_loop's own `service`
        # recorder already times; the duplication is necessary because that
        # recorder has no hit/miss split, and adding one would mean changing
        # harness/loadgen.py for every other caller (Phase 0's baselines
        # included) rather than just this one experiment.
        started = time.perf_counter_ns()
        response = client.dispatch(query, k=hits)
        elapsed_us = (time.perf_counter_ns() - started) / 1000
        if response.cache_hit:
            hits_count += 1
            hit_latency.record(elapsed_us)
        else:
            misses_count += 1
            miss_latency.record(elapsed_us)

    result = run_open_loop(
        dispatch=dispatch, queries=queries, qps=qps, duration_s=duration_s,
        workers=workers, seed=seed, zipf_s=zipf_s,
    )
    total = hits_count + misses_count
    return {
        "zipf_s": zipf_s,
        "hit_rate": hits_count / total if total else 0.0,
        "hits": hits_count,
        "misses": misses_count,
        "hit_latency": hit_latency.summary(),
        "miss_latency": miss_latency.summary(),
        "overall": result.summary(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zipf-s", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--qps", type=float, default=20.0,
                         help="fixed QPS, chosen below the throughput knee")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--client-workers", type=int, default=4)
    parser.add_argument("--hits", type=int, default=10)
    parser.add_argument("--server-workers", type=int, default=4)
    parser.add_argument("--queue-depth", type=int, default=64)
    parser.add_argument("--cache-capacity", type=int, default=200)
    parser.add_argument("--algorithm", default="wand")
    parser.add_argument("--port", type=int, default=50162)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    queries = [text for _, text in sorted(load_queries("dev").items())]

    with ServerProcess(
        INDEX_DIR, port=args.port, workers=args.server_workers,
        queue_depth=args.queue_depth, cache_capacity=args.cache_capacity,
        algorithm=args.algorithm,
    ) as server:
        client = SearchClient(server.address)
        points = []
        for zipf_s in args.zipf_s:
            point = measure(client, zipf_s, args.qps, args.duration, args.client_workers,
                             args.hits, queries, args.seed)
            points.append(point)
            hit_p99 = point["hit_latency"].get("p99_us", 0) / 1000
            miss_p99 = point["miss_latency"].get("p99_us", 0) / 1000
            print(f"s={zipf_s:>4.1f}  hit_rate={point['hit_rate']:.1%}  "
                  f"hit_p99={hit_p99:.2f}ms  miss_p99={miss_p99:.2f}ms")
        client.close()

    output = {
        "server": "cascade-grpc",
        "query_set": "dev",
        "num_distinct_queries": len(queries),
        "qps": args.qps,
        "duration_s": args.duration,
        "cache_capacity": args.cache_capacity,
        "server_workers": args.server_workers,
        "queue_depth": args.queue_depth,
        "algorithm": args.algorithm,
        "hits": args.hits,
        "points": points,
        "provenance": run_metadata(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "server-cache-sensitivity.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nwrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and sanity-check the output**

```bash
cd /Users/tanhonjung/cascade-search-engine
export PYTHONPATH=py
uv run python -m server.cache_sensitivity --zipf-s 0.5 2.0 --duration 5
```

Expected: two printed lines (s=0.5, s=2.0). The s=2.0 (more skewed) line
should show a **higher** hit rate than s=0.5 — that's the whole point of the
sweep (skewed traffic repeats the same head queries more, which is what a
cache exploits). If hit rate is ~0% at both, `--cache-capacity` (default
200) may be too small relative to how the Zipf sampler is drawing from
~6,980 dev queries, or too large if hit rate is ~100% at both regardless of
skew — note the actual numbers; Task 10's verification is where this gets
tuned if needed, per the design spec's §6.

- [ ] **Step 3: Commit**

```bash
git add py/server/cache_sensitivity.py
git commit -m "phase2: add the cache-sensitivity experiment"
```

---

## Task 10: Report generator and full experiment run

**Files:**
- Create: `py/server/report.py`

**Interfaces:**
- Consumes: `bench/results/server-throughput-knee.json` (Task 8), `bench/results/server-cache-sensitivity.json` (Task 9).
- Produces: `bench/phase2a.md`, the sub-project's exit artifact.

- [ ] **Step 1: Write report.py**

Create `py/server/report.py`:

```python
"""Renders bench/phase2a.md from the throughput-knee and cache-sensitivity
results — the two of the plan's four Phase 2 experiments a single server
instance can produce (sharding and hedging are sub-project B).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"


def find_knee(points: list[dict], duration_s: float) -> tuple[float | None, float | None]:
    """(knee_qps, sustained_qps): the offered rate where achieved throughput
    falls short *and* the run's wall time outlives the arrival window —
    Phase 0's validated criterion (bench/baselines.md), duplicated rather
    than imported from baselines/report.py to avoid touching that
    already-shipped Phase 0 report generator for an unrelated phase.
    """
    knee = None
    sustained = None
    for point in points:
        reps = [r for r in point["repeats"] if r["latency"].get("count", 0) > 0]
        if not reps:
            continue
        achieved = statistics.median([r["achieved_qps"] for r in reps])
        wall = statistics.median([r["wall_s"] for r in reps])
        if achieved < 0.95 * point["offered_qps"] and wall > 1.1 * duration_s:
            knee = point["offered_qps"]
            break
        sustained = point["offered_qps"]
    return knee, sustained


def render_markdown(knee_data: dict, cache_data: dict) -> str:
    lines = [
        "# Phase 2, sub-project A — single-node gRPC server",
        "",
        "Exit artifact for Phase 2 sub-project A (see "
        "`docs/superpowers/specs/2026-08-15-phase2-server-design.md`). "
        "`bench/results/server-throughput-knee.json` and "
        "`bench/results/server-cache-sensitivity.json` are committed; the "
        "built index is not (see README's Phase 1 setup). Sharding, the "
        "broker, and hedged requests are sub-project B.",
        "",
        "## Server configuration",
        "",
        f"{knee_data['server_workers']} worker threads, queue depth "
        f"{knee_data['queue_depth']}, {knee_data['cache_capacity']}-entry LRU "
        f"cache, `{knee_data['algorithm']}` (Phase 1 measured WAND faster than "
        "BlockMax-WAND in wall-clock on this corpus at k=10 despite scoring "
        "more postings — see `bench/phase1.md` — so it's the server default), "
        f"top-{knee_data['hits']}.",
        "",
        "## Throughput knee",
        "",
        f"Open-loop, Poisson arrivals, Zipfian query popularity (s="
        f"{knee_data['zipf_s']}) over {knee_data['num_distinct_queries']:,} dev "
        f"queries, {knee_data['client_workers']} client dispatch threads. Each "
        f"point is the median of {knee_data['repeats']} runs of "
        f"{knee_data['duration_s']:.0f}s.",
        "",
        "| Offered QPS | Achieved QPS | p50 | p95 | p99 | p99.9 | errors (median) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in knee_data["points"]:
        reps = [r for r in point["repeats"] if r["latency"].get("count", 0) > 0]
        if not reps:
            lines.append(f"| {point['offered_qps']:.0f} | no samples recorded |")
            continue
        mid = sorted(reps, key=lambda r: r["latency"]["p99_us"])[len(reps) // 2]
        lat = mid["latency"]
        errors = statistics.median([r["errors"] for r in reps])
        lines.append(
            f"| {point['offered_qps']:.0f} | {mid['achieved_qps']:.1f} | "
            f"{lat['p50_us']/1000:.2f}ms | {lat['p95_us']/1000:.2f}ms | "
            f"{lat['p99_us']/1000:.2f}ms | {lat['p999_us']/1000:.2f}ms | {errors:.0f} |"
        )

    knee, sustained = find_knee(knee_data["points"], knee_data["duration_s"])
    if sustained is not None:
        sustained_row = next(p for p in knee_data["points"] if p["offered_qps"] == sustained)
        sustained_p99 = statistics.median(
            [r["latency"]["p99_us"] for r in sustained_row["repeats"]]
        )
        cache_note = ""
        if cache_data["points"]:
            rates = [p["hit_rate"] for p in cache_data["points"]]
            cache_note = (
                f", {statistics.median(rates):.0%} median cache hit rate "
                "across the cache-sensitivity sweep"
            )
        lines += [
            "",
            f"**Capacity: sustains {sustained:.0f} QPS at p99 < "
            f"{sustained_p99/1000:.0f}ms** with {knee_data['server_workers']} "
            f"worker threads, queue depth {knee_data['queue_depth']}{cache_note}.",
        ]
        if knee is not None:
            lines.append(
                f"At {knee:.0f} QPS the server stops keeping up: achieved throughput "
                "falls below offered and the queue outlives the arrival window."
            )
    else:
        lines += [
            "",
            "No offered QPS in this sweep stayed under the knee — widen `--qps` "
            "with lower values and re-run.",
        ]

    lines += [
        "",
        "## Cache sensitivity",
        "",
        f"Fixed QPS ({cache_data['qps']}, chosen below the throughput knee above "
        "so this reflects cache behavior rather than queueing) across a sweep "
        "of the Zipfian skew. Cache-hit and cache-miss latency are reported "
        "separately — a blended number would hide which one actually matters.",
        "",
        "| Zipf s | hit rate | hit p50 | hit p99 | miss p50 | miss p99 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for point in cache_data["points"]:
        hit = point["hit_latency"]
        miss = point["miss_latency"]
        hit_p50 = f"{hit['p50_us']/1000:.2f}ms" if hit.get("count", 0) else "n/a"
        hit_p99 = f"{hit['p99_us']/1000:.2f}ms" if hit.get("count", 0) else "n/a"
        miss_p50 = f"{miss['p50_us']/1000:.2f}ms" if miss.get("count", 0) else "n/a"
        miss_p99 = f"{miss['p99_us']/1000:.2f}ms" if miss.get("count", 0) else "n/a"
        lines.append(
            f"| {point['zipf_s']:.1f} | {point['hit_rate']:.1%} | {hit_p50} | "
            f"{hit_p99} | {miss_p50} | {miss_p99} |"
        )

    lines += [
        "",
        "## Known limitations",
        "",
        "- Single server process, single machine. Sharding and its own",
        "  tail-latency amplification are sub-project B's subject, not this",
        "  document's.",
        "- The gRPC layer's own thread pool is bounded via `ResourceQuota`",
        "  (workers + queue depth) so it can't grow unbounded, but the fixed",
        "  worker pool and bounded queue are the actual concurrency control —",
        "  see the design spec for why.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    knee_path = RESULTS_DIR / "server-throughput-knee.json"
    cache_path = RESULTS_DIR / "server-cache-sensitivity.json"
    if not knee_path.exists():
        raise SystemExit(f"no {knee_path}; run `python -m server.throughput_knee` first")
    if not cache_path.exists():
        raise SystemExit(f"no {cache_path}; run `python -m server.cache_sensitivity` first")

    knee_data = json.loads(knee_path.read_text())
    cache_data = json.loads(cache_path.read_text())

    (REPO_ROOT / "bench" / "phase2a.md").write_text(render_markdown(knee_data, cache_data))
    print("wrote bench/phase2a.md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the real experiments (not the short smoke runs from Tasks 8-9) and generate the report**

```bash
cd /Users/tanhonjung/cascade-search-engine
export PYTHONPATH=py
uv run python -m server.throughput_knee
uv run python -m server.cache_sensitivity
uv run python -m server.report
```

This uses each script's real defaults (`--qps 20 40 60 80 100 120`,
`--duration 10`, `--repeats 3` for the knee sweep; `--zipf-s 0.5 1.0 1.5 2.0`
for cache sensitivity) — expect this to take several minutes total.

- [ ] **Step 3: Read bench/phase2a.md and sanity-check it**

Read the generated file. Confirm:
- The capacity statement names a real sustained QPS and a real p99 — if
  `sustained` came back `None` (every offered QPS in the default sweep
  already exceeded the knee, or none did), widen `--qps` on
  `server.throughput_knee` (lower values if everything looks saturated,
  higher if nothing shows a knee at all) and re-run Step 2.
- The cache-sensitivity table shows hit rate genuinely increasing as `s`
  increases from 0.5 to 2.0. If it's flat near 0% or flat near 100% across
  the whole sweep, `--cache-capacity` (default 200, both in
  `cache_sensitivity.py`'s own default and the server's) needs adjusting —
  smaller if hit rate is saturating near 100% regardless of skew, larger if
  it's staying near 0% even at high skew. Re-run
  `uv run python -m server.cache_sensitivity --cache-capacity <new value>`
  and `uv run python -m server.report` until the sweep shows real variation,
  and note the tuned value in the report's server configuration section
  (it's already templated to display whatever `cache_data['cache_capacity']`
  was actually used).

- [ ] **Step 4: Commit**

```bash
git add py/server/report.py bench/phase2a.md bench/results/server-throughput-knee.json bench/results/server-cache-sensitivity.json
git commit -m "phase2: add the report generator and the real bench/phase2a.md results"
```

---

## Task 11: README and final verification

**Files:**
- Modify: `README.md`

**Interfaces:** none — this task documents what Tasks 1-10 built.

- [ ] **Step 1: Update README.md**

Update the status table row:

```
| 2 — Serving, sharding, tail latency | capacity statement | sub-project A done |
```

Update the `Layout` code block, adding after the existing `cpp/tests/` line:

```
cpp/server/     gRPC service: bounded queue + worker pool + LRU cache
py/server/      server client, process manager, throughput/cache experiments
```

and after `bench/          baselines.md, phase1.md, plots, REPORT.md`, note
`phase2a.md` too (edit that line to read
`bench/          baselines.md, phase1.md, phase2a.md, plots, REPORT.md`).

Add a new section after "Running Phase 1" (before "## Ground rules"):

```markdown
## Running Phase 2 (sub-project A: single-node server)

Needs the Phase 1 index and `pkg-config` pointed at Anaconda's grpc/protobuf
(`brew install pkg-config` if you don't have it; the Makefile defaults
`PKG_CONFIG_PATH` to `/opt/anaconda3/lib/pkgconfig`, override it if your
grpc/protobuf live elsewhere).

```bash
export PYTHONPATH=py
uv run python -m server.gen_proto        # generates py/server/generated/*.py
make -C cpp all                          # builds cpp/build/server_bin
uv run python -m server.throughput_knee  # -> bench/results/server-throughput-knee.json
uv run python -m server.cache_sensitivity  # -> bench/results/server-cache-sensitivity.json
uv run python -m server.report           # -> bench/phase2a.md
```

The server itself (`cpp/build/server_bin <index_dir> [--port=P] [--workers=N]
[--queue-depth=D] [--cache-capacity=C] [--algorithm=wand|blockmax-wand|daat-or]`)
can also be run standalone for manual testing.
```

- [ ] **Step 2: Full clean-build verification**

```bash
cd /Users/tanhonjung/cascade-search-engine/cpp
make clean
make all
make test
```

Expected: clean build, `make test` prints `all checks passed` for all four
C++ test binaries, exits 0.

```bash
cd /Users/tanhonjung/cascade-search-engine
uv run pytest -v
```

Expected: every test in `py/tests` and `py/server/tests` passes (the
`py/server/tests` ones should now genuinely run, not skip, since the index
and generated stubs both exist on this machine at this point in the plan).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "phase2: document sub-project A setup and update the status table"
```

---

## Self-Review Notes

(Filled in during plan writing, not left for the executor to redo.)

1. **Spec coverage:** §2 architecture → Tasks 2-6. §3 BoundedQueue/WorkerPool
   → Tasks 2, 4. §4 LruCache + algorithm choice → Task 3, Task 5/6 (algorithm
   flag), Task 10 (report cites Phase 1's finding). §5 proto → Task 1 (with
   the noted `Result`→`SearchResult` rename). §6 experiments → Tasks 8-9. §7
   testing → Tasks 2-5, 7. §8 exit artifact → Task 10. §9 out of scope →
   correctly excluded (no sharding/hedging/broker code anywhere in this
   plan).
2. **Placeholder scan:** none found — every step has literal code or literal
   shell commands, no "add error handling" or "similar to Task N" phrasing.
3. **Type consistency:** `Algorithm`, `Index`, `Result`, `SearchStats` used
   identically to their existing declarations in `cpp/query/searcher.h`
   (checked against the actual file, not from memory). `SearchServiceImpl`'s
   constructor signature and `Query` override match between its Task 5
   declaration and every call site in Task 6 (`main.cc`) and Task 5's own
   test. `SearchClient`/`ServerProcess`'s constructor parameters match
   between their Task 7 definitions and every caller in Tasks 8-10.
