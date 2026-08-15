# Phase 2, sub-project A: single-node gRPC server, cache, throughput knee, cache sensitivity

Status: approved for implementation planning
Scope: the first of two Phase 2 sub-projects (see CASCADE_SEARCH_PLAN.md §Phase 2).
This one stops at a single server process — no sharding, no hedging, no broker.
Those are sub-project B, once this is built and its numbers are in.

## 1. Purpose

Phase 0/1 measured retrieval quality and single-query cost through Python calling
directly into the C++ index via pybind11 — no network, no concurrency control, no
serving layer. Phase 2 asks a different question: what happens when many
concurrent clients hit one process with a bounded amount of work it can do at
once? That requires an actual server with its own concurrency policy, which is
what this sub-project builds, plus the two of the plan's four Phase 2 experiments
that only need one server instance:

- **Throughput knee**: QPS on x-axis, p99 on y-axis; find where latency goes
  vertical. Capacity statement: "sustains X QPS at p99 < Yms with N=1, C%
  cache hit rate."
- **Cache sensitivity**: sweep the load generator's Zipf parameter, plot hit
  rate against p99; report cache-hit and cache-miss latency *separately* (a
  blended number hides which one actually matters).

Tail-latency-amplification-vs-shard-count and hedged requests need a broker
fanning out to multiple server processes — sub-project B, once this one's
server is running and its numbers are checked in.

## 2. Architecture

```
                    ┌──────────────────────────────┐
py client  ───RPC──▶│  grpc::Server (sync service)  │
(loadgen)           │      Search(query, k)         │
                    └───────────────┬────────────────┘
                                    │ enqueue-or-shed
                                    ▼
                    ┌──────────────────────────────┐
                    │  BoundedQueue<Request>         │  depth D (flag, default 64)
                    │  full → RESOURCE_EXHAUSTED      │
                    └───────────────┬────────────────┘
                                    │
                      N worker threads (flag, default 4)
                                    │
                    ┌───────────────▼────────────────┐
                    │  LruCache (normalized query)     │──hit──▶ respond
                    │       miss                       │
                    │        ▼                         │
                    │  cascade::Index::search()          │  WAND, k=10 (see §4)
                    └───────────────────────────────────┘
```

The gRPC layer's own sync-handler thread pool is bounded via
`grpc::ResourceQuota().SetMaxThreads(M)`, `M = worker_count + queue_depth`
(enough that a gRPC-dispatch thread is never itself the bottleneck: every
request that isn't immediately shed either occupies a worker or sits in the
queue, so that many gRPC threads can be blocked on a future at once without
growing further) so it can't grow unbounded under load, but it is not the
concurrency control the plan asks for. The `BoundedQueue` + fixed worker pool
is: this is the only place where "how much work is in flight" is decided,
and it's the thing the throughput-knee experiment measures directly.

## 3. Component: BoundedQueue + worker pool

`cpp/server/bounded_queue.h` — a small, self-contained, header-only class:

```cpp
template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(size_t capacity);
  // Returns false (does not block, does not enqueue) if already at capacity.
  bool try_push(T item);
  // Blocks until an item is available or shutdown() is called.
  std::optional<T> pop();
  void shutdown();  // wakes all blocked pop() calls, they return nullopt
  size_t size() const;
 private:
  mutable std::mutex mu_;
  std::condition_variable not_empty_;
  std::deque<T> items_;
  size_t capacity_;
  bool shutting_down_ = false;
};
```

`cpp/server/worker_pool.h` — `N` `std::thread`s, each looping `pop()` and
invoking a `std::function<void(T)>` handler; join on `shutdown()`.

The RPC handler:

```cpp
Status Search(ServerContext*, const SearchRequest* req, SearchResponse* resp) override {
  auto task = std::make_shared<Task>(req, resp);
  if (!queue_.try_push(task)) {
    return Status(StatusCode::RESOURCE_EXHAUSTED, "queue full");
  }
  task->done.get_future().wait();   // worker thread fulfills this promise
  return task->status;
}
```

`Task` carries a `std::promise<void>` the worker fulfills after populating
`resp`, and a `queued_at` timestamp captured before `try_push` so the worker
can compute `queue_wait_us` (queue delay = dequeued_at − queued_at) and put it
in the response — the harness needs this to separate "the server was slow" from
"the request sat in line," matching how `harness/loadgen.py` already separates
`service` from `queue_delay` client-side. Server-side `queue_wait_us` is a
second, corroborating measurement, not a replacement — the client-observed
number (scheduled → done) is still what gets reported as *the* latency, per
Phase 0's convention.

## 4. Component: LruCache

`cpp/server/lru_cache.h` — capacity in entries (flag, default 200; see §6 for
why entry count and default matter for the cache-sensitivity experiment).
Classic `unordered_map<key, list::iterator>` + intrusive `list<Entry>` for
O(1) get/put/evict. Key is the query string **normalized by lowercasing and
collapsing/trimming whitespace only** — not full tokenization. The plan says
"keyed on the *normalized* query string"; tokenizing before the cache lookup
would make the cache and the analyzer's stopword/stemming choices coupled in a
way that isn't necessary for what a query cache is for (skip repeated
identical-enough requests), and keeps `LruCache` a generic string-keyed
component with no dependency on `Analyzer`.

Cached value: the serialized `(docid, score)` result list. A cache hit skips
`Index::search()` entirely — no analyzer, no WAND, nothing — which is what
makes hit-latency and miss-latency worth reporting separately.

**Algorithm choice for the server: WAND, not BlockMax-WAND.** Phase 1's
query-cost table found BlockMax-WAND scores far fewer postings than plain WAND
on this corpus but is slower at *every* latency percentile — `mean_blocks_decoded`
being 49% higher than WAND's is the traced cause (see `bench/phase1.md`).
Postings-scored was the plan's Phase 1 metric; wall-clock is what actually
serves a request, so the server defaults to whichever one that measurement
says is faster. Configurable via flag for anyone who wants to re-run the
comparison at the server layer.

## 5. Proto

`proto/search.proto`:

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

message Result {
  uint32 docid = 1;
  float score = 2;
}

message SearchResponse {
  repeated Result results = 1;
  bool cache_hit = 2;
  uint32 queue_wait_us = 3;
}
```

`docid` here is the internal uint32 docid, not the external MS MARCO id — this
server is a latency/throughput testbed, not a quality re-evaluation (Phase 1
already established quality); external-id resolution is an unnecessary string
lookup for every result on every request.

## 6. Experiments

Both reuse `harness/loadgen.py` and `harness/histogram.py` completely
unchanged — the point of `run_open_loop(dispatch, ...)` taking a plain
`Callable[[str], object]` is that a gRPC stub call is a drop-in `dispatch`,
same shape as Phase 0's Tantivy `dispatch`.

**Throughput knee** (`py/server/throughput_knee.py`, mirrors
`baselines/latency.py`'s structure): sweep QPS points, 3 repeats each, find
the point where achieved QPS falls below offered *and* the run's wall time
exceeds the arrival window (Phase 0's already-validated knee criterion —
reused, not reinvented). Output: `bench/results/server-throughput-knee.json`,
capacity statement in the exit artifact.

**Cache sensitivity** (`py/server/cache_sensitivity.py`): sweep the Zipf `s`
parameter (e.g. 0.5, 1.0, 1.5, 2.0) at a fixed moderate QPS (below the knee,
so the number reflects cache behavior, not queueing). For each point, pull
`cache_hit` out of every response and split the `LatencyRecorder` into two —
one for hits, one for misses — then report hit rate and both percentile sets.
Cache capacity (200 entries, §4) needs to be small relative to the ~6,980
distinct dev queries the Zipf sampler draws from, or hit rate would trend to
~100% regardless of skew once the cache warms up; 200 is a starting point,
tuned after the first real run if the hit-rate spread across `s` values turns
out too flat or too saturated.

## 7. Testing

- `cpp/server/tests/test_bounded_queue.cc`: FIFO order, `try_push` returns
  false at capacity without blocking or losing state, concurrent
  producer/consumer stress test (multiple threads pushing/popping, assert no
  item lost or duplicated).
- `cpp/server/tests/test_lru_cache.cc`: hit/miss, eviction order (least
  recently *used*, not inserted — a `get` on an old entry must move it to the
  front), capacity boundary.
- `py/server/tests/test_server_integration.py`: Python integration test,
  since it exercises the real network path the experiments themselves use
  (a C++ gRPC test client would only prove the C++ side can talk to itself).
  Starts the server as a subprocess, makes one real gRPC call end-to-end,
  asserts a well-formed response; separately, starts a server with
  `queue_depth=1` and one worker, fires a concurrent burst, and asserts some
  calls come back `RESOURCE_EXHAUSTED` rather than all succeeding or hanging.
- No correctness re-testing of WAND/BMW itself — Phase 1 already proved that
  against exhaustive DAAT-OR on the real corpus; this layer only adds
  concurrency and caching on top of a search path already known correct.

## 8. Exit artifact

`bench/phase2a.md` (mirroring `bench/phase1.md`'s structure): throughput-knee
table + capacity statement, cache-sensitivity table (hit rate vs. `s`, hit vs.
miss latency at each point), server configuration (thread count, queue depth,
cache capacity, algorithm). Two of the plan's four required plots/artifacts;
the other two (tail-latency-vs-N, hedging) are sub-project B's exit bar.

## 9. Out of scope (this sub-project)

- Sharding / document partitioning, the broker, scatter-gather.
- Hedged requests (needs replicas, which needs the broker).
- TLS, auth, production deployment concerns — this is a benchmark harness,
  not a service anyone else's traffic hits.
- Dynamic thread pool sizing — fixed-size is what the plan asks for, and
  what makes the throughput knee interpretable (a pool that grows under load
  would blur "capacity" into "how aggressively did it scale up").
