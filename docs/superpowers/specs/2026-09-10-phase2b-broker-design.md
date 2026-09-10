# Phase 2, sub-project B: sharded broker, tail-latency amplification, hedged requests

Status: approved for implementation planning
Scope: the second of Phase 2's two sub-projects (see `CASCADE_SEARCH_PLAN.md`
§Phase 2 and `docs/superpowers/specs/2026-08-15-phase2-server-design.md` §9,
which explicitly deferred this work: "Tail-latency-amplification-vs-shard-count
and hedged requests need a broker fanning out to multiple server processes —
sub-project B, once this one's server is running and its numbers are
checked in."). Sub-project A's server (`cpp/server/`), proto, and Python
harness (`py/server/`) are the unmodified foundation this fans out to.

## 1. Purpose

Sub-project A answered "what happens when many concurrent clients hit *one*
process." Sub-project B asks the plan's other question: what happens to tail
latency when a query has to wait on the slowest of *N* processes, and can a
backup request fix it. Two of the plan's four Phase 2 experiments remain:

- **Tail-latency amplification**: measure single-shard p99, then N-shard
  fan-out p99 for N ∈ {4, 8, 16}. Plot p99 vs. N. The fan-out p99 is expected
  to be materially worse than the single-shard p99 because the broker's
  response waits on the slowest of N parallel calls — reproducing Dean &
  Barroso's "Tail at Scale" result directly, not citing it.
- **Hedged requests**: once a shard call exceeds that shard's own measured
  p95, fire a backup request to a replica of the same shard; take whichever
  of the two returns first. Report both the p99 improvement *and* the extra
  request load (plan's stated expectation: ~5%) — a hedging result without
  the load-cost number is not the artifact the plan asks for.

## 2. Architecture

```
py client (loadgen)  ───dispatch───▶  Broker (py/server/broker.py)
                                          │  ThreadPoolExecutor, N futures
                          ┌───────────────┼───────────────┬ ... ┐
                          ▼               ▼               ▼     ▼
                   shard 0 server   shard 1 server   shard 2   shard N-1
                   (server_bin,     (server_bin,      ...      (server_bin,
                    port 60000)      port 60001)                port 6000N-1)
                          │               │
                    [hedge only]    [hedge only]
                          ▼               ▼
                   shard 0 replica  shard 1 replica
                   (port 61000)      (port 61001)
```

Each shard server is an **unmodified** `server_bin` from sub-project A,
pointed at its own shard's index directory. The broker is Python, not C++.
Justification: sub-project A already delivered the C++ signal (gRPC service,
bounded queue, worker pool, LRU cache, all in C++); the broker does no
retrieval work of its own — it fans out real network calls to real OS
processes and merges responses, which is orchestration, not a performance-
critical path. Writing it in Python lets the fan-out use plain
`concurrent.futures.ThreadPoolExecutor` and keeps `harness/loadgen.py`
unchanged: `Broker.dispatch(query, k)` has the same
`Callable[[str], object]` shape as `SearchClient.dispatch` from sub-project A,
so the load generator cannot tell the difference between hitting one server
and hitting a broker fanning out to sixteen.

The **real tail-latency amplification this experiment measures is in the
shard servers and the network, not the broker** — each shard server has its
own process, its own `BoundedQueue`/`WorkerPool`/`LruCache` (sub-project A,
unmodified), and its own OS scheduling; the broker's own overhead (spawning
futures, computing max/argsort over ≤16 small responses) is sub-millisecond
and does not materially change which effect the plot shows.

## 3. Component: corpus partitioner and shard index builder

Document-partitioned sharding needs N index directories, each built from a
disjoint contiguous slice of the corpus — `cpp/index/builder.cc` takes
`<corpus.tsv> <index_dir> [max_docs]` and always starts from line 0, so
there is no existing doc-id-range or offset flag. Rather than modify the
C++ builder (out of scope — it is proven correct by Phase 1 and untouched by
every phase since), a new Python script splits the corpus file itself:

`py/server/partition_corpus.py`:
- Input: `data/msmarco-passage.tsv` (8.8M lines, `docid\tpassage_text`).
- For a given N, writes N files
  `data/shards/n{N}/shard{i}.tsv` (i = 0..N-1), splitting by **line count**
  into contiguous, non-overlapping, roughly-equal chunks (last shard absorbs
  the remainder). Document-partitioned, not term-partitioned, per the plan.
- Idempotent / skip-if-exists per N, the same convention Phase 3's subset
  builder already uses for its own expensive one-time artifacts.

Then, for each shard file, invoke the existing `build_index` binary
unmodified: `build_index data/shards/n{N}/shard{i}.tsv indexes/shards/n{N}/shard{i}`.
A thin driver (`py/server/build_shards.py`) loops i = 0..N-1 and shells out to
the binary, skipping shards whose index directory already has a complete
`index.terms` file (rebuild is ~O(minutes) per shard at this corpus size per
Phase 1's numbers, scaled down by N — still worth not repeating on every
re-run of a later task).

**Docid semantics change, and this is a real limitation, stated up front, not
discovered by a reviewer:** each shard's internal `docid` (0..shard_doc_count-1)
is local to that shard — shard 3's docid 100 and shard 7's docid 100 are
different passages, and each shard's BM25 IDF is computed from that shard's
own document frequencies, not the corpus-global df. Sub-project A's spec
already established that `docid` in the proto is "internal, not external...
this server is a latency/throughput testbed, not a quality re-evaluation"
(§5) — sub-project B inherits and extends that framing: the broker's merged
top-k is a genuine, real computation (real scores, real ranking across real
shard responses), but it is not a quality-equivalent substitute for Phase 1's
monolithic-index NDCG numbers, and this spec makes no NDCG claim for the
sharded configuration. If a future phase needs sharded quality, that is new
scope (global df sharing across shards, or a rerank pass) — not this one.

## 4. Component: Broker

`py/server/broker.py`:

```python
class Broker:
    def __init__(self, shard_addresses: list[str], timeout_s: float = 30.0): ...
    def dispatch(self, query: str, k: int = 10): ...
    def close(self) -> None: ...

class HedgedBroker(Broker):
    def __init__(self, shard_addresses: list[str], replica_addresses: list[str],
                 hedge_delay_s: float, timeout_s: float = 30.0): ...
```

`Broker.dispatch`: submits one `SearchClient.dispatch(query, k)` call per
shard address to a `ThreadPoolExecutor(max_workers=N)`, blocks on
`concurrent.futures.wait(..., return_when=ALL_COMPLETED)` (every shard must
answer — a shard covers a disjoint slice of the corpus, so unlike a
cache/replica lookup there is no "good enough" partial answer), merges each
shard's returned results by score, descending, and truncates to k. Records
per-shard latency (submit → future done) for hedging's p95 threshold and for
diagnosing which shard was the tail.

`HedgedBroker.dispatch`: for each shard, submits the primary call, then
schedules (via the executor, `threading.Timer`, or an equivalent delayed
submit) a backup call to that shard's replica if the primary has not
completed after `hedge_delay_s`. Takes whichever of the two completes first
via `concurrent.futures.wait(..., return_when=FIRST_COMPLETED)` per shard,
cancels/discards the other's result when it later arrives (gRPC calls
already in flight are not forcibly killed — `grpc`'s Python client has no
clean mid-call cancel that is safe to rely on here — the discarded response
is just not used; this matches "extra load" being the honest cost the plan
asks to report, not something to hide by cancelling it away). Counts every
backup call actually sent, for the ~5%-extra-load number.

`hedge_delay_s` is measured, not guessed: a short warm-up run against the
un-hedged broker at the same N and QPS produces each shard's own p95
service latency (client-observed, per sub-project A's convention); the
hedging experiment then uses the **max across shards' p95** as one fixed
delay (simpler than a per-shard delay table, and conservative — no shard
gets hedged before its own measured p95, so this cannot overstate the
technique's benefit).

## 5. Component: shard clusters for the experiments

`py/server/shard_cluster.py`: a context manager wrapping N (or N replica-pairs)
`ServerProcess` instances (sub-project A's, unmodified), one per shard index
directory, on consecutive ports starting from a caller-supplied base port.
`ShardCluster(index_dirs, base_port, **server_kwargs)` — enter starts every
process and waits for every one to become ready (reuses
`ServerProcess.__enter__`'s existing ready-wait, just N times, concurrently
via a thread pool so N startups don't serialize); exit tears all down.
For the hedging experiment, `ShardCluster` is instantiated twice at the same
N (once for primaries, once for replicas, disjoint port ranges) — the two
clusters point at **the same index directories** (a replica is a second
process serving an identical shard, not a second copy of the data on disk;
only the process is duplicated, not the index files).

Server flags for every shard process: `--workers=4 --queue-depth=64
--cache-capacity=200 --algorithm=wand`, matching sub-project A's defaults —
this experiment is about the network/scheduling effect of fan-out, not a
re-sweep of per-server tuning.

## 6. Experiments

Both reuse `harness/loadgen.py`/`harness/histogram.py` unchanged, exactly as
sub-project A's did — `Broker.dispatch` and `HedgedBroker.dispatch` are
drop-in `dispatch` callables.

**Tail-latency amplification** (`py/server/tail_latency.py`):
1. N=1 baseline: **re-measured through the broker path** (a single-shard
   `Broker` with one shard address), not reused from sub-project A's direct
   `SearchClient` number — sub-project A's throughput-knee ran the client
   straight against one server with no broker hop in between, so its number
   is not comparable to the N≥4 broker-mediated points. Re-measuring N=1
   through `Broker` costs one more run and keeps the whole plot's x-axis
   apples-to-apples (every point pays the same broker overhead).
2. N ∈ {4, 8, 16}: partition + build shards (§3), start a `ShardCluster`,
   run the open-loop load generator at a fixed moderate QPS (below every
   configuration's own throughput knee — this experiment is about tail
   amplification from fan-out width, not about re-finding each N's own
   capacity limit), record full latency distribution.
3. Output: `bench/results/server-tail-latency.json` (all four points, full
   percentile sets), `bench/plots/phase2b-tail-latency.png` (p99 vs. N, and
   for context p50 vs. N on the same axes — the plan's ask is specifically
   the p99 curve).

**Hedged requests** (`py/server/hedging.py`), at one representative N (N=8,
matching the plan's own worked example of an N=8 capacity statement):
1. Warm-up run, un-hedged, to measure each shard's p95 (§4).
2. Baseline run: `Broker` (no hedging) at the same fixed QPS as the N=8 point
   above.
3. Hedged run: `HedgedBroker` at the same QPS and hedge delay from step 1.
4. Report: p99 (baseline vs. hedged, and the delta), and
   `backup_calls_sent / total_shard_calls` as the extra-load percentage —
   both numbers in the same table row, per the plan's explicit "show...both."

## 7. Testing

- `py/server/tests/test_partition_corpus.py`: a small synthetic TSV (not the
  real 8.8M-line corpus) partitioned at a few N values — every input line
  appears in exactly one output shard, shard sizes are within one line of
  equal, concatenating shards in order reproduces the input.
- `py/server/tests/test_broker.py`: fake `SearchClient`-shaped stubs (no real
  gRPC/server_bin — this test is about `Broker`'s merge/wait logic, not the
  network); asserts merged top-k is the true top-k of the union of shard
  results, and that `dispatch` waits for every shard (a stub with an
  injected delay must still appear in the final merge, not be dropped for
  being slow, since every shard is required per §4).
- `py/server/tests/test_hedged_broker.py`: fake stubs where the "primary" for
  one shard is rigged to hang past `hedge_delay_s` and the "replica" answers
  immediately; asserts the hedged result comes from the replica and that
  exactly one backup call was counted; a second case where the primary
  answers before the delay elapses asserts zero backup calls (no hedge fired
  when it wasn't needed — this is what keeps the extra-load number honest
  at ~5%, not close to 100%).
- `py/server/tests/test_shard_cluster.py`: integration-level, mirrors
  sub-project A's `test_server_integration.py` convention — starts a
  `ShardCluster` of 2 tiny real shard indexes (built from a synthetic
  corpus in a tmp dir, not the real 8.8M corpus), makes one real dispatch
  through a real `Broker`, asserts a well-formed merged response; separately
  asserts `ShardCluster.__exit__` actually terminates every child process
  (no orphaned `server_bin`s left running after a test — checked via
  `psutil` or `process.poll()` per PID, not by trusting `terminate()` alone).
- No re-test of WAND/BMW correctness or of `BoundedQueue`/`LruCache` — all
  proven by Phase 1 and sub-project A; this layer only adds fan-out, merge,
  and hedging on top of components already known correct.

## 8. Exit artifact

`bench/phase2b.md`: the tail-latency-vs-N plot + table, the hedging table
(baseline vs. hedged p99, extra-load %), cluster configuration for each run
(N, replicas, ports, per-shard server flags, QPS, corpus/shard doc counts),
and an explicit callout of §3's docid-semantics limitation so a reader does
not mistake the sharded broker's merged results for a quality claim.
Combined with sub-project A's `bench/phase2a.md`, this completes Phase 2's
four-plot exit bar from `CASCADE_SEARCH_PLAN.md` §Phase 2, and the capacity
statement can now be extended: "...with N=8 shards, hedged requests cut p99
by [Z]% at a [W]% extra-request cost."

## 9. Resource constraints (checked before committing to N and R)

- Disk: partitioning does not duplicate the corpus (shard files sum to the
  original 3.0 GB); building indexes for N ∈ {4, 8, 16} sums to roughly one
  more monolithic index's worth of bytes each (~2.6 GB total across shards,
  per N — same documents, split, not duplicated), so three N configurations
  add ~7.8 GB. The hedging experiment's replica set duplicates only the
  N=8 shards' index files a second time (~1.3 GB, since indexes/ scales
  with shard doc count) for two processes to load. All well inside the
  62 GB free at spec-writing time (`df -h /`).
- Memory: N=16 means 16 concurrent `server_bin` processes; each shard's
  index is ~1/16 the monolithic 2.6 GB (~165 MB) plus fixed per-process
  overhead. The N=8-with-replicas hedging config is the high-water mark:
  16 processes at ~1/8 the monolithic index each (~325 MB) — budget this
  explicitly against the machine's RAM before running it, and reduce
  `--workers` per shard (already 4, could drop to 2) if the run thrashes.

## 10. Out of scope (this sub-project)

- Any change to `cpp/server/` or `cpp/index/builder.cc` — both proven and
  frozen; sharding is achieved entirely by feeding the existing, unmodified
  builder N separate corpus slices.
- Global (cross-shard) BM25 statistics, or any NDCG/quality claim for the
  sharded configuration — see §3.
- Rebalancing, shard failure/retry beyond what hedging already covers,
  consistent hashing, or any dynamic shard membership — this is a fixed,
  static N-way partition for one benchmark run, not a production sharding
  scheme.
- Cache sensitivity and throughput knee — already sub-project A's exit
  artifact; not repeated here at the sharded level.
