# Phase 2 Sub-Project B: Sharded Broker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python broker that fans a query out to N document-partitioned shard servers (sub-project A's unmodified `server_bin`, one process per shard), merges results, and — for the hedging experiment — races a backup call to a shard's replica once that shard's own measured p95 elapses. Produce `bench/phase2b.md` with the tail-latency-vs-N plot and the hedging table, completing Phase 2's four-plot exit bar.

**Architecture:** `partition_corpus.py` splits `data/msmarco-passage.tsv` into N contiguous TSV files; `build_shards.py` shells out to the existing `cpp/build/build_index` binary once per shard file (no C++ changes). `ShardCluster` starts/stops N `ServerProcess` instances concurrently. `Broker` fans a query out to every shard client via a thread pool, waits for all of them (every shard is required — no partial merge), and returns the score-sorted top-k. `HedgedBroker` extends this: after a fixed delay (the max of each shard's own measured p95 from a warm-up run), any shard whose primary hasn't answered gets a backup call to its replica; whichever answers first wins.

**Tech Stack:** Python 3.12, `grpcio` (already a dev dependency), `concurrent.futures`, reusing `harness/loadgen.py`/`harness/histogram.py`/`harness/datasets.py`/`harness/runmeta.py` and sub-project A's `server.client.SearchClient`/`server.process.ServerProcess` unchanged. `matplotlib` (already present via the `dense` extra) for the tail-latency plot.

**Spec:** `docs/superpowers/specs/2026-09-10-phase2b-broker-design.md`

## Global Constraints

- Broker is Python, not C++ — sub-project A already delivered the C++ signal; the broker does no retrieval work, only orchestration (spec §2).
- `cpp/server/` and `cpp/index/builder.cc` are **not modified**. Sharding is achieved entirely by feeding the existing, unmodified `build_index` binary N separate corpus-slice TSV files (spec §3).
- Corpus partitioning is document-partitioned: contiguous line-count chunks of `data/msmarco-passage.tsv`, not term-partitioned, not randomly sampled.
- Every shard's `docid` is local to that shard (different documents at the same numeric docid across shards); the sharded configuration makes no NDCG/quality claim (spec §3). This is a latency benchmark, extending sub-project A's own "docid is internal, not external" framing.
- `Broker.dispatch` waits for **every** shard (`ALL_COMPLETED`) — a shard covers a disjoint slice of the corpus, so there is no "good enough" partial answer (spec §4).
- `hedge_delay_s` is measured from a real warm-up run (max across shards' own client-observed p95), never a guessed constant (spec §4).
- The N=1 point in the tail-latency-vs-N experiment is measured **through `Broker`**, not reused from sub-project A's direct-`SearchClient` throughput-knee number — the two aren't comparable because sub-project A's number has no broker hop (spec §6).
- No mean latency anywhere (project-wide rule, `harness/histogram.py` already enforces this at the recorder level).
- Python modules run with `PYTHONPATH=py` (`uv run python -m server.foo`), matching every existing script's convention.
- Server flags for every shard process: `--workers=4 --queue-depth=64 --cache-capacity=200 --algorithm=wand` — sub-project A's defaults; this experiment is about the fan-out/hedging effect, not a re-sweep of per-server tuning (spec §5).
- Ports are disjoint across scripts to avoid collisions when tests and experiment drivers run in the same session: sub-project A already uses 50151-50153 (integration tests), 50161 (`throughput_knee.py` default), 50162 (`cache_sensitivity.py` default). This plan uses 50251-50252 (Task 5's integration test), 50300+ (`tail_latency.py` default `--base-port`), 50400+/50500+ (`hedging.py` default primary/replica base ports).

---

## File Structure

```
py/server/partition_corpus.py            new — corpus TSV -> N contiguous shard TSVs
py/server/tests/test_partition_corpus.py new

py/server/build_shards.py                new — shells out to build_index per shard, skip-if-built
py/server/tests/test_build_shards.py     new

py/server/broker.py                      new — Broker (Task 3) + HedgedBroker (Task 4)
py/server/tests/test_broker.py           new
py/server/tests/test_hedged_broker.py    new

py/server/shard_cluster.py               new — starts/stops N ServerProcess instances concurrently
py/server/tests/test_shard_cluster.py    new — real 2-shard integration test

py/server/tail_latency.py                new — experiment 1 driver (tail-latency vs. N)
py/server/hedging.py                     new — experiment 2 driver (hedged vs. baseline)

py/server/report_2b.py                   new — renders bench/phase2b.md + the tail-latency plot

README.md                                modified — "Running Phase 2 (sub-project B)" + status table
```

---

## Task 1: Corpus partitioner

**Files:**
- Create: `py/server/partition_corpus.py`
- Test: `py/server/tests/test_partition_corpus.py`

**Interfaces:**
- Produces (for Task 2): `partition(corpus_path: Path, n: int, shards_dir: Path = SHARDS_DIR) -> list[Path]` (writes/returns N shard TSV paths, idempotent), `shard_paths(n: int, shards_dir: Path = SHARDS_DIR) -> list[Path]`, module constants `CORPUS_PATH` (`data/msmarco-passage.tsv`), `SHARDS_DIR` (`data/shards`).

- [ ] **Step 1: Write the failing tests**

Create `py/server/tests/test_partition_corpus.py`:

```python
"""partition_corpus splits a corpus TSV into N contiguous, document-
partitioned shard files. These tests use a small synthetic corpus, never
the real 8.8M-line one -- that's exercised for real in Task 9."""

from __future__ import annotations

from server.partition_corpus import partition, shard_paths


def test_partition_covers_every_line_exactly_once(tmp_path):
    corpus = tmp_path / "corpus.tsv"
    lines = [f"{i}\tpassage number {i}\n" for i in range(23)]
    corpus.write_text("".join(lines))

    paths = partition(corpus, 4, shards_dir=tmp_path / "shards")
    assert len(paths) == 4

    seen: list[str] = []
    for path in paths:
        seen.extend(path.read_text().splitlines(keepends=True))
    assert seen == lines


def test_shard_sizes_differ_by_at_most_one_line(tmp_path):
    corpus = tmp_path / "corpus.tsv"
    corpus.write_text("".join(f"{i}\tx\n" for i in range(23)))

    paths = partition(corpus, 4, shards_dir=tmp_path / "shards")
    sizes = [len(p.read_text().splitlines()) for p in paths]
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == 23


def test_partition_is_idempotent(tmp_path):
    corpus = tmp_path / "corpus.tsv"
    corpus.write_text("0\ta\n1\tb\n2\tc\n3\td\n")
    shards_dir = tmp_path / "shards"

    first = partition(corpus, 2, shards_dir=shards_dir)
    first_contents = [p.read_text() for p in first]

    # Corrupt the corpus after the first call: a second call, seeing the
    # shard files already exist, must not touch them again.
    corpus.write_text("garbage")
    second = partition(corpus, 2, shards_dir=shards_dir)
    assert [p.read_text() for p in second] == first_contents


def test_shard_paths_names_files_by_index(tmp_path):
    paths = shard_paths(3, shards_dir=tmp_path)
    assert [p.name for p in paths] == ["shard0.tsv", "shard1.tsv", "shard2.tsv"]
    assert paths[0].parent.name == "n3"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd py && PYTHONPATH=. uv run --project .. pytest server/tests/test_partition_corpus.py -v`
Expected: FAIL / collection error — `server.partition_corpus` doesn't exist yet.

- [ ] **Step 3: Write `py/server/partition_corpus.py`**

```python
"""Splits the corpus TSV into N contiguous, document-partitioned shards.

cpp/index/builder.cc has no doc-id-range or offset flag -- it always reads
its input TSV from line 0 -- so sharding is done by writing N separate TSV
files rather than by changing the (proven, frozen) C++ builder. Splitting by
line count keeps each shard's underlying documents contiguous and disjoint,
matching the plan's "document-partitioned sharding," and concatenating the
N output files in shard order reproduces the input exactly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "msmarco-passage.tsv"
SHARDS_DIR = REPO_ROOT / "data" / "shards"


def shard_paths(n: int, shards_dir: Path = SHARDS_DIR) -> list[Path]:
    return [shards_dir / f"n{n}" / f"shard{i}.tsv" for i in range(n)]


def partition(corpus_path: Path, n: int, shards_dir: Path = SHARDS_DIR) -> list[Path]:
    """Writes n shard files under shards_dir/n{n}/, skipping if all already
    exist. Returns the n shard paths in shard order."""
    paths = shard_paths(n, shards_dir)
    if all(p.exists() for p in paths):
        return paths

    lines = corpus_path.read_text().splitlines(keepends=True)
    total = len(lines)
    base, extra = divmod(total, n)
    # The first `extra` shards get one extra line each, so sizes differ by
    # at most one line -- "roughly equal," not "equal except a giant last
    # shard absorbing the whole remainder."
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    start = 0
    for i, path in enumerate(paths):
        size = base + (1 if i < extra else 0)
        path.write_text("".join(lines[start:start + size]))
        start += size
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    args = parser.parse_args()
    for n in args.n:
        paths = partition(args.corpus, n)
        print(f"n={n}: {len(paths)} shards written under {paths[0].parent}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd py && PYTHONPATH=. uv run --project .. pytest server/tests/test_partition_corpus.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add py/server/partition_corpus.py py/server/tests/test_partition_corpus.py
git commit -m "phase2b: corpus partitioner for document-partitioned sharding"
```

---

## Task 2: Shard index builder driver

**Files:**
- Create: `py/server/build_shards.py`
- Test: `py/server/tests/test_build_shards.py`

**Interfaces:**
- Consumes: nothing from Task 1 directly (takes shard TSV paths as input; `main()` wires it to `partition_corpus.partition`).
- Produces (for Tasks 6, 7): `build_shards(shard_tsv_paths: list[Path], n: int, indexes_dir: Path = INDEXES_DIR, build_index_bin: Path = BUILD_INDEX_BIN) -> list[Path]`, `shard_index_dirs(n: int, indexes_dir: Path = INDEXES_DIR) -> list[Path]`, `is_built(index_dir: Path) -> bool`.

- [ ] **Step 1: Write the failing tests**

Create `py/server/tests/test_build_shards.py`:

```python
"""build_shards invokes cpp/build/build_index once per shard TSV, skipping
shards that already have a complete index (index.terms exists -- the
builder's last write, per cpp/index/builder.cc). The real binary is never
invoked here: subprocess.run is replaced by a small fake so this test runs
in milliseconds and doesn't need the C++ build."""

from __future__ import annotations

from pathlib import Path

import pytest

import server.build_shards as build_shards_module
from server.build_shards import build_shards, is_built


class _FakeSubprocess:
    """Stands in for the real `subprocess` module inside build_shards'
    namespace only -- monkeypatch.setattr(build_shards_module, "subprocess",
    ...) replaces the *name* build_shards.py looks up, not the process-wide
    subprocess module, so no other code in the test process is affected."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def run(self, args: list[str], check: bool) -> None:
        assert check is True
        self.calls.append(args)
        _, shard_tsv, index_dir = args
        Path(index_dir, "index.terms").write_bytes(b"")


def test_build_shards_invokes_binary_once_per_shard(tmp_path, monkeypatch):
    fake = _FakeSubprocess()
    monkeypatch.setattr(build_shards_module, "subprocess", fake)

    fake_bin = tmp_path / "build_index"
    fake_bin.write_text("#!/bin/sh\n")
    shard_tsvs = [tmp_path / f"shard{i}.tsv" for i in range(3)]
    for p in shard_tsvs:
        p.write_text("0\tx\n")

    index_dirs = build_shards(
        shard_tsvs, 3, indexes_dir=tmp_path / "idx", build_index_bin=fake_bin
    )

    assert len(fake.calls) == 3
    assert [c[1] for c in fake.calls] == [str(p) for p in shard_tsvs]
    assert all(d.exists() for d in index_dirs)


def test_build_shards_skips_already_built(tmp_path, monkeypatch):
    fake = _FakeSubprocess()
    monkeypatch.setattr(build_shards_module, "subprocess", fake)

    fake_bin = tmp_path / "build_index"
    fake_bin.write_text("#!/bin/sh\n")
    index_dir = tmp_path / "idx" / "n1" / "shard0"
    index_dir.mkdir(parents=True)
    (index_dir / "index.terms").write_bytes(b"")

    shard_tsv = tmp_path / "shard0.tsv"
    shard_tsv.write_text("0\tx\n")
    build_shards([shard_tsv], 1, indexes_dir=tmp_path / "idx", build_index_bin=fake_bin)

    assert fake.calls == []


def test_is_built_checks_index_terms(tmp_path):
    assert not is_built(tmp_path / "missing")
    (tmp_path / "index.terms").write_bytes(b"")
    assert is_built(tmp_path)


def test_build_shards_raises_if_binary_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_shards(
            [tmp_path / "shard0.tsv"], 1, indexes_dir=tmp_path / "idx",
            build_index_bin=tmp_path / "does-not-exist",
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd py && PYTHONPATH=. uv run --project .. pytest server/tests/test_build_shards.py -v`
Expected: FAIL / collection error — `server.build_shards` doesn't exist yet.

- [ ] **Step 3: Write `py/server/build_shards.py`**

```python
"""Builds one index per shard by invoking the existing, unmodified
cpp/build/build_index binary once per shard TSV -- sharding needs no
change to the C++ builder, only N separate inputs (see partition_corpus.py).
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_INDEX_BIN = REPO_ROOT / "cpp" / "build" / "build_index"
INDEXES_DIR = REPO_ROOT / "indexes" / "shards"


def shard_index_dirs(n: int, indexes_dir: Path = INDEXES_DIR) -> list[Path]:
    return [indexes_dir / f"n{n}" / f"shard{i}" for i in range(n)]


def is_built(index_dir: Path) -> bool:
    # index.terms is the builder's last write (cpp/index/builder.cc), so its
    # presence means a previous run finished rather than crashed partway.
    return (index_dir / "index.terms").exists()


def build_shards(
    shard_tsv_paths: list[Path],
    n: int,
    indexes_dir: Path = INDEXES_DIR,
    build_index_bin: Path = BUILD_INDEX_BIN,
) -> list[Path]:
    if not build_index_bin.exists():
        raise FileNotFoundError(f"{build_index_bin} not built; run `make -C cpp all` first")
    index_dirs = shard_index_dirs(n, indexes_dir)
    for shard_tsv, index_dir in zip(shard_tsv_paths, index_dirs):
        if is_built(index_dir):
            continue
        index_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(build_index_bin), str(shard_tsv), str(index_dir)], check=True)
    return index_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[4, 8, 16])
    args = parser.parse_args()
    from server.partition_corpus import CORPUS_PATH, partition

    for n in args.n:
        shard_tsvs = partition(CORPUS_PATH, n)
        index_dirs = build_shards(shard_tsvs, n)
        print(f"n={n}: {len(index_dirs)} shard indexes ready under {index_dirs[0].parent}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd py && PYTHONPATH=. uv run --project .. pytest server/tests/test_build_shards.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add py/server/build_shards.py py/server/tests/test_build_shards.py
git commit -m "phase2b: shard index builder driving the unmodified build_index binary"
```

---

## Task 3: Broker

**Files:**
- Create: `py/server/broker.py`
- Test: `py/server/tests/test_broker.py`

**Interfaces:**
- Consumes: `server.client.SearchClient` (existing; `.dispatch(query, k=10, timeout_s=30.0)` returning an object with `.results` (each with `.docid`, `.score`) and `.cache_hit`; `.close()`).
- Produces (for Tasks 4, 5, 6, 7): `class MergedResult(docid, score, shard_index)` (NamedTuple), `class MergedResponse` (`.results: list[MergedResult]`, `.cache_hit: bool`, always `False`), `class Broker`: `__init__(self, clients: list, timeout_s: float = 30.0)` (takes already-constructed client-like objects — dependency injection, so unit tests never need a real gRPC connection), classmethod `Broker.for_addresses(shard_addresses: list[str], timeout_s: float = 30.0) -> Broker` (the real-usage constructor, builds `SearchClient`s), `dispatch(self, query: str, k: int = 10) -> MergedResponse`, `close(self) -> None`. Internal attributes `self._clients`, `self._pool`, `self._timeout_s` are used by Task 4's subclass.

- [ ] **Step 1: Write the failing tests**

Create `py/server/tests/test_broker.py`:

```python
"""Broker.dispatch fans out to every client, waits for all of them
(ALL_COMPLETED -- no shard is optional), and merges by score. Fake
client-shaped stubs, not real gRPC/server_bin: this test is about the
merge/wait logic, not the network -- see Task 5 for the real end-to-end
integration test."""

from __future__ import annotations

import time

import pytest

from server.broker import Broker


class _FakeResult:
    def __init__(self, docid: int, score: float) -> None:
        self.docid = docid
        self.score = score


class _FakeResponse:
    def __init__(self, results: list[_FakeResult]) -> None:
        self.results = results


class _FakeClient:
    def __init__(self, results: list[_FakeResult], delay_s: float = 0.0) -> None:
        self._results = results
        self._delay_s = delay_s
        self.closed = False

    def dispatch(self, query: str, k: int = 10, timeout_s: float = 30.0):
        time.sleep(self._delay_s)
        return _FakeResponse(self._results[:k])

    def close(self) -> None:
        self.closed = True


def test_dispatch_merges_and_sorts_by_score_descending():
    client_a = _FakeClient([_FakeResult(1, 0.5), _FakeResult(2, 0.9)])
    client_b = _FakeClient([_FakeResult(3, 0.7)])
    broker = Broker([client_a, client_b])

    response = broker.dispatch("q", k=10)

    assert [r.docid for r in response.results] == [2, 3, 1]
    assert [r.score for r in response.results] == [0.9, 0.7, 0.5]
    broker.close()


def test_dispatch_truncates_to_k():
    client_a = _FakeClient([_FakeResult(1, 0.1), _FakeResult(2, 0.2)])
    client_b = _FakeClient([_FakeResult(3, 0.3), _FakeResult(4, 0.4)])
    broker = Broker([client_a, client_b])

    response = broker.dispatch("q", k=2)

    assert [r.docid for r in response.results] == [4, 3]
    broker.close()


def test_dispatch_waits_for_every_shard_even_a_slow_one():
    fast = _FakeClient([_FakeResult(1, 1.0)])
    slow = _FakeClient([_FakeResult(2, 2.0)], delay_s=0.2)
    broker = Broker([fast, slow])

    response = broker.dispatch("q", k=10)

    assert {r.docid for r in response.results} == {1, 2}
    broker.close()


def test_close_closes_every_client():
    client_a = _FakeClient([])
    client_b = _FakeClient([])
    broker = Broker([client_a, client_b])
    broker.close()
    assert client_a.closed
    assert client_b.closed


def test_constructor_rejects_empty_client_list():
    with pytest.raises(ValueError):
        Broker([])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd py && PYTHONPATH=. uv run --project .. pytest server/tests/test_broker.py -v`
Expected: FAIL / collection error — `server.broker` doesn't exist yet.

- [ ] **Step 3: Write `py/server/broker.py`**

```python
"""Fans a query out to every shard, waits for all of them, merges results
by score. Every shard covers a disjoint slice of the corpus, so unlike a
cache/replica lookup there's no answer without all N -- see
docs/superpowers/specs/2026-09-10-phase2b-broker-design.md §4.
"""

from __future__ import annotations

from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from typing import NamedTuple

from server.client import SearchClient


class MergedResult(NamedTuple):
    docid: int
    score: float
    shard_index: int


class MergedResponse:
    """Duck-types just enough of search_pb2.SearchResponse for
    harness.loadgen callers and experiment code: .results (score-sorted,
    truncated to k) and .cache_hit (always False -- the broker has no
    cache of its own, only the shard servers do, and a merged response
    spans shards so "was this a hit" isn't a single bool anymore; see
    design spec §4)."""

    def __init__(self, results: list[MergedResult]) -> None:
        self.results = results
        self.cache_hit = False


class Broker:
    def __init__(self, clients: list, timeout_s: float = 30.0) -> None:
        if not clients:
            raise ValueError("Broker needs at least one client")
        self._clients = clients
        self._timeout_s = timeout_s
        self._pool = ThreadPoolExecutor(max_workers=len(clients))

    @classmethod
    def for_addresses(cls, shard_addresses: list[str], timeout_s: float = 30.0) -> "Broker":
        return cls([SearchClient(addr) for addr in shard_addresses], timeout_s=timeout_s)

    def dispatch(self, query: str, k: int = 10) -> MergedResponse:
        futures = {
            self._pool.submit(client.dispatch, query, k=k, timeout_s=self._timeout_s): i
            for i, client in enumerate(self._clients)
        }
        wait(futures, return_when=ALL_COMPLETED)
        merged: list[MergedResult] = []
        for future, shard_index in futures.items():
            response = future.result()
            merged.extend(
                MergedResult(r.docid, r.score, shard_index) for r in response.results
            )
        merged.sort(key=lambda r: r.score, reverse=True)
        return MergedResponse(merged[:k])

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        for client in self._clients:
            client.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd py && PYTHONPATH=. uv run --project .. pytest server/tests/test_broker.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add py/server/broker.py py/server/tests/test_broker.py
git commit -m "phase2b: Broker fans out to every shard and merges by score"
```

---

## Task 4: HedgedBroker

**Files:**
- Modify: `py/server/broker.py` (add `HedgedBroker`)
- Create: `py/server/tests/test_hedged_broker.py`

**Interfaces:**
- Consumes: `Broker` (Task 3) — `self._clients`, `self._pool`, `self._timeout_s`.
- Produces (for Tasks 6, 7): `class HedgedBroker(Broker)`: `__init__(self, clients: list, replica_clients: list, hedge_delay_s: float, timeout_s: float = 30.0)`, classmethod `HedgedBroker.for_addresses(shard_addresses, replica_addresses, hedge_delay_s, timeout_s=30.0) -> HedgedBroker`, `dispatch(self, query, k=10) -> MergedResponse` (overrides `Broker.dispatch`), `close(self) -> None`, public counters `self.backup_calls_sent: int`, `self.total_shard_calls: int`.

- [ ] **Step 1: Write the failing tests**

Create `py/server/tests/test_hedged_broker.py`:

```python
"""HedgedBroker fires a backup call to a shard's replica only if that
shard's primary hasn't answered by hedge_delay_s, and takes whichever
answers first. Deterministic via small real sleeps in fake clients (same
technique test_server_integration.py already uses for timing-sensitive
concurrency), not real gRPC."""

from __future__ import annotations

import time

import pytest

from server.broker import HedgedBroker


class _FakeResult:
    def __init__(self, docid: int, score: float) -> None:
        self.docid = docid
        self.score = score


class _FakeResponse:
    def __init__(self, results: list[_FakeResult]) -> None:
        self.results = results


class _FakeClient:
    def __init__(self, docid: int, score: float, delay_s: float = 0.0) -> None:
        self._result = _FakeResult(docid, score)
        self._delay_s = delay_s
        self.calls = 0

    def dispatch(self, query: str, k: int = 10, timeout_s: float = 30.0):
        self.calls += 1
        time.sleep(self._delay_s)
        return _FakeResponse([self._result])

    def close(self) -> None:
        pass


def test_hedged_call_uses_replica_when_primary_is_slow():
    primary = _FakeClient(docid=1, score=0.5, delay_s=0.5)
    replica = _FakeClient(docid=2, score=0.9, delay_s=0.0)
    broker = HedgedBroker([primary], [replica], hedge_delay_s=0.05)

    response = broker.dispatch("q", k=10)

    assert [r.docid for r in response.results] == [2]
    assert broker.backup_calls_sent == 1
    assert broker.total_shard_calls == 1
    broker.close()


def test_no_backup_sent_when_primary_answers_within_delay():
    primary = _FakeClient(docid=1, score=0.5, delay_s=0.0)
    replica = _FakeClient(docid=2, score=0.9, delay_s=0.0)
    broker = HedgedBroker([primary], [replica], hedge_delay_s=0.2)

    response = broker.dispatch("q", k=10)

    assert [r.docid for r in response.results] == [1]
    assert broker.backup_calls_sent == 0
    assert replica.calls == 0
    broker.close()


def test_constructor_requires_matching_client_and_replica_counts():
    with pytest.raises(ValueError):
        HedgedBroker([_FakeClient(1, 0.1)], [], hedge_delay_s=0.1)


def test_multi_shard_hedging_is_independent_per_shard():
    # Shard 0's primary is slow (hedge fires); shard 1's primary is fast
    # (no hedge) -- proves the not_done set from one wait() call is applied
    # correctly per shard, not globally.
    primary0 = _FakeClient(docid=10, score=1.0, delay_s=0.5)
    replica0 = _FakeClient(docid=11, score=2.0, delay_s=0.0)
    primary1 = _FakeClient(docid=20, score=3.0, delay_s=0.0)
    replica1 = _FakeClient(docid=21, score=4.0, delay_s=0.0)
    broker = HedgedBroker([primary0, primary1], [replica0, replica1], hedge_delay_s=0.05)

    response = broker.dispatch("q", k=10)

    assert {r.docid for r in response.results} == {11, 20}
    assert broker.backup_calls_sent == 1
    assert replica1.calls == 0
    broker.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd py && PYTHONPATH=. uv run --project .. pytest server/tests/test_hedged_broker.py -v`
Expected: FAIL / collection error — `HedgedBroker` doesn't exist yet.

- [ ] **Step 3: Append `HedgedBroker` to `py/server/broker.py`**

Add to the end of `py/server/broker.py`:

```python
from concurrent.futures import FIRST_COMPLETED  # noqa: E402 (grouped with the class that uses it)


class HedgedBroker(Broker):
    """See design spec §4: after hedge_delay_s, any shard whose primary
    hasn't answered gets a backup call to its replica; whichever of the
    two answers first wins. hedge_delay_s is one fixed value (the max
    across shards' own measured p95 from a warm-up run — see hedging.py),
    not a per-shard table -- simpler, and conservative (no shard is hedged
    before its own measured p95, so this can't overstate the technique's
    benefit)."""

    def __init__(
        self, clients: list, replica_clients: list, hedge_delay_s: float,
        timeout_s: float = 30.0,
    ) -> None:
        if len(clients) != len(replica_clients):
            raise ValueError("need one replica client per shard")
        super().__init__(clients, timeout_s=timeout_s)
        self._replica_clients = replica_clients
        # Sized to N, not 2N: self._pool (inherited) already handles the N
        # concurrent primaries; this pool only ever runs backup calls, of
        # which there are at most N.
        self._backup_pool = ThreadPoolExecutor(max_workers=len(clients))
        self._hedge_delay_s = hedge_delay_s
        self.backup_calls_sent = 0
        self.total_shard_calls = 0

    @classmethod
    def for_addresses(
        cls, shard_addresses: list[str], replica_addresses: list[str],
        hedge_delay_s: float, timeout_s: float = 30.0,
    ) -> "HedgedBroker":
        return cls(
            [SearchClient(addr) for addr in shard_addresses],
            [SearchClient(addr) for addr in replica_addresses],
            hedge_delay_s=hedge_delay_s, timeout_s=timeout_s,
        )

    def dispatch(self, query: str, k: int = 10) -> MergedResponse:
        n = len(self._clients)
        self.total_shard_calls += n
        primaries = [
            self._pool.submit(client.dispatch, query, k=k, timeout_s=self._timeout_s)
            for client in self._clients
        ]
        # ALL_COMPLETED with a timeout: wait up to hedge_delay_s for every
        # primary. Whatever's left in not_done is exactly the set that
        # exceeded the delay -- FIRST_COMPLETED here would return as soon
        # as the *fastest* shard answers, which is not what "once a shard
        # exceeds its own p95" means.
        _, not_done = wait(primaries, timeout=self._hedge_delay_s, return_when=ALL_COMPLETED)

        results: list = [None] * n
        for i, primary in enumerate(primaries):
            if primary in not_done:
                backup = self._backup_pool.submit(
                    self._replica_clients[i].dispatch, query, k=k, timeout_s=self._timeout_s
                )
                self.backup_calls_sent += 1
                done, _ = wait([primary, backup], return_when=FIRST_COMPLETED)
                results[i] = next(iter(done)).result()
            else:
                results[i] = primary.result()

        merged: list[MergedResult] = []
        for shard_index, response in enumerate(results):
            merged.extend(
                MergedResult(r.docid, r.score, shard_index) for r in response.results
            )
        merged.sort(key=lambda r: r.score, reverse=True)
        return MergedResponse(merged[:k])

    def close(self) -> None:
        self._backup_pool.shutdown(wait=False)
        for client in self._replica_clients:
            client.close()
        super().close()
```

Move the `from concurrent.futures import FIRST_COMPLETED` into the existing top-of-file import instead of a second import line — the final file should read:

```python
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, ThreadPoolExecutor, wait
```

as its one `concurrent.futures` import, with the inline `# noqa` comment above deleted.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd py && PYTHONPATH=. uv run --project .. pytest server/tests/test_hedged_broker.py server/tests/test_broker.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add py/server/broker.py py/server/tests/test_hedged_broker.py
git commit -m "phase2b: HedgedBroker races a backup call once a shard exceeds its p95"
```

---

## Task 5: ShardCluster

**Files:**
- Create: `py/server/shard_cluster.py`
- Test: `py/server/tests/test_shard_cluster.py`

**Interfaces:**
- Consumes: `server.process.ServerProcess` (existing), `server.broker.Broker.for_addresses` (Task 3), `cpp/build/build_index` and `cpp/build/server_bin` binaries (built already for sub-project A).
- Produces (for Tasks 6, 7): `class ShardCluster`: `__init__(self, index_dirs: list[Path], base_port: int, **server_kwargs)`, context manager (`__enter__`/`__exit__`), `.addresses -> list[str]` (index-aligned with `index_dirs`).

- [ ] **Step 1: Write the failing test**

Create `py/server/tests/test_shard_cluster.py`:

```python
"""Integration test: build two tiny real shard indexes from a synthetic
corpus, start a real ShardCluster, dispatch one real query through a real
Broker end to end. Mirrors test_server_integration.py's skip-if-absent
convention -- skips (not a collection failure) if cpp/build/build_index,
server_bin, or the generated proto stubs aren't built."""

from __future__ import annotations

import subprocess
from pathlib import Path

import grpc
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD_INDEX_BIN = REPO_ROOT / "cpp" / "build" / "build_index"
SERVER_BIN = REPO_ROOT / "cpp" / "build" / "server_bin"
GENERATED_DIR = Path(__file__).resolve().parents[1] / "generated"
_READY = (
    BUILD_INDEX_BIN.exists()
    and SERVER_BIN.exists()
    and (GENERATED_DIR / "search_pb2.py").exists()
)

pytestmark = pytest.mark.skipif(
    not _READY,
    reason="needs cpp/build/build_index, server_bin, and generated proto stubs "
    "(see README's Phase 1 and Phase 2 setup)",
)

if _READY:
    from server.broker import Broker
    from server.shard_cluster import ShardCluster


def _build_shard(tmp_path: Path, name: str, docs: list[tuple[str, str]]) -> Path:
    corpus = tmp_path / f"{name}.tsv"
    corpus.write_text("".join(f"{docid}\t{text}\n" for docid, text in docs))
    index_dir = tmp_path / name
    subprocess.run([str(BUILD_INDEX_BIN), str(corpus), str(index_dir)], check=True)
    return index_dir


def test_broker_merges_across_two_real_shards(tmp_path):
    shard0 = _build_shard(tmp_path, "shard0", [
        ("1", "bank teller job description"),
        ("2", "river bank erosion"),
    ])
    shard1 = _build_shard(tmp_path, "shard1", [
        ("3", "teller machine withdrawal"),
        ("4", "unrelated passage about weather"),
    ])

    with ShardCluster([shard0, shard1], base_port=50251) as cluster:
        addresses = cluster.addresses
        broker = Broker.for_addresses(addresses)
        response = broker.dispatch("bank teller", k=10)
        assert 0 < len(response.results) <= 10
        scores = [r.score for r in response.results]
        assert scores == sorted(scores, reverse=True)
        broker.close()

    # Processes are down: connecting to either address now fails fast
    # rather than hanging or succeeding.
    for address in addresses:
        channel = grpc.insecure_channel(address)
        with pytest.raises(grpc.FutureTimeoutError):
            grpc.channel_ready_future(channel).result(timeout=1.0)
        channel.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd py && PYTHONPATH=. uv run --project .. pytest server/tests/test_shard_cluster.py -v`
Expected: FAIL / collection error — `server.shard_cluster` doesn't exist yet (or skipped if `cpp/build/` binaries aren't present in this environment; in that case this step's "failure" is the import error you'd see once `_READY` is forced true locally — proceed to Step 3 regardless).

- [ ] **Step 3: Write `py/server/shard_cluster.py`**

```python
"""Context manager for N ServerProcess instances -- one per shard index
directory, all started concurrently so N startups don't serialize. Used
both for a plain N-shard cluster (tail_latency.py) and, called twice at
disjoint port ranges over the *same* index directories, for a
primary/replica pair (hedging.py) -- see design spec §5.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from server.process import ServerProcess


class ShardCluster:
    def __init__(self, index_dirs: list[Path], base_port: int, **server_kwargs) -> None:
        self._index_dirs = index_dirs
        self._base_port = base_port
        self._server_kwargs = server_kwargs
        self._servers: list[ServerProcess] = []

    @property
    def addresses(self) -> list[str]:
        return [s.address for s in self._servers]

    def __enter__(self) -> "ShardCluster":
        servers = [
            ServerProcess(index_dir, port=self._base_port + i, **self._server_kwargs)
            for i, index_dir in enumerate(self._index_dirs)
        ]
        started: list[ServerProcess] = []
        failure: Exception | None = None
        with ThreadPoolExecutor(max_workers=len(servers)) as pool:
            futures = [pool.submit(s.__enter__) for s in servers]
            for future in futures:
                try:
                    started.append(future.result())
                except Exception as exc:  # noqa: BLE001 - still must tear down the rest below
                    failure = exc
        if failure is not None:
            # A partial cluster is not a usable cluster: tear down whatever
            # did start before propagating, so a failed __enter__ never
            # leaks running processes.
            for server in started:
                server.__exit__(None, None, None)
            raise failure
        self._servers = started
        return self

    def __exit__(self, *exc_info) -> None:
        with ThreadPoolExecutor(max_workers=max(len(self._servers), 1)) as pool:
            list(pool.map(lambda s: s.__exit__(*exc_info), self._servers))
        self._servers = []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd py && PYTHONPATH=. uv run --project .. pytest server/tests/test_shard_cluster.py -v`
Expected: 1 passed (or 1 skipped, if `cpp/build/build_index`/`server_bin`/generated stubs are not present in this environment — build them per README's Phase 1/Phase 2 setup, then re-run to confirm a real pass before moving on).

- [ ] **Step 5: Commit**

```bash
git add py/server/shard_cluster.py py/server/tests/test_shard_cluster.py
git commit -m "phase2b: ShardCluster starts/stops N shard servers concurrently"
```

---

## Task 6: Tail-latency-amplification experiment driver

**Files:**
- Create: `py/server/tail_latency.py`

**Interfaces:**
- Consumes: `harness.datasets.load_queries`, `harness.loadgen.run_open_loop`, `harness.runmeta.run_metadata` (existing), `server.broker.Broker.for_addresses` (Task 3), `server.build_shards.build_shards` (Task 2), `server.partition_corpus.{CORPUS_PATH, partition}` (Task 1), `server.shard_cluster.ShardCluster` (Task 5).
- Produces (for Task 8): `bench/results/server-tail-latency.json` with top-level `points: list[{n, repeats, p99_us_across_repeats}]` plus the run's own config/provenance — same shape family as sub-project A's `server-throughput-knee.json`.

No unit test file for this task — it is an experiment driver against real shard servers, in the same category as sub-project A's `throughput_knee.py`/`cache_sensitivity.py` (neither of which has a test file either); it's verified by actually running it (Task 9) and inspecting the output.

- [ ] **Step 1: Write `py/server/tail_latency.py`**

```python
"""Tail-latency-amplification experiment: single-shard p99 (through the
broker, so every point pays the same broker overhead -- see design spec
§6) vs. N-shard fan-out p99, for N in {1, 4, 8, 16}. Mirrors
throughput_knee.py's structure: same open-loop generator, same dev-query
Zipfian sampler, same repeats-per-point protocol.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from harness.datasets import load_queries
from harness.loadgen import run_open_loop
from harness.runmeta import run_metadata
from server.broker import Broker
from server.build_shards import build_shards
from server.partition_corpus import CORPUS_PATH, partition
from server.shard_cluster import ShardCluster

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
SINGLE_INDEX_DIR = REPO_ROOT / "indexes" / "cascade-msmarco-passage"


def measure(broker: Broker, qps: float, duration_s: float, workers: int,
            hits: int, queries: list[str], zipf_s: float, seed: int) -> dict:
    def dispatch(query: str) -> None:
        broker.dispatch(query, k=hits)

    result = run_open_loop(
        dispatch=dispatch, queries=queries, qps=qps, duration_s=duration_s,
        workers=workers, seed=seed, zipf_s=zipf_s,
    )
    return result.summary()


def run_point(n: int, index_dirs: list[Path], base_port: int, args, queries: list[str]) -> dict:
    with ShardCluster(
        index_dirs, base_port=base_port, workers=args.server_workers,
        queue_depth=args.queue_depth, cache_capacity=args.cache_capacity,
        algorithm=args.algorithm,
    ) as cluster:
        broker = Broker.for_addresses(cluster.addresses)
        repeats = [
            measure(broker, args.qps, args.duration, args.client_workers, args.hits,
                    queries, args.zipf_s, seed)
            for seed in range(args.repeats)
        ]
        broker.close()
    p99s = [r["latency"]["p99_us"] for r in repeats]
    return {
        "n": n,
        "repeats": repeats,
        "p99_us_across_repeats": {
            "min": min(p99s), "median": statistics.median(p99s), "max": max(p99s),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--qps", type=float, default=20.0,
                         help="fixed, moderate QPS below every configuration's own knee")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--client-workers", type=int, default=32)
    parser.add_argument("--hits", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--zipf-s", type=float, default=1.0)
    parser.add_argument("--server-workers", type=int, default=4)
    parser.add_argument("--queue-depth", type=int, default=64)
    parser.add_argument("--cache-capacity", type=int, default=200)
    parser.add_argument("--algorithm", default="wand")
    parser.add_argument("--base-port", type=int, default=50300)
    args = parser.parse_args()

    queries = [text for _, text in sorted(load_queries("dev").items())]

    points = []
    for n in args.n:
        if n == 1:
            index_dirs = [SINGLE_INDEX_DIR]
        else:
            shard_tsvs = partition(CORPUS_PATH, n)
            index_dirs = build_shards(shard_tsvs, n)
        point = run_point(n, index_dirs, args.base_port, args, queries)
        points.append(point)
        p99 = point["p99_us_across_repeats"]["median"]
        print(f"N={n:>3}   p99={p99/1000:7.2f}ms")

    output = {
        "server": "cascade-grpc-broker",
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
        "qps": args.qps,
        "duration_s": args.duration,
        "repeats": args.repeats,
        "points": points,
        "provenance": run_metadata(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "server-tail-latency.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nwrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity-check on a tiny local slice (no full corpus needed)**

Run: `cd py && PYTHONPATH=. uv run --project .. python -m server.tail_latency --n 1 4 --qps 5 --duration 3 --repeats 1`

Expected: prints two `N=  1   p99=...ms` / `N=  4   p99=...ms` lines and writes `bench/results/server-tail-latency.json`. This partitions/builds real N=4 shards from the real corpus (takes a few minutes the first time; cached after via `is_built`/`partition`'s idempotence) and needs `SINGLE_INDEX_DIR` (Phase 1's index) to already exist. The full N sweep and real repeat/duration values run in Task 9.

- [ ] **Step 3: Commit**

```bash
git add py/server/tail_latency.py
git commit -m "phase2b: tail-latency-vs-shard-count experiment driver"
```

---

## Task 7: Hedged-requests experiment driver

**Files:**
- Create: `py/server/hedging.py`

**Interfaces:**
- Consumes: `harness.datasets.load_queries`, `harness.histogram.LatencyRecorder`, `harness.loadgen.run_open_loop`, `harness.runmeta.run_metadata`, `server.client.SearchClient` (existing), `server.broker.{Broker, HedgedBroker}` (Tasks 3, 4), `server.build_shards.build_shards` (Task 2), `server.partition_corpus.{CORPUS_PATH, partition}` (Task 1), `server.shard_cluster.ShardCluster` (Task 5).
- Produces (for Task 8): `bench/results/server-hedging.json` with `baseline_p99_us_median`, `hedged_p99_us_median`, `extra_load_fraction`, `shard_p95_us_warmup`, `hedge_delay_us`, plus `baseline_repeats`/`hedged_repeats` and config/provenance.

No unit test file — same experiment-driver category as Task 6.

- [ ] **Step 1: Write `py/server/hedging.py`**

```python
"""Hedged-requests experiment, at one representative N (default 8, matching
the plan's own worked capacity-statement example): warm-up to measure each
shard's own p95, then baseline (no hedging) vs. hedged runs at the same
fixed QPS, reporting both the p99 delta and the extra-request cost -- see
design spec §6.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from harness.datasets import load_queries
from harness.histogram import LatencyRecorder
from harness.loadgen import run_open_loop
from harness.runmeta import run_metadata
from server.broker import Broker, HedgedBroker
from server.build_shards import build_shards
from server.client import SearchClient
from server.partition_corpus import CORPUS_PATH, partition
from server.shard_cluster import ShardCluster

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"


def measure_shard_p95(client: SearchClient, qps: float, duration_s: float, workers: int,
                       hits: int, queries: list[str], zipf_s: float, seed: int) -> float:
    recorder = LatencyRecorder()

    def dispatch(query: str) -> None:
        started = time.perf_counter_ns()
        client.dispatch(query, k=hits)
        recorder.record((time.perf_counter_ns() - started) / 1000)

    run_open_loop(dispatch=dispatch, queries=queries, qps=qps, duration_s=duration_s,
                   workers=workers, seed=seed, zipf_s=zipf_s)
    return recorder.percentile(95)


def measure(broker, qps: float, duration_s: float, workers: int, hits: int,
            queries: list[str], zipf_s: float, seed: int) -> dict:
    def dispatch(query: str) -> None:
        broker.dispatch(query, k=hits)

    result = run_open_loop(dispatch=dispatch, queries=queries, qps=qps, duration_s=duration_s,
                            workers=workers, seed=seed, zipf_s=zipf_s)
    return result.summary()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--qps", type=float, default=20.0)
    parser.add_argument("--warmup-duration", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--client-workers", type=int, default=32)
    parser.add_argument("--hits", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--zipf-s", type=float, default=1.0)
    parser.add_argument("--server-workers", type=int, default=4)
    parser.add_argument("--queue-depth", type=int, default=64)
    parser.add_argument("--cache-capacity", type=int, default=200)
    parser.add_argument("--algorithm", default="wand")
    parser.add_argument("--base-port", type=int, default=50400)
    parser.add_argument("--replica-base-port", type=int, default=50500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    queries = [text for _, text in sorted(load_queries("dev").items())]
    shard_tsvs = partition(CORPUS_PATH, args.n)
    index_dirs = build_shards(shard_tsvs, args.n)
    server_kwargs = dict(
        workers=args.server_workers, queue_depth=args.queue_depth,
        cache_capacity=args.cache_capacity, algorithm=args.algorithm,
    )

    # 1. Warm-up: measure each shard's own p95 service latency,
    #    client-side, one shard at a time -- Broker doesn't expose
    #    per-shard timing on its own merged view, so this wraps each
    #    shard's SearchClient individually instead.
    with ShardCluster(index_dirs, base_port=args.base_port, **server_kwargs) as cluster:
        shard_p95s = []
        for address in cluster.addresses:
            client = SearchClient(address)
            shard_p95s.append(
                measure_shard_p95(client, args.qps, args.warmup_duration, args.client_workers,
                                   args.hits, queries, args.zipf_s, args.seed)
            )
            client.close()
    hedge_delay_us = max(shard_p95s)
    print(f"warm-up shard p95s (us): {[round(p) for p in shard_p95s]}, "
          f"hedge_delay={hedge_delay_us/1000:.2f}ms")

    # 2. Baseline: no hedging.
    with ShardCluster(index_dirs, base_port=args.base_port, **server_kwargs) as cluster:
        broker = Broker.for_addresses(cluster.addresses)
        baseline_repeats = [
            measure(broker, args.qps, args.duration, args.client_workers, args.hits,
                    queries, args.zipf_s, seed)
            for seed in range(args.repeats)
        ]
        broker.close()

    # 3. Hedged: replica cluster serves the same index directories as a
    #    second set of processes on a disjoint port range -- a replica
    #    duplicates the process, not the on-disk index (design spec §5).
    with ShardCluster(index_dirs, base_port=args.base_port, **server_kwargs) as primary_cluster, \
            ShardCluster(index_dirs, base_port=args.replica_base_port, **server_kwargs) as replica_cluster:
        hedge_delay_s = hedge_delay_us / 1e6
        hedged_repeats = []
        for seed in range(args.repeats):
            broker = HedgedBroker.for_addresses(
                primary_cluster.addresses, replica_cluster.addresses, hedge_delay_s=hedge_delay_s,
            )
            summary = measure(broker, args.qps, args.duration, args.client_workers, args.hits,
                               queries, args.zipf_s, seed)
            summary["backup_calls_sent"] = broker.backup_calls_sent
            summary["total_shard_calls"] = broker.total_shard_calls
            hedged_repeats.append(summary)
            broker.close()

    baseline_p99 = statistics.median([r["latency"]["p99_us"] for r in baseline_repeats])
    hedged_p99 = statistics.median([r["latency"]["p99_us"] for r in hedged_repeats])
    total_backup = sum(r["backup_calls_sent"] for r in hedged_repeats)
    total_calls = sum(r["total_shard_calls"] for r in hedged_repeats)
    extra_load_pct = total_backup / total_calls if total_calls else 0.0
    print(f"baseline p99={baseline_p99/1000:.2f}ms   hedged p99={hedged_p99/1000:.2f}ms   "
          f"extra load={extra_load_pct:.1%}")

    output = {
        "server": "cascade-grpc-hedged-broker",
        "n": args.n,
        "query_set": "dev",
        "num_distinct_queries": len(queries),
        "qps": args.qps,
        "duration_s": args.duration,
        "repeats": args.repeats,
        "zipf_s": args.zipf_s,
        "client_workers": args.client_workers,
        "server_workers": args.server_workers,
        "queue_depth": args.queue_depth,
        "cache_capacity": args.cache_capacity,
        "algorithm": args.algorithm,
        "hits": args.hits,
        "shard_p95_us_warmup": shard_p95s,
        "hedge_delay_us": hedge_delay_us,
        "baseline_repeats": baseline_repeats,
        "hedged_repeats": hedged_repeats,
        "baseline_p99_us_median": baseline_p99,
        "hedged_p99_us_median": hedged_p99,
        "extra_load_fraction": extra_load_pct,
        "provenance": run_metadata(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "server-hedging.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nwrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity-check on a short run**

Run: `cd py && PYTHONPATH=. uv run --project .. python -m server.hedging --n 4 --qps 5 --warmup-duration 3 --duration 3 --repeats 1`

Expected: prints the warm-up p95s, then `baseline p99=...ms   hedged p99=...ms   extra load=...%`, and writes `bench/results/server-hedging.json`. Uses N=4 rather than the real N=8 default for a fast sanity pass — Task 9 runs the real N=8, full-duration version.

- [ ] **Step 3: Commit**

```bash
git add py/server/hedging.py
git commit -m "phase2b: hedged-requests experiment driver"
```

---

## Task 8: Report rendering

**Files:**
- Create: `py/server/report_2b.py`

**Interfaces:**
- Consumes: `bench/results/server-tail-latency.json` (Task 6's output shape), `bench/results/server-hedging.json` (Task 7's output shape).
- Produces: `bench/phase2b.md`, `bench/plots/phase2b-tail-latency.png`.

Kept as a separate module from `server.report` (sub-project A's renderer) rather than merged into it — matching this project's established practice (see `server/report.py`'s own comment) of not touching an already-shipped phase's report generator for a different phase's data. No unit test file — same convention as sub-project A's `report.py` (verified by rendering real output and visual inspection, Steps 2-3 below).

- [ ] **Step 1: Write `py/server/report_2b.py`**

```python
"""Renders bench/phase2b.md from the tail-latency and hedging results --
sub-project B's two remaining Phase 2 experiments (sharded broker,
hedged requests). Kept separate from server.report (sub-project A's
renderer) rather than merged into it, matching this project's established
practice of not touching an already-shipped phase's report generator for a
different phase's data.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
PLOTS_DIR = REPO_ROOT / "bench" / "plots"


def render_tail_latency_plot(tail_data: dict, output_path: Path) -> None:
    ns = [p["n"] for p in tail_data["points"]]
    p50s = [
        statistics.median([r["latency"]["p50_us"] for r in p["repeats"]]) / 1000
        for p in tail_data["points"]
    ]
    p99s = [p["p99_us_across_repeats"]["median"] / 1000 for p in tail_data["points"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ns, p99s, marker="o", label="p99")
    ax.plot(ns, p50s, marker="o", label="p50")
    ax.set_xlabel("shard count (N)")
    ax.set_ylabel("client-observed latency (ms)")
    ax.set_xticks(ns)
    ax.set_title("Tail-latency amplification vs. shard count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_tail_latency_table(tail_data: dict) -> str:
    lines = [
        "| N | p50 | p95 | p99 | p99.9 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for point in tail_data["points"]:
        reps = point["repeats"]
        mid = sorted(reps, key=lambda r: r["latency"]["p99_us"])[len(reps) // 2]
        lat = mid["latency"]
        lines.append(
            f"| {point['n']} | {lat['p50_us']/1000:.2f}ms | {lat['p95_us']/1000:.2f}ms | "
            f"{lat['p99_us']/1000:.2f}ms | {lat['p999_us']/1000:.2f}ms |"
        )
    return "\n".join(lines)


def render_hedging_table(hedging_data: dict) -> str:
    baseline_p99 = hedging_data["baseline_p99_us_median"] / 1000
    hedged_p99 = hedging_data["hedged_p99_us_median"] / 1000
    delta_pct = (hedged_p99 - baseline_p99) / baseline_p99 * 100 if baseline_p99 else 0.0
    extra_load = hedging_data["extra_load_fraction"]
    return "\n".join([
        "| discipline | p99 | delta vs. baseline | extra load |",
        "|---|---:|---:|---:|",
        f"| baseline (no hedging) | {baseline_p99:.2f}ms | - | - |",
        f"| hedged | {hedged_p99:.2f}ms | {delta_pct:+.1f}% | {extra_load:.1%} |",
    ])


def render_markdown(tail_data: dict, hedging_data: dict) -> str:
    warmup_ms = [round(p / 1000, 2) for p in hedging_data["shard_p95_us_warmup"]]
    lines = [
        "# Phase 2, sub-project B — sharded broker, tail latency, hedging",
        "",
        "Exit artifact for Phase 2 sub-project B (see "
        "`docs/superpowers/specs/2026-09-10-phase2b-broker-design.md`). "
        "Combined with `bench/phase2a.md`, this completes Phase 2's four-plot "
        "exit bar.",
        "",
        "## Docid-semantics limitation",
        "",
        "Each shard's internal docid is local to that shard and BM25 IDF is "
        "computed from that shard's own document frequencies, not the "
        "corpus-global df. The broker's merged top-k is a real computation "
        "over real shard responses, but this document makes no NDCG/quality "
        "claim for the sharded configuration — see the design spec §3.",
        "",
        "## Tail-latency amplification vs. shard count",
        "",
        "![Tail latency vs N](plots/phase2b-tail-latency.png)",
        "",
        f"Open-loop, Poisson arrivals, Zipfian query popularity (s="
        f"{tail_data['zipf_s']}) over {tail_data['num_distinct_queries']:,} dev "
        f"queries, fixed QPS={tail_data['qps']}, "
        f"{tail_data['client_workers']} client dispatch threads. Each row is "
        f"the median of {tail_data['repeats']} runs of "
        f"{tail_data['duration_s']:.0f}s. N=1 is measured through the same "
        "`Broker` path as every other row (a single-shard broker, not "
        "sub-project A's direct-client number), so every point pays the "
        "same broker overhead.",
        "",
        render_tail_latency_table(tail_data),
        "",
        "## Hedged requests",
        "",
        f"At N={hedging_data['n']} shards, fixed QPS={hedging_data['qps']}. "
        f"Hedge delay ({hedging_data['hedge_delay_us']/1000:.2f}ms) is the max "
        f"across shards' own measured p95 service latency from an un-hedged "
        f"warm-up run ({warmup_ms} ms per shard).",
        "",
        render_hedging_table(hedging_data),
        "",
        "## Configuration",
        "",
        f"{tail_data['server_workers']} worker threads per shard, queue depth "
        f"{tail_data['queue_depth']}, {tail_data['cache_capacity']}-entry LRU "
        f"cache per shard, `{tail_data['algorithm']}`, top-{tail_data['hits']}.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    tail_path = RESULTS_DIR / "server-tail-latency.json"
    hedging_path = RESULTS_DIR / "server-hedging.json"
    if not tail_path.exists():
        raise SystemExit(f"no {tail_path}; run `python -m server.tail_latency` first")
    if not hedging_path.exists():
        raise SystemExit(f"no {hedging_path}; run `python -m server.hedging` first")

    tail_data = json.loads(tail_path.read_text())
    hedging_data = json.loads(hedging_path.read_text())

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    render_tail_latency_plot(tail_data, PLOTS_DIR / "phase2b-tail-latency.png")

    (REPO_ROOT / "bench" / "phase2b.md").write_text(render_markdown(tail_data, hedging_data))
    print("wrote bench/phase2b.md and bench/plots/phase2b-tail-latency.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the report renderer**

Run: `cd py && PYTHONPATH=. uv run --project .. python -m server.report_2b`
Expected: `wrote bench/phase2b.md and bench/plots/phase2b-tail-latency.png` (requires Task 6/7's real JSON outputs from Task 9 to exist first — if run before Task 9's real data, this correctly raises `SystemExit` naming the missing file).

- [ ] **Step 3: Visually inspect the output**

Open `bench/phase2b.md` and `bench/plots/phase2b-tail-latency.png`. Confirm: the table's N values are sorted ascending, p99 is monotonically non-decreasing with N (if not, that's a real finding worth a caveat sentence in the markdown, not a bug to silently paper over — the same standard `bench/phase4.md` held itself to), and the hedging table's extra-load percentage is in a plausible range (single-digit to low double-digit percent, not near 0% or near 100% — near-0% means the hedge delay was set too high to ever fire, near-100% means it was set too low and every request hedges, per design spec §7's test already guarding this in miniature).

- [ ] **Step 4: Commit**

```bash
git add py/server/report_2b.py
git commit -m "phase2b: render bench/phase2b.md and the tail-latency plot"
```

---

## Task 9: Real data run, README, and status table

**Files:**
- Modify: `README.md`
- (No new source files — this task runs Tasks 6-8's scripts for real and commits their outputs.)

- [ ] **Step 1: Run the full local test suite**

Run: `uv run pytest` (from repo root)
Expected: all prior tests still pass (the ones from Tasks 1-5), plus sub-project A's and every earlier phase's — no regressions.

- [ ] **Step 2: Partition and build shards for every N up front**

```bash
cd py
PYTHONPATH=. uv run --project .. python -m server.build_shards --n 4 8 16
```

This partitions `data/msmarco-passage.tsv` into `data/shards/n{4,8,16}/` and builds `indexes/shards/n{4,8,16}/shard{i}/` via the real `build_index` binary — real compute against the real 8.8M-passage corpus, on the order of the time Phase 1's own monolithic build took, split across shards. Confirm disk headroom first (spec §9): `df -h /` should show comfortably more than ~8GB free before starting.

- [ ] **Step 3: Run the tail-latency-amplification experiment for real**

```bash
PYTHONPATH=. uv run --project .. python -m server.tail_latency
```

Uses the script's real defaults (`--n 1 4 8 16 --qps 20 --duration 10 --repeats 3`). Writes `bench/results/server-tail-latency.json`. Watch memory while N=16 is running (spec §9) — if the machine thrashes, re-run with `--server-workers 2` for that point specifically and note the reduced worker count in the eventual report's Configuration section (accept the mismatch across rows rather than silently forcing 4 everywhere, and say so in `bench/phase2b.md` if it happens).

- [ ] **Step 4: Run the hedging experiment for real**

```bash
PYTHONPATH=. uv run --project .. python -m server.hedging
```

Uses the script's real defaults (`--n 8 --qps 20 --warmup-duration 10 --duration 10 --repeats 3`). Writes `bench/results/server-hedging.json`.

- [ ] **Step 5: Render the report and inspect it**

```bash
PYTHONPATH=. uv run --project .. python -m server.report_2b
```

Re-run Task 8 Step 3's inspection checklist against the real output this time.

- [ ] **Step 6: Commit the real results**

```bash
cd ..
git add bench/results/server-tail-latency.json bench/results/server-hedging.json \
        bench/phase2b.md bench/plots/phase2b-tail-latency.png
git commit -m "phase2b: real tail-latency and hedging results from the sharded broker"
```

- [ ] **Step 7: Add the README section**

Add to `README.md`, after the existing "## Running Phase 2 (sub-project A: single-node server)" section (before "## Running Phase 3"):

```markdown
## Running Phase 2 (sub-project B: sharded broker, tail latency, hedging)

Needs sub-project A's built server binary and generated proto stubs (see
above), plus the Phase 1 monolithic index for the N=1 baseline point.
Partitions the corpus and builds shard indexes on first use (cached after —
see `server.partition_corpus`/`server.build_shards`'s skip-if-exists logic).

```bash
cd py
PYTHONPATH=. uv run --project .. python -m server.build_shards --n 4 8 16
PYTHONPATH=. uv run --project .. python -m server.tail_latency
PYTHONPATH=. uv run --project .. python -m server.hedging
PYTHONPATH=. uv run --project .. python -m server.report_2b   # -> bench/phase2b.md
```

Each shard's `docid` is local to that shard and BM25 statistics are
per-shard, not corpus-global — this phase makes a latency claim about the
sharded configuration, not a quality (NDCG) one; see
`docs/superpowers/specs/2026-09-10-phase2b-broker-design.md` §3.
```

- [ ] **Step 8: Update the status table**

Change the Phase 2 row in the `## Status` table from:
```
| 2 — Serving, sharding, tail latency | capacity statement | sub-project A done |
```
to:
```
| 2 — Serving, sharding, tail latency | capacity statement | **done** |
```
(Only after `bench/results/server-tail-latency.json` and `bench/results/server-hedging.json` are real, committed results and `bench/phase2b.md` has been rendered from them — if this step runs before Steps 2-6 complete, leave the status table unchanged and note that in the task report instead of marking it done prematurely.)

- [ ] **Step 9: Commit**

```bash
git add README.md
git commit -m "phase2b: document setup and mark Phase 2 done in the status table"
```

---
