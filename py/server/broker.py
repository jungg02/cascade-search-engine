"""Fans a query out to every shard, waits for all of them, merges results
by score. Every shard covers a disjoint slice of the corpus, so unlike a
cache/replica lookup there's no answer without all N -- see
docs/superpowers/specs/2026-09-10-phase2b-broker-design.md §4.
"""

from __future__ import annotations

import threading
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, NamedTuple

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
        merged: list[MergedResult] = []
        # Unconditional .result() calls block until every future is done, propagate the first exception encountered, and ensure no partial merge is possible.
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
        # dispatch() is called from up to 32 concurrent client threads
        # against one broker instance (see tail_latency.py/hedging.py) --
        # plain `+=` on these counters is not atomic, so a lock protects
        # both increments below.
        self._counter_lock = threading.Lock()
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
        with self._counter_lock:
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

        # Pass 1: Submit all backups immediately, without waiting. By the time
        # Pass 2 runs, all backups have been executing concurrently since here,
        # not serially stalled behind the previous shard's race resolution.
        backups: dict[int, Any] = {}  # shard_index -> backup_future
        for i, primary in enumerate(primaries):
            if primary in not_done:
                backup = self._backup_pool.submit(
                    self._replica_clients[i].dispatch, query, k=k, timeout_s=self._timeout_s
                )
                with self._counter_lock:
                    self.backup_calls_sent += 1
                backups[i] = backup

        # Pass 2: Resolve each shard's result. For shards with a backup, race
        # primary and backup; for shards without, just take the primary.
        results: list = [None] * n
        for i, primary in enumerate(primaries):
            if i in backups:
                done, _ = wait([primary, backups[i]], return_when=FIRST_COMPLETED)
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
        self._backup_pool.shutdown(wait=True)
        for client in self._replica_clients:
            client.close()
        super().close()
