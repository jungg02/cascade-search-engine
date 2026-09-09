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
