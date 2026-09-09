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
