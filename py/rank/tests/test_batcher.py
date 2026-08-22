"""DynamicBatcher's flush decision is pure given an injected clock -- tested
with a fake clock rather than real wall-clock sleeps, so these run in
milliseconds and never flake on timing."""

from __future__ import annotations

from rank.batcher import DynamicBatcher


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, ms: float) -> None:
        self.now += ms


def test_flushes_at_max_batch_size_before_wait_elapses():
    clock = FakeClock()
    batcher = DynamicBatcher(max_batch_size=2, max_wait_ms=100, clock=clock)
    batcher.add("a")
    assert not batcher.should_flush()
    batcher.add("b")
    assert batcher.should_flush()
    assert batcher.flush() == ["a", "b"]


def test_flushes_at_max_wait_ms_before_batch_size_reached():
    clock = FakeClock()
    batcher = DynamicBatcher(max_batch_size=10, max_wait_ms=5, clock=clock)
    batcher.add("a")
    assert not batcher.should_flush()
    clock.advance(5.0)
    assert batcher.should_flush()
    assert batcher.flush() == ["a"]


def test_empty_batcher_never_flushes():
    clock = FakeClock()
    batcher = DynamicBatcher(max_batch_size=2, max_wait_ms=5, clock=clock)
    clock.advance(1000.0)
    assert not batcher.should_flush()


def test_flush_resets_the_batch_and_wait_window():
    clock = FakeClock()
    batcher = DynamicBatcher(max_batch_size=2, max_wait_ms=100, clock=clock)
    batcher.add("a")
    batcher.add("b")
    assert batcher.flush() == ["a", "b"]
    assert not batcher.should_flush()
    batcher.add("c")
    assert not batcher.should_flush()


def test_len_reflects_current_queue_depth():
    clock = FakeClock()
    batcher = DynamicBatcher(max_batch_size=10, max_wait_ms=100, clock=clock)
    assert len(batcher) == 0
    batcher.add("a")
    batcher.add("b")
    assert len(batcher) == 2
    batcher.flush()
    assert len(batcher) == 0
