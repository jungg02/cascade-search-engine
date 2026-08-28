"""Dynamic-batching queue logic: accumulate requests until either
max_batch_size or max_wait_ms is hit, whichever comes first. Pure given an
injected clock, so the batching *decision* is unit-tested without any real
concurrency or GPU -- crossencoder_harness.py wraps this in a flusher thread
that actually calls the model, guarding every add/should_flush/flush call
with its own lock (this class is deliberately not internally synchronized).
"""

from __future__ import annotations

from typing import Callable


class DynamicBatcher:
    def __init__(self, max_batch_size: int, max_wait_ms: float, clock: Callable[[], float]) -> None:
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._clock = clock
        self._items: list = []
        self._window_start: float | None = None

    def add(self, item) -> None:
        if not self._items:
            self._window_start = self._clock()
        self._items.append(item)

    def __len__(self) -> int:
        return len(self._items)

    def should_flush(self) -> bool:
        if not self._items:
            return False
        if len(self._items) >= self.max_batch_size:
            return True
        elapsed = self._clock() - self._window_start
        return elapsed >= self.max_wait_ms

    def flush(self) -> list:
        # Capped at max_batch_size even when should_flush() fired on the
        # max_wait_ms branch with more items already queued -- under a single
        # writer this rarely mattered (should_flush()'s count check fires
        # before the queue can overshoot by much), but with many concurrent
        # producers adding faster than one flusher thread can drain, the
        # queue can accumulate well past max_batch_size between checks. An
        # uncapped flush would silently turn "max_batch_size" into "however
        # much piled up," confounding the very axis the batching sweep exists
        # to measure.
        items = self._items[: self.max_batch_size]
        self._items = self._items[self.max_batch_size :]
        self._window_start = self._clock() if self._items else None
        return items
