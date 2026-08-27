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
        items, self._items = self._items, []
        self._window_start = None
        return items
