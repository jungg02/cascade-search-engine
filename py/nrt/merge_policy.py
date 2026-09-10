"""A simplified tiered merge policy, not full Lucene multi-tier: bounds
live segment count from above at max_segments by collapsing the
merge_factor smallest segments into one as soon as the count exceeds it.
Smallest-first mirrors why real tiered policies merge small segments
before large ones -- amortized merge cost per document stays bounded.
See design spec §4a.
"""

from __future__ import annotations


class TieredMergePolicy:
    def __init__(self, merge_factor: int = 4, max_segments: int = 8) -> None:
        self.merge_factor = merge_factor
        self.max_segments = max_segments

    def maybe_merge(self, segments: list) -> list | None:
        if len(segments) <= self.max_segments:
            return None
        by_size = sorted(segments, key=lambda s: s.doc_count)
        return by_size[: min(self.merge_factor, len(by_size))]
