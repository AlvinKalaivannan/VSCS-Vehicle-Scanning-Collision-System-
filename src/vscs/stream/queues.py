"""Bounded queue that drops the stalest item instead of blocking (P6-T2, ADR 0008).

Between every two pipeline stages sits one of these. If the downstream stage falls
behind, a normal queue grows without limit, and every frame waits longer than the one
before: latency climbs for as long as the overload lasts (R-21). A *blocking* bounded
queue stops the growth but pushes the stall upstream, so the source falls behind real
time instead.

For a risk system the newest frame is the valuable one: an obstacle position from half a
second ago is worse than useless. So when the queue is full, ``put`` discards the
**oldest** item and never waits. Latency is then bounded by roughly
``maxsize x (downstream service time)``, whatever the input rate, and the cost of overload
shows up as counted drops, which the metrics surface (P6-T3).

Thread-safe: producers and consumers may be different threads.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any


class DropOldestQueue:
    def __init__(self, maxsize: int) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize must be >= 1, got {maxsize}")
        self.maxsize = maxsize
        self._items: deque[Any] = deque()
        self._cond = threading.Condition()
        self._closed = False
        self.puts = 0
        self.dropped = 0
        self.max_depth = 0

    def put(self, item: Any) -> Any | None:
        """Add ``item``. If full, evict and return the oldest item (else ``None``)."""
        with self._cond:
            if self._closed:
                raise RuntimeError("put on a closed queue")
            evicted = None
            if len(self._items) >= self.maxsize:
                evicted = self._items.popleft()
                self.dropped += 1
            self._items.append(item)
            self.puts += 1
            self.max_depth = max(self.max_depth, len(self._items))
            self._cond.notify()
            return evicted

    def get(self, timeout: float | None = None) -> Any | None:
        """The oldest item still queued, or ``None`` on timeout or once closed and empty."""
        with self._cond:
            if not self._items and not self._closed:
                self._cond.wait(timeout)
            return self._items.popleft() if self._items else None

    def close(self) -> None:
        """No more puts; waiting consumers wake and drain what is left."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed

    def __len__(self) -> int:
        with self._cond:
            return len(self._items)
