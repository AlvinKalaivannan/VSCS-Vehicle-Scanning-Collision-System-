"""Staged concurrent pipeline: one thread per stage, drop-oldest queues between (P6-T2).

    source ──▶ [q] decode ──▶ [q] detect ──▶ ... ──▶ [q] output ──▶ sink
                  │ thread        │ thread                │ thread

Each stage is a plain function ``payload -> payload``, so the real stages (perception,
risk) plug in unchanged and the pipeline is tested with stubs. A stage thread:

1. takes the oldest frame still waiting in its input queue,
2. stamps the time it took it, runs its function, and stamps the time it finished,
3. hands the frame to the next queue. If that queue is full, its *oldest* frame is
   evicted and counted as a drop against that queue (backpressure without blocking).

The last stage hands frames to ``LatencyRecorder.finish`` and then to the sink. A stage
that raises is counted as an error for that frame, which is then dropped. One bad frame
must not stop the stream.

Shutdown flows downstream: when the source is exhausted the first queue is closed; each
stage drains its queue, then closes the next. So every frame already accepted is either
finished or counted as dropped. None vanishes.

On CPython, threads share one interpreter lock. That is fine for stages whose time is
spent in NumPy, OpenCV or model inference (they release the lock), and it is what the
latency metrics are for: P6-T4 measures whether it holds for the real stages.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from vscs.common.log import get_logger
from vscs.stream.metrics import LatencyRecorder, StreamFrame
from vscs.stream.queues import DropOldestQueue

logger = get_logger("stream.pipeline")

#: How long an idle stage waits on its queue before re-checking for shutdown (s).
_POLL_S = 0.05


@dataclass(frozen=True)
class Stage:
    name: str
    fn: Callable[[Any], Any]


class Pipeline:
    def __init__(
        self,
        stages: list[Stage],
        queue_size: int,
        recorder: LatencyRecorder,
        *,
        sink: Callable[[StreamFrame], None] | None = None,
        clock: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if not stages:
            raise ValueError("a pipeline needs at least one stage")
        self.stages = stages
        self.queues = [DropOldestQueue(queue_size) for _ in stages]
        self.recorder = recorder
        self.sink = sink
        self.clock = clock
        self.errors: dict[str, int] = {s.name: 0 for s in stages}
        self._lock = threading.Lock()

    def _queue_name(self, i: int) -> str:
        return f"{self.stages[i].name}_in"

    def _hand_on(self, i: int, frame: StreamFrame) -> None:
        if i + 1 < len(self.stages):
            if self.queues[i + 1].put(frame) is not None:
                with self._lock:
                    self.recorder.drop(self._queue_name(i + 1))
        else:
            with self._lock:
                self.recorder.finish(frame)
            if self.sink is not None:
                self.sink(frame)

    def _worker(self, i: int) -> None:
        stage, q = self.stages[i], self.queues[i]
        while True:
            frame = q.get(timeout=_POLL_S)
            if frame is None:
                if q.closed and len(q) == 0:
                    break
                continue
            t_in = self.clock()
            try:
                frame.payload = stage.fn(frame.payload)
            except Exception:
                logger.exception(
                    "stage %s failed on frame %d; frame dropped", stage.name, frame.seq
                )
                with self._lock:
                    self.errors[stage.name] += 1
                continue
            frame.stamp(stage.name, t_in, self.clock())
            self._hand_on(i, frame)
        if i + 1 < len(self.queues):
            self.queues[i + 1].close()

    def run(self, source: Iterable[StreamFrame]) -> dict[str, Any]:
        """Push every source frame through; return the final latency summary."""
        threads = [
            threading.Thread(target=self._worker, args=(i,), name=f"stage-{s.name}", daemon=True)
            for i, s in enumerate(self.stages)
        ]
        for t in threads:
            t.start()
        for frame in source:
            if self.queues[0].put(frame) is not None:
                with self._lock:
                    self.recorder.drop(self._queue_name(0))
        self.queues[0].close()
        for t in threads:
            t.join()
        summary = self.recorder.summary()
        summary["errors"] = dict(self.errors)
        return summary
