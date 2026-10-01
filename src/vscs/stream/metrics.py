"""Per-frame timing and live latency statistics for the streaming pipeline (P6-T3).

Every frame carries its own stopwatch. The replay source stamps when it *emitted* the
frame; each stage stamps when it *took* the frame from its input queue and when it
*finished*. From those:

    service(stage k) = t_out[k] - t_in[k]                 (time spent working)
    wait(stage k)    = t_in[k]  - t_out[k-1]               (time spent queued before it;
                                                             for k = 0, since emission)
    end_to_end       = t_out[last] - t_emit

and ``end_to_end == sum(service) + sum(wait)`` *exactly*: there is no unaccounted time.
That identity is CLAUDE.md §5's "stage latencies sum to the end-to-end latency" test.
It is also why waits are reported separately: a slow pipeline with fast stages is a
queueing problem, not a compute problem.

``LatencyRecorder`` keeps a rolling window of finished frames and reports, per stage and
end to end, p50/p95/p99 in milliseconds, throughput (fps), and frames dropped per queue.
``over_budget`` compares p95 against the per-stage budget in ``configs/stream.yaml``.
All times are monotonic nanoseconds (``time.perf_counter_ns``), never wall-clock dates.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

NS_PER_MS = 1_000_000


@dataclass
class StreamFrame:
    """One frame moving through the pipeline, with its stopwatch."""

    seq: int
    t_capture_ns: int  # the recording's own timestamp (§4.1), for synchronising outputs
    t_emit_ns: int  # monotonic time the source emitted it
    payload: Any = None
    stamps: list[tuple[str, int, int]] = field(default_factory=list)  # (stage, t_in, t_out)

    def stamp(self, stage: str, t_in_ns: int, t_out_ns: int) -> None:
        self.stamps.append((stage, int(t_in_ns), int(t_out_ns)))


def breakdown(frame: StreamFrame) -> dict[str, Any]:
    """``{"service": {stage: ns}, "wait": {stage: ns}, "end_to_end": ns}`` for one frame."""
    if not frame.stamps:
        raise ValueError(f"frame {frame.seq} has no stage stamps")
    service, wait = {}, {}
    prev_out = frame.t_emit_ns
    for stage, t_in, t_out in frame.stamps:
        wait[stage] = t_in - prev_out
        service[stage] = t_out - t_in
        prev_out = t_out
    return {"service": service, "wait": wait, "end_to_end": prev_out - frame.t_emit_ns}


def _pcts(values_ns: list[int]) -> dict[str, float]:
    a = np.asarray(values_ns, dtype=np.float64) / NS_PER_MS
    p50, p95, p99 = np.percentile(a, [50, 95, 99])
    return {"p50_ms": float(p50), "p95_ms": float(p95), "p99_ms": float(p99), "n": len(a)}


class LatencyRecorder:
    """Rolling statistics over the last ``window`` finished frames."""

    def __init__(self, window: int) -> None:
        if window < 2:
            raise ValueError("window must hold at least two frames")
        self._done: deque[tuple[int, dict[str, Any]]] = deque(maxlen=window)
        self.dropped: dict[str, int] = defaultdict(int)
        self.finished = 0

    def finish(self, frame: StreamFrame) -> dict[str, Any]:
        b = breakdown(frame)
        self._done.append((frame.stamps[-1][2], b))
        self.finished += 1
        return b

    def drop(self, queue_name: str, n: int = 1) -> None:
        self.dropped[queue_name] += n

    def summary(self) -> dict[str, Any]:
        if not self._done:
            return {
                "stages": {},
                "waits": {},
                "end_to_end": None,
                "fps": 0.0,
                "dropped": dict(self.dropped),
            }
        service: dict[str, list[int]] = defaultdict(list)
        waits: dict[str, list[int]] = defaultdict(list)
        e2e = []
        for _, b in self._done:
            for s, v in b["service"].items():
                service[s].append(v)
            for s, v in b["wait"].items():
                waits[s].append(v)
            e2e.append(b["end_to_end"])
        t_first, t_last = self._done[0][0], self._done[-1][0]
        span_s = (t_last - t_first) / 1e9
        fps = (len(self._done) - 1) / span_s if span_s > 0 else 0.0
        return {
            "stages": {s: _pcts(v) for s, v in service.items()},
            "waits": {s: _pcts(v) for s, v in waits.items()},
            "end_to_end": _pcts(e2e),
            "fps": fps,
            "dropped": dict(self.dropped),
        }


def over_budget(summary: dict[str, Any], budget_ms: dict[str, float]) -> dict[str, float]:
    """Stages (and ``end_to_end``) whose p95 exceeds its budget: ``{name: p95_ms}``."""
    out = {}
    for name, limit in budget_ms.items():
        stats = summary["end_to_end"] if name == "end_to_end" else summary["stages"].get(name)
        if stats is not None and stats["p95_ms"] > float(limit):
            out[name] = stats["p95_ms"]
    return out
