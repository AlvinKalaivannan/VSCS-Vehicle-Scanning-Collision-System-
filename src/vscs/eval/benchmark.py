"""Per-stage throughput benchmark (P5-T3): time each stage on real inputs, tag the hardware.

Acceptance (CLAUDE.md §6): FPS per stage and end to end, logged.

Each stage is a function run over the same inputs. The first ``warmup`` calls are
discarded (model loading, CUDA kernel compilation and cache warm-up would otherwise
dominate), then every call is timed. Reported per stage: p50 / p95 latency in ms and
throughput (calls per second). The pipeline figure is the sum of the stages' p50 latencies
run back to back. A concurrent pipeline can do better, which the streaming metrics
measure separately (Phase 6).

Every result carries a ``hardware`` string. §9: a latency or fps figure is meaningless,
and must never be quoted, without the hardware it was measured on.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def hardware_tag() -> str:
    """CPU, plus the CUDA device if torch is importable and sees one."""
    tag = f"{platform.system()} {platform.machine()} {platform.processor() or 'cpu'}".strip()
    try:
        import torch

        if torch.cuda.is_available():
            tag += f" | CUDA {torch.cuda.get_device_name(0)}"
    except ImportError:
        pass
    return tag


@dataclass(frozen=True)
class StageTiming:
    stage: str
    n: int
    p50_ms: float
    p95_ms: float

    @property
    def fps(self) -> float:
        return 1000.0 / self.p50_ms if self.p50_ms > 0 else float("inf")


def time_stage(
    name: str,
    fn: Callable[[Any], Any],
    inputs: Sequence[Any],
    *,
    warmup: int,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> StageTiming:
    if len(inputs) <= warmup:
        raise ValueError(f"{name}: need more than {warmup} inputs to time anything after warm-up")
    times = []
    for i, x in enumerate(inputs):
        t0 = clock()
        fn(x)
        if i >= warmup:
            times.append((clock() - t0) / 1e6)
    a = np.asarray(times)
    return StageTiming(name, len(a), float(np.percentile(a, 50)), float(np.percentile(a, 95)))


def pipeline_fps(timings: Sequence[StageTiming]) -> float:
    """Sequential end-to-end rate: one frame through every stage in turn (p50s summed)."""
    total = sum(t.p50_ms for t in timings)
    return 1000.0 / total if total > 0 else float("inf")
