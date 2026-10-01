"""Replay a recording as a live stream, at its own timestamps (P6-T1, ADR 0008).

"Live" in VSCS means a recording played back in real time (§1, recorded input only). The
replay releases frame ``i`` at ``start + (t_i - t_0) / speed``, using the recording's own
``t_ns``, never ``i / fps``. Phone footage has a variable frame rate (R-03), and a replay
that assumed a constant rate would quietly change the speed of everything in it.

This is the bottom rung of the §8.2 stream-ingest ladder: direct file replay, with no
network and no ffmpeg/GStreamer. The RTSP rungs wrap the same frame list once the LGPL
question for ffmpeg/GStreamer is settled (§10).

The source never drops or skips frames. If the consumer is too slow, frames are emitted
*late* and counted. Dropping is the pipeline queues' job (P6-T2), where it is measured.

The clock and sleep are injectable, so tests check the schedule exactly.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from vscs.common.io import read_jsonl
from vscs.stream.metrics import StreamFrame


def frames_from_capture_run(run_dir: Path) -> list[tuple[int, Path]]:
    """``(t_ns, image path)`` for every frame of an ``extract_frames.py`` run, in time order."""
    run_dir = Path(run_dir)
    rows = list(read_jsonl(run_dir / "frames.jsonl"))
    if not rows:
        raise ValueError(f"{run_dir}: frames.jsonl is empty")
    return sorted(((int(r["t_ns"]), run_dir / r["file"]) for r in rows), key=lambda x: x[0])


class FileReplay:
    """Iterate ``StreamFrame`` objects released on the recording's schedule."""

    def __init__(
        self,
        frames: Sequence[tuple[int, Any]],
        *,
        speed: float = 1.0,
        late_tolerance_ns: int = 0,
        clock: Callable[[], int] = time.perf_counter_ns,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not frames:
            raise ValueError("nothing to replay")
        t = [int(f[0]) for f in frames]
        if any(b <= a for a, b in pairwise(t)):
            raise ValueError("frame timestamps must be strictly increasing (R-03)")
        if speed <= 0:
            raise ValueError(f"speed must be positive, got {speed}")
        self.frames = list(frames)
        self.speed = float(speed)
        self.late_tolerance_ns = int(late_tolerance_ns)
        self.clock = clock
        self.sleep = sleep
        self.late = 0
        self.max_lateness_ns = 0

    def __iter__(self) -> Iterator[StreamFrame]:
        t0_rec = int(self.frames[0][0])
        t0_wall = self.clock()
        for seq, (t_ns, payload) in enumerate(self.frames):
            due = t0_wall + round((int(t_ns) - t0_rec) / self.speed)
            wait = due - self.clock()
            if wait > 0:
                self.sleep(wait / 1e9)
            now = self.clock()
            lateness = now - due
            if lateness > self.late_tolerance_ns:
                self.late += 1
            self.max_lateness_ns = max(self.max_lateness_ns, lateness)
            yield StreamFrame(seq=seq, t_capture_ns=int(t_ns), t_emit_ns=now, payload=payload)
