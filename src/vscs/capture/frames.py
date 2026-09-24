"""Frame extraction with real timestamps, safe for variable frame rate video (P1-T4).

CLAUDE.md §4.1: timestamps are ``int64`` nanoseconds, and **frame index is never used as
time**. Phones record at a variable frame rate, so the gap between consecutive frames is
not constant and assuming otherwise corrupts every velocity, time-to-contact and
ego-motion estimate downstream (R-03).

Two timestamp backends
----------------------
``ffprobe`` (preferred)
    Reads the container's per-frame presentation timestamps directly. This is the
    authoritative source and the one to use for real phone footage.

``opencv`` (fallback)
    Uses ``CAP_PROP_POS_MSEC``. Convenient, requires no external binary, and *usually*
    derived from the container - but it is backend-dependent and has been known to
    return zeros or interpolated constant-rate values for some files. Because a
    constant-rate lie is exactly the failure R-03 is about, :func:`analyse_timing`
    reports whether the result actually looks variable, and
    :func:`probe_timestamps` warns when the fallback is used.

**ffprobe is not installed on the development laptop** (see ADR 0003). The parser is
tested against captured ffprobe output, but the end-to-end ffprobe path, and the VFR
behaviour of real phone footage, remain unverified until ffmpeg is installed and a real
clip exists. Do not treat P1-T4 as accepted until then.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from vscs.common.log import get_logger
from vscs.common.types import NS_PER_S, validate_timestamp_stream

logger = get_logger("capture.frames")

Backend = Literal["auto", "ffprobe", "opencv"]

#: A clip whose frame intervals vary by more than this fraction of the median is
#: treated as variable frame rate.
VFR_JITTER_THRESHOLD = 0.05

#: An interval this many times the median counts as a dropped-frame gap.
GAP_FACTOR = 1.8


@dataclass
class TimingReport:
    """What the timestamps say about how the clip was actually recorded."""

    n_frames: int
    duration_s: float
    mean_fps: float
    median_dt_ns: int
    min_dt_ns: int
    max_dt_ns: int
    jitter_ratio: float
    is_vfr: bool
    gap_indices: list[int] = field(default_factory=list)

    def summary(self) -> str:
        kind = "VFR" if self.is_vfr else "CFR"
        return (
            f"{self.n_frames} frames, {self.duration_s:.2f} s, {self.mean_fps:.2f} fps mean, "
            f"{kind} (jitter {self.jitter_ratio:.1%}), "
            f"dt median {self.median_dt_ns / 1e6:.2f} ms "
            f"[{self.min_dt_ns / 1e6:.2f}, {self.max_dt_ns / 1e6:.2f}], "
            f"{len(self.gap_indices)} gap(s)"
        )


@dataclass
class VideoTimestamps:
    """Per-frame container timestamps for one video."""

    path: Path
    t_ns: list[int]
    backend: str
    image_size: tuple[int, int] | None = None

    @property
    def n_frames(self) -> int:
        return len(self.t_ns)


# --------------------------------------------------------------------------- #
# ffprobe backend                                                              #
# --------------------------------------------------------------------------- #
def ffprobe_available() -> bool:
    """True if an ``ffprobe`` binary is on PATH."""
    return shutil.which("ffprobe") is not None


def parse_ffprobe_frames(payload: str) -> list[int]:
    """Parse ``ffprobe -show_entries frame=... -of json`` output into ns timestamps.

    Pure function, so the parsing is testable without ffmpeg installed.

    ``best_effort_timestamp_time`` is preferred over ``pts_time``: ffmpeg falls back to a
    sensible estimate when a container's pts is missing for a frame, and a missing pts on
    one frame should not discard the whole clip.
    """
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ffprobe output is not valid JSON: {exc}") from exc

    frames = data.get("frames")
    if not frames:
        raise ValueError("ffprobe returned no frames; is this a video file?")

    out: list[int] = []
    skipped = 0
    for fr in frames:
        raw = fr.get("best_effort_timestamp_time", fr.get("pts_time"))
        if raw is None or raw == "N/A":
            skipped += 1
            continue
        try:
            out.append(round(float(raw) * NS_PER_S))
        except (TypeError, ValueError):
            skipped += 1
    if skipped:
        logger.warning("ffprobe: %d frame(s) had no usable timestamp and were skipped", skipped)
    if not out:
        raise ValueError("ffprobe returned frames but none carried a usable timestamp")
    return out


def _ffprobe_timestamps(video: Path) -> list[int]:
    """Run ffprobe and return per-frame timestamps in nanoseconds."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time,pts_time",
        "-of",
        "json",
        str(video),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffprobe failed on {video.name}: {exc.stderr.strip()}") from exc
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is not installed (see ADR 0003)") from exc
    return parse_ffprobe_frames(proc.stdout)


# --------------------------------------------------------------------------- #
# OpenCV backend                                                               #
# --------------------------------------------------------------------------- #
def _opencv_timestamps(video: Path) -> tuple[list[int], tuple[int, int]]:
    """Per-frame timestamps via ``CAP_PROP_POS_MSEC``, plus the image size.

    ``POS_MSEC`` is read **after** each ``read()``, which returns the timestamp of the
    frame that ``read()`` just produced. Measured on this backend, that yields
    ``0.00, 33.33, 66.67, ...`` for a 30 fps clip - correct and strictly increasing.

    Reading it *before* ``read()`` instead returns the previously decoded frame's
    timestamp, which duplicates 0.0 across the first two frames and makes the stream
    non-monotonic. That is a real trap: the duplicate would be rejected by
    :func:`validate_timestamp_stream` and look like corrupt input rather than a
    programming error, so the order here matters and is pinned by a test.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video}")
    try:
        size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        out: list[int] = []
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            t_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            out.append(round(t_ms * 1e6))
    finally:
        cap.release()
    if not out:
        raise RuntimeError(f"{video.name} yielded no frames")
    return out, size


# --------------------------------------------------------------------------- #
# Public entry point                                                           #
# --------------------------------------------------------------------------- #
def probe_timestamps(video: Path, *, backend: Backend = "auto") -> VideoTimestamps:
    """Read per-frame timestamps from a video.

    ``auto`` uses ffprobe when available and falls back to OpenCV with a warning, since
    the fallback is the less trustworthy source for VFR footage.
    """
    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(video)

    chosen = backend
    if backend == "auto":
        chosen = "ffprobe" if ffprobe_available() else "opencv"
        if chosen == "opencv":
            logger.warning(
                "ffprobe not found; falling back to OpenCV CAP_PROP_POS_MSEC for %s. "
                "Container timestamps are the authoritative source for variable frame "
                "rate footage (R-03) - install ffmpeg before trusting this on real "
                "phone video (ADR 0003).",
                video.name,
            )

    if chosen == "ffprobe":
        t_ns = _ffprobe_timestamps(video)
        size: tuple[int, int] | None = None
        cap = cv2.VideoCapture(str(video))
        if cap.isOpened():
            size = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        cap.release()
    elif chosen == "opencv":
        t_ns, size = _opencv_timestamps(video)
    else:
        raise ValueError(f"unknown backend {backend!r}")

    return VideoTimestamps(path=video, t_ns=t_ns, backend=chosen, image_size=size)


def analyse_timing(t_ns: list[int]) -> TimingReport:
    """Describe the frame timing, and say whether it is genuinely variable.

    Raises via :func:`validate_timestamp_stream` if the timestamps are unusable, which is
    the monotonic check P1-T4 has to pass.
    """
    if len(t_ns) < 2:
        raise ValueError(f"need at least 2 timestamps to analyse timing, got {len(t_ns)}")
    validate_timestamp_stream(t_ns, name="video timestamps")

    dt = np.diff(np.asarray(t_ns, dtype=np.int64))
    median = int(np.median(dt))
    span_ns = int(t_ns[-1] - t_ns[0])
    # Mean fps over intervals, i.e. (n - 1) intervals spanning `span_ns`.
    mean_fps = (len(t_ns) - 1) * NS_PER_S / span_ns if span_ns > 0 else 0.0
    jitter = float((dt.max() - dt.min()) / median) if median > 0 else 0.0
    gaps = [int(i) + 1 for i in np.nonzero(dt > GAP_FACTOR * median)[0]]

    return TimingReport(
        n_frames=len(t_ns),
        duration_s=span_ns / NS_PER_S,
        mean_fps=float(mean_fps),
        median_dt_ns=median,
        min_dt_ns=int(dt.min()),
        max_dt_ns=int(dt.max()),
        jitter_ratio=jitter,
        is_vfr=jitter > VFR_JITTER_THRESHOLD,
        gap_indices=gaps,
    )


def extract_frames(
    video: Path,
    out_dir: Path,
    *,
    backend: Backend = "auto",
    stride: int = 1,
    image_format: str = "png",
    jpeg_quality: int = 95,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Extract frames to ``out_dir`` and write ``frames.jsonl`` carrying real timestamps.

    Each index line is ``{"frame_index", "t_ns", "file"}``. ``frame_index`` is the index
    in the source video and exists only for traceability - **it is never time**
    (CLAUDE.md §4.1). Anything downstream reads ``t_ns``.

    Returns a dict describing the extraction, suitable for writing into a run folder.
    """
    video = Path(video)
    out_dir = Path(out_dir)
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    fmt = image_format.lower().lstrip(".")
    if fmt not in ("png", "jpg", "jpeg"):
        raise ValueError(f"unsupported image_format {image_format!r}; use png or jpg")

    stamps = probe_timestamps(video, backend=backend)
    report = analyse_timing(stamps.t_ns)
    logger.info("%s: %s (backend=%s)", video.name, report.summary(), stamps.backend)

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video}")

    params: list[int] = []
    if fmt in ("jpg", "jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]

    index: list[dict[str, Any]] = []
    written = 0
    try:
        for i in range(stamps.n_frames):
            ok, frame = cap.read()
            if not ok:
                logger.warning(
                    "%s: decode stopped at frame %d but %d timestamps were reported",
                    video.name,
                    i,
                    stamps.n_frames,
                )
                break
            if i % stride:
                continue
            name = f"{i:06d}.{fmt}"
            if not cv2.imwrite(str(frames_dir / name), frame, params):
                raise RuntimeError(f"failed to write {frames_dir / name}")
            index.append({"frame_index": i, "t_ns": int(stamps.t_ns[i]), "file": f"frames/{name}"})
            written += 1
            if max_frames is not None and written >= max_frames:
                break
    finally:
        cap.release()

    if not index:
        raise RuntimeError(f"{video.name}: no frames extracted")

    # The extracted subset must itself be a valid timestamp stream.
    validate_timestamp_stream([r["t_ns"] for r in index], name="extracted frames")

    from vscs.common.io import write_json, write_jsonl

    write_jsonl(out_dir / "frames.jsonl", index)
    meta = {
        "video": video.name,
        "backend": stamps.backend,
        "image_size": list(stamps.image_size) if stamps.image_size else None,
        "stride": stride,
        "image_format": fmt,
        "n_frames_source": stamps.n_frames,
        "n_frames_written": written,
        "timing": {
            "duration_s": report.duration_s,
            "mean_fps": report.mean_fps,
            "median_dt_ns": report.median_dt_ns,
            "min_dt_ns": report.min_dt_ns,
            "max_dt_ns": report.max_dt_ns,
            "jitter_ratio": report.jitter_ratio,
            "is_vfr": report.is_vfr,
            "gap_indices": report.gap_indices,
        },
    }
    write_json(out_dir / "frames_meta.json", meta)
    return meta
