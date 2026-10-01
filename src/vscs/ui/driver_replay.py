"""Driver replay: top-down silhouette, per-component colours, path, and an audio cue (P5-T4).

Acceptance (CLAUDE.md §6): works on any recorded run.

What a driver would see and hear, rebuilt from a recorded drive's ``RiskFrame`` stream:

* **Silhouette:** the van seen from above, forward up the image, each component
  coloured by the alert level it reaches on its own (the same grading as the alerts, via
  ``risk.alerts.raw_level``). Obstacles are drawn as boxes, and the predicted path as a
  line.
* **Banner** on every image: "advisory research prototype - not a safety device" (§1).
* **Audio:** silence when there is no alert; beeps that speed up through caution and
  warning; a continuous tone at critical. One WAV track for the whole drive, aligned to
  the frames' own ``t_ns``.

Output is PNG frames + ``replay.wav`` + ``index.json`` (frame file, t_ns, alert). Using
only PIL and the standard ``wave`` module keeps this free of ffmpeg, whose LGPL terms are
under the developer's review (§10). Stitching into a video is a later, separate step.
"""

from __future__ import annotations

import json
import wave
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from vscs.common.types import Obstacle, RiskFrame
from vscs.risk.alerts import raw_level

NS_PER_S = 1_000_000_000


class TopDown:
    """Maps ``veh`` metres to image pixels: forward is up, left is left."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.w, self.h = (int(v) for v in cfg["image_px"])
        self.s = float(cfg["px_per_m"])
        self.ox = float(cfg["origin_frac"][0]) * self.w
        self.oy = float(cfg["origin_frac"][1]) * self.h

    def px(self, x_m: float, y_m: float) -> tuple[float, float]:
        return self.ox - y_m * self.s, self.oy - x_m * self.s


def render_frame(
    footprints: Mapping[str, tuple[Polygon, float, float]],
    frame: RiskFrame,
    cfg: dict[str, Any],
    alert_colours: Mapping[str, Iterable[int]],
    risk_cfg: dict[str, Any],
    *,
    obstacles: Iterable[Obstacle] = (),
    path_xy: npt.ArrayLike | None = None,
    obstacle_colour: Iterable[int] = (80, 160, 230),
    path_colour: Iterable[int] = (120, 220, 120),
) -> Image.Image:
    view = TopDown(cfg)
    img = Image.new("RGB", (view.w, view.h), tuple(int(c) for c in cfg["background"]))
    d = ImageDraw.Draw(img)
    level_of = {cr.component: raw_level([cr], risk_cfg) for cr in frame.per_component}
    for name, (poly, _zmin, _zmax) in footprints.items():
        colour = tuple(int(c) for c in alert_colours[level_of.get(name, "none")])[:3]
        d.polygon([view.px(x, y) for x, y in poly.exterior.coords], fill=colour, outline=(0, 0, 0))
    for ob in obstacles:
        cx, cy = ob.center_veh[0], ob.center_veh[1]
        hx, hy = ob.extent[0] / 2.0, ob.extent[1] / 2.0
        corners = [view.px(cx + a, cy + b) for a, b in ((hx, hy), (hx, -hy), (-hx, -hy), (-hx, hy))]
        d.polygon(corners, outline=tuple(int(c) for c in obstacle_colour)[:3], width=3)
    if path_xy is not None and len(np.asarray(path_xy)) > 1:
        d.line(
            [view.px(x, y) for x, y in np.asarray(path_xy)],
            fill=tuple(int(c) for c in path_colour)[:3],
            width=2,
        )
    d.text((8, 8), str(cfg["banner"]), fill=tuple(int(c) for c in cfg["banner_colour"]))
    d.text(
        (8, 24),
        f"alert: {frame.alert_level}",
        fill=tuple(int(c) for c in alert_colours[frame.alert_level])[:3],
    )
    return img


def tone(
    level: str, duration_s: float, audio_cfg: dict[str, Any], t0_s: float = 0.0
) -> npt.NDArray[np.int16]:
    """``duration_s`` of the cue for ``level``, phase-continuous from absolute time ``t0_s``."""
    sr = int(audio_cfg["sample_rate_hz"])
    n = max(0, round(duration_s * sr))
    t = t0_s + np.arange(n) / sr
    rate = float(audio_cfg["beeps_per_s"][level])
    if rate == 0:
        gate = np.zeros(n)
    elif rate < 0:
        gate = np.ones(n)
    else:
        gate = ((t * rate) % 1.0 < float(audio_cfg["duty"])).astype(np.float64)
    wave_ = (
        np.sin(2 * np.pi * float(audio_cfg["tone_hz"]) * t) * gate * float(audio_cfg["amplitude"])
    )
    return (wave_ * 32767).astype(np.int16)


def audio_track(
    frames: list[RiskFrame], audio_cfg: dict[str, Any], last_frame_s: float
) -> npt.NDArray[np.int16]:
    """The whole drive's cue: each frame's level held until the next frame's ``t_ns``."""
    if not frames:
        return np.zeros(0, dtype=np.int16)
    t0 = frames[0].t_ns
    parts = []
    for f, nxt in zip(frames, [*frames[1:], None], strict=True):
        start = (f.t_ns - t0) / NS_PER_S
        end = (nxt.t_ns - t0) / NS_PER_S if nxt else start + last_frame_s
        parts.append(tone(f.alert_level, end - start, audio_cfg, t0_s=start))
    return np.concatenate(parts)


def write_wav(path: Path, samples: npt.NDArray[np.int16], sample_rate: int) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(np.ascontiguousarray(samples, dtype="<i2").tobytes())
    return path


def write_replay(
    out_dir: Path,
    footprints: Mapping[str, tuple[Polygon, float, float]],
    frames: list[RiskFrame],
    ui_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
    *,
    obstacles_by_t: Mapping[int, list[Obstacle]] | None = None,
    paths_by_t: Mapping[int, npt.ArrayLike] | None = None,
) -> Path:
    """Write ``frames/*.png``, ``replay.wav`` and ``index.json`` into ``out_dir``."""
    cfg = ui_cfg["driver_replay"]
    colours = ui_cfg["view"]["alert_colours"]
    out_dir = Path(out_dir)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    index = []
    for i, f in enumerate(frames):
        img = render_frame(
            footprints,
            f,
            cfg,
            colours,
            risk_cfg,
            obstacles=(obstacles_by_t or {}).get(f.t_ns, ()),
            path_xy=(paths_by_t or {}).get(f.t_ns),
            obstacle_colour=ui_cfg["view"]["obstacle_colour"],
            path_colour=ui_cfg["view"]["path_colour"],
        )
        name = f"frames/{i:06d}.png"
        img.save(out_dir / name)
        index.append(
            {
                "file": name,
                "t_ns": f.t_ns,
                "alert_level": f.alert_level,
                "worst_component": f.worst_component,
            }
        )
    gap = (frames[-1].t_ns - frames[-2].t_ns) / NS_PER_S if len(frames) > 1 else 0.1
    audio = cfg["audio"]
    write_wav(out_dir / "replay.wav", audio_track(frames, audio, gap), int(audio["sample_rate_hz"]))
    (out_dir / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return out_dir
