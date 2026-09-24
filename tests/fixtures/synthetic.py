"""Loader and geometry helpers for the synthetic fixture world (P0-T4).

The scene is declared in ``scene.yaml``; this module turns it into objects and
provides the exact distance queries the known-answer tests assert against
(CLAUDE.md section 5).

Everything here is analytic. There is no sampling in any distance function, so a test
can demand agreement to 1e-6 and mean it. Point-cloud sampling exists separately, is
seeded, and is only used where a test genuinely needs points (label fusion, P2-T3).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from vscs.common.frames import look_at_T_world_cam
from vscs.common.types import NS_PER_S

FloatArray = npt.NDArray[np.float64]

SCENE_PATH = Path(__file__).resolve().parent / "scene.yaml"


# --------------------------------------------------------------------------- #
# Boxes                                                                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Box:
    """An axis-aligned box in the veh frame.

    ``bounds`` in the YAML is ``[xmin, ymin, zmin, xmax, ymax, zmax]``.
    """

    name: str
    lo: FloatArray
    hi: FloatArray

    @classmethod
    def from_bounds(cls, name: str, bounds: list[float]) -> Box:
        if len(bounds) != 6:
            raise ValueError(f"{name}: bounds must be 6 numbers, got {len(bounds)}")
        lo = np.array(bounds[:3], dtype=np.float64)
        hi = np.array(bounds[3:], dtype=np.float64)
        if np.any(hi < lo):
            raise ValueError(f"{name}: bounds are inverted: lo={lo}, hi={hi}")
        return cls(name=name, lo=lo, hi=hi)

    @property
    def center(self) -> FloatArray:
        return (self.lo + self.hi) / 2.0

    @property
    def extent(self) -> FloatArray:
        """Full extent (dx, dy, dz), matching ``Obstacle.extent`` semantics."""
        return self.hi - self.lo

    @property
    def min_z(self) -> float:
        """Lowest z, for ``Component.min_z`` and underbody clearance checks."""
        return float(self.lo[2])

    @property
    def volume(self) -> float:
        return float(np.prod(self.extent))

    def corners(self) -> FloatArray:
        """The 8 corners, shape ``(8, 3)``."""
        return np.array(
            [
                [x, y, z]
                for x in (self.lo[0], self.hi[0])
                for y in (self.lo[1], self.hi[1])
                for z in (self.lo[2], self.hi[2])
            ]
        )

    def nearest_point_to(self, p: npt.ArrayLike) -> FloatArray:
        """The point of this box closest to ``p`` (clamping, so exact)."""
        p = np.asarray(p, dtype=np.float64).reshape(3)
        return np.clip(p, self.lo, self.hi)

    def distance_to_point(self, p: npt.ArrayLike) -> float:
        """Euclidean distance to ``p``; 0 if ``p`` is inside."""
        p = np.asarray(p, dtype=np.float64).reshape(3)
        d = np.maximum(np.maximum(self.lo - p, p - self.hi), 0.0)
        return float(np.linalg.norm(d))

    def distance_to_box(self, other: Box) -> float:
        """Euclidean distance between two axis-aligned boxes; 0 if they overlap."""
        d = np.maximum(np.maximum(self.lo - other.hi, other.lo - self.hi), 0.0)
        return float(np.linalg.norm(d))

    def distance_to_box_horizontal(self, other: Box) -> float:
        """Distance between two box *footprints*, ignoring height.

        This is the right query for a kerb or a speed bump: the vehicle will drive
        over the spot, so what matters is the ground-plane gap now and the height
        clearance separately (P4-T6). The full 3D distance would report a larger
        number simply because a bumper sits above kerb height, which is not the
        clearance a driver cares about.
        """
        d = np.maximum(np.maximum(self.lo[:2] - other.hi[:2], other.lo[:2] - self.hi[:2]), 0.0)
        return float(np.linalg.norm(d))

    def distance_to_vertical_axis(self, axis_xy: npt.ArrayLike) -> float:
        """Distance in the ground plane from this box's footprint to a vertical line.

        This is the query a pole poses: the pole is tall enough that height does not
        limit the contact, so the meaningful clearance is purely horizontal.
        """
        q = np.asarray(axis_xy, dtype=np.float64).reshape(2)
        d = np.maximum(np.maximum(self.lo[:2] - q, q - self.hi[:2]), 0.0)
        return float(np.linalg.norm(d))

    def sample_surface_points(self, n: int, *, seed: int) -> FloatArray:
        """``n`` seeded points on the box surface, for label-fusion style tests.

        Faces are chosen with probability proportional to their area, so the sampling
        is uniform over the surface rather than over the faces.
        """
        rng = np.random.default_rng(seed)
        dx, dy, dz = self.extent
        # Areas of the (+x,-x), (+y,-y), (+z,-z) face pairs.
        areas = np.array([dy * dz, dy * dz, dx * dz, dx * dz, dx * dy, dx * dy])
        total = areas.sum()
        if total <= 0:
            return np.repeat(self.center[None, :], n, axis=0)
        face = rng.choice(6, size=n, p=areas / total)
        u = rng.random(n)
        v = rng.random(n)
        pts = np.empty((n, 3))
        for f in range(6):
            m = face == f
            if not np.any(m):
                continue
            axis = f // 2
            at_hi = f % 2 == 0
            other = [a for a in (0, 1, 2) if a != axis]
            pts[m, axis] = self.hi[axis] if at_hi else self.lo[axis]
            pts[m, other[0]] = self.lo[other[0]] + u[m] * self.extent[other[0]]
            pts[m, other[1]] = self.lo[other[1]] + v[m] * self.extent[other[1]]
        return pts


@dataclass(frozen=True)
class Pole:
    """A thin vertical cylinder standing on the ground."""

    name: str
    axis_xy: FloatArray
    radius_m: float
    height_m: float

    def distance_to_box_axis(self, box: Box) -> float:
        """Horizontal distance from the box footprint to the pole *axis*."""
        return box.distance_to_vertical_axis(self.axis_xy)

    def distance_to_box_surface(self, box: Box) -> float:
        """Horizontal distance to the pole *surface*, i.e. axis distance minus radius.

        Clamped at 0: a negative value would mean interpenetration, not a distance.
        """
        return max(0.0, self.distance_to_box_axis(box) - self.radius_m)


# --------------------------------------------------------------------------- #
# Scene                                                                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CameraPose:
    """One analytically known camera pose with its timestamp."""

    t_ns: int
    T_world_cam: FloatArray


@dataclass(frozen=True)
class Scene:
    """The whole fixture world."""

    raw: dict[str, Any]
    vehicle: Box
    components: dict[str, Box]
    component_meta: dict[str, dict[str, Any]]
    pole: Pole
    curb: Box
    box_obstacle: Box
    K: FloatArray
    image_size: tuple[int, int]

    # -- convenience ------------------------------------------------------- #
    def component(self, name: str) -> Box:
        try:
            return self.components[name]
        except KeyError as exc:
            raise KeyError(
                f"no fixture component {name!r}; have {sorted(self.components)}"
            ) from exc

    @property
    def wheelbase_m(self) -> float:
        return float(self.raw["vehicle"]["wheelbase_m"])

    def nearest_component_to_pole(self) -> tuple[str, float]:
        """``(name, horizontal distance to the pole axis)`` for the closest component.

        This is the ground truth for component attribution accuracy - the metric that
        decides whether VSCS beats the single-bounding-box baseline (CLAUDE.md
        section 11).
        """
        dists = {n: self.pole.distance_to_box_axis(b) for n, b in self.components.items()}
        name = min(dists, key=lambda k: dists[k])
        return name, dists[name]

    def camera_trajectory(self) -> list[CameraPose]:
        """The 20-pose scan loop, with jittered but strictly increasing timestamps."""
        t = self.raw["trajectory"]
        n = int(t["n_frames"])
        radius = float(t["radius_m"])
        height = float(t["height_m"])
        cx, cy = (float(v) for v in t["center_xy"])
        target = np.array([float(v) for v in t["look_at"]])
        fps = float(t["nominal_fps"])
        jitter = float(t["jitter_fraction"])
        rng = np.random.default_rng(int(t["seed"]))

        nominal_dt_ns = round(NS_PER_S / fps)
        # Jitter each interval, never below half the nominal, so timestamps stay
        # strictly increasing while still being irregular.
        gaps = nominal_dt_ns * (1.0 + jitter * (rng.random(n) - 0.5) * 2.0)
        gaps = np.maximum(gaps, nominal_dt_ns * 0.5)
        t_ns = np.cumsum(np.round(gaps).astype(np.int64))

        poses: list[CameraPose] = []
        for i in range(n):
            angle = 2.0 * np.pi * i / n
            eye = np.array([cx + radius * np.cos(angle), cy + radius * np.sin(angle), height])
            poses.append(
                CameraPose(t_ns=int(t_ns[i]), T_world_cam=look_at_T_world_cam(eye, target))
            )
        return poses

    def labelled_point_cloud(self, points_per_component: int = 200, *, seed: int = 20260924):
        """Seeded labelled points over all components.

        Returns ``(points (N, 3), labels (N,))`` where labels are component names.
        Stands in for a hand-labelled 3D subset so P2-T3 fusion can be tested against
        a known answer before any real labelling exists (P2-T4).
        """
        pts: list[FloatArray] = []
        labels: list[str] = []
        for offset, (name, box) in enumerate(sorted(self.components.items())):
            p = box.sample_surface_points(points_per_component, seed=seed + offset)
            pts.append(p)
            labels.extend([name] * points_per_component)
        return np.vstack(pts), np.array(labels)


def load_scene(path: Path | None = None) -> Scene:
    """Load ``scene.yaml`` into a :class:`Scene`."""
    p = Path(path) if path is not None else SCENE_PATH
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    components = {
        name: Box.from_bounds(name, spec["bounds"]) for name, spec in raw["components"].items()
    }
    meta = {name: dict(spec) for name, spec in raw["components"].items()}

    pole_spec = raw["obstacles"]["pole"]
    pole = Pole(
        name="pole",
        axis_xy=np.array(pole_spec["axis_xy"], dtype=np.float64),
        radius_m=float(pole_spec["radius_m"]),
        height_m=float(pole_spec["height_m"]),
    )

    cam = raw["camera"]
    K = np.array(
        [
            [float(cam["fx"]), 0.0, float(cam["cx"])],
            [0.0, float(cam["fy"]), float(cam["cy"])],
            [0.0, 0.0, 1.0],
        ]
    )

    return Scene(
        raw=raw,
        vehicle=Box.from_bounds("vehicle", raw["vehicle"]["bounds"]),
        components=components,
        component_meta=meta,
        pole=pole,
        curb=Box.from_bounds("curb", raw["obstacles"]["curb"]["bounds"]),
        box_obstacle=Box.from_bounds("box", raw["obstacles"]["box"]["bounds"]),
        K=K,
        image_size=(int(cam["image_size"][0]), int(cam["image_size"][1])),
    )
