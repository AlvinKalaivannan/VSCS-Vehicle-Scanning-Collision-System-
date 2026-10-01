"""Multi-object tracking on the ground plane: Kalman filter + two-stage association (P4-T4).

Acceptance (CLAUDE.md §6): tracks stable on a staged walk-behind clip (person at a safe
distance, van stationary).

Why track at all: one frame gives a position; the risk engine needs a *velocity* to
predict where a person or car will be (time to contact), and it needs the same object to
keep the same id so alerts do not flicker as detections come and go (R-10).

Kalman filter (constant velocity, per object, in the fixed ``world`` frame)
---------------------------------------------------------------------------
State ``x = [px, py, vx, vy]``. Over a step ``dt`` the motion model is ``x' = F x`` with

    F = [[1, 0, dt, 0],
         [0, 1, 0, dt],
         [0, 0, 1,  0],
         [0, 0, 0,  1]]

and unknown accelerations enter as process noise ``Q = G G^T a^2``, with
``G = [dt^2/2, dt^2/2, dt, dt]`` per axis (``a = accel_std_mps2``). A detection measures
position only: ``z = H x + noise``, ``H = [[1,0,0,0],[0,1,0,0]]``, ``R = I m^2``
(``m = measurement_std_m``). Predict: ``x = F x``, ``P = F P F^T + Q``. Update:
``S = H P H^T + R``, ``K = P H^T S^-1``, ``x += K (z - H x)``, ``P = (I - K H) P``.

Association (after ByteTrack, Zhang et al., "ByteTrack: Multi-Object Tracking by
Associating Every Detection Box", ECCV 2022 - the idea, applied on the ground plane)
-------------------------------------------------------------------------------------
1. Predict every track to the frame time.
2. Match **high-score** detections to tracks: Hungarian assignment on the Mahalanobis
   distance ``d^2 = r^T S^-1 r`` of the innovation ``r``, rejecting pairs above
   ``gate_chi2``.
3. Match the **remaining tracks** to **low-score** detections the same way. A partly
   hidden person still yields a weak detection, and it keeps their track alive instead of
   breaking it into two ids.
4. Offer the high-score detections still unmatched to the tracks still unmatched, under a
   looser gate (``gate_recover_chi2``). A 99% gate rejects ~1% of genuine detections,
   and without this step every such outlier started a duplicate track.
5. Unmatched high-score detections start *tentative* tracks. Low-score ones never do: a
   weak blip is not evidence of a new object.
6. A track is *confirmed* after ``min_hits`` matches, and only confirmed tracks are
   reported. A track unmatched for more than ``max_age_frames`` frames is deleted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from vscs.common.types import Obstacle, ObstacleKind

FloatArray = npt.NDArray[np.float64]
NS_PER_S = 1_000_000_000
_H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])


@dataclass(frozen=True)
class Detection:
    """One detection on the ground, in ``world``: centre, size, class and score."""

    xy: tuple[float, float]
    extent: tuple[float, float, float]
    kind: ObstacleKind
    score: float


@dataclass
class Track:
    id: int
    x: FloatArray  # [px, py, vx, vy]
    P: FloatArray
    kind: ObstacleKind
    extent: tuple[float, float, float]
    t_ns: int
    hits: int = 1
    misses: int = 0
    history: list[int] = field(default_factory=list)

    @property
    def position(self) -> FloatArray:
        return self.x[:2]

    @property
    def velocity(self) -> FloatArray:
        return self.x[2:]


class Tracker:
    def __init__(self, track_cfg: dict[str, Any]) -> None:
        k = track_cfg["kalman"]
        self.accel = float(k["accel_std_mps2"])
        self.R = np.eye(2) * float(k["measurement_std_m"]) ** 2
        self.max_age = int(track_cfg["max_age_frames"])
        self.min_hits = int(track_cfg["min_hits"])
        self.high = float(track_cfg["high_score"])
        self.low = float(track_cfg["low_score"])
        self.gate = float(track_cfg["gate_chi2"])
        self.gate_recover = float(track_cfg["gate_recover_chi2"])
        self.tracks: list[Track] = []
        self._next_id = 0

    # ----------------------------------------------------------- Kalman
    def _predict(self, tr: Track, t_ns: int) -> None:
        dt = (t_ns - tr.t_ns) / NS_PER_S
        if dt <= 0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([dt * dt / 2, dt * dt / 2, dt, dt])
        Q = np.zeros((4, 4))
        for axis in (0, 1):  # x and y accelerations are independent
            g = np.zeros(4)
            g[axis], g[axis + 2] = G[axis], G[axis + 2]
            Q += np.outer(g, g) * self.accel**2
        tr.x = F @ tr.x
        tr.P = F @ tr.P @ F.T + Q
        tr.t_ns = t_ns

    def _innovation(self, tr: Track, z: FloatArray) -> tuple[FloatArray, FloatArray]:
        S = _H @ tr.P @ _H.T + self.R
        return z - _H @ tr.x, S

    def _update(self, tr: Track, det: Detection) -> None:
        r, S = self._innovation(tr, np.asarray(det.xy, dtype=np.float64))
        K = tr.P @ _H.T @ np.linalg.inv(S)
        tr.x = tr.x + K @ r
        tr.P = (np.eye(4) - K @ _H) @ tr.P
        # Size: smooth gently; class: keep the first confident one.
        tr.extent = tuple(0.8 * a + 0.2 * b for a, b in zip(tr.extent, det.extent, strict=True))
        tr.hits += 1
        tr.misses = 0

    # ------------------------------------------------------- association
    def _match(
        self, tracks: list[Track], dets: list[Detection], gate: float | None = None
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Hungarian assignment on Mahalanobis distance, gated. Returns matches, unmatched."""
        gate = self.gate if gate is None else gate
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        cost = np.full((len(tracks), len(dets)), np.inf)
        for i, tr in enumerate(tracks):
            for j, d in enumerate(dets):
                r, S = self._innovation(tr, np.asarray(d.xy, dtype=np.float64))
                cost[i, j] = float(r @ np.linalg.solve(S, r))
        big = 1e9
        rows, cols = linear_sum_assignment(np.where(np.isfinite(cost), cost, big))
        matches = [(i, j) for i, j in zip(rows, cols, strict=True) if cost[i, j] <= gate]
        mt = {i for i, _ in matches}
        md = {j for _, j in matches}
        return (
            matches,
            [i for i in range(len(tracks)) if i not in mt],
            [j for j in range(len(dets)) if j not in md],
        )

    def update(self, detections: list[Detection], t_ns: int) -> list[Track]:
        """Fold in one frame of detections; return the confirmed tracks."""
        for tr in self.tracks:
            self._predict(tr, t_ns)
        high = [d for d in detections if d.score >= self.high]
        low = [d for d in detections if self.low <= d.score < self.high]

        m1, rest_t, rest_high = self._match(self.tracks, high)
        for i, j in m1:
            self._update(self.tracks[i], high[j])
        remaining = [self.tracks[i] for i in rest_t]
        m2, still_t, _ = self._match(remaining, low)
        for i, j in m2:
            self._update(remaining[i], low[j])
        lost = [remaining[i] for i in still_t]
        leftover = [high[j] for j in rest_high]
        # Stage 3: before starting a new track, offer each leftover confident detection to
        # the tracks still unmatched, under a looser gate. Otherwise every rare outlier
        # (a 99% gate rejects ~1% of genuine detections) becomes a duplicate track.
        m3, still_lost, new_dets = self._match(lost, leftover, gate=self.gate_recover)
        for i, j in m3:
            self._update(lost[i], leftover[j])
        for i in still_lost:
            lost[i].misses += 1

        for j in new_dets:  # only confident detections may start a track
            d = leftover[j]
            P0 = np.diag([self.R[0, 0], self.R[1, 1], 4.0, 4.0])  # velocity unknown: wide
            self.tracks.append(
                Track(
                    self._next_id,
                    np.array([d.xy[0], d.xy[1], 0.0, 0.0]),
                    P0,
                    d.kind,
                    d.extent,
                    int(t_ns),
                )
            )
            self._next_id += 1
        self.tracks = [tr for tr in self.tracks if tr.misses <= self.max_age]
        for tr in self.tracks:
            tr.history.append(int(t_ns))
        return self.confirmed()

    @property
    def next_id(self) -> int:
        """The id the next new track will get (ids are never reused)."""
        return self._next_id

    def confirmed(self) -> list[Track]:
        return [tr for tr in self.tracks if tr.hits >= self.min_hits]

    def obstacles(
        self, T_veh_world: npt.ArrayLike, t_ns: int, pos_sigma_m: float
    ) -> list[Obstacle]:
        """Confirmed tracks as §4.2 ``Obstacle`` objects in the current ``veh`` frame."""
        T = np.asarray(T_veh_world, dtype=np.float64)
        out = []
        for tr in self.confirmed():
            c = T @ np.array([tr.x[0], tr.x[1], tr.extent[2] / 2.0, 1.0])
            v = T[:3, :3] @ np.array([tr.x[2], tr.x[3], 0.0])
            out.append(
                Obstacle(
                    id=tr.id,
                    t_ns=int(t_ns),
                    kind=tr.kind,
                    center_veh=(float(c[0]), float(c[1]), float(c[2])),
                    extent=tuple(float(e) for e in tr.extent),
                    velocity_veh=(float(v[0]), float(v[1]), float(v[2])),
                    pos_sigma_m=float(pos_sigma_m),
                    source="detector",
                )
            )
        return out
