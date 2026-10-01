"""Wheelbase from the segmented wheels, for P1-T6's wheelbase check (ADR 0013).

The vehicle frame (P1-T6) has no wheel centres, so its own step cannot validate the
wheelbase. Once the scan is segmented (P2-T5), each wheel is a labelled set of points in
``veh``. A wheel and tyre seen from the side is a disc in the x-z plane, so its hub's x is
the middle of its x-extent. The extent is trimmed at both ends, so a few stray labels
cannot stretch it. The wheel arch hides the top of the tyre and the ground hides the
bottom, but neither changes the x-extent, which is set at hub height.

``wheelbase = mean front-hub x - mean rear-hub x``.

Two diagnostics come with it, so a bad number can be traced:

* **left/right disagreement** per axle, which points at segmentation;
* **rear axle x**, which should be 0 by definition of ``veh`` (origin below the rear axle,
  placed from the tape-measured rear overhang), which points at the frame or the tape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Wheelbase:
    wheelbase_m: float
    hub_x_m: dict[str, float]
    front_left_right_gap_m: float
    rear_left_right_gap_m: float
    rear_axle_x_m: float


def hub_x(points_x: npt.ArrayLike, trim_percentile: float) -> float:
    x = np.asarray(points_x, dtype=np.float64)
    if x.size < 10:
        raise ValueError(f"only {x.size} wheel points: too few to place a hub")
    lo, hi = np.percentile(x, [trim_percentile, 100.0 - trim_percentile])
    return float((lo + hi) / 2.0)


def wheelbase_from_wheels(
    points: npt.ArrayLike,
    labels: npt.ArrayLike,
    names: list[str],
    front_wheels: list[str],
    rear_wheels: list[str],
    trim_percentile: float,
) -> Wheelbase:
    pts = np.asarray(points, dtype=np.float64)
    lab = np.asarray(labels)
    missing = [w for w in (*front_wheels, *rear_wheels) if w not in names]
    if missing:
        raise ValueError(f"wheel components {missing} are not in the labelled cloud")
    hubs = {
        w: hub_x(pts[lab == names.index(w), 0], trim_percentile)
        for w in (*front_wheels, *rear_wheels)
    }
    front = [hubs[w] for w in front_wheels]
    rear = [hubs[w] for w in rear_wheels]
    return Wheelbase(
        wheelbase_m=float(np.mean(front) - np.mean(rear)),
        hub_x_m=hubs,
        front_left_right_gap_m=float(np.ptp(front)),
        rear_left_right_gap_m=float(np.ptp(rear)),
        rear_axle_x_m=float(np.mean(rear)),
    )
