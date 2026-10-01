"""Single-bounding-box baseline: the van as ONE oriented box (P5-T1).

Acceptance (CLAUDE.md §6): runs on the same splits. Comparing VSCS against this baseline
on every risk metric is the core result, and is never cut.

The baseline is the *same* pipeline, with one change: instead of one footprint per
component, the vehicle is one footprint, the minimum-area rectangle around every
component's plan outline (``shapely.minimum_rotated_rectangle``), from the lowest to the
highest point. It is one component, named by ``eval.yaml baseline.name``, with severity
``baseline.severity``.

What it gets wrong, by construction (shown by the fixture tests):

* **It invents clearance loss.** Its sides run along the furthest-protruding parts (the
  mirrors), so an obstacle level with a flat body panel is reported closer than it is.
  On the fixture, a pole 0.40 m from the sliding door is 0.15 m from the box.
* **It cannot attribute.** Every alert names "the vehicle", so component attribution
  accuracy is not defined for it. Report "n/a", never 0 or 1.

Everything else (motion model, sweep, TTC, alerts) is shared, so any difference in the
metrics comes from the geometry alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shapely.geometry import Polygon
from shapely.ops import unary_union


@dataclass(frozen=True)
class BoxFootprint:
    """Duck-types the footprint the risk engine sweeps (name, polygon, z range)."""

    name: str
    polygon: Polygon
    z_min: float
    z_max: float


def single_box_footprint(
    footprints: dict[str, tuple[Polygon, float, float]], name: str
) -> BoxFootprint:
    """The minimum-area rectangle around every component footprint (``urdf.plan_footprints``)."""
    if not footprints:
        raise ValueError("no component footprints to enclose")
    outline = unary_union([fp[0] for fp in footprints.values()])
    rect = outline.minimum_rotated_rectangle
    z_min = min(fp[1] for fp in footprints.values())
    z_max = max(fp[2] for fp in footprints.values())
    return BoxFootprint(name=name, polygon=rect, z_min=z_min, z_max=z_max)


def baseline_severity_cfg(
    severity_cfg: dict[str, Any], baseline_cfg: dict[str, Any]
) -> dict[str, Any]:
    """``severity.yaml`` with its components replaced by the single baseline component.

    Obstacle multipliers are kept (a person still outranks a cone); per-component
    overrides are dropped, because the baseline has no components to override.
    """
    return {
        **severity_cfg,
        "components": {str(baseline_cfg["name"]): float(baseline_cfg["severity"])},
        "overrides": {},
    }
