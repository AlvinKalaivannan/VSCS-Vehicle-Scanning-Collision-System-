"""Data contracts shared between VSCS modules.

These schemas are the *only* way modules talk to each other (CLAUDE.md section 2).
No module imports another module's internals, so a change here is a change to every
stage at once.

Changing a schema requires all three of (CLAUDE.md section 4.2):

1. a bump of :data:`SCHEMA_VERSION`,
2. a migration note in an ADR,
3. updated tests.

Models are pydantic v2 with ``extra="forbid"``: a typo'd or stale field is an error
at construction time rather than a silently ignored key that shows up three stages
later as a wrong number (risk R-16). See ADR 0002.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

#: Bump on ANY change to the models below. Serialized streams carry it so a stale
#: file is recognisable rather than silently misread.
SCHEMA_VERSION = 1

AlertLevel = Literal["none", "caution", "warning", "critical"]
ObstacleKind = Literal["static_geom", "person", "vehicle", "cyclist", "unknown"]
JointType = Literal["revolute", "prismatic", "fixed", "continuous"]

Vec3 = tuple[float, float, float]

#: One second in nanoseconds. Timestamps are int64 ns everywhere (CLAUDE.md 4.1).
NS_PER_S = 1_000_000_000


class _Base(BaseModel):
    """Shared config: reject unknown fields, validate on assignment too."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------- #
# Timestamps                                                                   #
# --------------------------------------------------------------------------- #
def validate_timestamp_stream(t_ns: list[int], *, name: str = "stream") -> None:
    """Raise if a timestamp sequence is unusable. Enforces CLAUDE.md 4.1 and R-03.

    Phones record at a variable frame rate, so the two ways this goes wrong are:

    * **frame index used as time** - the giveaway is a sequence that increments by
      about 1 per frame, which at nanosecond scale would be a frame every billionth
      of a second;
    * **non-monotonic timestamps** - out-of-order or duplicated container
      timestamps, which make every velocity and time-to-contact downstream wrong.

    Used by ingest (P1-T4, "timestamp monotonic check passes").
    """
    if len(t_ns) < 2:
        return
    arr = list(t_ns)
    for i in range(1, len(arr)):
        if arr[i] <= arr[i - 1]:
            raise ValueError(
                f"{name}: timestamps are not strictly increasing at index {i}: "
                f"{arr[i - 1]} -> {arr[i]}. Use container presentation timestamps, "
                "never frame index (CLAUDE.md 4.1, R-03)."
            )
    span_ns = arr[-1] - arr[0]
    # A real clip spanning N frames covers far more than N nanoseconds. If the whole
    # span is smaller than a millisecond, these are almost certainly frame indices.
    if span_ns < NS_PER_S // 1000:
        raise ValueError(
            f"{name}: {len(arr)} timestamps span only {span_ns} ns. These look like "
            "frame indices or seconds, not int64 nanoseconds (R-03)."
        )


# --------------------------------------------------------------------------- #
# Vehicle model (output of the offline pipeline: capture -> recon -> seg -> model)
# --------------------------------------------------------------------------- #
class Component(_Base):
    """One mechanical component of the scanned vehicle."""

    name: str = Field(description="Machine name, e.g. 'rear_right_bumper_corner'.")
    display_name: str = Field(description="Driver-facing name, e.g. 'rear right corner'.")
    link: str = Field(description="URDF link name.")
    severity: float = Field(ge=0.0, description="Default unitless cost, from severity.yaml.")
    min_z: float = Field(
        description="Lowest point of the component in the veh frame, metres. Used for "
        "underbody clearance checks (P4-T6). May be negative if the "
        "reconstruction dips below the fitted ground plane."
    )


class JointSpec(_Base):
    """A hinge or slide for an articulated part (door, mirror).

    Not defined in CLAUDE.md section 4.2, which references ``JointSpec`` from
    ``ComponentModel`` without giving its fields. Defined here to match URDF joint
    semantics so export is a direct mapping. See ADR 0002.

    ``origin_xyz`` / ``origin_rpy`` place the joint frame in the parent link's frame,
    exactly as URDF's ``<origin>`` does. ``axis`` is a direction in the joint frame,
    so it is rotated, never translated (see ``frames.rotate_vectors``).

    Limits are radians for revolute joints and metres for prismatic ones.
    """

    type: JointType
    parent_link: str
    child_link: str
    axis: Vec3 = (0.0, 0.0, 1.0)
    origin_xyz: Vec3 = (0.0, 0.0, 0.0)
    origin_rpy: Vec3 = (0.0, 0.0, 0.0)
    limit_lower: float = 0.0
    limit_upper: float = 0.0

    @field_validator("axis")
    @classmethod
    def _axis_non_zero(cls, v: Vec3) -> Vec3:
        if all(abs(c) < 1e-12 for c in v):
            raise ValueError("joint axis must be a non-zero direction")
        return v

    @model_validator(mode="after")
    def _limits_ordered(self) -> JointSpec:
        if self.type in ("revolute", "prismatic") and self.limit_upper < self.limit_lower:
            raise ValueError(
                f"joint limits out of order: upper {self.limit_upper} < lower {self.limit_lower}"
            )
        return self


class ComponentModel(_Base):
    """The exported collision model: URDF plus per-component metadata (P2-T8)."""

    urdf_path: Path
    components: list[Component]
    joints: dict[str, JointSpec] = Field(default_factory=dict)
    scale_error_m: float = Field(
        ge=0.0,
        description="Worst dimensional error from recon validation (P1-T6). "
        "Acceptance gate is 0.02 m.",
    )
    schema_version: int = SCHEMA_VERSION

    @field_validator("components")
    @classmethod
    def _names_unique(cls, v: list[Component]) -> list[Component]:
        names = [c.name for c in v]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate component names: {sorted(dupes)}")
        return v

    def by_name(self, name: str) -> Component:
        """Look up a component, with a helpful error listing what does exist."""
        for c in self.components:
            if c.name == name:
                return c
        raise KeyError(f"no component {name!r}; have {[c.name for c in self.components]}")


# --------------------------------------------------------------------------- #
# Perception output                                                            #
# --------------------------------------------------------------------------- #
class Obstacle(_Base):
    """One obstacle around the vehicle at one instant, in the veh frame."""

    id: int
    t_ns: StrictInt = Field(description="int64 nanoseconds. Never a frame index (R-03).")
    kind: ObstacleKind
    center_veh: Vec3
    extent: Vec3 = Field(description="Full box extent (dx, dy, dz) in metres, not half-extent.")
    velocity_veh: Vec3 = (0.0, 0.0, 0.0)
    pos_sigma_m: float = Field(
        ge=0.0,
        description="1-sigma position uncertainty. Grows with range (perception.yaml).",
    )
    source: Literal["occupancy", "detector"]

    @field_validator("t_ns")
    @classmethod
    def _t_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("t_ns must be non-negative")
        return v

    @field_validator("extent")
    @classmethod
    def _extent_positive(cls, v: Vec3) -> Vec3:
        if any(c <= 0 for c in v):
            raise ValueError(f"extent must be positive in every axis, got {v}")
        return v


# --------------------------------------------------------------------------- #
# Risk output                                                                  #
# --------------------------------------------------------------------------- #
class ComponentRisk(_Base):
    """Risk for one component against its closest-threatening obstacle."""

    component: str
    min_distance_m: float
    ttc_s: float | None = Field(
        default=None,
        description="None means no predicted contact within the horizon. "
        "This is NOT the same as a large TTC and must not be "
        "coerced to one.",
    )
    p_contact: float = Field(ge=0.0, le=1.0)
    severity: float = Field(ge=0.0)
    risk: float = Field(ge=0.0, description="p_contact * severity * speed_factor.")
    obstacle_id: int | None = None

    @field_validator("ttc_s")
    @classmethod
    def _ttc_non_negative(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("ttc_s must be non-negative or None")
        return v


def p_any_contact(probabilities: list[float]) -> float:
    """Probability that *at least one* component makes contact.

    ``p_any = 1 - prod(1 - p_i)``, i.e. the complement of "every component stays
    clear". This assumes the per-component events are independent, which they are
    not - two adjacent components usually face the same obstacle, so the true value
    is lower than this. It therefore errs toward over-warning, which is the right
    direction for an advisory system, but it is an approximation and is reported as
    one (CLAUDE.md section 9).

    Computed in the complement domain so that many small probabilities do not lose
    precision the way ``1 - (1 - p)`` repeated would.
    """
    q = 1.0
    for p in probabilities:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"probability out of range: {p}")
        q *= 1.0 - p
    return 1.0 - q


class RiskFrame(_Base):
    """Per-component risk at one instant. Serialized as JSON Lines."""

    t_ns: StrictInt
    ego_speed_mps: float
    per_component: list[ComponentRisk] = Field(default_factory=list)
    p_any_contact: float = Field(ge=0.0, le=1.0, description="1 - prod(1 - p_i).")
    expected_damage: float = Field(ge=0.0, description="Sum of risk_i.")
    worst_component: str | None = None
    alert_level: AlertLevel = "none"
    schema_version: int = SCHEMA_VERSION

    @field_validator("t_ns")
    @classmethod
    def _t_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("t_ns must be non-negative")
        return v

    @model_validator(mode="after")
    def _worst_component_exists(self) -> RiskFrame:
        if self.worst_component is not None and self.per_component:
            names = {c.component for c in self.per_component}
            if self.worst_component not in names:
                raise ValueError(
                    f"worst_component {self.worst_component!r} is not in per_component {names}"
                )
        return self

    @classmethod
    def from_components(
        cls,
        t_ns: int,
        ego_speed_mps: float,
        per_component: list[ComponentRisk],
        alert_level: AlertLevel = "none",
    ) -> RiskFrame:
        """Build a frame, deriving the aggregate fields from the per-component list.

        Only the aggregation the schema itself defines lives here. Choosing the
        speed factor, looking up severity and running the alert hysteresis belong to
        ``risk/aggregate.py`` and ``risk/alerts.py`` (P3-T5).
        """
        worst = max(per_component, key=lambda c: c.risk).component if per_component else None
        return cls(
            t_ns=t_ns,
            ego_speed_mps=ego_speed_mps,
            per_component=per_component,
            p_any_contact=p_any_contact([c.p_contact for c in per_component]),
            expected_damage=float(sum(c.risk for c in per_component)),
            worst_component=worst,
            alert_level=alert_level,
        )
