"""Motion model tests - the P3-T3 acceptance criterion.

"Fixture test: straight + arc paths correct" (CLAUDE.md §6). The known answers are
closed-form: a straight path is ``x = v t``, and an arc of curvature ``kappa`` is a circle
of radius ``1/kappa`` centred at ``(0, 1/kappa)``, whose quarter turn lands at
``(R, R)`` with heading ``pi/2``.
"""

from __future__ import annotations

import numpy as np
import pytest

from vscs.common import frames as F
from vscs.common.config import load_config
from vscs.risk import motion as M

EXACT = 1e-9


# --------------------------------------------------------------------------- #
# Straight paths                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("speed", [1.0, -1.0, 0.5, -2.3])
def test_straight_path_is_x_equals_vt(speed):
    fan = M.path_fan(speed, horizon_s=3.0, dt_s=0.1, curvatures=[0.0])
    np.testing.assert_allclose(fan.x[0], speed * fan.t_s, atol=EXACT)
    np.testing.assert_allclose(fan.y[0], 0.0, atol=EXACT)
    np.testing.assert_allclose(fan.heading[0], 0.0, atol=EXACT)


def test_reversing_moves_backwards():
    """VSCS is a reversing system: negative speed must move the vehicle in -x."""
    fan = M.path_fan(-1.0, horizon_s=2.0, dt_s=0.5, curvatures=[0.0])
    assert fan.x[0, -1] == pytest.approx(-2.0, abs=EXACT)


# --------------------------------------------------------------------------- #
# Arcs                                                                         #
# --------------------------------------------------------------------------- #
def test_quarter_turn_lands_at_R_R_with_heading_pi_over_2():
    """kappa = 0.2 => R = 5 m. A quarter circle is s = (pi/2) R."""
    kappa, R = 0.2, 5.0
    x, y, h = M.constant_curvature_pose(kappa, np.pi / 2 * R)
    assert float(x) == pytest.approx(R, abs=EXACT)
    assert float(y) == pytest.approx(R, abs=EXACT)
    assert float(h) == pytest.approx(np.pi / 2, abs=EXACT)


def test_half_turn_lands_at_0_2R_facing_backwards():
    kappa, R = 0.2, 5.0
    x, y, h = M.constant_curvature_pose(kappa, np.pi * R)
    assert float(x) == pytest.approx(0.0, abs=1e-9)
    assert float(y) == pytest.approx(2 * R, abs=1e-9)
    assert float(h) == pytest.approx(np.pi, abs=EXACT)


@pytest.mark.parametrize("kappa", [0.05, 0.12, 0.2, -0.05, -0.2])
@pytest.mark.parametrize("speed", [1.5, -1.5])
def test_every_point_lies_on_the_turning_circle(kappa, speed):
    """Rear-axle centre stays on the circle of radius 1/kappa centred at (0, 1/kappa)."""
    R = 1.0 / kappa
    fan = M.path_fan(speed, horizon_s=3.0, dt_s=0.1, curvatures=[kappa])
    radius = np.hypot(fan.x[0], fan.y[0] - R)
    np.testing.assert_allclose(radius, abs(R), atol=1e-9)


def test_positive_curvature_turns_left():
    """y points left in veh (REP-103), so kappa > 0 going forward must gain +y."""
    fan = M.path_fan(1.0, horizon_s=2.0, dt_s=0.1, curvatures=[0.2])
    assert fan.y[0, -1] > 0
    assert fan.heading[0, -1] > 0


def test_curvature_sign_mirrors_the_path():
    left = M.path_fan(-1.0, horizon_s=3.0, dt_s=0.1, curvatures=[0.12])
    right = M.path_fan(-1.0, horizon_s=3.0, dt_s=0.1, curvatures=[-0.12])
    np.testing.assert_allclose(left.x, right.x, atol=EXACT)
    np.testing.assert_allclose(left.y, -right.y, atol=EXACT)
    np.testing.assert_allclose(left.heading, -right.heading, atol=EXACT)


def test_heading_rate_equals_kappa_times_speed():
    """d(theta)/dt = kappa * v - the defining property of a constant-curvature path."""
    kappa, v = 0.12, -1.2
    fan = M.path_fan(v, horizon_s=3.0, dt_s=0.1, curvatures=[kappa])
    rate = np.diff(fan.heading[0]) / np.diff(fan.t_s)
    np.testing.assert_allclose(rate, kappa * v, atol=1e-9)


def test_travelled_arc_length_matches_speed():
    """Chord between samples is slightly shorter than the arc |v| dt, never longer."""
    fan = M.path_fan(-1.0, horizon_s=3.0, dt_s=0.1, curvatures=[0.2])
    chords = np.hypot(np.diff(fan.x[0]), np.diff(fan.y[0]))
    assert np.all(chords <= 0.1 + 1e-12)
    assert np.all(chords > 0.1 * 0.999)


# --------------------------------------------------------------------------- #
# The kappa -> 0 limit                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kappa", [1e-12, 1e-9, -1e-9, 1e-6])
def test_tiny_curvature_converges_to_straight(kappa):
    """No if-branch for straight: the arc formulas must degrade continuously."""
    s = np.linspace(-3, 3, 13)
    x, y, _ = M.constant_curvature_pose(kappa, s)
    np.testing.assert_allclose(x, s, atol=1e-6)
    np.testing.assert_allclose(y, 0.0, atol=1e-5)
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))


def test_series_and_closed_form_agree_across_the_threshold():
    """No jump where the implementation switches from Taylor series to closed form."""
    u = np.array([0.5, 0.9, 0.99, 1.01, 1.1, 2.0]) * M._SERIES_THRESHOLD
    closed_sinc = np.sin(u) / u
    closed_cosc = (1 - np.cos(u)) / u
    np.testing.assert_allclose(M._sinc(u), closed_sinc, rtol=1e-9)
    # The closed form of cosc is itself noisy here; the series is the reference.
    np.testing.assert_allclose(M._cosc(u), u / 2 - u**3 / 24, rtol=1e-6)
    assert np.isfinite(closed_cosc).all()


# --------------------------------------------------------------------------- #
# Transforms                                                                   #
# --------------------------------------------------------------------------- #
def test_first_pose_is_identity():
    fan = M.path_fan(-1.0, horizon_s=3.0, dt_s=0.1, curvatures=[-0.2, 0.0, 0.2])
    for k in range(fan.n_curvatures):
        np.testing.assert_allclose(fan.T_veh0_veh(k, 0), np.eye(4), atol=EXACT)


def test_transforms_are_rigid_and_carry_the_origin_to_the_path():
    fan = M.path_fan(-1.0, horizon_s=3.0, dt_s=0.1, curvatures=[0.12])
    Ts = fan.transforms(0)
    assert Ts.shape == (fan.n_steps, 4, 4)
    for i, T in enumerate(Ts):
        R, t = F.Rt_from_T(T)
        assert F.is_rotation(R)
        assert t[2] == 0.0  # planar model: never leaves the ground
        # The veh origin (rear-axle centre) at step i sits at (x_i, y_i) in veh0.
        p = F.transform_points(T, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(p[:2], [fan.x[0, i], fan.y[0, i]], atol=EXACT)


def test_a_point_on_the_rear_bumper_moves_as_the_rigid_body_does():
    """A body point 1 m behind the axle, straight reverse at 1 m/s for 1 s: x = -2."""
    fan = M.path_fan(-1.0, horizon_s=1.0, dt_s=0.5, curvatures=[0.0])
    p = F.transform_points(fan.T_veh0_veh(0, fan.n_steps - 1), [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(p, [-2.0, 0.0, 0.0], atol=EXACT)


# --------------------------------------------------------------------------- #
# Config wiring and helpers                                                    #
# --------------------------------------------------------------------------- #
def test_fan_from_config_matches_risk_yaml():
    cfg = load_config("risk")
    fan = M.path_fan_from_config(-1.0, cfg)
    assert fan.n_curvatures == len(cfg["motion"]["curvatures_inv_m"]) == 7
    horizon, dt = cfg["motion"]["horizon_s"], cfg["motion"]["dt_s"]
    assert fan.n_steps == round(horizon / dt) + 1
    assert fan.t_s[-1] == pytest.approx(horizon)


def test_step_times_are_int64_nanoseconds():
    fan = M.path_fan(-1.0, horizon_s=1.0, dt_s=0.1, curvatures=[0.0])
    assert fan.t_ns.dtype == np.int64
    assert fan.t_ns[1] == 100_000_000


def test_index_of_curvature():
    fan = M.path_fan(-1.0, horizon_s=1.0, dt_s=0.1, curvatures=[-0.2, 0.0, 0.2])
    assert fan.index_of_curvature(0.0) == 1
    assert fan.index_of_curvature(0.17) == 2
    assert fan.index_of_curvature(-5.0) == 0


def test_is_moving_uses_the_configured_threshold():
    cfg = load_config("risk")
    threshold = cfg["motion"]["min_speed_mps"]
    assert not M.is_moving(0.0, cfg)
    assert not M.is_moving(threshold * 0.5, cfg)
    assert M.is_moving(-threshold * 2, cfg)


def test_curvature_from_yaw_rate():
    assert M.curvature_from_yaw_rate(0.24, -1.2) == pytest.approx(-0.2)
    assert M.curvature_from_yaw_rate(0.5, 0.0) == 0.0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"horizon_s": 3.0, "dt_s": 0.0, "curvatures": [0.0]}, "dt_s"),
        ({"horizon_s": -1.0, "dt_s": 0.1, "curvatures": [0.0]}, "horizon_s"),
        ({"horizon_s": 3.0, "dt_s": 0.1, "curvatures": []}, "empty"),
    ],
)
def test_rejects_bad_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        M.path_fan(-1.0, **kwargs)


# --------------------------------------------------------------------------- #
# Truncation to the reachable horizon (ADR 0005)                               #
# --------------------------------------------------------------------------- #
def test_truncated_ends_exactly_at_t_max():
    fan = M.path_fan(-1.0, horizon_s=3.0, dt_s=0.1, curvatures=[-0.2, 0.0, 0.2])
    cut = fan.truncated(0.37)
    assert cut.t_s[-1] == pytest.approx(0.37)
    np.testing.assert_allclose(cut.t_s[:-1], [0.0, 0.1, 0.2, 0.3])
    assert cut.n_curvatures == 3


def test_truncated_final_pose_is_the_exact_continuous_pose():
    fan = M.path_fan(-1.0, horizon_s=3.0, dt_s=0.1, curvatures=[0.12])
    cut = fan.truncated(1.234)
    x, y, h = M.constant_curvature_pose(0.12, -1.234)
    assert cut.x[0, -1] == pytest.approx(float(x), abs=EXACT)
    assert cut.y[0, -1] == pytest.approx(float(y), abs=EXACT)
    assert cut.heading[0, -1] == pytest.approx(float(h), abs=EXACT)


def test_truncated_on_a_step_boundary_does_not_duplicate_it():
    fan = M.path_fan(-1.0, horizon_s=3.0, dt_s=0.1, curvatures=[0.0])
    cut = fan.truncated(0.3)
    np.testing.assert_allclose(cut.t_s, [0.0, 0.1, 0.2, 0.3], atol=1e-12)


def test_truncated_at_zero_is_the_current_pose_only():
    cut = M.path_fan(-1.0, horizon_s=3.0, dt_s=0.1, curvatures=[0.0]).truncated(0.0)
    assert cut.n_steps == 1
    np.testing.assert_allclose(cut.T_veh0_veh(0, 0), np.eye(4), atol=EXACT)


def test_truncating_beyond_the_horizon_is_a_no_op():
    fan = M.path_fan(-1.0, horizon_s=3.0, dt_s=0.1, curvatures=[0.0])
    assert fan.truncated(5.0) is fan


def test_truncated_rejects_negative_time():
    with pytest.raises(ValueError, match="non-negative"):
        M.path_fan(-1.0, horizon_s=3.0, dt_s=0.1, curvatures=[0.0]).truncated(-0.1)
