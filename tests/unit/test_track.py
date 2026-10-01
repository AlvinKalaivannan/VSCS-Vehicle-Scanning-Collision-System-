"""P4-T4 tracking: known-answer scenarios on the ground plane (seeded noise)."""

from __future__ import annotations

import numpy as np
import pytest

from vscs.common.config import load_config
from vscs.common.frames import T_from_Rt, rot_z
from vscs.perception.track import Detection, Tracker

CFG = load_config("perception")["track"]
NOISE = float(CFG["kalman"]["measurement_std_m"])
S = 1_000_000_000
DT = 0.1  # 10 Hz
PERSON = (0.5, 0.5, 1.7)


def _det(xy, score=0.9):
    return Detection(xy=(float(xy[0]), float(xy[1])), extent=PERSON, kind="person", score=score)


def _walk(start, vel, n, rng):
    return [
        np.asarray(start) + np.asarray(vel) * k * DT + rng.normal(0, NOISE, 2) for k in range(n)
    ]


def test_velocity_of_a_walker_converges():
    """A person walking 1.2 m/s across behind the van (the P4-T4 staged clip, van still)."""
    rng = np.random.default_rng(1)
    tr = Tracker(CFG)
    for k, z in enumerate(_walk((-4.0, 3.0), (0.0, -1.2), 40, rng)):
        tracks = tr.update([_det(z)], round(k * DT * S))
    (t,) = tracks
    np.testing.assert_allclose(t.velocity, [0.0, -1.2], atol=0.15)
    np.testing.assert_allclose(t.position, [-4.0, 3.0 - 1.2 * 39 * DT], atol=2 * NOISE)


def test_two_people_crossing_keep_their_ids():
    rng = np.random.default_rng(2)
    a = _walk((-4.0, 2.0), (0.0, -1.0), 50, rng)
    b = _walk((-4.6, -2.0), (0.0, 1.0), 50, rng)  # passes 0.6 m from A at the crossing
    tr = Tracker(CFG)
    ids_a, ids_b = set(), set()
    for k in range(50):
        tr.update([_det(a[k]), _det(b[k])], round(k * DT * S))
        for t in tr.confirmed():
            (ids_a if t.position[0] > -4.3 else ids_b).add(t.id)
    assert len(ids_a) == 1 and len(ids_b) == 1 and ids_a != ids_b


def test_weak_detections_bridge_an_occlusion_but_never_start_a_track():
    rng = np.random.default_rng(3)
    path = _walk((-5.0, 2.0), (0.0, -1.0), 40, rng)
    tr = Tracker(CFG)
    for k in range(40):
        score = 0.2 if 15 <= k < 25 else 0.9  # partly hidden for 1 s: weak detections only
        tr.update([_det(path[k], score)], round(k * DT * S))
    assert len({t.id for t in tr.tracks}) == 1  # one object, one id, throughout
    lone = Tracker(CFG)
    for k in range(10):
        lone.update([_det((-3.0, 0.0), 0.2)], round(k * DT * S))
    assert lone.tracks == []  # low-score blips alone never become a track


def test_a_single_false_detection_is_never_reported():
    tr = Tracker(CFG)
    tr.update([_det((-3.0, 0.0))], 0)  # tentative
    for k in range(1, 5):
        out = tr.update([], round(k * DT * S))
    assert out == []
    assert all(t.hits < CFG["min_hits"] for t in tr.tracks)


def test_a_track_is_deleted_after_max_age():
    tr = Tracker(CFG)
    for k in range(5):
        tr.update([_det((-3.0, 0.0))], round(k * DT * S))
    assert len(tr.confirmed()) == 1
    for k in range(5, 5 + CFG["max_age_frames"] + 1):
        tr.update([], round(k * DT * S))
    assert tr.tracks == []


def test_obstacles_are_reported_in_the_vehicle_frame_with_velocity():
    tr = Tracker(CFG)
    for k in range(10):
        tr.update([_det((-3.0, 1.0 - 1.0 * k * DT))], round(k * DT * S))
    # Van at world (1, 0), turned 90 deg left: world -y is veh +x... check by construction.
    T_world_veh = T_from_Rt(rot_z(np.pi / 2), [1.0, 0.0, 0.0])
    (ob,) = tr.obstacles(np.linalg.inv(T_world_veh), round(9 * DT * S), pos_sigma_m=0.2)
    p_world = np.array([-3.0, 1.0 - 0.9, PERSON[2] / 2, 1.0])
    np.testing.assert_allclose(ob.center_veh, (np.linalg.inv(T_world_veh) @ p_world)[:3], atol=0.05)
    # World velocity (0, -1) seen from a frame rotated +90 deg: veh velocity (-1, 0).
    np.testing.assert_allclose(ob.velocity_veh[:2], [-1.0, 0.0], atol=0.2)
    assert ob.kind == "person" and ob.source == "detector" and ob.pos_sigma_m == pytest.approx(0.2)


def test_one_outlier_measurement_does_not_spawn_a_duplicate():
    """The defect behind the crossing failure: a single 3-sigma detection (beyond the 99%
    gate) must update its track, not start a second one."""
    tr = Tracker(CFG)
    for k in range(10):
        tr.update([_det((-4.0, 2.0 - 1.0 * k * DT))], round(k * DT * S))
    z = (-4.0 + 3.2 * NOISE, 2.0 - 1.0 * 10 * DT + 2.5 * NOISE)  # far out, still the same person
    tr.update([_det(z)], round(10 * DT * S))
    assert len(tr.tracks) == 1


def test_kalman_filter_is_statistically_consistent_nees():
    """Monte Carlo NEES check (Bar-Shalom): simulate the filter's own model - white
    acceleration with std accel_std, position measurements with std measurement_std - and
    the normalised state error e^T P^-1 e must average ~4 (chi-square, 4 dof). A wrong F,
    Q or update shifts it away from 4; this is the strongest test of the maths."""
    from vscs.perception.track import Track

    rng = np.random.default_rng(11)
    tr = Tracker(CFG)
    a_std = CFG["kalman"]["accel_std_mps2"]
    runs, steps, dt = 300, 30, 0.1
    nees = []
    for _ in range(runs):
        x = np.array([0.0, 0.0, 1.0, -0.5])
        z0 = x[:2] + rng.normal(0, NOISE, 2)
        P0 = np.diag([NOISE**2, NOISE**2, 4.0, 4.0])
        trk = Track(0, np.array([z0[0], z0[1], 0.0, 0.0]), P0, "person", PERSON, 0)
        # True initial velocity drawn from the prior, so the start is consistent too.
        x[2:] = rng.normal(0, 2.0, 2)
        for k in range(1, steps + 1):
            acc = rng.normal(0, a_std, 2)
            x = np.array(
                [
                    x[0] + x[2] * dt + 0.5 * acc[0] * dt**2,
                    x[1] + x[3] * dt + 0.5 * acc[1] * dt**2,
                    x[2] + acc[0] * dt,
                    x[3] + acc[1] * dt,
                ]
            )
            tr._predict(trk, round(k * dt * S))
            tr._update(trk, _det(x[:2] + rng.normal(0, NOISE, 2)))
        e = trk.x - x
        nees.append(float(e @ np.linalg.solve(trk.P, e)))
    mean = float(np.mean(nees))
    # chi2(4): mean 4, std of the mean over 300 runs ~ sqrt(8/300) = 0.16 -> a 4-sigma band.
    assert 3.35 < mean < 4.65, mean
