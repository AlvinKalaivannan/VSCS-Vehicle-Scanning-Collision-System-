"""Rigid-body transforms, rotations, and pinhole projection.

This module is the single place where CLAUDE.md section 4.1 conventions are
implemented. Every other module depends on these being right, so read this before
changing any of it: a silent convention change breaks every downstream stage at once
(risk R-16).

Conventions (do not change without an ADR and the developer's approval)
-----------------------------------------------------------------------
Units are metres, seconds, radians. All frames are right-handed.

    veh     x forward, y left, z up (ROS REP-103). Origin on the ground plane
            directly below the centre of the rear axle.
    cam     x right, y down, z forward (OpenCV).
    world   fixed at the vehicle pose at the start of a drive.

Naming: ``T_a_b`` maps points *expressed in frame b* into *frame a*:

    p_a = T_a_b @ p_b

and composition chains adjacent subscripts, reading right to left:

    T_a_c = T_a_b @ T_b_c

The mental model (MAT188 notation)
----------------------------------
A rigid transform is a linear map on *homogeneous* coordinates. Write a point as
p~ = [x, y, z, 1]^T in R^4. Then

    T = [ R  t ]        with R in SO(3), t in R^3
        [ 0  1 ]

and T p~ = [Rp + t, 1]^T: rotate, then translate.

The fourth row and the appended 1 exist purely so that a translation - which is
*not* a linear map on R^3, because it moves the origin - becomes a linear map on
R^4. That is the whole trick, and it is why transforms compose by plain matrix
multiplication.

Two consequences used constantly below.

**Inversion is free.** R is orthogonal (R^T R = I), so R^-1 = R^T. Solving
q = Rp + t for p gives p = R^T q - R^T t, i.e.

    T^-1 = [ R^T  -R^T t ]
           [ 0     1     ]

No general 4x4 inverse is ever needed, and using one (``np.linalg.inv``) is both
slower and numerically worse. :func:`invert` uses the closed form.

**Order matters.** T_ab T_bc != T_bc T_ab, because SO(3) is not commutative. The
subscript-cancellation rule is the guard rail: if the inner subscripts do not match,
the product is meaningless even though numpy will happily compute it. Hence the
``T_a_b`` naming discipline in every signature here.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

#: Tolerance used when checking that a matrix is a valid rotation.
ROTATION_TOL = 1e-8

#: Points with camera-frame depth at or below this are at/behind the pinhole and
#: cannot be projected.
MIN_PROJECTION_DEPTH_M = 1e-6


def _as_float(a: npt.ArrayLike) -> FloatArray:
    """Return ``a`` as a contiguous float64 array."""
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64))


# --------------------------------------------------------------------------- #
# Rotations                                                                    #
# --------------------------------------------------------------------------- #
def rot_x(angle_rad: float) -> FloatArray:
    """Rotation by ``angle_rad`` about the x axis (right-handed)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle_rad: float) -> FloatArray:
    """Rotation by ``angle_rad`` about the y axis (right-handed)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle_rad: float) -> FloatArray:
    """Rotation by ``angle_rad`` about the z axis (right-handed)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def is_rotation(R: npt.ArrayLike, tol: float = 1e-6) -> bool:
    """True if ``R`` is 3x3, orthogonal, and right-handed (``det == +1``).

    ``det == -1`` means a reflection has crept in, usually a mirrored axis
    convention somewhere upstream, which silently swaps left and right.
    """
    R = _as_float(R)
    if R.shape != (3, 3):
        return False
    if not np.allclose(R.T @ R, np.eye(3), atol=tol):
        return False
    return bool(np.isclose(np.linalg.det(R), 1.0, atol=tol))


def axis_angle_to_R(axis: npt.ArrayLike, angle_rad: float) -> FloatArray:
    """Rodrigues rotation of ``angle_rad`` about ``axis``.

    ``R = I + sin(t) K + (1 - cos(t)) K^2``, where ``K`` is the skew-symmetric
    cross-product matrix of the unit axis.
    """
    axis = _as_float(axis).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm < ROTATION_TOL:
        raise ValueError("axis must be a non-zero vector")
    kx, ky, kz = axis / norm
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + np.sin(angle_rad) * K + (1.0 - np.cos(angle_rad)) * (K @ K)


def R_to_axis_angle(R: npt.ArrayLike) -> tuple[FloatArray, float]:
    """Inverse of :func:`axis_angle_to_R`. Returns ``(unit_axis, angle_rad)``.

    For the identity the axis is arbitrary; ``[0, 0, 1]`` is returned with angle 0.
    """
    R = _as_float(R)
    if not is_rotation(R):
        raise ValueError("R is not a valid rotation matrix")
    # trace(R) = 1 + 2 cos(theta)
    cos_theta = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cos_theta))
    if angle < ROTATION_TOL:
        return np.array([0.0, 0.0, 1.0]), 0.0
    if abs(angle - np.pi) < 1e-6:
        # Near 180 degrees the skew part vanishes, so recover the axis from
        # (R + I)/2, whose columns are all parallel to the axis.
        M = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(M), 0.0, None))
        i = int(np.argmax(axis))
        signs = np.sign([M[i, 0], M[i, 1], M[i, 2]])
        signs[signs == 0] = 1.0
        axis = axis * signs * np.sign(signs[i])
        return axis / float(np.linalg.norm(axis)), angle
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis / (2.0 * np.sin(angle)), angle


def quat_to_R(q: npt.ArrayLike) -> FloatArray:
    """Rotation from a quaternion in ``(w, x, y, z)`` order.

    Note the order. Phone IMUs and several libraries use ``(x, y, z, w)``; getting
    this wrong produces a plausible-looking but wrong rotation, so it is stated here
    and asserted in the tests.
    """
    q = _as_float(q).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < ROTATION_TOL:
        raise ValueError("quaternion must be non-zero")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def R_to_quat(R: npt.ArrayLike) -> FloatArray:
    """Quaternion ``(w, x, y, z)`` from a rotation matrix, with ``w >= 0``.

    ``q`` and ``-q`` are the same rotation; the sign is fixed so round-trips compare
    equal.
    """
    axis, angle = R_to_axis_angle(R)
    half = angle / 2.0
    q = np.concatenate([[np.cos(half)], np.sin(half) * axis])
    if q[0] < 0:
        q = -q
    return q


# --------------------------------------------------------------------------- #
# Homogeneous transforms                                                       #
# --------------------------------------------------------------------------- #
def identity() -> FloatArray:
    """The 4x4 identity transform."""
    return np.eye(4)


def T_from_Rt(R: npt.ArrayLike, t: npt.ArrayLike) -> FloatArray:
    """Assemble ``[[R, t], [0, 1]]`` from a rotation and a translation."""
    R = _as_float(R)
    t = _as_float(t).reshape(3)
    if R.shape != (3, 3):
        raise ValueError(f"R must be 3x3, got {R.shape}")
    if not is_rotation(R):
        raise ValueError("R is not a valid rotation matrix (orthogonal, det +1)")
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def Rt_from_T(T: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Split a 4x4 transform into ``(R, t)``."""
    T = _as_float(T)
    if T.shape != (4, 4):
        raise ValueError(f"T must be 4x4, got {T.shape}")
    return T[:3, :3].copy(), T[:3, 3].copy()


def invert(T_a_b: npt.ArrayLike) -> FloatArray:
    """Return ``T_b_a``, the inverse of ``T_a_b``.

    Uses the closed form ``[[R^T, -R^T t], [0, 1]]`` rather than a general matrix
    inverse; see the module docstring.
    """
    R, t = Rt_from_T(T_a_b)
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def compose(*transforms: npt.ArrayLike) -> FloatArray:
    """Chain transforms left to right: ``compose(T_a_b, T_b_c) -> T_a_c``.

    The caller is responsible for matching inner subscripts; numpy cannot check it.
    """
    if not transforms:
        return identity()
    out = _as_float(transforms[0])
    if out.shape != (4, 4):
        raise ValueError(f"transforms must be 4x4, got {out.shape}")
    for T_next in transforms[1:]:
        T = _as_float(T_next)
        if T.shape != (4, 4):
            raise ValueError(f"transforms must be 4x4, got {T.shape}")
        out = out @ T
    return out


def transform_points(T_a_b: npt.ArrayLike, points_b: npt.ArrayLike) -> FloatArray:
    """Map ``(N, 3)`` points from frame ``b`` into frame ``a``.

    Vectorised as ``P R^T + t``, so no homogeneous column is ever materialised.
    Accepts a single ``(3,)`` point and returns a single point in that case.
    """
    P = _as_float(points_b)
    single = P.ndim == 1
    P = np.atleast_2d(P)
    if P.shape[-1] != 3:
        raise ValueError(f"points must be (N, 3), got {P.shape}")
    R, t = Rt_from_T(_as_float(T_a_b))
    out = P @ R.T + t
    return out[0] if single else out


def rotate_vectors(T_a_b: npt.ArrayLike, vectors_b: npt.ArrayLike) -> FloatArray:
    """Rotate ``(N, 3)`` *directions* from ``b`` into ``a``, ignoring translation.

    Use this for velocities, surface normals and joint axes: anything that is a
    direction rather than a position. Translating a direction is a common and silent
    bug.
    """
    V = _as_float(vectors_b)
    single = V.ndim == 1
    V = np.atleast_2d(V)
    if V.shape[-1] != 3:
        raise ValueError(f"vectors must be (N, 3), got {V.shape}")
    R, _ = Rt_from_T(_as_float(T_a_b))
    out = V @ R.T
    return out[0] if single else out


# --------------------------------------------------------------------------- #
# The veh <-> cam convention                                                   #
# --------------------------------------------------------------------------- #
# A camera rigidly mounted to the vehicle, looking along its forward axis. Reading
# the rows:
#     cam x (right)   = -veh y (left)
#     cam y (down)    = -veh z (up)
#     cam z (forward) = +veh x (forward)
# Row i is the i-th cam basis vector written in veh coordinates, which is exactly
# what makes p_cam = R_CAM_VEH @ p_veh true.
R_CAM_VEH: FloatArray = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ]
)

#: Axis-convention-only transform, zero translation. A real mounted camera also has
#: a lever arm; compose this with that translation rather than editing it.
T_CAM_VEH: FloatArray = T_from_Rt(R_CAM_VEH, np.zeros(3))
T_VEH_CAM: FloatArray = invert(T_CAM_VEH)


def look_at_T_world_cam(
    eye_world: npt.ArrayLike,
    target_world: npt.ArrayLike,
    up_world: npt.ArrayLike = (0.0, 0.0, 1.0),
) -> FloatArray:
    """Camera pose ``T_world_cam`` for a camera at ``eye`` looking at ``target``.

    Builds an OpenCV camera frame (x right, y down, z forward). Used to generate the
    synthetic fixture trajectory with analytically known poses.
    """
    eye = _as_float(eye_world).reshape(3)
    target = _as_float(target_world).reshape(3)
    up = _as_float(up_world).reshape(3)

    z_c = target - eye
    n = float(np.linalg.norm(z_c))
    if n < ROTATION_TOL:
        raise ValueError("eye and target coincide; view direction is undefined")
    z_c = z_c / n

    x_c = np.cross(z_c, up)
    nx = float(np.linalg.norm(x_c))
    if nx < 1e-9:
        raise ValueError("view direction is parallel to `up`; choose a different up vector")
    x_c = x_c / nx
    y_c = np.cross(z_c, x_c)  # completes a right-handed frame, pointing down

    # Columns are the cam basis vectors in world coordinates -> R_world_cam.
    R_world_cam = np.column_stack([x_c, y_c, z_c])
    return T_from_Rt(R_world_cam, eye)


# --------------------------------------------------------------------------- #
# Pinhole camera                                                               #
# --------------------------------------------------------------------------- #
def project(
    K: npt.ArrayLike,
    points_cam: npt.ArrayLike,
    image_size: tuple[int, int] | None = None,
) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """Project camera-frame points to pixels. No distortion (undistort first).

    Returns ``(uv, valid)``, where ``uv`` is ``(N, 2)`` and ``valid`` marks points in
    front of the camera and, when ``image_size`` is given, inside the image.

    Points at or behind the pinhole have no projection. They come back as NaN and
    marked invalid, rather than silently producing a finite pixel the way a bare
    divide would for ``z < 0``. That bare divide places objects *behind* the camera
    onto the image, which is the classic way an occlusion test goes wrong (relevant
    to the multi-view label fusion in P2-T3).
    """
    K = _as_float(K)
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K.shape}")
    P = _as_float(points_cam)
    single = P.ndim == 1
    P = np.atleast_2d(P)
    if P.shape[-1] != 3:
        raise ValueError(f"points must be (N, 3), got {P.shape}")

    z = P[:, 2]
    valid = z > MIN_PROJECTION_DEPTH_M
    uv = np.full((P.shape[0], 2), np.nan)
    if np.any(valid):
        xy = P[valid, :2] / z[valid, None]
        uv[valid, 0] = K[0, 0] * xy[:, 0] + K[0, 1] * xy[:, 1] + K[0, 2]
        uv[valid, 1] = K[1, 1] * xy[:, 1] + K[1, 2]

    if image_size is not None:
        w, h = image_size
        # NaN compares False, so invalid points stay invalid without special-casing.
        inside = (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
        valid = valid & inside

    if single:
        return uv[0], bool(valid[0])
    return uv, valid


def unproject(K: npt.ArrayLike, uv: npt.ArrayLike, depth_m: npt.ArrayLike) -> FloatArray:
    """Back-project pixels at a known depth into camera-frame points.

    ``depth_m`` is depth along the optical axis (the z coordinate), *not* distance
    from the pinhole. Confusing the two is a systematic error that grows toward the
    image edges.
    """
    K = _as_float(K)
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K.shape}")
    uv_a = _as_float(uv)
    single = uv_a.ndim == 1
    uv_a = np.atleast_2d(uv_a)
    if uv_a.shape[-1] != 2:
        raise ValueError(f"uv must be (N, 2), got {uv_a.shape}")
    z = np.atleast_1d(_as_float(depth_m)).reshape(-1)
    if z.size == 1:
        z = np.full(uv_a.shape[0], z[0])
    if z.shape[0] != uv_a.shape[0]:
        raise ValueError("depth_m must be scalar or one value per pixel")

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    skew = K[0, 1]
    y_n = (uv_a[:, 1] - cy) / fy
    x_n = (uv_a[:, 0] - cx - skew * y_n) / fx
    out = np.column_stack([x_n * z, y_n * z, z])
    return out[0] if single else out
