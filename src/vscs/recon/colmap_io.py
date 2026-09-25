"""Read and write COLMAP's text model format (cameras.txt, images.txt, points3D.txt).

COLMAP is the reconstruction engine (§8.2 primary); this module is the boundary where its
conventions are converted into ours, so it is the one place a convention error could creep
in unnoticed. Two conventions matter:

**Pose direction.** COLMAP stores each image's pose as a *world-to-camera* transform:
``X_cam = R(q) X_world + t``. In §4.1 naming that is ``T_cam_world``. The camera's pose in
the world - where the camera *is* - is its inverse, ``T_world_cam``, with the camera centre
at ``-R^T t``. Mixing these up gives a reconstruction that looks plausible and is wrong.

**Quaternion order.** COLMAP writes ``QW QX QY QZ`` - scalar first - which is the order
``common.frames.quat_to_R`` expects. Stated here because many other tools use scalar last.

COLMAP's camera frame is the OpenCV one (x right, y down, z forward), matching §4.1's
``cam``, so no axis permutation is needed.

The writer exists for two reasons: tests build synthetic models with exactly known poses
from the fixture world, and a model can be exported after scaling (P1-T6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from vscs.common.frames import R_to_quat, Rt_from_T, T_from_Rt, invert, quat_to_R

FloatArray = npt.NDArray[np.float64]

#: Parameter layout of the COLMAP camera models this project accepts.
_MODEL_PARAMS = {
    "SIMPLE_PINHOLE": ("f", "cx", "cy"),
    "PINHOLE": ("fx", "fy", "cx", "cy"),
    "SIMPLE_RADIAL": ("f", "cx", "cy", "k"),
    "RADIAL": ("f", "cx", "cy", "k1", "k2"),
    "OPENCV": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"),
}


@dataclass
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: FloatArray

    def __post_init__(self) -> None:
        if self.model not in _MODEL_PARAMS:
            raise ValueError(
                f"unsupported COLMAP camera model {self.model!r}; expected one of "
                f"{sorted(_MODEL_PARAMS)}"
            )
        self.params = np.asarray(self.params, dtype=np.float64)
        expected = len(_MODEL_PARAMS[self.model])
        if self.params.size != expected:
            raise ValueError(
                f"camera {self.camera_id}: {self.model} takes {expected} params, "
                f"got {self.params.size}"
            )

    def _p(self) -> dict[str, float]:
        return dict(zip(_MODEL_PARAMS[self.model], map(float, self.params), strict=True))

    @property
    def K(self) -> FloatArray:
        p = self._p()
        fx = p.get("fx", p.get("f"))
        fy = p.get("fy", p.get("f"))
        return np.array([[fx, 0.0, p["cx"]], [0.0, fy, p["cy"]], [0.0, 0.0, 1.0]])

    @property
    def dist(self) -> FloatArray:
        """Distortion in OpenCV order ``(k1, k2, p1, p2)``; zeros where the model has none."""
        p = self._p()
        k1 = p.get("k1", p.get("k", 0.0))
        return np.array([k1, p.get("k2", 0.0), p.get("p1", 0.0), p.get("p2", 0.0)])


@dataclass
class Image:
    image_id: int
    qvec: FloatArray  # (w, x, y, z), world -> camera
    tvec: FloatArray  # world -> camera
    camera_id: int
    name: str
    xys: FloatArray = field(default_factory=lambda: np.zeros((0, 2)))
    point3D_ids: npt.NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, np.int64))

    @property
    def T_cam_world(self) -> FloatArray:
        """COLMAP's stored pose: maps world points into this camera."""
        return T_from_Rt(quat_to_R(self.qvec), self.tvec)

    @property
    def T_world_cam(self) -> FloatArray:
        """Where the camera is: maps camera points into the world."""
        return invert(self.T_cam_world)

    @property
    def centre(self) -> FloatArray:
        """Camera centre in the world, ``-R^T t``."""
        return self.T_world_cam[:3, 3].copy()

    @classmethod
    def from_T_world_cam(
        cls, image_id: int, T_world_cam: npt.ArrayLike, camera_id: int, name: str, **kw
    ) -> Image:
        R, t = Rt_from_T(invert(np.asarray(T_world_cam, dtype=np.float64)))
        return cls(image_id, R_to_quat(R), t, camera_id, name, **kw)


@dataclass
class Point3D:
    point3D_id: int
    xyz: FloatArray
    rgb: npt.NDArray[np.int64]
    error: float  # mean reprojection error over the track, px
    image_ids: npt.NDArray[np.int64]
    point2D_idxs: npt.NDArray[np.int64]

    @property
    def track_length(self) -> int:
        return int(self.image_ids.size)


@dataclass
class Model:
    cameras: dict[int, Camera]
    images: dict[int, Image]
    points: dict[int, Point3D]

    @property
    def n_registered(self) -> int:
        return len(self.images)

    def mean_reprojection_error_px(self) -> float:
        """Observation-weighted mean reprojection error, px.

        COLMAP stores one error per 3D point, averaged over that point's track. Weighting
        by track length turns that into the mean over *observations*, which is the figure
        COLMAP's own model analyser reports and what P1-T5's 1.5 px gate refers to.
        """
        if not self.points:
            return float("nan")
        errs = np.array([p.error for p in self.points.values()])
        weights = np.array([max(p.track_length, 1) for p in self.points.values()])
        return float(np.average(errs, weights=weights))

    def registered_fraction(self, n_frames: int) -> float:
        """Fraction of the extracted frames that ended up in this model (P1-T5 gate)."""
        if n_frames <= 0:
            raise ValueError("n_frames must be positive")
        return self.n_registered / n_frames


# --------------------------------------------------------------------------- #
# Reading                                                                      #
# --------------------------------------------------------------------------- #
def _data_lines(path: Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [ln.rstrip("\n") for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]


def read_cameras_text(path: Path) -> dict[int, Camera]:
    out: dict[int, Camera] = {}
    for ln in _data_lines(path):
        f = ln.split()
        cam = Camera(int(f[0]), f[1], int(f[2]), int(f[3]), np.array(f[4:], dtype=np.float64))
        out[cam.camera_id] = cam
    return out


def read_images_text(path: Path) -> dict[int, Image]:
    """Images take two lines each: the pose line, then its 2D observations.

    The observation line may be empty (an image with no triangulated points), so blank
    lines cannot simply be skipped here - comments are, but pairing is by position.
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh if not ln.lstrip().startswith("#")]
    # Drop trailing blank lines only; interior blanks are legitimate empty observations.
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) % 2:
        lines.append("")
    out: dict[int, Image] = {}
    for pose_ln, obs_ln in zip(lines[0::2], lines[1::2], strict=True):
        f = pose_ln.split()
        if not f:
            continue
        obs = obs_ln.split()
        if len(obs) % 3:
            raise ValueError(f"image {f[0]}: observation line length is not a multiple of 3")
        triples = np.array(obs, dtype=np.float64).reshape(-1, 3) if obs else np.zeros((0, 3))
        img = Image(
            image_id=int(f[0]),
            qvec=np.array(f[1:5], dtype=np.float64),
            tvec=np.array(f[5:8], dtype=np.float64),
            camera_id=int(f[8]),
            name=" ".join(f[9:]),
            xys=triples[:, :2],
            point3D_ids=triples[:, 2].astype(np.int64),
        )
        out[img.image_id] = img
    return out


def read_points3D_text(path: Path) -> dict[int, Point3D]:
    out: dict[int, Point3D] = {}
    for ln in _data_lines(path):
        f = ln.split()
        track = np.array(f[8:], dtype=np.int64).reshape(-1, 2)
        pt = Point3D(
            point3D_id=int(f[0]),
            xyz=np.array(f[1:4], dtype=np.float64),
            rgb=np.array(f[4:7], dtype=np.int64),
            error=float(f[7]),
            image_ids=track[:, 0],
            point2D_idxs=track[:, 1],
        )
        out[pt.point3D_id] = pt
    return out


def read_model(model_dir: Path) -> Model:
    """Read a text model directory. Raises with guidance if it is a binary model."""
    d = Path(model_dir)
    if not (d / "cameras.txt").is_file():
        if (d / "cameras.bin").is_file():
            raise FileNotFoundError(
                f"{d} holds a binary COLMAP model. Convert it first with "
                "`colmap model_converter --output_type TXT`."
            )
        raise FileNotFoundError(f"no COLMAP model in {d}")
    return Model(
        cameras=read_cameras_text(d / "cameras.txt"),
        images=read_images_text(d / "images.txt"),
        points=read_points3D_text(d / "points3D.txt"),
    )


# --------------------------------------------------------------------------- #
# Writing                                                                      #
# --------------------------------------------------------------------------- #
def _fmt(values) -> str:
    return " ".join(
        repr(float(v)) if isinstance(v, float | np.floating) else str(v) for v in values
    )


def write_model(model: Model, model_dir: Path) -> Path:
    """Write a text model that COLMAP itself can read back."""
    d = Path(model_dir)
    d.mkdir(parents=True, exist_ok=True)
    with (d / "cameras.txt").open("w", encoding="utf-8") as fh:
        fh.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for c in model.cameras.values():
            fh.write(f"{c.camera_id} {c.model} {c.width} {c.height} {_fmt(list(c.params))}\n")
    with (d / "images.txt").open("w", encoding="utf-8") as fh:
        fh.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        fh.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for im in model.images.values():
            fh.write(f"{im.image_id} {_fmt([*im.qvec, *im.tvec])} {im.camera_id} {im.name}\n")
            obs = [
                f"{float(x)!r} {float(y)!r} {int(pid)}"
                for (x, y), pid in zip(im.xys, im.point3D_ids, strict=True)
            ]
            fh.write(" ".join(obs) + "\n")
    with (d / "points3D.txt").open("w", encoding="utf-8") as fh:
        fh.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for p in model.points.values():
            track = " ".join(
                f"{int(i)} {int(j)}" for i, j in zip(p.image_ids, p.point2D_idxs, strict=True)
            )
            fh.write(
                f"{p.point3D_id} {_fmt(list(p.xyz))} {' '.join(str(int(v)) for v in p.rgb)} "
                f"{float(p.error)!r} {track}\n"
            )
    return d
