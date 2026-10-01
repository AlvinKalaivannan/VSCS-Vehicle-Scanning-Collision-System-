"""Inputs for 3D label fusion, read from a COLMAP dense workspace (P2-T3 plumbing).

The fusion itself is the developer's ``seg/fusion3d.py``. This module only gathers what it
needs, per frame: intrinsics, pose, the 2D label mask and the depth map, all pixel-aligned
and all in the **metric vehicle frame** ``veh``.

Pixel alignment
---------------
COLMAP's dense depth maps are computed on the *undistorted* images that ``image_undistorter``
writes into ``<workspace>/images`` (resampled to an ideal pinhole camera, and downscaled to
``recon.dense.max_image_size``). The label masks must come from those same images, so
``scripts/seg.py`` is run on ``<workspace>/images``, not on the original frames.
:func:`load_view` refuses a mask whose size differs from its depth map, which catches the
common mistake (original frames are larger) but not every one.

From SfM units to ``veh`` (MAT188 notation)
-------------------------------------------
SfM is correct only up to scale. P1-T6 supplies a scale ``s`` (metres per SfM unit) and
``T_veh_world``, the rigid transform from the *scaled* reconstruction frame into ``veh``.
A point therefore maps as ``p_veh = T_veh_world @ (s * p_sfm)``.

A camera stores ``x_cam = R p_sfm + t`` (SfM units). Multiplying by ``s`` gives the
camera-frame point in metres, ``s x_cam = R (s p_sfm) + s t``: the rotation is unchanged
and the translation scales. Composing with ``T_world_veh = T_veh_world^-1`` then gives
``T_cam_veh = [R | s t] @ T_world_veh``. Depth is a camera-frame z, so it scales by ``s``
as well.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from vscs.common.frames import T_from_Rt, invert, transform_points
from vscs.common.io import read_json
from vscs.recon.colmap_io import Camera, Image, Model
from vscs.seg.masks2d import read_label_png

FloatArray = npt.NDArray[np.float64]


# --------------------------------------------------------------------------- #
# COLMAP dense arrays                                                          #
# --------------------------------------------------------------------------- #
def read_colmap_array(path: Path) -> npt.NDArray[np.float32]:
    """Read a COLMAP dense ``.bin`` array (depth or normal map).

    Format: an ASCII header ``width&height&channels&``, then float32 values in column-major
    (Fortran) order. Returns ``(H, W)`` for one channel, ``(H, W, C)`` otherwise.
    """
    data = Path(path).read_bytes()
    pos = 0
    fields = []
    for _ in range(3):
        end = data.index(b"&", pos)
        fields.append(int(data[pos:end]))
        pos = end + 1
    width, height, channels = fields
    arr = np.frombuffer(data, dtype="<f4", offset=pos)
    if arr.size != width * height * channels:
        raise ValueError(
            f"{path}: header says {width}x{height}x{channels} but holds {arr.size} values"
        )
    arr = arr.reshape((width, height, channels), order="F").transpose(1, 0, 2)
    return np.ascontiguousarray(arr.squeeze(axis=2) if channels == 1 else arr)


def write_colmap_array(path: Path, array: npt.ArrayLike) -> Path:
    """Write the format :func:`read_colmap_array` reads (test fixtures)."""
    a = np.asarray(array, dtype="<f4")
    if a.ndim == 2:
        a = a[:, :, None]
    height, width, channels = a.shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(f"{width}&{height}&{channels}&".encode("ascii"))
        fh.write(a.transpose(1, 0, 2).reshape(-1, order="F").tobytes())
    return path


def depth_map_path(workspace: Path, image_name: str, kind: str) -> Path:
    """``kind`` is ``geometric`` or ``photometric`` (``recon.dense.geom_consistency``)."""
    return Path(workspace) / "stereo" / "depth_maps" / f"{image_name}.{kind}.bin"


# --------------------------------------------------------------------------- #
# SfM -> veh                                                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SfmToVeh:
    """P1-T6's result: metres per SfM unit, and the scaled reconstruction frame -> ``veh``."""

    scale: float
    T_veh_world: FloatArray

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError(f"scale must be a positive number, got {self.scale}")
        T = np.asarray(self.T_veh_world, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"T_veh_world must be 4x4, got {T.shape}")
        object.__setattr__(self, "T_veh_world", T)

    def points(self, points_sfm: npt.ArrayLike) -> FloatArray:
        return transform_points(self.T_veh_world, self.scale * np.asarray(points_sfm, float))

    def T_cam_veh(self, T_cam_sfm: npt.ArrayLike) -> FloatArray:
        T = np.asarray(T_cam_sfm, dtype=np.float64)
        return T_from_Rt(T[:3, :3], self.scale * T[:3, 3]) @ invert(self.T_veh_world)

    def depth(self, depth_sfm: npt.ArrayLike) -> FloatArray:
        return self.scale * np.asarray(depth_sfm, dtype=np.float64)


def read_sfm_to_veh(path: Path) -> SfmToVeh:
    """``{"scale": s, "T_veh_world": 4x4}`` - the P1-T6 output ``fuse.py`` consumes."""
    raw = read_json(path)
    missing = {"scale", "T_veh_world"} - set(raw)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    return SfmToVeh(float(raw["scale"]), np.asarray(raw["T_veh_world"], dtype=np.float64))


# --------------------------------------------------------------------------- #
# Views                                                                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ViewData:
    """One frame, ready for ``fusion3d.PosedView`` (same fields, same conventions)."""

    name: str
    K: FloatArray
    T_cam_world: FloatArray  # here "world" is veh: the fused points are in veh
    labels: npt.NDArray[np.int64]
    depth: FloatArray  # metres; inf where COLMAP has no depth (it stores 0 there)


def load_view(
    image: Image,
    camera: Camera,
    labels_png: Path,
    depth_bin: Path,
    sfm_to_veh: SfmToVeh,
    n_classes: int | None = None,
) -> ViewData:
    if camera.model not in ("PINHOLE", "SIMPLE_PINHOLE"):
        raise ValueError(
            f"{image.name}: camera model {camera.model} is not a pinhole - fusion needs the "
            "undistorted model from the dense workspace, not the SfM model"
        )
    labels = read_label_png(labels_png)
    raw = read_colmap_array(depth_bin)
    size = (camera.height, camera.width)
    if labels.shape != size or raw.shape != size:
        raise ValueError(
            f"{image.name}: mask {labels.shape}, depth {raw.shape}, camera {size} (H, W) "
            "differ. Run scripts/seg.py on the dense workspace's undistorted images."
        )
    if n_classes is not None and labels.max(initial=-1) >= n_classes:
        raise ValueError(
            f"{image.name}: mask holds label {labels.max()} but the seg run names only "
            f"{n_classes} components"
        )
    depth = sfm_to_veh.depth(raw)
    depth[~(raw > 0)] = np.inf
    return ViewData(image.name, camera.K, sfm_to_veh.T_cam_veh(image.T_cam_world), labels, depth)


def iter_views(
    model: Model,
    workspace: Path,
    labels_dir: Path,
    sfm_to_veh: SfmToVeh,
    depth_kind: str,
    missing: list[str] | None = None,
    n_classes: int | None = None,
) -> Iterator[ViewData]:
    """Every registered image that has both a mask and a depth map, in image-id order.

    Images lacking either are skipped and their names appended to ``missing``, so the
    caller can report them rather than silently fusing fewer views.
    """
    for image_id in sorted(model.images):
        image = model.images[image_id]
        png = Path(labels_dir) / f"{Path(image.name).stem}.png"
        dbin = depth_map_path(workspace, image.name, depth_kind)
        if not (png.is_file() and dbin.is_file()):
            if missing is not None:
                missing.append(image.name)
            continue
        yield load_view(image, model.cameras[image.camera_id], png, dbin, sfm_to_veh, n_classes)
