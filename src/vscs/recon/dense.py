"""COLMAP dense reconstruction, run on Colab (P1-T7).

Acceptance (CLAUDE.md §6): dense geometry and a splat, both viewable in Rerun in the ``veh``
frame. This module produces the dense point cloud; viewing and the splat are separate.

COLMAP's dense step has three stages, all driven from the sparse model P1-T5 produced,
plus a conversion:

1. ``image_undistorter`` - resample every registered frame to an ideal pinhole image,
   because patch-match assumes straight epipolar geometry;
2. ``patch_match_stereo`` - estimate a depth and normal map per image by matching patches
   against neighbouring views, optionally filtered for geometric consistency between them;
3. ``stereo_fusion`` - merge the per-image depth maps into one point cloud (``fused.ply``);
4. ``model_converter`` - write the *undistorted* camera model (``<workspace>/sparse``, binary)
   as text into ``<workspace>/sparse_txt``. Label fusion (``scripts/fuse.py``) needs these
   pinhole cameras: they, not the SfM model's, match the depth maps pixel for pixel.

**Stage 2 requires a CUDA build of COLMAP.** The laptop has no NVIDIA GPU (ADR 0003), which
is why this runs on Colab. A COLMAP binary built without CUDA does not fail until it reaches
stage 2, after the undistortion has already run - so :func:`run_dense` checks the build
first, from COLMAP's own version banner, and refuses to start on a build that reports no
CUDA. If the banner is unrecognised it warns and proceeds, rather than blocking on a guess.

The runner is injectable, so everything except COLMAP itself is tested on the laptop.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from vscs.common.log import get_logger

logger = get_logger("recon.dense")

Runner = Callable[..., Any]
CudaStatus = Literal["cuda", "no_cuda", "unknown"]


@dataclass
class DenseResult:
    ply_path: str
    n_points: int
    min_points: int

    @property
    def plausible(self) -> bool:
        """A cloud below the floor almost certainly means a silent failure upstream."""
        return self.n_points >= self.min_points


def parse_cuda_status(banner: str) -> CudaStatus:
    """Read CUDA support from COLMAP's banner.

    COLMAP 3.x says ``... with CUDA)`` / ``... without CUDA)``; COLMAP 4.x says
    ``... with GPU support)`` / ``... without GPU support)`` (seen on 4.2.0). The "without"
    forms are checked first, since each contains its "with" form as a substring.
    """
    text = banner.lower()
    if "without cuda" in text or "without gpu support" in text:
        return "no_cuda"
    if "with cuda" in text or "with gpu support" in text:
        return "cuda"
    return "unknown"


def colmap_cuda_status(colmap: str, runner: Runner = subprocess.run) -> CudaStatus:
    """Ask the COLMAP binary how it was built."""
    try:
        proc = runner([colmap, "help"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return "unknown"
    return parse_cuda_status(f"{getattr(proc, 'stdout', '')}\n{getattr(proc, 'stderr', '')}")


def build_dense_commands(
    colmap: str,
    image_dir: Path,
    sparse_model_dir: Path,
    workspace: Path,
    dense_cfg: dict[str, Any],
) -> list[list[str]]:
    """The three COLMAP dense stages, then the text conversion, as argument lists."""
    ws = str(workspace)
    geometric = bool(dense_cfg.get("geom_consistency", True))
    return [
        [
            colmap,
            "image_undistorter",
            "--image_path",
            str(image_dir),
            "--input_path",
            str(sparse_model_dir),
            "--output_path",
            ws,
            "--output_type",
            "COLMAP",
            "--max_image_size",
            str(int(dense_cfg["max_image_size"])),
        ],
        [
            colmap,
            "patch_match_stereo",
            "--workspace_path",
            ws,
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.geom_consistency",
            "true" if geometric else "false",
        ],
        [
            colmap,
            "stereo_fusion",
            "--workspace_path",
            ws,
            "--workspace_format",
            "COLMAP",
            "--input_type",
            "geometric" if geometric else "photometric",
            "--output_path",
            str(Path(workspace) / "fused.ply"),
        ],
        [
            colmap,
            "model_converter",
            "--input_path",
            str(Path(workspace) / "sparse"),
            "--output_path",
            str(Path(workspace) / "sparse_txt"),
            "--output_type",
            "TXT",
        ],
    ]


_VERTEX_RE = re.compile(rb"^element\s+vertex\s+(\d+)\s*$")


def read_ply_vertex_count(path: Path) -> int:
    """Point count from a PLY header, without loading the (possibly large) body."""
    with Path(path).open("rb") as fh:
        first = fh.readline().strip()
        if first != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        for _ in range(200):
            line = fh.readline()
            if not line:
                break
            line = line.strip()
            m = _VERTEX_RE.match(line)
            if m:
                return int(m.group(1))
            if line == b"end_header":
                break
    raise ValueError(f"{path}: no 'element vertex' line in the PLY header")


def run_dense(
    image_dir: Path,
    sparse_model_dir: Path,
    workspace: Path,
    dense_cfg: dict[str, Any],
    *,
    colmap: str = "colmap",
    runner: Runner = subprocess.run,
) -> DenseResult:
    """Run the three dense stages and report the fused cloud."""
    status = colmap_cuda_status(colmap, runner)
    if status == "no_cuda":
        raise RuntimeError(
            "this COLMAP build reports no CUDA support, and patch_match_stereo requires "
            "CUDA. Install a CUDA-enabled COLMAP on the Colab GPU runtime (ADR 0003), and "
            "record the install route that works there (R-12)."
        )
    if status == "unknown":
        logger.warning("could not read CUDA support from COLMAP's banner; proceeding anyway")

    workspace = Path(workspace)
    (workspace / "sparse_txt").mkdir(parents=True, exist_ok=True)  # model_converter needs it
    for cmd in build_dense_commands(colmap, image_dir, sparse_model_dir, workspace, dense_cfg):
        logger.info("running: %s", " ".join(cmd[:2]))
        runner(cmd, check=True)

    ply = workspace / "fused.ply"
    if not ply.is_file():
        raise RuntimeError(f"stereo_fusion finished but wrote no {ply}")
    result = DenseResult(str(ply), read_ply_vertex_count(ply), int(dense_cfg["min_fused_points"]))
    if not result.plausible:
        logger.warning(
            "fused cloud has only %d points (floor %d): the dense step probably failed "
            "quietly. Check the patch-match logs before using it.",
            result.n_points,
            result.min_points,
        )
    return result
