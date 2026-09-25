"""COLMAP structure-from-motion wrapper and the P1-T5 acceptance metrics.

Acceptance criterion (CLAUDE.md §6): at least 90% of extracted frames registered, and mean
reprojection error below 1.5 px. Both gates live in ``configs/recon.yaml``.

This module builds the COLMAP command lines from configuration, runs them, converts the
result to text, and measures it. It does no reconstruction itself.

Design notes
------------
* **Calibrated intrinsics are reused, not re-solved** (``use_prior_intrinsics``,
  ``refine_intrinsics: false``). The P0-T7 calibration was taken with locked focus and
  checked to < 0.5 px; letting COLMAP re-estimate focal length from a van with large
  featureless panels would trade a measured value for a guessed one (R-04).
* **More than one model is a warning sign, not a detail.** COLMAP splits the reconstruction
  when it cannot connect all the frames, and the largest piece may still look healthy. On
  this van the likely cause is reflective or textureless panels (R-01). The wrapper reports
  the model count and evaluates the largest model, but counts registration against *all*
  extracted frames, so a split cannot hide inside a passing number.
* The command runner is injectable, so everything except COLMAP itself is tested now,
  before COLMAP is installed.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vscs.common.log import get_logger
from vscs.recon.colmap_io import Model, read_model

logger = get_logger("recon.sfm")

Runner = Callable[..., Any]


@dataclass
class SfmResult:
    """What P1-T5 is judged on, plus the context needed to interpret it."""

    model_dir: str
    n_models: int
    n_frames: int
    n_registered: int
    n_points: int
    registered_fraction: float
    mean_reprojection_error_px: float
    min_registered_fraction: float
    max_mean_reprojection_error_px: float

    @property
    def registration_ok(self) -> bool:
        return self.registered_fraction >= self.min_registered_fraction

    @property
    def reprojection_ok(self) -> bool:
        return self.mean_reprojection_error_px < self.max_mean_reprojection_error_px

    @property
    def passed(self) -> bool:
        return self.registration_ok and self.reprojection_ok

    def summary(self) -> str:
        lines = [
            f"models: {self.n_models}"
            + (
                "  <-- reconstruction split (R-01: reflective/textureless panels?)"
                if self.n_models > 1
                else ""
            ),
            f"registered: {self.n_registered}/{self.n_frames} = {self.registered_fraction:.1%} "
            f"(gate >= {self.min_registered_fraction:.0%}) "
            f"{'PASS' if self.registration_ok else 'FAIL'}",
            f"reprojection: {self.mean_reprojection_error_px:.3f} px "
            f"(gate < {self.max_mean_reprojection_error_px} px) "
            f"{'PASS' if self.reprojection_ok else 'FAIL'}",
            f"3D points: {self.n_points}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(
            registration_ok=self.registration_ok,
            reprojection_ok=self.reprojection_ok,
            passed=self.passed,
        )
        return d


def colmap_binary(explicit: str | None = None) -> str:
    """Locate the COLMAP executable, or explain how to get one (ADR 0003)."""
    if explicit:
        if Path(explicit).is_file() or shutil.which(explicit):
            return explicit
        raise FileNotFoundError(f"COLMAP not found at {explicit}")
    for name in ("colmap", "COLMAP.bat", "colmap.exe"):
        found = shutil.which(name)
        if found:
            return found
    raise FileNotFoundError(
        "COLMAP is not installed or not on PATH. Per ADR 0003, install the prebuilt "
        "Windows no-CUDA release and add its folder to PATH, or pass --colmap <path>."
    )


def camera_params_from_intrinsics(intrinsics: dict[str, Any]) -> str | None:
    """COLMAP ``OPENCV`` parameter string from ``configs/capture.yaml`` intrinsics.

    Returns ``None`` when uncalibrated, in which case COLMAP estimates intrinsics itself
    and the caller should say so loudly. OpenCV's calibration may return a fifth
    coefficient (k3); COLMAP's OPENCV model has no slot for it, so a non-negligible k3 is
    reported rather than silently dropped.
    """
    if not intrinsics.get("calibrated"):
        return None
    K = np.asarray(intrinsics["K"], dtype=np.float64)
    dist = np.asarray(intrinsics["dist"], dtype=np.float64).reshape(-1)
    k = np.zeros(4)
    k[: min(4, dist.size)] = dist[:4]
    if dist.size > 4 and abs(dist[4]) > 1e-3:
        logger.warning(
            "calibration has k3 = %.4g, which COLMAP's OPENCV model cannot represent; "
            "it is dropped. If reprojection error is high, consider the FULL_OPENCV model.",
            dist[4],
        )
    vals = [K[0, 0], K[1, 1], K[0, 2], K[1, 2], *k]
    return ",".join(f"{v:.10g}" for v in vals)


def build_commands(
    colmap: str,
    image_dir: Path,
    workspace: Path,
    sfm_cfg: dict[str, Any],
    camera_params: str | None,
) -> list[list[str]]:
    """Feature extraction, matching and mapping, as argument lists (never a shell string)."""
    db = str(Path(workspace) / "database.db")
    sparse = str(Path(workspace) / "sparse")
    extract = [
        colmap,
        "feature_extractor",
        "--database_path",
        db,
        "--image_path",
        str(image_dir),
        "--ImageReader.camera_model",
        str(sfm_cfg["camera_model"]),
        "--ImageReader.single_camera",
        "1",
    ]
    if sfm_cfg.get("use_prior_intrinsics") and camera_params:
        extract += ["--ImageReader.camera_params", camera_params]

    matcher = str(sfm_cfg.get("matcher", "exhaustive"))
    if matcher not in ("exhaustive", "sequential"):
        raise ValueError(f"unsupported matcher {matcher!r}; use exhaustive or sequential")
    match = [colmap, f"{matcher}_matcher", "--database_path", db]

    mapper = [
        colmap,
        "mapper",
        "--database_path",
        db,
        "--image_path",
        str(image_dir),
        "--output_path",
        sparse,
    ]
    if not sfm_cfg.get("refine_intrinsics", True):
        mapper += [
            "--Mapper.ba_refine_focal_length",
            "0",
            "--Mapper.ba_refine_principal_point",
            "0",
            "--Mapper.ba_refine_extra_params",
            "0",
        ]
    return [extract, match, mapper]


def converter_command(colmap: str, model_dir: Path, out_dir: Path) -> list[str]:
    return [
        colmap,
        "model_converter",
        "--input_path",
        str(model_dir),
        "--output_path",
        str(out_dir),
        "--output_type",
        "TXT",
    ]


def evaluate_model(
    model: Model, n_frames: int, sfm_cfg: dict[str, Any], *, model_dir: str, n_models: int
) -> SfmResult:
    """Measure a model against the P1-T5 gates."""
    return SfmResult(
        model_dir=model_dir,
        n_models=n_models,
        n_frames=int(n_frames),
        n_registered=model.n_registered,
        n_points=len(model.points),
        registered_fraction=model.registered_fraction(n_frames),
        mean_reprojection_error_px=model.mean_reprojection_error_px(),
        min_registered_fraction=float(sfm_cfg["min_registered_frame_fraction"]),
        max_mean_reprojection_error_px=float(sfm_cfg["max_mean_reprojection_error_px"]),
    )


def count_images(image_dir: Path) -> int:
    exts = {".png", ".jpg", ".jpeg"}
    return sum(1 for p in Path(image_dir).iterdir() if p.suffix.lower() in exts)


def run_sfm(
    image_dir: Path,
    workspace: Path,
    sfm_cfg: dict[str, Any],
    intrinsics: dict[str, Any],
    *,
    colmap: str | None = None,
    runner: Runner = subprocess.run,
    n_frames: int | None = None,
) -> tuple[SfmResult, Model]:
    """Run COLMAP end to end, convert every model to text, evaluate the largest."""
    image_dir, workspace = Path(image_dir), Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    exe = colmap_binary(colmap) if runner is subprocess.run else (colmap or "colmap")

    params = camera_params_from_intrinsics(intrinsics)
    if params is None:
        logger.warning(
            "intrinsics are uncalibrated (P0-T7 not done): COLMAP will estimate them itself, "
            "which is weaker on a van with large featureless panels (R-01, R-04)."
        )
    (workspace / "sparse").mkdir(exist_ok=True)
    for cmd in build_commands(exe, image_dir, workspace, sfm_cfg, params):
        logger.info("running: %s", " ".join(cmd[:2]))
        runner(cmd, check=True)

    model_dirs = sorted(
        (p for p in (workspace / "sparse").iterdir() if p.is_dir()), key=lambda p: p.name
    )
    if not model_dirs:
        raise RuntimeError(
            "COLMAP produced no model at all - no two frames could be matched. Check "
            "texture, overlap and lighting (R-01) before anything else."
        )

    models: list[tuple[Path, Model]] = []
    for md in model_dirs:
        txt = workspace / "sparse_txt" / md.name
        txt.mkdir(parents=True, exist_ok=True)
        runner(converter_command(exe, md, txt), check=True)
        models.append((txt, read_model(txt)))

    best_dir, best = max(models, key=lambda m: m[1].n_registered)
    if len(models) > 1:
        logger.warning(
            "COLMAP split the reconstruction into %d models; evaluating the largest (%d "
            "images). A split usually means frames could not be linked across a "
            "reflective or textureless region (R-01).",
            len(models),
            best.n_registered,
        )
    total = n_frames if n_frames is not None else count_images(image_dir)
    result = evaluate_model(best, total, sfm_cfg, model_dir=str(best_dir), n_models=len(models))
    logger.info("SfM result:\n%s", result.summary())
    return result, best
