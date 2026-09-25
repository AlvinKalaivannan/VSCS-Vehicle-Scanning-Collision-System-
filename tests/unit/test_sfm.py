"""COLMAP wrapper tests (P1-T5).

COLMAP is not installed (ADR 0003), so the runner is replaced by a fake that creates what
COLMAP would: model folders on ``mapper``, and text models on ``model_converter``. The text
models are the synthetic fixture reconstructions, so the metrics have known answers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fixtures.colmap_synth import build_fixture_model
from vscs.common.config import load_config
from vscs.recon import sfm
from vscs.recon.colmap_io import write_model

SFM = load_config("recon")["sfm"]
CALIBRATED = {
    "calibrated": True,
    "K": [[900.0, 0.0, 639.5], [0.0, 900.0, 359.5], [0.0, 0.0, 1.0]],
    "dist": [0.1, -0.05, 0.001, 0.002, 0.0],
}


def _fake_runner(models):
    calls: list[list[str]] = []

    def runner(cmd, check=True):
        calls.append(list(cmd))
        if cmd[1] == "mapper":
            out = Path(cmd[cmd.index("--output_path") + 1])
            for i in range(len(models)):
                (out / str(i)).mkdir(parents=True, exist_ok=True)
        elif cmd[1] == "model_converter":
            src = Path(cmd[cmd.index("--input_path") + 1])
            dst = Path(cmd[cmd.index("--output_path") + 1])
            write_model(models[int(src.name)], dst)

    return runner, calls


def _images(tmp_path, n=20):
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(n):
        (d / f"{i:06d}.png").write_bytes(b"x")
    return d


# --------------------------------------------------------------------------- #
# Command construction                                                         #
# --------------------------------------------------------------------------- #
def test_commands_are_argument_lists_never_shell_strings(tmp_path):
    cmds = sfm.build_commands("colmap", tmp_path / "img", tmp_path / "ws", SFM, "1,2,3,4,0,0,0,0")
    assert [c[1] for c in cmds] == ["feature_extractor", "exhaustive_matcher", "mapper"]
    assert all(isinstance(c, list) and all(isinstance(a, str) for a in c) for c in cmds)


def test_calibrated_intrinsics_are_passed_and_not_refined(tmp_path):
    params = sfm.camera_params_from_intrinsics(CALIBRATED)
    extract, _, mapper = sfm.build_commands("colmap", tmp_path, tmp_path, SFM, params)
    assert extract[extract.index("--ImageReader.camera_params") + 1] == params
    assert extract[extract.index("--ImageReader.camera_model") + 1] == "OPENCV"
    for flag in (
        "--Mapper.ba_refine_focal_length",
        "--Mapper.ba_refine_principal_point",
        "--Mapper.ba_refine_extra_params",
    ):
        assert mapper[mapper.index(flag) + 1] == "0"


def test_refinement_flags_absent_when_refinement_enabled(tmp_path):
    cfg = {**SFM, "refine_intrinsics": True}
    _, _, mapper = sfm.build_commands("colmap", tmp_path, tmp_path, cfg, None)
    assert "--Mapper.ba_refine_focal_length" not in mapper


def test_no_camera_params_when_uncalibrated(tmp_path):
    extract, _, _ = sfm.build_commands("colmap", tmp_path, tmp_path, SFM, None)
    assert "--ImageReader.camera_params" not in extract


def test_sequential_matcher_and_bad_matcher(tmp_path):
    _, match, _ = sfm.build_commands(
        "c", tmp_path, tmp_path, {**SFM, "matcher": "sequential"}, None
    )
    assert match[1] == "sequential_matcher"
    with pytest.raises(ValueError, match="unsupported matcher"):
        sfm.build_commands("c", tmp_path, tmp_path, {**SFM, "matcher": "vocab_tree"}, None)


def test_camera_params_string():
    s = sfm.camera_params_from_intrinsics(CALIBRATED)
    vals = [float(v) for v in s.split(",")]
    assert vals == pytest.approx([900.0, 900.0, 639.5, 359.5, 0.1, -0.05, 0.001, 0.002])
    assert sfm.camera_params_from_intrinsics({"calibrated": False}) is None


def test_large_k3_is_reported_not_silently_dropped(caplog):
    intr = {**CALIBRATED, "dist": [0.1, -0.05, 0.0, 0.0, 0.2]}
    with caplog.at_level("WARNING"):
        sfm.camera_params_from_intrinsics(intr)
    assert "k3" in caplog.text


def test_missing_colmap_explains_how_to_install(monkeypatch):
    monkeypatch.setattr(sfm.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="ADR 0003"):
        sfm.colmap_binary()
    with pytest.raises(FileNotFoundError, match="not found"):
        sfm.colmap_binary("C:/nowhere/colmap.exe")


# --------------------------------------------------------------------------- #
# Running and evaluating                                                       #
# --------------------------------------------------------------------------- #
def test_full_run_on_a_perfect_reconstruction(tmp_path):
    runner, calls = _fake_runner([build_fixture_model()])
    result, model = sfm.run_sfm(_images(tmp_path), tmp_path / "ws", SFM, CALIBRATED, runner=runner)
    assert [c[1] for c in calls] == [
        "feature_extractor",
        "exhaustive_matcher",
        "mapper",
        "model_converter",
    ]
    assert result.n_models == 1 and result.n_registered == 20 and result.n_frames == 20
    assert result.registered_fraction == pytest.approx(1.0)
    assert result.mean_reprojection_error_px == pytest.approx(0.4)
    assert result.passed
    assert model.n_registered == 20


def test_a_split_reconstruction_cannot_hide_behind_its_largest_part(tmp_path):
    """COLMAP split 25 frames into 20 + 5. The largest model is healthy on its own, but
    registration is counted against all 25 frames: 80%, below the 90% gate."""
    runner, _ = _fake_runner([build_fixture_model(n_images=5), build_fixture_model()])
    result, _ = sfm.run_sfm(_images(tmp_path, 25), tmp_path / "ws", SFM, CALIBRATED, runner=runner)
    assert result.n_models == 2
    assert result.n_registered == 20
    assert result.registered_fraction == pytest.approx(0.8)
    assert not result.registration_ok and not result.passed
    assert "split" in result.summary()


def test_high_reprojection_error_fails_the_gate(tmp_path):
    runner, _ = _fake_runner([build_fixture_model(point_error_px=2.0)])
    result, _ = sfm.run_sfm(_images(tmp_path), tmp_path / "ws", SFM, CALIBRATED, runner=runner)
    assert result.registration_ok and not result.reprojection_ok and not result.passed


def test_no_model_at_all_is_an_error(tmp_path):
    runner, _ = _fake_runner([])
    with pytest.raises(RuntimeError, match="no model"):
        sfm.run_sfm(_images(tmp_path), tmp_path / "ws", SFM, CALIBRATED, runner=runner)


def test_uncalibrated_run_warns(tmp_path, caplog):
    runner, calls = _fake_runner([build_fixture_model()])
    with caplog.at_level("WARNING"):
        sfm.run_sfm(_images(tmp_path), tmp_path / "ws", SFM, {"calibrated": False}, runner=runner)
    assert "uncalibrated" in caplog.text
    assert "--ImageReader.camera_params" not in calls[0]


def test_explicit_frame_count_overrides_the_folder(tmp_path):
    runner, _ = _fake_runner([build_fixture_model()])
    result, _ = sfm.run_sfm(
        _images(tmp_path), tmp_path / "ws", SFM, CALIBRATED, runner=runner, n_frames=40
    )
    assert result.registered_fraction == pytest.approx(0.5)


def test_result_dict_is_serialisable(tmp_path):
    import json

    runner, _ = _fake_runner([build_fixture_model()])
    result, _ = sfm.run_sfm(_images(tmp_path), tmp_path / "ws", SFM, CALIBRATED, runner=runner)
    d = result.to_dict()
    json.dumps(d)
    assert d["passed"] is True


def test_count_images_ignores_other_files(tmp_path):
    d = _images(tmp_path, 3)
    (d / "notes.txt").write_text("x", encoding="utf-8")
    assert sfm.count_images(d) == 3


def test_gate_values_come_from_config():
    assert SFM["min_registered_frame_fraction"] == 0.90
    assert SFM["max_mean_reprojection_error_px"] == 1.5
    assert np.isfinite(SFM["max_mean_reprojection_error_px"])
