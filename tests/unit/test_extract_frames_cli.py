"""Tests for scripts/extract_frames.py - the step between ingest and SfM."""

from __future__ import annotations

import importlib.util

import cv2
import numpy as np
import pytest

from vscs.common.config import load_config, repo_root
from vscs.common.io import read_json, read_jsonl
from vscs.common.types import NS_PER_S, validate_timestamp_stream


@pytest.fixture(scope="module")
def cli():
    path = repo_root() / "scripts" / "extract_frames.py"
    spec = importlib.util.spec_from_file_location("extract_frames_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("v") / "scan.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 120))
    assert w.isOpened()
    for i in range(45):
        f = np.zeros((120, 160, 3), np.uint8)
        f[:, (i * 3) % 150 : (i * 3) % 150 + 10] = 255
        w.write(f)
    w.release()
    return path


def _only_run_dir(root):
    runs = list((root / "capture").iterdir())
    assert len(runs) == 1
    return runs[0]


def test_extracts_into_a_new_capture_run_folder(cli, clip, tmp_path):
    assert (
        cli.main(
            [
                "--video",
                str(clip),
                "--stride",
                "3",
                "--backend",
                "opencv",
                "--out-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    run = _only_run_dir(tmp_path)
    assert (run / "provenance.json").is_file()
    assert (run / "resolved_config.yaml").is_file()
    assert (run / "run.log").is_file()
    meta = read_json(run / "frames_meta.json")
    assert meta["n_frames_source"] == 45 and meta["n_frames_written"] == 15
    assert len(list((run / "frames").glob("*.png"))) == 15


def test_timestamps_are_real_and_widen_with_stride(cli, clip, tmp_path):
    cli.main(
        ["--video", str(clip), "--stride", "5", "--backend", "opencv", "--out-root", str(tmp_path)]
    )
    rows = list(read_jsonl(_only_run_dir(tmp_path) / "frames.jsonl"))
    t = [r["t_ns"] for r in rows]
    validate_timestamp_stream(t)
    assert np.diff(t).min() > 4 * NS_PER_S / 30  # five source frames apart, not one
    assert [r["frame_index"] for r in rows][:3] == [0, 5, 10]


def test_default_stride_comes_from_config(cli, clip, tmp_path):
    cli.main(["--video", str(clip), "--backend", "opencv", "--out-root", str(tmp_path)])
    stride = load_config("capture")["ingest"]["extract_stride"]
    meta = read_json(_only_run_dir(tmp_path) / "frames_meta.json")
    assert meta["stride"] == stride
    assert meta["n_frames_written"] == -(-45 // stride)


def test_output_feeds_recon(cli, clip, tmp_path):
    """recon.py --frames-run reads frames/ and frames_meta.json's frame count."""
    cli.main(
        ["--video", str(clip), "--stride", "9", "--backend", "opencv", "--out-root", str(tmp_path)]
    )
    run = _only_run_dir(tmp_path)
    from vscs.recon.sfm import count_images

    assert count_images(run / "frames") == read_json(run / "frames_meta.json")["n_frames_written"]


def test_missing_video_is_reported(cli, tmp_path):
    assert cli.main(["--video", str(tmp_path / "nope.mp4"), "--out-root", str(tmp_path)]) == 2
