"""Detector output filtering (the RT-DETR adapter itself is verified on CPU; ADR 0011)."""

from __future__ import annotations

from vscs.common.config import load_config
from vscs.perception.detect import filter_detections

CFG = load_config("perception")["detect"]


def test_keeps_configured_classes_above_threshold_and_clips():
    out = filter_detections(
        boxes=[[10, 10, 50, 80], [-5, 0, 30, 900], [0, 0, 5, 5], [100, 100, 100, 150]],
        scores=[0.9, 0.6, 0.95, 0.9],
        labels=["person", "car", "dining table", "car"],
        image_hw=(720, 1280),
        classes=CFG["classes"],
        score_threshold=CFG["score_threshold"],
    )
    assert [(b.cls, b.xyxy) for b in out] == [
        ("person", (10.0, 10.0, 50.0, 80.0)),
        ("car", (0.0, 0.0, 30.0, 719.0)),  # clipped to the frame
    ]  # 'dining table' is not a configured class; the zero-width box is dropped


def test_threshold_is_respected():
    out = filter_detections(
        [[0, 0, 10, 10]],
        [CFG["score_threshold"] - 0.01],
        ["person"],
        (100, 100),
        CFG["classes"],
        CFG["score_threshold"],
    )
    assert out == []


def test_detector_licence_rule_holds_in_config():
    """§10: the detector must not be Ultralytics (AGPL)."""
    assert "ultralytics" not in CFG["engine"].lower() and "yolo" not in CFG["model_id"].lower()
