"""Object detection for the per-drive pipeline: images -> ``Box2D`` detections (P4-T4).

The §8.2 primary is an Apache/MIT-licensed detector. RT-DETR (Zhao et al., "DETRs Beat
YOLOs on Real-time Object Detection", CVPR 2024) qualifies: code and published weights are
Apache-2.0 (checked 2026-10-01, ADR 0011). Ultralytics YOLO (AGPL) does not (§10).

Split as elsewhere: ``filter_detections`` (pure logic: keep the configured classes above
the score threshold, clip to the image, map to ``Box2D``) is tested on the laptop; the
model adapter ``RTDetrDetector`` imports torch/transformers lazily and is verified on
CPU in an isolated environment, then runs on Colab for speed.

COCO class names come from the model's own ``id2label``; ``perception.yaml pipeline
class_to_kind`` maps them to §4.2 obstacle kinds.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy.typing as npt

from vscs.perception.pipeline import Box2D


def filter_detections(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    labels: Sequence[str],
    image_hw: tuple[int, int],
    classes: Iterable[str],
    score_threshold: float,
) -> list[Box2D]:
    """Keep wanted classes above the threshold; clip boxes to the image; drop empty ones."""
    keep = set(classes)
    h, w = image_hw
    out = []
    for b, s, lab in zip(boxes, scores, labels, strict=True):
        if lab not in keep or float(s) < score_threshold:
            continue
        x0, y0, x1, y1 = (float(v) for v in b)
        x0, x1 = min(max(x0, 0.0), w - 1.0), min(max(x1, 0.0), w - 1.0)
        y0, y1 = min(max(y0, 0.0), h - 1.0), min(max(y1, 0.0), h - 1.0)
        if x1 > x0 and y1 > y0:
            out.append(Box2D((x0, y0, x1, y1), str(lab), float(s)))
    return out


class RTDetrDetector:
    """RT-DETR through Hugging Face ``transformers`` (Apache-2.0 code and weights)."""

    def __init__(self, detect_cfg: dict[str, Any], device: str = "cuda") -> None:
        import torch
        from transformers import AutoImageProcessor, RTDetrForObjectDetection

        self._torch = torch
        self.processor = AutoImageProcessor.from_pretrained(detect_cfg["model_id"])
        self.model = (
            RTDetrForObjectDetection.from_pretrained(detect_cfg["model_id"]).to(device).eval()
        )
        self.device = device
        self.classes = list(detect_cfg["classes"])
        self.threshold = float(detect_cfg["score_threshold"])

    def detect(self, image: npt.NDArray) -> list[Box2D]:
        """``image``: (H, W, 3) RGB uint8."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        h, w = image.shape[:2]
        res = self.processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=[(h, w)]
        )[0]
        id2label = self.model.config.id2label
        labels = [id2label[int(i)] for i in res["labels"].tolist()]
        return filter_detections(
            res["boxes"].tolist(),
            res["scores"].tolist(),
            labels,
            (h, w),
            self.classes,
            self.threshold,
        )
