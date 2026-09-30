"""2D component masks for every scan frame: Grounding DINO boxes -> SAM 2 tracking (P2-T2).

Acceptance (CLAUDE.md §6): a visual check on 20 random frames, with failures logged.

Pipeline (GPU, on Colab; ``notebooks/colab/20_seg.ipynb``):

1. **Keyframes.** Every ``segmenter.keyframe_stride``-th frame (and the last), so a
   component entering view between keyframes is still picked up.
2. **Detect.** On each keyframe, the open-vocabulary detector is asked for each component
   by its text prompt (``seg.yaml components``). Boxes below ``detector.box_threshold``
   are dropped. Each component keeps at most its best box per keyframe: every component
   exists once on the van.
3. **Track.** Each component becomes one SAM 2 object, prompted with its boxes on the
   keyframes where it was detected, and propagated through the whole video. The output
   is per-frame, per-object mask *logits* (> 0 means inside).
4. **Compose.** Per pixel, the component with the highest positive logit wins; no positive
   logit means ``NONE_LABEL``. Overlaps are therefore settled by the model's own
   confidence, not by the order components happen to be listed in.
5. **Save and check.** One label PNG per frame (uint8, 255 = none), a JSON log of the
   detections, per-component coverage (components never detected are listed as
   failures), and colour overlays of 20 seeded-random frames for the visual check.

The detector and segmenter are injected (the ``Detector`` / ``VideoSegmenter``
protocols), so steps 1, 2, 4 and 5 are tested on the laptop against the fixture
renderer. The real adapters (``GroundingDinoDetector``, ``Sam2VideoSegmenter``) import
their GPU libraries lazily and are only exercised on Colab. Record the working
library versions in an ADR on the first run (R-12).

Left/right: a text prompt's "left" is a hint only. The side is settled in 3D by the sign
of y (see ``seg.yaml``).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from vscs.common.log import get_logger
from vscs.seg.labels import NONE_LABEL

logger = get_logger("seg.masks2d")

#: Stored value for NONE_LABEL in the uint8 label PNGs.
PNG_NONE = 255

ImageArray = npt.NDArray[np.uint8]  # (H, W, 3) RGB


@dataclass(frozen=True)
class Detection:
    component: str
    box_xyxy: tuple[float, float, float, float]
    score: float


class Detector(Protocol):
    def detect(
        self, image: ImageArray, prompt: str
    ) -> list[tuple[tuple[float, float, float, float], float]]:
        """Boxes ``(x0, y0, x1, y1)`` in pixels, with scores, for one text prompt."""
        ...


class VideoSegmenter(Protocol):
    def segment(
        self,
        frame_paths: Sequence[Path],
        box_prompts: dict[int, dict[int, tuple[float, float, float, float]]],
    ) -> Iterator[tuple[int, dict[int, npt.NDArray[np.float32]]]]:
        """``{frame_pos: {obj_id: box}}`` prompts -> a stream of ``(frame_pos, {obj_id: logits})``.

        Streamed, one frame at a time: holding every frame's logits for every object would
        need tens of GB on a real scan. A frame may be yielded more than once (e.g. by a
        forward and a reverse tracking pass); the results are merged by highest logit.
        """
        ...


# --------------------------------------------------------------------------- #
# Steps 1-2                                                                    #
# --------------------------------------------------------------------------- #
def select_keyframes(n_frames: int, stride: int) -> list[int]:
    if n_frames < 1 or stride < 1:
        raise ValueError(f"need n_frames >= 1 and stride >= 1; got {n_frames}, {stride}")
    keys = list(range(0, n_frames, stride))
    if keys[-1] != n_frames - 1:
        keys.append(n_frames - 1)
    return keys


def detect_components(
    image: ImageArray, prompts: dict[str, str], detector: Detector, box_threshold: float
) -> list[Detection]:
    """Best box per component above the threshold, on one image."""
    out: list[Detection] = []
    for name, prompt in prompts.items():
        hits = [(b, s) for b, s in detector.detect(image, prompt) if s >= box_threshold]
        if hits:
            box, score = max(hits, key=lambda h: h[1])
            out.append(Detection(name, tuple(float(v) for v in box), float(score)))
    return out


# --------------------------------------------------------------------------- #
# Step 4                                                                       #
# --------------------------------------------------------------------------- #
def compose_best(
    logits: dict[int, npt.NDArray[np.float32]], shape: tuple[int, int]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    """Per pixel: ``(winning object or NONE_LABEL, its logit)``.

    Keeping only the winner and its logit is enough to merge later passes exactly:
    the best over (pass, object) is the better of each pass's best.
    """
    labels = np.full(shape, NONE_LABEL, dtype=np.int64)
    best = np.full(shape, -np.inf, dtype=np.float32)
    if not logits:
        return labels, best
    ids = sorted(logits)
    stack = np.stack([np.asarray(logits[i], dtype=np.float32) for i in ids])
    if stack.shape[1:] != tuple(shape):
        raise ValueError(f"logits {stack.shape[1:]} do not match the frame {shape}")
    arg = np.argmax(stack, axis=0)
    best = np.max(stack, axis=0)
    positive = best > 0
    labels[positive] = np.asarray(ids)[arg[positive]]
    return labels, best


def compose_labels(
    logits: dict[int, npt.NDArray[np.float32]], shape: tuple[int, int]
) -> npt.NDArray[np.int64]:
    """Per pixel: the object with the highest positive logit, else ``NONE_LABEL``."""
    return compose_best(logits, shape)[0]


def merge_best(
    a: tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]],
    b: tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    """Combine two passes over the same frame: per pixel, the higher logit wins."""
    take_b = b[1] > a[1]
    return np.where(take_b, b[0], a[0]), np.where(take_b, b[1], a[1])


# --------------------------------------------------------------------------- #
# Step 5                                                                       #
# --------------------------------------------------------------------------- #
def write_label_png(path: Path, labels: npt.ArrayLike) -> None:
    from PIL import Image

    lab = np.asarray(labels)
    if lab.max(initial=NONE_LABEL) >= PNG_NONE:
        raise ValueError(f"at most {PNG_NONE} components fit a uint8 label image")
    img = np.where(lab == NONE_LABEL, PNG_NONE, lab).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="L").save(path)


def read_label_png(path: Path) -> npt.NDArray[np.int64]:
    from PIL import Image

    img = np.asarray(Image.open(path)).astype(np.int64)
    return np.where(img == PNG_NONE, NONE_LABEL, img)


def check_frames(n_frames: int, k: int, seed: int) -> list[int]:
    """The seeded random frames for the P2-T2 visual check (all of them if fewer)."""
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(n_frames, size=min(k, n_frames), replace=False))


def palette(n: int) -> npt.NDArray[np.uint8]:
    """``n`` well-separated colours (golden-angle hues), fixed so overlays are comparable."""
    import colorsys

    return np.array(
        [
            [round(255 * c) for c in colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.85, 0.95)]
            for i in range(n)
        ],
        dtype=np.uint8,
    )


def overlay(
    image: ImageArray, labels: npt.ArrayLike, n_classes: int, alpha: float = 0.5
) -> ImageArray:
    lab = np.asarray(labels)
    out = image.astype(np.float32).copy()
    colours = palette(n_classes).astype(np.float32)
    m = lab != NONE_LABEL
    out[m] = (1 - alpha) * out[m] + alpha * colours[lab[m]]
    return out.round().astype(np.uint8)


@dataclass
class MasksSummary:
    n_frames: int
    keyframes: list[int]
    detections: dict[str, list[dict[str, Any]]]  # frame file -> detections
    frames_with_component: dict[str, int]
    never_detected: list[str]
    check_frames: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_masks(
    frame_paths: Sequence[Path],
    out_dir: Path,
    seg_cfg: dict[str, Any],
    detector: Detector,
    segmenter: VideoSegmenter,
) -> MasksSummary:
    """Steps 1-5 over one scan. Writes ``labels/``, ``check/`` and ``masks_summary.json``."""
    from PIL import Image

    frame_paths = [Path(p) for p in frame_paths]
    if not frame_paths:
        raise ValueError("no frames")
    names = list(seg_cfg["components"])
    prompts = dict(seg_cfg["components"])
    out_dir = Path(out_dir)

    keys = select_keyframes(len(frame_paths), int(seg_cfg["segmenter"]["keyframe_stride"]))
    box_prompts: dict[int, dict[int, tuple[float, float, float, float]]] = {}
    detections: dict[str, list[dict[str, Any]]] = {}
    for k in keys:
        image = np.asarray(Image.open(frame_paths[k]).convert("RGB"))
        dets = detect_components(
            image, prompts, detector, float(seg_cfg["detector"]["box_threshold"])
        )
        detections[frame_paths[k].name] = [asdict(d) for d in dets]
        if dets:
            box_prompts[k] = {names.index(d.component): d.box_xyxy for d in dets}

    detected = {names[i] for p in box_prompts.values() for i in p}
    never = [n for n in names if n not in detected]
    for n in never:
        logger.warning("%s: not detected on any keyframe - no masks (P2-T2 failure to log)", n)

    # Stream frames from the segmenter, merging repeat visits via small per-frame files
    # (winner + its logit), so memory stays at one frame regardless of scan length.
    merge_dir = out_dir / "_merge"
    merge_dir.mkdir(parents=True, exist_ok=True)
    seen: set[int] = set()
    stream = segmenter.segment(frame_paths, box_prompts) if box_prompts else iter(())
    for pos, frame_logits in stream:
        with Image.open(frame_paths[pos]) as im:
            shape = (im.height, im.width)
        cur = compose_best(frame_logits, shape)
        f = merge_dir / f"{pos}.npz"
        if pos in seen:
            with np.load(f) as z:
                cur = merge_best((z["labels"], z["best"]), cur)
        np.savez(f, labels=cur[0].astype(np.int16), best=cur[1].astype(np.float16))
        seen.add(pos)

    counts = dict.fromkeys(names, 0)
    for pos, path in enumerate(frame_paths):
        if pos in seen:
            with np.load(merge_dir / f"{pos}.npz") as z:
                labels = z["labels"].astype(np.int64)
        else:
            with Image.open(path) as im:
                labels = np.full((im.height, im.width), NONE_LABEL, dtype=np.int64)
        write_label_png(out_dir / "labels" / f"{path.stem}.png", labels)
        for i in np.unique(labels):
            if i != NONE_LABEL:
                counts[names[i]] += 1
    shutil.rmtree(merge_dir)

    check = check_frames(len(frame_paths), 20, int(seg_cfg["seed"]))
    for pos in check:
        image = np.asarray(Image.open(frame_paths[pos]).convert("RGB"))
        labels = read_label_png(out_dir / "labels" / f"{frame_paths[pos].stem}.png")
        (out_dir / "check").mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay(image, labels, len(names))).save(
            out_dir / "check" / f"{frame_paths[pos].stem}.png"
        )

    summary = MasksSummary(
        n_frames=len(frame_paths),
        keyframes=keys,
        detections=detections,
        frames_with_component=counts,
        never_detected=never,
        check_frames=[frame_paths[p].name for p in check],
    )
    (out_dir / "masks_summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
    )
    return summary


# --------------------------------------------------------------------------- #
# Real adapters - Colab only (GPU). Not exercised by the laptop tests.         #
# --------------------------------------------------------------------------- #
class GroundingDinoDetector:
    """Grounding DINO through Hugging Face ``transformers`` (code and weights Apache-2.0).

    One text query per component, lower-case and ending in ".", as the model expects.
    """

    def __init__(self, model_id: str, text_threshold: float, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.text_threshold = text_threshold

    def detect(self, image: ImageArray, prompt: str):
        text = prompt.lower().strip().rstrip(".") + "."
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.0,  # box threshold is applied (and logged) in detect_components
            text_threshold=self.text_threshold,
            target_sizes=[image.shape[:2]],
        )[0]
        return [
            (tuple(b.tolist()), float(s)) for b, s in zip(res["boxes"], res["scores"], strict=True)
        ]


class Sam2VideoSegmenter:
    """SAM 2 video predictor (``facebookresearch/sam2``; code and checkpoints Apache-2.0).

    SAM 2 reads a folder of JPEGs named by position (00000.jpg, ...). Extracted frames are
    named by *source* index with gaps (stride), so a positional JPEG copy is made first.
    """

    def __init__(
        self, model_cfg: str, checkpoint: str, work_dir: Path, device: str = "cuda"
    ) -> None:
        from sam2.build_sam import build_sam2_video_predictor

        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
        self.work_dir = Path(work_dir)

    def segment(self, frame_paths, box_prompts):
        import torch
        from PIL import Image

        jpg_dir = self.work_dir / "sam2_frames"
        jpg_dir.mkdir(parents=True, exist_ok=True)
        for pos, p in enumerate(frame_paths):
            Image.open(p).convert("RGB").save(jpg_dir / f"{pos:05d}.jpg", quality=95)
        try:
            yield from self._track(torch, jpg_dir, box_prompts)
        finally:
            shutil.rmtree(jpg_dir, ignore_errors=True)  # temporary copies, not outputs

    def _track(self, torch, jpg_dir, box_prompts):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # Keep frames and per-frame state in CPU RAM: a few hundred frames would
            # otherwise take several GB of the T4's 16 GB.
            state = self.predictor.init_state(
                video_path=str(jpg_dir), offload_video_to_cpu=True, offload_state_to_cpu=True
            )
            for pos, objs in box_prompts.items():
                for obj_id, box in objs.items():
                    self.predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=pos, obj_id=obj_id, box=np.asarray(box)
                    )
            # Forward from the earliest prompt covers every object from its own first
            # detection onward; reverse from the latest prompt covers every object back to
            # frame 0. Together each object is tracked across the whole scan. Frames seen by
            # both passes are merged by the caller (highest logit wins).
            for reverse, start in ((False, min(box_prompts)), (True, max(box_prompts))):
                for pos, obj_ids, mask_logits in self.predictor.propagate_in_video(
                    state, start_frame_idx=start, reverse=reverse
                ):
                    yield (
                        pos,
                        {
                            int(o): mask_logits[i, 0].float().cpu().numpy()
                            for i, o in enumerate(obj_ids)
                        },
                    )
