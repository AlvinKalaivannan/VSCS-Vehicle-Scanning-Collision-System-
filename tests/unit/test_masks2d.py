"""P2-T2 masks stage on the fixture, with a fake detector and a fake video segmenter.

The fakes read the fixture renderer's exact labels, so a correct pipeline must reproduce
them pixel for pixel. The real Grounding DINO / SAM 2 adapters run only on Colab.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from fixtures.render import component_names, render_views
from fixtures.synthetic import load_scene
from vscs.common.config import load_config
from vscs.seg import masks2d as M
from vscs.seg.labels import NONE_LABEL

SCENE = load_scene()
FIXTURE_NAMES = component_names(SCENE)


@pytest.fixture(scope="module")
def scan(tmp_path_factory):
    """20 fixture views written as frames named like extract_frames does (with gaps)."""
    root = tmp_path_factory.mktemp("scan")
    truth, paths = [], []
    for i, (_, _, labels, _) in enumerate(render_views(SCENE, 0.25)):
        img = np.full((*labels.shape, 3), 90, dtype=np.uint8)
        img[labels != NONE_LABEL] = M.palette(len(FIXTURE_NAMES))[labels[labels != NONE_LABEL]]
        p = root / f"{i * 10:06d}.png"
        Image.fromarray(img).save(p)
        paths.append(p)
        truth.append(labels)
    return paths, truth


def _cfg(stride=5):
    seg = load_config("seg")
    # The fixture's components, in the fixture's (sorted) order, as the vocabulary.
    return {
        **seg,
        "components": {n: f"the {n.replace('_', ' ')}" for n in FIXTURE_NAMES},
        "segmenter": {**seg["segmenter"], "keyframe_stride": stride},
    }


class FakeDetector:
    """Returns each visible component's exact bounding box, plus a weak decoy."""

    def __init__(self, truth, paths):
        self.by_image = {}
        self.truth, self.paths = truth, paths
        self.calls = 0

    def detect(self, image, prompt):
        self.calls += 1
        name = prompt.removeprefix("the ").replace(" ", "_")
        labels = next(
            t
            for t, p in zip(self.truth, self.paths, strict=True)
            if np.array_equal(np.asarray(Image.open(p).convert("RGB")), image)
        )
        ys, xs = np.nonzero(labels == FIXTURE_NAMES.index(name))
        hits = [((0.0, 0.0, 5.0, 5.0), 0.05)]  # decoy below any sensible threshold
        if len(xs):
            hits.append(((float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())), 0.8))
        return hits


class FakeSegmenter:
    """Perfect tracker, streamed like SAM 2: a forward pass, then a reverse pass that
    revisits every frame with weaker (but still correct) logits, so merging is exercised."""

    def __init__(self, truth):
        self.truth = truth
        self.prompts = None

    def segment(self, frame_paths, box_prompts):
        self.prompts = box_prompts
        objs = {o for p in box_prompts.values() for o in p}
        for strength in (2.0, 1.0):
            order = range(len(self.truth)) if strength == 2.0 else reversed(range(len(self.truth)))
            for pos in order:
                t = self.truth[pos]
                yield (
                    pos,
                    {o: np.where(t == o, strength, -strength).astype(np.float32) for o in objs},
                )


def test_keyframes_cover_both_ends():
    assert M.select_keyframes(20, 5) == [0, 5, 10, 15, 19]
    assert M.select_keyframes(1, 15) == [0]
    with pytest.raises(ValueError):
        M.select_keyframes(0, 5)


def test_compose_takes_the_most_confident_object_per_pixel():
    a = np.array([[3.0, 1.0, -1.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0, -0.5]], dtype=np.float32)
    assert M.compose_labels({4: a, 7: b}, (1, 3)).tolist() == [[4, 7, NONE_LABEL]]
    assert (M.compose_labels({}, (2, 2)) == NONE_LABEL).all()
    with pytest.raises(ValueError, match="do not match"):
        M.compose_labels({0: a}, (2, 3))


def test_merging_two_passes_keeps_the_higher_logit_per_pixel():
    """Keeping only (winner, logit) per pass merges exactly like keeping every logit."""
    rng = np.random.default_rng(4)
    pass_a = {o: rng.normal(size=(6, 6)).astype(np.float32) for o in (0, 1, 2)}
    pass_b = {o: rng.normal(size=(6, 6)).astype(np.float32) for o in (0, 1, 2)}
    merged = M.merge_best(M.compose_best(pass_a, (6, 6)), M.compose_best(pass_b, (6, 6)))[0]
    everything = {o: np.maximum(pass_a[o], pass_b[o]) for o in pass_a}
    assert np.array_equal(merged, M.compose_labels(everything, (6, 6)))


def test_label_png_round_trip(tmp_path):
    lab = np.array([[0, 13, NONE_LABEL], [NONE_LABEL, 2, 2]])
    M.write_label_png(tmp_path / "x.png", lab)
    assert np.array_equal(M.read_label_png(tmp_path / "x.png"), lab)
    with pytest.raises(ValueError, match="uint8"):
        M.write_label_png(tmp_path / "y.png", np.array([[300]]))


def test_check_frames_are_seeded_and_bounded():
    a = M.check_frames(500, 20, seed=1)
    assert a == M.check_frames(500, 20, seed=1) and len(a) == 20 and a == sorted(set(a))
    assert M.check_frames(7, 20, seed=1) == list(range(7))


def test_pipeline_reproduces_the_exact_masks(scan, tmp_path):
    paths, truth = scan
    det, seg = FakeDetector(truth, paths), FakeSegmenter(truth)
    summary = M.run_masks(paths, tmp_path, _cfg(stride=5), det, seg)
    for p, t in zip(paths, truth, strict=True):
        assert np.array_equal(M.read_label_png(tmp_path / "labels" / p.name), t), p.name
    # Only keyframes were sent to the detector (5 keyframes x 12 prompts).
    assert det.calls == 5 * len(FIXTURE_NAMES)
    assert set(seg.prompts) <= set(summary.keyframes)
    # The decoy box (score 0.05) never became a prompt.
    assert all(box != (0.0, 0.0, 5.0, 5.0) for p in seg.prompts.values() for box in p.values())


def test_pipeline_logs_failures_and_writes_the_visual_check(scan, tmp_path):
    paths, truth = scan
    summary = M.run_masks(paths, tmp_path, _cfg(), FakeDetector(truth, paths), FakeSegmenter(truth))
    # The underbody is never visible from a standing-height scan: a logged failure.
    assert summary.never_detected == ["underbody"]
    assert summary.frames_with_component["underbody"] == 0
    assert summary.frames_with_component["rear_bumper"] > 0
    assert len(list((tmp_path / "check").glob("*.png"))) == min(20, len(paths))
    assert not (tmp_path / "_merge").exists()  # scratch merge files are cleaned up
    saved = json.loads((tmp_path / "masks_summary.json").read_text(encoding="utf-8"))
    assert saved["never_detected"] == ["underbody"] and saved["n_frames"] == len(paths)


def test_overlay_colours_only_labelled_pixels():
    img = np.zeros((1, 2, 3), dtype=np.uint8)
    out = M.overlay(img, np.array([[NONE_LABEL, 0]]), 3, alpha=1.0)
    assert out[0, 0].tolist() == [0, 0, 0]
    assert out[0, 1].tolist() == M.palette(3)[0].tolist()
