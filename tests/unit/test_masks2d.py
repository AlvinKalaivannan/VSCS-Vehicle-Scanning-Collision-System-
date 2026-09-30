"""P2-T2 masks stage on the fixture, with a fake detector and a fake video segmenter.

The fakes read the fixture renderer's exact labels, so a correct pipeline must reproduce
them pixel for pixel. The real Grounding DINO / SAM 2 adapters run only on Colab.
"""

from __future__ import annotations

import contextlib
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


class _FakeTorch:
    """Just enough of torch to check which autocast the adapter would ask for."""

    bfloat16, float16 = "bf16", "fp16"

    def __init__(self, bf16_ok):
        self.cuda = type("cuda", (), {"is_bf16_supported": staticmethod(lambda: bf16_ok)})
        self.requested = None

    def autocast(self, device_type, dtype):
        self.requested = (device_type, dtype)
        return contextlib.nullcontext()


@pytest.mark.parametrize(
    "device,bf16_ok,expected",
    [
        ("cuda", True, ("cuda", "bf16")),  # Ampere and newer (e.g. A100)
        ("cuda", False, ("cuda", "fp16")),  # the Colab T4: bfloat16 would raise
        ("cpu", True, None),  # no autocast at all
    ],
)
def test_autocast_matches_the_hardware(device, bf16_ok, expected):
    torch = _FakeTorch(bf16_ok)
    with M.autocast_for(torch, device):
        pass
    assert torch.requested == expected


def test_filter_drops_whole_vehicle_boxes_and_clips():
    """The real-photo finding: 'side mirror of a car' put the whole car on top, as 'car'."""
    kept = M.filter_grounded(
        boxes=[[27, 32, 354, 279], [326, 69, 354, 87], [-5, 10, 20, 400], [50, 50, 50, 60]],
        scores=[0.64, 0.45, 0.4, 0.9],
        labels=["car", "side mirror", "side mirror", "side mirror"],
        image_hw=(308, 414),
        ignore_labels=["car", "van"],
    )
    assert kept == [((326.0, 69.0, 354.0, 87.0), 0.45), ((0.0, 10.0, 20.0, 307.0), 0.4)]


def test_no_prompt_names_the_whole_vehicle():
    """A prompt like '... of a van' makes the detector return the whole van (see above).

    'vehicle' is allowed inside a longer part name ('vehicle underbody'): a box matched to
    that phrase is kept, and one matched to 'vehicle' alone is dropped by ignore_labels.
    """
    seg = load_config("seg")
    whole_vehicle = set(seg["detector"]["ignore_labels"]) - {"vehicle"}
    for name, prompt in seg["components"].items():
        words = prompt.lower().replace(".", "").split()
        assert not whole_vehicle & set(words), (name, prompt)
        assert " of a " not in f" {prompt.lower()} ", (name, prompt)


class _T(list):
    """Stands in for a tensor: only ``.tolist()`` is used."""

    def tolist(self):
        return list(self)


def test_empty_result_with_a_stray_label_is_no_detections():
    """transformers 5.17 returns labels [''] beside zero boxes."""
    res = {"boxes": _T([]), "scores": _T([]), "text_labels": [""], "labels": [""]}
    assert M.grounded_to_hits(res, (10, 10), ["car"]) == []


def test_mismatched_labels_are_an_error_not_a_guess():
    res = {"boxes": _T([[0, 0, 5, 5]]), "scores": _T([0.9]), "text_labels": ["a", "b"]}
    with pytest.raises(ValueError, match="1 boxes but 2 labels"):
        M.grounded_to_hits(res, (10, 10), [])
