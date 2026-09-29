"""External dataset registry and use guard (ADR 0006)."""

from __future__ import annotations

import copy
from datetime import date

import pytest

from vscs.common import datasets as D
from vscs.common.config import load_config

CFG = load_config("datasets")
POLICY = CFG["policy"]
REGISTRY = D.load_registry(CFG)


def _cfg_with(entry: dict) -> dict:
    cfg = copy.deepcopy(CFG)
    cfg["datasets"] = {"x": entry}
    return cfg


def _entry(**over) -> dict:
    base = {
        "name": "x",
        "url": "https://example.org",
        "license": "CC-BY-4.0",
        "license_url": "https://example.org/l",
        "license_status": "clear",
        "verified": date(2026, 9, 29),
        "adopted": True,
        "covers": ["part_vocabulary"],
        "notes": "",
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------- #
# The registry itself                                                          #
# --------------------------------------------------------------------------- #
def test_registry_loads_and_every_entry_is_complete():
    assert REGISTRY
    for e in REGISTRY.values():
        assert e.url.startswith("https://"), e.id
        assert e.covers, f"{e.id}: say which gap it covers"
        if e.license_status in ("clear", "conflicting"):
            assert e.verified is not None, f"{e.id}: a checked licence needs a check date"
            assert e.license_url, e.id
        if e.license_status == "none":
            assert e.license is None, f"{e.id}: status none but a licence is named"


def test_policy_uses_are_known():
    for cls, uses in POLICY["uses_by_class"].items():
        assert set(uses) <= set(D.USES), cls


def test_only_permissive_clear_licences_can_publish_media_or_train():
    for e in REGISTRY.values():
        uses = D.allowed_uses(e, POLICY)
        if "publish_media" in uses or "train" in uses:
            assert e.license_status == "clear" and e.license in POLICY["permissive_licenses"]


def test_the_known_licence_conflicts_are_private_only():
    """3DRealCar and Tanks and Temples claim CC BY / Apache but contradict themselves."""
    for ds in ("threedrealcar", "tanks_and_temples"):
        assert D.allowed_uses(REGISTRY[ds], POLICY) == {"rehearsal", "private_eval"}


def test_roboflow_sets_match_what_the_api_reported_on_2026_09_30():
    """21-class: clear CC BY, but undocumented photos -> never shown publicly.
    19-class: DSMLR's unlicensed classes re-uploaded as CC BY -> private only."""
    assert D.allowed_uses(REGISTRY["roboflow_car_parts_21"], POLICY) == {
        "rehearsal",
        "private_eval",
        "publish_results",
        "train",
    }
    assert D.allowed_uses(REGISTRY["roboflow_car_parts_19"], POLICY) == {
        "rehearsal",
        "private_eval",
    }


def test_external_data_never_enters_our_lot_splits():
    """R-09: dev/test are our lot passes only."""
    splits = load_config("eval")["splits"]
    ours = set(splits.get("dev") or []) | set(splits.get("test") or [])
    assert not ours & set(REGISTRY)


# --------------------------------------------------------------------------- #
# Classification                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "over,cls",
    [
        ({}, "permissive"),
        ({"license": "CC-BY-NC-4.0"}, "noncommercial"),
        ({"license_status": "conflicting"}, "conflicting"),
        ({"license_status": "unverified", "verified": None}, "unverified"),
        ({"license_status": "none", "license": None}, "none"),
        # clear but not in either list: a human classifies it, so it grants nothing
        ({"license": "GPL-3.0-only"}, "unverified"),
    ],
)
def test_license_class(over, cls):
    e = D.load_registry(_cfg_with(_entry(**over)))["x"]
    assert D.license_class(e, POLICY) == cls


def test_a_claimed_permissive_licence_is_not_trusted_until_consistent():
    e = D.load_registry(_cfg_with(_entry(license_status="conflicting")))["x"]
    assert "publish_media" not in D.allowed_uses(e, POLICY)


# --------------------------------------------------------------------------- #
# The guard                                                                    #
# --------------------------------------------------------------------------- #
def test_require_use_returns_the_external_dir():
    path = D.require_use("x", "publish_media", cfg=_cfg_with(_entry()))
    assert path.parts[-3:] == ("data", "external", "x")


@pytest.mark.parametrize(
    "over,use",
    [
        ({"license": "CC-BY-NC-4.0"}, "publish_media"),
        ({"license": "CC-BY-NC-4.0"}, "train"),
        ({"license_status": "conflicting"}, "publish_results"),
        ({"license_status": "unverified", "verified": None}, "rehearsal"),
        ({"license_status": "none", "license": None}, "rehearsal"),
    ],
)
def test_require_use_refuses(over, use):
    with pytest.raises(D.DatasetUseError, match="not allowed"):
        D.require_use("x", use, cfg=_cfg_with(_entry(**over)))


def test_unregistered_dataset_is_refused():
    with pytest.raises(D.DatasetUseError, match=r"not in configs/datasets\.yaml"):
        D.require_use("some_youtube_video", "rehearsal")


def test_unknown_use_is_a_programming_error():
    with pytest.raises(ValueError, match="unknown use"):
        D.require_use("uco3d", "sell")


def test_undocumented_provenance_blocks_publishing_images_only():
    e = D.load_registry(_cfg_with(_entry(provenance="undocumented")))["x"]
    uses = D.allowed_uses(e, POLICY)
    assert "publish_media" not in uses
    assert {"train", "publish_results"} <= uses


def test_bad_provenance_is_rejected_at_load():
    with pytest.raises(ValueError, match="provenance"):
        D.load_registry(_cfg_with(_entry(provenance="trust_me")))


def test_bad_status_is_rejected_at_load():
    with pytest.raises(ValueError, match="license_status"):
        D.load_registry(_cfg_with(_entry(license_status="probably_fine")))
