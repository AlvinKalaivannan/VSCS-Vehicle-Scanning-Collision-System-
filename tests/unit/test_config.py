"""Config loading tests, plus assertions that the shipped configs are well formed.

CLAUDE.md section 4.3: all tunable numbers live in ``configs/*.yaml``. These tests
make a missing or malformed config a test failure rather than a surprise at runtime.
"""

from __future__ import annotations

import itertools

import pytest

from vscs.common import config as C


def test_repo_root_looks_like_the_repo():
    root = C.repo_root()
    assert (root / "CLAUDE.md").is_file()
    assert (root / "configs").is_dir()
    assert (root / "src" / "vscs").is_dir()


@pytest.mark.parametrize("name", C.KNOWN_CONFIGS)
def test_every_declared_config_exists_and_loads(name):
    cfg = C.load_config(name)
    assert isinstance(cfg, dict) and cfg
    assert cfg.get("schema_version") == 1, f"{name}.yaml must carry schema_version"


def test_load_all_configs():
    all_cfg = C.load_all_configs()
    assert set(all_cfg) == set(C.KNOWN_CONFIGS)


def test_no_stray_configs_on_disk():
    """configs/ and KNOWN_CONFIGS must not drift apart."""
    on_disk = {p.stem for p in C.config_dir().glob("*.yaml")}
    assert on_disk == set(C.KNOWN_CONFIGS)


def test_load_config_accepts_filename_and_path():
    by_stem = C.load_config("risk")
    by_name = C.load_config("risk.yaml")
    by_path = C.load_config(str(C.config_dir() / "risk.yaml"))
    assert by_stem == by_name == by_path


def test_missing_config_raises_and_lists_alternatives():
    with pytest.raises(FileNotFoundError, match="Available"):
        C.load_config("does_not_exist")


def test_empty_config_is_rejected(tmp_path):
    (tmp_path / "empty.yaml").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        C.load_config("empty", config_dir_override=tmp_path)


def test_non_mapping_config_is_rejected(tmp_path):
    (tmp_path / "list.yaml").write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(TypeError, match="mapping"):
        C.load_config("list", config_dir_override=tmp_path)


# --------------------------------------------------------------------------- #
# Nested access                                                                #
# --------------------------------------------------------------------------- #
def test_get_reads_nested_keys():
    cfg = C.load_config("risk")
    assert C.get(cfg, "alerts.hysteresis.min_dwell_s") == 0.60
    assert C.get(cfg, "motion.horizon_s") == 3.0


def test_get_raises_on_typo_rather_than_defaulting():
    """A config typo must stop the run, not silently substitute a value."""
    cfg = C.load_config("risk")
    with pytest.raises(KeyError, match="not found"):
        C.get(cfg, "alerts.hysteresis.min_dwel_s")


def test_get_default_is_used_when_supplied():
    cfg = C.load_config("risk")
    assert C.get(cfg, "nope.nope", default=7) == 7
    # None must remain a usable default, distinct from "no default given".
    assert C.get(cfg, "nope.nope", default=None) is None


# --------------------------------------------------------------------------- #
# Overrides                                                                    #
# --------------------------------------------------------------------------- #
def test_merge_overrides_applies_and_does_not_mutate():
    cfg = C.load_config("risk")
    out = C.merge_overrides(cfg, {"motion.horizon_s": 1.5})
    assert C.get(out, "motion.horizon_s") == 1.5
    assert C.get(cfg, "motion.horizon_s") == 3.0


def test_merge_overrides_rejects_unknown_key():
    """Prevents a typo'd --set flag from looking like it worked."""
    cfg = C.load_config("risk")
    with pytest.raises(KeyError, match="does not exist"):
        C.merge_overrides(cfg, {"motion.horzon_s": 1.0})
    with pytest.raises(KeyError, match="does not exist"):
        C.merge_overrides(cfg, {"nonexistent.branch.key": 1.0})


def test_dump_config_round_trip(tmp_path):
    cfg = C.load_config("risk")
    path = C.dump_config(cfg, tmp_path / "sub" / "resolved.yaml")
    assert path.is_file()
    assert C.load_config(str(path)) == cfg


# --------------------------------------------------------------------------- #
# Contents the rest of the codebase relies on                                  #
# --------------------------------------------------------------------------- #
def test_risk_config_has_a_symmetric_curvature_fan():
    cfg = C.load_config("risk")
    fan = C.get(cfg, "motion.curvatures_inv_m")
    assert 5 <= len(fan) <= 7, "P3-T3 calls for 5-7 curvatures"
    assert 0.0 in fan, "the straight-ahead path must be in the fan"
    assert sorted(fan) == fan, "fan should be sorted for stable output ordering"
    assert [-c for c in reversed(fan)] == fan, "fan should be symmetric left/right"


def test_alert_levels_escalate_monotonically():
    """caution -> warning -> critical must tighten every threshold, or hysteresis
    cannot behave sensibly."""
    levels = C.get(C.load_config("risk"), "alerts.levels")
    order = ["caution", "warning", "critical"]
    for a, b in itertools.pairwise(order):
        assert levels[b]["min_distance_m_below"] < levels[a]["min_distance_m_below"]
        assert levels[b]["ttc_s_below"] < levels[a]["ttc_s_below"]
        assert levels[b]["p_contact_above"] > levels[a]["p_contact_above"]


def test_hysteresis_deescalates_slower_than_it_escalates():
    """R-10: alerts must not flicker."""
    h = C.get(C.load_config("risk"), "alerts.hysteresis")
    assert h["frames_to_deescalate"] > h["frames_to_escalate"]
    assert h["min_dwell_s"] > 0
    assert h["deescalate_distance_hysteresis_m"] > 0


def test_eval_thresholds_match_the_acceptance_criteria():
    """Values transcribed from CLAUDE.md section 6 - report.py depends on these."""
    th = C.get(C.load_config("eval"), "thresholds")
    assert th["scale_error_m"]["limit"] == 0.02
    assert th["scale_error_m"]["direction"] == "below"
    assert th["component_iou_mean"]["limit"] == 0.70
    assert th["component_iou_mean"]["direction"] == "above"
    assert th["reprojection_error_px"]["limit"] == 0.5
    assert th["frames_registered_frac"]["limit"] == 0.90
    for name, spec in th.items():
        assert spec["direction"] in ("above", "below"), name
        assert "task" in spec, name


def test_test_split_is_still_untouched():
    """R-09 tripwire. This must stay False until P5-T2 actually runs."""
    cfg = C.load_config("eval")
    assert C.get(cfg, "splits.test_split_used") is False, (
        "configs/eval.yaml says the held-out test split has been used. "
        "It may only be touched once, at P5-T2 (CLAUDE.md section 11, R-09)."
    )


def test_baseline_is_configured_because_it_is_never_cut():
    """CLAUDE.md section 6: the baseline comparison is the core result."""
    cfg = C.load_config("eval")
    assert C.get(cfg, "baseline.name") == "single_obb"
    assert "component_attribution_accuracy" in C.get(cfg, "report_metrics")


def test_capture_config_starts_uncalibrated():
    """Intrinsics are written by scripts/calibrate.py at P0-T7, not by hand."""
    cfg = C.load_config("capture")
    assert C.get(cfg, "intrinsics.calibrated") is False
    assert C.get(cfg, "intrinsics.K") is None
    assert C.get(cfg, "checkerboard.max_reprojection_error_px") == 0.5


def test_capture_never_uses_frame_index_as_time():
    """R-03 / section 4.1."""
    cfg = C.load_config("capture")
    assert C.get(cfg, "ingest.timestamp_source") == "container_pts"
    assert C.get(cfg, "ingest.require_monotonic_timestamps") is True


def test_component_vocabulary_is_approved():
    """P2-T1 acceptance: developer-approved list of at least 8 components."""
    seg = C.load_config("seg")
    assert seg["vocabulary_status"].startswith("APPROVED_"), (
        "the component vocabulary needs the developer's approval before it is used (P2-T1)"
    )
    assert C.load_config("severity")["status"].startswith("APPROVED_")
    assert len(seg["components"]) >= 8, "P2-T1 requires at least 8 components"


def test_severity_covers_exactly_the_approved_vocabulary():
    """The two files must not drift apart.

    A component with no severity would silently fall back to the default cost, and a
    severity for a component that no longer exists is dead weight that hides a rename.
    Either way the risk ranking would be quietly wrong rather than loudly broken.
    """
    vocabulary = set(C.load_config("seg")["components"])
    weighted = set(C.load_config("severity")["components"])
    assert weighted == vocabulary, (
        f"missing a severity: {sorted(vocabulary - weighted)}; "
        f"severity for an unknown component: {sorted(weighted - vocabulary)}"
    )


def test_every_component_has_a_text_prompt():
    """The prompt is what Grounding DINO is given at P2-T2; an empty one is useless."""
    for name, prompt in C.load_config("seg")["components"].items():
        assert isinstance(prompt, str) and len(prompt.strip()) > 3, name


def test_rear_geometry_is_finer_than_front():
    """This is a reversing system, so the rear carries the component resolution.

    Recorded as a test because it is a deliberate asymmetry, not an oversight: the
    rear bumper is split into corners and the front is not.
    """
    vocabulary = set(C.load_config("seg")["components"])
    assert {"rear_left_bumper_corner", "rear_right_bumper_corner"} <= vocabulary
    assert "rear_bumper" in vocabulary and "front_bumper" in vocabulary


def test_severity_ranks_a_person_far_above_any_panel():
    """Any contact involving a person must dominate the ranking."""
    sev = C.load_config("severity")
    mult = sev["obstacle_multiplier"]
    worst_component = max(sev["components"].values())
    assert mult["person"] * min(sev["components"].values()) > worst_component * mult["static_geom"]
    assert mult["person"] >= mult["cyclist"] > mult["vehicle"] >= mult["static_geom"]


def test_recon_scale_gate_is_two_centimetres():
    """P1-T6 acceptance criterion."""
    cfg = C.load_config("recon")
    assert C.get(cfg, "scale.max_dimension_error_m") == 0.02
    assert len(C.get(cfg, "scale.validate_against")) == 4


def test_every_random_stage_declares_a_seed():
    """Section 4.3: deterministic, seeded, logged."""
    for name in ("recon", "seg", "model", "perception", "eval"):
        cfg = C.load_config(name)
        dumped = str(cfg)
        assert "seed" in dumped, f"{name}.yaml declares no seed"
