"""COLMAP dense-stage tests (P1-T7). COLMAP is replaced by a fake runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from vscs.common.config import load_config
from vscs.recon import dense as D

DENSE = load_config("recon")["dense"]


def _ply(path: Path, n: int, binary: bool = False) -> Path:
    fmt = "binary_little_endian 1.0" if binary else "ascii 1.0"
    header = f"ply\nformat {fmt}\nelement vertex {n}\nproperty float x\nproperty float y\n"
    header += "property float z\nend_header\n"
    path.write_bytes(header.encode() + (b"\x00" * 12 * min(n, 10) if binary else b""))
    return path


def _runner(banner: str, n_points: int | None = 50_000):
    calls: list[list[str]] = []

    def run(cmd, **kw):
        calls.append(list(cmd))
        if cmd[1] == "help":
            return SimpleNamespace(stdout=banner, stderr="")
        if cmd[1] == "stereo_fusion" and n_points is not None:
            _ply(Path(cmd[cmd.index("--output_path") + 1]), n_points)
        return SimpleNamespace(stdout="", stderr="")

    return run, calls


# --------------------------------------------------------------------------- #
# CUDA detection                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "banner,expected",
    [
        ("COLMAP 3.9.1 (Commit 0d9a4e3 on 2024-01-01 with CUDA)", "cuda"),
        ("COLMAP 3.9.1 (Commit 0d9a4e3 on 2024-01-01 without CUDA)", "no_cuda"),
        # COLMAP 4.x wording; the no-GPU line is verbatim from the 4.2.0 Windows release.
        ("COLMAP 4.2.0 (Commit be5e291 on 2026-08-31 without GPU support)", "no_cuda"),
        ("COLMAP 4.2.0 (Commit be5e291 on 2026-08-31 with GPU support)", "cuda"),
        ("COLMAP 3.9.1", "unknown"),
        ("", "unknown"),
    ],
)
def test_parse_cuda_status(banner, expected):
    """'without CUDA' contains 'with CUDA', so the order of the checks matters."""
    assert D.parse_cuda_status(banner) == expected


def test_status_is_read_from_the_binary():
    run, calls = _runner("COLMAP 3.9 (with CUDA)")
    assert D.colmap_cuda_status("colmap", run) == "cuda"
    assert calls == [["colmap", "help"]]


def test_missing_binary_is_unknown_not_a_crash():
    def run(cmd, **kw):
        raise FileNotFoundError(cmd[0])

    assert D.colmap_cuda_status("colmap", run) == "unknown"


# --------------------------------------------------------------------------- #
# Commands                                                                     #
# --------------------------------------------------------------------------- #
def test_three_stages_then_text_model_in_order_as_argument_lists(tmp_path):
    cmds = D.build_dense_commands(
        "colmap", tmp_path / "img", tmp_path / "sp", tmp_path / "ws", DENSE
    )
    assert [c[1] for c in cmds] == [
        "image_undistorter",
        "patch_match_stereo",
        "stereo_fusion",
        "model_converter",
    ]
    conv = cmds[3]
    assert conv[conv.index("--output_type") + 1] == "TXT"
    assert conv[conv.index("--output_path") + 1].endswith("sparse_txt")
    assert all(isinstance(a, str) for c in cmds for a in c)
    und = cmds[0]
    assert und[und.index("--max_image_size") + 1] == str(DENSE["max_image_size"])
    assert cmds[2][cmds[2].index("--output_path") + 1].endswith("fused.ply")


def test_geometric_consistency_switch(tmp_path):
    on = D.build_dense_commands(
        "c", tmp_path, tmp_path, tmp_path, {**DENSE, "geom_consistency": True}
    )
    off = D.build_dense_commands(
        "c", tmp_path, tmp_path, tmp_path, {**DENSE, "geom_consistency": False}
    )
    assert on[1][on[1].index("--PatchMatchStereo.geom_consistency") + 1] == "true"
    assert on[2][on[2].index("--input_type") + 1] == "geometric"
    assert off[2][off[2].index("--input_type") + 1] == "photometric"


# --------------------------------------------------------------------------- #
# PLY header                                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("binary", [False, True])
def test_reads_vertex_count_from_the_header(tmp_path, binary):
    assert D.read_ply_vertex_count(_ply(tmp_path / "c.ply", 123_456, binary)) == 123_456


def test_rejects_non_ply_and_headerless(tmp_path):
    bad = tmp_path / "x.ply"
    bad.write_bytes(b"not a ply\n")
    with pytest.raises(ValueError, match="not a PLY"):
        D.read_ply_vertex_count(bad)
    empty = tmp_path / "e.ply"
    empty.write_bytes(b"ply\nformat ascii 1.0\nend_header\n")
    with pytest.raises(ValueError, match="element vertex"):
        D.read_ply_vertex_count(empty)


# --------------------------------------------------------------------------- #
# Running                                                                      #
# --------------------------------------------------------------------------- #
def test_full_run(tmp_path):
    run, calls = _runner("COLMAP 3.9 (with CUDA)", n_points=250_000)
    res = D.run_dense(tmp_path / "img", tmp_path / "sp", tmp_path / "ws", DENSE, runner=run)
    assert [c[1] for c in calls] == [
        "help",
        "image_undistorter",
        "patch_match_stereo",
        "stereo_fusion",
        "model_converter",
    ]
    assert (tmp_path / "ws" / "sparse_txt").is_dir()
    assert res.n_points == 250_000 and res.plausible


def test_refuses_to_start_without_cuda(tmp_path):
    """Fail before undistortion, not an hour later at patch_match_stereo."""
    run, calls = _runner("COLMAP 3.9 (without CUDA)")
    with pytest.raises(RuntimeError, match="requires CUDA"):
        D.run_dense(tmp_path, tmp_path, tmp_path / "ws", DENSE, runner=run)
    assert [c[1] for c in calls] == ["help"]


def test_unknown_build_warns_and_proceeds(tmp_path, caplog):
    run, _ = _runner("COLMAP 3.9")
    with caplog.at_level("WARNING"):
        res = D.run_dense(tmp_path, tmp_path, tmp_path / "ws", DENSE, runner=run)
    assert "could not read CUDA" in caplog.text and res.plausible


def test_implausibly_small_cloud_is_flagged(tmp_path, caplog):
    run, _ = _runner("with CUDA", n_points=500)
    with caplog.at_level("WARNING"):
        res = D.run_dense(tmp_path, tmp_path, tmp_path / "ws", DENSE, runner=run)
    assert not res.plausible and "failed quietly" in caplog.text


def test_missing_fused_cloud_is_an_error(tmp_path):
    run, _ = _runner("with CUDA", n_points=None)
    with pytest.raises(RuntimeError, match="wrote no"):
        D.run_dense(tmp_path, tmp_path, tmp_path / "ws", DENSE, runner=run)
