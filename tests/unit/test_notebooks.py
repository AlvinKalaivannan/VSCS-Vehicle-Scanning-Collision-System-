"""Notebook hygiene (P0-T5).

CLAUDE.md section 4.3: notebooks contain no logic - only install, mount, and a call
into ``src/vscs``. These tests cannot judge intent, but they do catch the two things
that actually go wrong: committed outputs (which bloat the repo and can leak data) and
a notebook that has stopped being valid JSON.
"""

from __future__ import annotations

import nbformat
import pytest

from vscs.common.config import repo_root

NOTEBOOK_DIR = repo_root() / "notebooks" / "colab"


def _notebooks():
    return sorted(NOTEBOOK_DIR.glob("*.ipynb"))


def test_the_template_exists():
    assert (NOTEBOOK_DIR / "00_template.ipynb").is_file()


@pytest.mark.parametrize("path", _notebooks(), ids=lambda p: p.name)
def test_notebook_is_valid(path):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)


@pytest.mark.parametrize("path", _notebooks(), ids=lambda p: p.name)
def test_notebook_has_no_committed_outputs(path):
    """Outputs can carry recorded frames, file paths and GPU details. Keep them out."""
    nb = nbformat.read(path, as_version=4)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            assert not cell.get("outputs"), f"{path.name} cell {i} has committed outputs"
            assert cell.get("execution_count") in (None, 0), (
                f"{path.name} cell {i} carries an execution count"
            )


def test_template_checks_the_gpu_and_mounts_drive():
    """The two things P0-T5 must demonstrate."""
    nb = nbformat.read(NOTEBOOK_DIR / "00_template.ipynb", as_version=4)
    source = "\n".join(c.source for c in nb.cells)
    assert "nvidia-smi" in source
    assert "torch.cuda.is_available" in source
    assert "drive.mount" in source
    # Writes something to Drive - the hello-world acceptance check.
    assert "colab_hello.txt" in source


def test_template_does_not_pin_torch():
    """Overriding Colab's torch build is the usual way to break the runtime (R-12)."""
    nb = nbformat.read(NOTEBOOK_DIR / "00_template.ipynb", as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for line in cell.source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "pip install" not in stripped or "torch" not in stripped, (
                f"notebook installs torch: {stripped!r}"
            )


def test_colab_requirements_avoids_agpl_ultralytics():
    """CLAUDE.md section 10: Ultralytics YOLO is AGPL-3.0 and is not to be used."""
    text = (repo_root() / "envs" / "colab_requirements.txt").read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert "ultralytics" not in stripped.lower(), f"AGPL dependency declared: {stripped!r}"
