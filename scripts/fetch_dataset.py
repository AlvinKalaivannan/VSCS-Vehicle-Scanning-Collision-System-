#!/usr/bin/env python
"""Download a registered external dataset into data/external/<id>/ (ADR 0006).

Thin CLI only - logic lives in src/vscs/common/fetch.py. The dataset must be listed in
configs/datasets.yaml with a licence that allows at least private rehearsal.

    python scripts/fetch_dataset.py roboflow_car_parts_21
    python scripts/fetch_dataset.py roboflow_car_parts_19 --format coco-segmentation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vscs.common.datasets import DatasetUseError
from vscs.common.fetch import fetch_roboflow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", help="id in configs/datasets.yaml")
    parser.add_argument("--format", default="coco-segmentation", help="Roboflow export format")
    args = parser.parse_args(argv)
    try:
        out = fetch_roboflow(args.dataset, args.format)
    except (DatasetUseError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"extracted to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
