#!/usr/bin/env python
"""Ingest raw capture files: checksum, manifest row, dev/test split (P1-T4).

Thin CLI only - logic lives in src/vscs/capture/ingest.py (CLAUDE.md §3).

    python scripts/ingest.py --src <path> --kind scan|lot|calib
    python scripts/ingest.py --src <folder> --kind lot --stratum pole
    python scripts/ingest.py --src <file>   --kind lot --dry-run

The source is copied, never moved, and the copy is verified by re-hashing it afterwards
(R-13). Raw data is irreplaceable and is never modified after ingest (CLAUDE.md §0).

`lot` files are assigned to the dev or test split here, once and permanently (R-09). The
test split is opened exactly once, at P5-T2.

After this, and before deleting anything from the phone:
  * copy to an external drive and to cloud - three copies total
  * verify the sha256 after each copy, not before
  * update the `backups` column in data/MANIFEST.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vscs.capture.ingest import (
    STRATA,
    assign_split,
    infer_stratum,
    ingest_file,
    ingest_folder,
    make_raw_id,
)
from vscs.common.config import load_config
from vscs.common.log import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )
    parser.add_argument("--src", type=Path, required=True, help="file or folder to ingest")
    parser.add_argument("--kind", required=True, choices=["scan", "lot", "calib"])
    parser.add_argument("--description", default="", help="free text for the manifest row")
    parser.add_argument(
        "--stratum",
        default=None,
        choices=list(STRATA),
        help="obstacle category for split stratification (default: inferred from filename)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be ingested and which split it would land in; copy nothing",
    )
    args = parser.parse_args(argv)

    setup_logging(level=logging.INFO)

    if not args.src.exists():
        print(f"no such path: {args.src}")
        return 2

    if args.dry_run:
        return _dry_run(args)

    if args.src.is_dir():
        records = ingest_folder(
            args.src, kind=args.kind, description=args.description, stratum=args.stratum
        )
    else:
        records = [
            ingest_file(
                args.src, kind=args.kind, description=args.description, stratum=args.stratum
            )
        ]

    print(f"\ningested {len(records)} file(s):")
    for r in records:
        split = f"split={r.split}" if r.split else "no split"
        print(
            f"  {r.id}  {r.size_bytes / 1e6:7.1f} MB  {r.stratum:6s}  {split}  sha={r.sha256[:12]}"
        )

    n_unverified = sum(1 for r in records if not r.verified)
    if n_unverified:
        print(f"\nWARNING: {n_unverified} file(s) failed checksum verification.")
        return 1

    print(
        "\nAll copies verified against the source checksum.\n"
        "NEXT, before deleting anything from the phone:\n"
        "  1. copy to an external drive, then verify the sha256 again\n"
        "  2. copy to cloud, then verify again\n"
        "  3. update the `backups` column in data/MANIFEST.md\n"
        "  4. write today's devlog entry while you still remember what went wrong"
    )
    return 0


def _dry_run(args: argparse.Namespace) -> int:
    """Report ids, strata and splits without touching anything."""
    cfg = load_config("eval")["splits"]
    ratio_dev, seed = float(cfg["ratio_dev"]), int(cfg["seed"])

    files = (
        sorted(p for p in args.src.iterdir() if p.is_file()) if args.src.is_dir() else [args.src]
    )
    print(f"dry run - would ingest {len(files)} file(s) as kind={args.kind}\n")
    for p in files:
        raw_id = make_raw_id(args.kind, p)
        strat = args.stratum or infer_stratum(p.name)
        if args.kind == "lot":
            split = assign_split(raw_id, stratum=strat, ratio_dev=ratio_dev, seed=seed)
        else:
            split = "-"
        print(f"  {p.name:40s} -> {raw_id:38s} {strat:6s} split={split}")
    print("\nNothing was copied. Re-run without --dry-run to ingest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
