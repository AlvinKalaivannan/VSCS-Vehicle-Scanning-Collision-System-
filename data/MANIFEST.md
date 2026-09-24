# Raw data manifest

Every file in `data/raw/` is recorded here. **This is the only file under `data/` that is ever committed**
(CLAUDE.md §4.4). `data/raw/` is irreplaceable and is never modified after ingest - ask before touching it.

Backup policy (R-13): **three copies** - laptop, external drive, cloud - with the sha256 verified *after* each
copy, not before.

Written by `scripts/ingest.py`; the `verified` column is set only after a post-copy checksum comparison.

| id | date | kind | sha256 | duration / frames | description | backups | split | verified |
|---|---|---|---|---|---|---|---|---|
| _(no raw data ingested yet)_ | | | | | | | | |

## Columns

- **id** - stable identifier, e.g. `scan_20261003_van_loop1`, `lot_20261011_pass07`.
- **kind** - `scan` (vehicle scan) or `lot` (parking-lot drive) or `calib` (checkerboard).
- **sha256** - of the original file as written by the phone.
- **backups** - which of the three locations currently hold it.
- **split** - `dev` or `test`, assigned at ingest and never changed afterwards (R-09). The `test` split is
  touched exactly once, at P5-T2.
