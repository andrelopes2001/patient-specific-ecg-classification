"""Download the MIT-BIH Arrhythmia Database (mitdb) from PhysioNet.

The pipeline consumes per-record CSVs derived from the WFDB records. This
script fetches the raw records only; `build_csv.py` derives the CSVs.

Idempotent: records already present with a non-empty .dat/.hea/.atr triple are
skipped. Run with --force to re-download everything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The 48 records of the MIT-BIH Arrhythmia Database. Records 102/104/107/217
# are paced and are excluded later by the AAMI convention, but they are
# downloaded for completeness.
RECORDS = [
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112,
    113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200,
    201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215,
    217, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234,
]

EXTENSIONS = (".dat", ".hea", ".atr")
DB_NAME = "mitdb"


def missing_records(dl_dir: Path) -> list[int]:
    """Return records that are absent or incomplete in dl_dir."""
    missing = []
    for rec in RECORDS:
        if not all((dl_dir / f"{rec}{ext}").stat().st_size > 0
                   if (dl_dir / f"{rec}{ext}").exists() else False
                   for ext in EXTENSIONS):
            missing.append(rec)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dl-dir", type=Path, default=Path("data/raw/mitdb"),
                        help="destination directory (default: data/raw/mitdb)")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if records are already present")
    args = parser.parse_args()

    try:
        import wfdb
    except ImportError:
        print("error: wfdb is not installed. Run: uv pip install wfdb",
              file=sys.stderr)
        return 1

    dl_dir = args.dl_dir.resolve()
    dl_dir.mkdir(parents=True, exist_ok=True)

    todo = RECORDS if args.force else missing_records(dl_dir)
    if not todo:
        print(f"All {len(RECORDS)} records already present in {dl_dir}")
        return 0

    print(f"Downloading {len(todo)} of {len(RECORDS)} records "
          f"from PhysioNet '{DB_NAME}' into {dl_dir}")

    try:
        wfdb.dl_database(DB_NAME, dl_dir=str(dl_dir),
                         records=[str(r) for r in todo])
    except Exception as exc:
        print(f"error: download failed: {exc}", file=sys.stderr)
        print("Check your network connection and that physionet.org is "
              "reachable.", file=sys.stderr)
        return 1

    still_missing = missing_records(dl_dir)
    if still_missing:
        print(f"error: {len(still_missing)} records still missing after "
              f"download: {still_missing}", file=sys.stderr)
        return 1

    total_mb = sum(f.stat().st_size for f in dl_dir.iterdir() if f.is_file())
    total_mb /= 1024 * 1024
    print(f"OK: {len(RECORDS)} records ({total_mb:.1f} MB) in {dl_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
