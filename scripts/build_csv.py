"""Derive the per-record CSVs this pipeline consumes from WFDB records.

Reimplements the original data-import notebook and
the original data-merge notebook, which originally ran with the
working directory set to the record folder.

For each record it writes:
  {id}.csv               signal columns, one row per sample, header = lead names
  {id}_annotations.csv   Sample, Classification, Aux Note
  {id}_ecg_annotated.csv signal left-joined with annotations on sample index

Idempotent: records whose three CSVs already exist are skipped. Use --force to
rebuild. The full set is roughly 7 GB, so rebuilding by accident is expensive.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from download_data import RECORDS


SUFFIXES = ("", "_annotations", "_ecg_annotated")


def is_built(rec: int, dst: Path) -> bool:
    """True if all three CSVs for this record exist and are non-empty."""
    return all(
        (dst / f"{rec}{suffix}.csv").exists()
        and (dst / f"{rec}{suffix}.csv").stat().st_size > 0
        for suffix in SUFFIXES
    )


def build_record(rec: int, src: Path, dst: Path) -> None:
    import pandas as pd
    import wfdb

    record = wfdb.rdrecord(str(src / str(rec)))
    annotation = wfdb.rdann(str(src / str(rec)), "atr")

    # {id}.csv -- p_signal with sig_name as header
    with open(dst / f"{rec}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(record.sig_name)
        w.writerows(record.p_signal)

    # {id}_annotations.csv
    with open(dst / f"{rec}_annotations.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Sample", "Classification", "Aux Note"])
        w.writerows(zip(annotation.sample, annotation.symbol,
                        annotation.aux_note))

    # {id}_ecg_annotated.csv -- left join of signal onto annotations by index
    ecg = pd.read_csv(dst / f"{rec}.csv")
    ann = pd.read_csv(dst / f"{rec}_annotations.csv")
    merged = pd.merge(ecg, ann, left_index=True, right_on="Sample", how="left")
    if "Sample" in merged.columns:
        merged.set_index("Sample", inplace=True)
    merged.to_csv(dst / f"{rec}_ecg_annotated.csv")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=Path("data/raw/mitdb"))
    p.add_argument("--dst", type=Path, default=Path("data/interim"))
    p.add_argument("--records", type=int, nargs="*", default=RECORDS)
    p.add_argument("--force", action="store_true",
                   help="rebuild records that are already present")
    args = p.parse_args()

    if not args.src.exists():
        print(f"error: {args.src} not found. Run scripts/download_data.py "
              f"first.", file=sys.stderr)
        return 1
    args.dst.mkdir(parents=True, exist_ok=True)

    todo = [
        rec for rec in args.records
        if args.force or not is_built(rec, args.dst)
    ]
    skipped = len(args.records) - len(todo)
    if skipped:
        print(f"skipping {skipped} record(s) already built (use --force to rebuild)")
    if not todo:
        print(f"All {len(args.records)} records present in {args.dst.resolve()}")
        return 0

    for rec in todo:
        try:
            build_record(rec, args.src, args.dst)
        except FileNotFoundError as exc:
            print(f"error: record {rec}: {exc}", file=sys.stderr)
            print("Run scripts/download_data.py first.", file=sys.stderr)
            return 1
        print(f"{rec} done")

    print(f"OK: {len(todo)} record(s) written to {args.dst.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
