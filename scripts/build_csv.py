"""Derive the per-record CSVs the thesis pipeline consumes from WFDB records.

Reimplements the original data-import notebook and
the original data-merge notebook, which originally ran with the
working directory set to the record folder.

For each record it writes:
  {id}.csv               signal columns, one row per sample, header = lead names
  {id}_annotations.csv   Sample, Classification, Aux Note
  {id}_ecg_annotated.csv signal left-joined with annotations on sample index
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from download_data import RECORDS


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
    args = p.parse_args()

    if not args.src.exists():
        print(f"error: {args.src} not found. Run scripts/download_data.py "
              f"first.", file=sys.stderr)
        return 1
    args.dst.mkdir(parents=True, exist_ok=True)

    for rec in args.records:
        build_record(rec, args.src, args.dst)
        print(f"{rec} done")
    print(f"OK: {len(args.records)} records written to {args.dst.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
