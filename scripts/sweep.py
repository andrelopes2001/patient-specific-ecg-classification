"""Sweep the amount of patient-specific fine-tuning data.

Produces results/reproduced_metrics.csv, the data behind the README's headline
figure. Equivalent to running scripts/train.py once per --train-minutes value,
but segments the records a single time instead of once per setting.

    python scripts/sweep.py                      # 1-5 minutes, one seed
    python scripts/sweep.py --seeds 12 13 14     # averaged over seeds
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecg.config import CLASSES, DS1, DS2, Config, seed_everything  # noqa: E402
from ecg.models import build  # noqa: E402
from ecg.segmentation import segment_records  # noqa: E402
from ecg.train import evaluate_patient_specific, train_global  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", default="mlp", choices=["mlp", "cnn"])
    p.add_argument("--minutes", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--seeds", type=int, nargs="+", default=[12])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--class-weight", action="store_true",
                   help="enable inverse-frequency class weighting "
                        "(measured worse on DS2; see docs/experiments.md)")
    p.add_argument("--interim-dir", type=Path, default=Path("data/interim"))
    p.add_argument("--out", type=Path, default=Path("results/reproduced_metrics.csv"))
    args = p.parse_args()

    base = Config(interim_dir=args.interim_dir, max_epochs=args.epochs,
                  class_weight=args.class_weight)
    seed_everything(base.seed)

    print("segmenting DS1 and DS2 once ...")
    try:
        df1 = segment_records(DS1, base)
        df2 = segment_records(DS2, base)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"  DS1 {len(df1):,} beats   DS2 {len(df2):,} beats")

    rows = []
    for minutes in args.minutes:
        cfg = Config(interim_dir=args.interim_dir, max_epochs=args.epochs,
                     class_weight=args.class_weight,
                     train_minutes=minutes)
        per_seed = []
        for seed in args.seeds:
            seed_everything(seed)
            model, _ = train_global(df1, build(args.arch, cfg=cfg), cfg)
            _, m = evaluate_patient_specific(df2, model, DS2, cfg)
            per_seed.append(m)
            print(f"  {minutes} min, seed {seed}: {m.summary()}", flush=True)

        agg = lambda fn: float(np.mean([fn(m) for m in per_seed]))  # noqa: E731
        std = lambda fn: float(np.std([fn(m) for m in per_seed]))   # noqa: E731
        row = {
            "augmentation": "none", "extra_features": "none",
            "train_minutes": minutes, "test_from_minute": cfg.test_minute,
            "seeds": len(args.seeds),
            "accuracy": round(agg(lambda m: m.micro["Accuracy"]), 4),
            "accuracy_std": round(std(lambda m: m.micro["Accuracy"]), 4),
            "macro_f1": round(agg(lambda m: m.macro["F1 Score"]), 4),
            "macro_f1_std": round(std(lambda m: m.macro["F1 Score"]), 4),
            "cohen_kappa": round(agg(lambda m: m.cohen_kappa), 4),
        }
        for i, name in enumerate(CLASSES):
            row[f"f1_{name}"] = round(agg(lambda m, i=i: m.per_class["F1 Score"][i]), 3)
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
