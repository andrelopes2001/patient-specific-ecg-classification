"""Train the global model and run patient-specific fine-tuning.

    python scripts/train.py --dry-run          # build everything, train nothing
    python scripts/train.py --epochs 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecg.config import DS1, DS2, Config, seed_everything  # noqa: E402
from ecg.models import build  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", default="mlp", choices=["mlp", "cnn"])
    p.add_argument("--denoise", default="fir",
                   choices=["fir", "median", "wavelet", "none"])
    p.add_argument("--epochs", type=int, default=10,
                   help="epochs for the global model")
    p.add_argument("--class-weight", action="store_true",
                   help="enable inverse-frequency class weighting "
                        "(measured worse on DS2; see docs/experiments.md)")
    p.add_argument("--balance", default="none",
                   choices=["none", "oversample", "smote", "cgan"],
                   help="rebalance the DS1 training set before global training")
    p.add_argument("--cgan-epochs", type=int, default=20_000,
                   help="cGAN minibatch steps when --balance cgan")
    p.add_argument("--cgan-per-class", type=int, default=10_000,
                   help="synthetic beats per class when --balance cgan")
    p.add_argument("--train-minutes", type=int, default=5)
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--interim-dir", type=Path, default=Path("data/interim"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--dry-run", action="store_true",
                   help="build the dataset and model, report shapes, train nothing")
    args = p.parse_args()

    cfg = Config(
        interim_dir=args.interim_dir,
        denoise=None if args.denoise == "none" else args.denoise,
        train_minutes=args.train_minutes,
        seed=args.seed,
        max_epochs=args.epochs,
        class_weight=args.class_weight,
    )
    seed_everything(cfg.seed)

    model = build(args.arch, cfg=cfg)
    print(f"architecture : {args.arch} ({model.count_params():,} parameters)")
    print(f"window       : {cfg.window_length} samples ({cfg.window_ms:.0f} ms)")
    print(f"denoise      : {cfg.denoise}   balance: {args.balance}")
    print(f"training     : <={cfg.max_epochs} epochs, early stop patience "
          f"{cfg.early_stopping_patience}, class_weight={cfg.class_weight}")
    print(f"records      : DS1 {len(DS1)} train / DS2 {len(DS2)} evaluate")

    if args.dry_run:
        print("\n--dry-run: nothing trained.")
        return 0

    from ecg.segmentation import segment_records
    from ecg.train import evaluate_patient_specific, train_global

    try:
        print("\nsegmenting DS1 ...")
        df_train = segment_records(DS1, cfg)
        print(f"  {len(df_train):,} beats")

        print(f"training global model (<={cfg.max_epochs} epochs) ...")
        balance_kwargs = ({"epochs": args.cgan_epochs,
                           "n_per_class": args.cgan_per_class}
                          if args.balance == "cgan" else {})
        if args.balance != "none":
            print(f"rebalancing DS1 with {args.balance} ...")
        model, hist = train_global(df_train, model, cfg, balance=args.balance,
                                   balance_kwargs=balance_kwargs)
        print(f"  stopped after {len(hist.history['loss'])} epochs, "
              f"final loss {hist.history['loss'][-1]:.4f}")

        print("segmenting DS2 ...")
        df_test = segment_records(DS2, cfg)
        print(f"  {len(df_test):,} beats")
    except FileNotFoundError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1

    print("patient-specific fine-tuning ...")
    try:
        _, metrics = evaluate_patient_specific(df_test, model, DS2, cfg)
    except ValueError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    print("\n" + metrics.summary())

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(metrics.to_dict(), indent=2))
    print(f"wrote {args.out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
