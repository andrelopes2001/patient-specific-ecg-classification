"""Score saved predictions, or re-render a metrics table.

    python scripts/evaluate.py --predictions tests/fixtures/predictions_ds2.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecg.evaluate import compute_metrics, metrics_table  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, required=True,
                   help="npz with 'ground_truth' and 'predictions' arrays")
    p.add_argument("--json-out", type=Path)
    args = p.parse_args()

    if not args.predictions.exists():
        print(f"error: {args.predictions} not found", file=sys.stderr)
        return 1

    data = np.load(args.predictions)
    missing = {"ground_truth", "predictions"} - set(data.files)
    if missing:
        print(f"error: {args.predictions} is missing {sorted(missing)}",
              file=sys.stderr)
        return 1

    metrics = compute_metrics(data["ground_truth"], data["predictions"])
    print(metrics_table(metrics))
    print("\n" + metrics.summary())

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(metrics.to_dict(), indent=2))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
