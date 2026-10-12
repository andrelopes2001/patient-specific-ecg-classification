"""Export to ONNX, quantise to INT8, and measure accuracy, size and CPU latency.

Writes two tables:

  results/deployment_metrics.csv   pooled DS2 metrics per model format
  results/deployment_latency.csv   size on disk and single-thread CPU latency

Accuracy is measured on the real deliverable: every DS2 patient's own
fine-tuned model, exported and quantised individually, its INT8 ranges
calibrated on that patient's fine-tuning window. All formats score the *same*
weights in the same process, so the differences between rows are the cost of
the format alone, not of seed or run-to-run variation.

Latency is model inference only. Filtering and beat segmentation run once per
record upstream and are not timed.

    python scripts/benchmark.py                   # seeds 12 13 14
    python scripts/benchmark.py --seeds 12        # quicker

Needs the deploy extra: pip install -e ".[keras,deploy]"
"""

from __future__ import annotations

import os

# Benchmark on CPU, which is the target. This also makes training repeatable
# across processes; on a GPU the same seed gives slightly different weights.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import argparse  # noqa: E402
import csv  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecg import export  # noqa: E402
from ecg.config import CLASSES, DS1, DS2, Config, seed_everything  # noqa: E402
from ecg.evaluate import compute_metrics  # noqa: E402
from ecg.models import build  # noqa: E402
from ecg.segmentation import segment_records, to_features  # noqa: E402
from ecg.train import (  # noqa: E402
    evaluate_patient_specific,
    split_by_minute,
    train_global,
)

FORMATS = ("keras_fp32", "onnx_fp32", "onnx_int8_static", "onnx_int8_dynamic")


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _export_all(model, calibration: np.ndarray, workdir: Path) -> dict[str, Path]:
    """FP32 ONNX plus both INT8 variants of one model."""
    fp32 = export.to_onnx(model, workdir / "fp32.onnx")
    return {
        "onnx_fp32": fp32,
        "onnx_int8_static": export.quantize(
            fp32, workdir / "int8_static.onnx", "static", calibration),
        "onnx_int8_dynamic": export.quantize(
            fp32, workdir / "int8_dynamic.onnx", "dynamic"),
    }


def score_seed(df1, df2, cfg: Config, seed: int):
    """Train, personalise and score every format on one seed.

    Returns ``(metrics_by_format, agreement_by_format, one_patient)`` where
    ``one_patient`` is ``(model, calibration_beats, test_beats)`` for timing.
    """
    seed_everything(seed)
    model, _ = train_global(df1, build("mlp", cfg=cfg), cfg)
    results, keras_metrics = evaluate_patient_specific(
        df2, model, DS2, cfg, keep_models=True)

    preds = {name: [] for name in FORMATS[1:]}
    for r in results:
        train_df, test_df = split_by_minute(df2, r.patient, cfg)
        X_train, _ = to_features(train_df, cfg=cfg)
        X_test, y_test = to_features(test_df, cfg=cfg)
        assert np.array_equal(y_test, r.y_true), "test split drifted"

        with tempfile.TemporaryDirectory() as tmp:
            for name, path in _export_all(r.model, X_train, Path(tmp)).items():
                probs = export.predict(export.session(path), X_test)
                preds[name].append(np.argmax(probs, axis=1))

    y_true = np.concatenate([r.y_true for r in results])
    y_keras = np.concatenate([r.y_pred for r in results])
    metrics = {"keras_fp32": keras_metrics}
    agreement = {"keras_fp32": 1.0}
    for name, parts in preds.items():
        y_pred = np.concatenate(parts)
        metrics[name] = compute_metrics(y_true, y_pred)
        agreement[name] = float(np.mean(y_pred == y_keras))

    first = results[0]
    train_df, test_df = split_by_minute(df2, first.patient, cfg)
    one_patient = (first.model, to_features(train_df, cfg=cfg)[0],
                   to_features(test_df, cfg=cfg)[0])
    return metrics, agreement, one_patient


def _time_calls(fn, x: np.ndarray, warmup: int, iters: int) -> np.ndarray:
    """Per-call wall time in microseconds."""
    for _ in range(warmup):
        fn(x)
    times = np.empty(iters)
    for i in range(iters):
        t0 = time.perf_counter_ns()
        fn(x)
        times[i] = time.perf_counter_ns() - t0
    return times / 1e3


def measure_latency(model, calibration, test_beats, args) -> list[dict]:
    """Size and latency of each format for one patient's model."""
    import keras
    import onnxruntime as ort

    beat = np.asarray(test_beats[:1], dtype=np.float32)
    batch = np.asarray(
        np.resize(test_beats, (args.batch, test_beats.shape[1])), dtype=np.float32)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        keras_path = tmp / "model.keras"
        model.save(keras_path)
        paths = {"keras_fp32": keras_path,
                 **_export_all(model, calibration, tmp)}

        for name, path in paths.items():
            if name == "keras_fp32":
                def run(x):
                    return model(x, training=False)
                threads = "tf default"
            else:
                sess = export.session(path, threads=args.threads)
                def run(x, sess=sess):
                    return export.predict(sess, x)
                threads = str(args.threads)

            single = _time_calls(run, beat, args.warmup, args.iters)
            many = _time_calls(run, batch, args.warmup // 10 or 1,
                               max(args.iters // 50, 20))
            rows.append({
                "format": name,
                "size_kb": round(path.stat().st_size / 1024, 1),
                "batch1_median_us": round(float(np.median(single)), 1),
                "batch1_p99_us": round(float(np.percentile(single, 99)), 1),
                f"batch{args.batch}_beats_per_s": int(
                    args.batch / (np.median(many) / 1e6)),
                "threads": threads,
                "cpu": _cpu_name(),
                "os": platform.platform(terse=True),
                "runtime": (f"onnxruntime {ort.__version__}"
                            if name != "keras_fp32"
                            else f"keras {keras.__version__}"),
            })
            print(f"  {name:18s} {rows[-1]['size_kb']:7.1f} KB  "
                  f"median {rows[-1]['batch1_median_us']:8.1f} us  "
                  f"p99 {rows[-1]['batch1_p99_us']:8.1f} us", flush=True)
    return rows


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[12, 13, 14])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--threads", type=int, default=1,
                   help="ONNX Runtime threads for the latency runs")
    p.add_argument("--iters", type=int, default=5000,
                   help="timed single-beat calls per format")
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--batch", type=int, default=256,
                   help="batch size for the throughput measurement")
    p.add_argument("--interim-dir", type=Path, default=Path("data/interim"))
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    args = p.parse_args()

    cfg = Config(interim_dir=args.interim_dir, max_epochs=args.epochs)

    print("segmenting DS1 and DS2 once ...")
    try:
        df1 = segment_records(DS1, cfg)
        df2 = segment_records(DS2, cfg)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    per_seed = {name: [] for name in FORMATS}
    agree = {name: [] for name in FORMATS}
    one_patient = None
    for seed in args.seeds:
        metrics, agreement, sample = score_seed(df1, df2, cfg, seed)
        one_patient = one_patient or sample
        for name in FORMATS:
            per_seed[name].append(metrics[name])
            agree[name].append(agreement[name])
            print(f"  seed {seed} {name:18s} {metrics[name].summary()}  "
                  f"agrees with keras on {agreement[name]:.4f}", flush=True)

    rows = []
    for name in FORMATS:
        ms = per_seed[name]
        mean = lambda fn, ms=ms: float(np.mean([fn(m) for m in ms]))  # noqa: E731
        std = lambda fn, ms=ms: float(np.std([fn(m) for m in ms]))    # noqa: E731
        row = {
            "format": name, "seeds": len(args.seeds),
            "train_minutes": cfg.train_minutes,
            "accuracy": round(mean(lambda m: m.micro["Accuracy"]), 4),
            "macro_f1": round(mean(lambda m: m.macro["F1 Score"]), 4),
            "macro_f1_std": round(std(lambda m: m.macro["F1 Score"]), 4),
            "cohen_kappa": round(mean(lambda m: m.cohen_kappa), 4),
        }
        for i, cls in enumerate(CLASSES):
            row[f"f1_{cls}"] = round(
                mean(lambda m, i=i: m.per_class["F1 Score"][i]), 3)
        row["agreement_with_keras"] = round(float(np.mean(agree[name])), 4)
        rows.append(row)
    _write(args.out_dir / "deployment_metrics.csv", rows)

    print(f"\nlatency on one patient's model, {_cpu_name()} ...")
    _write(args.out_dir / "deployment_latency.csv",
           measure_latency(*one_patient, args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
