"""Metrics and confusion matrices.

Reimplements the ``metrics_from_confusion_matrix`` with the same
definitions (micro / macro / weighted / per-class from a one-vs-rest breakdown
of the confusion matrix) but without the 35-line duplicated early-return block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, matthews_corrcoef

from .config import CLASSES

_METRICS = ("Accuracy", "Sensitivity", "Specificity", "Precision", "F1 Score")


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return float(num / den) if den else default


def _f1(precision: float, sensitivity: float) -> float:
    return _safe_div(2 * precision * sensitivity, precision + sensitivity)


@dataclass
class Metrics:
    """Classification metrics at micro, macro, weighted and per-class level."""

    micro: dict[str, float]
    macro: dict[str, float]
    weighted: dict[str, float]
    per_class: dict[str, list[float]]
    confusion: np.ndarray
    labels: tuple[str, ...] = CLASSES
    cohen_kappa: float = 0.0
    matthews_corr: float = 0.0

    def to_dict(self) -> dict:
        return {
            "labels": list(self.labels),
            "micro": self.micro,
            "macro": self.macro,
            "weighted": self.weighted,
            "per_class": {
                name: {m: self.per_class[m][i] for m in _METRICS}
                for i, name in enumerate(self.labels)
            },
            "cohen_kappa": self.cohen_kappa,
            "matthews_corr": self.matthews_corr,
            "confusion_matrix": self.confusion.tolist(),
        }

    def summary(self) -> str:
        """One-line summary: the numbers usually quoted."""
        return (
            f"accuracy={self.micro['Accuracy']:.3f} "
            f"macro-F1={self.macro['F1 Score']:.3f} "
            f"kappa={self.cohen_kappa:.3f}"
        )


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: tuple[str, ...] = CLASSES
) -> Metrics:
    """Compute all metrics from integer class indices.

    ``y_true``/``y_pred`` index into ``labels``, whose order is pinned in config
    rather than inferred from the data.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    indices = list(range(len(labels)))

    cm = confusion_matrix(y_true, y_pred, labels=indices)
    total = cm.sum()

    # One-vs-rest counts per class.
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = total - tp - fp - fn
    support = cm.sum(axis=1).astype(float)

    per_class = {
        "Accuracy": [_safe_div(tp[i] + tn[i], total) for i in indices],
        "Sensitivity": [_safe_div(tp[i], tp[i] + fn[i]) for i in indices],
        "Specificity": [_safe_div(tn[i], tn[i] + fp[i]) for i in indices],
        "Precision": [_safe_div(tp[i], tp[i] + fp[i]) for i in indices],
    }
    per_class["F1 Score"] = [
        _f1(per_class["Precision"][i], per_class["Sensitivity"][i]) for i in indices
    ]

    # Micro: pool counts across classes first, then compute.
    g_tp, g_fp, g_fn, g_tn = tp.sum(), fp.sum(), fn.sum(), tn.sum()
    micro_precision = _safe_div(g_tp, g_tp + g_fp, 1.0)
    micro_sensitivity = _safe_div(g_tp, g_tp + g_fn, 1.0)
    micro = {
        "Accuracy": _safe_div(g_tp, total, 1.0),
        "Sensitivity": micro_sensitivity,
        "Specificity": _safe_div(g_tn, g_tn + g_fp, 1.0),
        "Precision": micro_precision,
        "F1 Score": _f1(micro_precision, micro_sensitivity),
    }

    macro = {m: float(np.mean(per_class[m])) for m in _METRICS}
    weights = support / total if total else np.zeros_like(support)
    weighted = {m: float(np.dot(per_class[m], weights)) for m in _METRICS}

    return Metrics(
        micro=micro,
        macro=macro,
        weighted=weighted,
        per_class=per_class,
        confusion=cm,
        labels=labels,
        cohen_kappa=float(cohen_kappa_score(y_true, y_pred, labels=indices)),
        matthews_corr=float(matthews_corrcoef(y_true, y_pred)),
    )


def metrics_table(metrics: Metrics) -> str:
    """Render metrics as a text table in the standard AAMI layout."""
    from tabulate import tabulate

    header = ["Metric", "Micro", "Macro", "Weighted"] + [
        f"Class {n}" for n in metrics.labels
    ]
    rows = [
        [
            m,
            f"{metrics.micro[m]:.3f}",
            f"{metrics.macro[m]:.3f}",
            f"{metrics.weighted[m]:.3f}",
        ]
        + [f"{v:.3f}" for v in metrics.per_class[m]]
        for m in _METRICS
    ]
    pad = ["N/A"] * (len(metrics.labels) + 2)
    rows.append(["Cohen Kappa", f"{metrics.cohen_kappa:.3f}"] + pad)
    rows.append(["Matthews Corr", f"{metrics.matthews_corr:.3f}"] + pad)
    return tabulate(rows, headers=header, tablefmt="grid")
