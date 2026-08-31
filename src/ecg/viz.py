"""Plotting.

Replaces four near-identical 60-line ``signal_visualization_by_*`` functions
from the original with one filterable plot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CLASSES, DEFAULT, Config


def plot_beats(
    df: pd.DataFrame,
    patient: int | None = None,
    label: str | None = None,
    n: int = 10,
    ax=None,
    cfg: Config = DEFAULT,
):
    """Overlay up to ``n`` beat windows, optionally filtered by patient/class."""
    import matplotlib.pyplot as plt

    subset = df
    if patient is not None:
        subset = subset[subset["Patient"] == patient]
    if label is not None:
        subset = subset[subset["Label"] == label]
    subset = subset.head(n)

    ax = ax or plt.subplots(figsize=(8, 4))[1]
    time_ms = np.arange(cfg.window_length) * 1000 / cfg.sampling_rate
    for _, row in subset.iterrows():
        ax.plot(time_ms, row["beat"], alpha=0.6, linewidth=0.9)

    ax.axvline(
        cfg.window_left * 1000 / cfg.sampling_rate,
        color="k", linestyle="--", linewidth=0.8, label="R-peak",
    )
    title = "Beats" + (f" - patient {patient}" if patient else "")
    ax.set(xlabel="Time (ms)", ylabel="Amplitude (mV)",
           title=title + (f" - class {label}" if label else ""))
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_class_balance(df: pd.DataFrame, ax=None):
    """Bar chart of beats per class, on a log scale."""
    import matplotlib.pyplot as plt

    counts = df["Label"].value_counts().reindex(CLASSES).fillna(0)
    ax = ax or plt.subplots(figsize=(5, 3.5))[1]
    ax.bar([str(c) for c in counts.index], counts.to_numpy())
    ax.set(ylabel="Beats (log scale)", yscale="log", title="Class balance")
    for i, v in enumerate(counts.to_numpy()):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9)
    return ax


def plot_confusion(metrics, ax=None, normalize: bool = True):
    """Confusion matrix heatmap."""
    import matplotlib.pyplot as plt

    cm = metrics.confusion.astype(float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    ax = ax or plt.subplots(figsize=(5, 4.5))[1]
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() or 1)
    ticks = range(len(metrics.labels))
    ax.set(
        xticks=ticks, yticks=ticks,
        xticklabels=metrics.labels, yticklabels=metrics.labels,
        xlabel="Predicted", ylabel="True", title="Confusion matrix",
    )
    for i in ticks:
        for j in ticks:
            ax.text(
                j, i, f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}",
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=9,
            )
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    return ax
