"""Beat segmentation and AAMI labelling.

Beats are fixed-length windows centred on the annotated R-peak, which is what
lets a plain MLP consume them. Beats too close to a neighbour to fill the window
are dropped rather than padded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AAMI_GROUPS, CLASSES, DEFAULT, Config
from .data import Record, load_record

# Symbols that denote an actual heartbeat. Everything else in the annotation
# stream (rhythm changes '+', signal quality, etc.) is not a beat.
BEAT_SYMBOLS: frozenset[str] = frozenset(
    sym for group in AAMI_GROUPS.values() for sym in group
)

# Label modes: multiclass4 drops the paced/unknown Q class entirely.
_LABEL_MODES = {
    "multiclass4": ("F", "N", "S", "V"),
    "multiclass5": ("F", "N", "S", "V", "Q"),
}


def _symbol_to_aami() -> dict[str, str]:
    return {sym: aami for aami, group in AAMI_GROUPS.items() for sym in group}


def aami_label(labels: pd.Series, mode: str = "multiclass4") -> pd.Series:
    """Map WFDB annotation symbols onto AAMI classes.

    Uses an explicit ``.map`` rather than the original chained ``.loc`` writes.
    Under pandas >= 3 copy-on-write those chained assignments can become silent
    no-ops -- they would not raise, the labels would simply not be remapped.
    """
    if mode in _LABEL_MODES:
        return labels.map(_symbol_to_aami())
    if mode in CLASSES:
        # Binary one-vs-rest mode, e.g. mode="V" gives {"V", "notV"}.
        mapped = labels.map(_symbol_to_aami())
        return mapped.where(mapped == mode, f"not{mode}")
    raise ValueError(
        f"unknown label_mode {mode!r}; expected one of "
        f"{sorted(_LABEL_MODES) + list(CLASSES)}"
    )


def beat_positions(record: Record, cfg: Config = DEFAULT) -> pd.DataFrame:
    """R-peak positions with the gap to the previous and next beat.

    The gaps determine whether a fixed-length window fits around each beat, and
    ``Heart rate`` derived from the left gap is used as an optional feature.
    """
    # Gaps are measured between consecutive *annotations*, not between beats.
    # The annotation stream also carries non-beat markers (rhythm changes '+',
    # signal quality), and those occupy sample positions. Measuring gaps to the
    # nearest beat instead would widen them, letting ~25 extra beats per record
    # through the window filter below and silently changing the dataset. This
    # matches the original thesis implementation.
    peaks = record.annotated.dropna(subset=["Label"]).reset_index(drop=True)

    samples = peaks["Sample"]
    # Distance to the previous annotation; the first measures from sample 0.
    length_left = samples.diff().fillna(samples.iloc[0]).astype(int) - 1
    # Distance to the next annotation; the last measures to the record end.
    length_right = (
        samples.shift(-1).fillna(record.n_samples).astype(int) - samples - 1
    )

    positions = pd.DataFrame(
        {
            "Patient": record.record_id,
            "Sample": samples.astype(int),
            "Label": peaks["Label"].to_numpy(),
            "Length left": length_left.to_numpy(),
            "Length right": length_right.to_numpy(),
            "Heart rate": 60.0 / (length_left / cfg.sampling_rate),
            "X1_type": record.leads[0],
            "X2_type": record.leads[1] if len(record.leads) > 1 else None,
        }
    )

    # Only now drop the non-beat annotations.
    return positions[positions["Label"].isin(BEAT_SYMBOLS)].reset_index(drop=True)


def segment_record(
    record: Record, cfg: Config = DEFAULT, lead: str | None = None
) -> pd.DataFrame:
    """Cut fixed-length windows around every usable beat in one record.

    Returns a frame with one row per beat: the windowed signal as ``beat``,
    plus position, label and rate metadata.
    """
    lead = lead or cfg.lead
    if lead not in record.leads:
        return pd.DataFrame()

    positions = beat_positions(record, cfg)
    # A beat is usable only if both neighbours are far enough away for the
    # window to sit entirely inside the record.
    usable = positions[
        (positions["Length left"] > cfg.window_left)
        & (positions["Length right"] > cfg.window_right)
    ].reset_index(drop=True)

    signal = record.signal[lead].to_numpy()
    beats = np.empty((len(usable), cfg.window_length), dtype=np.float64)
    for i, sample in enumerate(usable["Sample"].to_numpy()):
        beats[i] = signal[sample - cfg.window_left : sample + cfg.window_right + 1]

    out = usable.copy()
    out["beat"] = list(beats)
    out["Label"] = aami_label(out["Label"], cfg.label_mode)
    if cfg.label_mode in _LABEL_MODES:
        out = out[out["Label"].isin(_LABEL_MODES[cfg.label_mode])]
    return out.reset_index(drop=True)


def segment_records(record_ids, cfg: Config = DEFAULT) -> pd.DataFrame:
    """Segment several records into one frame."""
    frames = [
        segment_record(load_record(rid, cfg), cfg)
        for rid in record_ids
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def to_matrix(df: pd.DataFrame) -> np.ndarray:
    """Stack the ``beat`` column into an ``(n_beats, window_length)`` array."""
    return np.stack(df["beat"].to_numpy())


def to_features(
    df: pd.DataFrame, extra: tuple[str, ...] = (), cfg: Config = DEFAULT
) -> tuple[np.ndarray, np.ndarray]:
    """Build the model input matrix and integer targets.

    ``extra`` optionally prepends scalar features -- ``("Length left",)`` gives
    the 152-input variant, ``("Length left", "Heart rate")`` the 153-input one.

    Targets are integer indices into :data:`~ecg.config.CLASSES`, whose order is
    pinned in config rather than inferred from the data.
    """
    beats = to_matrix(df)
    if extra:
        scalars = df[list(extra)].to_numpy(dtype=np.float64)
        beats = np.hstack([scalars, beats])

    class_index = {name: i for i, name in enumerate(CLASSES)}
    targets = df["Label"].map(class_index).to_numpy()
    if np.isnan(targets.astype(float)).any():
        unknown = sorted(set(df["Label"]) - set(CLASSES))
        raise ValueError(f"labels outside CLASSES={CLASSES}: {unknown}")
    return beats, targets.astype(int)


def one_hot(targets: np.ndarray, n_classes: int = len(CLASSES)) -> np.ndarray:
    """Integer targets to one-hot, in the pinned class order."""
    out = np.zeros((len(targets), n_classes), dtype=np.float64)
    out[np.arange(len(targets)), targets] = 1.0
    return out
