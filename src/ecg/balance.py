"""Class balancing.

DS2 is 89% class N and 0.8% class F, so every training run needs some form of
rebalancing. Three strategies are available; all of them are seeded, which the
original beat-shifting augmentation was not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT, Config


def balanced_class_weights(targets: np.ndarray, n_classes: int = 4) -> dict[int, float]:
    """Inverse-frequency weights, as used by sklearn's "balanced" mode.

    DS1 is 90% class N and 0.8% class F. Without this the global model can
    minimise its loss by ignoring the rare classes entirely.
    """
    counts = np.bincount(np.asarray(targets), minlength=n_classes).astype(float)
    weights = len(targets) / (n_classes * np.maximum(counts, 1.0))
    return {i: float(weights[i]) for i in range(n_classes)}


def class_counts(df: pd.DataFrame, column: str = "Label") -> pd.Series:
    """Beats per class, most frequent first."""
    return df[column].value_counts()


def imbalance_report(df: pd.DataFrame, column: str = "Label") -> pd.DataFrame:
    """Per-class counts and shares."""
    counts = class_counts(df, column)
    return pd.DataFrame(
        {"count": counts, "share": (counts / counts.sum()).round(4)}
    )


def undersample_majority(
    df: pd.DataFrame, n_samples: int, cfg: Config = DEFAULT
) -> pd.DataFrame:
    """Cut the majority class down to ``n_samples``, leave the rest untouched."""
    majority = class_counts(df).idxmax()
    kept = df[df["Label"] == majority].sample(
        n=min(n_samples, (df["Label"] == majority).sum()), random_state=cfg.seed
    )
    return pd.concat([kept, df[df["Label"] != majority]], ignore_index=True)


def random_oversample(
    df: pd.DataFrame, n_samples: int | None = None, cfg: Config = DEFAULT
) -> pd.DataFrame:
    """Undersample the majority, then duplicate minority beats to match it."""
    from imblearn.over_sampling import RandomOverSampler

    if n_samples is not None:
        df = undersample_majority(df, n_samples, cfg)

    sampler = RandomOverSampler(random_state=cfg.seed)
    features, labels = sampler.fit_resample(
        df.drop(columns=["Label"]), df["Label"]
    )
    return pd.concat([features, labels], axis=1)


def smote(df: pd.DataFrame, cfg: Config = DEFAULT) -> pd.DataFrame:
    """SMOTE over the beat windows.

    Interpolates between neighbouring minority beats. Classes with a single
    member are dropped first -- SMOTE needs at least two points to interpolate.
    """
    from imblearn.over_sampling import SMOTE

    singletons = class_counts(df)[lambda s: s == 1].index
    df = df[~df["Label"].isin(singletons)]
    if df.empty:
        raise ValueError("no class has more than one beat; nothing to interpolate")

    # SMOTE interpolates towards k nearest neighbours of the same class, so k
    # must be smaller than the rarest class. Class F is under 1% of DS2 and a
    # single patient can easily have fewer than the default k=5, which would
    # otherwise raise deep inside sklearn.
    k_neighbors = min(5, int(class_counts(df).min()) - 1)
    if k_neighbors < 1:
        raise ValueError("rarest class has too few beats for SMOTE")

    beats = np.stack(df["beat"].to_numpy())
    resampled, labels = SMOTE(
        random_state=cfg.seed, k_neighbors=k_neighbors
    ).fit_resample(beats, df["Label"])

    out = pd.DataFrame({"Label": labels})
    out["beat"] = list(np.asarray(resampled))
    # Flag which rows are synthetic; the originals come first.
    out["synthetic"] = np.arange(len(out)) >= len(df)
    return out


def shift_augment(
    df: pd.DataFrame,
    samples_per_class: int,
    signal_lookup: dict[int, np.ndarray],
    cfg: Config = DEFAULT,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Augment minority classes by re-cutting each beat window off-centre.

    Picks a random side and a random shift of up to half that side's window,
    then re-extracts the window from the original signal at the shifted centre.

    Unlike the original, this takes an explicit ``rng``. The thesis version used
    unseeded module-level ``random``/``np.random`` calls, so the augmented set
    -- and therefore every model trained on it -- differed between runs.
    """
    rng = rng or np.random.default_rng(cfg.seed)

    majority = class_counts(df).idxmax()
    kept_majority = df[df["Label"] == majority].sample(
        n=min(samples_per_class, (df["Label"] == majority).sum()),
        random_state=cfg.seed,
    )
    base = pd.concat(
        [df[df["Label"] != majority], kept_majority], ignore_index=True
    )

    picks = []
    for label in base.loc[base["Label"] != majority, "Label"].unique():
        pool = base.index[base["Label"] == label]
        shortfall = samples_per_class - len(pool)
        if shortfall > 0:
            picks.extend(rng.choice(pool, shortfall, replace=True))

    if not picks:
        return base

    augmented = base.loc[picks].reset_index(drop=True)
    beats, samples = [], []
    for _, row in augmented.iterrows():
        signal = signal_lookup[row["Patient"]]
        centre = int(row["Sample"])

        # Room available on each side before running off the record.
        room_left = centre - cfg.window_left
        room_right = len(signal) - (centre + cfg.window_right + 1)
        to_right = bool(rng.integers(2))

        limit = (
            min(room_right, cfg.window_right // 2)
            if to_right
            else min(room_left, cfg.window_left // 2)
        )
        shift = int(rng.integers(1, limit)) if limit > 1 else 0
        centre = centre + shift if to_right else centre - shift

        beats.append(signal[centre - cfg.window_left : centre + cfg.window_right + 1])
        samples.append(centre)

    augmented["beat"] = beats
    augmented["Sample"] = samples
    return pd.concat([base, augmented], ignore_index=True)


STRATEGIES = {
    "none": lambda df, cfg=DEFAULT, **kw: df,
    "oversample": lambda df, cfg=DEFAULT, n_samples=None, **kw: random_oversample(
        df, n_samples, cfg
    ),
    "smote": lambda df, cfg=DEFAULT, **kw: smote(df, cfg),
}


def rebalance(df: pd.DataFrame, strategy: str, cfg: Config = DEFAULT, **kwargs):
    """Apply a named balancing strategy."""
    try:
        return STRATEGIES[strategy](df, cfg=cfg, **kwargs)
    except KeyError:
        raise ValueError(
            f"unknown strategy {strategy!r}; expected one of "
            f"{sorted(STRATEGIES)} or 'shift' via shift_augment()"
        ) from None
