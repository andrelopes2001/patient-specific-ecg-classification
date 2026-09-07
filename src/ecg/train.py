"""Global training and patient-specific fine-tuning.

The patient-specific protocol, from ``retrain_split`` in the original:

  1. Train a global model on DS1 -- 22 records, none of which appear in DS2.
  2. For each DS2 patient, fine-tune a *copy* of that global model on that
     patient's own first ``train_minutes`` of recording.
  3. Test on the same patient's beats from ``test_minute`` to the end.

Train and test are temporally disjoint within a patient, and the global model
never saw any DS2 patient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .balance import balanced_class_weights
from .config import CLASSES, DEFAULT, DS1, DS2, Config
from .evaluate import Metrics, compute_metrics
from .preprocessing import fit_scaler
from .segmentation import one_hot, to_features


def split_by_minute(
    df: pd.DataFrame, patient: int, cfg: Config = DEFAULT
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one patient's beats into fine-tuning and test portions by time."""
    own = df[df["Patient"] == patient]
    train_end = cfg.minute_to_sample(cfg.train_minutes)
    test_start = cfg.minute_to_sample(cfg.test_minute)
    return own[own["Sample"] < train_end], own[own["Sample"] >= test_start]


def _clone_compiled(model):
    """Return an independent copy of a compiled Keras model.

    The original ``tuned_model_training`` did ``tuned_model = model``, which
    binds a reference rather than copying. Every patient's fine-tuning therefore
    started from the previous patient's weights instead of from the global
    model, and the loop's results depended on patient ordering. Cloning fixes
    that.
    """
    import keras

    clone = keras.models.clone_model(model)
    clone.set_weights(model.get_weights())
    clone.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return clone


def fine_tune(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Config = DEFAULT,
    max_epochs: int | None = None,
    loss_threshold: float | None = None,
):
    """Fine-tune a copy of ``model``; stop early once the loss is small enough.

    Returns ``(tuned_model, history)``. ``model`` itself is left untouched.
    """
    max_epochs = cfg.finetune_max_epochs if max_epochs is None else max_epochs
    loss_threshold = (cfg.finetune_loss_threshold if loss_threshold is None
                      else loss_threshold)
    tuned = _clone_compiled(model)
    losses = []
    for _ in range(max_epochs):
        history = tuned.fit(
            X_train, y_train, epochs=1, batch_size=cfg.batch_size, verbose=0
        )
        losses.append(history.history["loss"][0])
        if losses[-1] < loss_threshold:
            break
    return tuned, losses


@dataclass
class PatientResult:
    patient: int
    n_train: int
    n_test: int
    y_true: np.ndarray
    y_pred: np.ndarray


def evaluate_patient_specific(
    df: pd.DataFrame,
    global_model,
    patients=DS2,
    cfg: Config = DEFAULT,
    extra_features: tuple[str, ...] = (),
    normalize: bool = False,
) -> tuple[list[PatientResult], Metrics]:
    """Run the full patient-specific protocol and pool the predictions.

    Metrics are computed over all patients' test beats pooled into one
    confusion matrix, which is the AAMI reporting convention.
    """
    results: list[PatientResult] = []

    for patient in patients:
        train_df, test_df = split_by_minute(df, patient, cfg)
        if train_df.empty or test_df.empty:
            continue

        X_train, y_train = to_features(train_df, extra_features, cfg)
        X_test, y_test = to_features(test_df, extra_features, cfg)

        if normalize:
            # Fit on the patient's fine-tuning window only -- never on test.
            scaler = fit_scaler(X_train, "MinMax")
            X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

        tuned, _ = fine_tune(global_model, X_train, one_hot(y_train), cfg)
        y_pred = np.argmax(tuned.predict(X_test, verbose=0), axis=1)

        results.append(
            PatientResult(patient, len(train_df), len(test_df), y_test, y_pred)
        )

    if not results:
        raise ValueError("no patient produced both a train and a test split")

    pooled_true = np.concatenate([r.y_true for r in results])
    pooled_pred = np.concatenate([r.y_pred for r in results])
    return results, compute_metrics(pooled_true, pooled_pred)


def train_global(
    df: pd.DataFrame,
    model,
    cfg: Config = DEFAULT,
    epochs: int | None = None,
    extra_features: tuple[str, ...] = (),
    balance: str = "none",
    balance_kwargs: dict | None = None,
):
    """Train the global model on DS1.

    Defaults: a fixed 10 epochs, no class weighting.

    ``cfg.class_weight`` and ``cfg.early_stopping_patience`` enable
    inverse-frequency weighting and early stopping. Both raised macro F1 on
    held-out DS1 patients and then *lowered* it on DS2 by 0.024, outside the
    seed spread -- see docs/experiments.md. They are kept as options because
    the negative result is worth being able to reproduce, not because they help.
    """
    import keras

    from .balance import rebalance

    epochs = cfg.max_epochs if epochs is None else epochs
    train_df = df[df["Patient"].isin(DS1)]

    if balance != "none":
        if extra_features:
            raise ValueError(
                "balancing operates on beat windows only; it cannot be combined "
                "with extra scalar features"
            )
        # Balancing applies to the global training set only. The per-patient
        # personalisation split is never resampled: it is what the deployed
        # system would actually see.
        train_df = rebalance(train_df, balance, cfg, **(balance_kwargs or {}))

    X, y = to_features(train_df, extra_features, cfg)

    callbacks = []
    if cfg.early_stopping_patience:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="loss",
                patience=cfg.early_stopping_patience,
                restore_best_weights=True,
            )
        )
    history = model.fit(
        X,
        one_hot(y),
        epochs=epochs,
        batch_size=cfg.batch_size,
        verbose=0,
        callbacks=callbacks,
        class_weight=balanced_class_weights(y, cfg.n_classes) if cfg.class_weight else None,
    )
    return model, history
