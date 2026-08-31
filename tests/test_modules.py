"""Structural tests for models, metrics, balancing and the fine-tuning loop.

No real training happens here: the only ``fit`` call runs one epoch on 20
synthetic rows, purely to verify that fine-tuning copies the model rather than
mutating it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecg.balance import class_counts, imbalance_report, smote
from ecg.config import CLASSES, Config
from ecg.evaluate import compute_metrics, metrics_table
from ecg.models import build, mlp

FIXTURES = Path(__file__).parent / "fixtures"


# --- step 6: models ----------------------------------------------------------

def test_mlp_matches_shipped_checkpoint():
    """151-128-64-4 with no dropout or regularisation: 27,972 parameters."""
    model = mlp()
    assert model.count_params() == 27972
    assert model.input_shape == (None, 151)
    assert model.output_shape == (None, 4)


@pytest.mark.parametrize("extra,expected_input", [(0, 151), (1, 152), (2, 153)])
def test_mlp_accepts_extra_features(extra, expected_input):
    assert mlp(input_size=151 + extra).input_shape == (None, expected_input)


def test_unknown_architecture_raises():
    with pytest.raises(ValueError, match="unknown architecture"):
        build("transformer")


# --- step 7: metrics ---------------------------------------------------------

def test_metrics_match_sklearn():
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

    d = np.load(FIXTURES / "predictions_ds2.npz")
    y_true, y_pred = d["ground_truth"], d["predictions"]
    m = compute_metrics(y_true, y_pred)

    assert m.micro["Accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert m.macro["F1 Score"] == pytest.approx(
        f1_score(y_true, y_pred, average="macro")
    )
    assert m.cohen_kappa == pytest.approx(cohen_kappa_score(y_true, y_pred))


def test_metrics_perfect_prediction():
    y = np.array([0, 1, 2, 3, 1, 2])
    m = compute_metrics(y, y)
    assert m.micro["Accuracy"] == 1.0
    assert m.macro["F1 Score"] == 1.0
    assert m.cohen_kappa == 1.0


def test_metrics_handle_absent_class():
    """A class with no support must not produce NaN."""
    y_true = np.array([1, 1, 1, 3])
    y_pred = np.array([1, 1, 3, 3])
    m = compute_metrics(y_true, y_pred)
    assert np.isfinite(list(m.macro.values())).all()
    assert m.confusion.shape == (4, 4)


def test_metrics_serialise_and_render():
    y_true = np.array([0, 1, 2, 3] * 5)
    y_pred = np.array([0, 1, 2, 2] * 5)
    m = compute_metrics(y_true, y_pred)
    assert json.loads(json.dumps(m.to_dict()))["labels"] == list(CLASSES)
    assert "Cohen Kappa" in metrics_table(m)


# --- step 5: balancing -------------------------------------------------------

@pytest.fixture
def toy_beats():
    rng = np.random.default_rng(0)
    labels = ["N"] * 40 + ["V"] * 8 + ["S"] * 4 + ["F"] * 2
    return pd.DataFrame({
        "Label": labels,
        "Patient": 100,
        "Sample": np.arange(len(labels)) * 400 + 500,
        "beat": [rng.normal(size=151) for _ in labels],
    })


def test_imbalance_report_shares_sum_to_one(toy_beats):
    assert imbalance_report(toy_beats)["share"].sum() == pytest.approx(1.0, abs=1e-3)


def test_smote_balances_and_is_seeded(toy_beats):
    cfg = Config()
    first = smote(toy_beats, cfg)
    second = smote(toy_beats, cfg)

    counts = class_counts(first)
    assert counts.nunique() == 1, "SMOTE should equalise class counts"
    assert len(first) == len(second)
    np.testing.assert_allclose(
        np.stack(first["beat"].to_numpy()), np.stack(second["beat"].to_numpy())
    )


def test_smote_beats_keep_window_length(toy_beats):
    out = smote(toy_beats, Config())
    assert np.stack(out["beat"].to_numpy()).shape[1] == 151


# --- step 8: fine-tuning contract -------------------------------------------

def test_fine_tune_does_not_mutate_the_global_model():
    """The original bound `tuned_model = model`, so each patient's fine-tuning
    continued from the previous patient's weights."""
    from ecg.train import fine_tune

    global_model = mlp()
    before = [w.copy() for w in global_model.get_weights()]

    X = np.random.default_rng(0).normal(size=(20, 151))
    y = np.eye(4)[np.array([0, 1, 2, 3] * 5)]
    tuned, losses = fine_tune(global_model, X, y, Config(), max_epochs=1)

    assert len(losses) == 1
    for w_before, w_after in zip(before, global_model.get_weights()):
        np.testing.assert_array_equal(
            w_before, w_after, err_msg="global model was mutated by fine_tune"
        )
    assert tuned is not global_model


def test_split_by_minute_is_temporally_disjoint():
    from ecg.train import split_by_minute

    cfg = Config(train_minutes=5, test_minute=5)
    df = pd.DataFrame({
        "Patient": [100] * 10 + [200] * 5,
        "Sample": list(np.linspace(0, 640000, 10).astype(int)) + [0] * 5,
    })
    train, test = split_by_minute(df, 100, cfg)

    assert (train["Patient"] == 100).all() and (test["Patient"] == 100).all()
    assert train["Sample"].max() < cfg.minute_to_sample(5)
    assert test["Sample"].min() >= cfg.minute_to_sample(5)
    assert set(train.index).isdisjoint(test.index)
