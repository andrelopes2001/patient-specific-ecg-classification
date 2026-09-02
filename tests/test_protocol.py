"""Guards on the evaluation protocol.

These are the assertions that matter most for trusting the reported numbers:
that no beat is ever used for both personalisation and testing, that the two
record sets share no patients, and that every test beat is counted exactly once.

They ran as ad-hoc checks while validating the results; they belong here so
they run on every change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ecg.config import CLASSES, DS1, DS2, Config
from ecg.segmentation import segment_records
from ecg.train import evaluate_patient_specific, split_by_minute

pytestmark = pytest.mark.needs_data

# Two DS2 records are enough to exercise the protocol without a full run.
SAMPLE_RECORDS = (213, 100)


def _cfg(**kw) -> Config:
    return Config(interim_dir=Path("data/interim"), **kw)


# --- split integrity, no training required -----------------------------------

def test_ds1_and_ds2_share_no_patients():
    """The global model must never see a patient it is later evaluated on."""
    assert set(DS1).isdisjoint(DS2)
    assert len(DS1) == len(DS2) == 22


def test_paced_records_excluded():
    """AAMI excludes the paced records; they must be in neither set."""
    for record in (102, 104, 107, 217):
        assert record not in DS1 and record not in DS2


@pytest.mark.parametrize("record", SAMPLE_RECORDS)
def test_no_beat_is_both_finetuned_and_tested(record):
    """The core leakage guard, on real segmented beats."""
    cfg = _cfg()
    df = segment_records([record], cfg)
    train, test = split_by_minute(df, record, cfg)

    assert len(train) > 0 and len(test) > 0
    key = lambda d: set(map(tuple, d[["Patient", "Sample"]].to_numpy()))  # noqa: E731
    assert key(train).isdisjoint(key(test))
    # and they are genuinely ordered in time, not merely disjoint
    assert train["Sample"].max() < cfg.minute_to_sample(cfg.train_minutes)
    assert test["Sample"].min() >= cfg.minute_to_sample(cfg.test_minute)


def test_split_only_returns_the_requested_patient():
    cfg = _cfg()
    df = segment_records(list(SAMPLE_RECORDS), cfg)
    train, test = split_by_minute(df, SAMPLE_RECORDS[0], cfg)
    for part in (train, test):
        assert set(part["Patient"]) == {SAMPLE_RECORDS[0]}


# --- accounting, needs a short fine-tuning run -------------------------------

def test_every_test_beat_counted_exactly_once():
    """The pooled confusion matrix must total the number of test beats.

    Catches double-counting and silently dropped patients -- both of which
    would move the headline metric without raising anything.
    """
    from ecg.models import mlp

    cfg = _cfg()
    df = segment_records(list(SAMPLE_RECORDS), cfg)
    expected = sum(len(split_by_minute(df, r, cfg)[1]) for r in SAMPLE_RECORDS)

    results, metrics = evaluate_patient_specific(
        df, mlp(cfg=cfg), SAMPLE_RECORDS, cfg
    )
    assert metrics.confusion.sum() == expected
    assert sum(r.n_test for r in results) == expected
    assert len(results) == len(SAMPLE_RECORDS)


def test_predictions_stay_within_the_pinned_class_set():
    from ecg.models import mlp

    cfg = _cfg()
    df = segment_records([SAMPLE_RECORDS[0]], cfg)
    results, _ = evaluate_patient_specific(
        df, mlp(cfg=cfg), [SAMPLE_RECORDS[0]], cfg
    )
    for r in results:
        assert set(np.unique(r.y_pred)) <= set(range(len(CLASSES)))
        assert set(np.unique(r.y_true)) <= set(range(len(CLASSES)))
