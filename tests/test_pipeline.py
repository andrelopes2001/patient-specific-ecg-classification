"""Regression tests against fixtures captured from the original thesis code.

The fixtures in tests/fixtures/ were produced by calling the original
functions.py. They pin the two boundaries where a
refactor can silently change results: the filtered signal and the segmented
beat tensor.

Fixtures are checksums and summary statistics, not the arrays themselves: the
underlying signal is PhysioNet-derived and is not redistributed. An MD5 over the
full float64 buffer is a stricter check than an allclose comparison anyway --
any change at all fails it.

These tests need the data locally:
    python scripts/download_data.py && python scripts/build_csv.py --records 213
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ecg.config import CLASSES, Config, seed_everything
from ecg.data import load_record
from ecg.preprocessing import denoise
from ecg.segmentation import segment_record, to_features, to_matrix, one_hot

FIXTURES = Path(__file__).parent / "fixtures"
RECORD = 213
TOL = 1e-9


@pytest.fixture(scope="module")
def baseline():
    return json.loads((FIXTURES / "baseline.json").read_text())


def _md5(a: np.ndarray) -> str:
    return hashlib.md5(np.asarray(a, dtype=np.float64).tobytes()).hexdigest()


def _cfg(**kw) -> Config:
    return Config(interim_dir=Path("data/interim"), **kw)


# --- step 2: record loading --------------------------------------------------

def test_record_shape_and_leads(baseline):
    rec = load_record(RECORD, _cfg(denoise=None))
    assert rec.n_samples == baseline["preprocessing"]["full_signal_shape"][0]
    assert rec.leads[0] == baseline["preprocessing"]["lead"]


# --- step 3: filtering -------------------------------------------------------

@pytest.mark.parametrize("method", ["fir", "median"])
def test_denoise_matches_original(method, baseline):
    """Filtered output must match the original implementation bit for bit."""
    rec = load_record(RECORD, _cfg(denoise=None))
    got = denoise(rec.signal, _cfg(denoise=method), method=method)
    window = got["MLII"].to_numpy()[:5000]

    expected = baseline["preprocessing"][method]
    assert _md5(window) == expected["md5"]
    np.testing.assert_allclose(window[:5], expected["first5"], rtol=0, atol=TOL)
    np.testing.assert_allclose(window[-5:], expected["last5"], rtol=0, atol=TOL)


def test_raw_signal_matches(baseline):
    rec = load_record(RECORD, _cfg(denoise=None))
    window = rec.signal["MLII"].to_numpy()[:5000]
    assert _md5(window) == baseline["preprocessing"]["raw"]["md5"]


def test_wavelet_path_runs():
    """The original raised ValueError here under pandas 3 (read-only buffer)."""
    rec = load_record(RECORD, _cfg(denoise=None))
    out = denoise(rec.signal, _cfg(denoise="wavelet"), method="wavelet")
    assert out.shape == rec.signal.shape
    assert np.isfinite(out["MLII"].to_numpy()).all()


# --- step 4: segmentation ----------------------------------------------------

def test_segmentation_tensor_identical(baseline):
    """Shape, contents and MD5 of the beat tensor must be unchanged.

    This is the single most important assertion in the suite: a changed filter
    order or an off-by-one in windowing shows up here and nowhere else.
    """
    cfg = _cfg(denoise="fir")
    df = segment_record(load_record(RECORD, cfg), cfg)
    X = to_matrix(df)
    expected = baseline["segmentation"]

    assert X.shape == tuple(expected["tensor_shape"])
    assert _md5(X) == expected["X_md5"]
    assert X.mean() == pytest.approx(expected["X_mean"], abs=1e-9)
    np.testing.assert_allclose(
        X[0, :5], expected["first_beat_first5"], rtol=0, atol=TOL
    )
    np.testing.assert_allclose(
        X[-1, -5:], expected["last_beat_last5"], rtol=0, atol=TOL
    )


def test_segmentation_label_counts(baseline):
    cfg = _cfg(denoise="fir")
    df = segment_record(load_record(RECORD, cfg), cfg)
    counts = df["Label"].value_counts().to_dict()
    assert counts == baseline["segmentation"]["label_counts"]


def test_window_geometry():
    cfg = _cfg()
    assert cfg.window_length == 151
    assert cfg.window_ms == pytest.approx(419.4, abs=0.1)


def test_beat_windows_are_centred_on_r_peak():
    cfg = _cfg(denoise=None)
    rec = load_record(RECORD, cfg)
    df = segment_record(rec, cfg)
    signal = rec.signal["MLII"].to_numpy()
    for _, row in df.head(20).iterrows():
        s = int(row["Sample"])
        np.testing.assert_allclose(
            row["beat"], signal[s - cfg.window_left : s + cfg.window_right + 1]
        )


# --- class ordering ----------------------------------------------------------

def test_class_order_is_pinned():
    assert CLASSES == ("F", "N", "S", "V")


def test_targets_use_pinned_order():
    cfg = _cfg(denoise="fir")
    df = segment_record(load_record(RECORD, cfg), cfg)
    _, y = to_features(df, cfg=cfg)
    for i, name in enumerate(CLASSES):
        assert (df["Label"].to_numpy()[y == i] == name).all()


def test_one_hot_roundtrip():
    y = np.array([0, 1, 2, 3, 1])
    assert (np.argmax(one_hot(y), axis=1) == y).all()


def test_extra_features_widen_input():
    cfg = _cfg(denoise="fir")
    df = segment_record(load_record(RECORD, cfg), cfg)
    assert to_features(df, cfg=cfg)[0].shape[1] == 151
    assert to_features(df, ("Length left",), cfg)[0].shape[1] == 152
    assert to_features(df, ("Length left", "Heart rate"), cfg)[0].shape[1] == 153


# --- step 1: seeding ---------------------------------------------------------

def test_seeding_is_reproducible():
    import random
    seed_everything(12); a = (random.random(), np.random.rand())
    seed_everything(12); b = (random.random(), np.random.rand())
    assert a == b


def test_config_is_immutable():
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        _cfg().sampling_rate = 500
