"""Signal denoising and normalisation.

Three denoising strategies are available; ``Config.denoise`` selects one for the
whole pipeline. All three preserve the shape of the input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import firwin, lfilter, medfilt
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .config import DEFAULT, Config

_SCALERS = {
    "MinMax": MinMaxScaler,
    "Standard": StandardScaler,
    "Robust": RobustScaler,
}


def _odd(n: int) -> int:
    """Median filter kernels must be odd."""
    return n if n % 2 else n + 1


def _median_windows(cfg: Config) -> tuple[int, int]:
    """Median filter kernel sizes, in samples, from the configured ms windows."""
    return tuple(
        _odd(int(ms / (1000 / cfg.sampling_rate))) for ms in cfg.median_window_ms
    )


def _baseline(signal: pd.Series, cfg: Config) -> np.ndarray:
    """Estimate baseline wander as the mean of two median-filtered signals.

    A 200 ms median filter removes QRS complexes, a 600 ms one removes P and T
    waves; averaging them leaves the baseline drift.
    """
    w1, w2 = _median_windows(cfg)
    return 0.5 * (medfilt(signal, kernel_size=w1) + medfilt(signal, kernel_size=w2))


def denoise_median(
    record: pd.DataFrame, columns=None, cfg: Config = DEFAULT
) -> pd.DataFrame:
    """Remove baseline wander by subtracting the median-filter estimate."""
    columns = list(record.columns if columns is None else columns)
    return pd.DataFrame(
        {col: record[col] - _baseline(record[col], cfg) for col in columns},
        index=record.index,
    )


def denoise_fir(
    record: pd.DataFrame, columns=None, cfg: Config = DEFAULT
) -> pd.DataFrame:
    """Baseline removal followed by a low-pass FIR filter.

    Note this uses ``lfilter``, which is causal and therefore introduces a phase
    shift of roughly ``fir_order / 2`` samples. That matches the original thesis
    implementation and is kept deliberately: switching to ``filtfilt`` would
    change every downstream beat window.
    """
    columns = list(record.columns if columns is None else columns)
    nyquist = 0.5 * cfg.sampling_rate
    coeffs = firwin(
        cfg.fir_order + 1, cfg.fir_cutoff_hz / nyquist, window="hamming"
    )
    baseline_removed = denoise_median(record, columns, cfg)
    return pd.DataFrame(
        {col: lfilter(coeffs, 1, baseline_removed[col]) for col in columns},
        index=record.index,
    )


def denoise_wavelet(
    record: pd.DataFrame, columns=None, cfg: Config = DEFAULT
) -> pd.DataFrame:
    """Soft-threshold selected wavelet detail coefficients."""
    import pywt

    columns = list(record.columns if columns is None else columns)
    levels = list(cfg.wavelet_levels)
    out = {}

    for col in columns:
        # np.asarray(...).copy() rather than .values: under pandas >= 3 the
        # array returned by .values is read-only, and pywt needs a writable
        # buffer. The original code raised ValueError here.
        data = np.asarray(record[col], dtype=float).copy()
        coeffs = pywt.wavedec(data, cfg.wavelet, level=max(levels))

        # Level n counts back from the end, so it always selects detail
        # coefficients rather than the approximation at index 0.
        for idx in (len(coeffs) - level for level in levels):
            coeffs[idx] = pywt.threshold(
                coeffs[idx],
                cfg.wavelet_threshold * np.max(coeffs[idx]),
                mode="soft",
            )

        reconstructed = pywt.waverec(coeffs, cfg.wavelet)
        # waverec can return one extra sample for odd-length inputs.
        out[col] = reconstructed[: len(data)]

    return pd.DataFrame(out, index=record.index)


_DENOISERS = {
    "fir": denoise_fir,
    "median": denoise_median,
    "wavelet": denoise_wavelet,
}


def denoise(
    record: pd.DataFrame, cfg: Config = DEFAULT, method: str | None = "__cfg__"
) -> pd.DataFrame:
    """Apply the configured denoising method. ``None`` returns the input as-is."""
    method = cfg.denoise if method == "__cfg__" else method
    if method is None:
        return record
    try:
        return _DENOISERS[method](record, None, cfg)
    except KeyError:
        raise ValueError(
            f"unknown denoise method {method!r}; "
            f"expected one of {sorted(_DENOISERS)} or None"
        ) from None


def normalize(data, method: str):
    """Scale with the named scaler. Fits on whatever it is given.

    Only use this where fitting on the input is correct -- i.e. per record at
    load time. For train/test splits use :func:`fit_scaler` so the scaler is
    fit on the training portion alone.
    """
    try:
        scaler = _SCALERS[method]()
    except KeyError:
        raise ValueError(
            f"unknown normalization {method!r}; expected one of {sorted(_SCALERS)}"
        ) from None
    return scaler.fit_transform(data)


def fit_scaler(train, method: str):
    """Fit a scaler on training data only; returns the fitted scaler."""
    try:
        return _SCALERS[method]().fit(train)
    except KeyError:
        raise ValueError(
            f"unknown normalization {method!r}; expected one of {sorted(_SCALERS)}"
        ) from None
