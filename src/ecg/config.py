"""Central configuration.

Every magic number that was previously inline in the thesis notebooks lives
here. Nothing in ``src/ecg`` should hardcode a sampling rate, window length or
class name.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

# --- Record splits -----------------------------------------------------------
# The de Chazal et al. inter-patient division of the MIT-BIH Arrhythmia
# Database. DS1 trains the global model, DS2 is held out for patient-specific
# evaluation. The two sets share no patients.
DS1: tuple[int, ...] = (
    101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
    124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230,
)
DS2: tuple[int, ...] = (
    100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
    212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234,
)

# Records 102, 104, 107 and 217 are paced beats and are excluded by the AAMI
# recommendation; they are absent from both DS1 and DS2 by construction.
PACED_RECORDS: tuple[int, ...] = (102, 104, 107, 217)

# --- Classes -----------------------------------------------------------------
# Pinned explicitly. Never derive this order from ``sorted()`` or from
# ``pd.get_dummies`` column order at runtime: that order is a pandas
# implementation detail, and a silent change to it permutes the model's output
# classes without raising anything.
CLASSES: tuple[str, ...] = ("F", "N", "S", "V")

# AAMI grouping of the WFDB annotation symbols.
AAMI_GROUPS: dict[str, tuple[str, ...]] = {
    "N": ("N", "L", "R", "e", "j"),
    "S": ("A", "a", "J", "S"),
    "V": ("V", "E"),
    "F": ("F",),
    "Q": ("/", "f", "Q"),
}


@dataclass(frozen=True)
class Config:
    """Immutable pipeline configuration."""

    # Data location
    raw_dir: Path = Path("data/raw/mitdb")
    interim_dir: Path = Path("data/interim")

    # Signal
    sampling_rate: int = 360
    lead: str = "MLII"

    # Denoising: one setting for the whole pipeline.
    # "fir" | "median" | "wavelet" | None
    denoise: str | None = "fir"
    fir_cutoff_hz: float = 35.0
    fir_order: int = 12
    median_window_ms: tuple[int, int] = (200, 600)
    wavelet: str = "sym4"
    wavelet_threshold: float = 0.1
    wavelet_levels: tuple[int, ...] = (2, 3, 9, 10)

    # Normalisation applied at load time. None means "leave the signal alone";
    # scaling is then done per-split during training so it is fit on train only.
    normalization: str | None = None

    # Beat segmentation: window is [R - left, R + right], inclusive.
    window_left: int = 50
    window_right: int = 100

    # Labelling
    label_mode: str = "multiclass4"

    # Patient-specific protocol: fine-tune on the patient's first
    # ``train_minutes``, test from ``test_minute`` to the end of the record.
    train_minutes: int = 5
    test_minute: int = 5

    # Training
    batch_size: int = 50
    seed: int = 12

    @property
    def window_length(self) -> int:
        """Total beat window in samples (151 by default)."""
        return self.window_left + self.window_right + 1

    @property
    def window_ms(self) -> float:
        """Beat window duration in milliseconds."""
        return 1000.0 * self.window_length / self.sampling_rate

    @property
    def n_classes(self) -> int:
        return len(CLASSES)

    def minute_to_sample(self, minute: float) -> int:
        """Convert a minute mark to a sample index."""
        return int(minute * 60 * self.sampling_rate)


DEFAULT = Config()


def seed_everything(seed: int = DEFAULT.seed) -> None:
    """Seed every RNG the pipeline touches.

    The original code seeded only the imbalanced-learn samplers, leaving beat
    augmentation and all weight initialisation non-deterministic.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
