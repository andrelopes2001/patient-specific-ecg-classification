"""Loading MIT-BIH records.

Reads the CSVs produced by ``scripts/build_csv.py``. Paths come from
:class:`~ecg.config.Config`; the original code derived the data directory from
``os.getcwd()``, which only worked when run from a notebook nested exactly two
levels deep.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DEFAULT, Config
from .preprocessing import denoise, normalize


@dataclass
class Record:
    """One MIT-BIH record: signal, annotations, and the two joined."""

    record_id: int
    signal: pd.DataFrame       # (n_samples, n_leads), column names are lead names
    annotations: pd.DataFrame  # Sample, Label, Aux Note
    annotated: pd.DataFrame    # signal left-joined with annotations on Sample

    @property
    def leads(self) -> list[str]:
        return list(self.signal.columns)

    @property
    def n_samples(self) -> int:
        return len(self.signal)

    def __repr__(self) -> str:
        return (
            f"Record({self.record_id}, {self.n_samples} samples, "
            f"leads={self.leads}, {len(self.annotations)} annotations)"
        )


def _require(path: Path, record_id: int) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path} for record {record_id}. "
            f"Run: python scripts/download_data.py && python scripts/build_csv.py"
        )
    return path


def load_record(record_id: int, cfg: Config = DEFAULT) -> Record:
    """Load one record, applying the configured denoising and normalisation."""
    base = Path(cfg.interim_dir)

    signal = pd.read_csv(_require(base / f"{record_id}.csv", record_id))
    annotations = pd.read_csv(
        _require(base / f"{record_id}_annotations.csv", record_id)
    )
    annotated = pd.read_csv(
        _require(base / f"{record_id}_ecg_annotated.csv", record_id),
        low_memory=False,
    )

    annotations = annotations.rename(columns={"Classification": "Label"})
    annotated = annotated.rename(columns={"Classification": "Label"})

    lead_columns = list(signal.columns)
    signal = denoise(signal, cfg)
    if cfg.normalization is not None:
        signal[lead_columns] = normalize(signal[lead_columns], cfg.normalization)

    # Keep the annotated frame's signal columns consistent with the processed
    # signal, so downstream code can use either interchangeably.
    annotated[lead_columns] = signal[lead_columns]

    return Record(record_id, signal, annotations, annotated)


def load_records(record_ids, cfg: Config = DEFAULT):
    """Yield records one at a time.

    A generator rather than a list: the full database is several GB once
    expanded, and every caller consumes records sequentially.
    """
    for record_id in record_ids:
        yield load_record(record_id, cfg)
