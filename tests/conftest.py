"""Shared test configuration.

Tests that need the MIT-BIH records skip cleanly when the data has not been
downloaded, so a fresh clone can run `pytest` and see the data-free tests pass
rather than a wall of FileNotFoundError.
"""

from __future__ import annotations

from pathlib import Path

import pytest

INTERIM = Path("data/interim")
FIXTURE_RECORD = 213

_REQUIRED = [
    INTERIM / f"{FIXTURE_RECORD}.csv",
    INTERIM / f"{FIXTURE_RECORD}_annotations.csv",
    INTERIM / f"{FIXTURE_RECORD}_ecg_annotated.csv",
]


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "needs_data: requires the MIT-BIH records to be downloaded"
    )


def pytest_collection_modifyitems(config, items):
    if all(p.exists() for p in _REQUIRED):
        return
    skip = pytest.mark.skip(
        reason=(
            f"record {FIXTURE_RECORD} not found in {INTERIM}. Run:\n"
            f"  python scripts/download_data.py\n"
            f"  python scripts/build_csv.py --records {FIXTURE_RECORD}"
        )
    )
    for item in items:
        if "needs_data" in item.keywords:
            item.add_marker(skip)
