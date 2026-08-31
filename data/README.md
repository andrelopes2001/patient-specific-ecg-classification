# Data

Nothing in this directory is tracked by git. The MIT-BIH Arrhythmia Database is
redistributed by PhysioNet under terms that do not permit re-hosting, so the
records are downloaded on demand.

## Getting the data

```bash
python scripts/download_data.py     # -> data/raw/mitdb   (48 records, ~90 MB)
python scripts/build_csv.py         # -> data/interim     (~7 GB, 144 CSVs)
```

Both scripts are idempotent. `download_data.py` skips records already present;
pass `--force` to re-fetch.

## Layout

| Path | Contents |
|---|---|
| `data/raw/mitdb/` | WFDB records as published: `{id}.dat`, `.hea`, `.atr` |
| `data/interim/` | Derived CSVs the thesis pipeline consumes (see below) |

`build_csv.py` reproduces the conversion originally done by
`data_import.ipynb` and `data_merge.ipynb`, writing three files per record:

| File | Contents |
|---|---|
| `{id}.csv` | `record.p_signal`, one row per sample, header = lead names |
| `{id}_annotations.csv` | `Sample`, `Classification` (WFDB symbol), `Aux Note` |
| `{id}_ecg_annotated.csv` | signal left-joined with annotations on sample index |

## Source

- **Database:** MIT-BIH Arrhythmia Database (`mitdb`), version 1.0.0
- **Records:** all 48 (`100`–`234`). Sampled at 360 Hz, 2 leads, 30 min each.
- **URL:** https://physionet.org/content/mitdb/1.0.0/
- **Licence:** Open Data Commons Attribution License v1.0

If you use this data, cite both the database and PhysioNet:

> Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
> IEEE Eng in Med and Biol 20(3):45-50 (2001).

> Goldberger A, et al. PhysioBank, PhysioToolkit, and PhysioNet: Components of
> a New Research Resource for Complex Physiologic Signals. Circulation
> 101(23):e215-e220 (2000).

## Verification against the original working copy

Checked on 2026-08-31 against the original working copy, to confirm
the downloaded data is the same data the thesis results were produced from.

**Raw records — 144/144 files byte-identical** (`.dat`, `.hea`, `.atr` for all
48 records) between a fresh PhysioNet download and the local copy.

**Per-record checks** (records `100` and `213`):

| Check | Result |
|---|---|
| Signal shape | `(650000, 2)` both |
| Sampling frequency | 360 Hz both |
| Lead names | identical (`MLII`/`V5`, `MLII`/`V1`) |
| First 100 samples | identical |
| Last 100 samples | identical |
| Full signal array | identical |
| Annotation count | 2274 / 3294, identical |
| Annotation counts by symbol | identical |

**Derived CSVs**: `{id}.csv` and `{id}_annotations.csv` regenerate
**byte-identical**. `{id}_ecg_annotated.csv` matches on every value in every
column but is 650000 bytes smaller — exactly one byte per row — because the
original was written on Windows with CRLF line endings. After `CRLF -> LF`
normalisation the files are byte-identical.

**Conclusion: the local files were raw, not preprocessed.** The pipeline is
fully reproducible from PhysioNet.
