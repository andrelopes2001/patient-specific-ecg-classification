# Data

Nothing in this directory is tracked by git. The MIT-BIH Arrhythmia Database is
redistributed by PhysioNet under terms that do not permit re-hosting, so the
records are downloaded on demand.

## Getting the data

```bash
python scripts/download_data.py     # -> data/raw/mitdb   (48 records, ~90 MB)
python scripts/build_csv.py         # -> data/interim     (~7 GB, 144 CSVs)
```

Both scripts are idempotent and skip work already done; pass `--force` to
either one to redo it. `build_csv.py` also takes `--records` to build a subset,
which is enough to run the tests:

```bash
python scripts/build_csv.py --records 213    # just the fixture record
```

## Layout

| Path | Contents |
|---|---|
| `data/raw/mitdb/` | WFDB records as published: `{id}.dat`, `.hea`, `.atr` |
| `data/interim/` | Derived CSVs the pipeline consumes (see below) |

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

## Integrity

The download is verified against the published record checksums: all 48 records
load at `(650000, 2)` samples, 360 Hz, with annotation counts matching the
database. `build_csv.py` is a lossless re-encoding of `wfdb` output — the signal
CSVs and annotation CSVs regenerate byte-identically from a fresh download.

`tests/test_pipeline.py` asserts an MD5 over the segmented beat tensor, so any
change to the download, the CSV derivation, or the preprocessing is caught.
