# Patient-Specific ECG Arrhythmia Classification

Classifying individual heartbeats into AAMI arrhythmia classes on the MIT-BIH
Arrhythmia Database, where a small global model is **personalised to each patient
using the first few minutes of their own recording**.

A general ECG classifier has to cope with the fact that healthy beat morphology
varies enormously between people — one patient's normal beat looks like another's
abnormality. Personalising the model to the individual sidesteps that, and the
practical question becomes: *how little of a patient's own data do you need?*

![Personalisation curve](results/personalisation_curve.png)

**Three minutes.** Beyond that, more fine-tuning data adds very little — and it
matters far more than any of the synthetic-data augmentation tried here.

---

## What "patient-specific" means here

Two stages, and the distinction matters:

1. **Global model** — trained on **DS1** (22 records), a small MLP over single
   beats.
2. **Personalisation** — for each **DS2** patient, a *copy* of the global model
   is fine-tuned on that patient's own first *n* minutes, then evaluated on
   minutes 5–30 of the same record.

```
DS1 (22 patients) ──train──> global model
                                  │
DS2 patient, min 0–5 ──fine-tune──┤ (a copy, per patient)
                                  ▼
DS2 patient, min 5–30 ─────test───> pooled confusion matrix
```

This is the de Chazal *et al.* inter-patient protocol. **DS1 and DS2 share no
patients**, and within a patient the fine-tuning and test windows are temporally
disjoint. The model does see the first five minutes of the patient it is tested
on — that is the point of the method, not a leak — but it never sees a test beat.

## Results

Reported over all 22 DS2 patients pooled into one confusion matrix, 4 AAMI
classes (F, N, S, V), 5 minutes of fine-tuning.

| Augmentation | Extra features | Accuracy | Macro F1 | Kappa | F1 (F) | F1 (N) | F1 (S) | F1 (V) |
|---|---|---|---|---|---|---|---|---|
| cGAN | length + rate | **0.981** | 0.851 | **0.901** | 0.612 | 0.990 | 0.843 | 0.957 |
| cGAN | heart rate | 0.980 | 0.848 | 0.897 | 0.604 | 0.990 | 0.840 | 0.956 |
| **none** | — | 0.974 | **0.862** | 0.869 | **0.758** | 0.986 | 0.754 | 0.951 |
| SMOTE | — | 0.970 | 0.813 | 0.852 | 0.558 | 0.984 | 0.759 | 0.950 |
| cGAN | — | 0.969 | 0.806 | 0.849 | 0.532 | 0.984 | 0.758 | 0.951 |

Full sweep in [`results/thesis_metrics.csv`](results/thesis_metrics.csv);
the analysis is in [`notebooks/02_results.ipynb`](notebooks/02_results.ipynb).

**Reading these honestly.** The metric that matters on data this imbalanced is
macro F1, and by that measure **the no-augmentation baseline wins**. Accuracy is
near-meaningless here — class N is 89% of beats, so predicting "normal" for
everything scores 0.89. The augmentation work earns its place at *short*
personalisation windows, not at the ceiling; by three minutes the advantage is
gone. Class F (fusion beats, under 1% of the data) is the weak point in every
configuration.

> ⚠️ **These numbers are from the original thesis experiments, not reproduced by
> this repository.** The saved checkpoints do not align with the documented
> pipeline, so the metrics are cited from the recorded results rather than
> re-derived.
> Training is not run here.

## Model

```
input 151 (beat window)  ─>  Dense 128 relu  ─>  Dense 64 relu  ─>  Dense 4 softmax
```

**27,972 parameters.** Adam, categorical cross-entropy. Optional variants
prepend the gap to the previous beat and the instantaneous heart rate, giving
152 or 153 inputs — cheap rhythm context an isolated beat cannot carry.

The small model is deliberate: the thesis motivation was wearable-scale
inference. **Inference latency was never benchmarked**, so this repo makes no
latency claim — only the architectural observation that the model is 28k
parameters over a 419 ms window.

## Data and preprocessing

MIT-BIH Arrhythmia Database — 48 records, 30 minutes each, 360 Hz, 2 leads.
Lead **MLII** is used throughout.

| Stage | Detail |
|---|---|
| Denoising | Baseline wander removed via 200 ms + 600 ms median filters, then a 35 Hz low-pass FIR |
| Resampling | None — native 360 Hz |
| Segmentation | 151-sample window centred on the annotated R-peak: 50 before, 100 after (**419 ms**) |
| Beat rejection | Beats whose neighbours are too close for the window to fit |
| Normalisation | Fit on the fine-tuning split only, never on test |
| Classes | AAMI F / N / S / V (`multiclass4`); paced records 102, 104, 107, 217 excluded |

Class balance on DS2 — the core difficulty:

| N | V | S | F |
|---|---|---|---|
| 89.02% | 6.51% | 3.69% | 0.78% |

R-peak locations come from the database's expert annotations; this work
classifies beats, it does not detect them.

## Quickstart

```bash
git clone https://github.com/andrelopes2001/patient-specific-ecg-classification
cd patient-specific-ecg-classification

uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[keras,torch,dev]"
```

Fetch the data — it is **not** in this repository, and cannot be: PhysioNet's
terms do not permit redistribution.

```bash
python scripts/download_data.py    # 48 records, ~90 MB, from PhysioNet
python scripts/build_csv.py        # derive per-record CSVs (~7 GB)
```

Then:

```bash
pytest                                    # 29 tests, ~10 s, no training
python scripts/train.py --dry-run         # inspect the config, train nothing
python scripts/evaluate.py --predictions tests/fixtures/predictions_ds2.npz
```

To actually train — **hours on CPU**, and not required to explore the repo:

```bash
python scripts/train.py --epochs 10 --balance smote --train-minutes 5
```

## Layout

```
src/ecg/
  config.py         Config dataclass — every parameter, DS1/DS2, class order
  data.py           record loading
  preprocessing.py  FIR / median / wavelet denoising, scaling
  segmentation.py   R-peak windowing, AAMI labelling, feature assembly
  balance.py        oversampling, SMOTE, shift augmentation
  models.py         Keras architectures
  cgan.py           conditional GAN for minority-class synthesis (PyTorch)
  train.py          global training, patient-specific fine-tuning
  evaluate.py       metrics and confusion matrices
  viz.py            plotting
scripts/            download_data, build_csv, train, evaluate
notebooks/          01_signal_exploration, 02_results
tests/              fixture-based regression tests
```

## Reproducibility

`seed_everything()` seeds Python, NumPy, TensorFlow and PyTorch. All parameters
live in a frozen `Config`; nothing downstream hardcodes a sampling rate or window
bound.

The **thesis** results predate this and came from an unseeded run — the original
seeded only the imbalanced-learn samplers, leaving beat augmentation and weight
initialisation non-deterministic. Those numbers are therefore not bit-reproducible
even with the original code, which is part of why they are cited rather than
re-derived.

What *is* pinned exactly: [`tests/`](tests/) asserts the segmented beat tensor
against an MD5 captured from the original implementation, so any change to
filtering or windowing fails loudly rather than quietly costing F1.

## Provenance

This is a refactor of the code behind an MSc dissertation (2024), *Patient-specific Real-Time Architecture for ECG Classification*.

The original was ~70 notebooks against a single 2,213-line `functions.py`, with
one function defined 39 times across a single notebook.

Two things worth calling out, since a reader will wonder:

- **An earlier version of this work had a leaky evaluation** — a random 80/20
  split applied *after* oversampling, putting duplicated beats on both sides. It
  was abandoned before the final results in favour of the patient-disjoint
  DS1/DS2 split. It is documented rather than quietly deleted.
- **Fine-tuning used to mutate the global model in place**, so each patient's
  personalisation resumed from the previous patient's weights. Fixed here
  (`train.py`), and covered by a regression test.

## Citation

If you use the data, cite the database and PhysioNet:

> Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
> *IEEE Eng in Med and Biol* 20(3):45-50 (2001).

> Goldberger A, et al. PhysioBank, PhysioToolkit, and PhysioNet: Components of a
> New Research Resource for Complex Physiologic Signals.
> *Circulation* 101(23):e215-e220 (2000).

## License

MIT — see [LICENSE](LICENSE). The MIT-BIH data is separately licensed by
PhysioNet under the Open Data Commons Attribution License v1.0.
