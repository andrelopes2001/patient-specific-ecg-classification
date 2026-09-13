# Patient-Specific ECG Arrhythmia Classification

Classifying individual heartbeats into AAMI arrhythmia classes on the MIT-BIH
Arrhythmia Database, using a compact model that is **personalised to each patient
from a few minutes of their own recording**.

Healthy beat morphology varies enormously between people — one patient's normal
beat can look like another's abnormality — which is why a single global ECG
classifier generalises poorly to a new patient. This project quantifies the
alternative: adapt the model to the individual, and measure how little of their
data is actually needed.

The research was carried out in 2024; the code was cleaned up and the results
re-derived in this repository in 2026.

![Personalisation curve](results/personalisation_curve.png)

**Three minutes.** Beyond that, additional patient data buys very little.

## Headline result

Evaluated on the 22 held-out DS2 patients, pooled into one confusion matrix.
4 AAMI classes, 5 minutes of personalisation, averaged over 3 seeds.

| | Accuracy | Macro F1 | Cohen's κ |
|---|---|---|---|
| Always predict "normal" | 0.889 | 0.235 | 0.000 |
| Global model, no personalisation | 0.790 | 0.379 | 0.296 |
| **Personalised** | **0.978 ± 0.001** | **0.857 ± 0.009** | **0.889** |

**Personalisation is worth +0.48 macro F1.** The global model on its own scores
*below* the trivial always-normal baseline on accuracy — a direct measurement of
how badly inter-patient morphology shift hurts, and the reason the personalised
approach exists.

Accuracy is close to meaningless on this data: class N is 89% of beats, so
predicting "normal" everywhere already scores 0.889. Macro F1 and κ are the
metrics that carry information.

### Per-class performance

Reported in the standard AAMI format — sensitivity (Se), positive predictive
value (+P) and specificity — so these are directly comparable with the published
literature on this benchmark.

| Class | Beats | Se | +P | Spec | F1 |
|---|---|---|---|---|---|
| N — normal | 36,566 | 0.990 | 0.987 | 0.898 | 0.989 |
| S — supraventricular ectopic | 1,590 | 0.747 | 0.895 | 0.996 | 0.814 |
| V — ventricular ectopic | 2,674 | 0.966 | 0.942 | 0.996 | 0.954 |
| F — fusion | 286 | 0.776 | 0.593 | 0.996 | 0.672 |

```
confusion matrix          predicted
(mean of 3 seeds)      F      N      S      V
                  F  222     33      1     30
                  N  115  36208    132    111
                  S    5    382   1187     16
                  V   34     51      7   2582
```

S and F are the hard classes, as they are throughout this literature: S beats
are morphologically close to N and differ mainly in timing, and F beats are by
definition intermediate between N and V. F is 0.6% of the data.

### Comparison with published work

Direct comparison on this benchmark needs care: published patient-specific
results are usually reported over 5 AAMI classes (including the "unknown" class
Q) while this project uses 4, and per-class accuracy is often averaged across
records rather than pooled. The figures below are the closest like-for-like
available — same database, same patient-specific protocol, same per-class
one-vs-rest metrics.

| | SVEB (S) accuracy | SVEB F1 | VEB (V) accuracy | VEB F1 |
|---|---|---|---|---|
| 1D Self-ONN, [Malik et al. 2021](https://arxiv.org/abs/2110.02215) | 0.980 | 0.766 | 0.990 | 0.937 |
| **This project** | **0.988** | **0.814** | **0.995** | **0.954** |

Ahead on both classes, with the caveats above and with a 27,972-parameter model.
Per-class results are reported in the standard Se / +P / specificity format
above precisely so they can be checked against any paper on this benchmark.

### What did not help

Every attempt to improve on the compact model made results worse, which is
itself the useful finding. Full study in [`docs/experiments.md`](docs/experiments.md);
all model selection was done on **held-out DS1 patients**, with DS2 used once.

| | Macro F1 (validation, N/S/V) |
|---|---|
| **MLP, 27,972 params** | **0.759** |
| MLP, 80k params | 0.750 |
| 1-D CNN, 163k params | 0.721 |
| MLP, 243k params + dropout | 0.670 |
| Residual CNN, 130k params | 0.620 |
| BiLSTM, 42k params | 0.588 |

Capacity hurts, monotonically. With roughly 370 personalisation beats per
patient there is nothing for a larger model to exploit, and bigger networks fit
the training records harder while generalising worse to unseen ones. Class
weighting and early stopping gained +0.035 on validation and then **lost 0.024
on DS2** — recorded as a negative result rather than quietly dropped.

## Class imbalance

DS1 is 90% class N and 0.8% class F, so the obvious move is to rebalance the
training set. Two approaches were tried: SMOTE, the standard interpolation
baseline, and a conditional GAN (cGAN) that generates beats of a requested
class. Both were measured like everything else: 22 held-out DS2 patients,
3 seeds, identical test beats.

| Balancing | Global model alone | After personalisation | F1 (F) | F1 (S) | F1 (V) |
|---|---|---|---|---|---|
| None | 0.387 ± 0.002 | **0.857 ± 0.009** | 0.672 | 0.814 | **0.954** |
| **cGAN** | **0.409 ± 0.023** | 0.854 ± 0.017 | 0.655 | **0.824** | 0.949 |
| SMOTE | 0.364 ± 0.028 | 0.813 ± 0.012 | 0.553 | 0.785 | 0.931 |

*(macro F1; "global model alone" is the patient-independent model with no
fine-tuning, on the same DS2 beats)*

The cGAN beats SMOTE at both stages, and is the only method that nudges the
global model up (+0.021, about one seed standard deviation). After
personalisation, though, it only ties no balancing. Its generated beats are
noisy and weakest for class S, so it needs serious work — class-balanced
training, and a check that generated beats actually vary with the requested
class and the noise input — before synthetic data can add real information.
Details are in [`docs/experiments.md`](docs/experiments.md).

Why neither method helps after personalisation is the interesting part:
**personalisation already corrects the imbalance**. Fine-tuning on a patient's first five minutes adapts the model to
the class mix seen in that window, so globally rebalancing DS1 corrects a
problem that the next stage was going to correct anyway — and SMOTE's synthetic
interpolants actively distort the prior that fine-tuning must then undo.

The global model still matters: it is the only source of knowledge about
classes a patient has not shown yet. Only 3 of the 22 test patients show all
four classes in their first five minutes.

**Limitation.** Personalisation adapts to the classes seen during calibration,
and largely overwrites the rest. Where S beats first appear only after the first
five minutes — almost entirely one record, 222 — detection is low. Two things
compound: fine-tuning to near-zero loss on a window with no S beats teaches the
model that this patient has none, and S differs from N mainly in timing and
P-wave shape, which a single 419 ms beat window carries only weakly, so the
global model's knowledge of S is fragile to begin with. V beats, which are
morphologically distinct, survive much better. Keeping fine-tuning closer to the
global model, for example by replaying DS1 minority beats during fine-tuning, is
the natural next step.

Even with balancing, the global model reaches only 0.409 macro F1 at ~0.82
accuracy, still short of the 0.889 accuracy of predicting "normal" everywhere.
Rebalancing moves a patient-independent classifier from unusable to marginally
less unusable; it is not a substitute for adaptation.

## Method

Two stages:

1. **Global model** — a small MLP trained on DS1 (22 records).
2. **Personalisation** — for each DS2 patient, a *copy* of that model is
   fine-tuned on that patient's first *n* minutes, then evaluated on the
   remainder of their record.

```
DS1 (22 patients) ──train──> global model
                                  │
DS2 patient, min 0–5 ──fine-tune──┤ (an independent copy per patient)
                                  ▼
DS2 patient, min 5–30 ─────test───> pooled confusion matrix
```

This is the [de Chazal *et al.*](https://doi.org/10.3390/a13040075) inter-patient
protocol. **DS1 and DS2 share no patients**, and within a patient the
personalisation and test windows are disjoint in time. The model does see the
first five minutes of the patient it is tested on — that is the method, not a
leak — but never a beat it is scored on. The guarantee is enforced by tests, not
by convention: see [`tests/test_protocol.py`](tests/test_protocol.py).

## Model

```
input 151  →  Dense 128 (relu)  →  Dense 64 (relu)  →  Dense 4 (softmax)
```

**27,972 parameters.** Input is a 151-sample window centred on the R-peak — 50
samples before, 100 after — at 360 Hz, so 419 ms of single-lead (MLII) signal.

The size is deliberate. The whole pipeline trains and personalises on a **laptop
CPU with no GPU**, which is what makes on-device personalisation practical: a
patient's ECG never has to leave their own hardware to adapt the model to them.
A 28k-parameter network is also small enough to be a credible target for
wearable-class inference, though **inference latency is not benchmarked here** and
no latency claim is made.

Optional variants prepend the interval to the previous beat and the
instantaneous heart rate, giving 152 or 153 inputs — cheap rhythm context that a
single isolated beat cannot carry.

## Data and preprocessing

MIT-BIH Arrhythmia Database: 48 records, 30 minutes each, 360 Hz, 2 leads.

| Stage | Detail |
|---|---|
| Denoising | Baseline wander removed with 200 ms + 600 ms median filters, then a 35 Hz low-pass FIR |
| Resampling | None — native 360 Hz |
| Segmentation | 151-sample window on the annotated R-peak: 50 before, 100 after |
| Beat rejection | Beats whose neighbours sit too close for the window to fit |
| Normalisation | Fit on the personalisation split only, never on test |
| Classes | AAMI F / N / S / V; paced records 102, 104, 107, 217 excluded per AAMI |

Class balance on DS2, and the central difficulty:

| N | V | S | F |
|---|---|---|---|
| 89.0% | 6.5% | 3.7% | 0.8% |

R-peak locations come from the database's expert annotations: this classifies
beats, it does not detect them.

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
