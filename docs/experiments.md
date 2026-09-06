# Experiments

Why the model is a 27,972-parameter MLP and why training departs from the
baseline. Every number here comes from **held-out DS1 patients** — DS2 was used
once, for the final result, after these decisions were fixed.

## Selection protocol

Tuning on DS2 would invalidate the reported result, so DS1 is split
patient-disjointly:

| Pool | Records |
|---|---|
| Train | 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208 |
| Validation | 209, 215, 220, 223, 230 |

The validation patients then go through the **full patient-specific protocol** —
global model, per-patient fine-tuning on their first 5 minutes, test on the rest
— so they stand in for DS2 rather than acting as a plain held-out split.

### Selection metric: macro F1 over N, S and V only

Class F is excluded from *selection*. DS1's fusion beats are 90% concentrated in
record 208, so whichever pool holds 208 starves the other: the validation
patients contain **9 F beats**. F1 computed on 9 samples is noise, and macro-F1
over four classes would make a quarter of the selection metric a coin flip. In
an early run the same configuration scored 0.571 and 0.663 on 4-class macro-F1
purely from that instability.

Restricting to N/S/V cut run-to-run variation to a mean of **0.024** between
repeated runs of the same configuration. F is still reported on DS2, which has
386 F beats.

## Architectures

Each trained on the DS1 pool with class weighting and early stopping, then run
through the patient-specific protocol on the validation patients. Two
independent runs per architecture.

| Architecture | Params | N/S/V run 1 | N/S/V run 2 | Fit time |
|---|---|---|---|---|
| **mlp_128_64** | 27,972 | 0.743 | **0.759** | 22–50 s |
| mlp_256_128_64 | 80,324 | 0.722 | 0.750 | 33–70 s |
| cnn_32_64 | 162,756 | 0.690 | 0.721 | 47–198 s |
| mlp_512_256_128 + dropout | 242,564 | 0.691 | 0.670 | 45–79 s |
| cnn_64_128 + dropout | 649,092 | 0.562 | 0.695 | 78–306 s |
| cnn_residual | 129,604 | 0.662 | 0.620 | 144–682 s |
| bilstm_64 | 42,308 | — | 0.588 | 851 s |

**Capacity hurts.** `mlp_128_64` is the smallest model tried and the only one
top-ranked in both runs. `mlp_256_128_64` is second in both, but the 0.009–0.016
gap sits inside run-to-run noise, so the two are tied and the smaller wins on
size alone. Everything from `cnn_32_64` down is worse by a margin that exceeds
the noise, and the BiLSTM is both the worst and 17× slower to train than the MLP.

The mechanism is visible in the numbers: the global model reaches **0.99
training accuracy on DS1 but ~0.79 on DS2 beats**. Larger models fit DS1 harder
and generalise worse to unseen patients, while the ~370 fine-tuning beats per
patient are far too few to exploit extra capacity.

## Training changes — a negative result

Two changes looked like clear wins on the validation patients:

| Change | Validation macro F1 (N/S/V) |
|---|---|
| Inverse-frequency class weights | **+0.035** |
| Early stopping (26–35 epochs) vs fixed 10 | **+0.030** |
| Per-beat z-normalisation | −0.03 (rejected immediately) |

The training loss was still falling at epoch 10, so the reference schedule really
does stop short, and DS1 really is 90% class N. Both changes were adopted.

**They then made DS2 worse.** Measured across the same three seeds, 5 minutes of
fine-tuning:

| Configuration | Macro F1 (DS2) |
|---|---|
| Baseline — 10 epochs, no class weighting | **0.857 ± 0.009** |
| Class weights + early stopping | 0.833 ± 0.011 |

The 0.024 loss is wider than the seed spread, so it is not noise. Both changes
were reverted; they remain available as `cfg.class_weight` and
`cfg.early_stopping_patience`, off by default, so the result can be reproduced:

```bash
python scripts/sweep.py --minutes 5 --seeds 12 13 14                  # 0.857
python scripts/sweep.py --minutes 5 --seeds 12 13 14 --class-weight \
                        --epochs 40                                   # 0.833
```

### Why it did not transfer

Five validation patients is too small a sample for the *decision*, even once the
metric itself is stable. Restricting to N/S/V fixed the metric's variance
(0.024 run-to-run) but did nothing about sampling error across patients — and
inter-patient morphology variation is the dominant source of difficulty in this
problem. A grouped k-fold across all of DS1, so every patient is validated once,
would have exposed this; it was not run because a single split was cheaper.

There is also a plausible mechanism. The global model is class-weighted but the
per-patient fine-tuning stage is not, so the two stages optimise different
objectives; personalisation then has to undo the global model's bias toward rare
classes using only ~370 beats.

## Conclusion

The baseline architecture **and** its training schedule were both left unchanged.
Every attempt to improve on them — larger MLPs, three CNN variants, a BiLSTM,
class weighting, early stopping, per-beat normalisation — either made no
difference or made results worse.

That is the honest outcome of the study, and a more useful one for a reader than
a tuned number would have been: on this problem, with ~370 personalisation beats
per patient, the binding constraint is the amount of patient-specific data, not
model capacity or training schedule.

## Reproducing

```bash
python scripts/sweep.py --seeds 12 13 14      # the fine-tuning-data curve
python scripts/train.py --epochs 10 --seed 12 # a single configuration
```

The architecture comparison is not wired into a CLI: it was a one-off selection
study whose conclusion — keep the MLP — is the current default.
