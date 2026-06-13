# Bayes-HDC tutorials

A pedagogical, read-in-order tour of the library. Each file is a
runnable `.py` with copy-pasteable sections — no notebooks required,
but every file opens cleanly in Colab too.

[![Open 01_quickstart in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rlogger/bayes-hdc/blob/main/tutorials/01_quickstart.py)

## Numbering convention

The leading digits encode reading order, not difficulty.

| Prefix | Meaning                                                            |
|--------|--------------------------------------------------------------------|
| `00_`  | Prerequisites and library setup (JAX, install verification).       |
| `01_`  | Quickstart — installation to first prediction in 90 seconds.       |
| `02_*` | Core PVSA: Gaussian/Dirichlet hypervectors, binding, posteriors.   |
| `03_*` | Calibration and conformal prediction (Lei et al. 2018; Liang 2026).|
| `04_*` | Factorisation, resonator networks, structured queries.             |
| `05_*` | End-to-end on real datasets (EEG, EMG, ISOLET, MNIST).             |

Higher-prefixed files assume you have read the lower-prefixed ones.

## Tutorials

Available now:

- [`01_quickstart.py`](01_quickstart.py) — install, first Gaussian HV,
  iris classification, temperature + conformal prediction, anomaly
  detection. Start here.
- [`02_anomaly_detection.py`](02_anomaly_detection.py) — calibrated
  one-shot anomaly detection from first principles: conformal
  p-values, the coverage guarantee shown empirically, naive-threshold
  comparison, multi-VSA, streaming, and a tabular fraud-style demo.
- [`03_sequences.py`](03_sequences.py) — `Sequence` and
  `HierarchicalSequence` encoding; the flat-vs-hierarchical capacity
  comparison (why hierarchical stays near-perfect at T=800 where flat
  collapses to ~31%).

In progress (numbers reserve reading order; files land as they are written):

- `03_calibration_and_coverage.py` — ECE/MCE, reliability curves,
  split-conformal classification and regression with coverage audits.
- `04_resonator_factorisation.py` — deterministic and probabilistic
  resonator networks for decoding bound compositions.
- `05_real_data_eeg.py` — seizure detection on a real EEG benchmark,
  end-to-end with calibrated probabilities.

## Tutorials vs `examples/`

`tutorials/` is a curriculum: read top to bottom, each file builds on
the last. [`examples/`](../examples/) is a cookbook: each file solves
one applied problem (image classification, gesture recognition, song
matching, etc.) and is meant to be read in isolation.

## Where to next

- Main project [README](../README.md) — install, citation, scope.
- Full API reference at <https://rlogger.github.io/bayes-hdc>.
- [`DESIGN.md`](../DESIGN.md) — architecture notes and design tradeoffs.
- [`BENCHMARKS.md`](../BENCHMARKS.md) — honest pointwise-CPU speedups
  vs eager TorchHD (1.4 - 3.5x range, not a single headline number).

If you spot a bug or a confusing passage in any tutorial, please open
an issue at <https://github.com/rlogger/bayes-hdc/issues>.
