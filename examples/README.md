# bayes-hdc Examples

Runnable examples demonstrating the PVSA (Probabilistic Vector Symbolic
Architectures) layer and classical HDC primitives.

## Quick start

```bash
pip install -e ".[examples]"   # core + matplotlib + scikit-learn
python examples/pvsa_quickstart.py
```

## Classical HDC

| Example | What it shows |
| --- | --- |
| [`basic_operations.py`](basic_operations.py) | Binding, bundling, permutation, similarity across MAP / BSC / HRR. |
| [`classification_simple.py`](classification_simple.py) | End-to-end pipeline on synthetic data with `RandomEncoder` + `CentroidClassifier`. |
| [`kanerva_example.py`](kanerva_example.py) | Kanerva's "Dollar of Mexico" — role-filler binding and analogical reasoning. |
| [`sequence_memory.py`](sequence_memory.py) | Position-addressable sequence memory: encode a 12-token sentence as one HV, retrieve each token by un-permuting + cleanup, confidence from top-1/top-2 gap. |

## PVSA — probabilistic layer

| Example | What it shows |
| --- | --- |
| [`pvsa_quickstart.py`](pvsa_quickstart.py) | 90-second tour: construct `GaussianHV`, bind / bundle with closed-form moment propagation, expected similarity + similarity variance, `BayesianCentroidClassifier`, conformal coverage. |
| [`language_identification.py`](language_identification.py) | Character trigram language ID on 5 European languages (Joshi, Halseth, Kanerva 2016 encoding) with temperature calibration + conformal prediction sets. Long unambiguous sentences collapse to singletons; short ambiguous ones expand. |
| [`medical_selective_prediction.py`](medical_selective_prediction.py) | UCI Breast Cancer Wisconsin Diagnostic with ridge + temperature scaling + conformal abstention. Gates predictions on `conformal_set_size == 1`, reporting accuracy-on-confident vs. overall. |
| [`anomaly_detection.py`](anomaly_detection.py) | OOD detection on UCI digits (train 0–7, score 8–9). MSP baseline vs. the PVSA-exclusive *posterior Mahalanobis distance* that uses both `mu_c` and `var_c` of the class posterior. |

## Requirements

```bash
pip install -e .                # core library
pip install -e ".[examples]"    # + matplotlib + scikit-learn (needed for the UCI examples)
```

All examples are self-contained — each one prints a short header describing what
it demonstrates, runs in under 30 seconds on a laptop CPU, and reports numbers
you can sanity-check against the docstring.
