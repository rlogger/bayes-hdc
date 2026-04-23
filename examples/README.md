# bayes-hdc Examples

Runnable examples demonstrating the probabilistic VSA layer, the classical
HDC primitives, and the research-connection demos (weight-space posteriors,
equivariance-respecting pipelines).

## Quick start

```bash
pip install -e ".[examples]"   # core + matplotlib + scikit-learn
python examples/pvsa_quickstart.py
```

## Research-connection demos

| Example | What it shows |
| --- | --- |
| [`weight_space_posterior.py`](weight_space_posterior.py) | A `BayesianCentroidClassifier`'s weights are a `GaussianHV` posterior — a distribution over weight vectors. Sample from it, predict with each draw, read off epistemic uncertainty, and verify the posterior commutes with the cyclic-shift action on hypervectors. |
| [`pvsa_quickstart.py`](pvsa_quickstart.py) | 90-second tour through every PVSA primitive: construct `GaussianHV`, bind / bundle with closed-form moment propagation, expected similarity + similarity variance, `BayesianCentroidClassifier`, conformal coverage. |

## PVSA applications

| Example | What it shows |
| --- | --- |
| [`language_identification.py`](language_identification.py) | Character trigram language ID on 5 European languages (Joshi, Halseth, Kanerva 2016 encoding) with temperature calibration + conformal prediction sets. Long unambiguous sentences collapse to singletons; short ambiguous ones expand. |
| [`medical_selective_prediction.py`](medical_selective_prediction.py) | UCI Breast Cancer Wisconsin Diagnostic with ridge + temperature scaling + conformal abstention. Gates predictions on `conformal_set_size == 1`, reporting accuracy-on-confident vs. overall. |
| [`anomaly_detection.py`](anomaly_detection.py) | OOD detection on UCI digits (train 0–7, score 8–9). MSP baseline vs. the PVSA-exclusive posterior Mahalanobis distance that uses both `mu_c` and `var_c` of the class posterior. |
| [`sequence_memory.py`](sequence_memory.py) | Position-addressable sequence memory: encode a 12-token sentence as one HV, retrieve each token by un-permuting + cleanup, confidence from top-1/top-2 gap. |

## Classical HDC

| Example | What it shows |
| --- | --- |
| [`basic_operations.py`](basic_operations.py) | Binding, bundling, permutation, similarity across MAP / BSC / HRR. |
| [`classification_simple.py`](classification_simple.py) | End-to-end pipeline on synthetic data with `RandomEncoder` + `CentroidClassifier`. |
| [`kanerva_example.py`](kanerva_example.py) | Kanerva's "Dollar of Mexico" — role-filler binding and analogical reasoning. |
| [`song_matching.py`](song_matching.py) | Fun demo — eight pseudo-songs across four themes, each encoded as a bag-of-words bundle. Cosine similarity recovers the theme pairs and the overlap of shared words is visible on every match. |

## Requirements

```bash
pip install -e .                # core library
pip install -e ".[examples]"    # + matplotlib + scikit-learn (needed for the UCI examples)
```

Each example is self-contained, prints a short header describing what it
demonstrates, runs in under a minute on a laptop CPU, and reports numbers
you can sanity-check against the docstring.
