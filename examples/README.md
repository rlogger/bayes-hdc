# bayes-hdc Examples

Runnable examples organised by what they demonstrate.

The application set mirrors what HDC is most-applied to in the recent
literature (Kleyko et al. 2022 survey): biosignal classification first
and foremost (EMG gestures, activity recognition from IMU streams),
then language identification, then sequence memory. Each application
example uses the literature-canonical encoder for its domain.

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

## PVSA applications — literature-canonical HDC tasks

| Example | What it shows |
| --- | --- |
| [`emg_gesture_recognition.py`](emg_gesture_recognition.py) | The marquee modern HDC application. 8-channel sEMG → RMS-per-channel → discretise → channel-value binding + bundle → `BayesianCentroidClassifier` with calibrated probabilities and per-gesture posterior variance. Pipeline is literature-canonical (Rahimi et al. 2016, Burrello et al. 2018, Hersche et al. 2019); synthetic data here, real data via `bayes_hdc.datasets.load_emg()`. |
| [`activity_recognition.py`](activity_recognition.py) | UCIHAR-style 6-class daily-living recognition (walking, stairs up/down, sitting, standing, laying) with feature-value binding, temperature calibration, conformal sets at α = 0.1, and selective-abstention reporting. Pipeline matches Anguita et al. (2013), Hassan et al. (2018), Schmuck et al. (2019); synthetic 36-feature windows here, real data via `bayes_hdc.datasets.load_ucihar()`. |
| [`language_identification.py`](language_identification.py) | Character trigram language ID on 5 European languages (Joshi, Halseth, Kanerva 2016 encoding) with temperature calibration + conformal prediction sets. Long unambiguous sentences collapse to singletons; short ambiguous ones expand. |
| [`sequence_memory.py`](sequence_memory.py) | Position-addressable sequence memory: encode a 12-token sentence as one HV, retrieve each token by un-permuting + cleanup, confidence from top-1/top-2 gap. |

## Classical HDC

| Example | What it shows |
| --- | --- |
| [`basic_operations.py`](basic_operations.py) | Binding, bundling, permutation, similarity across MAP / BSC / HRR. |
| [`classification_simple.py`](classification_simple.py) | End-to-end pipeline on synthetic data with `RandomEncoder` + `CentroidClassifier`. |
| [`kanerva_example.py`](kanerva_example.py) | Kanerva's "Dollar of Mexico" — role-filler binding and analogical reasoning. |
| [`song_matching.py`](song_matching.py) | Bag-of-words song similarity. The sum of word hypervectors is legible by eye; cosine similarity recovers theme pairs and the overlap of shared words is visible on every match. |

## Requirements

```bash
pip install -e .                # core library
pip install -e ".[examples]"    # + matplotlib + scikit-learn (needed for several examples)
```

Each example is self-contained, prints a short header describing what it
demonstrates, runs in under a minute on a laptop CPU, and reports numbers
you can sanity-check against the docstring.
