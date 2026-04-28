# Examples

Runnable end-to-end examples. Each one is self-contained, prints a short
header describing what it does, runs in under a minute on a laptop CPU,
and reports numbers you can sanity-check against the docstring.

## Quick start

```bash
pip install -e ".[examples]"   # core + matplotlib + scikit-learn
python examples/pvsa_quickstart.py
```

## Get started

| Example | What it shows |
| --- | --- |
| [`pvsa_quickstart.py`](pvsa_quickstart.py) | 90-second tour through every PVSA primitive: construct `GaussianHV`, bind / bundle with closed-form moment propagation, expected similarity, similarity variance, `BayesianCentroidClassifier`, conformal coverage. |
| [`basic_operations.py`](basic_operations.py) | Binding, bundling, permutation, similarity across MAP / BSC / HRR. |
| [`classification_simple.py`](classification_simple.py) | End-to-end pipeline with `RandomEncoder` + `CentroidClassifier`. |

## Applications

| Example | What it shows |
| --- | --- |
| [`emg_gesture_recognition.py`](emg_gesture_recognition.py) | 8-channel sEMG hand-gesture classification: RMS-per-channel → discretise → channel-value binding + bundle → `BayesianCentroidClassifier` with calibrated probabilities and per-gesture posterior variance. Synthetic data; the same pipeline runs on real data via `bayes_hdc.datasets.load_emg()`. |
| [`activity_recognition.py`](activity_recognition.py) | UCIHAR-style 6-class daily-living activity recognition (walking, stairs up/down, sitting, standing, laying) with feature-value binding, temperature calibration, conformal sets at α = 0.1, and selective-abstention reporting. Pass `--real-data` to load the real UCIHAR benchmark. |
| [`image_classification.py`](image_classification.py) | Classical HDC for vision — random-projection encoding + `CentroidClassifier` (one-shot bundling), `AdaptiveHDC` (5-epoch refinement), and `RegularizedLSClassifier` (closed-form ridge) compared on the same encoded hypervectors. Bundled 8×8 digits offline; pass `--real-data` to load real MNIST 28×28. |
| [`language_identification.py`](language_identification.py) | Character-trigram language ID on 5 European languages with temperature calibration and conformal prediction sets. Long unambiguous sentences collapse to singletons; short ambiguous ones expand. |
| [`sequence_memory.py`](sequence_memory.py) | Position-addressable sequence memory: encode a 12-token sentence as one HV, retrieve each token by un-permuting + cleanup, confidence from top-1/top-2 gap. |
| [`weight_space_posterior.py`](weight_space_posterior.py) | A `BayesianCentroidClassifier`'s weights as a `GaussianHV` posterior. Sample from it, predict with each draw, read off epistemic uncertainty, and verify the posterior commutes with the cyclic-shift action. |
| [`song_matching.py`](song_matching.py) | Bag-of-words song similarity. The sum of word hypervectors is legible by eye; cosine similarity recovers theme pairs and the overlap of shared words is visible on every match. |
| [`kanerva_example.py`](kanerva_example.py) | "Dollar of Mexico" — role-filler binding and analogical reasoning. |

## Requirements

```bash
pip install -e .                # core library
pip install -e ".[examples]"    # + matplotlib + scikit-learn (needed for several examples)
```
