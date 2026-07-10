<p align="center">
  <a href="https://github.com/rlogger/bayes-hdc">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/banner_dark.svg">
      <img src="assets/banner.svg" alt="bayes-hdc — probabilistic hyperdimensional computing in JAX" width="100%">
    </picture>
  </a>
</p>

<p align="center">
  <a href="https://github.com/rlogger/bayes-hdc/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/rlogger/bayes-hdc/actions/workflows/tests.yml/badge.svg?branch=main" /></a>
  <a href="https://codecov.io/gh/rlogger/bayes-hdc"><img alt="Coverage" src="https://codecov.io/gh/rlogger/bayes-hdc/graph/badge.svg" /></a>
  <a href="https://pypi.org/project/bayes-hdc/"><img alt="PyPI" src="https://img.shields.io/pypi/v/bayes-hdc" /></a>
  <a href="https://doi.org/10.5281/zenodo.20635099"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.20635099.svg" /></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.9–3.13-blue.svg" />
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
</p>

<p align="center">
  <a href="https://rlogger.github.io/bayes-hdc/"><strong>Docs</strong></a> ·
  <a href="https://colab.research.google.com/github/rlogger/bayes-hdc/blob/main/tutorials/01_quickstart.ipynb">Colab quickstart</a> ·
  <a href="examples/">Examples</a> ·
  <a href="BENCHMARKS.md">Benchmarks</a> ·
  <a href="https://github.com/rlogger/bayes-hdc/discussions">Discussions</a>
</p>

Hyperdimensional computing is fast, noise-robust, and edge-friendly — but its
predictions are raw similarity scores with no notion of confidence.
**bayes-hdc** fixes that: calibrated probabilities, conformal prediction sets,
and anomaly detection with a *guaranteed* false-positive rate, all in JAX.

```bash
pip install bayes-hdc
```

## Anomaly detection with a guaranteed false-positive rate

Fit on normal data only. Flag outliers at a false-positive rate that holds by
theorem, not by threshold tuning — finite-sample, distribution-free.

<p align="center">
  <img src="assets/quickstart.gif" alt="Live demo: fit on normal data, FPR 0.049 against a 0.050 target, 50/50 anomalies caught" width="700">
</p>

```python
import numpy as np
from bayes_hdc.sklearn import HDAnomalyDetector

rng = np.random.default_rng(0)
X_normal = 5.0 + rng.normal(size=(500, 16))          # sensors around a setpoint
X_test   = np.vstack([5.0 + rng.normal(size=(50, 16)),   # healthy
                      rng.normal(size=(50, 16))])        # signal dropout

det = HDAnomalyDetector(alpha=0.05).fit(X_normal)
labels = det.predict(X_test)        # +1 inlier / -1 outlier; marginal FP rate <= alpha
pvals  = det.score_samples(X_test)  # split-conformal p-values
```

Against IsolationForest, LOF, and OneClassSVM it takes the best AUROC on two
of three one-class benchmarks — while holding the false-positive rate at
target, a knob none of those baselines have. The JAX-native pipeline
(custom encoders, batch FDR control) is in
[`tutorials/02_anomaly_detection.py`](tutorials/02_anomaly_detection.py).

## Calibrated probabilities and prediction sets

Hypervectors carry distributions (`GaussianHV`, `DirichletHV`) with
closed-form moment propagation. Any classifier's outputs wrap into
temperature-scaled probabilities and conformal sets:

```python
from bayes_hdc import TemperatureCalibrator, ConformalClassifier

probs = TemperatureCalibrator.create().fit(logits_cal, y_cal).calibrate(logits_test)

conformal = ConformalClassifier.create(alpha=0.1).fit(probs_cal, y_cal)
sets      = conformal.predict_set(probs)        # (n, k) bool mask
coverage  = conformal.coverage(probs, y_test)   # >= 1-alpha in expectation (marginal)
```

scikit-learn users get the whole thing as a drop-in estimator:

```python
from bayes_hdc.sklearn import HDClassifier

HDClassifier(encoder="kernel").fit(X_train, y_train).predict_proba(X_test)
```

## Benchmarks

Standard HDC datasets, 5 seeds, both encoders given the same bandwidth search
on identical splits. Full protocol: [BENCHMARKS.md](BENCHMARKS.md).

| Dataset | bayes-hdc accuracy | TorchHD accuracy (tuned) | ECE raw → calibrated | Coverage @ α=0.1 |
|---|---|---|---|---|
| ISOLET | **0.895 ± 0.004** | 0.882 ± 0.006 | 0.845 → **0.022** | 0.901 |
| UCI-HAR | 0.849 ± 0.006 | **0.871 ± 0.005** | 0.633 → **0.031** | 0.904 |
| EMG gestures | **0.944 ± 0.014** | 0.892 ± 0.005 | 0.618 → **0.045** | 0.947 |

Accuracy is competitive — ahead on two, behind on one, printed as-is. The
right columns are the point: calibration and coverage the deterministic
libraries don't provide. Every number reproduces via `make bench-canonical`.

## In the HDC library landscape

Eight VSA models (BSC, MAP, HRR, FHRR, BSBC, CGR, MCR, VTB) under one
API; the differentiation is the probabilistic layer.

| Library | Backend | VSA models | Probabilistic / UQ | Differentiable |
|---|---|---:|---|---|
| [TorchHD](https://github.com/hyperdimensional-computing/torchhd) | PyTorch | 8 | — | partial |
| [HoloVec](https://github.com/Twistient/HoloVec) | NumPy / PyTorch / JAX | 8 | — | partial |
| [hdlib](https://github.com/cumbof/hdlib) | NumPy | generic | — | — |
| [vsapy](https://github.com/vsapy/vsapy) | NumPy | 6 | — | — |
| [NengoSPA](https://github.com/nengo/nengo-spa) | Nengo (spiking) | 3 | — | — |
| **bayes-hdc** | **JAX** | **8** | **Gaussian/Dirichlet HVs, conformal classifier + regressor + anomaly detector** | **end-to-end** |

Design rationale and per-primitive paper attributions:
[`DESIGN.md`](DESIGN.md) · [`docs/LITERATURE_AUDIT.md`](docs/LITERATURE_AUDIT.md).

## Examples

| | |
|---|---|
| [`emg_gesture_recognition.py`](examples/emg_gesture_recognition.py) | sEMG gestures with calibrated per-gesture probabilities |
| [`anomaly_detection_intrusion.py`](examples/anomaly_detection_intrusion.py) | network intrusion flags at a guaranteed FP rate |
| [`vision_action_policy.py`](examples/vision_action_policy.py) | vision-action policy with per-DOF conformal intervals and abstention |
| [`kanerva_example.py`](examples/kanerva_example.py) | "What's the Dollar of Mexico?" role-filler analogy |

Sixteen more in [`examples/`](examples/README.md), two worked tutorials in
[`tutorials/`](tutorials/README.md).

## Status

Alpha (`0.5.0a1`); API may shift before 1.0. 666 tests, 93% coverage, CI on
Ubuntu + macOS × Python 3.9–3.13: algebraic laws on randomized inputs,
gradients vs finite differences, and the coverage/FDR guarantees tested
directly. Pure Python on `jax` + `numpy`, no compiled extensions. Sharp
edges: GPU/TPU tested on CPU CI only; the variational-training API is the
most likely to change.

## Contributing

[Good first issues](https://github.com/rlogger/bayes-hdc/labels/good%20first%20issue)
are scoped and mentored; setup in [`CONTRIBUTING.md`](CONTRIBUTING.md).
Questions and show-and-tell:
[Discussions](https://github.com/rlogger/bayes-hdc/discussions). If this is
useful to you, a star helps others find it.

## Citing

```bibtex
@software{bayeshdc2026,
  author  = {Singh, Rajdeep},
  title   = {bayes-hdc: Calibrated, Differentiable Hyperdimensional Computing in JAX},
  url     = {https://github.com/rlogger/bayes-hdc},
  doi     = {10.5281/zenodo.20635099},
  version = {0.5.0a1},
  year    = {2026}
}
```

Or the "Cite this repository" button ([`CITATION.cff`](CITATION.cff)).

## License

MIT. See also: [JAX](https://github.com/jax-ml/jax) ·
[TorchHD](https://github.com/hyperdimensional-computing/torchhd) ·
[awesome-jax](https://github.com/n2cholas/awesome-jax) ·
[Kleyko et al.'s HDC/VSA surveys](https://arxiv.org/abs/2111.06077).
