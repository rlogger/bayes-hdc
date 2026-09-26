# Bayes-HDC

**Probabilistic hyperdimensional computing in JAX.**

[![Main CI](https://github.com/rlogger/bayes-hdc/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/rlogger/bayes-hdc/actions/workflows/tests.yml)
[![PyPI release](https://img.shields.io/pypi/v/bayes-hdc)](https://pypi.org/project/bayes-hdc/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20635099.svg)](https://doi.org/10.5281/zenodo.20635099)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[Examples](examples/) · [Tutorials](tutorials/) ·
[Published documentation](https://rlogger.github.io/bayes-hdc/) ·
[Discussions](https://github.com/rlogger/bayes-hdc/discussions)

Bayes-HDC is a research library for hyperdimensional computing (HDC) and
vector symbolic architectures (VSAs). It combines binding, bundling and
structured representations with distribution-valued hypervectors, selected
Bayesian models, and held-out calibration for prediction sets, regression
intervals and anomaly p-values.

Use it to explore representation uncertainty and calibrated HDC pipelines in
JAX. The library is experimental: its current evidence does not establish
state-of-the-art accuracy, hardware efficiency or deployment reliability.

## Install this development branch

Python 3.9 or later is required. The `examples` extra includes scikit-learn
for the two self-contained examples below. From a terminal with Git:

```bash
git clone --branch fix/audit-and-citations https://github.com/rlogger/bayes-hdc.git
cd bayes-hdc
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[examples]"
```

This branch contains unreleased correctness and citation fixes. The published
PyPI/Zenodo release, `0.5.0a1`, and the published documentation predate these
changes. The examples below target this checkout; record its commit when
reporting an experiment.

## Anomaly detection with split-conformal calibration

Fit on normal observations. `HDAnomalyDetector.fit` reserves 30% of them for
calibration by default and fits its scoring model on the remaining 70%.
This toy example generates independent synthetic setpoint and dropout data:

```python
import numpy as np
from bayes_hdc.sklearn import HDAnomalyDetector

rng = np.random.default_rng(0)
X_normal = 5.0 + rng.normal(size=(500, 16))
X_test = np.vstack([
    5.0 + rng.normal(size=(50, 16)),  # New normal samples.
    rng.normal(size=(50, 16)),        # Synthetic signal dropout.
])
det = HDAnomalyDetector(alpha=0.05, random_state=0).fit(X_normal)
labels = det.predict(X_test)  # +1 inlier, -1 outlier.
pvals = det.pvalue(X_test)  # Small conformal p-values suggest anomalies.
```

With a scoring pipeline fixed before calibration and exchangeable normal
calibration/test observations, flagging when `p <= alpha` bounds the
**marginal probability** of a false positive by `alpha`. It does not bound
the observed fraction in every test batch or guarantee detection of arbitrary
anomalies. Distribution shift can invalidate the guarantee.

Learned preprocessing must also stay independent of calibration data. For
explicit splits, custom encoders and batch multiple-testing examples, see
the [JAX-native anomaly tutorial](tutorials/02_anomaly_detection.py).
Conformal calibration can wrap non-HDC scores too; comparisons should give
baselines the same calibration protocol.

## Temperature scaling and prediction sets

`HDClassifier.predict_proba` returns softmax-normalized similarity scores;
they are **not automatically calibrated**. Here, training, temperature
fitting, conformal calibration and testing use four disjoint splits. The
scaler is learned from training data only.

```python
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from bayes_hdc import ConformalClassifier, TemperatureCalibrator
from bayes_hdc.sklearn import HDClassifier

X, y = load_iris(return_X_y=True)
# Random row splits; no label-based selection of calibration or test rows.
train, temp, cal, test = np.split(
    np.random.default_rng(0).permutation(len(y)), [60, 90, 120]
)
scaler = StandardScaler().fit(X[train])
model = HDClassifier(encoder="kernel", dimensions=2048, random_state=0).fit(
    scaler.transform(X[train]), y[train]
)
y_index = np.searchsorted(model.classes_, y)  # Probability-column indices.


def logits(rows):
    # Log probabilities are valid logits for temperature scaling.
    return jnp.log(jnp.asarray(model.predict_proba(scaler.transform(X[rows]))))


temperature = TemperatureCalibrator.create().fit(logits(temp), y_index[temp])
conformal = ConformalClassifier.create(alpha=0.1).fit(
    temperature.calibrate(logits(cal)), y_index[cal]
)
test_probs = temperature.calibrate(logits(test))
sets = np.asarray(conformal.predict_set(test_probs))
print("First prediction set:", model.classes_[sets[0]])
print("Observed test coverage:", sets[np.arange(len(test)), y_index[test]].mean())
```

With the fitted predictor held fixed and exchangeable conformal-calibration
and future test observations, split conformal gives marginal coverage of at
least `1 - alpha`. This is not a guarantee for each class, individual input,
or realized test batch. Temperature scaling alone gives no distribution-free
coverage guarantee. The tiny Iris example demonstrates the workflow; its
printed coverage is an observation, not a validation of the theorem.

See [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html) for temperature
scaling, [Romano et al.](https://arxiv.org/abs/2006.02544) for adaptive
prediction sets, and [Bates et al.](https://arxiv.org/abs/2104.08279) for
conformal outlier testing.

## Representation uncertainty

`GaussianHV` and `DirichletHV` describe uncertainty in hypervector
representations. Selected operations propagate moments analytically under
their stated assumptions. Gaussian representations use diagonal covariance,
and their binding/bundling rules assume independent operands. Nonlinear
normalization and inversion use approximations, and Dirichlet binding is
heuristic. These representations do not automatically yield a
calibrated predictive distribution or a Bayesian posterior over model weights.

Start with the [probabilistic VSA example](examples/pvsa_quickstart.py) and
the [conjugate weight-space model](examples/weight_space_posterior.py).
[Design notes](DESIGN.md) and the [literature audit](docs/LITERATURE_AUDIT.md)
describe individual primitives and their sources.

## Benchmark status and reproduction

Earlier committed tables, JSON results and graphics require regeneration
after corrections to model identity, preprocessing and data splits. Treat
[BENCHMARKS.md](BENCHMARKS.md) and those artifacts as historical results,
not evidence for the corrected branch. No performance leaderboard is claimed
here until full experiments are rerun and their provenance is published.

The current [calibration benchmark](benchmarks/benchmark_calibration.py)
compares JAX and TorchHD normalized cosine-centroid implementations on the
**same encoded features**, checks score parity, and uses separate training,
temperature, conformal and test data. It saves per-example calibrated
probabilities, labels, split IDs and source/environment metadata. This is a
matched centroid comparison, not a comparison against every TorchHD
classifier or the strongest HDC methods.

From the checkout, generate fresh results and figures from those results:

```bash
python -m pip install -e ".[examples,benchmark]"
python benchmarks/benchmark_calibration.py --output results/calibration.json
python benchmarks/generate_figures.py --results results/calibration.json --output-dir results/figures
```

The default run uses four bundled scikit-learn datasets and one seed;
`--include-mnist` adds a network download. Publication experiments need
multiple seeds, matched tuning budgets, strong baselines, and uncertainty
metrics alongside accuracy and set/interval size. For subject-based or
temporal data, evaluate dependence and the intended generalization target
before applying exchangeability-based guarantees. Timing or accuracy on a
development machine does not establish edge-device efficiency.

## In the HDC library landscape

These projects cover different parts of the HDC/VSA ecosystem. The table
highlights selected documented capabilities to help choose tools for a task.

| Library / runtime | Use it for | Concrete tools |
|---|---|---|
| [TorchHD](https://torchhd.readthedocs.io/en/stable/classifiers.html)<br>PyTorch | Comparing HDC classifiers and encoders | OnlineHD, AdaptHD and other classifiers; feature encoders; [dataset loaders](https://torchhd.readthedocs.io/en/stable/datasets.html). |
| [hdlib](https://cumbof.github.io/hdlib/model/classification.html)<br>NumPy | HDC classification and feature selection | Forward/backward feature selection; binary and bipolar `Vector` / `Space` APIs. |
| [HoloVec](https://github.com/Twistient/HoloVec)<br>NumPy; optional PyTorch/JAX | Comparing VSA representations and retrieval methods | Scalar, sequence and spatial encoders; item stores; resonator cleanup. |
| [vsapy](https://github.com/vsapy/vsapy)<br>NumPy | Encoding sequences, documents and cyclic values | `CSPvec` hierarchies; linear/circular scalar encoders; JSON encoding demo. |
| [NengoSPA](https://www.nengo.ai/nengo-spa/v1.3.0/)<br>Nengo | Building cognitive models with spiking neural networks | Semantic pointers; associative memories; action selection and routing. |
| **bayes-hdc**<br>JAX | Research on representation uncertainty and held-out calibration | Gaussian/Dirichlet hypervectors; selected Bayesian models; temperature scaling and conformal wrappers. |

Related work includes [DiceHD's uncertainty estimates](https://yang-ni-yn.github.io/files/ICCAD-2023.pdf),
[probabilistic computation with VSAs](https://link.springer.com/article/10.1007/s11571-023-10031-7),
and conformal HDC for [neural decoding](https://arxiv.org/abs/2602.21446)
and [activity recognition on microcontrollers](https://www.nature.com/articles/s41598-026-57375-8).
Bayes-HDC focuses on combining distribution-valued hypervectors and
calibration tools in JAX.

## More examples

| Example | What it demonstrates |
|---|---|
| [EMG-style classification](examples/emg_gesture_recognition.py) | Synthetic gesture data with train-only quantization and held-out temperature scaling. |
| [Network-style anomaly detection](examples/anomaly_detection_intrusion.py) | Synthetic observations with separate normal training and calibration data. |
| [Continuous-action regression](examples/vision_action_policy.py) | Synthetic multimodal inputs, marginal per-coordinate intervals and an abstention heuristic; no robot evaluation or control-safety guarantee. |
| [Role-filler analogy](examples/kanerva_example.py) | The "Dollar of Mexico" analogy using binding and retrieval. |
| [Sequences](tutorials/03_sequences.py) | Flat and hierarchical sequence encoding, including the cached chunk representations used for retrieval. |

Browse the [examples](examples/) and [tutorials](tutorials/) for executable
workflows. Synthetic demonstrations do not establish clinical, security or
robotics performance.

## Development status and compatibility

The package is alpha and APIs may change. The audit fixes change FHRR and
VTB representations and several calibration contracts; regenerate affected
stored vectors and refit models/calibrators when migrating. VTB uses right
unbinding. The native anomaly calibrator requires an already-fitted scorer;
the sklearn adapter performs its own training/calibration split.

Validation has been performed on CPU. JIT and gradient support depend on the
operation; some helpers use host-side loops. GPU/TPU performance and physical
edge deployments remain unverified. Tests exercise selected algebraic,
numerical and statistical contracts; passing them is not a proof of every
guarantee. See [CI](https://github.com/rlogger/bayes-hdc/actions/workflows/tests.yml)
for the revision and environments of each run.

For development setup and checks, see [CONTRIBUTING.md](CONTRIBUTING.md).
Questions and experience reports are welcome in
[Discussions](https://github.com/rlogger/bayes-hdc/discussions).

## Citing

The following citation identifies the published `0.5.0a1` release, which
does not include this branch's fixes. For experiments with unreleased code,
also record the exact Git commit and environment.

```bibtex
@software{bayeshdc2026,
  author  = {Singh, Rajdeep},
  title   = {bayes-hdc: Calibrated, Differentiable Hyperdimensional Computing in JAX},
  url     = {https://github.com/rlogger/bayes-hdc},
  doi     = {10.5281/zenodo.20635123},
  version = {0.5.0a1},
  year    = {2026}
}
```

Release metadata is also available in [CITATION.cff](CITATION.cff).

## License

[MIT](LICENSE).
