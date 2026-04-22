<h1 align="center">bayes-hdc</h1>

<p align="center">
  <strong>Probabilistic Vector Symbolic Architectures for JAX.</strong><br/>
  <em>Hypervectors are posteriors. Operations propagate them in closed form. Everything jits, vmaps, grads, pmaps.</em>
</p>

<p align="center">
  <a href="https://github.com/rlogger/bayes-hdc/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/rlogger/bayes-hdc/actions/workflows/tests.yml/badge.svg?branch=main" /></a>
  <a href="https://codecov.io/gh/rlogger/bayes-hdc"><img alt="Coverage" src="https://img.shields.io/badge/coverage-97%25-brightgreen.svg" /></a>
  <a href="https://github.com/rlogger/bayes-hdc/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg" />
  <img alt="JAX" src="https://img.shields.io/badge/JAX-%E2%89%A5%200.4.20-orange.svg" />
  <img alt="Tests" src="https://img.shields.io/badge/tests-467%20passing-brightgreen.svg" />
</p>

<p align="center">
  <a href="#why-hdc-at-all">Why HDC</a> ·
  <a href="#why-this-hdc">Why this HDC</a> ·
  <a href="#thirty-seconds">30s</a> ·
  <a href="#guarantees">Guarantees</a> ·
  <a href="#whats-inside">Inside</a> ·
  <a href="#examples">Examples</a> ·
  <a href="ORIGINALITY.md">Originality</a>
</p>

---

## Why HDC at all?

Neural networks are black boxes with 12 GB of floats inside. Hyperdimensional computing isn't.

In HDC, concepts are high-dimensional random vectors — typically 10 000 dimensions — and you compose them with algebra. Multiply two vectors to tie concepts together (`bind`). Add vectors to mix concepts (`bundle`). Shift indices to encode order (`permute`). That is the entire architecture. There is no hidden layer. There is no learned feature. The codebook has one vector per concept and you can point at it.

What that gets you that a neural net doesn't:

- **Transparent weights.** A classifier's "weights" are class centroids — one hypervector per class, each one a sum of the training examples that belong to it. You can decompose them, inspect them, edit them. No probing, no saliency, no mechanistic interpretability needed.
- **Symbolic composition.** New concepts are built from old ones by algebra, not training. `bind(dog, brown) + bind(cat, black)` is a two-row key-value store. Query with `bind(?, brown)` and you recover `dog` by similarity.
- **One-shot learning.** A new class is a new vector. Add it to the centroid bank. There is no gradient step, no retraining, no warm-up.
- **No catastrophic forgetting.** Adding a concept doesn't perturb the existing ones — 10 000-dim random vectors are nearly orthogonal, so new entries don't collide with old ones.
- **Noise-robust by construction.** Flip 30 % of a hypervector's dimensions and its cosine similarity to the clean version barely moves. Representations survive sensor noise, bit flips, quantisation.
- **Edge-feasible.** 10 000 dims × 1 byte = 10 KB per concept. Inference is a batch of dot products. HDC runs on microcontrollers and neuromorphic chips where a transformer would not boot.
- **No backprop.** Centroid classifiers fit in one pass over the data. Training is a matrix sum. Differentiable training is still available when you want it — this library gives you reparameterisation gradients through every op — but it's not the only way to learn.

What you give up: the universal approximation of deep nets. HDC is the right tool when you want *structure*, *interpretability*, and *deployability* more than you want the last 2 % on ImageNet.

## Why this HDC?

Classical HDC is fast, compositional, and blind. It hands you one vector and one answer. No confidence. No calibration. No idea when it is wrong.

**PVSA** fixes that at the level of the algebra. Every hypervector is a posterior distribution — a mean and a per-dimension variance. Every operation (bind, bundle, permute, cleanup, inverse, resonator) propagates that posterior in closed form. The result is HDC that *tells you what it knows and proves what it claims*.

This is the library for probabilistic HDC. No port. No wrapper. No magic. Pure JAX, pure pytrees, every claim a theorem verified in tests. See [`ORIGINALITY.md`](ORIGINALITY.md) — nothing here is lifted from another HDC package.

## Thirty seconds

```python
import jax
from bayes_hdc import GaussianHV, bind_gaussian, expected_cosine_similarity

key = jax.random.PRNGKey(0)
x = GaussianHV.random(key, dimensions=10_000, var=0.01)
y = GaussianHV.random(jax.random.fold_in(key, 1), dimensions=10_000, var=0.01)

z   = bind_gaussian(x, y)                  # exact moment propagation
sim = expected_cosine_similarity(x, z)     # uncertainty-aware similarity
```

Wrap any classifier — PVSA, classical HDC, or anything with logits — in calibration and coverage guarantees:

```python
from bayes_hdc import TemperatureCalibrator, ConformalClassifier

calibrator = TemperatureCalibrator.create().fit(logits_cal, y_cal)
probs      = calibrator.calibrate(logits_test)

conformal  = ConformalClassifier.create(alpha=0.1).fit(probs_cal, y_cal)
sets       = conformal.predict_set(probs)                    # (n, k) bool mask
coverage   = conformal.coverage(probs_test, y_test)          # ≥ 0.9 by construction
```

Deterministic pipelines lift into PVSA with `GaussianHV.from_sample(hv)` — a zero-variance posterior that behaves identically to classical MAP until you start injecting uncertainty. Nothing has to be rewritten.

## Guarantees

Not benchmarks. Theorems. These hold *by construction* on *every* input, independent of dataset, dimension, or training quality.

**Moment-exact algebra.** `bind_gaussian` returns the exact first and second moments of the element-wise product of two independent Gaussian hypervectors. Closed form, no Monte Carlo:

```
E[x · y]   = μ_x · μ_y
Var[x · y] = μ_x² σ_y² + μ_y² σ_x² + σ_x² σ_y²
```

Same story for `bundle_gaussian`, `permute_gaussian`, `kl_gaussian`, and their Dirichlet counterparts.

**Coverage ≥ 1 − α.** `ConformalClassifier` returns prediction sets whose marginal coverage satisfies `P(y ∈ set(x)) ≥ 1 − α` on exchangeable data. Holds for any underlying classifier. Holds in any dimension. Holds independently of how well the model was trained. This is the split-conformal guarantee with APS scores (Romano et al. 2020).

**Convex calibration.** `TemperatureCalibrator` minimises the NLL over a one-parameter temperature via L-BFGS in log-space. Convex objective, unique global minimum, the fitted temperature is the MLE.

**End-to-end differentiable.** Every distributional op admits a reparameterisation sampler. `jax.grad` composes through `bind_gaussian`, `bundle_gaussian`, `cleanup_gaussian`, `inverse_gaussian`, and the ELBO helpers in `bayes_hdc.inference`. Train codebooks and classifier posteriors variationally out of the box.

**Scales.** Every type is a JAX pytree. `jit`, `vmap`, `grad`, `pmap`, `shard_map` compose unconditionally. `pmap_bind_gaussian`, `shard_map_bind_gaussian`, and `shard_classifier_posteriors` are there for pod-scale training.

## What's inside

| Layer | Contents |
|---|---|
| **Probabilistic core** | `GaussianHV`, `DirichletHV`, `MixtureHV` · exact `bind_*`, `bundle_*`, `permute_*`, `cleanup_*`, `inverse_*`, `kl_*` · reparameterisation gradients everywhere |
| **VSA models** | BSC, MAP, HRR, FHRR, BSBC, CGR, MCR, VTB — shared `bind`, `bundle`, `inverse`, `similarity`, `random` API |
| **Encoders** | `RandomEncoder`, `LevelEncoder`, `ProjectionEncoder`, `KernelEncoder` (RFF), `GraphEncoder` |
| **Classifiers** | `CentroidClassifier`, `AdaptiveHDC`, `LVQClassifier`, `RegularizedLSClassifier`, `BayesianCentroidClassifier`, `BayesianAdaptiveHDC`, `StreamingBayesianHDC` |
| **Uncertainty** | `TemperatureCalibrator`, `ConformalClassifier`, `posterior_predictive_check`, `coverage_calibration_check` |
| **Memory** | `SparseDistributedMemory` (Kanerva), `HopfieldMemory` (modern), `AttentionMemory` (multi-head) |
| **Inference** | `elbo_gaussian`, `reconstruction_log_likelihood_mc`, `probabilistic_resonator` (MCMC multi-restart) |
| **Scale** | `pmap_bind_gaussian`, `shard_map_bind_gaussian`, `shard_classifier_posteriors` |
| **Datasets** | `iris`, `wine`, `breast_cancer`, `digits`, `mnist`, `fashion_mnist`, `isolet`, `ucihar`, `emg`, `pamap2`, `european_languages` |

Every component implemented directly from the primary paper (Kanerva 1988 / 1997 / 2009; Plate 1995, 2003; Gayler 2003; Rahimi & Recht 2007; Guo 2017; Romano 2020; Ramsauer 2020; Kleyko 2022). Nothing ported.

## Install

```bash
pip install -e .                 # core
pip install -e ".[examples]"     # + matplotlib + scikit-learn for the application examples
pip install -e ".[dev]"          # + pytest, ruff, mypy
```

## Examples

```bash
pip install -e ".[examples]"
python examples/<name>.py
```

### Fun first

[**`song_matching.py`**](examples/song_matching.py) — eight pseudo-songs, four themes, bag-of-words hypervectors. The code has 30 lines of actual logic. Songs with overlapping words end up in the same ballpark; you can read off *which* shared words drove every match. This is HDC at its most transparent — a sum of vectors, inspectable by eye, no training at all.

```
midnight_drive      → summer_road_trip   cos=0.487   shared: driving, highway, road, window
heartbreak_ballad   → lost_love          cos=0.741   shared: broken, crying, heart, lost, missing, tears
dance_floor         → party_anthem       cos=0.753   shared: dance, floor, lights, music, night, party
lullaby             → goodnight_song     cos=0.877   shared: baby, dream, moon, night, quiet, sleep, soft
```

### Everything else

| Example | What it shows |
|---|---|
| [`pvsa_quickstart.py`](examples/pvsa_quickstart.py) | 90-second tour through every PVSA primitive end-to-end. |
| [`language_identification.py`](examples/language_identification.py) | Character-trigram language ID over five European languages with calibrated probabilities and conformal sets that grow on ambiguous input. |
| [`medical_selective_prediction.py`](examples/medical_selective_prediction.py) | Conformal-gated abstention on Breast Cancer Wisconsin — predict or hand off to follow-up. |
| [`anomaly_detection.py`](examples/anomaly_detection.py) | Posterior-Mahalanobis OOD detection on UCI digits. A signal impossible without probabilistic hypervectors. |
| [`sequence_memory.py`](examples/sequence_memory.py) | A 12-token sentence encoded as one hypervector, retrieved per position via un-permute and cleanup. |
| [`kanerva_example.py`](examples/kanerva_example.py) | Dollar of Mexico — role-filler binding and analogical reasoning. |
| [`basic_operations.py`](examples/basic_operations.py) | bind / bundle / permute / similarity across all eight VSA models. |
| [`classification_simple.py`](examples/classification_simple.py) | Vanilla `RandomEncoder` + `CentroidClassifier` pipeline. |

## Status

**Alpha.** v0.2 through v1.0 shipped — Gaussian and Dirichlet posteriors, conformal prediction, temperature calibration, probabilistic resonator, posterior predictive checks, streaming Bayesian updates, multi-device sharding, 11 standard HDC datasets, workshop paper, containerised benchmarks. 461 tests, 97 % line coverage, Ubuntu + macOS × Python 3.9–3.13 on every push.

API may shift before 1.0.

## Development

```bash
pytest tests/ -v                                  # tests
pytest tests/ --cov=bayes_hdc --cov-report=html   # tests + coverage
ruff check bayes_hdc/ && ruff format bayes_hdc/   # lint + format
mypy bayes_hdc/                                   # types
make bench                                        # benchmarks
make figures                                      # paper figures
make docker-bench                                 # benchmarks in a container
```

## Citing

```bibtex
@software{bayes_hdc,
  author = {Singh, Rajdeep},
  title  = {{bayes-hdc: Probabilistic Vector Symbolic Architectures for Hyperdimensional Computing}},
  year   = {2026},
  url    = {https://github.com/rlogger/bayes-hdc},
}
```

## References

- Kanerva (2009). *Hyperdimensional Computing.*
- Plate (1995). *Holographic Reduced Representations.*
- Gayler (2003). *Vector Symbolic Architectures answer Jackendoff's challenges.*
- Joshi, Halseth, Kanerva (2016). *Language Geometry using Random Indexing.*
- Guo et al. (2017). *On Calibration of Modern Neural Networks.*
- Hendrycks & Gimpel (2017). *A Baseline for Detecting Misclassified and OOD Examples.*
- Romano et al. (2020). *Classification with Valid and Adaptive Coverage.*
- Ramsauer et al. (2020). *Hopfield Networks is All You Need.*
- Kleyko et al. (2022). *A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures.*

## License

MIT.
