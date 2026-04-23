<h1 align="center">bayes-hdc</h1>

<p align="center">
  <strong>A JAX library with serious algebraic depth.</strong><br/>
  <em>Hypervectors as a pytree-native algebra. Closed-form moments. Group actions. Equivariant bilinear operators. Reparameterisation gradients end-to-end.</em>
</p>

<p align="center">
  <a href="https://github.com/rlogger/bayes-hdc/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/rlogger/bayes-hdc/actions/workflows/tests.yml/badge.svg?branch=main" /></a>
  <a href="https://codecov.io/gh/rlogger/bayes-hdc"><img alt="Coverage" src="https://img.shields.io/badge/coverage-97%25-brightgreen.svg" /></a>
  <a href="https://github.com/rlogger/bayes-hdc/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg" />
  <img alt="JAX" src="https://img.shields.io/badge/JAX-%E2%89%A5%200.4.20-orange.svg" />
  <img alt="Tests" src="https://img.shields.io/badge/tests-475%20passing-brightgreen.svg" />
</p>

<p align="center">
  <a href="#what-this-is">What</a> ·
  <a href="#the-algebra">Algebra</a> ·
  <a href="#library-craft">Craft</a> ·
  <a href="#research-connections">Research</a> ·
  <a href="#thirty-seconds">30s</a> ·
  <a href="#guarantees">Guarantees</a> ·
  <a href="#examples">Examples</a> ·
  <a href="DESIGN.md">Design</a>
</p>

---

## What this is

A JAX library for hyperdimensional computing — with a probabilistic layer on top — that takes the mathematical structure seriously.

Every hypervector is an element of a small, well-designed algebra: a commutative binding, an associative bundling, a cyclic group action, a cosine measure, and a posterior distribution over the whole thing. Every type is a pytree. Every operation composes with `jit`, `vmap`, `grad`, `pmap`, `shard_map`. The implementation is careful; the API is small; the moments are closed-form; the claims are theorems.

The framework sits naturally at the intersection of three threads in current ML research, and that intersection is the reason this library exists.

## The algebra

A VSA is a compact algebraic object on :math:`\mathbb{R}^d`:

| Primitive | Signature | Law |
|---|---|---|
| `bind`       | `(R^d, R^d) → R^d`        | commutative, associative, invertible |
| `bundle`     | `(R^d)^n → R^d`           | commutative, associative |
| `permute`    | `R^d × Z → R^d`           | faithful action of `Z/d`, isometric |
| `similarity` | `(R^d, R^d) → R`          | cosine on the unit sphere |

**PVSA** lifts this to measures. A `GaussianHV` is a Dirac point when the variance is zero and a full posterior otherwise; all four primitives lift from `R^d` to `P(R^d)` in closed form, preserving the algebraic laws:

```python
bind_gaussian(x, y).mu  = x.mu * y.mu
bind_gaussian(x, y).var = x.mu**2 * y.var + y.mu**2 * x.var + x.var * y.var
```

These are not approximations. They are the exact first and second moments of the product distribution — which is what "closed form" means.

The group-theoretic structure is not folklore; it is first-class. `bayes_hdc.equivariance` exposes the cyclic-shift action, the single-argument vs. diagonal equivariances of the primitives, and property-based verifiers that reject any user-defined op that claims a symmetry it does not have.

## Library craft

- **Every type is a JAX pytree.** `jit`, `vmap`, `grad`, `pmap`, `shard_map` compose unconditionally, without ceremony at the call site. `@register_dataclass` is applied where it needs to be.
- **Immutable and functional.** All operations return new values. No in-place state. No hidden mutation.
- **Closed-form where possible, Monte Carlo where not.** `bind_gaussian`, `bundle_gaussian`, `kl_gaussian`, `kl_dirichlet` are analytic. Sampling fallbacks are explicit and reparameterised.
- **Small, typed API.** Every public function has a type signature and a docstring. No untyped kwargs. No surprise tensor shapes.
- **475 tests, 97 % coverage.** Property-based tests for algebraic identities (commutativity, associativity, inverse, equivariance, isometry). Shape tests, dtype tests, grad tests.
- **Zero transitive dependencies beyond JAX + numpy.** `matplotlib` and `scikit-learn` are extras for examples only.
- **CI on every push.** Ubuntu + macOS × Python 3.9–3.13. Ruff, mypy, pytest. CodeQL scheduled. Dependabot weekly.

See [`DESIGN.md`](DESIGN.md) for the long-form story.

## Research connections

The design is legible to three research programmes, each of which cares about the kind of thing this library provides by default.

### 1. Transformer weight-space research

Papers that treat a network's weights as data — model-zoo hypernets, permutation-invariant weight encoders, functa — ask for a representation that is typed, symmetry-respecting, and distribution-valued. A `BayesianCentroidClassifier` here stores `K` class hypervectors, each one a `GaussianHV` with mean `mu_c` and per-dimension variance `var_c`. That is literally a distribution over weight vectors.

```python
clf = BayesianCentroidClassifier.create(num_classes=K, dimensions=d).fit(X, y)
clf.mu       # (K, d)  — posterior mean weight matrix
clf.var      # (K, d)  — posterior variance
weight_sample = clf.mu + jax.random.normal(key, clf.mu.shape) * jnp.sqrt(clf.var)
```

[`examples/weight_space_posterior.py`](examples/weight_space_posterior.py) walks through this explicitly: draw from the posterior, predict with each draw, read off epistemic uncertainty as disagreement across draws, and verify that the whole pipeline commutes with the cyclic-shift symmetry of the representation.

### 2. Equivariant neural functionals (NFNs)

The NFN programme builds layers that respect the symmetries of weight-space. Cyclic shift is a `Z/d` action on hypervectors; element-wise-product binding is diagonally equivariant; circular-convolution binding is single-argument equivariant. `bayes_hdc.equivariance` makes all of this explicit, and ships `verify_shift_equivariance` / `verify_single_argument_shift_equivariance` so you can check at test-time that your custom op respects the symmetry it claims.

```python
from bayes_hdc import shift, verify_shift_equivariance, hrr_equivariant_bilinear

# Element-wise bind is diagonally Z/d-equivariant.
assert verify_shift_equivariance(bind_map, x, y)

# Circular convolution is the canonical single-argument equivariant bilinear operator.
z = hrr_equivariant_bilinear(x, filter_hv)
```

### 3. Meta-RL with structured representations

Task-conditioned agents that generalise systematically need a composable representation of "(task, state)". HDC gives you exactly that: `bind(task_hv, state_hv)` is a structured pair whose constituents can be unbound by similarity, and whose symmetries are inherited from the operands. `BayesianAdaptiveHDC` gives per-arm streaming posteriors for contextual bandits; `StreamingBayesianHDC` gives bounded-memory EMA posteriors for non-stationary environments; the posterior Mahalanobis distance from `BayesianCentroidClassifier` is a drop-in novelty / intrinsic-reward signal.

The reframe is this: **HDC is a structured-representation substrate, and PVSA is a weight-space algebra with uncertainty.** The rest is engineering.

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

Wrap any classifier in calibration and coverage guarantees:

```python
from bayes_hdc import TemperatureCalibrator, ConformalClassifier

calibrator = TemperatureCalibrator.create().fit(logits_cal, y_cal)
probs      = calibrator.calibrate(logits_test)

conformal  = ConformalClassifier.create(alpha=0.1).fit(probs_cal, y_cal)
sets       = conformal.predict_set(probs)                    # (n, k) bool mask
coverage   = conformal.coverage(probs_test, y_test)          # ≥ 0.9 by construction
```

Verify a custom op respects the cyclic group action:

```python
from bayes_hdc import verify_shift_equivariance

def my_layer(x, y):
    ...

assert verify_shift_equivariance(my_layer, x, y)
```

Deterministic pipelines lift into PVSA with `GaussianHV.from_sample(hv)` — a zero-variance posterior that behaves identically to classical MAP until you inject uncertainty. Nothing has to be rewritten.

## Guarantees

Not benchmarks. Theorems. These hold *by construction* on *every* input, independent of dataset, dimension, or training quality.

**Moment-exact algebra.** `bind_gaussian` returns the exact first and second moments of the element-wise product of two independent Gaussian hypervectors. Closed form, no Monte Carlo:

```
E[x · y]   = μ_x · μ_y
Var[x · y] = μ_x² σ_y² + μ_y² σ_x² + σ_x² σ_y²
```

Same story for `bundle_gaussian`, `permute_gaussian`, `kl_gaussian`, and their Dirichlet counterparts.

**Group-theoretic correctness.** The cyclic-shift action `T_k` is faithful, additive (`T_j ∘ T_k = T_{j+k}`), and isometric (`⟨T_k(x), T_k(y)⟩ = ⟨x, y⟩`). Element-wise binding is diagonally `Z/d`-equivariant; circular convolution is single-argument equivariant; cosine similarity is diagonally invariant. Every claim is in `tests/test_equivariance.py` as a property-based test.

**Coverage ≥ 1 − α.** `ConformalClassifier` returns prediction sets whose marginal coverage satisfies `P(y ∈ set(x)) ≥ 1 − α` on exchangeable data. Holds for any underlying classifier. Holds in any dimension. Holds independently of how well the model was trained.

**Convex calibration.** `TemperatureCalibrator` minimises the NLL over a one-parameter temperature via L-BFGS in log-space. Convex objective, unique global minimum, the fitted temperature is the MLE.

**End-to-end differentiable.** Every distributional op admits a reparameterisation sampler. `jax.grad` composes through `bind_gaussian`, `bundle_gaussian`, `cleanup_gaussian`, `inverse_gaussian`, and the ELBO helpers in `bayes_hdc.inference`.

**Scales.** Every type is a JAX pytree. `jit`, `vmap`, `grad`, `pmap`, `shard_map` compose unconditionally. `pmap_bind_gaussian`, `shard_map_bind_gaussian`, `shard_classifier_posteriors` for pod-scale training.

## What's inside

| Layer | Contents |
|---|---|
| **Probabilistic core** | `GaussianHV`, `DirichletHV`, `MixtureHV` · exact `bind_*`, `bundle_*`, `permute_*`, `cleanup_*`, `inverse_*`, `kl_*` · reparameterisation gradients everywhere |
| **Group structure** | `shift`, `compose_shifts`, `hrr_equivariant_bilinear`, `verify_shift_equivariance`, `verify_single_argument_shift_equivariance`, `verify_shift_invariance` |
| **VSA models** | BSC, MAP, HRR, FHRR, BSBC, CGR, MCR, VTB — shared `bind`, `bundle`, `inverse`, `similarity`, `random` API |
| **Encoders** | `RandomEncoder`, `LevelEncoder`, `ProjectionEncoder`, `KernelEncoder` (RFF), `GraphEncoder` |
| **Classifiers** | `CentroidClassifier`, `AdaptiveHDC`, `LVQClassifier`, `RegularizedLSClassifier`, `BayesianCentroidClassifier`, `BayesianAdaptiveHDC`, `StreamingBayesianHDC` |
| **Uncertainty** | `TemperatureCalibrator`, `ConformalClassifier`, `posterior_predictive_check`, `coverage_calibration_check` |
| **Memory** | `SparseDistributedMemory` (Kanerva), `HopfieldMemory` (modern), `AttentionMemory` (multi-head) |
| **Inference** | `elbo_gaussian`, `reconstruction_log_likelihood_mc`, `probabilistic_resonator` (MCMC multi-restart) |
| **Scale** | `pmap_bind_gaussian`, `shard_map_bind_gaussian`, `shard_classifier_posteriors` |
| **Datasets** | `iris`, `wine`, `breast_cancer`, `digits`, `mnist`, `fashion_mnist`, `isolet`, `ucihar`, `emg`, `pamap2`, `european_languages` |

Every component implemented directly from the primary paper (Kanerva 1988 / 1997 / 2009; Plate 1995, 2003; Gayler 2003; Rahimi & Recht 2007; Guo 2017; Romano 2020; Ramsauer 2020; Kleyko 2022). Nothing ported from another HDC library.

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

### Research-connection demos

| Example | What it shows |
|---|---|
| [`weight_space_posterior.py`](examples/weight_space_posterior.py) | A classifier's weights are a `GaussianHV` posterior. Sample from the posterior, predict with each draw, read off epistemic uncertainty, verify `Z/d`-equivariance of the whole pipeline. |
| [`pvsa_quickstart.py`](examples/pvsa_quickstart.py) | 90-second tour through every PVSA primitive end-to-end — construction, bind/bundle moment propagation, KL, conformal coverage. |

### Applications

| Example | What it shows |
|---|---|
| [`language_identification.py`](examples/language_identification.py) | Character-trigram language ID with calibrated probabilities and conformal sets that grow on ambiguous input. |
| [`medical_selective_prediction.py`](examples/medical_selective_prediction.py) | Conformal-gated abstention on Breast Cancer Wisconsin — predict or hand off to follow-up. |
| [`anomaly_detection.py`](examples/anomaly_detection.py) | Posterior-Mahalanobis OOD detection on UCI digits. A signal impossible without probabilistic hypervectors. |
| [`sequence_memory.py`](examples/sequence_memory.py) | A 12-token sentence encoded as one hypervector, retrieved per position via un-permute and cleanup. |

### Classical HDC

| Example | What it shows |
|---|---|
| [`song_matching.py`](examples/song_matching.py) | Bag-of-words song similarity; the sum of word hypervectors is legible by eye. |
| [`kanerva_example.py`](examples/kanerva_example.py) | Dollar of Mexico — role-filler binding and analogical reasoning. |
| [`basic_operations.py`](examples/basic_operations.py) | bind / bundle / permute / similarity across all eight VSA models. |
| [`classification_simple.py`](examples/classification_simple.py) | Vanilla `RandomEncoder` + `CentroidClassifier` pipeline. |

## Status

**Alpha.** v0.2 through v1.0 shipped — Gaussian and Dirichlet posteriors, equivariance module, conformal prediction, temperature calibration, probabilistic resonator, posterior predictive checks, streaming Bayesian updates, multi-device sharding, 11 standard HDC datasets, workshop paper, containerised benchmarks. 475 tests, 97 % line coverage, Ubuntu + macOS × Python 3.9–3.13 on every push.

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
