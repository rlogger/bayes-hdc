<h1 align="center">bayes-hdc</h1>

<p align="center">
  <strong>A JAX library with serious algebraic depth.</strong><br/>
  Hypervectors as a pytree-native algebra. Closed-form moments. Group actions.<br/>
  Equivariant bilinear operators. Reparameterisation gradients end-to-end.
</p>

<p align="center">
  <a href="https://github.com/rlogger/bayes-hdc/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/rlogger/bayes-hdc/actions/workflows/tests.yml/badge.svg?branch=main" /></a>
  <a href="https://github.com/rlogger/bayes-hdc/actions/workflows/codeql.yml"><img alt="CodeQL" src="https://github.com/rlogger/bayes-hdc/actions/workflows/codeql.yml/badge.svg?branch=main" /></a>
  <a href="https://codecov.io/gh/rlogger/bayes-hdc"><img alt="Coverage" src="https://img.shields.io/badge/coverage-97%25-brightgreen.svg" /></a>
  <img alt="Tests" src="https://img.shields.io/badge/tests-475%20passing-brightgreen.svg" />
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12%20|%203.13-blue.svg" />
  <img alt="JAX" src="https://img.shields.io/badge/JAX-%E2%89%A5%200.4.20-orange.svg" />
  <a href="https://github.com/rlogger/bayes-hdc/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <img alt="Code style: ruff" src="https://img.shields.io/badge/code%20style-ruff-000000.svg" />
  <img alt="Type checked: mypy" src="https://img.shields.io/badge/type%20checked-mypy-1f5082.svg" />
</p>

<p align="center">
  <a href="DESIGN.md">Design notes</a> ·
  <a href="examples/">Examples</a> ·
  <a href="docs/workshop_paper.tex">Paper</a> ·
  <a href="ORIGINALITY.md">Originality</a> ·
  <a href="BENCHMARKS.md">Benchmarks</a> ·
  <a href="https://github.com/rlogger/bayes-hdc/discussions">Discussions</a>
</p>

---

## About

bayes-hdc is a JAX library for hyperdimensional computing (HDC) with a probabilistic layer on top — **PVSA**, Probabilistic Vector Symbolic Architectures. Every hypervector is an element of a small, well-designed algebra: a commutative binding, an associative bundling, a cyclic group action, a cosine measure, and a posterior distribution over the whole thing. Every type is a pytree. Every operation composes with `jit`, `vmap`, `grad`, `pmap`, `shard_map`. The implementation is careful, the API is small, the moments are closed-form, and the claims are theorems.

The library is legibly useful at the intersection of three active research programmes: **transformer weight-space learning**, **equivariant neural functionals (NFNs)**, and **meta-RL with structured representations**. See [`DESIGN.md`](DESIGN.md) for the long-form story.

### Highlights

- **Pytree-native.** `jit` / `vmap` / `grad` / `pmap` / `shard_map` compose with every operation unconditionally.
- **Closed-form algebra.** `bind_gaussian`, `bundle_gaussian`, `kl_gaussian`, `kl_dirichlet` are analytic. No Monte Carlo where math is enough.
- **First-class group actions.** `Z/d` cyclic shift exposed as a real group object, with property-based verifiers for shift equivariance and invariance.
- **Calibration & coverage out of the box.** Temperature scaling (Guo 2017) and split-conformal prediction (Romano 2020), each with formal guarantees.
- **Differentiable end-to-end.** Reparameterisation samplers on every distributional op; `jax.grad` composes through everything.
- **Scales.** From a laptop CPU to a TPU pod with the same code via `pmap` / `shard_map` wrappers.
- **Deterministic VSA foundation.** Eight classical VSA models (BSC, MAP, HRR, FHRR, BSBC, CGR, MCR, VTB) implemented from the primary papers; nothing ported.
- **475 tests, 97 % coverage.** Property-based tests for every algebraic identity. CI on Ubuntu + macOS × Python 3.9–3.13 on every push.

## Quick tour

### Probabilistic hypervectors

```python
import jax
from bayes_hdc import GaussianHV, bind_gaussian, expected_cosine_similarity

key = jax.random.PRNGKey(0)
x = GaussianHV.random(key, dimensions=10_000, var=0.01)
y = GaussianHV.random(jax.random.fold_in(key, 1), dimensions=10_000, var=0.01)

z   = bind_gaussian(x, y)                  # exact moment propagation
sim = expected_cosine_similarity(x, z)     # uncertainty-aware similarity
```

### Calibration and coverage on any classifier

```python
from bayes_hdc import TemperatureCalibrator, ConformalClassifier

calibrator = TemperatureCalibrator.create().fit(logits_cal, y_cal)
probs      = calibrator.calibrate(logits_test)

conformal  = ConformalClassifier.create(alpha=0.1).fit(probs_cal, y_cal)
sets       = conformal.predict_set(probs)                # (n, k) bool mask
coverage   = conformal.coverage(probs_test, y_test)      # ≥ 0.9 by construction
```

### Verify a custom op respects the cyclic group action

```python
from bayes_hdc import bind_map, verify_shift_equivariance

assert verify_shift_equivariance(bind_map, x, y)         # diagonal Z/d-equivariant
```

### Lift a deterministic pipeline into PVSA

```python
from bayes_hdc import GaussianHV

x_pvsa = GaussianHV.from_sample(x_classical)             # zero-variance posterior
# behaves identically to classical MAP until you inject uncertainty
```

## More about bayes-hdc

### A pytree-native algebra

Every type in the library is a frozen JAX pytree, registered via `jax.tree_util.register_dataclass`. `jit`, `vmap`, `grad`, `pmap`, and `shard_map` compose with `GaussianHV`, `DirichletHV`, `BayesianCentroidClassifier`, and every other public type without any user-side flattening or unflattening. The library is deliberately functional — immutable values, pure operations, no hidden state.

### Closed-form moment propagation

For independent Gaussian hypervectors, the first and second moments of bind and bundle are exact:

```
E[x · y]   = μ_x · μ_y
Var[x · y] = μ_x² σ_y² + μ_y² σ_x² + σ_x² σ_y²

E[Σ xᵢ]    = Σ μᵢ
Var[Σ xᵢ]  = Σ σᵢ²
```

`bind_gaussian` and `bundle_gaussian` return these analytically. `kl_gaussian` and `kl_dirichlet` are likewise closed form and differentiable end-to-end. Monte Carlo fallbacks exist where the math is not closed; they are explicit and reparameterised.

### First-class group actions

The cyclic-shift action `T_k` of `Z/d` on `R^d` — what `permute` *is* — is a faithful, additive, isometric group action. The `bayes_hdc.equivariance` module exposes it, distinguishes the two flavours of equivariance correctly (element-wise bind is diagonally equivariant; circular convolution is single-argument equivariant), and ships property-based verifiers that reject any user-defined op claiming a symmetry it does not have.

```python
from bayes_hdc import shift, hrr_equivariant_bilinear, verify_single_argument_shift_equivariance

assert verify_single_argument_shift_equivariance(hrr_equivariant_bilinear, x, filter_hv)
```

### Reparameterisation gradients end-to-end

Every distributional operation admits a differentiable reparameterisation sampler. `jax.grad` composes through `bind_gaussian`, `bundle_gaussian`, `cleanup_gaussian`, `inverse_gaussian`, `permute_gaussian`, `kl_gaussian`, and the ELBO helpers in `bayes_hdc.inference`. End-to-end variational training of codebooks and classifier posteriors is one `jax.grad` away.

### Calibration and coverage with formal guarantees

`TemperatureCalibrator` minimises the negative log-likelihood over a one-parameter temperature via L-BFGS in log-space. Convex objective, unique global minimum, the fitted temperature is the maximum-likelihood estimator. `ConformalClassifier` uses split-conformal with APS scores (Romano et al. 2020) and returns prediction sets whose marginal coverage satisfies `P(y ∈ set(x)) ≥ 1 − α` on exchangeable data — independent of model, dimension, or training quality.

### Scales from laptop to pod

Single-device wrappers degrade gracefully on multi-device hosts via `pmap_bind_gaussian`, `pmap_bundle_gaussian`, `shard_map_bind_gaussian`, and `shard_classifier_posteriors`. The same code runs on a laptop CPU and on a TPU pod. `StreamingBayesianHDC` keeps EMA posteriors in bounded memory for non-stationary streams.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Applications                                                                │
│   language identification · selective classification · OOD detection        │
│   sequence memory · weight-space posteriors                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ Uncertainty                                                                 │
│   ConformalClassifier · TemperatureCalibrator · posterior_predictive_check  │
├─────────────────────────────────────────────────────────────────────────────┤
│ Bayesian models                                                             │
│   BayesianCentroidClassifier · BayesianAdaptiveHDC · StreamingBayesianHDC   │
├─────────────────────────────────────────────────────────────────────────────┤
│ PVSA core                                                                   │
│   GaussianHV · DirichletHV · MixtureHV                                      │
│   bind_gaussian · bundle_gaussian · permute_gaussian · cleanup_gaussian     │
│   inverse_gaussian · kl_gaussian · kl_dirichlet                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ Group structure                                                             │
│   shift · compose_shifts · hrr_equivariant_bilinear                         │
│   verify_shift_equivariance · verify_single_argument_shift_equivariance     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Classical VSA                                                               │
│   BSC · MAP · HRR · FHRR · BSBC · CGR · MCR · VTB                           │
│   five encoders · five classifiers · three memory modules                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ JAX                                                                         │
│   pytree · jit · vmap · grad · pmap · shard_map · CPU / GPU / TPU           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install -e .                 # core
pip install -e ".[examples]"     # + matplotlib + scikit-learn (for the application examples)
pip install -e ".[dev]"          # + pytest, ruff, mypy
```

### Compatibility

| Component | Supported versions |
|---|---|
| Python | 3.9, 3.10, 3.11, 3.12, 3.13 |
| JAX    | ≥ 0.4.20 |
| OS     | Linux (Ubuntu), macOS |
| Hardware | CPU, GPU (CUDA via JAX), TPU |

The library is pure Python on top of JAX. There are no compiled extensions, no C++ build steps, and no transitive dependencies beyond `jax`, `jaxlib`, and `numpy`. `matplotlib` and `scikit-learn` are extras for the examples only.

## Examples

```bash
pip install -e ".[examples]"
python examples/<name>.py
```

### Research-connection demos

| Example | What it shows |
|---|---|
| [`weight_space_posterior.py`](examples/weight_space_posterior.py) | A classifier's weights are a `GaussianHV` posterior — a distribution over weight vectors. Sample from it, predict with each draw, read off epistemic uncertainty, verify `Z/d`-equivariance of the whole pipeline. |
| [`pvsa_quickstart.py`](examples/pvsa_quickstart.py) | 90-second tour through every PVSA primitive end-to-end. |

### PVSA applications

| Example | What it shows |
|---|---|
| [`language_identification.py`](examples/language_identification.py) | Character-trigram language ID with calibrated probabilities and conformal sets that grow on ambiguous input. |
| [`medical_selective_prediction.py`](examples/medical_selective_prediction.py) | Conformal-gated abstention on Breast Cancer Wisconsin — predict or hand off to follow-up. |
| [`anomaly_detection.py`](examples/anomaly_detection.py) | Posterior-Mahalanobis OOD detection on UCI digits. |
| [`sequence_memory.py`](examples/sequence_memory.py) | A 12-token sentence encoded as one hypervector, retrieved per position via un-permute and cleanup. |

### Classical HDC

| Example | What it shows |
|---|---|
| [`song_matching.py`](examples/song_matching.py) | Bag-of-words song similarity; the sum of word hypervectors is legible by eye. |
| [`kanerva_example.py`](examples/kanerva_example.py) | "Dollar of Mexico" — role-filler binding and analogical reasoning. |
| [`basic_operations.py`](examples/basic_operations.py) | bind / bundle / permute / similarity across all eight VSA models. |
| [`classification_simple.py`](examples/classification_simple.py) | Vanilla `RandomEncoder` + `CentroidClassifier` pipeline. |

## Project status

**Alpha — `0.4.0a0`.** API may shift before `1.0`.

| | |
|---|---|
| **Tests** | 475 passing, 2 skipped (network-gated dataset loaders) |
| **Coverage** | 97 % line coverage |
| **Lint** | `ruff check`, `ruff format --check`, `mypy` clean on every push |
| **CI** | Ubuntu + macOS × Python 3.9–3.13 |
| **Security** | CodeQL on a weekly schedule; Dependabot weekly bumps |
| **Release** | Tag `vX.Y.Z` triggers TestPyPI then PyPI publish via OIDC |

See [`CHANGELOG.md`](CHANGELOG.md) for what's shipped and [`DESIGN.md`](DESIGN.md) for the design rationale.

## Community and contributing

- **Questions** — [GitHub Discussions → Q&A](https://github.com/rlogger/bayes-hdc/discussions/categories/q-a)
- **Bugs** — [open an issue](https://github.com/rlogger/bayes-hdc/issues/new?template=bug_report.yml) with a reproducer
- **Feature ideas** — [GitHub Discussions → Ideas](https://github.com/rlogger/bayes-hdc/discussions/categories/ideas)
- **Security** — see [`SECURITY.md`](SECURITY.md); do not open a public issue
- **Contributing** — read [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, style, and release process. All interactions are governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Citation

If you use bayes-hdc in research, please cite:

```bibtex
@software{bayes_hdc,
  author = {Singh, Rajdeep},
  title  = {{bayes-hdc: Probabilistic Vector Symbolic Architectures
             for Hyperdimensional Computing}},
  year   = {2026},
  url    = {https://github.com/rlogger/bayes-hdc},
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is at the repository root; the GitHub "Cite this repository" button reads from it. See [`ORIGINALITY.md`](ORIGINALITY.md) for per-component primary-source attribution.

## References

The library implements every component directly from the primary research paper. Selected core references:

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

[MIT](LICENSE).
