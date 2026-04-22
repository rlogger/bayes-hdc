<h1 align="center">bayes-hdc</h1>

<p align="center">
  <strong>Probabilistic Vector Symbolic Architectures (PVSA) — the algebra of uncertainty for hyperdimensional computing, built on JAX.</strong><br/>
  <em>Every hypervector is a posterior distribution. Every operation propagates that distribution in closed form.</em>
</p>

<p align="center">
  <a href="https://github.com/rlogger/bayes-hdc/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/rlogger/bayes-hdc/actions/workflows/tests.yml/badge.svg?branch=main" /></a>
  <a href="https://codecov.io/gh/rlogger/bayes-hdc"><img alt="Coverage" src="https://img.shields.io/badge/coverage-97%25-brightgreen.svg" /></a>
  <a href="https://github.com/rlogger/bayes-hdc/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg" />
  <img alt="JAX" src="https://img.shields.io/badge/JAX-%E2%89%A5%200.4.20-orange.svg" />
  <img alt="Status" src="https://img.shields.io/badge/status-alpha-yellow.svg" />
  <img alt="Tests" src="https://img.shields.io/badge/tests-467%20passing-brightgreen.svg" />
</p>

<p align="center">
  <a href="#what-is-this">What is this?</a> ·
  <a href="#what-you-can-build">What you can build</a> ·
  <a href="#thirty-second-tour">30-second tour</a> ·
  <a href="#formal-guarantees">Guarantees</a> ·
  <a href="#the-pvsa-algebra">PVSA</a> ·
  <a href="docs/workshop_paper.tex">Paper</a> ·
  <a href="ORIGINALITY.md">Originality</a>
</p>

---

## What is this?

> New to the field? This section is for you. If you already know HDC, skip to [What you can build](#what-you-can-build).

**Hyperdimensional computing (HDC)** is a way to represent concepts as very high-dimensional random vectors — typically 10 000 dimensions — called *hypervectors*. Instead of storing data as structured records or tensors, you store meaning as vectors and compute with them: you combine concepts by multiplying vectors ("bind"), average concepts by adding vectors ("bundle"), and reason by measuring how similar two vectors are. It sounds odd until you see it in action: a 10 000-dim random vector is almost orthogonal to every other random 10 000-dim vector, which gives the math enough room to encode entire sentences, images, or knowledge graphs into *one* fixed-size vector and pull them back out by similarity. HDC is fast on tiny hardware, robust to noise, and used today in edge ML, neuromorphic chips, and cognitive-inspired robotics.

**The catch:** classical HDC gives you a single vector and a single answer. "This sentence is French." "This tumour is benign." No uncertainty. No "I'm not sure." In the real world — medicine, fraud detection, safety-critical autonomy — that's not good enough. You need the model to say *when it doesn't know*, and you need formal guarantees on what it claims to know.

**What this library adds:** every hypervector here is a **posterior distribution** over hypervectors (a mean *and* a per-dimension variance), and every HDC operation — binding, bundling, similarity, cleanup, resonator search — propagates that distribution in closed form. You get the speed and robustness of HDC, plus:

- **calibrated probabilities** — temperature scaling reduces expected calibration error by construction, via a convex objective with a unique minimum;
- **coverage-guaranteed prediction sets** — conformal prediction returns a set whose true-label coverage is mathematically guaranteed to be at least 1 − α on exchangeable data, independent of the underlying model;
- **out-of-distribution signals** — the posterior stores per-dimension variance, so the Mahalanobis distance "how far is this query from any learned class" is a first-class citizen.

We call the framework **PVSA (Probabilistic Vector Symbolic Architectures)**. It's the first of its kind in the HDC literature; see [`ORIGINALITY.md`](ORIGINALITY.md) for an independence statement and primary-source attribution for every component.

Everything runs unchanged on CPU, GPU, and TPU via JAX's XLA backend. Every type is a JAX pytree, so `jit`, `vmap`, `grad`, and `pmap` compose with the whole library out of the box.

## What you can build

Four self-contained examples, each under 60 seconds on a laptop CPU, each printing a short report you can read top-to-bottom. Install with `pip install -e ".[examples]"` and run any of them with `python examples/<name>.py`.

### [Language identification →](examples/language_identification.py)
Tell whether a sentence is English, Spanish, French, German, or Italian — with a confidence set that *grows* when the input is ambiguous.

- 5 languages × 20 phrases each, character trigram encoding (Joshi, Halseth, Kanerva 2016)
- **84 % test accuracy** on held-out phrases
- Conformal set size 1 for long unambiguous sentences, 3–4 for short ambiguous ones
- Spanish and Italian correctly appear together when the input is genuinely between them

```
✓  [english  @ 0.99] set=['english']                            "actions speak louder than words"
✓  [spanish  @ 0.95] set=['spanish', 'italian']                 "la pluma es mas fuerte que la espada"
✓  [english  @ 0.47] set=['english', 'spanish', 'french', 'italian']  "how are you doing today my friend"
```

### [Medical selective classification →](examples/medical_selective_prediction.py)
Predict malignant vs. benign on the UCI Breast Cancer Wisconsin Diagnostic dataset — but *abstain* when the conformal set has more than one class. Abstentions get routed to human follow-up instead of risking a wrong call.

- **96.3 % accuracy on the 72 % of cases that get a confident prediction**
- 28 % of cases flagged for review (better than silently guessing)
- Target coverage 95 % achieved empirically at 97.4 %

This is the right default for high-stakes classification: the model knows what it knows and hands off what it doesn't.

### [Out-of-distribution detection →](examples/anomaly_detection.py)
Train a classifier on handwritten digits 0–7, then score digits 8 and 9 as *out of distribution*. The library exposes a signal you physically cannot compute without probabilistic hypervectors: **posterior Mahalanobis distance** — how far a query is from each class, weighted by the per-dimension variance the classifier learned during training.

- 0.84 AUROC from the posterior Mahalanobis signal alone
- Per-class variance is a PVSA-exclusive signal — a deterministic HDC library cannot produce it
- Combines cleanly with standard max-softmax baselines

### [Sequence memory →](examples/sequence_memory.py)
Encode a 12-token sentence as a *single* 10 000-dim vector. Retrieve each word back by its position. Repeated words ("the" at positions 0, 6, 10) are correctly disambiguated by the position binding.

- **12/12 (100 %)** retrieval accuracy
- One HV holds the whole sentence; no per-token storage needed
- Top-1 vs top-2 similarity gap reports retrieval confidence per position

This is the classical HDC "sequence as permute-bundle" trick — but with PVSA's closed-form permute, the same encoding extends to sequences with noise and uncertainty.

## Thirty-second tour

```python
import jax
from bayes_hdc import GaussianHV, bind_gaussian, expected_cosine_similarity

key = jax.random.PRNGKey(0)
x = GaussianHV.random(key, dimensions=10_000, var=0.01)
y = GaussianHV.random(jax.random.fold_in(key, 1), dimensions=10_000, var=0.01)

z   = bind_gaussian(x, y)                   # exact moment propagation
sim = expected_cosine_similarity(x, z)      # uncertainty-aware similarity
```

Post-hoc uncertainty on any classifier — works with PVSA, classical HDC, or anything that produces logits:

```python
from bayes_hdc import ConformalClassifier, TemperatureCalibrator

calibrator = TemperatureCalibrator.create().fit(logits_val, y_val)
probs      = calibrator.calibrate(logits_test)          # ECE-reducing softmax

conformal  = ConformalClassifier.create(alpha=0.1).fit(probs_val, y_val)
sets       = conformal.predict_set(probs)               # (n, k) bool mask
cov        = conformal.coverage(probs_test, y_test)     # ≥ 0.9 guaranteed
```

Deterministic pipelines lift into PVSA with `GaussianHV.from_sample(hv)` — a zero-variance posterior that behaves identically to classical MAP until you start injecting uncertainty. Nothing has to be rewritten.

## Formal guarantees

Every claim in this library is a theorem, not a measurement. These properties hold *by construction*, on *every* input, independent of the dataset.

**Moment-exact binding and bundling.** For independent Gaussian hypervectors `x ~ N(μ_x, diag(σ_x²))` and `y ~ N(μ_y, diag(σ_y²))`, `bind_gaussian(x, y)` returns the exact first and second moments of the element-wise product:

```
E[x · y]  = μ_x · μ_y
Var[x · y] = μ_x² · σ_y² + μ_y² · σ_x² + σ_x² · σ_y²
```

`bundle_gaussian` is exact under the same independence assumption. These are closed-form algebraic identities — no Monte Carlo, no approximation.

**Coverage guarantee for conformal prediction.** For any model producing scores on exchangeable data, `ConformalClassifier.fit(probs_cal, y_cal).predict_set(probs_test)` returns a set whose marginal coverage satisfies

```
P(y ∈ set(x)) ≥ 1 − α
```

on average over calibration-and-test draws. This is the split-conformal guarantee with APS scores (Romano et al. 2020) and holds independently of the classifier, the feature distribution, or the dimensionality.

**Convex calibration.** `TemperatureCalibrator.fit` minimises the negative log-likelihood over a one-parameter temperature via L-BFGS in log-space. The objective is convex; the global minimum is unique; the fitted temperature is the maximum-likelihood estimator.

**Closed-form KL.** `kl_gaussian` and `kl_dirichlet` return analytic KL divergences between two distributions of the same family. These are exact — again, no Monte Carlo — and differentiable end-to-end under JAX `grad`, so they drop into variational objectives directly.

**Reparameterisation gradients everywhere.** Every distributional operation admits a differentiable reparameterisation sampler. `jax.grad` composes through `bind_gaussian`, `bundle_gaussian`, `cleanup_gaussian`, `inverse_gaussian`, and the ELBO helpers in `bayes_hdc.inference` — enough for end-to-end variational training of codebooks or classifier posteriors.

**PyTree compatibility.** Every type in the library is a JAX pytree. `jit`, `vmap`, `grad`, `pmap`, and `shard_map` compose with every operation out of the box. Multi-device wrappers (`pmap_bind_gaussian`, `shard_map_bind_gaussian`, `shard_classifier_posteriors`) exist for scale-out.

## Use cases

PVSA is aimed at any problem where you need both the geometric reasoning of HDC *and* a quantitative account of uncertainty.

**Safety-critical classification.** Medical diagnosis, fraud detection, autonomous-system perception. The model either predicts with a coverage-guaranteed confidence set or abstains and routes the case to human review. See `examples/medical_selective_prediction.py`.

**Edge ML and neuromorphic inference.** HDC's original wheelhouse — tiny-memory, low-power, noise-tolerant inference on sensor data, speech, gesture, and biosignals. PVSA keeps the same hypervector representation while adding quantified noise tolerance via the posterior variance.

**Cognitive robotics and symbolic reasoning.** Role-filler binding ("Dollar of Mexico"), analogical inference, and compositional concept building via the core VSA algebra — now with uncertainty in every slot. See `examples/kanerva_example.py`.

**Structured knowledge representation.** Graphs, sequences, hierarchies, and finite state machines encoded as hypervectors via the encoders and `bayes_hdc.structures` module. Sequence memory (`examples/sequence_memory.py`) is one instance of a more general pattern.

**Streaming and online learning.** `StreamingBayesianHDC` maintains a bounded-memory exponential-moving-average posterior per class, handling concept drift without retraining from scratch. `BayesianAdaptiveHDC` is the Kalman-style conjugate update for stationary streams.

**Out-of-distribution detection.** Two signals ship out of the box: the standard max-softmax baseline (works on any classifier) and the PVSA-exclusive posterior Mahalanobis distance (uses the per-class variance the classifier learned during training). See `examples/anomaly_detection.py`.

**Language and sequence analytics.** Character-level trigram encoders for language identification, text classification, and short-sequence retrieval. See `examples/language_identification.py`.

**Research on probabilistic HDC itself.** Closed-form KL, ELBO helpers, reparameterisation gradients, posterior predictive checks, and a MCMC multi-restart resonator network make PVSA a usable platform for variational and Bayesian inference research on top of VSA.

## Installation

```bash
git clone https://github.com/rlogger/bayes-hdc.git
cd bayes-hdc
pip install -e .                 # core library
pip install -e ".[examples]"     # + matplotlib + scikit-learn (needed for the four application examples)
pip install -e ".[dev]"          # + pytest, ruff, mypy (for contributors)
```

## The PVSA algebra

PVSA is the library's original research contribution. Three formal claims, each verified in `tests/`:

1. **Moment-propagating algebra.** Every core operation (`bind_gaussian`, `bundle_gaussian`, `bind_dirichlet`, `bundle_dirichlet`, `kl_*`, `permute_gaussian`, `cleanup_gaussian`, `inverse_gaussian`) has closed-form moments under standard independence assumptions, with a Monte Carlo fallback for everything else.
2. **Calibrated predictive distributions.** Post-hoc temperature scaling (Guo et al. 2017) fit via L-BFGS in log-space, solving a convex one-parameter objective that reduces the expected calibration error of any classifier.
3. **Coverage-guaranteed prediction sets.** Split-conformal with APS scores (Romano et al. 2020), returning a prediction set whose true-label coverage is ≥ 1 − α on exchangeable data.

On top of the PVSA layer, bayes-hdc ships a complete **deterministic VSA foundation** — eight classical models (BSC, MAP, HRR, FHRR, BSBC, CGR, MCR, VTB), five encoders, five classifiers (including `ClusteringModel`), three associative memory modules, four symbolic data structures, and a capacity-and-noise analysis toolkit — each implemented directly from the primary research papers (Kanerva 1988 / 1997 / 2009; Plate 1995, 2003; Gayler 2003; Rahimi & Recht 2007; Ramsauer et al. 2020; and the Kleyko et al. 2022 VSA surveys). **No component is ported from another HDC library.**

## Library reference

| Category | What's in it |
|---|---|
| **VSA models** | BSC, MAP, HRR, FHRR, BSBC, CGR, MCR, VTB — all sharing the `bind`, `bundle`, `inverse`, `similarity`, `random` API |
| **Encoders** | `RandomEncoder` (discrete), `LevelEncoder` (continuous), `ProjectionEncoder` (random projection), `KernelEncoder` (RFF), `GraphEncoder` |
| **Classifiers** | `CentroidClassifier`, `AdaptiveHDC`, `LVQClassifier`, `RegularizedLSClassifier`, `BayesianCentroidClassifier` (PVSA), `StreamingBayesianHDC` (online with EMA posteriors) |
| **Uncertainty** | `TemperatureCalibrator`, `ConformalClassifier`, `posterior_predictive_check`, `coverage_calibration_check` |
| **Memory** | `SparseDistributedMemory` (Kanerva), `HopfieldMemory` (modern), `AttentionMemory` (multi-head) |
| **Inference** | `elbo_gaussian`, `reconstruction_log_likelihood_mc`, `probabilistic_resonator` (MCMC multi-restart) |
| **Scale** | `pmap_bind_gaussian`, `shard_map_bind_gaussian`, `shard_classifier_posteriors` for pod-scale training |
| **Plots** | `plot_reliability_diagram`, `plot_coverage_curve` (optional; requires matplotlib) |

## Roadmap

### v0.2 — Bayesian hypervector foundation ✅
- [x] `GaussianHV` with mean and diagonal variance
- [x] `bind_gaussian` — exact moment propagation under element-wise product
- [x] `bundle_gaussian` — exact sum of independent Gaussians + normalisation
- [x] `expected_cosine_similarity`, `similarity_variance`
- [x] `kl_gaussian` — closed-form KL for variational objectives
- [x] `sample` / `sample_batch` for Monte Carlo fallbacks

### v0.3 — Probabilistic VSA operations ✅
- [x] `DirichletHV` for probabilistic categorical codebooks
- [x] `bind_dirichlet`, `bundle_dirichlet`, `kl_dirichlet`
- [x] Calibration metrics (`expected_calibration_error`, `maximum_calibration_error`, `brier_score`, `sharpness`, `negative_log_likelihood`, `reliability_curve`)
- [x] `MixtureHV` for multi-modal representations
- [x] `permute_gaussian`, `cleanup_gaussian` derived operations
- [x] `inverse_gaussian` — delta-method approximate unbinding
- [x] Reparameterisation gradients through every distributional op

### v0.4 — Bayesian learning models ✅
- [x] `TemperatureCalibrator` — post-hoc temperature scaling (Guo et al. 2017)
- [x] `ConformalClassifier` — coverage-guaranteed prediction sets via APS (Romano et al. 2020)
- [x] `BayesianCentroidClassifier` — per-class Gaussian posteriors with `predict_uncertainty`
- [x] `BayesianAdaptiveHDC` — streaming Kalman-style online updates
- [x] `bayes_hdc.plots` — optional matplotlib helpers

### v0.5 — Inference & diagnostics ✅
- [x] `bayes_hdc.inference.elbo_gaussian` — closed-form ELBO for Gaussian-posterior PVSA models
- [x] `reconstruction_log_likelihood_mc` — MC reconstruction term for variational training
- [x] `probabilistic_resonator` — multi-restart MCMC factorisation with Gaussian factors
- [x] `posterior_predictive_check` — general PPC driver + `coverage_calibration_check`

### v0.6 — Distribution & scale ✅
- [x] `pmap_bind_gaussian`, `pmap_bundle_gaussian` — multi-device wrappers with single-device fallback
- [x] `shard_map_bind_gaussian` with explicit axis-annotated sharding (JAX ≥ 0.4.24)
- [x] `shard_classifier_posteriors` — reshapes `(K, d)` posteriors into `(n_devices, K/n_devices, d)` for pod-scale training
- [x] `StreamingBayesianHDC` — bounded-memory streaming with EMA posteriors; handles distribution shift

### v1.0 — Datasets, paper, examples ✅
- [x] `bayes_hdc.datasets` with **11 standard HDC benchmarks**: iris, wine, breast_cancer, digits, mnist, fashion_mnist, isolet, ucihar, emg, pamap2, european_languages
- [x] Workshop paper introducing PVSA (`docs/workshop_paper.tex`) with embedded figures
- [x] Containerised benchmarks (`make docker-bench`) with a `Dockerfile` benchmark stage
- [x] Paper figures (reliability, coverage, accuracy, ECE) in `benchmarks/figures/`
- [x] 4 application examples — language identification, medical selective classification, OOD detection, sequence memory

### Future directions
- Variational codebooks trained end-to-end via reparameterised ELBO
- Sparse PVSA posteriors for memory-bounded deployment
- JAX-native integration with `flax.nnx` for Bayesian HDC modules inside larger neural pipelines
- GPU-optimised sparse resonator kernels for large-vocabulary cleanup

## Development

```bash
pytest tests/ -v                                  # run tests (461 pass, 2 skip)
pytest tests/ --cov=bayes_hdc --cov-report=html   # with coverage
ruff check bayes_hdc/                             # lint
ruff format bayes_hdc/                            # format
mypy bayes_hdc/                                   # type check
make bench                                        # regenerate internal benchmark numbers
make figures                                      # regenerate paper figures
make docker-bench                                 # reproduce benchmarks in a container
```

## Citing

If you use bayes-hdc in research, please cite:

```bibtex
@software{bayes_hdc,
  author = {Singh, Rajdeep},
  title  = {{bayes-hdc: Probabilistic Vector Symbolic Architectures for Hyperdimensional Computing}},
  year   = {2026},
  url    = {https://github.com/rlogger/bayes-hdc},
}
```

See [`CITATION.cff`](CITATION.cff) for a machine-readable version and [`ORIGINALITY.md`](ORIGINALITY.md) for per-component primary-source attribution.

## References

- Kanerva, P. (2009). *Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors.*
- Plate, T. A. (1995). *Holographic Reduced Representations.*
- Gayler, R. W. (2003). *Vector Symbolic Architectures answer Jackendoff's challenges for cognitive neuroscience.*
- Joshi, Halseth, and Kanerva (2016). *Language Geometry using Random Indexing.*
- Guo, C. et al. (2017). *On Calibration of Modern Neural Networks.*
- Hendrycks, D. and Gimpel, K. (2017). *A Baseline for Detecting Misclassified and Out-of-Distribution Examples.*
- Romano, Y. et al. (2020). *Classification with Valid and Adaptive Coverage.*
- Ramsauer, H. et al. (2020). *Hopfield Networks is All You Need.*
- Kleyko, D. et al. (2022). *A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures.*

## License

MIT — see [LICENSE](LICENSE).
