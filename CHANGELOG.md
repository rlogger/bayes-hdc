# Changelog

All notable changes to Bayes-HDC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — datasets submodule (v1.0 / first cut)
- `bayes_hdc.datasets` subpackage with a uniform `HDCDataset` container and a name-based `load()` dispatcher.
- Sklearn-backed (offline) loaders: `load_iris`, `load_wine`, `load_breast_cancer`, `load_digits`.
- OpenML-backed (download + cache) loaders: `load_mnist`, `load_fashion_mnist`, `load_isolet` (Fanty & Cole 1990; the canonical HDC benchmark), `load_ucihar` (Anguita et al. 2013).
- Stratified 70/30 train/test splits by default with configurable `test_size` / `random_state`; automatic label normalisation to contiguous `int32`.
- 13 unit tests covering shape, dtype, split stratification, reproducibility, dispatch, and error handling; 1 network-gated integration test.
- Added `datasets` extras group in `pyproject.toml` for installing the sklearn dependency.
- New `network` pytest marker for tests that require a network connection; skipped by default.

### Added — Bayesian extensions (v0.3 completion)
- `MixtureHV` — mixture-of-Gaussian hypervector type with weights, component means and variances, uniform-default construction, law-of-total-variance `variance()`, moment-matched `collapse_to_gaussian()`, and categorical sampling.
- `permute_gaussian(x, shifts)` — cyclic shift of both the mean and variance vectors, matching the deterministic `permute` under the independent-component assumption.
- `cleanup_gaussian(query, memory)` — nearest-neighbour retrieval in a list of Gaussian hypervectors via expected cosine similarity; returns `(best_index, best_score)`.
- 22 new unit tests covering these features.

### Reframed — Probabilistic VSA (PVSA)

The project now defines and implements **Probabilistic Vector Symbolic Architectures (PVSA)** as a named research framework: an HDC algebra in which every hypervector is a posterior distribution, and every VSA primitive propagates moments in closed form. The README, paper, slide deck, quiz, and cover letter are rewritten to lead with this contribution.

### Fixed — `TemperatureCalibrator` optimisation
- Switched from naive gradient descent on `T` to L-BFGS in log-space (`log T`), with a gradient-descent fallback and a safety clip to `T ∈ [0.01, 100]`. The previous implementation could drift to `T ≈ 10¹¹` on tiny-logit inputs, collapsing softmax to uniform. Matches the Guo et al. (2017) reference implementation.

### Added — standard-HDC-pipeline benchmark
- `benchmarks/benchmark_calibration.py` rewritten to use the standard HDC pipeline: `KBinsDiscretizer` → `RandomEncoder` (codebook lookup) for tabular, `ProjectionEncoder` for MNIST, plus `AdaptiveHDC` with iterative refinement at `D = 10 000`. Added MNIST as a fifth benchmark dataset.
- Empirical results over {iris, wine, breast-cancer, digits, MNIST}:
  - Accuracy parity with Torchhd (both libraries at 82–95%).
  - **ECE reduction from temperature scaling:** iris 6.5×, wine 4.5×, digits **20×**, MNIST **25×**.
  - **Conformal coverage at α = 0.1:** 94.7% – 100% on every dataset, empirically validating the guarantee.
- `benchmarks/benchmark_selective.py` — selective classification via conformal sets of size 1, matched by empirical coverage to an MSP-threshold baseline.
- `benchmarks/benchmark_ood.py` — out-of-distribution detection comparing MSP, MSP+T, and conformal-set-size scores on digits and wine via leave-one-class-out AUROC.
- All three benchmarks output machine-readable JSON (`benchmark_calibration_results.json`, etc.) under `benchmarks/`.

### Added — Bayesian layer v0.3 + v0.4
- `DirichletHV` — distributions over the probability simplex; `mean`, `variance`, `concentration`, `sample`, `sample_batch`, `from_counts`, `uniform`
- `bind_dirichlet`, `bundle_dirichlet` — moment-matched composition for categorical Bayesian HDC
- `kl_dirichlet` — closed-form KL divergence between two Dirichlets
- `bayes_hdc.uncertainty` module:
  - `TemperatureCalibrator` — post-hoc temperature scaling fitted by gradient descent on NLL (Guo et al. 2017)
  - `ConformalClassifier` — split-conformal wrapper with marginal coverage guarantee, using APS nonconformity scores (Romano et al. 2020)
- Calibration metrics in `bayes_hdc.metrics`: `expected_calibration_error`, `maximum_calibration_error`, `brier_score`, `sharpness`, `negative_log_likelihood`, `reliability_curve`
- `benchmarks/benchmark_calibration.py` — head-to-head vs TorchHD on 5 datasets (iris, wine, breast-cancer, digits, synthetic), reports accuracy, ECE, Brier, NLL, conformal coverage, set size
- 42 new unit tests across `tests/test_distributions.py` (Dirichlet additions), `tests/test_uncertainty.py`, `tests/test_calibration_metrics.py`, `tests/test_dirichlet.py` — all passing, 99% coverage maintained

### Added — Bayesian layer (headline for the v0.2 release)
- `bayes_hdc.distributions` module — the Bayesian core of the library
- `GaussianHV`: hypervectors represented as mean + diagonal variance; `jax.pytree`-compatible
- `bind_gaussian`: exact moment propagation under element-wise product (MAP-style binding)
- `bundle_gaussian`: exact moment propagation under summation + normalisation
- `expected_cosine_similarity`: uncertainty-aware similarity at the moment-matched Gaussian
- `similarity_variance`: exact first-order variance of the dot product
- `kl_gaussian`: closed-form KL divergence suitable as a variational objective
- `sample` / `sample_batch`: Monte Carlo fallbacks for richer posterior-predictive quantities
- 24 unit tests for the Bayesian layer (test_distributions.py) — 100% line coverage

### Changed
- **Project pivot:** Bayes-HDC is now primarily a Bayesian / probabilistic framework for HDC. The eight deterministic VSA models, encoders, classifiers, memory modules, and structures remain as the foundation on which the Bayesian layer builds.
- Repository renamed from `jax-hdc` to `bayes-hdc`; Python package renamed from `jax_hdc` to `bayes_hdc`.
- Paper title, abstract, and introduction rewritten to lead with the Bayesian contribution.
- Version bumped to 0.2.0a0 to mark the pivot.

### Added
- BSBC (Binary Sparse Block Codes) VSA model
- CGR (Cyclic Group Representation) VSA model
- MCR (Modular Composite Representation) VSA model
- VTB (Vector-Derived Transformation Binding) VSA model
- KernelEncoder (RBF kernel approximation via random Fourier features)
- GraphEncoder for graph structures
- LVQClassifier (Learning Vector Quantization)
- RegularizedLSClassifier (regularized least squares)
- ClusteringModel (HDC-style k-means)
- SparseDistributedMemory, HopfieldMemory, and AttentionMemory modules
- Integration tests (end-to-end encode/train/predict)
- Performance benchmark suite, including `benchmarks/benchmark_compare.py` (bayes-hdc vs TorchHD)
- `cleanup()` with `return_similarity` support
- Metrics module (`bayes_hdc/metrics.py`) with `bundle_snr`, `bundle_capacity`, `effective_dimensions`, `sparsity`, `signal_energy`, `saturation`, `cosine_matrix`, `retrieval_confidence`
- Functional resonator-network skeleton (`functional.resonator`)
- Functional primitives: `fractional_power`, `jaccard_similarity`, `tversky_similarity`, `soft_quantize`, `hard_quantize`, `flip_fraction`, `add_noise_map`, `select_bsc`, `select_map`, `threshold`, `window`
- Symbolic data structures: `Multiset`, `HashTable`, `Sequence`, `Graph` (`bayes_hdc/structures.py`)
- `SLIDES.md` — full library walkthrough deck
- `QUIZ.md` — 58-question self-quiz with answer key
- `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1)
- `SECURITY.md` — vulnerability reporting policy
- `CITATION.cff` — machine-readable citation
- GitHub issue templates (bug report, feature request) and PR template
- `docs/MLOSS_CHECKLIST.md` and `docs/MLOSS_COVER_LETTER.md` for JMLR MLOSS submission preparation
- SPDX license header (`# SPDX-License-Identifier: MIT`) on every Python source file
- Roadmap (v0.2 → v1.0) targeting differentiable primitives, factorization, distributed / streaming, probabilistic HDC, neuro-symbolic reasoning, and a JMLR MLOSS paper

### Changed
- Replaced `black`/`isort`/`flake8` with `ruff` for linting and formatting
- Removed `numpy` and `optax` from core dependencies
- Centralized JAX dataclass registration in `_compat.py`
- Reduced `utils.py` to `normalize` and `benchmark_function`
- CI matrix expanded to Ubuntu + macOS + Windows across Python 3.9 through 3.13
- Tightened mypy types in `functional.py` (graph-encode closure annotation) and `models.py` (clustering loop variable)

### Removed
- Nix packaging files (`default.nix`, `flake.nix`, `shell.nix`, `.envrc`) and their documentation references

## [0.1.0-alpha] - 2024-11-03

### Added
- Core functional operations (bind, bundle, permute, similarity)
- Four VSA model implementations: BSC, MAP, HRR, FHRR
- Three encoder types: RandomEncoder, LevelEncoder, ProjectionEncoder
- Two classification models: CentroidClassifier, AdaptiveHDC
- Unit tests for core operations and VSA models
- Reference examples: basic operations, Kanerva's example, classification
- Documentation structure (Sphinx/ReadTheDocs ready)
- MIT License

[Unreleased]: https://github.com/rlogger/bayes-hdc/compare/v0.1.0-alpha...HEAD
[0.1.0-alpha]: https://github.com/rlogger/bayes-hdc/releases/tag/v0.1.0-alpha
