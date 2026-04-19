# Changelog

All notable changes to Bayes-HDC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
