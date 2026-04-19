# Changelog

All notable changes to JAX-HDC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- Performance benchmark suite, including `benchmarks/benchmark_compare.py` (jax-hdc vs TorchHD)
- `cleanup()` with `return_similarity` support
- Metrics module (`jax_hdc/metrics.py`) with `bundle_snr`, `bundle_capacity`, `effective_dimensions`, `sparsity`, `signal_energy`, `saturation`, `cosine_matrix`, `retrieval_confidence`
- Functional resonator-network skeleton (`functional.resonator`)
- Functional primitives: `fractional_power`, `jaccard_similarity`, `tversky_similarity`, `soft_quantize`, `hard_quantize`, `flip_fraction`, `add_noise_map`, `select_bsc`, `select_map`, `threshold`, `window`
- Symbolic data structures: `Multiset`, `HashTable`, `Sequence`, `Graph` (`jax_hdc/structures.py`)
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

[Unreleased]: https://github.com/rlogger/jax-hdc/compare/v0.1.0-alpha...HEAD
[0.1.0-alpha]: https://github.com/rlogger/jax-hdc/releases/tag/v0.1.0-alpha
