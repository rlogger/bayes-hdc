# Changelog

All notable changes to Bayes-HDC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Tier-3 TokenEncoder (2026-05-05)

Closes the language-tokenizer gap from the depth audit's PI-applicability
list. ``TokenEncoder`` is a tokenizer-agnostic vocabulary → hypervector
codebook + permute-bundle sequence encoding.

- **`bayes_hdc.embeddings.TokenEncoder`** — fresh codebook of
  ``vocab_size`` random L2-normalised hypervectors, plus
  ``encode(token_ids)`` (flat permute-bundle) and
  ``encode_hierarchical(token_ids, chunk_size=16)`` (returns a
  :class:`HierarchicalSequence` for T ≳ 200). Tokenizer choice is
  left to the caller — pass integer IDs from HuggingFace,
  SentencePiece, tiktoken, BPE, or character indices; the docstring
  shows the HuggingFace one-liner. ``lookup`` / ``lookup_batch`` are
  jitted for downstream pipelines that want raw per-token vectors.
- **6 tests** in ``tests/test_embeddings.py::TestTokenEncoder``:
  shape + unit-norm-rows, invalid-construction-args, lookup
  consistency, flat encode + codebook cleanup recovers all 12
  tokens of a length-12 sentence at d=2048, hierarchical encode
  recovers ≥ 18/20 spot-checked positions at T=300, and codebook
  determinism under seeded creation.

Test count: 561 → 567 passing (+6). Coverage holds at 93 %.

### Added — Tier-3 hierarchical Sequence + capacity benchmark (2026-05-05)

Closes the fourth of four "blocking-for-VLA" gaps from the depth
audit (long-horizon trajectory encoding). Flat ``Sequence`` saturates
around T ≈ 200 at d = 4 096 with cleanup; the audit flagged this as
the limit on trajectory length for HDC-policy work. The hierarchical
construction below pushes reliable retrieval past T = 800 at the
same d.

- **`bayes_hdc.structures.HierarchicalSequence`** — two-level
  chunked sequence. ``from_vectors(vectors, chunk_size=16)`` encodes
  the input as a flat permute-bundle within each chunk, then a flat
  permute-bundle over the chunks; the **clean chunk hypervectors are
  cached** on the dataclass as ``chunk_codebook`` and used by
  ``get(i)`` for *intermediate cleanup* between the outer and inner
  un-permute. Without that intermediate cleanup the noise from both
  layers sums to the same magnitude as the flat case — a subtle
  point easy to miss when reading only the structural definition.
  References: Plate (2003) §6.2; Frady, Kleyko & Sommer (2018,
  Neural Computation 30(6)) for the recurrent-network capacity
  theory the hierarchical construction is the static analogue of.
- **`benchmarks/benchmark_sequence_capacity.py`** — sweeps
  T ∈ {16, 32, 64, 128, 200, 300, 400, 600, 800} at d = 4 096,
  codebook = 256, 3 seeds. Numbers added to BENCHMARKS.md:

  | T | flat | hierarchical | gain |
  |---:|---:|---:|---:|
  | 200 | 0.958 | 1.000 | +0.042 |
  | 300 | 0.809 | 1.000 | +0.191 |
  | 400 | 0.631 | 1.000 | +0.369 |
  | 600 | 0.423 | 1.000 | +0.577 |
  | 800 | 0.309 | 1.000 | +0.691 |

- **6 new tests** in `tests/test_structures.py::TestHierarchicalSequence`:
  empty construction, out-of-range raise, invalid chunk_size raise,
  short-sequence retrieval (n ≤ chunk_size), beat-flat-by-margin at
  T = 400, uneven-chunk padding (n = 10, chunk_size = 4).

Test count: 555 → 561 passing (+6). Coverage holds at 93 %.
ruff / format / Sphinx -W all clean.

### Added — Tier-3 vision-bridge example (2026-05-05)

Closes the third of the four "blocking-for-VLA" gaps from the depth
audit: the example for plugging a frozen pretrained vision backbone
into the HDC pipeline.

- **`examples/vision_action_policy.py`** — calibrated multi-modal
  action prediction with two `ProjectionEncoder` heads (one for
  vision features, one for proprioception), `bundle_map` fusion into
  a single state hypervector, an `HDRegressor` policy head, and a
  `ConformalRegressor` for per-DOF action intervals with selective
  abstention. Synthesised data simulates DINOv2-S output (384-d
  vision features) + 7-DOF arm proprioception → 7-DOF velocity
  command. The docstring shows exactly how to swap in real DINOv2 /
  CLIP / SigLIP features (one-line change to the feature extractor;
  the rest of the pipeline is unchanged).
- Verified end-to-end on the synthetic task at d=4096:
  - test R² = 0.94 (a linear ground-truth map is recoverable through
    the random projections + bundle fusion + closed-form ridge);
  - per-DOF empirical coverage 0.86–0.93, mean 0.91 (target 0.90 —
    finite-sample-slack tight on every DOF);
  - hand-off-to-teleop abstention rule (predicted action norm vs.
    interval norm) keeps 86 % of points and abstains on 14 %, with
    the abstained set having 28 % higher relative error.
- Documents the bundle-vs-bind choice for additive multi-modal fusion
  vs. role-filler binding — a recurring pitfall when readers first
  apply VSA primitives to continuous-target prediction.

### Added — Tier-3 continuous-output regression stack (2026-05-05)

Closes the first two of the four "blocking-for-VLA" gaps surfaced by
the depth audit's Physical-Intelligence-applicability findings. The
Tier-3 plan called for `HDRegressor` and `ConformalRegressor`; this
batch ships both, end-to-end-tested, with a worked example that
demonstrates calibrated continuous-output prediction with selective
abstention.

- **`bayes_hdc.models.HDRegressor`** — closed-form ridge regression
  on hypervector features for continuous targets `Y ∈ R^{n×k}`.
  Mirrors the existing `RegularizedLSClassifier` API: `create()`,
  `fit()`, `predict()`, `score()` (multi-output R²). Auto-selects
  primal (`d×d`) or dual (`n×n`) form by training-set size; small-`n`
  high-`d` is the default HDC regime and the dual path is the better-
  conditioned of the two there. `jax.grad`-differentiable through
  `weights`, so the regression head plugs into a larger variational
  loss without change.
- **`bayes_hdc.uncertainty.ConformalRegressor`** — split-conformal
  absolute-residual prediction intervals with a finite-sample marginal
  coverage guarantee `P(y ∈ [ŷ - q, ŷ + q]) ≥ 1 - α` on exchangeable
  data (Lei et al. 2018, *Distribution-Free Predictive Inference for
  Regression*). One quantile per output column for multi-output
  targets; concurrent algorithmic work in HDC at the prototype level
  is Liang et al. 2026 *ConformalHDC*, cited explicitly. JIT-compiled
  `predict_interval`, `coverage`, `interval_width`.
- **`examples/calibrated_regression.py`** — end-to-end demo: a
  RandomEncoder over 8 features × 16 values → `HDRegressor` →
  `ConformalRegressor` → selective abstention rule. Verified output
  on a synthetic 2-D continuous-action task at d=4096:
  - test R² = 0.93,
  - empirical coverage 0.91 vs target 0.90 (clear of finite-sample
    slack),
  - abstention rule (zero-in-interval) cleanly separates harder
    cases: 63 % of test points acted on with relative error 0.22, 37 %
    abstained with relative error 0.53.
- **`tests/test_regression.py`** — 15 new tests across HDRegressor
  shape / fit correctness / dual-form / 1-D target reshaping / R² /
  gradient flow, ConformalRegressor alpha validation / quantile
  monotonicity in residual scale / per-output-column quantile / shape
  preservation / interval-width identity / minimum-calibration-size,
  and an end-to-end pipeline integration test that asserts
  multi-trial mean coverage clears `1 - α - 3·SE`.

Test count: 540 → 555 passing (+15 new). Coverage holds at 93 %.
ruff / format / Sphinx -W all clean.

### Added — Tier-2 test depth + code quality (audit follow-up, 2026-05-05)

The 2026-05 depth audit's test-rigor finding ("the suite passes 506
tests but a peer reviewer running `pytest --co -q | grep -c
associativity` gets zero hits") is closed in this batch. Reviewer-
expected mathematical guarantees are now part of the suite.

- **`tests/test_math_properties.py`** — 30 new property-based tests
  covering three previously unaddressed categories:
  1. **Reparameterisation gradient correctness** via
     `jax.test_util.check_grads` against finite differences for
     `bind_gaussian`, `bundle_gaussian`, `kl_gaussian`, and
     `inverse_gaussian` (relaxed tolerances appropriate for
     CPU float32: atol = rtol = 2e-2).
  2. **VSA algebraic laws** across BSC, MAP, and HRR: bind
     commutativity, bind associativity, BSC self-inverse chain
     (`bind(a, bind(a, b)) = b`), bind distributes over (un-
     normalised) bundle for MAP, approximate distributivity over the
     normalised bundle, bundle majority for BSC, full-cycle
     permutation identity, bind / unbind round-trip for MAP and BSC,
     and a noise-aware HRR bind/unbind test that asserts recovery is
     significantly above a random baseline (rather than chasing a
     tight cosine threshold across seeds).
  3. **Closed-form ↔ Monte-Carlo agreement.** The Gaussian moments
     in `bayes_hdc.distributions` are cross-checked against MC
     samples at d=64, n=20 000: bind moments match within 1e-2
     element-wise; un-normalised bundle moments within 2e-2; KL
     within 1 nat; the delta-method inverse mean correction
     `1/μ + σ²/μ³` matches its expansion within 1e-4. KL identities
     (`KL(p, p) = 0`, `KL ≥ 0`) are explicitly asserted.
- **Bug fix — `tests/test_functional.py:218` (`test_bind_hrr_inverse`)
  was degenerate.** It used a single `key` for both `x` and `y`,
  which made `x == y` and turned the bind-inverse test into a
  trivially-true identity. Fixed to split keys; added a sanity
  assertion that `|<x, y>|` is small.
- **Bug fix — `tests/test_v05_v06.py:62` resonator test was
  trivially true.** It only asserted `result.alignment > 0.0`,
  which is satisfied by any well-conditioned codebook. Strengthened
  to verify (a) recovered indices lie in valid codebook ranges, (b)
  the resonator achieves at least one factor match OR alignment >
  0.3 (a substantive partial-factorisation claim), and (c)
  `result.alignment ≤ 1.0 + ε`.
- **`shard_map_bind_gaussian`** (`distributed.py:107`): the bare
  `except Exception:` that swallowed all errors silently is replaced
  with a typed `(ImportError, AttributeError)` whitelist. Mis-
  configurations now raise; only old-JAX-without-shard_map falls
  through to the pmap path (and that path itself now uses a typed
  `(RuntimeError, ValueError)` fallback).
- **Documentation honesty in two `distributions.py` docstrings**:
  - `bind_gaussian` now carries an explicit `.. warning::` block
    that the closed-form variance assumes operand independence;
    pipelines with shared upstream randomness should treat the
    returned variance as a lower bound.
  - `bundle_gaussian` now carries an explicit `.. note::` block
    that the post-normalisation step treats `||sum_mu||` as a
    deterministic scalar — a plug-in approximation, exact only in
    the limit of large d.
- **`StreamingBayesianHDC` foregrounded** in README highlights and
  DESIGN.md §6 ("When to reach for this library") — bounded-memory
  EMA posteriors for non-stationary streams was previously buried
  in a single bullet despite being a primitive no competing HDC
  library exposes.

Test count: 510 → 540 passing (+30 new). Coverage holds at 93 % on
23 modules. ruff / format / Sphinx -W all clean.

### Fixed — Audit-driven Tier-1 corrections (depth audit, 2026-05-05)

A five-agent depth audit surfaced two semantic bugs, one unfair
benchmark, and several overstated novelty claims. All Tier-1 items
fixed in this batch.

- **`pmap_bundle_gaussian` semantic bug** (`distributed.py`) — the old
  implementation composed `bundle_gaussian` twice (per-device, then
  host), which double-normalises and is *not* algebraically identical
  to a single global normalised bundle. Replaced with a sum-only
  per-device kernel + a single host-side normalisation. Added a
  regression test `test_pmap_bundle_gaussian_matches_global_bundle`
  that asserts equivalence to `bundle_gaussian` on the un-sharded
  batch within float32 precision.
- **`recon_log_likelihood_mc` was not a log-likelihood**
  (`inference.py`). The old function returned a cosine-similarity
  proxy bounded in [-1, 1], which made the ELBO `recon - KL`
  dimensionally inconsistent (KL is in nats; cosine is unitless).
  Renamed to `reconstruction_score_mc` (similarity proxy, honestly
  labelled), and added a new
  `gaussian_reconstruction_log_likelihood_mc(observation_noise=...)`
  that computes the actual isotropic-Gaussian log-density. The legacy
  name is preserved as a back-compat alias. Example
  `examples/variational_codebook_learning.py` and the test in
  `tests/test_training.py` switched to the real log-density;
  cos-recovery on a 1024-d target reaches 0.95+ in 1500 Adam steps.
- **Benchmark `.item()` asymmetry** (`benchmarks/benchmark_compare.py`).
  TorchHD's cosine path called `.item()`, forcing a Tensor → Python-
  float host transfer; the JAX path only `block_until_ready`-ed a 0-d
  device array, so the two timers measured different operations.
  Removed the asymmetric `.item()`. Honest numbers: bind 1.41× (was
  1.54×), bundle 2.11× (was 1.89×), cosine 3.48× (was 4.07×), encoder
  0.85× (was 0.81×). Updated `BENCHMARKS.md` and both papers.
- **Five missing citations added** to `paper/paper.bib` and engaged
  with in `paper/paper.md` and `paper/paper_mloss.md`:
  Liang et al. 2026 *ConformalHDC* (arXiv:2602.21446) — concurrent
  conformal-HDC algorithm; Furlong & Eliasmith 2024 (Cogn. Neurodyn.)
  — probabilistic VSA via SSP / fractional binding; Rachkovskij 2024
  (Cogn. Comp.) — shift-equivariance for HDC sequences; Bryant et al.
  2024 *HDVQ-VAE* — static-codebook contrast to our trained-codebook
  contribution; Nesy-GeMs ICLR'23 HD-VAE — workshop precedent.
- **"First" claims softened throughout**. README, `paper/paper.md`
  §State-of-the-field, `paper/paper_mloss.md` §2 + §7, and
  `bayes_hdc/training.py` module docstring now distinguish "first
  *comprehensive* JAX-native HDC library" from the bare claim (two
  narrower JAX packages, `hyper-jax` and `hrr`, exist) and
  "first *open-source library to ship*" from "first to think of"
  (algorithmic priority on conformal HDC sits with Liang et al.).
- **Test count reconciled to one canonical number (510)** across
  README, BENCHMARKS, paper.md, paper_mloss.md (was 480/506/498/506).
- **3 new tests** in `tests/test_inference_and_distributed.py` cover
  the bounded-score property of `reconstruction_score_mc`, the
  d-extensive scale of `gaussian_reconstruction_log_likelihood_mc`,
  and the max-at-target property; +1 regression test for the
  `pmap_bundle_gaussian` fix. Total: **510 passing, 93 % coverage**.

### Added — Publishability push: JOSS / JMLR-MLOSS submission artefacts and SOTA features

- **`paper/paper.md`** — JOSS-format short paper (1107 words) with frontmatter,
  Summary, Statement of Need, State of the Field, Software Design,
  Research Impact Statement, AI Usage Disclosure, Acknowledgements, and
  References. Positions bayes-hdc as the differentiable, uncertainty-aware
  HDC stack — the empty lane in the open-source HDC ecosystem.
- **`paper/paper_mloss.md`** — longer JMLR-MLOSS-format paper (1686
  words, 7 sections) emphasising novel contributions: PVSA algebra,
  end-to-end variational training, conformal prediction, equivariance
  verifiers, and 8-VSA-model coverage.
- **`paper/paper.bib`** — BibTeX bibliography for both papers,
  20 entries spanning foundational HDC/VSA work, calibration /
  conformal prediction, the JAX numerical stack, and the four competing
  HDC libraries (TorchHD, hdlib, vsapy, NengoSPA).
- **`.zenodo.json`** — DOI archival metadata, ready for tagged-release
  Zenodo integration.
- **README "How to cite" + "In the HDC library landscape" table** —
  explicit positioning vs TorchHD / hdlib / vsapy / NengoSPA on the
  five differentiating axes, plus a BibTeX block for citation.
- **`bayes_hdc.training`** — new module with a minimal, dependency-free
  Adam optimiser (`adam_init` / `adam_update` / `AdamState`) and a
  high-level `train_variational_codebook` loop that compiles via
  `jax.lax.scan` so the full training trajectory lowers to one XLA
  program. The `TrainResult` is a registered JAX pytree, so the
  trainer can be wrapped in `jax.jit`. To our knowledge no other
  open-source HDC/VSA library exposes a comparable end-to-end
  variational training API. 8 unit tests, 98 % line coverage on the
  new module.
- **`examples/variational_codebook_learning.py`** — concrete
  demonstration: a 1024-d `GaussianHV` posterior initialised at
  μ = 0, σ² = 1 recovers a target μ-direction at cosine similarity
  0.9999 in 500 Adam steps under a -ELBO loss with a 32-sample MC
  reconstruction term. Verifies that `jax.grad` composes through every
  PVSA primitive end-to-end.
- **`examples/hopfield_cleanup_hdc.py`** — modern continuous Hopfield
  retrieval (Ramsauer et al. 2020) as a soft cleanup step in an HDC
  pipeline, contrasted against classical hard nearest-neighbour
  cleanup over the same codebook.
- **Wall-clock micro-benchmark vs TorchHD** in `BENCHMARKS.md`:
  pointwise ops on CPU at `d = 10 000`, 200 trials. Cosine similarity
  is **4.07× faster** under JAX-`jit` than TorchHD's eager kernels;
  `bind` and `bundle` are 1.54× and 1.89× respectively. Reproducible
  via `python benchmarks/benchmark_compare.py`.
- Test count grew from 498 to **506 passing**, line coverage 93 %
  on 23 modules.

### Added — Group-theoretic structure module and research-connection framing

- `bayes_hdc.equivariance` — new module exposing the cyclic-shift action of
  `Z/d` as first-class. Provides `shift`, `compose_shifts`, the canonical
  shift-equivariant bilinear operator `hrr_equivariant_bilinear`, and
  property-based verifiers (`verify_shift_equivariance`,
  `verify_single_argument_shift_equivariance`, `verify_shift_invariance`)
  that reject user-defined ops claiming a symmetry they do not have.
- `examples/weight_space_posterior.py` — new demo treating a
  `BayesianCentroidClassifier`'s posterior as a distribution over
  weight-space. Shows posterior sampling, epistemic uncertainty as
  disagreement across draws, and numerically verifies the posterior
  commutes with the cyclic-shift action.
- `DESIGN.md` — long-form design document covering the algebra, the PVSA
  lift to measures, the functional-programming commitments, the JAX
  idioms, and the research programmes the design serves (weight-space
  learning, equivariant neural functionals, meta-RL with structured
  representations).
- README rewritten with the new framing: a JAX library with serious
  algebraic depth, legibly useful to weight-space / NFN / structured-RL
  research.
- 14 new tests in `tests/test_equivariance.py` for the group action,
  primitive equivariances, and detector correctness. Suite now 475 tests,
  97 % line coverage, 100 % on the new module.

### Added — v0.5/v0.6 completion + containerised benchmarks

- **`bayes_hdc.resonator.probabilistic_resonator`** — multi-restart MCMC factorisation of a composite PVSA hypervector. Each chain samples factor indices from a softmax over residual similarities (Metropolis-style); returns the best chain's `(indices, alignment, history, n_restarts)`. Uses `inverse_gaussian` so uncertainty propagates through unbinding. Closes the v0.5 research-paper block.
- **`bayes_hdc.diagnostics`** — posterior predictive checks and coverage calibration audits. Ships `posterior_predictive_check` with two ready-made statistics (`statistic_mean_norm`, `statistic_cosine_to_reference`), a general `coverage_calibration_check` that sweeps α for any `ConformalClassifier`, and associated `PPCResult` / `CoverageCheckResult` dataclasses.
- **`StreamingBayesianHDC`** — bounded-memory streaming classifier with exponential-moving-average posteriors. Variance adapts to distribution shift (unlike the strict-shrinkage `BayesianAdaptiveHDC`); memory is O(K·d), independent of stream length.
- **`shard_map_bind_gaussian`** and **`shard_classifier_posteriors`** in `bayes_hdc.distributed` — explicit-axis sharding via `jax.experimental.shard_map`. On multi-device hosts shards across a `Mesh`; on single-device hosts degrades to plain `bind_gaussian` for API parity.
- **Containerised benchmarks** — updated `Dockerfile` with dedicated `benchmark` stage that runs all three benchmarks end-to-end and writes results to a mounted volume. New `make docker-bench` / `make docker-test` targets; `make bench` / `make figures` for local runs.
- **Paper updates** — `docs/workshop_paper.tex` now includes sample figures from `benchmarks/figures/`, references the Docker-based reproducibility path, and adds the Bayesian `inverse` and mixture hypervector types to the algebra listing.
- **23 new unit tests** covering resonator, PPC, streaming, and shard-map helpers. Total: **467 tests, 97% line coverage on 22 source files.**

### Added — v0.3/v0.4 closure + v0.5/v0.6 openers + paper figures

- **`inverse_gaussian`** — approximate distributional inverse via the delta method; exact in the zero-variance limit, matches classical MAP unbinding. Preserves `bind(bind(x, y), inverse(y)) ≈ x` on low-variance inputs.
- **`BayesianAdaptiveHDC`** — streaming Kalman-style online classifier. Maintains a proper conjugate-Gaussian posterior per class, with a configurable observation-noise variance. Supports streaming data, distribution shift, and anytime-valid uncertainty.
- **`bayes_hdc.inference`** module (v0.5 opener): `elbo_gaussian` for variational PVSA objectives, `reconstruction_log_likelihood_mc` as a convenient MC reconstruction term.
- **`bayes_hdc.distributed`** module (v0.6 opener): `batch_bind_gaussian` / `batch_similarity_gaussian` (vmap wrappers), `pmap_bind_gaussian` / `pmap_bundle_gaussian` (multi-device with single-device fallback).
- **Paper figures** (`benchmarks/generate_figures.py`): generates reliability diagrams, conformal-coverage curves, accuracy-comparison bars, and ECE-reduction bars for every dataset. 10 PDFs + 10 PNGs under `benchmarks/figures/`, paper-ready at 150 DPI.
- **34 new unit tests** across `test_inverse_gaussian.py`, `test_bayesian_adaptive.py`, `test_inference_and_distributed.py` — reparameterisation-gradient tests specifically demonstrate that `jax.grad` composes through every PVSA primitive (bind, bundle, KL), validating variational-codebook training paths.
- Total test count: **451 passing, 97% line coverage on 20 source files**.

### Added — Bayesian classifier + plotting + workshop paper + dataset expansion

- **`BayesianCentroidClassifier`** in `bayes_hdc/bayesian_models.py` — per-class Gaussian posterior fit by empirical Bayes (sample mean + regularised diagonal variance with configurable prior strength). Exposes `predict`, `predict_proba`, `predict_uncertainty` (per-class similarity variance — a PVSA-exclusive signal), and `predict_with_uncertainty` in one pass. 15 unit tests.
- **`bayes_hdc.plots`** module — optional matplotlib helpers. `plot_reliability_diagram` produces the Guo-et-al. 2017 reliability diagram (per-bin accuracy vs confidence with gap bars and the $y = x$ calibration line); `plot_coverage_curve` sweeps $\alpha$ for a conformal classifier and plots empirical coverage + mean set size against $1 - \alpha$. 6 unit tests; `pytest.importorskip("matplotlib")` guarded.
- **3 new datasets in `bayes_hdc.datasets`**: `load_emg` (multi-class hand gestures), `load_pamap2` (physical activity monitoring, Reiss & Stricker 2012; subsample-by-default for iteration speed), `load_european_languages` (21-class n-gram classification, Joshi / Halseth / Kanerva 2016). Registry now lists 11 datasets.
- **`docs/workshop_paper.tex`** — standalone short-paper draft introducing PVSA for NeurIPS/ICLR/UAI workshop submission. Sections: introduction, background (classical VSA + calibration + conformal), PVSA algebra, calibrated + coverage-guaranteed prediction, empirical validation (accuracy / ECE / coverage tables), related work, conclusion. References block included.

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
