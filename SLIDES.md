# bayes-hdc: library walkthrough

A deck for reading the `bayes-hdc` codebase with confidence. All file
references point to `bayes_hdc/` unless noted. Slides are separated by `---`
so the file renders as plain Markdown and as a Marp / Slidev deck.

---

## 0. The pitch: Probabilistic VSA (PVSA)

**Classical VSA:** a symbol is a point in $\mathbb{R}^d$ (or $\mathbb{F}_2^d$, etc.). Binding and bundling produce points.

**PVSA:** a symbol is a *distribution* over hypervectors. Binding and bundling produce distributions — with closed-form moments.

This library introduces PVSA and ships three layers that use it:

1. **Distributional algebra** — `GaussianHV`, `DirichletHV` with closed-form bind/bundle and KL.
2. **Calibration** — `TemperatureCalibrator` reduces ECE by 5–25× on real datasets.
3. **Coverage guarantees** — `ConformalClassifier` returns prediction sets with marginal coverage ≥ 1 − α.

No public HDC library previously offered any of these together. The **deterministic** VSA foundation (the 8 classical models, encoders, classifiers, memory, structures) is what PVSA builds on top of.

---

## 1. What is HDC / VSA in one page

**Hyperdimensional computing (HDC):** compute with long (≥ 1000-dim),
random-looking vectors called **hypervectors**. Information is distributed
across every dimension, so the representation is robust to noise and
component failure.

**Vector symbolic architecture (VSA):** the algebra on hypervectors. Four
primitive operations:

| Op | Produces | Result is... |
|----|----------|--------------|
| `bind(x, y)` | new hypervector | dissimilar to both `x` and `y` |
| `bundle([x₁..xₙ])` | new hypervector | similar to all inputs |
| `permute(x, k)` | new hypervector | dissimilar to `x`, reversible |
| `similarity(x, y)` | scalar | ∈ [-1, 1] or [0, 1] |

Plus two derived operations:

- `inverse(x)` — undoes a bind: `bind(bind(x, y), inverse(y)) ≈ x`
- `cleanup(q, M)` — snaps noisy `q` to the nearest entry in codebook `M`

---

## 2. The 10-000-ft view of this library

```
bayes_hdc/
├── functional.py     # low-level ops (bind_*, bundle_*, similarity, resonator, ...)
├── vsa.py            # 8 VSA model classes (BSC, MAP, HRR, FHRR, BSBC, CGR, MCR, VTB)
├── embeddings.py     # 5 encoders (Random, Level, Projection, Kernel, Graph)
├── models.py         # 5 classifiers (Centroid, Adaptive, LVQ, RegLS, Clustering)
├── memory.py         # 3 memory modules (SDM, Hopfield, Attention)
├── structures.py     # 4 data structures (Multiset, HashTable, Sequence, Graph)
├── metrics.py        # 14 analysis + calibration funcs (SNR, ECE, Brier, reliability, …)
├── distributions.py  # PVSA: GaussianHV, DirichletHV — Bayesian hypervector types
├── uncertainty.py    # PVSA: TemperatureCalibrator, ConformalClassifier
├── utils.py          # normalize, benchmark_function
├── constants.py      # EPS = 1e-8
└── _compat.py        # register_dataclass fallback for older JAX
```

1 370 statements of source, **363 tests, 99 % coverage**.

---

## 3. The public API surface

`from bayes_hdc import ...` exposes:

- **PVSA — Bayesian hypervectors**: `GaussianHV, DirichletHV, bind_gaussian, bundle_gaussian, bind_dirichlet, bundle_dirichlet, expected_cosine_similarity, similarity_variance, kl_gaussian, kl_dirichlet`
- **PVSA — uncertainty**: `TemperatureCalibrator, ConformalClassifier`
- **VSA models**: `BSC, BSBC, MAP, HRR, FHRR, CGR, MCR, VTB`
- **Encoders**: `RandomEncoder, LevelEncoder, ProjectionEncoder, KernelEncoder, GraphEncoder`
- **Classifiers**: `CentroidClassifier, AdaptiveHDC, LVQClassifier, RegularizedLSClassifier, ClusteringModel`
- **Memory**: `SparseDistributedMemory, HopfieldMemory, AttentionMemory`
- **Structures**: `Multiset, HashTable, Sequence, Graph`
- **Functional ops**: `bind_*, bundle_*, inverse_*, permute, cleanup, resonator, hash_table, ngrams, bundle_sequence, graph_encode, ...`
- **Similarity**: `cosine_similarity, hamming_similarity, dot_similarity, jaccard_similarity, tversky_similarity, matching_similarity, phasor_similarity`
- **Capacity metrics**: `bundle_snr, bundle_capacity, effective_dimensions, sparsity, saturation, signal_energy, cosine_matrix, retrieval_confidence`
- **Calibration metrics**: `expected_calibration_error, maximum_calibration_error, brier_score, sharpness, negative_log_likelihood, reliability_curve`
- **Utilities**: `normalize, benchmark_function`, noise injection (`flip_fraction`, `add_noise_map`), quantization (`soft_quantize, hard_quantize`), fractional power (`fractional_power`)

---

## 4. `functional.py` — the core

Two design rules:

1. **Stateless.** Every op is a pure function of inputs + JAX keys.
2. **JIT-friendly.** Most ops carry `@jax.jit`. PRNG keys thread explicitly.

The file is organised by VSA family:

```
bind_bsc / bundle_bsc / inverse_bsc / hamming_similarity     ← BSC (XOR, majority)
bind_map / bundle_map / inverse_map / cosine_similarity      ← MAP (mul, norm-sum)
bind_hrr (FFT) / inverse_hrr / bundle_hrr (=bundle_map)      ← HRR (circ. convolution)
bind_cgr / bundle_cgr / inverse_cgr / matching_similarity    ← CGR (mod add)
bind_mcr (=bind_cgr) / bundle_mcr (phasor) / phasor_sim      ← MCR (phase arithmetic)
bind_vtb (matmul) / inverse_vtb (pinv)                       ← VTB (matrix bind)
permute / cleanup                                             ← generic
multibind_map / multibind_bsc / cross_product                 ← n-ary ops
hash_table / ngrams / bundle_sequence / bind_sequence         ← composite
graph_encode / resonator                                      ← structured
```

---

## 5. The eight VSA models, side by side

| Model | Element type | `bind` | `bundle` | `inverse` | `similarity` |
|-------|--------------|--------|----------|-----------|---------------|
| **BSC** | `bool` | XOR | majority | identity | Hamming |
| **MAP** | `float32` | element-wise × | norm-sum | reciprocal | cosine |
| **HRR** | `float32` | circular conv (via FFT) | norm-sum | reverse-flip | cosine |
| **FHRR** | `complex64` | element-wise × | norm-sum | conjugate | cosine (real part) |
| **BSBC** | `bool`, block-sparse | XOR | majority | identity | Hamming |
| **CGR** | `int` ∈ ℤ_q | (x+y) mod q | element-wise mode | (q-x) mod q | matching fraction |
| **MCR** | `int` ∈ ℤ_q, phase | (x+y) mod q | phasor sum + snap-to-grid | (q-x) mod q | phasor inner product |
| **VTB** | `float32`, d = n² | reshape-and-matmul | norm-sum | matrix pseudoinverse | cosine |

Each class has an identical API: `create(dimensions, ...)`, `bind`, `bundle`, `inverse`, `similarity`, `random`.

---

## 6. VSA model internals — `vsa.py`

Every model is a **JAX pytree** registered via `@register_dataclass`:

- Data fields (`name`, `dimensions`) are traced by JAX.
- Static fields (`metadata=dict(static=True)`) are compile-time constants.

Example, BSBC (block-sparse binary):

```python
@register_dataclass
@dataclass
class BSBC(VSAModel):
    block_size: int = field(metadata=dict(static=True), default=100)
    k_active:   int = field(metadata=dict(static=True), default=5)
```

Constraints:

- `BSBC`: `dimensions % block_size == 0`, `1 ≤ k_active ≤ block_size`
- `VTB`: dimensions must be a perfect square
- `CGR` / `MCR`: `q ≥ 2`

Factory: `create_vsa_model("hrr", dimensions=10000)` builds any of the 8 by name.

---

## 7. Encoders — `embeddings.py`

| Encoder | Input | How it encodes |
|---------|-------|----------------|
| **RandomEncoder** | discrete features (`indices`) | lookup in a random codebook `(num_features, num_values, d)`, bundle |
| **LevelEncoder** | continuous values | interpolate between random level hypervectors |
| **ProjectionEncoder** | high-dim real vectors | multiply by normalized random projection matrix |
| **KernelEncoder** | real vectors | Random Fourier Features (cos(ωᵀx + b)); approximates RBF kernel |
| **GraphEncoder** | edge list | per-edge `bind(u, permute(v))`, bundle all edges |

Every encoder stores:

- **data fields** (codebook, projection, level_hvs) — traced by JAX
- **static fields** (num_features, dimensions, vsa_model_name) — constants

All encoders expose `encode(x)` and `encode_batch(x)`; both are `@jax.jit`.

---

## 8. Classifiers — `models.py`

| Classifier | Training rule | Gradient? |
|------------|---------------|-----------|
| **CentroidClassifier** | average hypervectors per class | none — closed form |
| **AdaptiveHDC** | centroid + iterative error-driven refinement | none |
| **LVQClassifier** | winner-take-all: move toward / away | none |
| **RegularizedLSClassifier** | solve `(XᵀX + λI) W = XᵀY` | closed form |
| **ClusteringModel** | k-means in hypervector space | none |

All share: `create(...)`, `fit(hvs, labels)`, `predict(queries)`, `score(hvs, labels)`, `replace(**updates)` for pytree-safe updates.

Common pattern: classifiers dispatch on `vsa_model_name` to choose between BSC (Hamming, majority) and real-valued (cosine, norm-sum) paths.

---

## 9. Memory modules — `memory.py`

| Module | What it is |
|--------|-----------|
| **SparseDistributedMemory** | Kanerva SDM — random addresses, content stored at addresses within `radius` |
| **HopfieldMemory** | Modern Hopfield — softmax-weighted retrieval, `beta` is inverse temperature |
| **AttentionMemory** | scaled dot-product attention, optional multi-head, temperature-controlled |

All three are JAX pytrees: writes return a new object (functional style) rather than mutating. `retrieve(query)` is `@jax.jit`. `AttentionMemory` also has `retrieve_with_weights` to inspect the attention distribution.

---

## 10. Symbolic structures — `structures.py`

Four HDC-backed data structures, all stored in a single hypervector:

| Structure | `add` | `get` / query |
|-----------|-------|---------------|
| **Multiset** | `value + hv` | cosine similarity of `hv` vs `value` |
| **HashTable** | `value + bind(key, val)` | `bind(value, inverse(key))` |
| **Sequence** | `permute(value, 1) + hv` | `permute(value, -(size-i-1))` |
| **Graph** | `value + bind(u, v)` (or `bind(u, permute(v))` if directed) | edge similarity via `dot_similarity` |

All structures are functional (return a new instance on modification) and compatible with `jax.jit`.

---

## 11. Metrics & analysis — `metrics.py` ⭐

First-class capacity-and-noise tooling that practitioners normally re-derive per paper — implemented from the theory (Kanerva 2009; Kleyko et al. 2022 surveys), not from another library:

| Function | What it measures |
|----------|------------------|
| `bundle_snr(d, n)` | expected signal-to-noise after bundling n vectors in d dims |
| `bundle_capacity(d, delta)` | largest n with retrieval error ≤ delta (conservative bound) |
| `effective_dimensions(x)` | participation ratio (Σxᵢ²)² / Σxᵢ⁴ |
| `sparsity(x, τ)` | fraction of elements with \|xᵢ\| < τ |
| `signal_energy(x)` | Σxᵢ² — L2 energy, flags collapsed representations |
| `saturation(x)` | fraction of elements near ±1 |
| `cosine_matrix(vectors)` | n×n pairwise cosine — check codebook orthogonality |
| `retrieval_confidence(q, codebook)` | gap between best and 2nd-best similarity |

Use these to **size dimensionality**, **diagnose collapse**, and **validate codebooks** during development.

---

## 12. JAX idioms used throughout

- **PRNG threading.** Every random op takes an explicit `jax.random.PRNGKey`. Split keys: `k1, k2 = jax.random.split(key)`.
- **Pytree registration.** All dataclasses register via `@register_dataclass` so they can cross `jit` / `vmap` / `pmap` boundaries.
- **Functional updates.** State is never mutated. `replace(**updates)` returns a new object via `dataclasses.replace`.
- **vmap for batching.** `encode_batch = jax.vmap(encode)`. Same pattern for classifier `predict`.
- **`jax.jit` on pure methods only.** Methods that loop over data in Python (e.g. `AdaptiveHDC.fit`) are not jitted — they use jitted primitives internally.
- **`block_until_ready()`** in benchmarks to force async dispatch to complete before timing.

---

## 13. Derived operations worth knowing

- **`cleanup(q, memory)`** — nearest-neighbour retrieval. The poor man's resonator.
- **`hash_table(keys, values)`** — `Σ bind(kᵢ, vᵢ)` in one call, reconstructs as `bind(h, inverse(k))`.
- **`ngrams(vectors, n)`** — bound, position-permuted n-grams, then bundled.
- **`bundle_sequence(v)`** vs **`bind_sequence(v)`** — additive vs multiplicative order encoding.
- **`fractional_power(x, p)`** — continuous power binding, basis for fractional-power encoding (FPE).
- **`resonator(codebooks, target)`** — iterative factoriser. Current implementation is a skeleton; the v0.3 roadmap expands this.
- **`cross_product(A, B, bind_fn)`** — all pairwise bindings, shape `(n, m, d)`.

---

## 14. Similarity metrics beyond cosine / Hamming

- **`jaccard_similarity(x, y)`** — `|x ∩ y| / |x ∪ y|` on binary.
- **`tversky_similarity(x, y, α, β)`** — asymmetric Jaccard; useful for prototype-vs-exemplar.
- **`matching_similarity(x, y)`** — fraction of equal components; used by CGR.
- **`phasor_similarity(x, y, q)`** — real part of normalised phasor inner product; used by MCR.
- **`dot_similarity(x, y)`** — raw `Σ xᵢ yᵢ`; used internally by Graph.

Similarity choice is tied to VSA model; `similarity` on the model class dispatches to the right one.

---

## 15. Noise & quantization primitives

- **`flip_fraction(x, p, key)`** — randomly flip `p` fraction of bits in a binary vector (BSC channel noise model).
- **`add_noise_map(x, σ, key)`** — additive Gaussian noise on MAP vectors.
- **`soft_quantize(x, q)`** — differentiable q-level quantisation via tanh.
- **`hard_quantize(x, q)`** — straight quantisation (non-differentiable).
- **`threshold(x, τ)`** — bipolar threshold.
- **`window(x, lo, hi)`** — clip to a range.

These enable noise-budget benchmarks (v0.3 roadmap) and future quantisation-aware learning (v0.2 roadmap).

---

## 16. Testing & coverage

```
tests/
├── conftest.py            # fixtures
├── test_vsa.py            # 8 models × primitive ops
├── test_functional.py     # every function in functional.py
├── test_embeddings.py     # 5 encoders
├── test_models.py         # 5 classifiers, fit / predict / score
├── test_memory.py         # 3 memory modules
├── test_structures.py     # 4 data structures
├── test_metrics.py        # capacity, SNR, participation
├── test_integration.py    # end-to-end pipelines
├── test_performance.py    # benchmark smoke tests (marked)
└── test_utils.py          # normalize, benchmark
```

Numbers: **297 tests, 99 % line coverage, 20 s full run.**
The 3 uncovered lines (`_compat.py:10–13`) are the `except TypeError` path
for JAX < 0.4.25 and will never hit on current JAX.

---

## 17. Dev commands

```bash
pytest tests/ -v                              # 297 tests in ~20 s
pytest tests/ --cov=bayes_hdc --cov-report=html # HTML coverage report
ruff check bayes_hdc/                           # lint (clean)
ruff format bayes_hdc/                          # format
mypy bayes_hdc/                                 # type check (clean)
python examples/basic_operations.py           # bind / bundle / permute demo
python examples/kanerva_example.py            # analogy: king = queen · man · woman⁻¹
python examples/classification_simple.py      # encoder → classifier pipeline
```

---

## 18. PVSA layer 1 — `distributions.py`

The algebra of Bayesian hypervectors. Two distribution types today; more on the roadmap.

**`GaussianHV(mu, var, dimensions)`** — diagonal-covariance Gaussian over $\mathbb{R}^d$.

| op | formula | closed-form? |
|---|---|---|
| `bind_gaussian(x, y)` | $\mu_z = \mu_x \mu_y$, $\sigma_z^2 = \mu_x^2 \sigma_y^2 + \mu_y^2 \sigma_x^2 + \sigma_x^2 \sigma_y^2$ | ✅ exact |
| `bundle_gaussian(hvs)` | $\mu = \sum \mu_i / \|\!\sum\mu_i\|$, $\sigma^2 = \sum \sigma_i^2 / \|\!\sum\mu_i\|^2$ | ✅ exact |
| `expected_cosine_similarity(x, y)` | plug-in at the means | approximate |
| `similarity_variance(x, y)` | $\sum_i (\mu_{x,i}^2 \sigma_{y,i}^2 + \ldots)$ | ✅ exact |
| `kl_gaussian(p, q)` | standard diagonal-Gaussian KL | ✅ exact |

**`DirichletHV(alpha, dimensions)`** — distribution over $\Delta_K$ for Bayesian categorical codebooks.

| op | formula | closed-form? |
|---|---|---|
| `bind_dirichlet(x, y)` | means multiplied and renormalised; concentrations summed | approximate |
| `bundle_dirichlet(hvs)` | concentrations summed (exact posterior update) | ✅ exact |
| `kl_dirichlet(p, q)` | standard Dirichlet KL (digamma / log-gamma) | ✅ exact |

Classical VSA is the zero-variance / point-mass limit of either.

---

## 19. PVSA layer 2 — `uncertainty.py`

Two post-hoc wrappers that turn any classifier producing logits into an uncertainty-aware one.

**`TemperatureCalibrator`** — fit $T > 0$ by minimising NLL in log-space with L-BFGS (Guo et al. 2017). Accuracy-preserving. Returns calibrated probabilities. Empirically reduces ECE **5–25×** on this library's benchmark.

**`ConformalClassifier(alpha)`** — split-conformal APS (Romano et al. 2020):

- `fit(probs_cal, labels_cal)` → learns quantile $\hat{q}$.
- `predict_set(probs)` → boolean mask of admitted classes; always non-empty (top-1 is always included).
- `coverage(probs_te, y_te)` → empirical coverage, guaranteed $\geq 1 - \alpha$ on exchangeable data.
- `set_size(probs_te)` → mean number of classes per prediction.

Both are JAX pytrees; `jit`/`vmap`/`grad` compose through them.

---

## 20. PVSA layer 3 — calibration metrics in `metrics.py`

Everything needed to *measure* calibration quality, JIT-compiled:

- `expected_calibration_error(probs, labels, n_bins=15)` — weighted mean gap between confidence and accuracy.
- `maximum_calibration_error` — worst-bin gap.
- `brier_score(probs, labels, n_classes)` — mean squared error vs one-hot.
- `sharpness(probs)` — mean top-1 confidence.
- `negative_log_likelihood(probs, labels)` — the objective temperature scaling minimises.
- `reliability_curve(probs, labels, n_bins)` — the four arrays needed to draw a reliability diagram (centers, accuracies, confidences, counts).

Use these to audit any classifier's calibration, including baselines from other HDC libraries.

---

## 21. Empirical headline

From `python benchmarks/benchmark_calibration.py`, D=10 000, 2 epochs of `AdaptiveHDC`:

**ECE reduction under temperature scaling (Bayes-HDC):**

| dataset | ECE raw | ECE + T | factor |
|---|---|---|---|
| iris | 0.523 | 0.081 | 6.5× |
| wine | 0.498 | 0.111 | 4.5× |
| digits | 0.792 | **0.039** | **20×** |
| MNIST | 0.683 | **0.027** | **25×** |

**Conformal coverage at α = 0.1 — empirical coverage (target ≥ 0.90):**

| dataset | coverage | mean set size |
|---|---|---|
| iris | 1.000 | 2.98 |
| wine | 0.981 | 1.63 |
| breast-cancer | 0.947 | 1.00 |
| digits | 0.996 | 4.62 |
| MNIST | 0.992 | 3.91 |

Every dataset clears the 90% coverage target. Set size tracks task difficulty (binary → 1, 10-class → 4).

---

## 22. Where PVSA goes beyond every existing HDC library

Already shipping in bayes-hdc, not in any other public HDC library:

- **Bayesian hypervector algebra** (`GaussianHV`, `DirichletHV` with closed-form bind/bundle/KL).
- **Calibrated prediction** (`TemperatureCalibrator` with L-BFGS in log-space).
- **Coverage-guaranteed prediction sets** (`ConformalClassifier` with APS scores).
- **Calibration metrics as JIT primitives** (ECE / MCE / Brier / NLL / reliability curves).
- **Capacity-and-noise toolkit** in `metrics.py`.
- **Resonator network skeleton** in `functional.py`.
- **Fractional-power binding**, Tversky / Jaccard similarities.
- **Stateless, pytree-native design** — every classifier / memory / structure is a pytree and works through `jit` / `vmap` / `pmap` / `grad` without wrappers.

Covers the same deterministic VSA layer the HDC literature defines — eight algebraic models, encoders, classifiers, memory, structures — implemented directly from the primary papers (Kanerva, Plate, Gayler, Rahimi, Ramsauer) rather than ported from another library. See `ORIGINALITY.md` for the per-component attribution. Still missing: a `datasets` submodule with the 14 HDC-standard benchmarks (v1.0 roadmap).

---

## 23. Reading this library in 30 minutes

Order to touch the files:

1. `bayes_hdc/functional.py` — the deterministic substrate.
2. `bayes_hdc/vsa.py` — the 8 models wrapping functional; note `@register_dataclass`.
3. `bayes_hdc/distributions.py` — **the PVSA layer**; `GaussianHV` first, then `DirichletHV`.
4. `bayes_hdc/uncertainty.py` — `TemperatureCalibrator` and `ConformalClassifier`.
5. `bayes_hdc/metrics.py` — capacity + calibration; this is the analysis toolkit.
6. `bayes_hdc/models.py` — classifiers; start with `CentroidClassifier`, then `AdaptiveHDC`.
7. `bayes_hdc/embeddings.py` — five encoders; `RandomEncoder` is canonical.
8. `bayes_hdc/memory.py` + `bayes_hdc/structures.py` — short, same pytree pattern.
9. `benchmarks/benchmark_calibration.py` — see the pipeline end-to-end.
10. `examples/*.py` + `tests/test_distributions.py` — sanity-check your reading against runnable code.

If you can answer `QUIZ.md` without opening the source, you are fluent.

---

## 24. Roadmap at a glance (see README for detail)

1. **v0.2** ✅ — PVSA foundation: `GaussianHV`, closed-form moment propagation.
2. **v0.3** ✅ — `DirichletHV`, calibration metrics, `benchmark_calibration.py`.
3. **v0.4** ✅ — `TemperatureCalibrator`, `ConformalClassifier`, head-to-head vs TorchHD.
4. **v0.5** — `MixtureHV`; `inverse_gaussian`, `permute_gaussian`; reparameterisation gradients.
5. **v0.6** — `BayesianCentroidClassifier`, `BayesianAdaptiveHDC`; ELBO for variational codebooks.
6. **v0.7** — `pmap` / `shard_map` kernels; streaming Bayesian updates.
7. **v1.0** — `bayes_hdc.datasets` (14 HDC-standard sets); JMLR MLOSS submission.
