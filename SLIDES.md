# jax-hdc: library walkthrough

A deck for reading the `jax-hdc` codebase with confidence. All file
references point to `jax_hdc/` unless noted. Slides are separated by `---`
so the file renders as plain Markdown and as a Marp / Slidev deck.

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
jax_hdc/
├── functional.py     # low-level ops (bind_*, bundle_*, similarity, resonator, ...)
├── vsa.py            # 8 VSA model classes (BSC, MAP, HRR, FHRR, BSBC, CGR, MCR, VTB)
├── embeddings.py     # 5 encoders (Random, Level, Projection, Kernel, Graph)
├── models.py         # 5 classifiers (Centroid, Adaptive, LVQ, RegLS, Clustering)
├── memory.py         # 3 memory modules (SDM, Hopfield, Attention)
├── structures.py     # 4 data structures (Multiset, HashTable, Sequence, Graph)
├── metrics.py        # 8 analysis funcs (capacity, SNR, sparsity, ...)
├── utils.py          # normalize, benchmark_function
├── constants.py      # EPS = 1e-8
└── _compat.py        # register_dataclass fallback for older JAX
```

1 129 statements of source, 297 tests, 99 % coverage.

---

## 3. The public API surface

`from jax_hdc import ...` exposes:

- **VSA models**: `BSC, BSBC, MAP, HRR, FHRR, CGR, MCR, VTB`
- **Encoders**: `RandomEncoder, LevelEncoder, ProjectionEncoder, KernelEncoder, GraphEncoder`
- **Classifiers**: `CentroidClassifier, AdaptiveHDC, LVQClassifier, RegularizedLSClassifier, ClusteringModel`
- **Memory**: `SparseDistributedMemory, HopfieldMemory, AttentionMemory`
- **Structures**: `Multiset, HashTable, Sequence, Graph`
- **Functional ops**: `bind_*, bundle_*, inverse_*, permute, cleanup, resonator, hash_table, ngrams, bundle_sequence, graph_encode, ...`
- **Similarity**: `cosine_similarity, hamming_similarity, dot_similarity, jaccard_similarity, tversky_similarity, matching_similarity, phasor_similarity`
- **Metrics**: `bundle_snr, bundle_capacity, effective_dimensions, sparsity, saturation, signal_energy, cosine_matrix, retrieval_confidence`
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

This module is a differentiator from TorchHD — capacity and noise tooling
that practitioners normally re-derive per paper:

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
pytest tests/ --cov=jax_hdc --cov-report=html # HTML coverage report
ruff check jax_hdc/                           # lint (clean)
ruff format jax_hdc/                          # format
mypy jax_hdc/                                 # type check (clean)
python examples/basic_operations.py           # bind / bundle / permute demo
python examples/kanerva_example.py            # analogy: king = queen · man · woman⁻¹
python examples/classification_simple.py      # encoder → classifier pipeline
```

---

## 18. Where this goes beyond TorchHD today

Already shipping in jax-hdc, not in TorchHD:

- **`metrics.py`** — capacity, SNR, participation ratio, retrieval confidence as first-class APIs.
- **`resonator` skeleton** in `functional.py` — iterative factorisation; TorchHD does not ship one.
- **Fractional-power binding** — continuous exponent on MAP / HRR.
- **Stateless, pytree-native design** — every classifier / memory / structure is a pytree, works through `jit` / `vmap` / `pmap` / `grad` without wrappers.
- **Tversky and Jaccard similarities** as library primitives.

Matched with TorchHD:

- 8 identical VSA models, identical core API surface.
- Encoders, classifiers, memory modules, symbolic structures.

---

## 19. What TorchHD has that we don't — yet

- **`torchhd.datasets`** — 14+ standardised HDC benchmarks. (v1.0 roadmap.)
- **Packaged in `torchhd.tensors`** — a specific tensor-subclass design we mirror with dataclasses.

Everything else in the roadmap (differentiable primitives, factorisation toolkit, distributed kernels, probabilistic HDC, neuro-symbolic reasoning, published benchmark suite) is **net-new for the HDC ecosystem**.

---

## 20. Reading this library in 30 minutes

Order to touch the files:

1. `jax_hdc/functional.py` — top-to-bottom; this is the substrate.
2. `jax_hdc/vsa.py` — how the 8 models wrap it; note the `@register_dataclass` pattern.
3. `jax_hdc/embeddings.py` — `RandomEncoder` is the canonical example; the rest parallel it.
4. `jax_hdc/models.py` — `CentroidClassifier` first; `ClusteringModel` if you want to see the iterative pattern.
5. `jax_hdc/memory.py` + `jax_hdc/structures.py` — both short, same pytree pattern.
6. `jax_hdc/metrics.py` — this is the paper-shaped differentiator; understand every function.
7. `examples/*.py` — read the three of them; they are your REPL.
8. `tests/test_vsa.py` — read one test file end-to-end so you know the invariants the code maintains.

If you can answer `QUIZ.md` without opening the code, you are fluent.

---

## 21. Roadmap at a glance (see README for detail)

1. **v0.2** — differentiable primitives + learnable codebooks
2. **v0.3** — factorisation & resonator network toolkit
3. **v0.4** — distributed (`pmap`/`shard_map`) & streaming
4. **v0.5** — probabilistic hypervectors
5. **v0.6** — neuro-symbolic reasoning (SME, Raven's, SCAN, COGS)
6. **v1.0** — `jax_hdc.datasets`, TorchHD head-to-head, JMLR MLOSS submission
