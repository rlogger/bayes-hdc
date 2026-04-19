<p align="center">
    <a href="https://github.com/rlogger/bayes-hdc/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat" /></a>
    <img alt="Development Status" src="https://img.shields.io/badge/status-alpha-orange.svg?style=flat" />
    <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue.svg?style=flat" />
</p>

# Bayes-HDC

**A Bayesian framework for Hyperdimensional Computing. Built on JAX.**

Bayes-HDC is the first HDC library where every hypervector can carry a
distribution. Classical VSAs represent symbols as single
high-dimensional points; Bayes-HDC represents them as posteriors —
Gaussian today, Dirichlet and mixture tomorrow — so that binding,
bundling, and retrieval propagate calibrated uncertainty end-to-end.

On top of that Bayesian core, the library ships a complete deterministic
VSA foundation: eight classical models (BSC, MAP, HRR, FHRR, BSBC, CGR,
MCR, VTB), five encoders, five classifiers, three memory modules, four
symbolic data structures, and a capacity-and-noise analysis toolkit. The
deterministic layer is the baseline; the Bayesian layer is the research
contribution.

All operations run unchanged on CPU, GPU, and TPU via JAX's XLA backend.
Every type is a JAX pytree, so `jit`, `vmap`, `grad`, and `pmap` compose
with the whole library out of the box.

## The Bayesian core

```python
import jax
from bayes_hdc import GaussianHV, bind_gaussian, expected_cosine_similarity

key = jax.random.PRNGKey(0)
x = GaussianHV.random(key, dimensions=10_000, var=0.01)
y = GaussianHV.random(jax.random.fold_in(key, 1), dimensions=10_000, var=0.01)

z = bind_gaussian(x, y)          # exact moment propagation
sim = expected_cosine_similarity(x, z)  # uncertainty-aware similarity
```

Every distributional operation has closed-form moment propagation where
possible and a Monte Carlo fallback otherwise. Deterministic pipelines
compose by lifting: `GaussianHV.from_sample(hv)` wraps any existing
hypervector in a zero-variance posterior that behaves identically to a
classical VSA model until you start injecting uncertainty.

## Roadmap

### v0.2 — Bayesian hypervector foundation ✅ (this release)
- [x] `GaussianHV` with mean and diagonal variance
- [x] `bind_gaussian` — exact moment propagation under element-wise product
- [x] `bundle_gaussian` — exact sum of independent Gaussians + normalisation
- [x] `expected_cosine_similarity`, `similarity_variance`
- [x] `kl_gaussian` — closed-form KL for variational objectives
- [x] `sample` / `sample_batch` for Monte Carlo fallbacks

### v0.3 — Probabilistic VSA operations
- [ ] `DirichletHV` for probabilistic categorical codebooks
- [ ] `MixtureHV` for multi-modal representations
- [ ] `inverse_gaussian`, `permute_gaussian`, `cleanup_gaussian`
- [ ] Reparameterisation gradients through every distributional op
- [ ] Low-rank Gaussian parameterisation for richer posteriors

### v0.4 — Bayesian learning models
- [ ] `BayesianCentroidClassifier` — posterior per class, calibrated predictions
- [ ] `BayesianAdaptiveHDC` with Kalman-style online updates
- [ ] `ConformalClassifier` wrapper — coverage-guaranteed prediction sets
- [ ] Temperature-scaled and Platt-scaled retrieval

### v0.5 — Inference and diagnostics
- [ ] ELBO optimisation for variational codebooks
- [ ] Probabilistic resonator networks (MCMC / multi-restart)
- [ ] Posterior predictive checks, coverage curves, reliability diagrams
- [ ] Brier score, ECE, MCE, and sharpness metrics in `bayes_hdc.metrics`

### v0.6 — Distribution and scale
- [ ] `pmap` / `shard_map` kernels for every distributional op
- [ ] Sharded posteriors across TPU pods with zero-copy transfer
- [ ] Streaming Bayesian updates with bounded memory

### v1.0 — Datasets, benchmarks, paper
- [ ] `bayes_hdc.datasets` module matching TorchHD's 14+ standard HDC benchmarks
- [ ] Head-to-head vs TorchHD on accuracy and throughput (deterministic parity)
- [ ] Head-to-head vs TorchHD + temperature scaling on expected calibration error (Bayesian contribution)
- [ ] Seeded, containerised runs with fixed hardware profiles
- [ ] JMLR MLOSS submission

## Features

- XLA compilation and automatic kernel fusion through JAX
- Native GPU/TPU support
- Functional design compatible with JAX transformations (`jit`, `vmap`, `pmap`)
- Eight VSA models: BSC, MAP, HRR, FHRR, BSBC, CGR, MCR, VTB
- Encoders for discrete, continuous, kernel, and graph data
- Classification models: centroid, adaptive, LVQ, regularized least squares
- Memory modules: SDM, Hopfield networks, attention-based retrieval

## Installation

```bash
git clone https://github.com/rlogger/bayes-hdc.git
cd bayes-hdc
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import jax
from bayes_hdc import MAP, RandomEncoder, CentroidClassifier

model = MAP.create(dimensions=10000)
key = jax.random.PRNGKey(42)

# Bind and bundle
x = model.random(key, (10000,))
y = model.random(key, (10000,))
bound = model.bind(x, y)
bundled = model.bundle(jax.numpy.stack([x, y]), axis=0)

# Classification pipeline
encoder = RandomEncoder.create(
    num_features=20, num_values=10, dimensions=10000,
    vsa_model=model, key=key,
)
data = jax.random.randint(key, (100, 20), 0, 10)
labels = jax.random.randint(key, (100,), 0, 5)
encoded = encoder.encode_batch(data)

classifier = CentroidClassifier.create(
    num_classes=5, dimensions=10000, vsa_model=model,
)
classifier = classifier.fit(encoded, labels)
accuracy = classifier.score(encoded, labels)
```

## VSA Models

| Model | Description |
|-------|-------------|
| **BSC** | Binary Spatter Codes — XOR binding, majority bundling |
| **MAP** | Multiply-Add-Permute — element-wise multiply, normalized sum |
| **HRR** | Holographic Reduced Representations — circular convolution |
| **FHRR** | Fourier HRR — complex-valued, element-wise multiply |
| **BSBC** | Binary Sparse Block Codes — block-sparse binary |
| **CGR** | Cyclic Group Representation — modular addition binding |
| **MCR** | Modular Composite Representation — phasor arithmetic |
| **VTB** | Vector-Derived Transformation Binding — matrix multiplication |

All models share the same API: `bind`, `bundle`, `inverse`, `similarity`, `random`.

## Encoders

- **RandomEncoder** — discrete features via codebook lookup
- **LevelEncoder** — continuous values via level interpolation
- **ProjectionEncoder** — high-dimensional data via random projection
- **KernelEncoder** — RBF kernel approximation (Random Fourier Features)
- **GraphEncoder** — graph structures via node binding

## Classification Models

- **CentroidClassifier** — single-pass centroid prototypes
- **AdaptiveHDC** — iterative prototype refinement
- **LVQClassifier** — Learning Vector Quantization
- **RegularizedLSClassifier** — regularized least squares

## Memory Modules

- **SparseDistributedMemory** — content-addressable storage (Kanerva SDM)
- **HopfieldMemory** — modern Hopfield network with softmax attention
- **AttentionMemory** — scaled dot-product attention with multi-head support

## Development

```bash
pytest tests/ -v                              # run tests
pytest tests/ --cov=bayes_hdc --cov-report=html # with coverage
ruff check bayes_hdc/                           # lint
ruff format bayes_hdc/                          # format
mypy bayes_hdc/                                 # type check
```

## Examples

```bash
python examples/basic_operations.py      # core HDC operations
python examples/kanerva_example.py       # analogical reasoning
python examples/classification_simple.py # classification pipeline
```

## License

MIT — see [LICENSE](LICENSE).

## References

- Kanerva, P. (2009). "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors"
- Plate, T. A. (1995). "Holographic Reduced Representations"
- Gayler, R. W. (2003). "Vector Symbolic Architectures answer Jackendoff's challenges for cognitive neuroscience"
