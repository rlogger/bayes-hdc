<p align="center">
    <a href="https://github.com/rlogger/jax-hdc/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat" /></a>
    <img alt="Development Status" src="https://img.shields.io/badge/status-alpha-orange.svg?style=flat" />
    <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue.svg?style=flat" />
</p>

# JAX-HDC

**A high-performance JAX library for Hyperdimensional Computing and Vector Symbolic Architectures**

JAX-HDC provides efficient implementations of Hyperdimensional Computing (HDC) and Vector Symbolic Architectures (VSA) using JAX. The library leverages XLA compilation, automatic vectorization, and hardware acceleration with a functional programming interface.

## Roadmap

### v0.2 — Differentiable VSA primitives
- [ ] Backprop through `bind`, `bundle`, `permute`, and `cleanup` for all eight VSA models
- [ ] Straight-through estimators for BSC, BSBC, and CGR
- [ ] Learnable codebooks, level tables, and projection matrices
- [ ] Flax, Equinox, and Optax interop

### v0.3 — Probabilistic hypervectors
- [ ] Posterior sampling over hypervector distributions
- [ ] Variational codebooks via reparameterization
- [ ] Conformal prediction for VSA classifiers
- [ ] Temperature calibration for similarity-based retrieval

### v0.4 — Distributed and streaming
- [ ] `pmap` and `shard_map` kernels for all VSA operations
- [ ] Sharded codebooks across accelerators
- [ ] Online classifiers with concept-drift handling
- [ ] Memory-mapped codebooks for >1B-symbol vocabularies

### v0.5 — Reasoning primitives
- [ ] Structure-mapping engine on VSAs
- [ ] Knowledge-graph embeddings and link prediction
- [ ] Rule induction on hypervectors
- [ ] Compositional generalization benchmarks (SCAN, COGS)

### v0.6 — Capacity and robustness analysis
- [ ] Capacity, crosstalk, and binding-noise probes per VSA model
- [ ] Analytic and empirical noise budgets
- [ ] Adversarial robustness audits
- [ ] Bundle decomposition and cleanup tracing

### v1.0 — Benchmark suite
- [ ] Reproducible head-to-head comparisons against TorchHD on 15+ datasets
- [ ] Accuracy, throughput, memory, and energy-per-inference metrics
- [ ] Seeded, containerized runs with fixed hardware profiles

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
git clone https://github.com/rlogger/jax-hdc.git
cd jax-hdc
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

### Nix

```bash
nix develop   # flakes
nix-shell     # traditional
```

## Quick Start

```python
import jax
from jax_hdc import MAP, RandomEncoder, CentroidClassifier

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
pytest tests/ --cov=jax_hdc --cov-report=html # with coverage
ruff check jax_hdc/                           # lint
ruff format jax_hdc/                          # format
mypy jax_hdc/                                 # type check
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
