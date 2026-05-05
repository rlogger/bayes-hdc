---
title: 'bayes-hdc: Differentiable, Uncertainty-Aware Hyperdimensional Computing in JAX'
tags:
  - Python
  - JAX
  - hyperdimensional computing
  - vector symbolic architectures
  - Bayesian deep learning
  - variational inference
  - conformal prediction
  - uncertainty quantification
authors:
  - name: Rajdeep Singh
    orcid: 0000-0000-0000-0000
    corresponding: true
    affiliation: 1
affiliations:
  - name: Department of Computer Science, University of Southern California, United States
    index: 1
    ror: 03taz7m60
date: 29 April 2026
bibliography: paper.bib
---

# Abstract

`bayes-hdc` is a JAX library for Hyperdimensional Computing (HDC) and
Vector Symbolic Architectures (VSA) that operationalises
*Probabilistic Vector Symbolic Architectures* (PVSA): an algebra in
which every hypervector is a posterior distribution and the primitives
propagate first and second moments in closed form. The library
combines (i) the eight classical VSA models (BSC, MAP, HRR, FHRR,
BSBC, CGR, MCR, VTB) under a uniform pytree-native API; (ii) Gaussian
and Dirichlet hypervector types with analytic moment propagation,
building on the SSP / fractional-binding theoretical line of
@furlong2024probabilistic and providing what is, to our knowledge, the
first open-source library exposing closed-form bind/bundle moments;
(iii) an end-to-end variational training module — distinct from
@bryant2024hdvqvae's static-codebook HDVQ-VAE in that the codebook
itself is *trained* via reparameterisation gradients — that is, to our
knowledge, the first such API in the open-source HDC ecosystem; (iv)
split-conformal prediction sets with finite-sample coverage guarantees,
operationalising the concurrent algorithmic proposal of
@liang2026conformalhdc as a JAX-native built-in; (v) runtime
equivariance verifiers for the cyclic-shift action of $\mathbb{Z}/d$,
extending the shift-equivariance theory of @rachkovskij2024shiftequivariance
into a property-based testing primitive; and (vi) a per-paper
literature-attribution audit trail for every primitive in the library.
We benchmark against `TorchHD` [@heddes2023torchhd]: pointwise eager-
mode speedups of 1.4×–3.5× on a CPU; ensemble accuracy is +3.9 % mean
across five canonical datasets; the conformal layer is, at the
implementation level, unique to `bayes-hdc`. The library is
MIT-licensed, ships 510 unit tests at 93 % line coverage, and runs
unmodified on CPU, GPU, and TPU.

# 1 Introduction

Hyperdimensional computing [@kanerva2009hyperdimensional] and Vector
Symbolic Architectures [@gayler2003vsa] represent symbols as
high-dimensional vectors and compose them with a small algebra:
*bind* (binding two hypervectors into a third dissimilar to both),
*bundle* (superposition into a vector similar to all summands),
*permute* (a fixed isometric reordering, used to encode order), and
*similarity*. The same primitives support classification [@rahimi2016hdc],
analogical mapping [@kanerva2010dollar], language modelling
[@jones2007beagle], factor recovery [@frady2020resonator], and
robotics control. The field has matured into the Kleyko et al. (2023)
two-part survey [@kleyko2023survey1; @kleyko2023survey2] which
catalogues over 200 application papers.

Despite this growth, deployed HDC inference remains *deterministic*:
hypervectors are point estimates, similarity is a scalar, and the
practitioner has no principled handle on uncertainty. For
safety-relevant deployments — biomedical signal classification, edge
robotics, anomaly detection — calibrated probability and coverage
guarantees are not optional. Existing HDC software does not provide
them. `TorchHD` [@heddes2023torchhd] (PyTorch) and `hdlib`
[@cumbo2023hdlib] (NumPy) expose only the deterministic substrate;
`NengoSPA` [@bekolay2014nengo] is biologically realistic but disjoint
from machine-learning workflows; no JAX-native HDC library exists.

`bayes-hdc` fills both gaps simultaneously. It introduces *Probabilistic
Vector Symbolic Architectures* (PVSA) — a generalisation of VSA in
which every hypervector carries a posterior — and packages the entire
stack as a pytree-native JAX library, so `jit`, `vmap`, `grad`, `pmap`,
and `shard_map` compose with every operation without user-side
flattening.

# 2 Related software

| Library | Backend | VSA models | Probabilistic / UQ | Diff. | Released |
|---|---|---:|---|---|---|
| TorchHD [@heddes2023torchhd] | PyTorch | 8 | — | partial | 2023 (mature) |
| hdlib [@cumbo2023hdlib] | NumPy | generic | — | — | 2023 |
| vsapy | NumPy | 5 | — | — | 2023 |
| NengoSPA [@bekolay2014nengo] | Nengo (spiking) | HRR, VTB | — | — | 2014 (active) |
| `hyper-jax` | JAX | 1 (MAP) | — | partial | 2024 |
| `hrr` | NumPy / Torch / TF / JAX | 1 (HRR) | — | partial | 2023 |
| **bayes-hdc** | **JAX** | **8** | **GaussianHV, DirichletHV, conformal sets** | **end-to-end** | 2026 |

`bayes-hdc` differs from each on a different axis. Against `TorchHD`,
on the JAX backend, the probabilistic algebra, and the conformal
prediction layer; against `hdlib` and `vsapy`, on speed,
autodifferentiation, and primitive coverage; against `NengoSPA`, on
machine-learning integration; against `hyper-jax` and `hrr`, on
breadth (full eight-model coverage rather than a single primitive).
To our knowledge `bayes-hdc` is the first comprehensive JAX-native
HDC library covering the eight VSA models, and the first open-source
HDC library shipping end-to-end variational training, split-conformal
prediction, and runtime equivariance verifiers as built-in modules.

On the *theoretical* side, the relevant prior and concurrent work
develops these directions on paper without released code: probabilistic
VSA via SSPs and fractional binding [@furlong2024probabilistic];
adaptive split-conformal scores for HDC prototypes
[@liang2026conformalhdc]; HDVQ-VAE [@bryant2024hdvqvae] which uses HDC
as a *static* binary codebook inside a VQ-VAE (the opposite of our
trained-codebook contribution); the Nesy-GeMs HD-VAE
[@nesygems2023hdvae]; and shift-equivariance of HDC sequence
encodings [@rachkovskij2024shiftequivariance]. `bayes-hdc`
operationalises this body of work as a JAX-native, JIT-compiled,
test-covered library.

# 3 Architecture

The library is layered.

**Classical VSA substrate.** `bayes_hdc.vsa` provides the eight
canonical models — Binary Spatter Codes [@kanerva1997fdr],
Multiply-Add-Permute [@gayler2003vsa], Holographic Reduced
Representations [@plate1995hrr; @plate2003hrr], Fourier Holographic
Reduced Representations, Binary Sparse Block Codes, Modular Composite
Representations, Cyclic Group Representations, and Vector-Tensor
Binding — under a uniform `bind / bundle / inverse / similarity /
random` interface. Pure functions live in `bayes_hdc.functional`.

**Probabilistic algebra (PVSA).** `bayes_hdc.distributions` introduces
`GaussianHV` (mean + diagonal variance), `DirichletHV` (probability
simplex), and `MixtureHV`, each registered as a JAX pytree. The
primitives `bind_gaussian`, `bundle_gaussian`, `permute_gaussian`,
`cleanup_gaussian`, and `inverse_gaussian` propagate exact first and
second moments under independence assumptions. For Gaussian inputs:
$$
\mathbb{E}[x \cdot y] = \mu_x \cdot \mu_y, \qquad
\mathrm{Var}[x \cdot y] = \mu_x^2 \sigma_y^2 + \mu_y^2 \sigma_x^2 + \sigma_x^2 \sigma_y^2,
$$
$$
\mathbb{E}\Bigl[\sum_i x_i\Bigr] = \sum_i \mu_i, \qquad
\mathrm{Var}\Bigl[\sum_i x_i\Bigr] = \sum_i \sigma_i^2.
$$
Closed-form Kullback–Leibler divergences `kl_gaussian` and
`kl_dirichlet` enable variational objectives.

**End-to-end variational training.** `bayes_hdc.training` exposes a
minimal Adam optimiser (`adam_init` / `adam_update`) and a high-level
`train_variational_codebook` loop that compiles via `jax.lax.scan` so
the full training run lowers to one XLA program. Because every PVSA
primitive is a pure JAX function on a registered pytree, `jax.grad`
composes through `bind`, `bundle`, `permute`, `cleanup`, the closed
form `kl_gaussian`, and the reparameterised sampler. This enables
end-to-end variational learning of probabilistic codebooks — a
capability not present in any other open-source HDC library, to our
knowledge.

**Equivariance verifiers.** `bayes_hdc.equivariance` exposes the
cyclic-shift action of $\mathbb{Z}/d$ as a first-class group object,
distinguishes diagonal-equivariance from single-argument-equivariance
(circular convolution), and ships property-based verifiers that reject
user-defined operations claiming a symmetry they do not have.

**Calibration and coverage.** `bayes_hdc.uncertainty` ships
`TemperatureCalibrator` — a one-parameter L-BFGS calibrator with a
unique global minimum [@guo2017calibration] — and
`ConformalClassifier`, a split-conformal Adaptive-Prediction-Set
classifier [@romano2020classification] whose output sets satisfy the
finite-sample coverage guarantee
$\mathbb{P}(y \in \hat C(x)) \geq 1 - \alpha$ on exchangeable data,
independent of model class, dimension, or training quality.

**Memory primitives.** `bayes_hdc.memory` provides three retrievers:
classical Sparse Distributed Memory [@kanerva1997fdr], a modern
continuous Hopfield network [@ramsauer2020hopfield] usable as a soft
cleanup memory, and a generic key-value attention memory.

**Probabilistic factorisation.** `bayes_hdc.resonator` provides a
multi-restart MCMC factorisation of a composite PVSA hypervector. The
deterministic Frady–Kleyko resonator network [@frady2020resonator] is
recovered exactly as the zero-temperature limit of this implementation:
``probabilistic_resonator(temperature=0)`` is the canonical
deterministic algorithm; positive temperatures interpolate
continuously between MCMC and the original Frady–Kleyko update. To our
knowledge this generalisation has not been packaged in a prior
library.

# 4 Empirical evaluation

We evaluate on five canonical HDC benchmarks (iris, wine,
breast-cancer, digits, MNIST) and benchmark wall-clock primitives
against `TorchHD`. Reproduction scripts under `benchmarks/` use
identical preprocessing, dimension ($d = 10\,000$), warmup (20),
and trial count (200).

**Accuracy** (`bayes-hdc` ensemble vs. `TorchHD` centroid):

| Dataset | n | Bayes-HDC | TorchHD | $\Delta$ |
|---|---:|---:|---:|---:|
| iris | 150 | **0.933** | 0.911 | +2.2 |
| wine | 178 | **0.852** | 0.815 | +3.7 |
| breast-cancer | 569 | **0.959** | 0.953 | +0.6 |
| digits | 1 797 | **0.943** | 0.900 | +4.3 |
| MNIST | 10 000 | **0.946** | 0.857 | +8.9 |
| mean $\Delta$ | | | | **+3.9** |

**Calibration** (Expected Calibration Error reduction under
temperature scaling):

| Dataset | ECE raw | ECE + T | reduction |
|---|---:|---:|---:|
| iris | 0.523 | **0.081** | 6.5× |
| wine | 0.498 | **0.111** | 4.5× |
| digits | 0.792 | **0.039** | **20×** |
| MNIST | 0.683 | **0.027** | **25×** |

**Conformal coverage** (split-conformal APS at $\alpha = 0.1$): all
five datasets clear the finite-sample coverage guarantee
($\geq 0.90$); set sizes scale with task difficulty. No comparable
capability exists in `TorchHD`.

**Wall-clock primitives** (CPU, $d = 10\,000$, eager-mode TorchHD; no
`torch.compile` baseline):

| Operation | bayes-hdc (ms) | TorchHD (ms) | Speedup |
|---|---:|---:|---:|
| MAP `bind` (2 HVs) | **0.009** | 0.012 | 1.41× |
| MAP `bundle` (10 HVs) | **0.025** | 0.053 | 2.11× |
| Cosine similarity | **0.021** | 0.075 | 3.48× |
| `RandomEncoder` (100×20) | 1.069 | **0.911** | 0.85× |

Pointwise operations are 1.4×–3.5× faster under JAX-`jit` than
TorchHD's eager kernels; the encoder result reverses on this CPU
configuration. The benchmark methodology returns the result as an
on-device tensor on both sides (no asymmetric host sync via `.item()`),
which is the one-line correction made for this submission relative to
earlier draft numbers. A `torch.compile` comparison is deferred to a
future suite. Accuracy and timing benchmarks are single-seed; the JSON
dumps under `benchmarks/` record exact configurations.

**Variational codebook recovery** (`examples/variational_codebook_learning.py`):
a 1024-dimensional `GaussianHV` posterior initialised at $\mu = 0$,
$\sigma^2 = 1$ recovers a target $\mu$-direction at cosine similarity
$0.9999$ in 500 Adam steps with negative-ELBO loss and a 32-sample
Monte-Carlo reconstruction term. The entire training trajectory
compiles to a single XLA program via `jax.lax.scan`.

# 5 Software engineering

The repository ships 510 unit tests at 93 % line coverage on 23
modules; CI runs the full matrix `ubuntu-latest × macos-latest × Python
{3.9, 3.10, 3.11, 3.12, 3.13}` on every push. Lint
(`ruff check`), format (`ruff format --check`), and type checks
(`mypy bayes_hdc/`) are clean. Documentation is built with Sphinx +
Furo under `-W` (warnings as errors) and deployed to GitHub Pages on
every push to `main`. CodeQL runs weekly; Dependabot bumps weekly.
Releases are tagged `v*.*.*` and published to TestPyPI then PyPI via
OIDC. The codebase is MIT-licensed, follows a documented contributor
ladder (see `COMMUNITY.md`), and ships a **per-paper literature
attribution audit** under `docs/audit/` — one Markdown report per
foundational HDC/VSA paper, mapping primitives to file:line locations
in the implementation. The audit (`docs/LITERATURE_AUDIT.md`) is the
strongest defence we know of against any "vibe-coded library"
characterisation: every algorithmic decision can be traced to a
primary source, and every primary source we depend on has been
re-read line-by-line during development.

# 6 Limitations and roadmap

The library currently focuses on the diagonal-Gaussian moment
parameterisation; richer posteriors (low-rank Gaussian, normalising-
flow hypervectors) are roadmap. Tensor-product binding [@smolensky1990tensor],
matrix-vector logic, and spiking-neuron VSA models [@bekolay2014nengo]
are out of scope and documented in `DESIGN.md` under "Related
approaches not implemented". Multi-device benchmarks beyond the
provided `pmap` / `shard_map` wrappers are deferred to the GPU and
TPU benchmark suite (containerised under `make docker-bench`).

# 7 Conclusion

`bayes-hdc` is the first comprehensive JAX-native HDC library covering
the eight canonical VSA models, the first to ship a built-in
probabilistic algebra with closed-form Gaussian moment propagation as
a library API, the first open-source library to operationalise
split-conformal coverage in HDC (concurrent with the algorithmic
proposals of @liang2026conformalhdc and the HDUQ-HAR line), and the
first to expose end-to-end gradient training of variational PVSA
codebooks. The library is production-grade (510 tests, 93 % coverage,
full CI matrix, deployed docs) and research-grade (per-paper
literature audit, 14 worked examples spanning EMG, EEG, NLP, and
analogical reasoning). Source, documentation, and benchmarks are at
<https://github.com/rlogger/bayes-hdc>.

# AI usage disclosure

Generative AI assistants (specifically, Anthropic's Claude) were used
during library development as a coding assistant for boilerplate
generation, refactoring, and the literature-attribution audit
(`docs/LITERATURE_AUDIT.md`). All algorithmic decisions, mathematical
derivations, citations, and benchmark interpretations were authored,
reviewed, and verified by the human author. No AI-suggested code or
prose was committed without human verification.

# Acknowledgements

The author thanks the JAX, BlackJAX, and Equinox developer communities
for the substrate that made this library possible, and Pentti Kanerva,
Tony Plate, Ross Gayler, Denis Kleyko, and the broader HDC/VSA research
community whose foundational work `bayes-hdc` rests on.

# References
