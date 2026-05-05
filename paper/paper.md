---
title: 'bayes-hdc: Probabilistic Vector Symbolic Architectures and Calibrated Hyperdimensional Computing in JAX'
tags:
  - Python
  - JAX
  - hyperdimensional computing
  - vector symbolic architectures
  - Bayesian machine learning
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

# Summary

`bayes-hdc` is a JAX [@jax2018github] library for hyperdimensional computing (HDC) and vector symbolic architectures (VSA) [@gayler2003vsa; @kanerva2009hyperdimensional] that pairs a comprehensive re-implementation of eight classical VSA models — BSC, MAP, HRR, FHRR, BSBC, CGR, MCR, and VTB — with a built-in *probabilistic* layer. The probabilistic layer, **PVSA** (*Probabilistic Vector Symbolic Architectures*), introduces Gaussian and Dirichlet hypervector types whose first and second moments propagate analytically through `bind`, `bundle`, `permute`, and `cleanup` (building on the SSP / fractional-binding theoretical line of @furlong2024probabilistic, here packaged as a library API for the first time); closed-form Kullback–Leibler divergences between hypervector posteriors; reparameterisation gradients for variational training of codebooks and classifier posteriors; runtime equivariance verifiers for the cyclic-shift action of $\mathbb{Z}/d$ (building on the shift-equivariance analysis of @rachkovskij2024shiftequivariance); temperature-scaling calibration [@guo2017calibration]; and split-conformal prediction sets with finite-sample coverage guarantees [@romano2020classification], operationalising the algorithmic conformal-HDC direction concurrently developed by @liang2026conformalhdc as a JAX-native, JIT-compiled component. Every type is registered as a JAX pytree, so `jit`, `vmap`, `grad`, `pmap`, and `shard_map` compose with every operation on CPU, GPU, and TPU without user-side flattening. The library ships 14 application examples spanning EMG gesture recognition, EEG seizure detection, character-trigram language identification, and the Kanerva "Dollar of Mexico" analogical-mapping benchmark, alongside 510 unit tests at 93 % line coverage.

# Statement of need

Existing open-source HDC software is dominated by `TorchHD` [@heddes2023torchhd], a mature PyTorch library covering the same eight VSA models. Two narrower JAX-backed libraries exist (`hyper-jax`, MAP only; `hrr`, HRR only), but no comprehensive JAX-native HDC library spans the full primitive set. `TorchHD` does not expose a probabilistic layer: its hypervectors carry no posterior variance, its classifiers have no calibration step, and it has no conformal-prediction module. Concurrent algorithmic work on conformal HDC [@liang2026conformalhdc] and probabilistic SSP [@furlong2024probabilistic] does not ship libraries; there is room for a library that *operationalises* these directions as JAX-native components. Two needs follow.

First, **JAX-native HDC**. The JAX research ecosystem — including BlackJAX, NumPyro, Flax, Equinox, Optax, and Dynamax — has converged on `jit` / `vmap` / `grad` / `pmap` / `shard_map` as the substrate for differentiable scientific computing. None of those tools compose with `TorchHD`'s in-place tensor methods. A practitioner who wants to embed a VSA codebook inside a probabilistic-programming model, run a HMC sampler over codebook posteriors, or differentiate a hyperdimensional classifier through a downstream loss currently has no first-class option. `bayes-hdc` fills that gap: every primitive is a pure JAX function on registered pytrees.

Second, **uncertainty-aware retrieval and classification**. HDC's classical bind / bundle / cleanup pipeline is well calibrated only by accident. For safety-relevant deployments — biomedical signal classification, edge robotics, anomaly detection — the practitioner needs quantified epistemic and aleatoric uncertainty. `bayes-hdc` propagates exact Gaussian moments through bind and bundle, exposes posterior variance as a first-class output of `predict_with_uncertainty`, and produces split-conformal prediction sets with proven $\geq 1-\alpha$ marginal coverage on exchangeable data. No competing HDC library offers this stack.

# State of the field

The HDC ecosystem has four maintained general libraries. `TorchHD` [@heddes2023torchhd] is dominant — PyTorch backend, JMLR-published, the same eight VSA models. `hdlib` [@cumbo2023hdlib] targets bioinformatics on a NumPy backend with a domain-application focus rather than research-primitive depth. `vsapy` is a NumPy library specialising in number-line and circular encodings. `NengoSPA` [@bekolay2014nengo] is the spiking-neuron lineage of HDC from the Eliasmith group, embedded in the Nengo neural simulator; it is biologically faithful but disjoint from machine-learning workflows. Two narrower JAX-backed packages exist (`hyper-jax`, MAP only; `hrr`, multi-backend with a JAX option) but neither covers the full primitive set or the probabilistic layer.

Algorithmic prior and concurrent work on the *probabilistic* and *uncertainty-aware* directions exists on the paper side without released libraries: probabilistic VSA via SSP / fractional binding [@furlong2024probabilistic], split-conformal prediction adapted to HDC prototypes [@liang2026conformalhdc], variational HDC encoders via VQ-VAE-style codebooks [@bryant2024hdvqvae; @nesygems2023hdvae]. `bayes-hdc` differs from `TorchHD`, `hdlib`, and `vsapy` on the JAX backend and the probabilistic / conformal stack, and from `NengoSPA` on machine-learning integration. It is, to our knowledge, the first comprehensive JAX-native HDC library shipping closed-form Gaussian moment propagation, end-to-end variational codebook training, and split-conformal coverage as built-in modules.

# Software design

The library is structured as a layered pytree-native algebra. At the bottom sit the eight classical VSA models in `bayes_hdc.vsa` and the corresponding pure functions in `bayes_hdc.functional`. Above that, `bayes_hdc.distributions` introduces `GaussianHV`, `DirichletHV`, and `MixtureHV` — frozen dataclasses registered with `jax.tree_util.register_dataclass`. The probabilistic primitives `bind_gaussian`, `bundle_gaussian`, `permute_gaussian`, `cleanup_gaussian`, and `inverse_gaussian` propagate exact moments under independence assumptions:

$$
\mathbb{E}[x \cdot y] = \mu_x \cdot \mu_y, \qquad
\mathrm{Var}[x \cdot y] = \mu_x^2 \sigma_y^2 + \mu_y^2 \sigma_x^2 + \sigma_x^2 \sigma_y^2.
$$

`bayes_hdc.equivariance` exposes the cyclic-shift action of $\mathbb{Z}/d$ on $\mathbb{R}^d$ as a first-class group object, distinguishes diagonal-equivariance (element-wise binding) from single-argument-equivariance (circular convolution), and ships property-based verifiers that reject any user-defined operation claiming a symmetry it does not have. `bayes_hdc.bayesian_models` provides `BayesianCentroidClassifier`, `BayesianAdaptiveHDC`, and `StreamingBayesianHDC` — closed-form, online, and bounded-memory variants respectively. `bayes_hdc.uncertainty` ships `TemperatureCalibrator` (a one-parameter L-BFGS calibrator with a unique global minimum) and `ConformalClassifier` (split-conformal APS sets). `bayes_hdc.resonator` provides a multi-restart MCMC factorisation procedure (`probabilistic_resonator`) of which the deterministic Frady–Kleyko resonator network [@frady2020resonator] is the zero-temperature limit.

A central design commitment is that every public type is *immutable, pytree-native, and JAX-functional*. There is no hidden state, no in-place mutation, and no Python control flow that would prevent `jit` compilation. The `pmap` and `shard_map` wrappers in `bayes_hdc.distributed` degrade gracefully on single-device hosts, so the same code runs on a laptop CPU and on a TPU pod.

# Research impact statement

`bayes-hdc` lowers the barrier to four research directions that are difficult on existing HDC stacks. (1) *Variational HDC*: end-to-end reparameterisation gradients enable training of codebooks and classifier posteriors as ELBO objectives. (2) *Calibrated edge inference*: the conformal layer produces prediction sets with finite-sample coverage on exchangeable streams, suitable for biomedical and IoT deployments. (3) *Equivariant neural functionals*: the cyclic-shift verifier surfaces in the same module as the classical permutation primitive, allowing weight-space symmetry analysis to compose with VSA primitives. (4) *Faithful literature reproduction*: every primitive cites its primary source, and the `docs/audit/` directory ships per-paper attribution reports for the 18 foundational HDC works that the library implements. The repository has been used internally to reproduce Kanerva's analogical-mapping example [@kanerva2010dollar], the Gayler–Levy 2009 distributed analogy benchmark [@gayler2009distributed], and the Frady et al. resonator factorisation [@frady2020resonator]. The MIT-licensed source, 498-test suite, and tagged release pipeline (TestPyPI then PyPI via OIDC) make the library suitable for both downstream research and classroom use.

# AI usage disclosure

Generative AI assistants (specifically, Anthropic's Claude) were used during library development as a coding assistant for boilerplate generation, refactoring, and the literature-attribution audit (`docs/LITERATURE_AUDIT.md`). All algorithmic decisions, mathematical derivations, citations, and benchmark interpretations were authored, reviewed, and verified by the human author. AI-suggested code and prose were treated as drafts requiring human verification before commit; no tool produced text that was checked in unread.

# Acknowledgements

The author thanks the JAX, BlackJAX, and Equinox developer communities for the substrate that made this library possible, and Pentti Kanerva, Tony Plate, Ross Gayler, and the broader HDC/VSA community whose foundational work `bayes-hdc` rests on.

# References
