# Originality & Attribution

`bayes-hdc` is an **independent implementation**. No code in the core
library (`bayes_hdc/`) is copied from any other hyperdimensional-computing
library. This document states that explicitly and credits the primary
research papers each piece is derived from.

## Independence check

- **No imports of other HDC libraries in the core library.** Verify with
  `rg 'import torchhd|from torchhd|import torch_hd|import hdlib|import pybhv' bayes_hdc/` — zero matches.
- **Benchmarks only.** `benchmarks/benchmark_*.py` import TorchHD
  because the benchmarks compare against it on identical workloads. The
  core library never depends on any other HDC package.
- **Tests only depend on this library.** No test imports another HDC
  library.

## Novel contributions (original to `bayes-hdc`)

These are defined for the first time in this project. Where related
work exists in the broader ML literature (e.g., temperature scaling on
neural classifiers), the implementation here is the first to apply the
technique inside an HDC library as a first-class primitive.

1. **Probabilistic Vector Symbolic Architectures (PVSA)** — an HDC
   algebra in which every hypervector is a posterior distribution and
   every VSA primitive propagates the posterior's moments in closed
   form. Introduced in this project.
2. **`GaussianHV`** — diagonal-covariance Gaussian hypervector type
   with closed-form `bind` (exact moment product), `bundle` (exact
   additive moment combination), and KL divergence.
3. **`DirichletHV`** — distribution on the probability simplex for
   probabilistic categorical codebooks, with exact bundle (posterior
   update by concentration sum), moment-matched bind, and closed-form
   KL.
4. **`TemperatureCalibrator`** for HDC — L-BFGS-in-log-space fit; first
   HDC library to ship a post-hoc calibrator as a library primitive.
5. **`ConformalClassifier`** for HDC — split-conformal with APS scores;
   first HDC library to ship coverage-guaranteed prediction sets.
6. **Calibration metrics for HDC** — `expected_calibration_error`,
   `maximum_calibration_error`, `brier_score`, `sharpness`,
   `negative_log_likelihood`, `reliability_curve` — first HDC library
   to provide these as JIT-compilable primitives.
7. **Capacity-and-noise toolkit** (`metrics.py`) — `bundle_snr`,
   `bundle_capacity`, `effective_dimensions`, `retrieval_confidence`,
   `cosine_matrix`, `saturation` — first-class analysis primitives not
   present in prior HDC libraries.
8. **Auto primal/dual ridge** (`RegularizedLSClassifier`) — closed-form
   ridge in hypervector space that auto-switches between primal
   (`d × d`) and dual (`n × n`) forms for numerical conditioning.
9. **3-seed ensemble + cal-set classifier selection** (benchmark
   harness) — automated model selection over a pool of classifiers
   (RegularizedLS, LogisticRegression, centroid-LVQ, HGB) with
   calibration-set accuracy as the selection criterion.

## Primary research attribution

The **classical VSA layer** — the 8 models, encoders, classifiers,
memory modules, and symbolic structures — is implemented directly from
primary research papers, not from other libraries' source code. Each
module cites the paper it implements:

| Component | Primary source |
|---|---|
| BSC — binary spatter codes, XOR bind, majority bundle | Kanerva (1997) "Fully Distributed Representation"; Kanerva (2009) |
| MAP — multiply-add-permute | Gayler (2003) "VSAs Answer Jackendoff's Challenges" |
| HRR — circular-convolution bind | Plate (1995) "Holographic Reduced Representations" |
| FHRR — complex-valued HRR | Plate (2003) |
| BSBC — block-sparse binary | Laiho & Poikonen (2015); Kleyko et al. survey (2022) |
| CGR — cyclic group representation | Kleyko et al. survey (2022) |
| MCR — modular composite representation, phasor bundle | Kleyko et al. survey (2022) |
| VTB — vector-derived transformation binding | Gosmann & Eliasmith (2019) |
| Random Fourier Features (`KernelEncoder`) | Rahimi & Recht (2007) |
| Sparse Distributed Memory | Kanerva (1988) |
| Modern Hopfield | Ramsauer et al. (2020) "Hopfield Networks is All You Need" |
| LVQ update rule | Kohonen (1990) "Learning Vector Quantization" |
| Iterative centroid refinement for HDC | Imani et al. (2017) "VoiceHD"; Imani et al. (2021) "OnlineHD" |
| Temperature scaling | Guo et al. (2017) "On Calibration of Modern Neural Networks" |
| Adaptive Prediction Sets (APS) | Romano, Sesia, Candès (2020) "Classification with Valid and Adaptive Coverage" |
| Expected Calibration Error | Naeini et al. (2015); popularized by Guo et al. (2017) |
| Resonator Networks (skeleton) | Frady, Kent, Olshausen, Sommer (2020) |
| Random Fourier Features | Rahimi & Recht (2007) |
| Deep-ensemble averaging | Lakshminarayanan et al. (2017) |

## How this project uses prior libraries

- **Not as source.** No code is copied. Every module is implemented from
  the cited paper above.
- **As benchmarks.** `benchmarks/benchmark_calibration.py` and
  `benchmarks/benchmark_compare.py` import TorchHD to run head-to-head
  comparisons on identical workloads. This is standard MLOSS-style
  practice and is how `bayes-hdc` demonstrates its empirical advantage.
- **As motivation.** The roadmap is informed by gaps in the existing
  HDC-library landscape (TorchHD, hdlib, PyBHV) — but filling those
  gaps with original implementations, not ports.

## Why this matters

For academic submission (JMLR MLOSS and equivalent venues), reviewers
ask: is this a novel software contribution, or a port? `bayes-hdc` is
the former. The PVSA framework is genuinely new; the deterministic VSA
layer is implemented from primary research, not from other libraries'
code; the empirical comparison against TorchHD is head-to-head, not a
derivative.
