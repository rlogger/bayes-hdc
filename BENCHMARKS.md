# Benchmarks

Reference numbers for `bayes-hdc`. All numbers are produced by
`make bench`; the included figures are checked into
`benchmarks/figures/`.

**Reproduce:**

```bash
make install-all
make bench                # local run (takes ~2 minutes on a laptop CPU)
make figures              # regenerate the paper PDFs/PNGs under benchmarks/figures/
# or, containerised:
make docker-bench         # writes to benchmarks/results/
```

Last refreshed from commit `0f761f0` (April 22, 2026), Python 3.14, JAX 0.4.28, macOS arm64.

## Accuracy (Bayes-HDC vs TorchHD, identical preprocessing)

The numbers below are not a like-for-like single-classifier comparison: the
Bayes-HDC column reports the *best of an ensemble* of four classifiers
(ridge regression on hypervectors, logistic regression on hypervectors,
centroid-LVQ matching TorchHD's default, and an off-the-shelf gradient-
boosted baseline on raw features), with the per-task winner chosen by
held-out calibration-set accuracy. The TorchHD column reports its default
centroid classifier on the same encoder. The setup is closer to "how well
does the ensemble + cal-set selection do" than to "is bind/bundle in JAX
faster than in PyTorch"; read the deltas accordingly.

| Dataset | n | classes | Bayes-HDC ensemble | TorchHD centroid | Δ |
|---|---:|---:|---:|---:|---:|
| iris          |    150 |  3 | **0.933** | 0.911 | +2.2 |
| wine          |    178 |  3 | **0.852** | 0.815 | +3.7 |
| breast-cancer |    569 |  2 | **0.959** | 0.953 | +0.6 |
| digits        |  1 797 | 10 | **0.943** | 0.900 | +4.3 |
| MNIST         | 10 000 | 10 | **0.946** | 0.857 | +8.9 |
| **mean Δ** | | | | | **+3.9** |

Numbers are single-seed; the JSON dump in
[`benchmarks/benchmark_calibration_results.json`](benchmarks/benchmark_calibration_results.json)
records exact configurations. A multi-seed sweep is a planned addition.

## Calibration (ECE reduction under temperature scaling, Bayes-HDC)

Both libraries use the *same* `TemperatureCalibrator` (L-BFGS in log-space),
isolating the effect of the underlying classifier's logit distribution.

| Dataset | ECE raw | ECE + T | reduction |
|---|---:|---:|---:|
| iris          | 0.523 | **0.081** | 6.5× |
| wine          | 0.498 | **0.111** | 4.5× |
| digits        | 0.792 | **0.039** | **20×** |
| MNIST         | 0.683 | **0.027** | **25×** |

## Conformal coverage (Bayes-HDC only — no equivalent in TorchHD)

Split-conformal APS (Romano et al. 2020) at α = 0.1 target (≥ 0.90 coverage):

| Dataset | target | empirical | mean set size |
|---|---:|---:|---:|
| iris          | 0.90 | **1.000** | 2.44 |
| wine          | 0.90 | **0.944** | 1.50 |
| breast-cancer | 0.90 | **1.000** | 1.29 |
| digits        | 0.90 | **0.969** | 2.81 |
| MNIST         | 0.90 | **0.956** | 2.92 |

All datasets clear the finite-sample coverage guarantee. Set size scales with
task difficulty — binary classification collapses sets to near-1, 10-class
problems admit 2–3 classes.

## Wall-clock micro-benchmark vs TorchHD (CPU, eager mode)

Reproduced from `benchmarks/benchmark_compare.py` on the same machine,
identical dimensions (`d = 10 000`), 20 warmup iterations, 200 timed
trials. Both libraries run on CPU. JAX side compiles via `jit` once
during warmup; PyTorch side uses TorchHD's default eager kernels under
`torch.no_grad`. Both timers wait for the result tensor on-device — no
asymmetric host sync via `.item()` — so the numbers measure
"compute-and-block" on each backend.

| Operation | bayes-hdc (ms) | TorchHD (ms) | Speedup |
|---|---:|---:|---:|
| MAP `bind` (2 HVs) | **0.009 ± 0.006** | 0.012 ± 0.014 | 1.41× |
| MAP `bundle` (10 HVs) | **0.025 ± 0.008** | 0.053 ± 0.021 | 2.11× |
| Cosine similarity | **0.021 ± 0.006** | 0.075 ± 0.025 | 3.48× |
| `RandomEncoder` (100×20) | 1.069 ± 0.075 | **0.911 ± 0.128** | 0.85× |

Pointwise operations (`bind`, `bundle`, cosine) are 1.4×–3.5× faster
under JAX-`jit` than TorchHD's eager kernels. The encoder benchmark
includes 200 random-sample lookups in addition to the bind+bundle
work, and TorchHD's `embeddings.Random` indexing path stays slightly
faster on this single-CPU configuration; the gap typically closes on
GPU and reverses with batching.

These numbers compare **eager-mode TorchHD against `jit`-compiled
bayes-hdc**. A `torch.compile` baseline would partially close the
pointwise gap and is deferred to a future suite. Reproduce locally
with `python benchmarks/benchmark_compare.py`; the script writes
`benchmarks/benchmark_results.json` (gitignored — local hardware
varies).

## Sequence-encoding capacity: flat vs. hierarchical

Per-position retrieval accuracy at fixed `d = 4 096`, codebook size
256, chunk size 16, averaged over 3 seeds. Each item is drawn
i.i.d. from the codebook; retrieval applies `get(i)` then cleanup
against the codebook (argmax cosine similarity).

| T   | flat `Sequence` | `HierarchicalSequence` | gain |
|---:|---:|---:|---:|
| 16  | 1.000 | 1.000 |  +0.000 |
| 32  | 1.000 | 1.000 |  +0.000 |
| 64  | 1.000 | 1.000 |  +0.000 |
| 128 | 0.992 | 1.000 |  +0.008 |
| 200 | 0.958 | 1.000 |  +0.042 |
| 300 | 0.809 | 1.000 |  +0.191 |
| 400 | 0.631 | 1.000 |  +0.369 |
| 600 | 0.423 | 1.000 |  +0.577 |
| 800 | 0.309 | 1.000 |  +0.691 |

Flat permute-bundle saturates around `T ≈ 200` and degrades
monotonically as `T` grows: by `T = 800` the flat representation
retrieves only 31 % of positions correctly. The hierarchical variant
stays at perfect retrieval throughout the swept range, because the
chunk-level cleanup (an `argmax` against the cached chunk
codebook) prunes the cross-chunk noise *before* the inner
un-permute. At `chunk_size = 16`, both layers carry only `O(√T)`
items, and the per-layer SNR is dominated by `1/√(chunk_size)` and
`1/√(n_chunks)` rather than `1/√T`. Reproduce with
`python benchmarks/benchmark_sequence_capacity.py`. References:
Plate (2003) §6.2 on flat-bundle capacity; Frady, Kleyko & Sommer
(2018) on hierarchical recurrent-network indexing.

## Canonical HDC benchmark datasets (script shipped; numbers pending)

The accuracy table above uses sklearn datasets (iris / wine /
breast-cancer / digits / MNIST) — useful as smoke checks, not as
HDC-canonical anchors. The datasets the HDC literature actually
benchmarks on are:

| Dataset | Task | Reference |
|---|---|---|
| ISOLET | 26-class spoken-letter recognition (617 features) | Fanty & Cole 1990; Rahimi et al. 2016 |
| UCI-HAR | 6-class daily-living activity recognition (561 features) | Anguita et al. 2013 |
| EMG | multi-class hand-gesture EMG | Rahimi et al. 2016 |
| European Languages | 21-class character-trigram language ID | Joshi, Halseth, Kanerva 2016 |

`benchmarks/benchmark_canonical_hdc_tasks.py` ships the full
calibration + conformal pipeline against these four datasets via the
`bayes_hdc.datasets.load_*` loaders. The reported columns are
**accuracy**, **ECE (raw)**, **ECE (post-temperature)**, **Brier**,
**NLL**, **conformal coverage at α = 0.1**, **mean conformal set
size**. At the time of writing, all four loaders fetch from OpenML and
on the development machine these fetches either error on the dataset
ID or require the `pyarrow` optional parser. Reproduce on a host with
OpenML-reachable networking + `pyarrow` installed; numbers go in this
section once produced. (This is the single most load-bearing
empirical gap the 2026-05-06 audit identified — anchoring on the
field's standard datasets is what turns "library-first" into
"library-first AND empirically validated.")

## Deferred comparisons

These are head-to-head comparisons the library will gain once the
matching datasets / configurations are available on the dev machine.
Documented here so they are not silently absent from the benchmarks
story.

- **ConformalHDC** (Liang, Poursiami, Yang, Cooper, Jaiswal, Parsa,
  Fortin & Shahbaba 2026, *arXiv:2602.21446*). Concurrent algorithmic
  work on conformal prediction for HDC; cited in the README, paper,
  and bibliography. A like-for-like comparison on their reported
  hippocampal-neural-decoding dataset would cash out the
  "concurrent algorithmic priority; library-first implementation"
  framing into hard numbers (e.g. matching their adaptive-score
  coverage within 1-2 pp at comparable set sizes). Deferred to the
  next benchmark cycle.
- **`torch.compile` head-to-head with TorchHD.** The Wall-clock
  table above compares eager-mode TorchHD against `jit`-compiled
  bayes-hdc; a `torch.compile` baseline would partially close the
  pointwise gap and isolate the JAX-XLA-vs-PyTorch-compile
  contribution from the JIT-vs-eager contribution.
- **GPU + TPU numbers.** All wall-clock measurements above are
  single-CPU. Multi-device numbers via `pmap_*` and `shard_map_*`
  wrappers are runnable but not yet reported on a TPU pod with
  multiple chips.

## Test / coverage / lint status

| Check | Value |
|---|---|
| Unit tests passing | 625 (+ 1 xfailed for GraphEncoder.encode_edges jit limitation) |
| Line coverage | 93 % on 23 modules |
| Lint (`ruff check`) | clean |
| Format (`ruff format --check`) | clean |
| Type check (`mypy bayes_hdc/`) | clean |
| CI matrix | Ubuntu + macOS × Python 3.9–3.13 |
| Core-library `torchhd` imports | 0 (independent implementation) |

## Figures

The commit includes paper-ready figures under
[`benchmarks/figures/`](benchmarks/figures/) — 10 PDFs + 10 PNGs at 150 DPI.

### Accuracy bar chart

![accuracy](benchmarks/figures/accuracy_comparison.png)

### ECE reduction

![ece](benchmarks/figures/ece_reduction.png)

### Reliability diagrams

| Dataset | Reliability |
|---|---|
| iris          | ![iris](benchmarks/figures/reliability_iris.png) |
| wine          | ![wine](benchmarks/figures/reliability_wine.png) |
| breast-cancer | ![bc](benchmarks/figures/reliability_breast_cancer.png) |
| digits        | ![digits](benchmarks/figures/reliability_digits.png) |

### Conformal coverage curves

| Dataset | Coverage |
|---|---|
| iris          | ![cov_iris](benchmarks/figures/coverage_iris.png) |
| wine          | ![cov_wine](benchmarks/figures/coverage_wine.png) |
| breast-cancer | ![cov_bc](benchmarks/figures/coverage_breast_cancer.png) |
| digits        | ![cov_digits](benchmarks/figures/coverage_digits.png) |
