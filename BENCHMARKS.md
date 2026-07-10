# Benchmarks

Reference numbers for `bayes-hdc`. The calibration / selective /
OOD numbers come from `make bench`; the canonical-dataset and paper
tables come from `make bench-canonical` and `make bench-paper`
(separate targets because they need the optional torch / torch-hd /
gdown extras and a network connection on first run). Figures are
checked into `benchmarks/figures/`.

**Reproduce:**

```bash
make install-all
make bench                # calibration/selective/OOD (~2 minutes, laptop CPU)
make bench-canonical      # ISOLET/UCI-HAR/EMG vs TorchHD (downloads data on first run)
make bench-paper          # classification calibration + multi-seed anomaly
make figures              # regenerate the figures under benchmarks/figures/
# or, containerised:
make docker-bench         # writes to benchmarks/results/
```

Each results JSON under `benchmarks/` embeds its own provenance
(commit, Python / JAX / TorchHD versions, platform). Canonical and
paper tables last refreshed June 10, 2026 on Python 3.13.5, JAX
0.10.0, torchhd 5.8.4, macOS arm64; the committed
`benchmarks/canonical_results.json` and `benchmarks/paper_results.json`
are the exact artifacts behind the tables below and in the paper.

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

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/anomaly_demo_dark.svg">
    <img src="assets/anomaly_demo.svg" alt="Conformal anomaly detection: empirical false-positive rate tracks the target alpha while a naive threshold drifts" width="640">
  </picture>
</p>

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

## Canonical HDC benchmark datasets

The accuracy table above uses sklearn datasets (iris / wine /
breast-cancer / digits / MNIST) — useful as smoke checks, not as
HDC-canonical anchors. The table below anchors on the datasets the HDC
literature actually benchmarks on, head-to-head against a TorchHD
centroid over its `Sinusoid` embedding (the matching random-Fourier
encoder family) on identical splits.

Protocol: the training pool is split 70/15/15 into fit /
model-selection / conformal-calibration slices. **Both** encoders get
the same RBF-bandwidth grid search on the model-selection slice (the
grid includes TorchHD's untuned default), and the conformal +
temperature calibration uses only the disjoint calibration slice, so
the finite-sample coverage guarantee is intact. Test data is never
touched before final evaluation. Mean ± std over 5 seeds, d = 10 000,
α = 0.1.

| Dataset | bayes-hdc | TorchHD (tuned) | ECE raw → calibrated | Coverage @ α=0.1 | Mean set size |
|---|---|---|---|---|---|
| ISOLET (canonical 6238/1559 split) | **0.895 ± 0.004** | 0.882 ± 0.006 | 0.845 → **0.022** | 0.901 | 1.08 |
| UCI-HAR (official subject-disjoint 7352/2947 split) | 0.849 ± 0.006 | **0.871 ± 0.005** | 0.633 → **0.031** | 0.904 | 1.52 |
| EMG hand gestures (Rahimi et al. 2016; stratified 70/30 window split) | **0.944 ± 0.014** | 0.892 ± 0.005 | 0.618 → **0.045** | 0.947 | 1.01 |

Under a matched tuning budget bayes-hdc is ahead on ISOLET and EMG and
behind on UCI-HAR — competitive with the deterministic reference, not
dominant. The value-add is the probabilistic layer: calibrated
probabilities (order-of-magnitude ECE reduction) and conformal coverage
at the target, guarantees the deterministic baseline does not provide.
Note the EMG split is window-level (subjects shared across train/test,
no canonical split ships with the dataset); UCI-HAR uses the official
subject-disjoint partition, so its accuracies are comparable to the
literature.

Reproduce with:

```bash
uv run --with scikit-learn --with torch --with torch-hd --with gdown \
    python benchmarks/benchmark_canonical.py
```

ISOLET is fetched via TorchHD's dataset loader (canonical
isolet1-4/isolet5 split); UCI-HAR via `bayes_hdc.datasets.load_ucihar`,
which downloads the official UCI archive and keeps the subject-disjoint
7352/2947 partition; EMG via `bayes_hdc.datasets.load_emg`, which
downloads the original authors' `dataset.mat` from
`abbas-rahimi/HDC-EMG` and cuts label-pure 256-sample windows.
European Languages (Joshi, Halseth, Kanerva 2016) remains future work:
the corpus has no stable OpenML/UCI mirror and needs the Wortschatz /
Europarl preprocessing pipeline.

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
| Unit tests passing | 666 (+ 2 skipped network-gated, 1 xfailed for GraphEncoder.encode_edges jit limitation) |
| Line coverage | 93 % on 23 modules |
| Lint (`ruff check`) | clean |
| Format (`ruff format --check`) | clean |
| Type check (`mypy bayes_hdc/`) | clean |
| CI matrix | Ubuntu + macOS × Python 3.9–3.13 |
| Core-library `torchhd` imports | 0 (independent implementation) |

## Figures

The commit includes paper-ready figures under
[`benchmarks/figures/`](benchmarks/figures/) — 10 PNGs at 150 DPI (PDF versions regenerable via `make figures`).

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
