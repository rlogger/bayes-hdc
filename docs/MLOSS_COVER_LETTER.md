# Cover letter — JMLR MLOSS submission

> Template. Fill in the bracketed fields at submission time. Keep under one page.

---

**Subject:** MLOSS submission — bayes-hdc

**Track:** Machine Learning Open Source Software (MLOSS)

**Software version under review:** `vX.Y.Z` — commit `[SHA]`
**Project website:** <https://github.com/rlogger/bayes-hdc>
**Documentation:** <https://bayes-hdc.readthedocs.io>
**License:** MIT (OSI-approved)
**Submission archive:** `singh26a-code.tar.gz`

---

## Summary

`bayes-hdc` is the first open-source HDC library in which **every
hypervector can carry a distribution**. Existing HDC software represents
symbols as single points in $\mathbb{R}^d$; `bayes-hdc` represents them
as posteriors and propagates the posterior forward through the full
Vector Symbolic Architecture (VSA) algebra — binding, bundling,
similarity, retrieval — so that classification and retrieval inherit
calibrated uncertainty.

The release contains three layers:

1. **Bayesian core.** A Gaussian hypervector type (`GaussianHV`) with
   closed-form moment propagation under element-wise product (MAP-style
   binding) and normalised sum (bundling); an uncertainty-aware
   similarity API (expected cosine similarity and exact dot-product
   variance); and a closed-form KL divergence for variational
   objectives.
2. **Deterministic VSA foundation.** Eight classical models (BSC, MAP,
   HRR, FHRR, BSBC, CGR, MCR, VTB) behind a uniform pytree-native API,
   five data encoders, five learning models (including k-means-style
   clustering), three associative memory modules, four symbolic data
   structures, a capacity-and-noise analysis toolkit, and a
   resonator-network primitive for factorisation.
3. **Portable implementation.** JAX + XLA throughout; runs unchanged on
   CPU, GPU, and TPU; `jit` / `vmap` / `grad` / `pmap` compose with the
   whole library.

The codebase is unit-tested to 99% line coverage (321 tests) and
exercised by continuous integration across Ubuntu, macOS, and Windows on
Python 3.9 through 3.13.

## How this contribution differs from prior software

- **Torchhd** (Heddes et al., JMLR MLOSS 2023) — the most comparable
  library — provides the deterministic functionality on PyTorch. It does
  not expose hypervectors as distributions, has no notion of
  expected-value or variance-of-similarity APIs, and does not ship KL
  divergences or variational objectives for codebooks. `bayes-hdc`
  provides all of these as first-class primitives, and matches Torchhd
  on the deterministic baseline.
- **hdlib** (Cumbo et al., JOSS 2023) focuses on bioinformatics and
  clustering workflows on CPU. `bayes-hdc` covers a superset of its
  deterministic API and adds hardware acceleration and the Bayesian
  layer.
- **PyBHV** (Vandervorst, 2023) is a Boolean-hypervector research
  framework with a rich threshold-logic layer. `bayes-hdc` incorporates
  several of its ideas (Tversky/Jaccard, threshold/window ops) as
  deterministic primitives.

To the best of our knowledge, no public HDC library has previously
offered hypervectors as distributions with closed-form moment propagation
and calibrated retrieval. A feature-level comparison table is in
Section 3 of the paper.

## Evidence of community adoption

- **Repository activity:** `[ X stars, Y forks, Z watchers ]` at submission time.
- **External uses / citations:** `[ list of downstream projects or papers, if any ]`.
- **Recent release cadence:** `[ date of last 3 tagged releases ]`.
- **Open issues / closed PRs:** `[ numbers ]` — indicating responsive
  maintainership.

## Authorship and prior publication

The method(s) underlying each VSA model have been previously published
(references in the paper). **The software itself has not been published
elsewhere** and is submitted here for the first time.

## Suggested reviewers

1. `[ Name, affiliation ]` — expert on `[HDC / JAX / VSA models / …]`,
   relevant because `[ reason ]`.
2. `[ Name, affiliation ]` — `[ reason ]`.
3. `[ Name, affiliation ]` — `[ reason ]`.

(Reviewers with whom the author has a current or recent collaboration have
been excluded.)

## Non-reviewer requests

`[ If applicable, list people you would not like to review, with a neutral
one-line reason. Otherwise: "None." ]`

## Ethics and research integrity

- All contributions are by the author(s) listed on the paper.
- No third-party code is included without attribution in the source file,
  repository, and paper.
- No proprietary dependencies are required to build, run, or use the
  software.
- All included example data is synthetic or public-domain.

## Contact

- **Corresponding author:** Rajdeep Singh — `rajdeeps@usc.edu`
- **GitHub handle:** `@rlogger`

Thank you for considering this submission.

---

Rajdeep Singh
University of Southern California
`[ date ]`
