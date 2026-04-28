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

`bayes-hdc` is the first open-source HDC library to implement
**Probabilistic Vector Symbolic Architectures (PVSA)** — a new research
framework in which every hypervector is a *posterior distribution* over
hypervectors rather than a single point, and every VSA primitive
propagates that posterior's moments in closed form. Existing HDC
software represents symbols as single points in $\mathbb{R}^d$;
`bayes-hdc` represents them as posteriors and derives closed-form
algebraic operations for binding, bundling, similarity, retrieval, and
divergence.

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

## Maturity and community adoption

The library is at version `0.4.0a0` and was first published on April 15,
2026; community adoption is therefore limited at submission time. The
following are the concrete maturity signals available today:

- **Continuous integration**: 475 unit tests and 97% line coverage on
  every push to `main`, across Ubuntu and macOS for Python 3.9–3.13;
  weekly CodeQL security scan; weekly Dependabot dependency bumps.
- **Release infrastructure**: tag-driven TestPyPI / PyPI publishing
  workflow with OIDC trust; release notes via `CHANGELOG.md`.
- **Issue and discussion templates** for bug reports, feature requests,
  and Q&A.
- **Repository activity at submission**: see the live shields at
  the top of `README.md` for the current snapshot.

We acknowledge that "evidence of an existing user community" is the
weakest dimension of this submission against MLOSS criteria. The
deliberate strategy is to ship a small, correct, well-tested foundation
first and grow the community after publication; the contributing
infrastructure (`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SUPPORT.md`)
is already in place to accept external contributions.

## Authorship and prior publication

The method(s) underlying each VSA model have been previously published
(references in the paper). **The software itself has not been published
elsewhere** and is submitted here for the first time.

## Suggested reviewers

The author defers to the action editor for reviewer assignment; the
following names are offered as illustrative of the relevant subfield
expertise rather than a hard request:

- HDC/VSA — researchers active in the Kleyko et al. (2022) survey
  community.
- JAX scientific software — authors of recent JMLR MLOSS submissions
  on JAX-based libraries.
- Calibration and conformal prediction — researchers familiar with
  the post-hoc calibration and split-conformal literatures.

The author has no current or recent collaboration with any researcher
identifiable by these descriptors.

## Non-reviewer requests

None.

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
_Date to be inserted at submission._
