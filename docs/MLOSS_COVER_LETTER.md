# Cover letter — JMLR MLOSS submission

> Template. Fill in the bracketed fields at submission time. Keep under one page.

---

**Subject:** MLOSS submission — jax-hdc

**Track:** Machine Learning Open Source Software (MLOSS)

**Software version under review:** `vX.Y.Z` — commit `[SHA]`
**Project website:** <https://github.com/rlogger/jax-hdc>
**Documentation:** <https://jax-hdc.readthedocs.io>
**License:** MIT (OSI-approved)
**Submission archive:** `singh26a-code.tar.gz`

---

## Summary

`jax-hdc` is an open-source Python library for hyperdimensional computing
(HDC) and vector symbolic architectures (VSA), built on JAX. It is, to the
best of our knowledge, the first HDC library that

1. implements eight VSA models (BSC, MAP, HRR, FHRR, BSBC, CGR, MCR, VTB)
   behind a single pytree-native API compatible with `jit`, `vmap`, and
   `grad`;
2. runs unchanged on CPUs, GPUs, and TPUs via XLA compilation;
3. ships a first-class capacity / noise / retrieval-confidence analysis
   module (`jax_hdc.metrics`);
4. includes a resonator-network primitive for factorising composite
   hypervectors.

Beyond the eight VSA models, the library provides five data encoders, five
learning models (including clustering), three associative-memory modules,
four symbolic data structures, and an extensive functional layer. All
code is unit-tested to 99% line coverage and exercised by continuous
integration across Ubuntu, macOS, and Windows on Python 3.9 through 3.13.

## How this contribution differs from prior software

- **Torchhd** (Heddes et al., JMLR MLOSS 2023) provides the most comparable
  functionality on PyTorch. `jax-hdc` differs by targeting JAX/XLA (adding
  native TPU support and transparent `jit`/`vmap`/`grad`) and by shipping
  capacity analysis, resonator factorisation, fractional-power binding,
  Tversky/Jaccard similarity, and a pytree-native functional design that
  Torchhd does not provide out of the box.
- **hdlib** (Cumbo et al., JOSS 2023) focuses on bioinformatics and
  clustering workflows on CPU. `jax-hdc` covers a superset of its core API
  and adds hardware acceleration.
- **PyBHV** (Vandervorst, 2023) is a Boolean-hypervector research
  framework with a rich threshold-logic layer. `jax-hdc` incorporates
  several of its ideas (Tversky/Jaccard, threshold/window ops) as
  first-class primitives.

A feature-level comparison table is in Section 3 of the paper.

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
