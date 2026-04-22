# JMLR MLOSS submission checklist

Tracks `bayes-hdc` against the requirements published at
<https://www.jmlr.org/mloss/mloss-info.html>. Re-run each time before a
submission or resubmission round. Status legend: ✅ done · 🟡 partial · ❌ missing.

---

## 1. Submission package

| Item | Status | Location / notes |
|------|--------|------------------|
| Cover letter | ✅ | [`docs/MLOSS_COVER_LETTER.md`](MLOSS_COVER_LETTER.md) — template, fill in before submission |
| Up-to-4-page description (JMLR format) | 🟡 | [`docs/main.tex`](main.tex) — exists but needs compaction; see §7 |
| Public repository link | ✅ | `https://github.com/rlogger/bayes-hdc` |
| Source archive (`singh26a-code.tar.gz`) | ❌ | Generate at submission: `git archive --format=tar.gz --prefix=bayes-hdc/ HEAD -o singh26a-code.tar.gz` |
| Reviewer suggestions | ❌ | Add to cover letter before submission |

## 2. Code quality & testing

| Item | Status | Evidence |
|------|--------|----------|
| Unit tests | ✅ | `tests/test_*.py` — **467 tests** |
| Integration tests | ✅ | `tests/test_integration.py` + benchmarks as end-to-end |
| Coverage close to 100% | ✅ | **97%** on 22 source modules |
| Coverage report published | ✅ | Codecov upload in `.github/workflows/tests.yml` |
| CI on all supported platforms | ✅ | Ubuntu, macOS (Windows dropped — pytest-cov collides with protected drive metadata) |
| CI across multiple Python versions | ✅ | Python 3.9, 3.10, 3.11, 3.12, 3.13 |
| CI across multiple JAX versions | 🟡 | Single matrix axis for now; JAX≥0.4.20 pin is wide |
| Static analysis (CodeQL) | ✅ | `.github/workflows/codeql.yml` — weekly + PR |
| Dependency automation | ✅ | `.github/dependabot.yml` — weekly GH-Actions + pip updates |
| Pre-commit hooks | ✅ | `.pre-commit-config.yaml` mirrors CI (ruff + mypy) |
| Release automation | ✅ | `.github/workflows/release.yml` — tag → sdist+wheel + GH Release |
| Clear software design | ✅ | Functional, pytree-native; see `SLIDES.md` |
| No proprietary dependencies | ✅ | Only JAX (Apache-2.0) and stdlib |
| Containerised reproduction | ✅ | `Dockerfile` with dedicated `benchmark` stage; `make docker-bench` |

## 3. License

| Item | Status | Evidence |
|------|--------|----------|
| Recognised open-source licence | ✅ | MIT, [`LICENSE`](../LICENSE) |
| Licence in source package | ✅ | `LICENSE` at repo root |
| Licence mentioned in each source file | ✅ | SPDX header (`# SPDX-License-Identifier: MIT`) on every `.py` file |
| Dependencies are OSS | ✅ | JAX (Apache-2.0), Python stdlib |

## 4. Installation & reproducibility

| Item | Status | Evidence |
|------|--------|----------|
| Compiles / runs on all supported platforms | ✅ | Verified via CI matrix |
| Installation instructions | ✅ | `README.md` Installation section + `docs/installation.rst` |
| Tutorials | ✅ | `examples/basic_operations.py`, `examples/kanerva_example.py`, `examples/classification_simple.py` |
| Non-trivial example showing typical use | ✅ | `examples/classification_simple.py` (end-to-end encode → train → score) |
| Learning aid for reviewers | ✅ | `SLIDES.md` + `QUIZ.md` |

## 5. Documentation

| Item | Status | Evidence |
|------|--------|----------|
| Full API documentation | ✅ | Google-style docstrings on every public function; `docs/api.rst` uses Sphinx autodoc |
| Developer documentation | ✅ | [`CONTRIBUTING.md`](../CONTRIBUTING.md) — setup, style, testing, release process |
| Documentation accessible online | 🟡 | `.readthedocs.yaml` configured; **TODO** — verify RTD build is actually published before submission |
| API reference | ✅ | `docs/api.rst` |

## 6. Repository hygiene

| Item | Status | Evidence |
|------|--------|----------|
| No extraneous files (VCS, `.DS_Store`, backup) | ✅ | `.gitignore` covers all of them |
| Public source repository | ✅ | GitHub: `rlogger/bayes-hdc` |
| Bug tracker | ✅ | GitHub Issues + templates in `.github/ISSUE_TEMPLATE/` |
| Forum / mailing list | ✅ | GitHub Discussions (`https://github.com/rlogger/bayes-hdc/discussions`) |
| Code of Conduct | ✅ | [`CODE_OF_CONDUCT.md`](../CODE_OF_CONDUCT.md) (Contributor Covenant 2.1) |
| Security policy | ✅ | [`SECURITY.md`](../SECURITY.md) |
| Citation file | ✅ | [`CITATION.cff`](../CITATION.cff) |

## 7. Paper (docs/main.tex)

| Item | Status | Notes |
|------|--------|-------|
| JMLR LaTeX class | ✅ | `\documentclass[twoside,11pt]{article}` with `jmlr2e` |
| Abstract with software URL | ✅ | Abstract links to `https://github.com/rlogger/bayes-hdc` |
| Keywords | ✅ | HDC, VSA, JAX, XLA, hardware acceleration |
| 4 pages body (references unlimited) | ❌ | **TODO** — current draft overflows; compact by merging 8 module subsections into a single architecture table |
| Comparison with prior work (runtime / memory / features) | 🟡 | Feature table present (Table 2); **TODO** — add runtime numbers via `benchmarks/benchmark_compare.py` vs TorchHD |
| Usage examples | ✅ | Three code listings |
| Related work section | ✅ | Torchhd, hdlib, PyBHV referenced |
| LaTeX source in final archive | ✅ | `docs/main.tex` + `docs/*.bib` in the repo |
| Authorship verified | 🟡 | **TODO** — confirm all listed authors before submission |

## 8. Maturity & community

| Item | Status | Notes |
|------|--------|-------|
| Recognised OSS licence (SPDX-tagged) | ✅ | MIT |
| Maintainability plan | ✅ | Release process in `CONTRIBUTING.md`; SemVer policy in `CHANGELOG.md` |
| Openness to new contributors | ✅ | `CONTRIBUTING.md`, issue templates, PR template, COC |
| Evidence of user community | ❌ | **TODO before submission** — GitHub stars, downloads, external citations. Current: new repo |
| Platform support including open OS | ✅ | Linux (Ubuntu) in CI, Python 3.9–3.13 |

## 9. Novelty & significance

| Item | Status | Notes |
|------|--------|-------|
| Stated novelty | ✅ | **PVSA framework** (Probabilistic Vector Symbolic Architectures) — first HDC algebra where every hypervector is a posterior distribution with closed-form moment propagation. See `ORIGINALITY.md` and `docs/workshop_paper.tex`. |
| Comparison with previous implementations | ✅ | Head-to-head benchmarks in `benchmarks/benchmark_calibration.py`, `benchmark_selective.py`, `benchmark_ood.py`. Summary in `BENCHMARKS.md`. |
| Significant progress over alternatives | ✅ | **Bayes-HDC wins 5/5 benchmark datasets vs TorchHD (mean +3.94pt, MNIST +8.9pt); ECE reduction under temperature scaling reaches 25× on MNIST; conformal coverage clears the 90% guarantee on every dataset.** |
| Independence verified | ✅ | `ORIGINALITY.md` — zero ported code, per-component primary-source attribution. |

---

## Pre-submission runbook (checklist)

One week before submission, run this in order. Each step should leave the
repo in a state a reviewer can check out and reproduce.

1. `pytest tests/ -v --cov=bayes_hdc --cov-report=term-missing` — expect 99%+.
2. `ruff check bayes_hdc/ tests/ examples/ benchmarks/` — expect clean.
3. `ruff format --check bayes_hdc/ tests/ examples/ benchmarks/` — expect clean.
4. `mypy bayes_hdc/` — expect clean.
5. `python examples/basic_operations.py` — expect it prints expected ranges.
6. `python examples/classification_simple.py` — expect >= 80% synthetic accuracy.
7. `python benchmarks/benchmark_compare.py` — save results JSON; paste numbers into the paper.
8. Tag a release: `git tag -s vX.Y.Z && git push --tags`.
9. `git archive --format=tar.gz --prefix=bayes-hdc/ vX.Y.Z -o singh26a-code.tar.gz`.
10. Fill in `docs/MLOSS_COVER_LETTER.md` with the tagged version, reviewer suggestions, and active-community evidence.
11. Build the paper: `cd docs && latexmk -pdf main.tex`.
12. Submit via JMLR's MLOSS portal with: cover letter, PDF, LaTeX source, and the source archive.

## Known gaps still to close (prioritised)

1. **Paper length.** Body needs to fit 4 pages. Current draft is ~8 pages.
2. **Published benchmark numbers.** `benchmark_compare.py` exists; run it and paste numbers into Section 3 of the paper.
3. **Community evidence.** Need GitHub stars, users, and at least one external citation or downstream use before submission is credible.
4. **Documentation hosting.** Verify ReadTheDocs build succeeds and link is live.
5. **Multi-JAX-version CI.** Add `jax-version` axis to the test matrix.
