# Community

bayes-hdc is open-source under MIT, and the design choice from day one is
to build a contributor community larger than any single research lab.
This page is the map.

## Where to talk

| Channel | Best for |
| --- | --- |
| [GitHub Discussions](https://github.com/rlogger/bayes-hdc/discussions) | Q&A, "is this a bug?", feature ideas, show-and-tell, RFCs |
| [GitHub Issues](https://github.com/rlogger/bayes-hdc/issues) | Confirmed bugs, concrete tasks |
| Pull requests | Code, docs, examples, benchmarks |
| Email (`rajdeeps@usc.edu`) | Security disclosures, private collaboration enquiries |

## How to get involved

There are five distinct ways to contribute, sorted from "ten minutes" to
"co-author":

### Ten-minute contributions

* **Star the repo.** Real signal — projects rank by stars on PyPI, OpenSSF, and
  most awesome-lists.
* **Open a [Discussion](https://github.com/rlogger/bayes-hdc/discussions/categories/show-and-tell)** describing what you used the library for. We read every one.
* **Report a bug** with a minimal reproducer. The issue template walks
  you through it.
* **Fix a typo.** The README, docstrings, and `docs/` are all fair game.
  Any PR with a plausible typo fix gets merged the same day.

### One-hour contributions

* **Pick a [`good first issue`](https://github.com/rlogger/bayes-hdc/labels/good%20first%20issue).** These are scoped, well-defined, and have a maintainer assigned for mentorship. If you can't find one that fits, ask on Discussions.
* **Add a docstring example** to any public API that lacks one.
* **Write a benchmark** for a function in `benchmarks/`.

### Half-day contributions

* **Build a new application example** for `examples/`. Pattern-match the
  existing ones (`emg_gesture_recognition.py`, `language_identification.py`).
  We are particularly interested in: ECG/EEG biosignal classification,
  sensor-fusion robotics, neural-symbolic reasoning, time-series anomaly
  detection.
* **Port a new dataset loader** to `bayes_hdc/datasets/loaders.py`. Open
  HDC datasets we'd love coverage for: ISOLET (added), TIMIT, CHiME,
  PhysioNet ECG, MNIST-1D, CIFAR-10 with random projection.
* **Add a new VSA model** to `bayes_hdc/vsa.py`. Candidates: Sparse
  Block Codes, Geometric Algebra HDC, FHRR variants.

### Multi-day contributions

* **Add a new probabilistic primitive** under `bayes_hdc/distributions.py` —
  e.g. a Wishart-prior covariance type, a structured-mixture posterior, or
  a Gaussian-process hypervector.
* **Wire bayes-hdc into a downstream library** (flax / equinox / brax /
  rlax) and PR a one-line integration test plus a short demo notebook.
  This is the highest-leverage contribution available.
* **Run a benchmark study** comparing a PVSA pipeline against a deep-net
  baseline on a non-standard task and write up the result in
  `benchmarks/`.

### Co-author contributions

* **Submit a PR with new theoretical work** — a closed-form result for a
  PVSA op we don't yet support, a tighter bound on the bundle capacity, a
  group-theoretic result on a new VSA family. We will work with you to
  publish it as a follow-up paper with shared authorship.

## Path to maintainership

Three commits with at least one substantive change — code, docs, or
examples — and a positive interaction history makes you eligible for
**triage permissions**: you can label issues, mark duplicates, and close
PRs that are clearly out of scope. Ask for it on Discussions.

Six commits with at least one PR that touched the public API, plus
demonstrated review-comment quality, makes you eligible for **commit
access**: you can merge approved PRs from others. Ask the lead maintainer
on Discussions.

The lead maintainer at present is the original author. Maintainership
will become a roster as the contributor pool grows.

## Recognition

Every PR-author appears in `CONTRIBUTORS.md` (sorted alphabetically; the
file is auto-updated by the all-contributors workflow when configured).
Multi-PR contributors get listed by category — code, docs, examples,
benchmarks, ideas, support — in the README's `Contributors` block. We
will keep this list pruned and accurate; we will not ghost anyone.

If your contribution leads to a citable result, you become a co-author on
that publication. There is no maintainer-only "core team" gatekeeping
authorship; what makes the cut is technical contribution.

## Code of Conduct

We follow the [Contributor Covenant](CODE_OF_CONDUCT.md). Standard
expectations: be civil, separate ideas from people, and prefer steel-man
readings. Harassment, doxxing, or sustained bad-faith behaviour will be
escalated and acted on.

## Security

See [`SECURITY.md`](SECURITY.md). Briefly: open a private security
advisory on GitHub or email `rajdeeps@usc.edu` for embargo handling.

## Funding and sponsorship

bayes-hdc has no commercial backing. If your lab or organisation
benefits from it, the most useful things you can do are:

1. Cite the paper (see `CITATION.cff`) once it is published.
2. Sponsor a graduate student or postdoc to contribute upstream.
3. Open a Discussion thread describing your use case so we know what
   to prioritise.

GitHub Sponsors / OpenCollective links will be added when the project
crosses the threshold where individual donations are useful.

## Where the project is heading

The roadmap lives in [`README.md`](README.md#roadmap) (high-level) and
in the GitHub issue tracker (concrete tasks). The two sections that are
most under-served right now are:

1. **Application examples in active research domains** — biosignals,
   neuro-symbolic reasoning, robotics, neuromorphic hardware.
2. **Variational learning of codebooks** — the PVSA primitives expose
   reparameterisation gradients end-to-end, but no example yet
   demonstrates training a codebook variationally.

If either of those resonates, please start a Discussion. We will route
you straight to the right issue or propose one together.
