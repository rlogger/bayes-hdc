# Outreach playbook

Internal checklist for pushing bayes-hdc out into the world. The
deliverable target is **a community larger and more active than
TorchHD's** — the comparable milestone is ≥ 200 stars, ≥ 5 cited papers
using the library, ≥ 3 external contributors, ≥ 1 downstream
integration in a major JAX library by the end of year one.

This is a working document. Tick items off as they ship; add new ones
as they come up.

## Cohort 1 — Discoverability foundations

- [x] Public-facing repository with README, examples, docs site
- [x] GitHub Pages auto-deploy on push to main
- [x] Sitemap.xml + robots.txt
- [x] OpenGraph + Twitter card metadata (1200×630 social card)
- [x] Schema.org JSON-LD `SoftwareApplication` markup
- [x] Page-level `<meta name="keywords">` on every docs page
- [x] PyPI metadata: keywords, classifiers, project URLs
- [ ] Publish to PyPI (`bayes-hdc`) — *prerequisite: tag a `v0.4.0` release once API stabilises*
- [ ] Publish to conda-forge — submit recipe to `conda-forge/staged-recipes`
- [ ] Register the repo on [Read the Docs](https://readthedocs.org/) so
      `bayes-hdc.readthedocs.io` works as an alternate docs host. The
      `.readthedocs.yaml` is already in the repo; once you click "Import
      a Project" on RTD it will build automatically.
- [ ] Add a [Papers With Code](https://paperswithcode.com/) entry for the
      JMLR paper once accepted, linked to the repo
- [ ] Add a [Zenodo](https://zenodo.org/) DOI by enabling the GitHub-Zenodo
      integration on a tagged release; cite the DOI in `CITATION.cff`

## Cohort 2 — Awesome-list submissions

Submit a one-line entry pointing at the repo to each of these. The line
should be:

> **bayes-hdc** — Probabilistic hyperdimensional computing in JAX. Closed-form Gaussian/Dirichlet hypervectors, conformal prediction sets, group-theoretic equivariance verifiers. \[[GitHub](https://github.com/rlogger/bayes-hdc) · [Docs](https://rlogger.github.io/bayes-hdc/)\]

- [ ] [`n2cholas/awesome-jax`](https://github.com/n2cholas/awesome-jax)
- [ ] [`patrick-kidger/awesome-jax-libraries`](https://github.com/patrick-kidger) (if it exists; otherwise a related list)
- [ ] [`vinta/awesome-python`](https://github.com/vinta/awesome-python) — under "Machine Learning"
- [ ] [`josephmisiti/awesome-machine-learning`](https://github.com/josephmisiti/awesome-machine-learning)
- [ ] [`hibayesian/awesome-automl-papers`](https://github.com/hibayesian/awesome-automl-papers) (uncertainty-quantification section)
- [ ] [`keiohtani/awesome-bayesian-deep-learning`](https://github.com/keiohtani/awesome-bayesian-deep-learning) (or similar)
- [ ] **Create** `awesome-hyperdimensional-computing` if no such list exists yet — owning the canonical awesome-list for a field is one of the highest-ROI moves available

## Cohort 3 — Research-community visibility

The HDC/VSA research community is small and tightly connected. A handful
of specific people and venues drive most of it.

- [ ] **HDComputing.com** — the canonical HDC community hub (run by Pentti
      Kanerva, Denis Kleyko, et al.). They maintain [a publications list
      via Zotero](https://www.hd-computing.com/publications); ask them to
      add the JMLR paper when it lands.
- [ ] Email the **Kleyko et al. 2022 survey** authors (Denis Kleyko,
      Dmitri Rachkovskij, Evgeny Osipov, Abbas Rahimi) about the library.
      They run a survey-update cadence and have cited every meaningful
      software contribution in the field.
- [ ] **IBM Zurich In-Memory Computing group** (Abbas Rahimi, Manuel
      Le Gallo, Abu Sebastian) — the EMG line started here. Pitch the
      `examples/emg_gesture_recognition.py` demo as a JAX-native
      reproduction of their pipeline.
- [ ] **ETH Zurich Integrated Systems Lab** (Luca Benini, Michael
      Hersche) — same neighbourhood as IBM Zurich, very active on
      neuromorphic HDC.
- [ ] **UC Irvine Cognitive Sciences / DNA computing group** (Mohsen
      Imani, formerly UCSD) — drives a lot of online-HDC work.
- [ ] **TorchHD's authors** (Mike Heddes, Igor Nunes, Pere Vergés). A
      cordial "we built a JAX-native sibling, here's what's different,
      let's cross-reference" email. Cooperative > competitive. Ask them
      to add a "see also: bayes-hdc for JAX + uncertainty" line in
      their README.

## Cohort 4 — JAX ecosystem visibility

- [ ] Open an issue on the [`jax`](https://github.com/google/jax) repo
      asking for inclusion in the "Resources & Awesome JAX" section of
      the README. Frame as: "Probabilistic VSA library for JAX, drop-in
      with flax/equinox; would value a mention."
- [ ] Submit a talk / poster to **JAX Day** (Google's annual JAX event) and
      to any JAX dev meetup
- [ ] Reach out to maintainers of **flax**, **equinox**, **distrax**,
      **blackjax**, **dynamax** — most relevant to: blackjax (probabilistic
      programming), distrax (distributions), dynamax (state-space). One-line
      integration demos open the door to citations.
- [ ] Post in the [JAX Discord](https://discord.gg/jax) `#show-and-tell` channel

## Cohort 5 — Calibration / conformal-prediction community

- [ ] Email **Anastasios Angelopoulos** (UC Berkeley) and **Stephen Bates**
      (MIT) — they maintain `mapie` and the canonical conformal-prediction
      tutorials. The library's `ConformalClassifier` could be cited in
      their software-comparison sections.
- [ ] Submit `bayes-hdc` to the [Conformal Prediction
      Resources](https://github.com/valeman/awesome-conformal-prediction)
      awesome-list

## Cohort 6 — Content marketing

Each blog post should be ≤ 1500 words, embed one runnable code block,
and link to the corresponding example file in the repo.

- [ ] **Post 1: "Why HDC needs a probabilistic layer."** The motivating
      argument. Three concrete failure modes of point-estimate HDC, and
      what PVSA does about them. Channels: personal blog, then cross-post
      to Medium / dev.to.
- [ ] **Post 2: "Closed-form moments under cyclic shift: the algebra
      behind PVSA."** The math, walked through. Targets the equivariant-
      ML / weight-space-learning audience.
- [ ] **Post 3: "Conformal prediction for hyperdimensional classifiers."**
      The conformal layer with worked numbers on UCIHAR. Targets the
      calibration / safe-ML audience.
- [ ] **Post 4: "Building a contextual-bandit agent on PVSA."** Bridge to
      meta-RL. Adds a fifth example to the repo as a side-effect.
- [ ] **Twitter/X thread** announcing v0.4 release with the social card,
      one GIF (PVSA quickstart), and links to the four posts above.
- [ ] **Hacker News submission** — title: "bayes-hdc: probabilistic
      hyperdimensional computing in JAX." Submit during US-morning weekday
      window. The discussion will inevitably ask "why HDC", which is
      exactly the conversation we want.
- [ ] **r/MachineLearning post** — same content, slightly more technical
      framing. Use the "Project" flair.
- [ ] **r/JAX, r/learnmachinelearning** — softer crossposts.

## Cohort 7 — Ongoing community work

- [ ] Set up the **all-contributors** bot to auto-update a Contributors
      section in the README on every merged PR
- [ ] **Mentorship-tagged issues** — every `good first issue` should have
      an explicit "PR me to claim it" comment template
- [ ] **Office hours** — a recurring 30-minute video call on Discussions
      where new users can ask questions. Once a fortnight is sustainable.
- [ ] **Newsletter** — short, monthly. New examples, new releases, new
      papers using the library. Keep it under 200 words / issue.
- [ ] **Annual contributor recognition post** — each Jan, a blog post
      thanking everyone who contributed in the previous year, with stats
      and pointers to follow-on work.

## Metrics to watch

| Metric | Target — month 1 | Target — year 1 |
| --- | ---: | ---: |
| GitHub stars | 50 | 500 |
| External PRs merged | 1 | 25 |
| Cited in arXiv papers | 1 | 5 |
| Downstream library integrations | 0 | 2 |
| Documentation site MAUs | 200 | 5 000 |
| PyPI monthly downloads | 100 | 5 000 |

The TorchHD reference numbers (as of this writing): ~80 stars, ~5
external PRs, multiple cited papers since 2023, no integrations into
PyTorch core libraries. Beating these is the explicit goal.

## What I am *not* going to do

- Buy stars, traffic, or fake engagement.
- Submit to spammy listicles or low-quality SEO-farm "best Python
  libraries" posts.
- Post the same content across more than three platforms in the same
  week.
- Solicit reviews from people whose opinion can be bought.

The reputation costs of any of those outweigh whatever short-term lift
they buy.
