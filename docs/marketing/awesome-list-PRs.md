# Awesome-list submissions — ready-to-file PRs

Passive-discovery surfaces for bayes-hdc. Each entry below is paste-ready.
File them one at a time, spaced out (don't fire all in one day from a fresh
account — looks like spam). Lead with the anomaly-detection + conformal +
JAX angle; that's what differentiates us from every other entry on these
lists.

Canonical one-liner (reuse, trim per list style):

> **bayes-hdc** — calibrated one-shot anomaly detection in JAX, plus a full
> probabilistic hyperdimensional-computing stack: conformal classifier /
> regressor / anomaly detector with finite-sample coverage, closed-form
> Gaussian & Dirichlet hypervectors, pytree-native (`jit`/`vmap`/`grad`).

---

## 1. n2cholas/awesome-jax

- **Repo:** https://github.com/n2cholas/awesome-jax
- **Section:** Libraries → (new line under the modelling / probabilistic entries)
- **Status:** PR #144 may already be pending from an earlier submission — check before re-filing; if it's stale, comment on it rather than opening a duplicate.
- **Entry:**
  ```
  - [bayes-hdc](https://github.com/rlogger/bayes-hdc) - Probabilistic hyperdimensional computing: conformal anomaly detection, Gaussian/Dirichlet hypervectors, calibrated prediction sets. <img src="https://img.shields.io/github/stars/rlogger/bayes-hdc?style=social" align="center">
  ```
- **PR title:** Add bayes-hdc (probabilistic hyperdimensional computing)
- **PR body:** bayes-hdc is a JAX-native library for hyperdimensional computing / vector symbolic architectures with a probabilistic layer — closed-form Gaussian and Dirichlet hypervectors, conformal prediction (classifier, regressor, one-class anomaly detector), and variational codebook training, all pytree-native so `jit`/`vmap`/`grad`/`pmap` compose. MIT, 644 tests, docs at rlogger.github.io/bayes-hdc. Fits under Libraries alongside the other probabilistic-JAX entries (NumPyro, BlackJAX, Distrax).

## 2. valeman/awesome-conformal-prediction

- **Repo:** https://github.com/valeman/awesome-conformal-prediction
- **Section:** Python → Libraries (or "Conformal Anomaly Detection" subsection if present)
- **Maintainer responsiveness:** Active; this list is well-curated and merges relevant high-quality entries.
- **Entry:**
  ```
  - [bayes-hdc](https://github.com/rlogger/bayes-hdc) - Split-conformal classifier, regressor, and one-class anomaly detector on hyperdimensional representations, in JAX. Finite-sample coverage, pytree-native.
  ```
- **PR title:** Add bayes-hdc — conformal classification, regression & anomaly detection in JAX
- **PR body:** bayes-hdc ships split-conformal prediction across three modes — APS classifier, absolute-residual regressor (Lei et al. 2018), and a one-class anomaly detector with conformal p-values (Laxhammar 2014; Bates et al. 2023) — on hyperdimensional-computing encoders, in JAX. Distinct from the rest of the list in that the conformal layer wraps an HDC backbone and composes with `jax.jit`/`vmap`/`grad`.

## 3. yzhao062/anomaly-detection-resources

- **Repo:** https://github.com/yzhao062/anomaly-detection-resources
- **Section:** Algorithms / Open-source libraries
- **Maintainer responsiveness:** Large, periodically curated; PRs may sit but do get batched.
- **Entry:**
  ```
  - [bayes-hdc](https://github.com/rlogger/bayes-hdc): calibrated one-class anomaly detection with finite-sample false-positive guarantees, on hyperdimensional encoders, in JAX.
  ```
- **PR title:** Add bayes-hdc (conformal HDC anomaly detection)
- **PR body:** bayes-hdc provides a one-class `ConformalAnomalyDetector`: fit on normal data, get split-conformal p-values back, control the false-positive rate at a target α with a finite-sample guarantee. HDC backbone means it works few-shot. JAX, MIT.

## 4. valeman/awesome-conformal-anomaly-detection (if it exists) / fallback hoya012/awesome-anomaly-detection

- **Repo:** https://github.com/hoya012/awesome-anomaly-detection
- **Section:** Survey / Methods (libraries are sparse here — frame as a method+implementation)
- **Maintainer responsiveness:** Slower; this is more a paper list, so only worth it if a libraries section exists.
- **Entry:** same canonical one-liner.

## 5. awesome-bayesian-deep-learning

- **Repo:** search current canonical (e.g. https://github.com/robi56/awesome-deep-learning-uncertainty or sjchoi86/bayes-nn) — verify the live one before filing.
- **Section:** Libraries / tools.
- **Entry:** canonical one-liner, emphasising Gaussian/Dirichlet posteriors + variational codebook training.

## 6. jbhuang0604/awesome-uncertainty-deeplearning (or jbsimoes equivalents)

- **Repo:** https://github.com/ENSTA-U2IS-AI/awesome-uncertainty-deeplearning
- **Section:** Libraries.
- **Maintainer responsiveness:** Active, academic.
- **Entry:**
  ```
  - [bayes-hdc](https://github.com/rlogger/bayes-hdc) - Calibration (temperature scaling), conformal prediction sets, and conformal anomaly detection on hyperdimensional encoders, JAX-native.
  ```

---

## Filing order & cadence

1. awesome-conformal-prediction (best fit, responsive) — file first.
2. awesome-jax (check PR #144 first).
3. awesome-uncertainty-deeplearning.
4. anomaly-detection-resources.
5. The rest as bandwidth allows.

Space them ~2–3 days apart. After each merge, the stars-badge in the entry
gives passive social proof. Re-verify each list's exact section name at file
time — awesome-lists reorganise.
