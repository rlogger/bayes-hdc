# SPDX-License-Identifier: MIT
# Copyright (c) 2026 R.S.

"""Bayes-HDC quickstart: Gaussian moments, classification and calibration.

Run sections in order. Proper training, temperature calibration, conformal
calibration and test observations are kept separate. Marginal guarantees
require exchangeability; the printed metrics are one finite sample.

Run: python tutorials/01_quickstart.py
"""

from __future__ import annotations

# =====================================================================
# 1. Install
# ---------------------------------------------------------------------
# Bayes-HDC is on PyPI:
#
#     pip install "bayes-hdc[datasets]"
#
# CPU JAX is pulled in as a default. For GPU/TPU follow the JAX install
# matrix at https://jax.readthedocs.io/en/latest/installation.html and
# install this package with a compatible accelerator JAX version.
# =====================================================================
# =====================================================================
# 2. A Gaussian hypervector + bind in 4 lines
# ---------------------------------------------------------------------
# GaussianHV describes a Gaussian distribution over coordinates. Independent
# product moments are exact; a product distribution is generally not Gaussian.
# A chosen representation distribution is not automatically a posterior.
# =====================================================================
import jax
import jax.numpy as jnp

from bayes_hdc import GaussianHV, bind_gaussian, expected_cosine_similarity

key_a, key_b = jax.random.split(jax.random.PRNGKey(0))
x = GaussianHV.random(key_a, dimensions=2048, var=1e-3)
y = GaussianHV.random(key_b, dimensions=2048, var=1e-3)
z = bind_gaussian(x, y)
print(
    f"[2] bound HV: mu[:3]={z.mu[:3]}  "
    f"approx independent E[cos(x, y)]={expected_cosine_similarity(x, y):+.3f}"
)


# =====================================================================
# 3. A tiny classifier on iris (RandomEncoder + CentroidClassifier)
# ---------------------------------------------------------------------
# RandomEncoder needs discrete features, so we quantise each iris
# feature into `n_bins` buckets first. CentroidClassifier then learns
# one prototype per class in a single pass — no backprop.
# =====================================================================

from bayes_hdc import CentroidClassifier, RandomEncoder  # noqa: E402
from bayes_hdc.datasets import load_iris  # noqa: E402

iris = load_iris(test_size=0.5, random_state=0)
n_bins, dims = 16, 4096


def quantise(x_raw, X_ref, n_bins=n_bins):
    lo, hi = X_ref.min(axis=0), X_ref.max(axis=0)
    return jnp.clip(
        jnp.floor((x_raw - lo) / (hi - lo + 1e-9) * n_bins).astype(jnp.int32), 0, n_bins - 1
    )


X_train = quantise(jnp.asarray(iris.X_train), iris.X_train)
X_test = quantise(jnp.asarray(iris.X_test), iris.X_train)
y_train = jnp.asarray(iris.y_train)
y_test = jnp.asarray(iris.y_test)

encoder = RandomEncoder.create(iris.n_features, n_bins, dims, key=jax.random.PRNGKey(1))
train_hvs, test_hvs = encoder.encode_batch(X_train), encoder.encode_batch(X_test)

clf = CentroidClassifier.create(iris.n_classes, dims).fit(train_hvs, y_train)
print(f"[3] iris test accuracy = {float(clf.score(test_hvs, y_test)):.3f}")


# =====================================================================
# 4. Calibrated + conformal prediction sets
# ---------------------------------------------------------------------
# Raw cosine similarities make weak probabilities. TemperatureCalibrator
# (Guo et al. 2017) learns one scalar T to fix that, and
# ConformalClassifier uses deterministic APS (Romano, Sesia & Candes 2020) for
# a marginal coverage guarantee Pr(y* in C(x*)) >= 1 - alpha on a held
# -out split with exchangeable calibration/test observations. Reserve separate
# temperature/conformal/test portions before fitting either calibration stage.
# =====================================================================

from bayes_hdc import ConformalClassifier, TemperatureCalibrator  # noqa: E402

n_part = test_hvs.shape[0] // 3
temp_hvs, temp_y = test_hvs[:n_part], y_test[:n_part]
cal_hvs, cal_y = test_hvs[n_part : 2 * n_part], y_test[n_part : 2 * n_part]
eval_hvs, eval_y = test_hvs[2 * n_part :], y_test[2 * n_part :]

logits_cal = jax.vmap(clf.similarity)(cal_hvs)
logits_eval = jax.vmap(clf.similarity)(eval_hvs)

calibrator = TemperatureCalibrator.create().fit(jax.vmap(clf.similarity)(temp_hvs), temp_y)
probs_cal = calibrator.calibrate(logits_cal)
probs_eval = calibrator.calibrate(logits_eval)

conformal = ConformalClassifier.create(alpha=0.1).fit(probs_cal, cal_y)
coverage = float(conformal.coverage(probs_eval, eval_y))
mean_size = float(conformal.set_size(probs_eval))
print(
    f"[4] T={float(calibrator.temperature):.2f}  coverage={coverage:.2f} "
    f"(target 0.90)  |C|={mean_size:.2f}"
)


# =====================================================================
# 5. Anomaly detection in 5 lines
# ---------------------------------------------------------------------
# Library-first split-conformal anomaly score: fit a centroid on a
# synthetic "in-distribution" cluster, take the (1 - alpha)-quantile of
# in-distribution cosine distances using the finite-sample rank correction.
# Test points exceeding the calibrated threshold have marginal
# false-alarm rate <= alpha (Bates, Candes, Lei, Romano & Sesia 2023).
# =====================================================================

key_in, key_out = jax.random.split(jax.random.PRNGKey(7))
X_in = jax.random.normal(key_in, (400, dims)) * 0.05 + jnp.ones(dims) / jnp.sqrt(dims)
X_out = jax.random.normal(key_out, (200, dims)) * 0.5
from bayes_hdc import ConformalAnomalyDetector, HDCAnomalyScorer  # noqa: E402

scorer = HDCAnomalyScorer.create(dimensions=dims).fit(X_in[:200])
detector = ConformalAnomalyDetector.create(scorer).fit(X_in[200:])
flags_out = detector.predict_batch(X_out, alpha=0.1)
print(f"[5] anomaly recall on synthetic OOD = {float(flags_out.mean()):.2f}")


# =====================================================================
# 6. Where to next
# ---------------------------------------------------------------------
# Longer worked examples live in `tutorials/`:
#
#   02_anomaly_detection.py — split-conformal anomaly examples.
#   03_sequences.py         — sequence retrieval and storage tradeoffs.
#   ../examples/resonator_factorisation.py — stochastic factor search.
#   ../examples/eeg_seizure_detection.py   — synthetic EEG-style features.
#
# The applied notebooks in `examples/` solve specific problems (EEG,
# EMG, image classification, language ID); the `tutorials/` series is
# pedagogical and meant to be read in order. Full API reference at
# https://rlogger.github.io/bayes-hdc.
# =====================================================================


if __name__ == "__main__":
    print("\nQuickstart complete. See tutorials/README.md for the rest of the series.")
