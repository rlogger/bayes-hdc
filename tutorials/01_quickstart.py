# SPDX-License-Identifier: MIT
# Copyright (c) 2026 R.S.

"""Bayes-HDC quickstart — installation to first prediction in 90 seconds.

Six self-contained sections. Each block is independent enough to paste
into a fresh REPL after the imports above it have been run. Total
runtime end-to-end is a few seconds on CPU.

References:
    Furlong & Eliasmith (2024) — distribution-valued VSAs.
    Liang et al. (2026) — ConformalHDC: prediction sets for HDC.
    Frady, Kleyko & Sommer (2020) — VSA capacity and resonator nets.
    Lei et al. (2018) — split-conformal classification.
"""

from __future__ import annotations

# =====================================================================
# 1. Install
# ---------------------------------------------------------------------
# Bayes-HDC is on PyPI:
#
#     pip install bayes-hdc
#
# CPU JAX is pulled in as a default. For GPU/TPU follow the JAX install
# matrix at https://jax.readthedocs.io/en/latest/installation.html and
# `pip install bayes-hdc --no-deps` to layer it on top.
# =====================================================================
# =====================================================================
# 2. A Gaussian hypervector + bind in 4 lines
# ---------------------------------------------------------------------
# PVSA (Furlong & Eliasmith 2024) treats every hypervector as a
# posterior. `bind_gaussian` propagates the first two moments in closed
# form, so binding is differentiable end-to-end.
# =====================================================================
import jax
import jax.numpy as jnp

from bayes_hdc import GaussianHV, bind_gaussian, expected_cosine_similarity

key_a, key_b = jax.random.split(jax.random.PRNGKey(0))
x = GaussianHV.random(key_a, dimensions=2048, var=1e-3)
y = GaussianHV.random(key_b, dimensions=2048, var=1e-3)
z = bind_gaussian(x, y)
print(f"[2] bound HV: mu[:3]={z.mu[:3]}  E[cos(x, z)]={expected_cosine_similarity(x, z):+.3f}")


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
# ConformalClassifier (Lei et al. 2018; Liang et al. 2026 for HDC) gives
# a marginal coverage guarantee Pr(y* in C(x*)) >= 1 - alpha on a held
# -out split. Split the test set into cal/test halves so the conformal
# guarantee actually applies.
# =====================================================================

from bayes_hdc import ConformalClassifier, TemperatureCalibrator  # noqa: E402

n_cal = test_hvs.shape[0] // 2
cal_hvs, cal_y = test_hvs[:n_cal], y_test[:n_cal]
eval_hvs, eval_y = test_hvs[n_cal:], y_test[n_cal:]

logits_cal = jax.vmap(clf.similarity)(cal_hvs)
logits_eval = jax.vmap(clf.similarity)(eval_hvs)

calibrator = TemperatureCalibrator.create().fit(logits_cal, cal_y)
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
# in-distribution cosine distances as the threshold, and any test
# point above that threshold is flagged as anomalous with marginal
# false-alarm rate <= alpha (Lei et al. 2018, applied to one class).
# =====================================================================

key_in, key_out = jax.random.split(jax.random.PRNGKey(7))
X_in = jax.random.normal(key_in, (400, dims)) * 0.05 + jnp.ones(dims) / jnp.sqrt(dims)
X_out = jax.random.normal(key_out, (200, dims)) * 0.5
centroid = X_in[:200].mean(axis=0)
centroid = centroid / (jnp.linalg.norm(centroid) + 1e-9)
scores_cal = 1.0 - jax.vmap(lambda v: jnp.dot(centroid, v) / (jnp.linalg.norm(v) + 1e-9))(
    X_in[200:]
)
threshold = jnp.quantile(scores_cal, 0.9)
scores_out = 1.0 - jax.vmap(lambda v: jnp.dot(centroid, v) / (jnp.linalg.norm(v) + 1e-9))(X_out)
print(f"[5] anomaly recall on synthetic OOD = {float((scores_out > threshold).mean()):.2f}")


# =====================================================================
# 6. Where to next
# ---------------------------------------------------------------------
# Longer worked examples live in `tutorials/`:
#
#   02_bayesian_hypervectors.py   — Dirichlet HVs, KL, posterior PPCs.
#   03_calibration_and_coverage.py — ECE/MCE, reliability curves, CP.
#   04_resonator_factorisation.py — probabilistic resonator networks.
#   05_real_data_eeg.py           — seizure detection end-to-end.
#
# The applied notebooks in `examples/` solve specific problems (EEG,
# EMG, image classification, language ID); the `tutorials/` series is
# pedagogical and meant to be read in order. Full API reference at
# https://rlogger.github.io/bayes-hdc.
# =====================================================================


if __name__ == "__main__":
    print("\nQuickstart complete. See tutorials/README.md for the rest of the series.")
