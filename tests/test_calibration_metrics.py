# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Tests for calibration metrics in bayes_hdc.metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from bayes_hdc.metrics import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    negative_log_likelihood,
    reliability_curve,
    sharpness,
)

# ----------------------------------------------------------------------
# ECE
# ----------------------------------------------------------------------


def test_ece_zero_for_perfectly_calibrated_classifier() -> None:
    # If we report uniform 1/k confidence and are uniformly random, bin
    # confidence and accuracy are both 1/k in the matching bin.
    k = 4
    n = 1000
    probs = jnp.full((n, k), 1.0 / k)
    labels = jax.random.randint(jax.random.PRNGKey(0), (n,), 0, k)
    ece = expected_calibration_error(probs, labels, n_bins=20)
    # Accuracy ≈ 1/k = 0.25, confidence = 0.25. ECE should be tiny.
    assert float(ece) < 0.05


def test_ece_positive_for_overconfident_classifier() -> None:
    # Report 99% confidence but only be correct 60% of the time.
    n = 1000
    k = 4
    probs = jnp.tile(jnp.array([0.99, 0.005, 0.0033, 0.0017]), (n, 1))
    labels = jax.random.randint(jax.random.PRNGKey(1), (n,), 0, k)
    ece = expected_calibration_error(probs, labels, n_bins=15)
    # Confidence 0.99, accuracy ~0.25, bin 15 gap ~0.74, weight ~1.0.
    assert float(ece) > 0.5


def test_ece_in_unit_interval() -> None:
    n = 500
    k = 10
    probs = jax.random.dirichlet(jax.random.PRNGKey(2), jnp.ones(k), shape=(n,))
    labels = jax.random.randint(jax.random.PRNGKey(3), (n,), 0, k)
    ece = expected_calibration_error(probs, labels, n_bins=15)
    assert 0.0 <= float(ece) <= 1.0


# ----------------------------------------------------------------------
# MCE
# ----------------------------------------------------------------------


def test_mce_at_least_ece() -> None:
    """The max bin gap can never be smaller than the weighted average."""
    n = 500
    k = 5
    probs = jax.random.dirichlet(jax.random.PRNGKey(10), jnp.ones(k), shape=(n,))
    labels = jax.random.randint(jax.random.PRNGKey(11), (n,), 0, k)
    ece = expected_calibration_error(probs, labels, n_bins=15)
    mce = maximum_calibration_error(probs, labels, n_bins=15)
    assert float(mce) >= float(ece) - 1e-6


# ----------------------------------------------------------------------
# Brier
# ----------------------------------------------------------------------


def test_brier_zero_for_perfectly_confident_correct_predictions() -> None:
    k = 3
    n = 10
    labels = jnp.arange(n) % k
    probs = jax.nn.one_hot(labels, k)
    score = brier_score(probs, labels, n_classes=k)
    assert jnp.isclose(score, 0.0)


def test_brier_bounded_by_one_minus_inv_k_for_uniform() -> None:
    k = 5
    n = 100
    probs = jnp.full((n, k), 1.0 / k)
    labels = jax.random.randint(jax.random.PRNGKey(20), (n,), 0, k)
    score = brier_score(probs, labels, n_classes=k)
    # Uniform prediction has Brier = 1 - 1/k = 0.8 for k=5.
    assert jnp.isclose(score, 1.0 - 1.0 / k, atol=0.01)


def test_brier_positive_for_wrong_confident_predictions() -> None:
    k = 4
    n = 100
    probs = jax.nn.one_hot(jnp.zeros(n, dtype=jnp.int32), k)
    labels = jnp.ones(n, dtype=jnp.int32)  # always wrong
    score = brier_score(probs, labels, n_classes=k)
    # Each sample contributes 1 + 1 = 2 (first index wrongly predicts 1;
    # true label slot wrongly predicts 0).
    assert jnp.isclose(score, 2.0)


# ----------------------------------------------------------------------
# Sharpness
# ----------------------------------------------------------------------


def test_sharpness_matches_1_over_k_for_uniform() -> None:
    k = 7
    probs = jnp.full((32, k), 1.0 / k)
    assert jnp.isclose(sharpness(probs), 1.0 / k)


def test_sharpness_is_one_for_deterministic_predictor() -> None:
    k = 5
    n = 10
    probs = jax.nn.one_hot(jnp.arange(n) % k, k)
    assert jnp.isclose(sharpness(probs), 1.0)


# ----------------------------------------------------------------------
# NLL
# ----------------------------------------------------------------------


def test_nll_zero_for_perfectly_confident_correct_predictions() -> None:
    k = 3
    n = 10
    labels = jnp.arange(n) % k
    probs = jax.nn.one_hot(labels, k)
    # log(1 - EPS) ≈ 0, so NLL is tiny (EPS clamp protects the log).
    assert float(negative_log_likelihood(probs, labels)) < 1e-5


def test_nll_positive_for_incorrect_predictions() -> None:
    k = 3
    n = 100
    labels = jnp.zeros(n, dtype=jnp.int32)
    probs = jax.nn.one_hot(jnp.ones(n, dtype=jnp.int32), k)
    nll = float(negative_log_likelihood(probs, labels))
    assert nll > 10.0  # log(EPS) is very negative


# ----------------------------------------------------------------------
# Reliability curve
# ----------------------------------------------------------------------


def test_reliability_curve_shapes() -> None:
    n = 200
    k = 4
    probs = jax.random.dirichlet(jax.random.PRNGKey(30), jnp.ones(k), shape=(n,))
    labels = jax.random.randint(jax.random.PRNGKey(31), (n,), 0, k)
    centers, accs, confs, counts = reliability_curve(probs, labels, n_bins=10)
    assert centers.shape == (10,)
    assert accs.shape == (10,)
    assert confs.shape == (10,)
    assert counts.shape == (10,)
    assert float(jnp.sum(counts)) == n


def test_reliability_curve_accs_and_confs_in_unit_interval() -> None:
    n = 100
    k = 3
    probs = jax.random.dirichlet(jax.random.PRNGKey(40), jnp.ones(k), shape=(n,))
    labels = jax.random.randint(jax.random.PRNGKey(41), (n,), 0, k)
    _, accs, confs, _ = reliability_curve(probs, labels, n_bins=10)
    assert jnp.all(accs >= 0.0) and jnp.all(accs <= 1.0)
    assert jnp.all(confs >= 0.0) and jnp.all(confs <= 1.0)
