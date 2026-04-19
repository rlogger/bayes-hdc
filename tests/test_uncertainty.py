# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Tests for TemperatureCalibrator and ConformalClassifier."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from bayes_hdc.metrics import (
    expected_calibration_error,
    negative_log_likelihood,
)
from bayes_hdc.uncertainty import ConformalClassifier, TemperatureCalibrator

# ----------------------------------------------------------------------
# Helper: synthetic overconfident classifier
# ----------------------------------------------------------------------


def _overconfident_logits(
    key: jax.Array, n: int, k: int, scale: float,
) -> tuple[jax.Array, jax.Array]:
    """Build logits and labels for a classifier whose scores are too confident.

    Labels are the argmax of noise-perturbed mean vectors. Logits are
    ``scale`` * mean — higher ``scale`` = more overconfident softmax.
    """
    key_mean, key_noise = jax.random.split(key)
    means = jax.random.normal(key_mean, (n, k))
    noise = 0.5 * jax.random.normal(key_noise, (n, k))
    labels = jnp.argmax(means, axis=-1)
    logits = scale * (means + noise)
    return logits, labels


# ----------------------------------------------------------------------
# TemperatureCalibrator
# ----------------------------------------------------------------------


def test_calibrator_starts_as_identity() -> None:
    cal = TemperatureCalibrator.create(initial_temperature=1.0)
    logits = jnp.array([[1.0, 2.0, 3.0], [0.5, 0.1, -0.3]])
    probs = cal.calibrate(logits)
    expected = jax.nn.softmax(logits, axis=-1)
    assert jnp.allclose(probs, expected)


def test_calibrator_preserves_argmax() -> None:
    key = jax.random.PRNGKey(0)
    logits, labels = _overconfident_logits(key, n=100, k=5, scale=5.0)
    cal = TemperatureCalibrator.create().fit(logits, labels, max_iters=100)

    raw_preds = jnp.argmax(logits, axis=-1)
    cal_probs = cal.calibrate(logits)
    cal_preds = jnp.argmax(cal_probs, axis=-1)
    # Temperature scaling is accuracy-preserving.
    assert jnp.all(raw_preds == cal_preds)


def test_calibrator_reduces_nll_on_overconfident_classifier() -> None:
    key = jax.random.PRNGKey(1)
    logits, labels = _overconfident_logits(key, n=500, k=5, scale=8.0)

    raw_probs = jax.nn.softmax(logits, axis=-1)
    raw_nll = negative_log_likelihood(raw_probs, labels)

    cal = TemperatureCalibrator.create().fit(logits, labels, max_iters=300)
    cal_probs = cal.calibrate(logits)
    cal_nll = negative_log_likelihood(cal_probs, labels)

    # Calibration should strictly reduce NLL on an overconfident classifier.
    assert cal_nll < raw_nll


def test_calibrator_temperature_greater_than_one_for_overconfident() -> None:
    key = jax.random.PRNGKey(2)
    logits, labels = _overconfident_logits(key, n=400, k=4, scale=10.0)
    cal = TemperatureCalibrator.create().fit(logits, labels, max_iters=300)
    # An overconfident classifier is tempered with T > 1.
    assert float(cal.temperature) > 1.0


def test_calibrator_reduces_ece_on_overconfident_classifier() -> None:
    key = jax.random.PRNGKey(3)
    logits, labels = _overconfident_logits(key, n=1000, k=5, scale=8.0)

    raw_probs = jax.nn.softmax(logits, axis=-1)
    raw_ece = expected_calibration_error(raw_probs, labels, n_bins=15)

    cal = TemperatureCalibrator.create().fit(logits, labels, max_iters=300)
    cal_probs = cal.calibrate(logits)
    cal_ece = expected_calibration_error(cal_probs, labels, n_bins=15)

    assert cal_ece < raw_ece


def test_calibrate_is_jit_compatible() -> None:
    cal = TemperatureCalibrator.create(initial_temperature=2.0)
    logits = jnp.array([[1.0, 2.0, 3.0]])
    jitted = jax.jit(cal.calibrate)
    out = jitted(logits)
    assert out.shape == (1, 3)
    assert jnp.isclose(jnp.sum(out), 1.0, atol=1e-5)


# ----------------------------------------------------------------------
# ConformalClassifier
# ----------------------------------------------------------------------


def _uniform_random_probs(key: jax.Array, n: int, k: int) -> jax.Array:
    """Random probability vectors (each row sums to 1)."""
    alphas = jnp.ones(k)
    return jax.random.dirichlet(key, alphas, shape=(n,))


def test_conformal_predict_set_shape() -> None:
    cal_probs = _uniform_random_probs(jax.random.PRNGKey(10), 200, 5)
    cal_labels = jax.random.randint(jax.random.PRNGKey(11), (200,), 0, 5)
    wrapper = ConformalClassifier.create(alpha=0.1).fit(cal_probs, cal_labels)
    test_probs = _uniform_random_probs(jax.random.PRNGKey(12), 50, 5)
    mask = wrapper.predict_set(test_probs)
    assert mask.shape == (50, 5)
    assert mask.dtype == jnp.bool_


def test_conformal_never_returns_empty_set() -> None:
    cal_probs = _uniform_random_probs(jax.random.PRNGKey(20), 100, 4)
    cal_labels = jax.random.randint(jax.random.PRNGKey(21), (100,), 0, 4)
    wrapper = ConformalClassifier.create(alpha=0.5).fit(cal_probs, cal_labels)
    test_probs = _uniform_random_probs(jax.random.PRNGKey(22), 50, 4)
    mask = wrapper.predict_set(test_probs)
    assert jnp.all(jnp.sum(mask.astype(jnp.int32), axis=-1) >= 1)


def test_conformal_set_size_monotonic_in_alpha() -> None:
    """Smaller alpha (tighter coverage) => larger prediction sets."""
    key = jax.random.PRNGKey(30)
    cal_probs = _uniform_random_probs(key, 500, 6)
    cal_labels = jax.random.randint(jax.random.fold_in(key, 1), (500,), 0, 6)
    test_probs = _uniform_random_probs(jax.random.fold_in(key, 2), 200, 6)

    sizes = []
    for alpha in [0.5, 0.2, 0.1, 0.05]:
        w = ConformalClassifier.create(alpha=alpha).fit(cal_probs, cal_labels)
        sizes.append(float(w.set_size(test_probs)))

    for i in range(len(sizes) - 1):
        assert sizes[i + 1] >= sizes[i]


def test_conformal_coverage_close_to_nominal() -> None:
    """On exchangeable data, empirical coverage should track 1 - alpha."""
    # Build a calibratable classifier: logits are signal + noise;
    # we softmax to get probs.
    key = jax.random.PRNGKey(40)
    k_classes = 5
    n_cal = 1000
    n_test = 2000
    scale = 2.0

    key_cal, key_test = jax.random.split(key)

    def build(key: jax.Array, n: int) -> tuple[jax.Array, jax.Array]:
        k1, k2 = jax.random.split(key)
        means = jax.random.normal(k1, (n, k_classes))
        noise = 0.3 * jax.random.normal(k2, (n, k_classes))
        labels = jnp.argmax(means, axis=-1)
        probs = jax.nn.softmax(scale * (means + noise), axis=-1)
        return probs, labels

    cal_probs, cal_labels = build(key_cal, n_cal)
    test_probs, test_labels = build(key_test, n_test)

    alpha = 0.1
    wrapper = ConformalClassifier.create(alpha=alpha).fit(cal_probs, cal_labels)
    cov = float(wrapper.coverage(test_probs, test_labels))
    # Split conformal guarantees marginal coverage >= 1 - alpha. The APS
    # score can over-cover due to the finite-sample ceiling correction;
    # a one-sided tolerance is the right test.
    assert cov >= (1.0 - alpha) - 0.03  # account for test-set sampling slack
    assert cov <= (1.0 - alpha) + 0.10  # should not wildly over-cover


def test_conformal_coverage_is_jit_compatible() -> None:
    cal_probs = _uniform_random_probs(jax.random.PRNGKey(50), 200, 3)
    cal_labels = jax.random.randint(jax.random.PRNGKey(51), (200,), 0, 3)
    wrapper = ConformalClassifier.create(alpha=0.2).fit(cal_probs, cal_labels)
    test_probs = _uniform_random_probs(jax.random.PRNGKey(52), 50, 3)
    test_labels = jax.random.randint(jax.random.PRNGKey(53), (50,), 0, 3)
    jitted = jax.jit(wrapper.coverage)
    cov = jitted(test_probs, test_labels)
    assert 0.0 <= float(cov) <= 1.0
