# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Tests for BayesianCentroidClassifier."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayes_hdc.bayesian_models import BayesianCentroidClassifier
from bayes_hdc.distributions import GaussianHV

DIMS = 32
N_CLASSES = 4


def _synthetic_data(key: jax.Array, n_per_class: int = 20, noise: float = 0.3):
    """Build a K-class cloud dataset: each class is a Gaussian centred on a random unit vector."""
    keys = jax.random.split(key, N_CLASSES + 1)
    centres = jax.random.normal(keys[0], (N_CLASSES, DIMS))
    centres = centres / (jnp.linalg.norm(centres, axis=-1, keepdims=True) + 1e-8)

    X_list, y_list = [], []
    for c in range(N_CLASSES):
        cloud = centres[c] + noise * jax.random.normal(keys[c + 1], (n_per_class, DIMS))
        X_list.append(cloud)
        y_list.append(jnp.full((n_per_class,), c, dtype=jnp.int32))
    return jnp.concatenate(X_list), jnp.concatenate(y_list)


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


def test_create_yields_standard_normal_prior() -> None:
    clf = BayesianCentroidClassifier.create(num_classes=N_CLASSES, dimensions=DIMS)
    assert clf.num_classes == N_CLASSES
    assert clf.dimensions == DIMS
    assert clf.mu.shape == (N_CLASSES, DIMS)
    assert clf.var.shape == (N_CLASSES, DIMS)
    assert jnp.allclose(clf.mu, 0.0)
    assert jnp.allclose(clf.var, 1.0)


def test_fit_empty_data_raises() -> None:
    clf = BayesianCentroidClassifier.create(num_classes=N_CLASSES, dimensions=DIMS)
    with pytest.raises(ValueError, match="empty"):
        clf.fit(jnp.zeros((0, DIMS)), jnp.zeros((0,), dtype=jnp.int32))


def test_fit_populates_mu_from_empirical_mean() -> None:
    key = jax.random.PRNGKey(0)
    X, y = _synthetic_data(key, n_per_class=50, noise=0.05)
    clf = BayesianCentroidClassifier.create(
        num_classes=N_CLASSES,
        dimensions=DIMS,
    ).fit(X, y)
    for c in range(N_CLASSES):
        class_mean = jnp.mean(X[y == c], axis=0)
        assert jnp.allclose(clf.mu[c], class_mean, atol=1e-5)


def test_fit_variance_is_non_negative_and_bounded() -> None:
    key = jax.random.PRNGKey(1)
    X, y = _synthetic_data(key, n_per_class=30)
    clf = BayesianCentroidClassifier.create(
        num_classes=N_CLASSES,
        dimensions=DIMS,
    ).fit(X, y, prior_strength=0.1)
    assert jnp.all(clf.var >= 0.0)
    assert jnp.all(clf.var < 10.0)  # bounded under reasonable noise


def test_fit_with_higher_prior_strength_widens_variance() -> None:
    key = jax.random.PRNGKey(2)
    X, y = _synthetic_data(key, n_per_class=30, noise=0.1)
    low_prior = BayesianCentroidClassifier.create(N_CLASSES, DIMS).fit(
        X,
        y,
        prior_strength=0.01,
    )
    high_prior = BayesianCentroidClassifier.create(N_CLASSES, DIMS).fit(
        X,
        y,
        prior_strength=5.0,
    )
    # Higher prior → pulls variance toward prior (higher), so mean variance grows.
    assert jnp.mean(high_prior.var) > jnp.mean(low_prior.var)


# ----------------------------------------------------------------------
# class_posterior
# ----------------------------------------------------------------------


def test_class_posterior_returns_gaussian_hv() -> None:
    key = jax.random.PRNGKey(3)
    X, y = _synthetic_data(key)
    clf = BayesianCentroidClassifier.create(N_CLASSES, DIMS).fit(X, y)
    post = clf.class_posterior(2)
    assert isinstance(post, GaussianHV)
    assert post.dimensions == DIMS
    assert jnp.allclose(post.mu, clf.mu[2])
    assert jnp.allclose(post.var, clf.var[2])


# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------


def test_predict_recovers_training_labels_on_clean_data() -> None:
    key = jax.random.PRNGKey(4)
    X, y = _synthetic_data(key, n_per_class=40, noise=0.02)
    clf = BayesianCentroidClassifier.create(N_CLASSES, DIMS).fit(X, y)
    preds = clf.predict(X)
    # Low-noise synthetic data → should recover nearly everything.
    assert float(jnp.mean(preds == y)) > 0.95


def test_predict_handles_single_and_batch() -> None:
    key = jax.random.PRNGKey(5)
    X, y = _synthetic_data(key)
    clf = BayesianCentroidClassifier.create(N_CLASSES, DIMS).fit(X, y)
    # Batch
    batch_preds = clf.predict(X)
    assert batch_preds.shape == (X.shape[0],)
    # Single
    single = clf.predict(X[0])
    assert single.shape == ()
    assert int(single) == int(batch_preds[0])


def test_predict_proba_rows_sum_to_one() -> None:
    key = jax.random.PRNGKey(6)
    X, y = _synthetic_data(key)
    clf = BayesianCentroidClassifier.create(N_CLASSES, DIMS).fit(X, y)
    probs = clf.predict_proba(X)
    assert probs.shape == (X.shape[0], N_CLASSES)
    assert jnp.allclose(jnp.sum(probs, axis=-1), 1.0, atol=1e-5)
    assert jnp.all(probs >= 0.0)


def test_predict_proba_argmax_matches_predict() -> None:
    key = jax.random.PRNGKey(7)
    X, y = _synthetic_data(key)
    clf = BayesianCentroidClassifier.create(N_CLASSES, DIMS).fit(X, y)
    probs = clf.predict_proba(X)
    assert jnp.all(jnp.argmax(probs, axis=-1) == clf.predict(X))


def test_logits_handles_single_and_batch_and_matches_softmax() -> None:
    """Public ``logits`` is the canonical input to TemperatureCalibrator / ConformalClassifier."""
    key = jax.random.PRNGKey(11)
    X, y = _synthetic_data(key)
    clf = BayesianCentroidClassifier.create(N_CLASSES, DIMS).fit(X, y)

    # Batch path returns (N, K).
    batch_logits = clf.logits(X)
    assert batch_logits.shape == (X.shape[0], N_CLASSES)

    # Single path returns (K,).
    single_logits = clf.logits(X[0])
    assert single_logits.shape == (N_CLASSES,)
    assert jnp.allclose(single_logits, batch_logits[0], atol=1e-6)

    # softmax(logits) must equal predict_proba — proves logits is the
    # genuine pre-softmax score, suitable for downstream calibration.
    assert jnp.allclose(jax.nn.softmax(batch_logits, axis=-1), clf.predict_proba(X), atol=1e-6)


def test_score_matches_manual_accuracy() -> None:
    key = jax.random.PRNGKey(8)
    X, y = _synthetic_data(key, noise=0.05)
    clf = BayesianCentroidClassifier.create(N_CLASSES, DIMS).fit(X, y)
    s = float(clf.score(X, y))
    manual = float(jnp.mean(clf.predict(X) == y))
    assert np.isclose(s, manual)


# ----------------------------------------------------------------------
# Uncertainty
# ----------------------------------------------------------------------


def test_predict_uncertainty_shape_batch_and_single() -> None:
    key = jax.random.PRNGKey(9)
    X, y = _synthetic_data(key)
    clf = BayesianCentroidClassifier.create(N_CLASSES, DIMS).fit(X, y)
    batch_unc = clf.predict_uncertainty(X)
    single_unc = clf.predict_uncertainty(X[0])
    assert batch_unc.shape == (X.shape[0], N_CLASSES)
    assert single_unc.shape == (N_CLASSES,)
    assert jnp.allclose(single_unc, batch_unc[0])


def test_predict_uncertainty_non_negative() -> None:
    key = jax.random.PRNGKey(10)
    X, y = _synthetic_data(key)
    clf = BayesianCentroidClassifier.create(N_CLASSES, DIMS).fit(X, y)
    unc = clf.predict_uncertainty(X)
    assert jnp.all(unc >= 0.0)


def test_predict_uncertainty_matches_analytic_formula() -> None:
    # With zero variance per class (hand-built), uncertainty is zero.
    clf = BayesianCentroidClassifier(
        mu=jnp.ones((2, DIMS)),
        var=jnp.zeros((2, DIMS)),
        num_classes=2,
        dimensions=DIMS,
    )
    query = jax.random.normal(jax.random.PRNGKey(0), (DIMS,))
    unc = clf.predict_uncertainty(query)
    assert jnp.allclose(unc, 0.0)


def test_predict_uncertainty_grows_with_posterior_variance() -> None:
    low = BayesianCentroidClassifier(
        mu=jnp.zeros((2, DIMS)),
        var=jnp.full((2, DIMS), 0.01),
        num_classes=2,
        dimensions=DIMS,
    )
    high = BayesianCentroidClassifier(
        mu=jnp.zeros((2, DIMS)),
        var=jnp.full((2, DIMS), 1.0),
        num_classes=2,
        dimensions=DIMS,
    )
    query = jnp.ones(DIMS)
    assert jnp.all(high.predict_uncertainty(query) > low.predict_uncertainty(query))


# ----------------------------------------------------------------------
# predict_with_uncertainty combined
# ----------------------------------------------------------------------


def test_predict_with_uncertainty_returns_three_arrays_consistent() -> None:
    key = jax.random.PRNGKey(11)
    X, y = _synthetic_data(key)
    clf = BayesianCentroidClassifier.create(N_CLASSES, DIMS).fit(X, y)
    preds, probs, unc = clf.predict_with_uncertainty(X)
    assert jnp.all(preds == clf.predict(X))
    assert jnp.allclose(probs, clf.predict_proba(X))
    assert jnp.allclose(unc, clf.predict_uncertainty(X))


# ----------------------------------------------------------------------
# Pytree compatibility
# ----------------------------------------------------------------------


def test_classifier_is_jit_compatible() -> None:
    key = jax.random.PRNGKey(12)
    X, y = _synthetic_data(key)
    clf = BayesianCentroidClassifier.create(N_CLASSES, DIMS).fit(X, y)
    jitted = jax.jit(clf.predict_proba)
    probs = jitted(X)
    assert probs.shape == (X.shape[0], N_CLASSES)
