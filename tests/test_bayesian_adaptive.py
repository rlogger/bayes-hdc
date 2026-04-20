# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Tests for BayesianAdaptiveHDC (Kalman-style online classifier)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from bayes_hdc.bayesian_models import BayesianAdaptiveHDC

DIMS = 32
N_CLASSES = 3


def _synth(key: jax.Array, n_per_class: int = 25, noise: float = 0.1):
    keys = jax.random.split(key, N_CLASSES + 1)
    centres = jax.random.normal(keys[0], (N_CLASSES, DIMS))
    centres = centres / (jnp.linalg.norm(centres, axis=-1, keepdims=True) + 1e-8)
    Xs, ys = [], []
    for c in range(N_CLASSES):
        cloud = centres[c] + noise * jax.random.normal(keys[c + 1], (n_per_class, DIMS))
        Xs.append(cloud)
        ys.append(jnp.full((n_per_class,), c, dtype=jnp.int32))
    return jnp.concatenate(Xs), jnp.concatenate(ys)


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


def test_create_has_broad_prior() -> None:
    clf = BayesianAdaptiveHDC.create(
        num_classes=N_CLASSES,
        dimensions=DIMS,
        prior_var=2.0,
        obs_var=0.5,
    )
    assert clf.num_classes == N_CLASSES
    assert clf.dimensions == DIMS
    assert clf.mu.shape == (N_CLASSES, DIMS)
    assert clf.var.shape == (N_CLASSES, DIMS)
    assert jnp.allclose(clf.mu, 0.0)
    assert jnp.allclose(clf.var, 2.0)
    assert clf.obs_var == 0.5


def test_fit_empty_raises() -> None:
    clf = BayesianAdaptiveHDC.create(num_classes=N_CLASSES, dimensions=DIMS)
    with pytest.raises(ValueError, match="empty"):
        clf.fit(jnp.zeros((0, DIMS)), jnp.zeros((0,), dtype=jnp.int32))


# ----------------------------------------------------------------------
# Kalman update correctness
# ----------------------------------------------------------------------


def test_single_update_shrinks_variance() -> None:
    clf = BayesianAdaptiveHDC.create(
        num_classes=2,
        dimensions=DIMS,
        prior_var=1.0,
        obs_var=0.1,
    )
    sample = jnp.ones(DIMS)
    updated = clf.update(sample, 0)
    # Class-0 variance must shrink; class-1 unchanged.
    assert jnp.all(updated.var[0] < clf.var[0])
    assert jnp.allclose(updated.var[1], clf.var[1])


def test_single_update_formula() -> None:
    # Known univariate: prior mu=0, var=1; obs=10, obs_var=0.1.
    # New mu = (0.1 * 0 + 1 * 10) / (0.1 + 1) = 10 / 1.1 ≈ 9.0909
    # New var = (0.1 * 1) / 1.1 ≈ 0.0909
    clf = BayesianAdaptiveHDC.create(
        num_classes=2,
        dimensions=1,
        prior_var=1.0,
        obs_var=0.1,
    )
    sample = jnp.array([10.0])
    updated = clf.update(sample, 0)
    assert jnp.isclose(updated.mu[0, 0], 10.0 / 1.1, atol=1e-5)
    assert jnp.isclose(updated.var[0, 0], 0.1 / 1.1, atol=1e-5)


def test_many_updates_converge_toward_sample_mean() -> None:
    clf = BayesianAdaptiveHDC.create(
        num_classes=2,
        dimensions=4,
        prior_var=1.0,
        obs_var=0.01,
    )
    target = jnp.array([1.0, 2.0, 3.0, 4.0])
    for _ in range(100):
        clf = clf.update(target, 0)
    # With many updates of the same vector, mean should be close to target.
    assert jnp.allclose(clf.mu[0], target, atol=0.1)


def test_variance_monotonically_shrinks() -> None:
    clf = BayesianAdaptiveHDC.create(
        num_classes=1,
        dimensions=DIMS,
        prior_var=1.0,
        obs_var=0.1,
    )
    key = jax.random.PRNGKey(0)
    prev = float(jnp.mean(clf.var))
    for i in range(10):
        sample = jax.random.normal(jax.random.fold_in(key, i), (DIMS,))
        clf = clf.update(sample, 0)
        curr = float(jnp.mean(clf.var))
        assert curr < prev
        prev = curr


# ----------------------------------------------------------------------
# fit / predict on synthetic data
# ----------------------------------------------------------------------


def test_fit_then_predict_recovers_labels() -> None:
    key = jax.random.PRNGKey(1)
    X, y = _synth(key, n_per_class=30, noise=0.03)
    clf = BayesianAdaptiveHDC.create(N_CLASSES, DIMS, obs_var=0.5).fit(X, y)
    preds = clf.predict(X)
    # With low noise and Kalman updates, accuracy should be high.
    assert float(jnp.mean(preds == y)) > 0.85


def test_predict_proba_sums_to_one() -> None:
    key = jax.random.PRNGKey(2)
    X, y = _synth(key)
    clf = BayesianAdaptiveHDC.create(N_CLASSES, DIMS).fit(X, y)
    probs = clf.predict_proba(X)
    assert jnp.allclose(jnp.sum(probs, axis=-1), 1.0, atol=1e-5)


def test_predict_uncertainty_shrinks_with_more_updates() -> None:
    """Repeated updates to a class should lower its similarity variance."""
    key = jax.random.PRNGKey(3)
    X, y = _synth(key, n_per_class=5)
    clf_few = BayesianAdaptiveHDC.create(N_CLASSES, DIMS, obs_var=0.1).fit(X, y)
    clf_many = BayesianAdaptiveHDC.create(N_CLASSES, DIMS, obs_var=0.1).fit(X, y, epochs=5)
    query = X[0]
    unc_few = float(jnp.sum(clf_few.predict_uncertainty(query)))
    unc_many = float(jnp.sum(clf_many.predict_uncertainty(query)))
    assert unc_many < unc_few


def test_predict_single_and_batch_consistent() -> None:
    key = jax.random.PRNGKey(4)
    X, y = _synth(key)
    clf = BayesianAdaptiveHDC.create(N_CLASSES, DIMS).fit(X, y)
    single = int(clf.predict(X[0]))
    batch = int(clf.predict(X)[0])
    assert single == batch


def test_score_matches_manual_accuracy() -> None:
    key = jax.random.PRNGKey(5)
    X, y = _synth(key, noise=0.05)
    clf = BayesianAdaptiveHDC.create(N_CLASSES, DIMS).fit(X, y)
    s = float(clf.score(X, y))
    manual = float(jnp.mean(clf.predict(X) == y))
    assert jnp.isclose(s, manual)


def test_multiple_epochs_reduce_variance() -> None:
    key = jax.random.PRNGKey(6)
    X, y = _synth(key, n_per_class=10)
    clf_1 = BayesianAdaptiveHDC.create(N_CLASSES, DIMS, obs_var=0.1).fit(X, y, epochs=1)
    clf_3 = BayesianAdaptiveHDC.create(N_CLASSES, DIMS, obs_var=0.1).fit(X, y, epochs=3)
    assert jnp.mean(clf_3.var) < jnp.mean(clf_1.var)


def test_jit_compatible() -> None:
    key = jax.random.PRNGKey(7)
    X, y = _synth(key)
    clf = BayesianAdaptiveHDC.create(N_CLASSES, DIMS).fit(X, y)
    probs = jax.jit(clf.predict_proba)(X)
    assert probs.shape == (X.shape[0], N_CLASSES)
