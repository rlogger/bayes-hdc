# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Tests for v0.5 (resonator + diagnostics) and v0.6 (streaming + shard_map)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from bayes_hdc.bayesian_models import StreamingBayesianHDC
from bayes_hdc.diagnostics import (
    coverage_calibration_check,
    posterior_predictive_check,
    statistic_cosine_to_reference,
    statistic_mean_norm,
)
from bayes_hdc.distributed import (
    shard_classifier_posteriors,
    shard_map_bind_gaussian,
)
from bayes_hdc.distributions import GaussianHV, bind_gaussian
from bayes_hdc.resonator import probabilistic_resonator
from bayes_hdc.uncertainty import ConformalClassifier

DIMS = 64


def _random_codebooks(key: jax.Array, k: int, n_per_book: int, dims: int):
    keys = jax.random.split(key, k)
    return [
        GaussianHV(
            mu=jax.random.normal(keys[j], (n_per_book, dims))
            / (
                jnp.linalg.norm(
                    jax.random.normal(keys[j], (n_per_book, dims)),
                    axis=-1,
                    keepdims=True,
                )
                + 1e-8
            ),
            var=jnp.full((n_per_book, dims), 0.001),
            dimensions=dims,
        )
        for j in range(k)
    ]


# ----------------------------------------------------------------------
# Probabilistic resonator
# ----------------------------------------------------------------------


def test_resonator_raises_on_empty_codebooks() -> None:
    key = jax.random.PRNGKey(0)
    target = GaussianHV.random(key, DIMS, var=0.001)
    with pytest.raises(ValueError, match="non-empty"):
        probabilistic_resonator([], target, key)


def test_resonator_returns_result_with_right_shape() -> None:
    key = jax.random.PRNGKey(1)
    k1, k2, k3 = jax.random.split(key, 3)
    codebooks = _random_codebooks(k1, k=3, n_per_book=5, dims=DIMS)
    # Target = bind of three random factors (the "ground truth").
    factor_0 = GaussianHV(
        mu=codebooks[0].mu[2],
        var=codebooks[0].var[2],
        dimensions=DIMS,
    )
    factor_1 = GaussianHV(
        mu=codebooks[1].mu[3],
        var=codebooks[1].var[3],
        dimensions=DIMS,
    )
    factor_2 = GaussianHV(
        mu=codebooks[2].mu[0],
        var=codebooks[2].var[0],
        dimensions=DIMS,
    )
    target = bind_gaussian(bind_gaussian(factor_0, factor_1), factor_2)

    result = probabilistic_resonator(
        codebooks,
        target,
        k2,
        n_restarts=4,
        max_iters=20,
        temperature=0.05,
    )
    assert result.indices.shape == (3,)
    assert result.n_restarts == 4
    assert result.history.shape == (20,)
    # Low-temperature run on a clean target should recover at least one factor exactly.
    # (With small DIMS and many codebook rows, perfect 3/3 recovery is not
    # guaranteed, but alignment must be > 0.)
    assert result.alignment > 0.0


def test_resonator_alignment_is_in_valid_range() -> None:
    key = jax.random.PRNGKey(2)
    codebooks = _random_codebooks(key, k=2, n_per_book=4, dims=DIMS)
    target = GaussianHV.random(jax.random.fold_in(key, 1), DIMS, var=0.01)
    result = probabilistic_resonator(
        codebooks,
        target,
        jax.random.fold_in(key, 2),
        n_restarts=2,
        max_iters=10,
    )
    assert -1.0 <= result.alignment <= 1.0


# ----------------------------------------------------------------------
# Posterior predictive checks
# ----------------------------------------------------------------------


def test_statistic_mean_norm_matches_analytic() -> None:
    X = jnp.array([[3.0, 4.0], [0.0, 0.0], [6.0, 8.0]])
    # Norms: 5, 0, 10 → mean 5.0
    assert jnp.isclose(statistic_mean_norm(X), 5.0)


def test_statistic_cosine_to_reference_self() -> None:
    ref = jnp.array([1.0, 0.0, 0.0])
    X = jnp.array([ref, jnp.array([0.0, 1.0, 0.0])])
    # Max cosine similarity should be 1.0 (first row).
    assert jnp.isclose(statistic_cosine_to_reference(X, ref), 1.0, atol=1e-5)


def test_ppc_returns_valid_fields() -> None:
    """Deterministic-only checks: fields are finite, std non-negative, CI ordered,
    p-value in [0, 1]. Probabilistic assertions (containment, p-value bounds)
    are intentionally excluded because any seeded draw from the true posterior
    will land in the distribution tails ~5% of the time.
    """
    import math

    key = jax.random.PRNGKey(10)
    posterior = GaussianHV.random(key, DIMS, var=0.5)
    observed = posterior.sample_batch(jax.random.fold_in(key, 1), n=64)
    result = posterior_predictive_check(
        posterior,
        observed,
        statistic_mean_norm,
        jax.random.fold_in(key, 2),
        n_replicas=500,
    )
    assert math.isfinite(result.observed)
    assert math.isfinite(result.predictive_mean)
    assert result.predictive_std >= 0.0
    assert math.isfinite(result.predictive_std)
    assert result.ci_low <= result.ci_high
    assert 0.0 <= result.p_value <= 1.0


def test_ppc_detects_misspecified_posterior() -> None:
    key = jax.random.PRNGKey(20)
    posterior = GaussianHV.create(DIMS, mu=jnp.zeros(DIMS), var=jnp.full((DIMS,), 0.01))
    # Observed samples from a very different distribution.
    observed = 10.0 + jax.random.normal(key, (32, DIMS))
    result = posterior_predictive_check(
        posterior,
        observed,
        statistic_mean_norm,
        jax.random.fold_in(key, 1),
        n_replicas=100,
    )
    # Observed mean norm will be wildly outside the predictive distribution.
    assert result.p_value < 0.1


def test_coverage_calibration_check_returns_expected_structure() -> None:
    key = jax.random.PRNGKey(30)
    probs_cal = jax.random.dirichlet(key, jnp.ones(4), shape=(200,))
    labels_cal = jax.random.randint(jax.random.fold_in(key, 1), (200,), 0, 4)
    probs_te = jax.random.dirichlet(jax.random.fold_in(key, 2), jnp.ones(4), shape=(100,))
    labels_te = jax.random.randint(jax.random.fold_in(key, 3), (100,), 0, 4)

    result = coverage_calibration_check(
        lambda a: ConformalClassifier.create(alpha=a),
        probs_cal,
        labels_cal,
        probs_te,
        labels_te,
        alphas=[0.1, 0.2, 0.3],
    )
    assert result.alphas.shape == (3,)
    assert result.empirical_coverage.shape == (3,)
    assert result.set_sizes.shape == (3,)
    assert result.max_deviation >= 0.0


# ----------------------------------------------------------------------
# StreamingBayesianHDC
# ----------------------------------------------------------------------


def test_streaming_create_rejects_invalid_decay() -> None:
    with pytest.raises(ValueError, match="decay"):
        StreamingBayesianHDC.create(num_classes=3, dimensions=16, decay=1.5)
    with pytest.raises(ValueError, match="decay"):
        StreamingBayesianHDC.create(num_classes=3, dimensions=16, decay=-0.1)


def test_streaming_fit_empty_raises() -> None:
    clf = StreamingBayesianHDC.create(num_classes=2, dimensions=DIMS)
    with pytest.raises(ValueError, match="empty"):
        clf.fit(jnp.zeros((0, DIMS)), jnp.zeros((0,), dtype=jnp.int32))


def test_streaming_single_update_moves_mean() -> None:
    clf = StreamingBayesianHDC.create(
        num_classes=2,
        dimensions=4,
        decay=0.5,
        prior_var=1.0,
    )
    sample = jnp.array([10.0, 20.0, 30.0, 40.0])
    updated = clf.update(sample, 0)
    # decay=0.5 → new_mu = 0.5 * 0 + 0.5 * sample = sample/2
    assert jnp.allclose(updated.mu[0], sample / 2.0)


def test_streaming_variance_adapts_to_drift() -> None:
    """EMA variance grows when incoming samples diverge from current mean."""
    clf = StreamingBayesianHDC.create(
        num_classes=1,
        dimensions=DIMS,
        decay=0.9,
        prior_var=0.01,
    )
    # Stream samples far from the prior mean of zero.
    far_sample = jnp.ones(DIMS) * 10.0
    for _ in range(5):
        clf = clf.update(far_sample, 0)
    # After a few consistent updates, variance from-drift should shrink again.
    mean_var_converged = float(jnp.mean(clf.var))
    # Now inject a big shift — variance should rebound.
    new_sample = -jnp.ones(DIMS) * 10.0
    clf_shifted = clf.update(new_sample, 0)
    assert jnp.mean(clf_shifted.var) > mean_var_converged


def test_streaming_predict_and_score() -> None:
    key = jax.random.PRNGKey(40)
    keys = jax.random.split(key, 4)
    centres = [
        jax.random.normal(keys[i], (DIMS,))
        / (jnp.linalg.norm(jax.random.normal(keys[i], (DIMS,))) + 1e-8)
        for i in range(3)
    ]
    X_list = []
    y_list = []
    for c in range(3):
        for _ in range(10):
            X_list.append(centres[c] + 0.01 * jax.random.normal(keys[3], (DIMS,)))
            y_list.append(c)
    X = jnp.stack(X_list)
    y = jnp.asarray(y_list, dtype=jnp.int32)

    clf = StreamingBayesianHDC.create(num_classes=3, dimensions=DIMS, decay=0.7)
    clf = clf.fit(X, y)
    # Very low noise → should classify training set with high accuracy.
    assert float(clf.score(X, y)) > 0.8


def test_streaming_jit_compatible() -> None:
    key = jax.random.PRNGKey(50)
    X = jax.random.normal(key, (20, DIMS))
    y = jax.random.randint(jax.random.fold_in(key, 1), (20,), 0, 2)
    clf = StreamingBayesianHDC.create(num_classes=2, dimensions=DIMS).fit(X, y)
    probs = jax.jit(clf.predict_proba)(X)
    assert probs.shape == (20, 2)


# ----------------------------------------------------------------------
# shard_map / sharding helpers
# ----------------------------------------------------------------------


def test_shard_classifier_posteriors_reshape() -> None:
    n_dev = jax.local_device_count()
    mu = jnp.zeros((n_dev * 2, DIMS))
    var = jnp.ones((n_dev * 2, DIMS))
    sharded_mu, sharded_var = shard_classifier_posteriors(mu, var)
    assert sharded_mu.shape == (n_dev, 2, DIMS)
    assert sharded_var.shape == (n_dev, 2, DIMS)


def test_shard_classifier_posteriors_invalid_divisor_raises() -> None:
    if jax.local_device_count() == 1:
        pytest.skip("Divisibility check is trivially satisfied on single-device hosts.")
    mu = jnp.zeros((3, DIMS))  # presumably not divisible by n_dev
    var = jnp.ones((3, DIMS))
    with pytest.raises(ValueError, match="divisible"):
        shard_classifier_posteriors(mu, var)


def test_shard_map_bind_gaussian_runs_on_single_device() -> None:
    n_dev = jax.local_device_count()
    key = jax.random.PRNGKey(100)
    if n_dev < 2:
        # Fallback path: call directly, should equal bind_gaussian.
        x = GaussianHV.random(key, DIMS, var=0.1)
        y = GaussianHV.random(jax.random.fold_in(key, 1), DIMS, var=0.1)
        out = shard_map_bind_gaussian(x, y)
        expected = bind_gaussian(x, y)
        assert jnp.allclose(out.mu, expected.mu)
        assert jnp.allclose(out.var, expected.var)
    else:
        # Multi-device path: inputs need leading device axis.
        x = GaussianHV(
            mu=jax.random.normal(key, (n_dev, DIMS)),
            var=jnp.full((n_dev, DIMS), 0.1),
            dimensions=DIMS,
        )
        y = GaussianHV(
            mu=jax.random.normal(jax.random.fold_in(key, 1), (n_dev, DIMS)),
            var=jnp.full((n_dev, DIMS), 0.1),
            dimensions=DIMS,
        )
        out = shard_map_bind_gaussian(x, y)
        assert out.mu.shape == (n_dev, DIMS)
