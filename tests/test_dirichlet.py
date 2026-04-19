# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Tests for the Dirichlet hypervector API."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from bayes_hdc.distributions import (
    DirichletHV,
    bind_dirichlet,
    bundle_dirichlet,
    kl_dirichlet,
)

K = 8  # categories


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


def test_create_defaults_to_symmetric_alpha_one() -> None:
    hv = DirichletHV.create(dimensions=K)
    assert hv.dimensions == K
    assert jnp.allclose(hv.alpha, 1.0)


def test_uniform_matches_create_alpha_one() -> None:
    a = DirichletHV.uniform(K)
    b = DirichletHV.create(K, concentration=1.0)
    assert jnp.allclose(a.alpha, b.alpha)


def test_from_counts_adds_prior() -> None:
    counts = jnp.array([5.0, 3.0, 0.0, 2.0])
    hv = DirichletHV.from_counts(counts, prior=1.0)
    assert jnp.allclose(hv.alpha, counts + 1.0)


# ----------------------------------------------------------------------
# Moments
# ----------------------------------------------------------------------


def test_mean_is_on_simplex() -> None:
    counts = jnp.array([10.0, 20.0, 5.0])
    hv = DirichletHV.from_counts(counts, prior=1.0)
    mean = hv.mean()
    assert jnp.isclose(jnp.sum(mean), 1.0)
    assert jnp.all(mean > 0.0)


def test_mean_matches_normalised_counts() -> None:
    counts = jnp.array([10.0, 20.0, 5.0])
    hv = DirichletHV.from_counts(counts, prior=0.0)
    assert jnp.allclose(hv.mean(), counts / jnp.sum(counts))


def test_variance_shrinks_with_concentration() -> None:
    low = DirichletHV.create(K, concentration=1.0)
    high = DirichletHV.create(K, concentration=100.0)
    assert jnp.all(high.variance() < low.variance())


def test_concentration_returns_sum_of_alpha() -> None:
    hv = DirichletHV.from_counts(jnp.array([2.0, 5.0, 3.0]))
    assert jnp.isclose(hv.concentration(), jnp.sum(hv.alpha))


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------


def test_sample_returns_vector_on_simplex() -> None:
    hv = DirichletHV.create(K, concentration=2.0)
    sample = hv.sample(jax.random.PRNGKey(0))
    assert sample.shape == (K,)
    assert jnp.isclose(jnp.sum(sample), 1.0, atol=1e-5)
    assert jnp.all(sample >= 0.0)


def test_sample_batch_shape_and_simplex() -> None:
    hv = DirichletHV.create(K, concentration=2.0)
    samples = hv.sample_batch(jax.random.PRNGKey(1), n=32)
    assert samples.shape == (32, K)
    assert jnp.allclose(jnp.sum(samples, axis=-1), 1.0, atol=1e-5)


def test_sample_mean_converges_to_analytic_mean() -> None:
    hv = DirichletHV.from_counts(jnp.array([1.0, 2.0, 3.0, 4.0]), prior=0.0)
    samples = hv.sample_batch(jax.random.PRNGKey(2), n=8192)
    empirical_mean = jnp.mean(samples, axis=0)
    assert jnp.allclose(empirical_mean, hv.mean(), atol=0.02)


# ----------------------------------------------------------------------
# Binding
# ----------------------------------------------------------------------


def test_bind_produces_mean_on_simplex() -> None:
    x = DirichletHV.from_counts(jnp.array([1.0, 4.0, 2.0]))
    y = DirichletHV.from_counts(jnp.array([3.0, 1.0, 5.0]))
    z = bind_dirichlet(x, y)
    assert jnp.isclose(jnp.sum(z.mean()), 1.0, atol=1e-4)


def test_bind_concentration_is_sum_of_inputs() -> None:
    x = DirichletHV.from_counts(jnp.array([2.0, 3.0, 1.0]))
    y = DirichletHV.from_counts(jnp.array([1.0, 4.0, 2.0]))
    z = bind_dirichlet(x, y)
    expected = x.concentration() + y.concentration()
    # Allow a small EPS slack from the additive guard.
    assert jnp.abs(z.concentration() - expected) < 0.01


def test_bind_with_uniform_prior_preserves_other_mean() -> None:
    """Binding against a high-concentration uniform should barely move the mean."""
    # A very sharp posterior.
    x = DirichletHV.from_counts(jnp.array([100.0, 1.0, 1.0]))
    # A flat uniform with low concentration relative to x.
    y = DirichletHV.create(3, concentration=0.01)
    z = bind_dirichlet(x, y)
    # The sharp posterior's top component should remain dominant.
    assert jnp.argmax(z.mean()) == jnp.argmax(x.mean())


# ----------------------------------------------------------------------
# Bundling
# ----------------------------------------------------------------------


def test_bundle_sums_concentrations() -> None:
    a = DirichletHV.from_counts(jnp.array([1.0, 2.0, 3.0]))
    b = DirichletHV.from_counts(jnp.array([3.0, 1.0, 2.0]))
    stacked = DirichletHV(alpha=jnp.stack([a.alpha, b.alpha]), dimensions=3)
    c = bundle_dirichlet(stacked)
    assert jnp.allclose(c.alpha, a.alpha + b.alpha)


def test_bundle_sharpens_with_more_observations() -> None:
    a = DirichletHV.from_counts(jnp.array([1.0, 2.0, 3.0]))
    stacked = DirichletHV(alpha=jnp.stack([a.alpha] * 10), dimensions=3)
    c = bundle_dirichlet(stacked)
    # Posterior variance should be much smaller with more observations.
    assert jnp.all(c.variance() < a.variance())


# ----------------------------------------------------------------------
# KL divergence
# ----------------------------------------------------------------------


def test_kl_self_is_zero() -> None:
    hv = DirichletHV.from_counts(jnp.array([2.0, 5.0, 3.0]))
    assert jnp.isclose(kl_dirichlet(hv, hv), 0.0, atol=1e-4)


def test_kl_non_negative() -> None:
    p = DirichletHV.from_counts(jnp.array([2.0, 5.0, 3.0]))
    q = DirichletHV.from_counts(jnp.array([1.0, 1.0, 1.0]))
    assert float(kl_dirichlet(p, q)) >= 0.0


def test_kl_asymmetric() -> None:
    p = DirichletHV.from_counts(jnp.array([10.0, 1.0, 1.0]))
    q = DirichletHV.from_counts(jnp.array([1.0, 1.0, 1.0]))
    # KL is not symmetric; both should be positive but unequal.
    assert not jnp.isclose(kl_dirichlet(p, q), kl_dirichlet(q, p))
