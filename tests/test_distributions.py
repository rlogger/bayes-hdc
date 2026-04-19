# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Tests for the Bayesian hypervector module (`bayes_hdc.distributions`)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from bayes_hdc.distributions import (
    GaussianHV,
    bind_gaussian,
    bundle_gaussian,
    expected_cosine_similarity,
    kl_gaussian,
    similarity_variance,
)

DIMS = 128


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


def test_create_defaults_to_standard_normal() -> None:
    hv = GaussianHV.create(dimensions=DIMS)
    assert hv.dimensions == DIMS
    assert hv.mu.shape == (DIMS,)
    assert hv.var.shape == (DIMS,)
    assert jnp.allclose(hv.mu, 0.0)
    assert jnp.allclose(hv.var, 1.0)


def test_from_sample_is_dirac_with_zero_variance() -> None:
    sample = jnp.arange(DIMS, dtype=jnp.float32)
    hv = GaussianHV.from_sample(sample)
    assert jnp.allclose(hv.mu, sample)
    assert jnp.allclose(hv.var, 0.0)


def test_from_sample_propagates_custom_variance() -> None:
    sample = jnp.ones(DIMS)
    hv = GaussianHV.from_sample(sample, var=0.5)
    assert jnp.allclose(hv.var, 0.5)


def test_random_is_unit_norm_with_isotropic_variance() -> None:
    key = jax.random.PRNGKey(0)
    hv = GaussianHV.random(key, dimensions=DIMS, var=0.01)
    assert jnp.isclose(jnp.linalg.norm(hv.mu), 1.0, atol=1e-5)
    assert jnp.allclose(hv.var, 0.01)


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------


def test_sample_returns_correct_shape() -> None:
    key = jax.random.PRNGKey(1)
    hv = GaussianHV.random(key, dimensions=DIMS, var=0.1)
    sample = hv.sample(jax.random.PRNGKey(2))
    assert sample.shape == (DIMS,)


def test_sample_batch_returns_correct_shape() -> None:
    key = jax.random.PRNGKey(3)
    hv = GaussianHV.random(key, dimensions=DIMS, var=0.1)
    samples = hv.sample_batch(jax.random.PRNGKey(4), n=16)
    assert samples.shape == (16, DIMS)


def test_sample_mean_matches_mu_in_expectation() -> None:
    key = jax.random.PRNGKey(5)
    hv = GaussianHV.random(key, dimensions=DIMS, var=1.0)
    samples = hv.sample_batch(jax.random.PRNGKey(6), n=4096)
    assert jnp.allclose(jnp.mean(samples, axis=0), hv.mu, atol=0.1)


def test_sample_variance_matches_var_in_expectation() -> None:
    key = jax.random.PRNGKey(7)
    hv = GaussianHV.random(key, dimensions=DIMS, var=0.25)
    samples = hv.sample_batch(jax.random.PRNGKey(8), n=8192)
    assert jnp.allclose(jnp.var(samples, axis=0), hv.var, atol=0.05)


def test_sample_clips_negative_variance_to_zero() -> None:
    # Even if a user constructs a pathological state, sampling stays finite.
    hv = GaussianHV(mu=jnp.zeros(DIMS), var=-jnp.ones(DIMS), dimensions=DIMS)
    sample = hv.sample(jax.random.PRNGKey(9))
    assert jnp.all(jnp.isfinite(sample))
    assert jnp.allclose(sample, 0.0)


# ----------------------------------------------------------------------
# Binding
# ----------------------------------------------------------------------


def test_bind_zero_variance_matches_deterministic_map() -> None:
    key = jax.random.PRNGKey(10)
    k1, k2 = jax.random.split(key)
    x_mu = jax.random.normal(k1, (DIMS,))
    y_mu = jax.random.normal(k2, (DIMS,))
    x = GaussianHV.from_sample(x_mu)
    y = GaussianHV.from_sample(y_mu)
    z = bind_gaussian(x, y)
    assert jnp.allclose(z.mu, x_mu * y_mu)
    assert jnp.allclose(z.var, 0.0)


def test_bind_propagates_variance_monotonically() -> None:
    x = GaussianHV.create(DIMS, mu=jnp.ones(DIMS), var=jnp.full((DIMS,), 0.01))
    y_low = GaussianHV.create(DIMS, mu=jnp.ones(DIMS), var=jnp.full((DIMS,), 0.01))
    y_high = GaussianHV.create(DIMS, mu=jnp.ones(DIMS), var=jnp.full((DIMS,), 0.10))
    z_low = bind_gaussian(x, y_low)
    z_high = bind_gaussian(x, y_high)
    assert jnp.all(z_high.var > z_low.var)


def test_bind_matches_analytic_formula() -> None:
    mu_x = jnp.array([2.0])
    mu_y = jnp.array([3.0])
    var_x = jnp.array([0.1])
    var_y = jnp.array([0.2])
    x = GaussianHV.create(1, mu=mu_x, var=var_x)
    y = GaussianHV.create(1, mu=mu_y, var=var_y)
    z = bind_gaussian(x, y)
    # Var[X*Y] = mu_x^2 var_y + mu_y^2 var_x + var_x var_y
    expected_var = mu_x**2 * var_y + mu_y**2 * var_x + var_x * var_y
    assert jnp.allclose(z.var, expected_var)
    assert jnp.allclose(z.mu, mu_x * mu_y)


def test_bind_is_commutative_in_moments() -> None:
    key = jax.random.PRNGKey(11)
    x = GaussianHV.random(key, DIMS)
    y = GaussianHV.random(jax.random.fold_in(key, 1), DIMS)
    z1 = bind_gaussian(x, y)
    z2 = bind_gaussian(y, x)
    assert jnp.allclose(z1.mu, z2.mu)
    assert jnp.allclose(z1.var, z2.var)


# ----------------------------------------------------------------------
# Bundling
# ----------------------------------------------------------------------


def test_bundle_two_hvs_produces_correct_shape() -> None:
    key = jax.random.PRNGKey(20)
    k1, k2 = jax.random.split(key)
    x = GaussianHV.random(k1, DIMS)
    y = GaussianHV.random(k2, DIMS)
    stacked = GaussianHV(
        mu=jnp.stack([x.mu, y.mu]),
        var=jnp.stack([x.var, y.var]),
        dimensions=DIMS,
    )
    bundled = bundle_gaussian(stacked)
    assert bundled.mu.shape == (DIMS,)
    assert bundled.var.shape == (DIMS,)
    # Normalised mean has unit norm.
    assert jnp.isclose(jnp.linalg.norm(bundled.mu), 1.0, atol=1e-5)


def test_bundle_preserves_signal() -> None:
    # Bundling n copies of the same HV should retrieve it with high similarity.
    key = jax.random.PRNGKey(21)
    x = GaussianHV.random(key, DIMS, var=0.01)
    stacked = GaussianHV(
        mu=jnp.stack([x.mu] * 10),
        var=jnp.stack([x.var] * 10),
        dimensions=DIMS,
    )
    bundled = bundle_gaussian(stacked)
    sim = expected_cosine_similarity(x, bundled)
    assert sim > 0.99


# ----------------------------------------------------------------------
# Similarity
# ----------------------------------------------------------------------


def test_expected_cosine_self_similarity_is_one() -> None:
    key = jax.random.PRNGKey(30)
    x = GaussianHV.random(key, DIMS, var=0.0)
    assert jnp.isclose(expected_cosine_similarity(x, x), 1.0, atol=1e-5)


def test_expected_cosine_similarity_random_vectors_is_small() -> None:
    key = jax.random.PRNGKey(31)
    x = GaussianHV.random(key, DIMS)
    y = GaussianHV.random(jax.random.fold_in(key, 1), DIMS)
    sim = expected_cosine_similarity(x, y)
    # Random unit vectors in 128 dims have |cos| < ~0.3 with very high prob.
    assert jnp.abs(sim) < 0.3


def test_similarity_variance_is_non_negative() -> None:
    key = jax.random.PRNGKey(32)
    x = GaussianHV.random(key, DIMS, var=0.1)
    y = GaussianHV.random(jax.random.fold_in(key, 1), DIMS, var=0.1)
    v = similarity_variance(x, y)
    assert v >= 0.0


def test_similarity_variance_zero_for_dirac_hvs() -> None:
    key = jax.random.PRNGKey(33)
    mu = jax.random.normal(key, (DIMS,))
    x = GaussianHV.from_sample(mu)
    y = GaussianHV.from_sample(mu + 1.0)
    assert jnp.isclose(similarity_variance(x, y), 0.0)


# ----------------------------------------------------------------------
# KL divergence
# ----------------------------------------------------------------------


def test_kl_self_is_zero() -> None:
    key = jax.random.PRNGKey(40)
    hv = GaussianHV.random(key, DIMS, var=0.5)
    assert jnp.isclose(kl_gaussian(hv, hv), 0.0, atol=1e-5)


def test_kl_non_negative() -> None:
    key = jax.random.PRNGKey(41)
    p = GaussianHV.random(key, DIMS, var=0.5)
    q = GaussianHV.random(jax.random.fold_in(key, 1), DIMS, var=0.5)
    assert kl_gaussian(p, q) >= 0.0


def test_kl_matches_univariate_formula() -> None:
    # Single-dim check against textbook formula:
    # KL(N(mu_p, var_p) || N(mu_q, var_q))
    #   = 0.5 * (log(var_q / var_p) + (var_p + (mu_p - mu_q)^2) / var_q - 1)
    p = GaussianHV.create(1, mu=jnp.array([0.0]), var=jnp.array([1.0]))
    q = GaussianHV.create(1, mu=jnp.array([1.0]), var=jnp.array([2.0]))
    kl = kl_gaussian(p, q)
    expected = 0.5 * (jnp.log(2.0 / 1.0) + (1.0 + 1.0**2) / 2.0 - 1.0)
    assert jnp.isclose(kl, expected)


# ----------------------------------------------------------------------
# Jit & vmap compatibility
# ----------------------------------------------------------------------


def test_bind_is_jit_compatible() -> None:
    key = jax.random.PRNGKey(50)
    x = GaussianHV.random(key, DIMS)
    y = GaussianHV.random(jax.random.fold_in(key, 1), DIMS)
    jitted = jax.jit(bind_gaussian)
    z = jitted(x, y)
    assert z.mu.shape == (DIMS,)


def test_expected_cosine_similarity_is_vmap_compatible() -> None:
    key = jax.random.PRNGKey(51)
    codebook_mu = jax.random.normal(key, (16, DIMS))
    query = GaussianHV.from_sample(jax.random.normal(jax.random.fold_in(key, 1), (DIMS,)))

    def one_sim(mu: jax.Array) -> jax.Array:
        codebook_hv = GaussianHV.from_sample(mu)
        return expected_cosine_similarity(query, codebook_hv)

    sims = jax.vmap(one_sim)(codebook_mu)
    assert sims.shape == (16,)
    assert jnp.all(sims >= -1.0) and jnp.all(sims <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
