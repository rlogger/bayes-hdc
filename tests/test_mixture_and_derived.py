# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Tests for MixtureHV, permute_gaussian, and cleanup_gaussian."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from bayes_hdc.distributions import (
    GaussianHV,
    MixtureHV,
    cleanup_gaussian,
    cleanup_gaussian_stacked,
    expected_cosine_similarity,
    permute_gaussian,
)

DIMS = 64


# ----------------------------------------------------------------------
# permute_gaussian
# ----------------------------------------------------------------------


def test_permute_shifts_mean_and_variance() -> None:
    mu = jnp.arange(DIMS, dtype=jnp.float32)
    var = jnp.arange(DIMS, dtype=jnp.float32) / 10.0
    hv = GaussianHV(mu=mu, var=var, dimensions=DIMS)
    shifted = permute_gaussian(hv, shifts=3)
    assert jnp.allclose(shifted.mu, jnp.roll(mu, 3))
    assert jnp.allclose(shifted.var, jnp.roll(var, 3))


def test_permute_with_full_cycle_returns_to_original() -> None:
    key = jax.random.PRNGKey(0)
    hv = GaussianHV.random(key, DIMS)
    shifted = permute_gaussian(hv, shifts=DIMS)
    assert jnp.allclose(shifted.mu, hv.mu)
    assert jnp.allclose(shifted.var, hv.var)


def test_permute_reverses_with_opposite_shift() -> None:
    key = jax.random.PRNGKey(1)
    hv = GaussianHV.random(key, DIMS)
    there = permute_gaussian(hv, shifts=5)
    back = permute_gaussian(there, shifts=-5)
    assert jnp.allclose(back.mu, hv.mu)
    assert jnp.allclose(back.var, hv.var)


def test_permute_is_jit_compatible() -> None:
    key = jax.random.PRNGKey(2)
    hv = GaussianHV.random(key, DIMS)
    jitted = jax.jit(permute_gaussian)
    out = jitted(hv, 1)
    assert out.mu.shape == (DIMS,)


# ----------------------------------------------------------------------
# cleanup_gaussian
# ----------------------------------------------------------------------


def test_cleanup_retrieves_exact_match_in_memory() -> None:
    key = jax.random.PRNGKey(10)
    keys = jax.random.split(key, 5)
    memory = [GaussianHV.random(k, DIMS, var=0.0) for k in keys]
    query = memory[3]  # pick a known entry
    idx, score = cleanup_gaussian(query, memory)
    assert idx == 3
    assert score > 0.99


def test_cleanup_picks_nearest_under_noise() -> None:
    key = jax.random.PRNGKey(11)
    keys = jax.random.split(key, 4)
    memory = [GaussianHV.random(k, DIMS, var=0.0) for k in keys]
    # Query is a noisy version of memory[1].
    noise_key = jax.random.PRNGKey(99)
    noise = 0.1 * jax.random.normal(noise_key, (DIMS,))
    query = GaussianHV(
        mu=memory[1].mu + noise,
        var=jnp.zeros(DIMS),
        dimensions=DIMS,
    )
    idx, _ = cleanup_gaussian(query, memory)
    assert idx == 1


def test_cleanup_empty_memory_raises() -> None:
    key = jax.random.PRNGKey(12)
    hv = GaussianHV.random(key, DIMS)
    try:
        cleanup_gaussian(hv, [])
    except ValueError as e:
        assert "non-empty" in str(e)
    else:
        raise AssertionError("cleanup_gaussian should raise on empty memory")


def test_cleanup_gaussian_stacked_matches_list_form() -> None:
    """Stacked-form cleanup is a JIT-friendly mirror of the list form."""
    keys = jax.random.split(jax.random.PRNGKey(33), 5)
    memory = [GaussianHV.random(k, DIMS) for k in keys[:4]]
    query = memory[2]  # exact match available

    list_idx, list_score = cleanup_gaussian(query, memory)

    stacked = GaussianHV(
        mu=jnp.stack([m.mu for m in memory]),
        var=jnp.stack([m.var for m in memory]),
        dimensions=DIMS,
    )
    stacked_idx, stacked_score = cleanup_gaussian_stacked(query, stacked)

    assert int(stacked_idx) == list_idx
    assert jnp.allclose(stacked_score, list_score, atol=1e-5)


def test_cleanup_gaussian_stacked_is_jit_compatible() -> None:
    """The stacked variant must compose with jit and vmap end-to-end."""
    keys = jax.random.split(jax.random.PRNGKey(34), 6)
    stacked = GaussianHV(
        mu=jnp.stack([GaussianHV.random(k, DIMS).mu for k in keys[:5]]),
        var=jnp.stack([GaussianHV.random(k, DIMS).var for k in keys[:5]]),
        dimensions=DIMS,
    )
    queries = jnp.stack([GaussianHV.random(keys[5], DIMS).mu for _ in range(3)])

    def best_idx_for(query_mu: jax.Array) -> jax.Array:
        q = GaussianHV(
            mu=query_mu,
            var=jnp.zeros_like(query_mu),
            dimensions=DIMS,
        )
        idx, _ = cleanup_gaussian_stacked(q, stacked)
        return idx

    indices = jax.vmap(best_idx_for)(queries)
    assert indices.shape == (3,)
    assert jnp.all((indices >= 0) & (indices < 5))


# ----------------------------------------------------------------------
# MixtureHV
# ----------------------------------------------------------------------


def test_from_components_uniform_weights_by_default() -> None:
    key = jax.random.PRNGKey(20)
    k1, k2 = jax.random.split(key)
    a = GaussianHV.random(k1, DIMS, var=0.1)
    b = GaussianHV.random(k2, DIMS, var=0.1)
    mix = MixtureHV.from_components([a, b])
    assert mix.weights.shape == (2,)
    assert jnp.allclose(mix.weights, 0.5)
    assert mix.mu.shape == (2, DIMS)
    assert mix.var.shape == (2, DIMS)


def test_from_components_custom_weights_normalised() -> None:
    key = jax.random.PRNGKey(21)
    k1, k2 = jax.random.split(key)
    a = GaussianHV.random(k1, DIMS)
    b = GaussianHV.random(k2, DIMS)
    mix = MixtureHV.from_components([a, b], weights=jnp.array([3.0, 1.0]))
    assert jnp.isclose(jnp.sum(mix.weights), 1.0)
    assert jnp.isclose(mix.weights[0], 0.75)


def test_create_uniform_mixture() -> None:
    mix = MixtureHV.create(dimensions=DIMS, n_components=3)
    assert mix.mu.shape == (3, DIMS)
    assert jnp.allclose(mix.weights, 1.0 / 3.0)


def test_mean_is_weighted_average_of_component_means() -> None:
    mu1 = jnp.ones(DIMS) * 2.0
    mu2 = jnp.ones(DIMS) * 4.0
    mix = MixtureHV.from_components(
        [
            GaussianHV.create(DIMS, mu=mu1, var=jnp.ones(DIMS)),
            GaussianHV.create(DIMS, mu=mu2, var=jnp.ones(DIMS)),
        ],
        weights=jnp.array([0.25, 0.75]),
    )
    # Expected: 0.25 * 2 + 0.75 * 4 = 3.5
    assert jnp.allclose(mix.mean(), 3.5)


def test_variance_follows_law_of_total_variance() -> None:
    # Two components at different means → total variance > any single component.
    mu1 = jnp.zeros(DIMS)
    mu2 = jnp.ones(DIMS)
    var = jnp.full((DIMS,), 0.1)
    mix = MixtureHV.from_components(
        [GaussianHV.create(DIMS, mu=mu1, var=var), GaussianHV.create(DIMS, mu=mu2, var=var)],
    )
    # Component variance = 0.1; inter-component separation adds extra.
    assert jnp.all(mix.variance() > 0.1)


def test_collapse_to_gaussian_preserves_moments() -> None:
    mu1 = jnp.array([1.0, 2.0])
    mu2 = jnp.array([3.0, 4.0])
    var = jnp.array([0.1, 0.2])
    mix = MixtureHV.from_components(
        [GaussianHV.create(2, mu=mu1, var=var), GaussianHV.create(2, mu=mu2, var=var)],
    )
    collapsed = mix.collapse_to_gaussian()
    assert jnp.allclose(collapsed.mu, mix.mean())
    assert jnp.allclose(collapsed.var, mix.variance())
    assert collapsed.dimensions == 2


def test_sample_returns_single_hypervector() -> None:
    mix = MixtureHV.create(dimensions=DIMS, n_components=3)
    sample = mix.sample(jax.random.PRNGKey(30))
    assert sample.shape == (DIMS,)


def test_sample_mean_matches_mixture_mean_in_expectation() -> None:
    key = jax.random.PRNGKey(31)
    k1, k2 = jax.random.split(key)
    mu1 = jnp.ones(DIMS) * 1.0
    mu2 = jnp.ones(DIMS) * 5.0
    mix = MixtureHV.from_components(
        [
            GaussianHV.create(DIMS, mu=mu1, var=jnp.full((DIMS,), 0.01)),
            GaussianHV.create(DIMS, mu=mu2, var=jnp.full((DIMS,), 0.01)),
        ],
    )
    samples = jnp.stack([mix.sample(jax.random.fold_in(k1, i)) for i in range(1000)])
    empirical_mean = jnp.mean(samples, axis=0)
    # Expected: 0.5 * 1 + 0.5 * 5 = 3.0
    assert jnp.allclose(empirical_mean, 3.0, atol=0.2)


def test_empty_components_raises() -> None:
    try:
        MixtureHV.from_components([])
    except ValueError as e:
        assert "non-empty" in str(e)
    else:
        raise AssertionError("MixtureHV.from_components should raise on empty list")


# ----------------------------------------------------------------------
# Cross-type: mixture → Gaussian → similarity
# ----------------------------------------------------------------------


def test_mixture_collapse_then_similarity() -> None:
    key = jax.random.PRNGKey(40)
    k1, k2, k3 = jax.random.split(key, 3)
    mix = MixtureHV.from_components(
        [GaussianHV.random(k1, DIMS, var=0.01), GaussianHV.random(k2, DIMS, var=0.01)],
    )
    other = GaussianHV.random(k3, DIMS, var=0.01)
    collapsed = mix.collapse_to_gaussian()
    sim = float(expected_cosine_similarity(collapsed, other))
    assert -1.0 <= sim <= 1.0
