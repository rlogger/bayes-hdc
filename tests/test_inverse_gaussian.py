# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Tests for inverse_gaussian and reparameterisation-gradient flow."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from bayes_hdc.distributions import (
    GaussianHV,
    bind_gaussian,
    bundle_gaussian,
    expected_cosine_similarity,
    inverse_gaussian,
    kl_gaussian,
)

DIMS = 64


# ----------------------------------------------------------------------
# inverse_gaussian — delta-method formula
# ----------------------------------------------------------------------


def test_inverse_zero_variance_recovers_classical_map_inverse() -> None:
    mu = jnp.array([2.0, -3.0, 0.5, 4.0], dtype=jnp.float32)
    hv = GaussianHV.from_sample(mu)  # var = 0
    inv = inverse_gaussian(hv)
    assert jnp.allclose(inv.mu, 1.0 / mu, atol=1e-5)
    assert jnp.allclose(inv.var, 0.0)


def test_inverse_mask_zeroes_tiny_mean_components() -> None:
    mu = jnp.array([1.0, 1e-12, -1.0, 1e-15], dtype=jnp.float32)
    var = jnp.full_like(mu, 0.01)
    hv = GaussianHV(mu=mu, var=var, dimensions=4)
    inv = inverse_gaussian(hv)
    # Components 1 and 3 were below eps — zeroed out.
    assert float(inv.mu[1]) == 0.0
    assert float(inv.mu[3]) == 0.0
    assert float(inv.var[1]) == 0.0
    assert float(inv.var[3]) == 0.0
    # Components 0 and 2 preserved (with correction).
    assert float(inv.mu[0]) != 0.0
    assert float(inv.mu[2]) != 0.0


def test_inverse_variance_matches_delta_method_formula() -> None:
    # Known univariate case: x ~ N(2, 0.1), 1/x ≈ N(1/2 + 0.1/2^3, 0.1/2^4)
    # = N(0.5125, 0.00625)
    hv = GaussianHV(
        mu=jnp.array([2.0]),
        var=jnp.array([0.1]),
        dimensions=1,
    )
    inv = inverse_gaussian(hv)
    assert jnp.isclose(inv.mu[0], 0.5125, atol=1e-4)
    assert jnp.isclose(inv.var[0], 0.00625, atol=1e-4)


def test_inverse_is_jit_compatible() -> None:
    key = jax.random.PRNGKey(0)
    hv = GaussianHV.random(key, DIMS, var=0.01)
    jitted = jax.jit(inverse_gaussian)
    out = jitted(hv)
    assert out.mu.shape == (DIMS,)
    assert out.var.shape == (DIMS,)


def test_inverse_bind_approximately_recovers_input_under_low_noise() -> None:
    """bind(bind(x, y), inverse(y)) ≈ x  when variance is small."""
    key = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(key)
    # Non-zero-centred means to avoid singularity; small variance.
    x = GaussianHV(
        mu=2.0 + jax.random.normal(k1, (DIMS,)) * 0.1,
        var=jnp.full((DIMS,), 1e-5),
        dimensions=DIMS,
    )
    y = GaussianHV(
        mu=2.0 + jax.random.normal(k2, (DIMS,)) * 0.1,
        var=jnp.full((DIMS,), 1e-5),
        dimensions=DIMS,
    )
    bound = bind_gaussian(x, y)
    y_inv = inverse_gaussian(y)
    recovered = bind_gaussian(bound, y_inv)
    # Cosine similarity should be extremely high.
    sim = float(expected_cosine_similarity(x, recovered))
    assert sim > 0.99


# ----------------------------------------------------------------------
# Reparameterisation gradients
# ----------------------------------------------------------------------


def test_grad_flows_through_bind_gaussian() -> None:
    """jax.grad composes through bind_gaussian — demonstrates PVSA is differentiable."""
    k_tgt, k_y, k_init = jax.random.split(jax.random.PRNGKey(10), 3)
    target_mu = jax.random.normal(k_tgt, (DIMS,))
    target_mu = target_mu / (jnp.linalg.norm(target_mu) + 1e-8)
    y_mu = jax.random.normal(k_y, (DIMS,))  # independent of target
    y_mu = y_mu / (jnp.linalg.norm(y_mu) + 1e-8)

    def loss(mu: jax.Array) -> jax.Array:
        x = GaussianHV(mu=mu, var=jnp.full((DIMS,), 0.1), dimensions=DIMS)
        y = GaussianHV.from_sample(y_mu)
        z = bind_gaussian(x, y)
        target = GaussianHV.from_sample(target_mu)
        return -expected_cosine_similarity(z, target)

    # Initialise at a generic (non-degenerate) random point so the loss
    # is not already at an argmax.
    init_mu = jax.random.normal(k_init, (DIMS,))
    g = jax.grad(loss)(init_mu)
    assert g.shape == (DIMS,)
    assert jnp.all(jnp.isfinite(g))
    assert jnp.linalg.norm(g) > 0.0


def test_grad_flows_through_bundle_gaussian() -> None:
    key = jax.random.PRNGKey(11)
    target_mu = jax.random.normal(key, (DIMS,))

    def loss(mus: jax.Array) -> jax.Array:
        hvs = GaussianHV(
            mu=mus,
            var=jnp.full(mus.shape, 0.01),
            dimensions=DIMS,
        )
        bundled = bundle_gaussian(hvs)
        target = GaussianHV.from_sample(target_mu)
        return -expected_cosine_similarity(bundled, target)

    init_mus = jax.random.normal(jax.random.PRNGKey(12), (5, DIMS))
    g = jax.grad(loss)(init_mus)
    assert g.shape == (5, DIMS)
    assert jnp.all(jnp.isfinite(g))


def test_grad_flows_through_kl_gaussian() -> None:
    """KL regulariser is differentiable — the ingredient variational codebooks need."""

    def kl_loss(mu_p: jax.Array, var_p: jax.Array) -> jax.Array:
        posterior = GaussianHV(mu=mu_p, var=var_p, dimensions=DIMS)
        prior = GaussianHV.create(DIMS)  # standard Normal
        return kl_gaussian(posterior, prior)

    mu = jnp.zeros(DIMS)
    var = jnp.full((DIMS,), 0.5)
    g_mu, g_var = jax.grad(kl_loss, argnums=(0, 1))(mu, var)
    assert g_mu.shape == (DIMS,)
    assert g_var.shape == (DIMS,)
    assert jnp.all(jnp.isfinite(g_mu))
    assert jnp.all(jnp.isfinite(g_var))


def test_variational_training_loop_converges() -> None:
    """Short gradient-descent loop: pull a Gaussian posterior toward a target via cosine sim."""
    k_tgt, k_init = jax.random.split(jax.random.PRNGKey(20))
    target_mu = jax.random.normal(k_tgt, (DIMS,))
    target_mu = target_mu / (jnp.linalg.norm(target_mu) + 1e-8)

    def loss(mu: jax.Array) -> jax.Array:
        post = GaussianHV(mu=mu, var=jnp.full((DIMS,), 0.1), dimensions=DIMS)
        target = GaussianHV.from_sample(target_mu)
        return -expected_cosine_similarity(post, target)

    grad_fn = jax.jit(jax.grad(loss))
    # Random init avoids the zero-gradient degeneracy at μ = 0.
    mu = jax.random.normal(k_init, (DIMS,))
    initial = float(loss(mu))
    for _ in range(100):
        mu = mu - 0.5 * grad_fn(mu)
    final = float(loss(mu))
    assert final < initial
