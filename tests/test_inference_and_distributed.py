# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Tests for bayes_hdc.inference (ELBO) and bayes_hdc.distributed (pmap wrappers)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from bayes_hdc.distributed import (
    batch_bind_gaussian,
    batch_similarity_gaussian,
    pmap_bind_gaussian,
    pmap_bundle_gaussian,
)
from bayes_hdc.distributions import GaussianHV, kl_gaussian
from bayes_hdc.inference import (
    elbo_gaussian,
    gaussian_reconstruction_log_likelihood_mc,
    reconstruction_log_likelihood_mc,  # back-compat alias
    reconstruction_score_mc,
)

DIMS = 64


# ----------------------------------------------------------------------
# elbo_gaussian
# ----------------------------------------------------------------------


def test_elbo_returns_scalar() -> None:
    prior = GaussianHV.create(DIMS)
    post = GaussianHV.random(jax.random.PRNGKey(0), DIMS, var=0.5)
    log_p = jnp.array(0.1)
    elbo = elbo_gaussian(post, prior, log_p)
    assert elbo.shape == ()


def test_elbo_equals_loglik_when_kl_is_zero() -> None:
    prior = GaussianHV.create(DIMS)
    post = GaussianHV.create(DIMS)  # same as prior → KL = 0
    log_p = jnp.array(3.0)
    elbo = elbo_gaussian(post, prior, log_p)
    assert jnp.isclose(elbo, log_p, atol=1e-4)


def test_elbo_decreases_with_larger_kl() -> None:
    prior = GaussianHV.create(DIMS)
    close_post = GaussianHV.random(jax.random.PRNGKey(1), DIMS, var=0.9)
    far_post = GaussianHV.random(
        jax.random.PRNGKey(2),
        DIMS,
        var=0.01,
    )  # sharp posterior → large KL
    log_p = jnp.array(0.0)
    elbo_close = float(elbo_gaussian(close_post, prior, log_p))
    elbo_far = float(elbo_gaussian(far_post, prior, log_p))
    # Close posterior has smaller KL → higher ELBO.
    assert elbo_close > elbo_far


def test_elbo_is_differentiable() -> None:
    prior = GaussianHV.create(DIMS)

    def loss(mu: jax.Array, log_var: jax.Array) -> jax.Array:
        post = GaussianHV(mu=mu, var=jnp.exp(log_var), dimensions=DIMS)
        return -elbo_gaussian(post, prior, jnp.array(0.0))

    g_mu, g_logvar = jax.grad(loss, argnums=(0, 1))(
        jnp.zeros(DIMS),
        jnp.log(jnp.full((DIMS,), 0.5)),
    )
    assert jnp.all(jnp.isfinite(g_mu))
    assert jnp.all(jnp.isfinite(g_logvar))


def test_reconstruction_score_mc_shape_and_bounds() -> None:
    """The similarity proxy is a scalar in [-1, 1]."""
    post = GaussianHV.random(jax.random.PRNGKey(10), DIMS, var=0.1)
    target = GaussianHV.random(jax.random.PRNGKey(11), DIMS, var=0.0)
    score = reconstruction_score_mc(post, target, jax.random.PRNGKey(12))
    assert score.shape == ()
    assert jnp.isfinite(score)
    assert -1.0 - 1e-6 <= float(score) <= 1.0 + 1e-6


def test_reconstruction_score_mc_respects_n_samples() -> None:
    post = GaussianHV.random(jax.random.PRNGKey(20), DIMS, var=0.1)
    target = GaussianHV.random(jax.random.PRNGKey(21), DIMS, var=0.0)
    score_small = reconstruction_score_mc(post, target, jax.random.PRNGKey(22), n_samples=4)
    score_large = reconstruction_score_mc(post, target, jax.random.PRNGKey(22), n_samples=128)
    assert jnp.isfinite(score_small)
    assert jnp.isfinite(score_large)


def test_legacy_alias_reconstruction_log_likelihood_mc() -> None:
    """The deprecated alias must still resolve to the same function."""
    assert reconstruction_log_likelihood_mc is reconstruction_score_mc


def test_gaussian_recon_log_likelihood_is_in_nats_scale() -> None:
    """The Gaussian log-density is on a scale comparable to KL — both negative,
    both d-extensive — unlike the cosine proxy which is bounded in [-1, 1]."""
    post = GaussianHV.random(jax.random.PRNGKey(30), DIMS, var=0.5)
    target = GaussianHV.random(jax.random.PRNGKey(31), DIMS, var=0.0)
    ll = gaussian_reconstruction_log_likelihood_mc(
        post, target, jax.random.PRNGKey(32), n_samples=64, observation_noise=1.0
    )
    # For d=64 random unit-ish vectors with σ_obs=1, |ll| ~ O(d). Sanity: > 1.
    assert jnp.isfinite(ll)
    assert abs(float(ll)) > 1.0


def test_gaussian_recon_log_likelihood_max_at_target() -> None:
    """Centring the posterior at the target should *raise* the log-density."""
    target = GaussianHV.random(jax.random.PRNGKey(40), DIMS, var=0.0)
    near = GaussianHV(
        mu=target.mu,
        var=jnp.full((DIMS,), 1e-3),
        dimensions=DIMS,
    )
    far = GaussianHV(
        mu=jnp.zeros(DIMS),
        var=jnp.full((DIMS,), 1e-3),
        dimensions=DIMS,
    )
    ll_near = float(
        gaussian_reconstruction_log_likelihood_mc(
            near, target, jax.random.PRNGKey(41), n_samples=64, observation_noise=0.5
        )
    )
    ll_far = float(
        gaussian_reconstruction_log_likelihood_mc(
            far, target, jax.random.PRNGKey(41), n_samples=64, observation_noise=0.5
        )
    )
    assert ll_near > ll_far


def test_elbo_gradient_descent_reduces_kl() -> None:
    """A toy variational loop — optimising the posterior toward the prior reduces KL."""
    prior = GaussianHV.create(DIMS)

    def kl_only(mu: jax.Array, log_var: jax.Array) -> jax.Array:
        post = GaussianHV(mu=mu, var=jnp.exp(log_var), dimensions=DIMS)
        return kl_gaussian(post, prior)

    grad_fn = jax.jit(jax.grad(kl_only, argnums=(0, 1)))
    mu = jnp.ones(DIMS) * 2.0
    log_var = jnp.log(jnp.full((DIMS,), 3.0))
    kl_0 = float(kl_only(mu, log_var))
    for _ in range(50):
        g_mu, g_lv = grad_fn(mu, log_var)
        mu = mu - 0.1 * g_mu
        log_var = log_var - 0.1 * g_lv
    kl_final = float(kl_only(mu, log_var))
    assert kl_final < kl_0


# ----------------------------------------------------------------------
# distributed.batch_* (vmap wrappers)
# ----------------------------------------------------------------------


def test_batch_bind_gaussian_produces_batched_hv() -> None:
    key = jax.random.PRNGKey(100)
    k1, k2 = jax.random.split(key)
    mus_x = jax.random.normal(k1, (8, DIMS))
    vars_x = jnp.full((8, DIMS), 0.1)
    mus_y = jax.random.normal(k2, (8, DIMS))
    vars_y = jnp.full((8, DIMS), 0.1)

    x = GaussianHV(mu=mus_x, var=vars_x, dimensions=DIMS)
    y = GaussianHV(mu=mus_y, var=vars_y, dimensions=DIMS)

    z = batch_bind_gaussian(x, y)
    assert z.mu.shape == (8, DIMS)
    assert z.var.shape == (8, DIMS)


def test_batch_similarity_gaussian() -> None:
    key = jax.random.PRNGKey(101)
    k1, k2 = jax.random.split(key)
    queries_mu = jax.random.normal(k1, (5, DIMS))
    queries_mu = queries_mu / (jnp.linalg.norm(queries_mu, axis=-1, keepdims=True) + 1e-8)
    target_mu = jax.random.normal(k2, (DIMS,))
    target_mu = target_mu / (jnp.linalg.norm(target_mu) + 1e-8)

    queries = GaussianHV(
        mu=queries_mu,
        var=jnp.full((5, DIMS), 0.01),
        dimensions=DIMS,
    )
    target = GaussianHV.from_sample(target_mu)
    sims = batch_similarity_gaussian(queries, target)
    assert sims.shape == (5,)
    assert jnp.all(sims >= -1.0) and jnp.all(sims <= 1.0)


# ----------------------------------------------------------------------
# distributed.pmap_* (single-device fallback)
# ----------------------------------------------------------------------


def test_pmap_bind_gaussian_single_device() -> None:
    """pmap degrades to single-device execution on a host with one device."""
    n_devices = jax.local_device_count()
    key = jax.random.PRNGKey(200)
    k1, k2 = jax.random.split(key)
    mus_x = jax.random.normal(k1, (n_devices, DIMS))
    mus_y = jax.random.normal(k2, (n_devices, DIMS))
    x = GaussianHV(
        mu=mus_x,
        var=jnp.full((n_devices, DIMS), 0.1),
        dimensions=DIMS,
    )
    y = GaussianHV(
        mu=mus_y,
        var=jnp.full((n_devices, DIMS), 0.1),
        dimensions=DIMS,
    )
    z = pmap_bind_gaussian(x, y)
    assert z.mu.shape == (n_devices, DIMS)


def test_pmap_bundle_gaussian_single_device() -> None:
    n_devices = jax.local_device_count()
    key = jax.random.PRNGKey(201)
    batch_per_device = 4
    mus = jax.random.normal(key, (n_devices, batch_per_device, DIMS))
    hvs = GaussianHV(
        mu=mus,
        var=jnp.full(mus.shape, 0.1),
        dimensions=DIMS,
    )
    out = pmap_bundle_gaussian(hvs)
    assert out.mu.shape == (DIMS,)
    assert out.var.shape == (DIMS,)


def test_pmap_bundle_gaussian_matches_global_bundle() -> None:
    """Sharded bundle is algebraically identical to bundling the un-sharded batch.

    Regression test for a subtle bug: composing ``bundle_gaussian`` twice
    (per-device then host) double-normalises, which is not the same
    operation as a single global bundle. The fix accumulates partial
    sums via ``pmap`` and normalises exactly once on the host.
    """
    from bayes_hdc.distributions import bundle_gaussian

    n_devices = jax.local_device_count()
    batch_per_device = 4
    total = n_devices * batch_per_device
    key = jax.random.PRNGKey(202)
    mus = jax.random.normal(key, (total, DIMS))
    vars_ = jnp.full((total, DIMS), 0.1)

    flat = GaussianHV(mu=mus, var=vars_, dimensions=DIMS)
    sharded = GaussianHV(
        mu=mus.reshape(n_devices, batch_per_device, DIMS),
        var=vars_.reshape(n_devices, batch_per_device, DIMS),
        dimensions=DIMS,
    )

    direct = bundle_gaussian(flat)
    via_pmap = pmap_bundle_gaussian(sharded)

    # Algebraic equivalence within float32 precision.
    assert jnp.allclose(via_pmap.mu, direct.mu, atol=1e-5)
    assert jnp.allclose(via_pmap.var, direct.var, atol=1e-5)
