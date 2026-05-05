# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Tests for bayes_hdc.training — Adam optimiser and the variational
codebook trainer."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from bayes_hdc.distributions import GaussianHV
from bayes_hdc.inference import (
    elbo_gaussian,
    gaussian_reconstruction_log_likelihood_mc,
)
from bayes_hdc.training import (
    AdamState,
    TrainResult,
    adam_init,
    adam_update,
    train_variational_codebook,
)

DIMS = 64


# ----------------------------------------------------------------------
# Adam state and update
# ----------------------------------------------------------------------


def test_adam_init_zeros_match_param_shape() -> None:
    params = {"a": jnp.ones((4,)), "b": jnp.ones((2, 3))}
    state = adam_init(params)
    assert isinstance(state, AdamState)
    assert state.m["a"].shape == (4,)
    assert state.m["b"].shape == (2, 3)
    assert state.v["a"].shape == (4,)
    assert state.v["b"].shape == (2, 3)
    assert int(state.step) == 0
    assert jnp.all(state.m["a"] == 0)
    assert jnp.all(state.v["b"] == 0)


def test_adam_step_advances_counter() -> None:
    params = jnp.zeros((3,))
    state = adam_init(params)
    grads = jnp.ones((3,))
    new_state, _ = adam_update(state, grads, learning_rate=1e-2)
    assert int(new_state.step) == 1


def test_adam_descends_quadratic() -> None:
    """A unit Adam loop on f(x) = ||x||^2 should drive x toward 0."""
    x = jnp.ones((5,))
    state = adam_init(x)
    for _ in range(200):
        grads = 2.0 * x  # df/dx of ||x||^2
        state, update = adam_update(state, grads, learning_rate=5e-2)
        x = x + update
    assert float(jnp.linalg.norm(x)) < 0.1


def test_adam_pytree_structure_preserved() -> None:
    """Adam state and update must preserve the parameter pytree shape."""
    params = {"mu": jnp.zeros((4,)), "log_var": jnp.zeros((4,))}
    state = adam_init(params)
    grads = jax.tree.map(jnp.ones_like, params)
    _, update = adam_update(state, grads)
    assert isinstance(update, dict)
    assert set(update.keys()) == {"mu", "log_var"}
    assert update["mu"].shape == (4,)


# ----------------------------------------------------------------------
# train_variational_codebook
# ----------------------------------------------------------------------


def test_train_returns_result_dataclass() -> None:
    init = jnp.zeros((3,))

    def loss_fn(params, key):
        return jnp.sum(params**2)

    out = train_variational_codebook(
        init_params=init,
        loss_fn=loss_fn,
        key=jax.random.PRNGKey(0),
        n_steps=10,
        learning_rate=1e-2,
    )
    assert isinstance(out, TrainResult)
    assert out.loss_history.shape == (10,)
    assert out.params.shape == (3,)


def test_train_descends_simple_quadratic() -> None:
    """Training f(x) = ||x - 1||^2 from x=0 should pull x toward 1."""
    init = jnp.zeros((4,))

    def loss_fn(params, key):
        return jnp.sum((params - 1.0) ** 2)

    out = train_variational_codebook(
        init_params=init,
        loss_fn=loss_fn,
        key=jax.random.PRNGKey(0),
        n_steps=300,
        learning_rate=1e-1,
    )
    # final params should be close to 1.0
    assert float(jnp.mean(jnp.abs(out.params - 1.0))) < 0.1
    # loss should have descended substantially
    assert float(out.loss_history[-1]) < float(out.loss_history[0])


def test_train_composes_through_pvsa_elbo() -> None:
    """End-to-end: train a GaussianHV posterior to recover a target via -ELBO."""
    key = jax.random.PRNGKey(2026)
    target_mu = jax.random.normal(key, (DIMS,))
    target_mu = target_mu / jnp.linalg.norm(target_mu)
    target = GaussianHV(mu=target_mu, var=jnp.full((DIMS,), 0.001), dimensions=DIMS)
    prior = GaussianHV.create(DIMS)

    def loss_fn(params, key):
        post = GaussianHV(
            mu=params["mu"],
            var=jnp.exp(params["log_var"]),
            dimensions=DIMS,
        )
        # Real Gaussian log-density (in nats) so the ELBO is dimensionally
        # consistent with kl_gaussian. Tight σ_obs = strong reconstruction
        # pressure relative to KL, which is needed to recover the target on
        # this small (d=64) problem.
        recon = gaussian_reconstruction_log_likelihood_mc(
            post, target, key, n_samples=16, observation_noise=0.05
        )
        return -elbo_gaussian(post, prior, recon)

    init = {"mu": jnp.zeros(DIMS), "log_var": jnp.zeros(DIMS)}
    out = train_variational_codebook(
        init_params=init,
        loss_fn=loss_fn,
        key=jax.random.fold_in(key, 1),
        n_steps=400,
        learning_rate=5e-2,
    )
    fitted_mu = out.params["mu"]
    cos = float(
        (fitted_mu @ target.mu) / (jnp.linalg.norm(fitted_mu) * jnp.linalg.norm(target.mu) + 1e-8)
    )
    # Should recover the target direction with high cosine similarity.
    assert cos > 0.9, f"cos={cos:.4f} — variational training did not recover target"


def test_train_jit_compiles_full_loop() -> None:
    """The trainer must be wrappable in jax.jit (single XLA program)."""
    init = jnp.ones((2,))

    def loss_fn(params, key):
        return jnp.sum(params**2)

    @jax.jit
    def run(init):
        return train_variational_codebook(
            init_params=init,
            loss_fn=loss_fn,
            key=jax.random.PRNGKey(0),
            n_steps=5,
            learning_rate=1e-2,
        )

    out = run(init)
    assert isinstance(out, TrainResult)
    assert out.loss_history.shape == (5,)
