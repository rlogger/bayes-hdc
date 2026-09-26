# SPDX-License-Identifier: MIT
# Copyright (c) 2026 R.S.

"""Variational inference for one Gaussian latent vector with JAX gradients.

A standard-normal prior and observation likelihood x|z ~ N(z,sigma**2 I)
produce an analytic Gaussian posterior. The demo uses a Monte Carlo ELBO to
fit its mean and diagonal variance and reports distance to that reference.
Only target.mu is the observation; target.var is not used by the likelihood.
The loss exercises the reparameterised sampler, Gaussian likelihood and KL;
it does not train through bind, bundle, permute or cleanup, nor establish a
unique or state-of-the-art codebook-learning method.

Run: python examples/variational_codebook_learning.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bayes_hdc import (
    GaussianHV,
    elbo_gaussian,
    gaussian_reconstruction_log_likelihood_mc,
    train_variational_codebook,
)

DIMS = 1024
SEED = 2026
N_STEPS = 1500
LEARNING_RATE = 5e-2
# Standard deviation of the Gaussian observation model. Smaller σ → tighter
# likelihood, stronger reconstruction pressure relative to KL. 0.05 gives
# clean recovery on this toy task; tune for your dataset.
OBSERVATION_NOISE = 0.05


def main() -> None:
    print("Variational codebook learning — end-to-end gradient training")
    print(f"  dimensions = {DIMS}    n_steps = {N_STEPS}    lr = {LEARNING_RATE}\n")

    # ----------------------------------------------------------------- 1.
    print("[1] Build a target GaussianHV (the 'truth' we want to recover).")
    key = jax.random.PRNGKey(SEED)
    key, sub = jax.random.split(key)
    target_mu = jax.random.normal(sub, (DIMS,))
    target_mu = target_mu / jnp.linalg.norm(target_mu)
    target = GaussianHV(mu=target_mu, var=jnp.full((DIMS,), 0.001), dimensions=DIMS)
    print(f"      target μ-norm = {float(jnp.linalg.norm(target.mu)):.4f}")
    print(f"      target var    = {float(target.var[0]):.4f}")

    # ----------------------------------------------------------------- 2.
    print("\n[2] Initialise a variational posterior far from the target.")
    init_params = {
        "mu": jnp.zeros(DIMS),
        "log_var": jnp.zeros(DIMS),  # log var = 0 → var = 1
    }
    init_post = GaussianHV(
        mu=init_params["mu"],
        var=jnp.exp(init_params["log_var"]),
        dimensions=DIMS,
    )
    print(f"      initial ‖μ‖ = {float(jnp.linalg.norm(init_post.mu)):.4f}")
    print(f"      initial var = {float(init_post.var[0]):.4f}")

    # ----------------------------------------------------------------- 3.
    print("\n[3] Define -ELBO(q, prior, target) as a JAX-differentiable loss.")
    prior = GaussianHV.create(DIMS)

    def loss_fn(params: dict, key: jax.Array) -> jax.Array:
        posterior = GaussianHV(
            mu=params["mu"],
            var=jnp.exp(params["log_var"]),
            dimensions=DIMS,
        )
        # True Gaussian-observation log-likelihood (in nats), so the
        # ``recon - KL`` combination below is a dimensionally consistent
        # ELBO. OBSERVATION_NOISE sets the reconstruction weight
        # relative to KL; tune for your task.
        recon = gaussian_reconstruction_log_likelihood_mc(
            posterior, target, key, n_samples=32, observation_noise=OBSERVATION_NOISE
        )
        # Negative ELBO so Adam minimises it.
        return -elbo_gaussian(posterior, prior, recon)

    # ----------------------------------------------------------------- 4.
    print("\n[4] Train the mean and variance with the sampler, likelihood and KL.")
    train_key = jax.random.fold_in(key, 1)
    result = train_variational_codebook(
        init_params=init_params,
        loss_fn=loss_fn,
        key=train_key,
        n_steps=N_STEPS,
        learning_rate=LEARNING_RATE,
    )
    initial_loss = float(result.loss_history[0])
    final_loss = result.final_loss
    print(f"      initial loss = {initial_loss:+.4f}")
    print(f"      final   loss = {final_loss:+.4f}")
    print(f"      reduction    = {initial_loss - final_loss:+.4f}")

    # ----------------------------------------------------------------- 5.
    print("\n[5] How close did the fitted posterior get to the target?")
    fitted = GaussianHV(
        mu=result.params["mu"],
        var=jnp.exp(result.params["log_var"]),
        dimensions=DIMS,
    )
    cos_sim = float(
        (fitted.mu @ target.mu) / (jnp.linalg.norm(fitted.mu) * jnp.linalg.norm(target.mu) + 1e-8)
    )
    print(f"      cos(μ_fitted, μ_target) = {cos_sim:.4f}    (1.0 is exact)")
    print(f"      mean fitted variance     = {float(jnp.mean(fitted.var)):.4f}")

    sigma2 = OBSERVATION_NOISE**2
    exact_mu = target.mu / (1.0 + sigma2)
    exact_var = jnp.full((DIMS,), sigma2 / (1.0 + sigma2))
    mu_rmse = float(jnp.sqrt(jnp.mean((fitted.mu - exact_mu) ** 2)))
    var_rmse = float(jnp.sqrt(jnp.mean((fitted.var - exact_var) ** 2)))
    print(f"      mean RMSE to analytic posterior: {mu_rmse:.6f}")
    print(f"      variance RMSE to analytic posterior: {var_rmse:.6f}")

    # ----------------------------------------------------------------- 7.
    print("\n[7] Loss trajectory (key steps):")
    history = np.asarray(result.loss_history)
    width = 30
    floor = float(history.min())
    ceil = float(history.max())
    span = max(ceil - floor, 1e-6)
    indices = sorted(set(list(range(0, N_STEPS, max(1, N_STEPS // 10))) + [N_STEPS - 1]))
    for i in indices:
        v = float(history[i])
        bar_len = max(0, int(width * (ceil - v) / span))
        print(f"  step {i:>4d}: loss = {v:+.4f}  {'█' * bar_len}")

    print("\nMonte Carlo loss values are noisy; a decrease alone does not prove convergence.")
    print("The analytic posterior errors above assess both mean and variance.")
    print("This example does not exercise the other differentiable VSA operations.")


if __name__ == "__main__":
    main()
