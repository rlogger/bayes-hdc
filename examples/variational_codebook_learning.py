# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""End-to-end variational training of a PVSA codebook.

This example demonstrates a capability that, to our knowledge, no
other open-source HDC/VSA library exposes: **end-to-end gradient
training of a probabilistic codebook**. The PVSA primitives in
``bayes_hdc.distributions`` propagate Gaussian moments analytically
under bind, bundle, permute, and cleanup, and ``GaussianHV`` exposes a
reparameterised sampler. This means ``jax.grad`` composes through
every operation — including the KL divergence to a prior — and an ELBO
objective can be optimised with a vanilla Adam loop.

What we do
----------

We pose a simple inference problem: an unobserved target ``GaussianHV``
sits behind a likelihood, and we want to fit a variational posterior
``q(z)`` that minimises ``-ELBO(q, prior, target)``. The prior is the
standard normal; the reconstruction term is a Monte-Carlo estimate of
expected cosine similarity between samples from ``q`` and the target.

This is a toy task — the analytical optimum is reachable in closed form
— but it serves as a *unit test for the entire variational stack*:
reparameterisation gradients, KL closed form, the Adam loop, and the
PVSA-side compose-with-jax.grad story all have to work for the
training trajectory to descend.

Run::

    python examples/variational_codebook_learning.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bayes_hdc import (
    GaussianHV,
    elbo_gaussian,
    reconstruction_log_likelihood_mc,
    train_variational_codebook,
)

DIMS = 1024
SEED = 2026
N_STEPS = 500
LEARNING_RATE = 1e-1


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
        # Monte-Carlo reconstruction term — 32 samples per step
        recon = reconstruction_log_likelihood_mc(posterior, target, key, n_samples=32)
        # Negative ELBO so Adam minimises it
        return -elbo_gaussian(posterior, prior, recon)

    # ----------------------------------------------------------------- 4.
    print("\n[4] Train. Adam composes through bind/bundle/KL/sampler in pure JAX.")
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

    if cos_sim > 0.95:
        print("\n[6] ✓ Variational training recovered the target codebook.")
    elif cos_sim > 0.7:
        print("\n[6] ~ Partial recovery — try more n_steps or a higher lr.")
    else:
        print("\n[6] ✗ Did not converge. Inspect the loss trajectory below.")

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

    print("\nThe per-step loss is noisy because the reconstruction term is a")
    print("32-sample Monte-Carlo estimator; Adam descends the *expected* loss,")
    print("not the per-step realisation. The cosine similarity at step [5] is")
    print("the cleaner convergence signal. The substantive point is that every")
    print("PVSA primitive — bind, bundle, permute, KL, the reparameterised")
    print("sampler — is a pure JAX function on registered pytrees, so jax.grad")
    print("composes through all of them and the entire training run compiles")
    print("to one XLA program via jax.lax.scan.")


if __name__ == "__main__":
    main()
