# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Inference utilities for variational PVSA.

This module provides building blocks for training variational
posteriors over PVSA hypervectors. The core primitive is
:func:`elbo_gaussian` — the standard evidence lower bound for a
diagonal-Gaussian latent-variable model, expressed in PVSA terms.

The ELBO is the right loss for learning any variational codebook,
prior, or encoder whose posterior is parameterised as a
:class:`~bayes_hdc.distributions.GaussianHV`. Because every PVSA
operation is a JAX function, :func:`jax.grad` composes through the
ELBO without any extra work — reparameterisation gradients are
"free" in the same sense that they are free for Flax / Equinox models.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from bayes_hdc.distributions import (
    GaussianHV,
    expected_cosine_similarity,
    kl_gaussian,
)


@jax.jit
def elbo_gaussian(
    posterior: GaussianHV,
    prior: GaussianHV,
    reconstruction_log_likelihood: jax.Array,
) -> jax.Array:
    r"""Evidence lower bound for a Gaussian-posterior PVSA model.

    .. math::
        \mathcal{L} = \mathbb{E}_{q}[\log p(x \mid z)] - \mathrm{KL}(q \| p)

    Maximising :math:`\mathcal{L}` with respect to the posterior
    parameters (via :func:`jax.grad` under the reparameterisation
    trick) tightens the variational bound on the log-evidence of the
    observed data.

    Args:
        posterior: Variational posterior ``q(z)`` as a
            :class:`GaussianHV`.
        prior: Prior ``p(z)`` — typically ``GaussianHV.create(d)``
            (standard Normal) or a learned prior.
        reconstruction_log_likelihood: Scalar Monte-Carlo estimate of
            :math:`\mathbb{E}_{q}[\log p(x \mid z)]` computed by the
            caller (usually via ``posterior.sample_batch`` and
            averaging a per-sample log-likelihood).

    Returns:
        Scalar ELBO. Higher is better.
    """
    return reconstruction_log_likelihood - kl_gaussian(posterior, prior)


def reconstruction_log_likelihood_mc(
    posterior: GaussianHV,
    target: GaussianHV,
    key: jax.Array,
    n_samples: int = 16,
) -> jax.Array:
    r"""Monte-Carlo estimate of a reconstruction log-likelihood.

    Uses :class:`GaussianHV`'s built-in reparameterised sampler to
    draw ``n_samples`` hypervectors from ``posterior`` and scores each
    against ``target`` via expected cosine similarity. Intended as a
    convenient default reconstruction term for toy variational models.

    For production use, pass a task-specific log-likelihood in
    :func:`elbo_gaussian` directly.
    """
    samples = posterior.sample_batch(key, n_samples)
    # Score each sample against the target mean via cosine similarity.
    target_norm = target.mu / (jnp.linalg.norm(target.mu) + 1e-8)
    sample_norms = samples / (jnp.linalg.norm(samples, axis=-1, keepdims=True) + 1e-8)
    # Use expected_cosine_similarity-style average over samples.
    _ = expected_cosine_similarity  # kept in scope for re-export clarity
    sims = sample_norms @ target_norm
    return jnp.mean(sims)


__all__ = [
    "elbo_gaussian",
    "reconstruction_log_likelihood_mc",
]
