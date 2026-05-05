# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Inference utilities for variational PVSA.

This module provides building blocks for training variational
posteriors over PVSA hypervectors. Two reconstruction terms are
exposed:

- :func:`gaussian_reconstruction_log_likelihood_mc` — a *true*
  Monte-Carlo log-density under an isotropic Gaussian observation
  model :math:`p(x \\mid z) = \\mathcal{N}(x \\mid z, \\sigma_{\\mathrm{obs}}^2 I)`.
  Use this when you want :func:`elbo_gaussian` to return a real
  evidence lower bound (in nats).

- :func:`reconstruction_score_mc` — a similarity-based proxy that
  averages cosine similarity between samples from the posterior and
  the target. **This is not a log-density**; it is bounded in
  ``[-1, 1]`` and lacks the normalisation constants of a true
  likelihood. Plugging it into :func:`elbo_gaussian` gives a
  similarity-regularised KL objective that is useful for codebook
  alignment but is not a tight bound on ``log p(data)``.

The legacy alias ``reconstruction_log_likelihood_mc`` is preserved for
backwards compatibility but is **deprecated** — its name is misleading
because it does not return a log-density. Prefer
:func:`reconstruction_score_mc` (when you want the similarity proxy)
or :func:`gaussian_reconstruction_log_likelihood_mc` (when you want a
proper ELBO). All three functions are JAX-pure;
:func:`jax.grad` composes through them via the reparameterisation
trick that ``GaussianHV.sample_batch`` already implements.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from bayes_hdc.distributions import (
    GaussianHV,
    kl_gaussian,
)


@jax.jit
def elbo_gaussian(
    posterior: GaussianHV,
    prior: GaussianHV,
    reconstruction_term: jax.Array,
) -> jax.Array:
    r"""Evidence lower bound for a Gaussian-posterior PVSA model.

    .. math::
        \mathcal{L} = \mathbb{E}_{q}[\log p(x \mid z)] - \mathrm{KL}(q \| p)

    Maximising :math:`\mathcal{L}` with respect to the posterior
    parameters (via :func:`jax.grad` under the reparameterisation
    trick) tightens the variational bound on the log-evidence of the
    observed data — *provided* the reconstruction term is a real
    log-density. If the caller passes a similarity proxy (e.g.
    :func:`reconstruction_score_mc`), the returned scalar is a
    similarity-regularised KL objective that is useful for codebook
    alignment but is **not** a tight evidence lower bound.

    Args:
        posterior: Variational posterior ``q(z)`` as a
            :class:`GaussianHV`.
        prior: Prior ``p(z)`` — typically ``GaussianHV.create(d)``
            (standard Normal) or a learned prior.
        reconstruction_term: Scalar Monte-Carlo estimate of
            :math:`\mathbb{E}_{q}[\log p(x \mid z)]` (true ELBO) or a
            similarity proxy (see module docstring). Use
            :func:`gaussian_reconstruction_log_likelihood_mc` for the
            former, :func:`reconstruction_score_mc` for the latter.

    Returns:
        Scalar objective. Higher is better. Whether this is a true
        ELBO depends on the reconstruction term passed in.
    """
    return reconstruction_term - kl_gaussian(posterior, prior)


def gaussian_reconstruction_log_likelihood_mc(
    posterior: GaussianHV,
    target: GaussianHV,
    key: jax.Array,
    n_samples: int = 16,
    observation_noise: float = 1.0,
) -> jax.Array:
    r"""Monte-Carlo estimate of :math:`\mathbb{E}_{q}[\log p(x \mid z)]`.

    Uses an isotropic Gaussian observation model
    :math:`p(x \mid z) = \mathcal{N}(x \mid z, \sigma_{\mathrm{obs}}^{2} I)` —
    the standard reconstruction term in a Gaussian VAE. For a sample
    :math:`z \sim q` and target mean :math:`x = \mu_{\mathrm{target}}`,

    .. math::
        \log p(x \mid z) = -\tfrac{1}{2 \sigma_{\mathrm{obs}}^{2}}
            \lVert x - z \rVert^{2}
            - \tfrac{d}{2} \log(2 \pi \sigma_{\mathrm{obs}}^{2}).

    The MC estimate averages this over ``n_samples`` reparameterised
    draws from ``posterior``. The return value is in **nats**, on the
    same scale as :func:`kl_gaussian`, so plugging it into
    :func:`elbo_gaussian` gives a dimensionally consistent ELBO.

    Args:
        posterior: Variational posterior to sample from.
        target: Target mean (only the ``mu`` field is used; the
            observation noise lives in ``observation_noise``).
        key: A ``jax.random.PRNGKey``.
        n_samples: Number of MC draws.
        observation_noise: Standard deviation of the isotropic
            Gaussian observation model. Controls the reconstruction
            weight relative to KL.

    Returns:
        Scalar log-likelihood estimate (nats).
    """
    d = posterior.dimensions
    sigma2 = observation_noise**2
    samples = posterior.sample_batch(key, n_samples)  # (n, d)
    diffs = samples - target.mu[None, :]
    sq_dist = jnp.sum(diffs * diffs, axis=-1)  # (n,)
    log_lik_per_sample = -0.5 * sq_dist / sigma2 - 0.5 * d * math.log(2.0 * math.pi * sigma2)
    return jnp.mean(log_lik_per_sample)


def reconstruction_score_mc(
    posterior: GaussianHV,
    target: GaussianHV,
    key: jax.Array,
    n_samples: int = 16,
) -> jax.Array:
    r"""Mean cosine similarity between posterior samples and the target.

    A **similarity proxy** in :math:`[-1, 1]`, *not* a log-density.
    Convenient for codebook-alignment objectives where the goal is "make
    samples from ``q`` look like the target direction" but the absolute
    scale of the loss does not need to match a probability.

    For a true variational lower bound, use
    :func:`gaussian_reconstruction_log_likelihood_mc` instead. The
    output of this function combined with :func:`elbo_gaussian` yields
    a similarity-regularised KL objective, useful empirically but not a
    tight bound on ``log p(data)``.

    Args:
        posterior: Variational posterior to sample from.
        target: Target hypervector (only its ``mu`` direction is used).
        key: A ``jax.random.PRNGKey``.
        n_samples: Number of MC draws.

    Returns:
        Scalar mean cosine similarity, bounded in ``[-1, 1]``.
    """
    samples = posterior.sample_batch(key, n_samples)
    target_norm = target.mu / (jnp.linalg.norm(target.mu) + 1e-8)
    sample_norms = samples / (jnp.linalg.norm(samples, axis=-1, keepdims=True) + 1e-8)
    sims = sample_norms @ target_norm
    return jnp.mean(sims)


# Deprecated alias — preserved for back-compat. Prefer
# ``reconstruction_score_mc`` (similarity proxy) or
# ``gaussian_reconstruction_log_likelihood_mc`` (true log-density).
reconstruction_log_likelihood_mc = reconstruction_score_mc


__all__ = [
    "elbo_gaussian",
    "gaussian_reconstruction_log_likelihood_mc",
    "reconstruction_score_mc",
    # Deprecated; do not use in new code.
    "reconstruction_log_likelihood_mc",
]
