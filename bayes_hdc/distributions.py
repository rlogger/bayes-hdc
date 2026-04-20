# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Bayesian hypervectors — distributions over hypervectors.

A Bayesian hypervector represents uncertainty explicitly. Rather than a
single point in :math:`\\mathbb{R}^d`, it carries a probability distribution.
Binding and bundling propagate the distribution forward through the VSA
algebra so that similarity, retrieval, and classification inherit
calibrated uncertainty end-to-end.

This module ships the foundational distribution types. The API mirrors the
deterministic :mod:`bayes_hdc.vsa` models: every distributional type exposes
``bind``, ``bundle``, ``similarity``, and ``sample``. Deterministic
hypervectors are recovered as the zero-variance limit, so any existing
pipeline composes cleanly.

Distributions currently provided:

- :class:`GaussianHV` — mean + diagonal variance; closed-form moment
  propagation under element-wise multiplication (MAP binding) and
  addition (bundling).

Planned (see ``README.md`` roadmap):

- ``DirichletHV`` — distributions over the probability simplex, for
  probabilistic categorical codebooks.
- ``MixtureHV`` — mixture-of-Gaussian hypervectors for multi-modal
  representations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from bayes_hdc._compat import register_dataclass
from bayes_hdc.constants import EPS


@register_dataclass
@dataclass
class GaussianHV:
    """A hypervector distributed as :math:`\\mathcal{N}(\\mu, \\mathrm{diag}(\\sigma^2))`.

    Diagonal covariance keeps binding and bundling in closed form and
    preserves the :math:`O(d)` memory footprint of deterministic MAP
    hypervectors. For richer posteriors use a mixture or a low-rank
    parameterisation (both roadmap).

    Attributes:
        mu: Mean of shape ``(d,)``.
        var: Element-wise variance of shape ``(d,)``; must be non-negative.
        dimensions: Dimensionality :math:`d`.
    """

    mu: jax.Array
    var: jax.Array
    dimensions: int = field(metadata=dict(static=True))

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @staticmethod
    def create(
        dimensions: int,
        mu: jax.Array | None = None,
        var: jax.Array | None = None,
    ) -> GaussianHV:
        """Construct a Gaussian hypervector.

        Defaults to a standard-normal prior :math:`\\mathcal{N}(0, I)`.
        """
        mu_arr = jnp.zeros(dimensions) if mu is None else mu
        var_arr = jnp.ones(dimensions) if var is None else var
        return GaussianHV(mu=mu_arr, var=var_arr, dimensions=dimensions)

    @staticmethod
    def from_sample(sample: jax.Array, var: float = 0.0) -> GaussianHV:
        """Lift a deterministic hypervector to a Gaussian with chosen variance.

        ``var=0`` gives a Dirac; ``var>0`` gives an isotropic posterior
        centred on the sample. Useful for wrapping existing pipelines in
        uncertainty without retraining.
        """
        d = sample.shape[-1]
        return GaussianHV(
            mu=sample,
            var=jnp.full((d,), float(var)),
            dimensions=d,
        )

    @staticmethod
    def random(
        key: jax.Array,
        dimensions: int,
        var: float = 1.0,
    ) -> GaussianHV:
        """Sample a random unit-norm mean with isotropic variance ``var``."""
        mu = jax.random.normal(key, (dimensions,))
        mu = mu / (jnp.linalg.norm(mu) + EPS)
        return GaussianHV(
            mu=mu,
            var=jnp.full((dimensions,), float(var)),
            dimensions=dimensions,
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, key: jax.Array) -> jax.Array:
        """Draw a single sample from this distribution, shape ``(d,)``."""
        eps = jax.random.normal(key, self.mu.shape)
        return self.mu + jnp.sqrt(jnp.maximum(self.var, 0.0)) * eps

    def sample_batch(self, key: jax.Array, n: int) -> jax.Array:
        """Draw ``n`` samples from this distribution, shape ``(n, d)``."""
        eps = jax.random.normal(key, (n, self.dimensions))
        std = jnp.sqrt(jnp.maximum(self.var, 0.0))
        return self.mu[None, :] + std[None, :] * eps


# ----------------------------------------------------------------------
# Distributional binding and bundling
# ----------------------------------------------------------------------


@jax.jit
def bind_gaussian(x: GaussianHV, y: GaussianHV) -> GaussianHV:
    """Bind two independent Gaussian HVs under element-wise multiplication.

    For independent :math:`X \\sim \\mathcal{N}(\\mu_x, \\sigma_x^2)` and
    :math:`Y \\sim \\mathcal{N}(\\mu_y, \\sigma_y^2)`, the exact first two
    moments of ``Z = X * Y`` are:

    .. math::
        \\mathbb{E}[Z] &= \\mu_x \\mu_y \\\\
        \\mathrm{Var}[Z] &= \\mu_x^2 \\sigma_y^2 + \\mu_y^2 \\sigma_x^2 + \\sigma_x^2 \\sigma_y^2

    This is the standard moment calculation for products of independent
    Gaussians and is applied element-wise here. The result is *not*
    itself exactly Gaussian, but ``GaussianHV`` takes a moment-matched
    view.

    Args:
        x: First Gaussian hypervector.
        y: Second Gaussian hypervector.

    Returns:
        Moment-matched Gaussian hypervector for the element-wise product.
    """
    new_mu = x.mu * y.mu
    new_var = (x.mu**2) * y.var + (y.mu**2) * x.var + x.var * y.var
    return GaussianHV(mu=new_mu, var=new_var, dimensions=x.dimensions)


@jax.jit
def bundle_gaussian(hvs: GaussianHV) -> GaussianHV:
    """Bundle a batch of independent Gaussian HVs by summation.

    If each ``hvs[i]`` is independent, the sum is exactly Gaussian with

    .. math::
        \\mathbb{E}\\!\\left[\\sum_i X_i\\right] &= \\sum_i \\mu_i \\\\
        \\mathrm{Var}\\!\\left[\\sum_i X_i\\right] &= \\sum_i \\sigma_i^2

    The result is then normalised (mean divided by its L2 norm, variance
    scaled by the squared inverse norm) to stay on the unit sphere, which
    matches deterministic MAP bundling.

    Args:
        hvs: A batched :class:`GaussianHV` with ``mu`` of shape ``(n, d)``
             and ``var`` of shape ``(n, d)``.

    Returns:
        Normalised Gaussian hypervector of dimension ``d``.
    """
    summed_mu = jnp.sum(hvs.mu, axis=0)
    summed_var = jnp.sum(hvs.var, axis=0)
    norm = jnp.linalg.norm(summed_mu) + EPS
    return GaussianHV(
        mu=summed_mu / norm,
        var=summed_var / (norm**2),
        dimensions=summed_mu.shape[-1],
    )


# ----------------------------------------------------------------------
# Expected similarity
# ----------------------------------------------------------------------


@jax.jit
def expected_cosine_similarity(x: GaussianHV, y: GaussianHV) -> jax.Array:
    """Plug-in estimator of the expected cosine similarity under ``p(x) p(y)``.

    Approximates :math:`\\mathbb{E}\\![\\langle X, Y \\rangle / (\\|X\\| \\|Y\\|)]`
    by evaluating cosine similarity at the means. This is exact in the
    zero-variance limit and is within :math:`O(\\sigma^2 / \\|\\mu\\|^2)` of
    the true expectation for small variance.

    For an unbiased Monte Carlo estimate, use :meth:`GaussianHV.sample_batch`
    and average the resulting cosine similarities.

    Args:
        x: First Gaussian hypervector.
        y: Second Gaussian hypervector.

    Returns:
        Scalar in ``[-1, 1]``.
    """
    x_norm = x.mu / (jnp.linalg.norm(x.mu) + EPS)
    y_norm = y.mu / (jnp.linalg.norm(y.mu) + EPS)
    return jnp.clip(jnp.sum(x_norm * y_norm), -1.0, 1.0)


@jax.jit
def similarity_variance(x: GaussianHV, y: GaussianHV) -> jax.Array:
    """First-order variance of the dot product :math:`\\langle X, Y \\rangle`.

    Under independence, :math:`\\mathrm{Var}[\\sum_i X_i Y_i]
    = \\sum_i (\\mu_{x,i}^2 \\sigma_{y,i}^2 + \\mu_{y,i}^2 \\sigma_{x,i}^2
    + \\sigma_{x,i}^2 \\sigma_{y,i}^2)`. This quantifies retrieval
    uncertainty and is the ingredient used by calibrated VSA classifiers.

    Args:
        x: First Gaussian hypervector.
        y: Second Gaussian hypervector.

    Returns:
        Non-negative scalar variance.
    """
    per_dim = (x.mu**2) * y.var + (y.mu**2) * x.var + x.var * y.var
    return jnp.sum(per_dim)


# ----------------------------------------------------------------------
# KL divergence
# ----------------------------------------------------------------------


@jax.jit
def kl_gaussian(p: GaussianHV, q: GaussianHV) -> jax.Array:
    """KL divergence :math:`\\mathrm{KL}(p \\| q)` for two diagonal Gaussians.

    Closed form:

    .. math::
        \\mathrm{KL}(p \\| q) = \\tfrac{1}{2} \\sum_i \\left[
            \\log\\frac{\\sigma_{q,i}^2}{\\sigma_{p,i}^2}
            + \\frac{\\sigma_{p,i}^2 + (\\mu_{p,i} - \\mu_{q,i})^2}{\\sigma_{q,i}^2}
            - 1
        \\right]

    Used as a regulariser when learning variational codebooks.

    Args:
        p: First Gaussian hypervector (posterior).
        q: Second Gaussian hypervector (prior).

    Returns:
        Non-negative scalar.
    """
    var_p = jnp.maximum(p.var, EPS)
    var_q = jnp.maximum(q.var, EPS)
    term = jnp.log(var_q / var_p) + (var_p + (p.mu - q.mu) ** 2) / var_q - 1.0
    return 0.5 * jnp.sum(term)


# ----------------------------------------------------------------------
# Derived Gaussian operations
# ----------------------------------------------------------------------


@jax.jit
def permute_gaussian(x: GaussianHV, shifts: int = 1) -> GaussianHV:
    """Cyclically permute a Gaussian hypervector.

    Applies the same cyclic shift to both the mean and variance vectors,
    which is the distributional analogue of classical permutation under
    the independent-component assumption.
    """
    return GaussianHV(
        mu=jnp.roll(x.mu, shifts, axis=-1),
        var=jnp.roll(x.var, shifts, axis=-1),
        dimensions=x.dimensions,
    )


def cleanup_gaussian(
    query: GaussianHV,
    memory: list[GaussianHV],
) -> tuple[int, float]:
    """Retrieve the entry in ``memory`` most similar to ``query``.

    Uses :func:`expected_cosine_similarity` as the scoring function —
    the uncertainty-aware analogue of classical cleanup. Returns
    ``(index, score)`` of the best match.

    Args:
        query: Query Gaussian hypervector.
        memory: Non-empty list of Gaussian hypervectors to search.

    Returns:
        ``(best_index, best_score)`` — the index into ``memory`` of the
        entry with the highest expected cosine similarity, and that
        similarity value.
    """
    if not memory:
        raise ValueError("cleanup_gaussian: memory must be non-empty")

    scores = jnp.array([float(expected_cosine_similarity(query, entry)) for entry in memory])
    idx = int(jnp.argmax(scores))
    return idx, float(scores[idx])


# ======================================================================
# Mixture hypervectors — mixture-of-Gaussian posteriors
# ======================================================================


@register_dataclass
@dataclass
class MixtureHV:
    r"""A mixture-of-Gaussian posterior over hypervectors.

    Represents a hypervector as a categorical mixture of component
    :class:`GaussianHV` posteriors — useful for multi-modal
    representations (e.g. a symbol with two plausible Gaussian
    interpretations, or a class prototype fit by EM).

    The mixture is parameterised by weights :math:`\boldsymbol{\pi}`
    (simplex-valued) and component means / variances stacked along a
    leading axis.

    Attributes:
        weights: Mixture weights of shape ``(K,)``. Must sum to 1.
        mu: Component means, shape ``(K, d)``.
        var: Component variances, shape ``(K, d)``.
        dimensions: Hypervector dimensionality :math:`d`.
    """

    weights: jax.Array
    mu: jax.Array
    var: jax.Array
    dimensions: int = field(metadata=dict(static=True))

    @staticmethod
    def from_components(
        components: list[GaussianHV],
        weights: jax.Array | None = None,
    ) -> MixtureHV:
        """Build a mixture from a list of :class:`GaussianHV` components.

        ``weights`` defaults to the uniform mixture ``1 / K``.
        """
        if not components:
            raise ValueError("MixtureHV.from_components: components must be non-empty")
        k = len(components)
        dims = components[0].dimensions
        mu = jnp.stack([c.mu for c in components], axis=0)
        var = jnp.stack([c.var for c in components], axis=0)
        if weights is None:
            weights = jnp.full((k,), 1.0 / k)
        else:
            weights = jnp.asarray(weights)
            weights = weights / (jnp.sum(weights) + EPS)
        return MixtureHV(weights=weights, mu=mu, var=var, dimensions=dims)

    @staticmethod
    def create(
        dimensions: int,
        n_components: int = 2,
    ) -> MixtureHV:
        """Uniform mixture of zero-mean, unit-variance components."""
        return MixtureHV(
            weights=jnp.full((n_components,), 1.0 / n_components),
            mu=jnp.zeros((n_components, dimensions)),
            var=jnp.ones((n_components, dimensions)),
            dimensions=dimensions,
        )

    def mean(self) -> jax.Array:
        """Overall mixture mean :math:`\\sum_k \\pi_k \\mu_k`."""
        return jnp.sum(self.weights[:, None] * self.mu, axis=0)

    def variance(self) -> jax.Array:
        r"""Overall mixture variance via the law of total variance.

        .. math::
            \mathrm{Var}[X] = \sum_k \pi_k (\sigma_k^2 + \mu_k^2)
                              - \left(\sum_k \pi_k \mu_k\right)^2
        """
        overall_mean = self.mean()
        second_moment = jnp.sum(self.weights[:, None] * (self.var + self.mu**2), axis=0)
        return second_moment - overall_mean**2

    def collapse_to_gaussian(self) -> GaussianHV:
        """Moment-matched Gaussian approximation of the mixture.

        Loses the multi-modal structure but is useful when a downstream
        operation expects a :class:`GaussianHV` and the mixture is
        unimodal-ish.
        """
        return GaussianHV(
            mu=self.mean(),
            var=self.variance(),
            dimensions=self.dimensions,
        )

    def sample(self, key: jax.Array) -> jax.Array:
        """Draw one sample from the mixture."""
        k_comp, k_gauss = jax.random.split(key)
        component = jax.random.categorical(k_comp, jnp.log(self.weights + EPS))
        eps = jax.random.normal(k_gauss, (self.dimensions,))
        return self.mu[component] + jnp.sqrt(jnp.maximum(self.var[component], 0.0)) * eps


# ======================================================================
# Dirichlet hypervectors — distributions on the probability simplex
# ======================================================================


@register_dataclass
@dataclass
class DirichletHV:
    r"""A Dirichlet distribution over the probability simplex :math:`\Delta_K`.

    Where :class:`GaussianHV` is the distributional analogue of a MAP
    hypervector in :math:`\mathbb{R}^d`, :class:`DirichletHV` is the
    distributional analogue of a categorical codebook: each "hypervector"
    is a posterior over categorical distributions on :math:`K` symbols,
    parameterised by a non-negative concentration vector
    :math:`\boldsymbol{\alpha} \in \mathbb{R}^K_{>0}`.

    Use cases:

    - Probabilistic categorical codebooks: each symbol is a distribution
      over :math:`K` token indices rather than a single index.
    - Bayesian counting: the concentration updates additively as evidence
      arrives (``from_counts`` implements add-:math:`\alpha` smoothing).
    - Variational posteriors over categorical latents.

    Attributes:
        alpha: Concentration of shape ``(d,)`` (or ``(n, d)`` batched).
            All entries must be strictly positive for a valid Dirichlet.
        dimensions: Number of categories :math:`K`.
    """

    alpha: jax.Array
    dimensions: int = field(metadata=dict(static=True))

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @staticmethod
    def create(dimensions: int, concentration: float = 1.0) -> DirichletHV:
        """Symmetric Dirichlet with all concentrations equal.

        ``concentration=1`` gives a uniform prior on the simplex;
        ``concentration<1`` concentrates mass near vertices;
        ``concentration>1`` concentrates mass near the centre.
        """
        return DirichletHV(
            alpha=jnp.full((dimensions,), float(concentration)),
            dimensions=dimensions,
        )

    @staticmethod
    def uniform(dimensions: int) -> DirichletHV:
        """Uniform distribution on the simplex (symmetric, alpha=1)."""
        return DirichletHV.create(dimensions, concentration=1.0)

    @staticmethod
    def from_counts(counts: jax.Array, prior: float = 1.0) -> DirichletHV:
        """Posterior Dirichlet from observation counts.

        Implements add-:math:`\\alpha` smoothing with ``prior`` as the
        Dirichlet prior concentration:

        .. math::
            \\boldsymbol{\\alpha}_{\\text{posterior}} = \\boldsymbol{\\alpha}_{\\text{prior}}
            + \\boldsymbol{c}

        where :math:`\\boldsymbol{c}` is the observation-count vector.
        """
        return DirichletHV(
            alpha=counts.astype(jnp.float32) + float(prior),
            dimensions=counts.shape[-1],
        )

    # ------------------------------------------------------------------
    # Moments and sampling
    # ------------------------------------------------------------------

    def mean(self) -> jax.Array:
        """Expected categorical distribution :math:`\\boldsymbol{\\alpha} / \\sum_k \\alpha_k`."""
        return self.alpha / (jnp.sum(self.alpha, axis=-1, keepdims=True) + EPS)

    def variance(self) -> jax.Array:
        r"""Per-category variance under the Dirichlet posterior.

        For component :math:`k`: :math:`\mathrm{Var}[p_k]
        = \bar{p}_k (1 - \bar{p}_k) / (\alpha_0 + 1)` where
        :math:`\bar{p}_k = \alpha_k / \alpha_0` and
        :math:`\alpha_0 = \sum_k \alpha_k`.
        """
        alpha_sum = jnp.sum(self.alpha, axis=-1, keepdims=True) + EPS
        mean = self.alpha / alpha_sum
        return mean * (1.0 - mean) / (alpha_sum + 1.0)

    def concentration(self) -> jax.Array:
        """Total concentration :math:`\\alpha_0 = \\sum_k \\alpha_k`.

        Larger :math:`\\alpha_0` means a tighter posterior; in the limit
        the Dirichlet collapses to a Dirac on :math:`\\bar{\\boldsymbol{p}}`.
        """
        return jnp.sum(self.alpha, axis=-1)

    def sample(self, key: jax.Array) -> jax.Array:
        """Draw a single categorical distribution from this Dirichlet."""
        return jax.random.dirichlet(key, jnp.maximum(self.alpha, EPS))

    def sample_batch(self, key: jax.Array, n: int) -> jax.Array:
        """Draw ``n`` categorical distributions from this Dirichlet."""
        return jax.random.dirichlet(key, jnp.maximum(self.alpha, EPS), shape=(n,))


# ----------------------------------------------------------------------
# Dirichlet binding and bundling
# ----------------------------------------------------------------------


@jax.jit
def bind_dirichlet(x: DirichletHV, y: DirichletHV) -> DirichletHV:
    r"""Bind two Dirichlet HVs by element-wise mean product, re-normalised.

    There is no canonical "binding" operation for Dirichlet posteriors in
    the VSA literature; we adopt the moment-matched approximation

    .. math::
        \bar{p}_z \propto \bar{p}_x \odot \bar{p}_y, \qquad
        \alpha_0^{(z)} = \alpha_0^{(x)} + \alpha_0^{(y)}

    — the element-wise product of expected categoricals combined with
    additive concentrations — so that binding two highly-concentrated
    Dirichlets yields a highly-concentrated result and binding against a
    flat prior leaves the other distribution approximately unchanged.
    This is the direct analogue of MAP binding on the means with
    uncertainty accumulation on the concentration.

    Args:
        x: First Dirichlet hypervector.
        y: Second Dirichlet hypervector.

    Returns:
        Moment-matched Dirichlet hypervector for the composed posterior.
    """
    mean_x = x.mean()
    mean_y = y.mean()
    new_mean = mean_x * mean_y
    new_mean = new_mean / (jnp.sum(new_mean, axis=-1, keepdims=True) + EPS)
    new_concentration = x.concentration() + y.concentration()
    new_alpha = new_mean * new_concentration[..., None] + EPS
    return DirichletHV(alpha=new_alpha, dimensions=x.dimensions)


@jax.jit
def bundle_dirichlet(hvs: DirichletHV) -> DirichletHV:
    r"""Bundle a batch of Dirichlet HVs by summing concentrations.

    Summing concentrations is the exact posterior update for a Dirichlet
    under independent observations: if each ``hvs[i]`` is a Dirichlet
    posterior from :math:`c_i` observations, then the combined posterior
    over the shared parameter is Dirichlet with
    :math:`\boldsymbol{\alpha} = \sum_i \boldsymbol{\alpha}_i`.

    Args:
        hvs: Batched :class:`DirichletHV` with ``alpha`` of shape ``(n, d)``.

    Returns:
        Combined Dirichlet hypervector of dimension ``d``.
    """
    summed = jnp.sum(hvs.alpha, axis=0)
    return DirichletHV(alpha=summed, dimensions=summed.shape[-1])


# ----------------------------------------------------------------------
# Dirichlet divergences
# ----------------------------------------------------------------------


@jax.jit
def kl_dirichlet(p: DirichletHV, q: DirichletHV) -> jax.Array:
    r"""Closed-form KL divergence :math:`\mathrm{KL}(p \| q)` for two Dirichlets.

    .. math::
        \mathrm{KL}(p \| q) &= \log \Gamma(\alpha_0^{(p)}) - \log \Gamma(\alpha_0^{(q)}) \\
        &+ \sum_i \left[\log \Gamma(\alpha_i^{(q)}) - \log \Gamma(\alpha_i^{(p)})\right] \\
        &+ \sum_i (\alpha_i^{(p)} - \alpha_i^{(q)}) \, [\psi(\alpha_i^{(p)}) - \psi(\alpha_0^{(p)})]

    where :math:`\psi` is the digamma function. Used as the KL term in
    variational bounds over categorical codebooks.
    """
    lgamma = jax.scipy.special.gammaln
    digamma = jax.scipy.special.digamma

    alpha_p = jnp.maximum(p.alpha, EPS)
    alpha_q = jnp.maximum(q.alpha, EPS)
    sum_p = jnp.sum(alpha_p, axis=-1)
    sum_q = jnp.sum(alpha_q, axis=-1)

    term1 = lgamma(sum_p) - lgamma(sum_q)
    term2 = jnp.sum(lgamma(alpha_q) - lgamma(alpha_p), axis=-1)
    term3 = jnp.sum(
        (alpha_p - alpha_q) * (digamma(alpha_p) - digamma(sum_p)[..., None]),
        axis=-1,
    )
    return term1 + term2 + term3


__all__ = [
    # Gaussian layer
    "GaussianHV",
    "bind_gaussian",
    "bundle_gaussian",
    "expected_cosine_similarity",
    "similarity_variance",
    "kl_gaussian",
    "permute_gaussian",
    "cleanup_gaussian",
    # Mixture layer
    "MixtureHV",
    # Dirichlet layer
    "DirichletHV",
    "bind_dirichlet",
    "bundle_dirichlet",
    "kl_dirichlet",
]
