# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Posterior-predictive checks and goodness-of-fit diagnostics for PVSA.

Posterior predictive checks (PPCs) assess whether samples drawn from
a fitted posterior look consistent with the observed data. The
workflow is:

1. Choose a statistic :math:`T(\\cdot)` of interest — e.g. mean,
   variance, max cosine similarity to a reference vector.
2. Compute :math:`T(x_\\mathrm{obs})` on the observed data.
3. Draw :math:`N` posterior-predictive samples and compute
   :math:`T(x_\\mathrm{sim}^{(i)})` on each.
4. Compare the observed statistic to the predictive distribution —
   the empirical CDF, a one-sided :math:`p`-value, or a 95 %
   credibility interval.

This module provides :func:`posterior_predictive_check` — the
general-purpose driver — plus two ready-made statistics
(:func:`statistic_mean_norm`, :func:`statistic_cosine_to_reference`)
and :func:`coverage_calibration_check` for auditing a
:class:`~bayes_hdc.uncertainty.ConformalClassifier` against its
claimed coverage level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from bayes_hdc.distributions import GaussianHV


@dataclass
class PPCResult:
    """Outcome of a posterior predictive check.

    Attributes:
        observed: :math:`T(x_\\mathrm{obs})` — the observed statistic.
        predictive_mean: Mean of the posterior-predictive distribution
            of :math:`T`.
        predictive_std: Standard deviation of the predictive
            distribution.
        ci_low, ci_high: 95 % equal-tailed credibility interval.
        p_value: Two-sided empirical :math:`p`-value — the fraction of
            simulated statistics at least as extreme (in absolute
            deviation from the mean) as the observed one. Values near
            0.5 indicate excellent fit; values near 0 / 1 indicate
            misspecification.
    """

    observed: float
    predictive_mean: float
    predictive_std: float
    ci_low: float
    ci_high: float
    p_value: float


def statistic_mean_norm(x: jax.Array) -> jax.Array:
    """Example statistic: mean L2 norm across a batch ``(n, d)``."""
    return jnp.mean(jnp.linalg.norm(x, axis=-1))


def statistic_cosine_to_reference(
    x: jax.Array,
    reference: jax.Array,
) -> jax.Array:
    """Example statistic: max cosine similarity to ``reference`` across a batch."""
    eps = 1e-8
    ref_norm = reference / (jnp.linalg.norm(reference) + eps)
    x_norm = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)
    return jnp.max(x_norm @ ref_norm)


def posterior_predictive_check(
    posterior: GaussianHV,
    observed: jax.Array,
    statistic: Callable[[jax.Array], jax.Array],
    key: jax.Array,
    n_replicas: int = 500,
) -> PPCResult:
    """Compare an observed statistic to its posterior-predictive distribution.

    Args:
        posterior: Fitted PVSA posterior to check.
        observed: Observed data, shape ``(n, d)``.
        statistic: Callable mapping ``(n, d)`` to a scalar.
        key: JAX random key.
        n_replicas: Number of posterior-predictive replicates to draw.
            Each replicate matches the leading size of ``observed``.

    Returns:
        :class:`PPCResult` with the observed statistic, the predictive
        mean / std / 95 % CI, and a two-sided empirical p-value.
    """
    n_obs = observed.shape[0]
    obs_stat = float(statistic(observed))

    sim_stats = []
    for r in range(n_replicas):
        sub_key = jax.random.fold_in(key, r)
        replicate = posterior.sample_batch(sub_key, n_obs)
        sim_stats.append(float(statistic(replicate)))
    sim_arr = jnp.asarray(sim_stats)

    mean = float(jnp.mean(sim_arr))
    std = float(jnp.std(sim_arr))
    ci_low = float(jnp.quantile(sim_arr, 0.025))
    ci_high = float(jnp.quantile(sim_arr, 0.975))

    # Two-sided p-value: fraction of replicates whose deviation from
    # the predictive mean is at least as large as the observed one.
    deviations = jnp.abs(sim_arr - mean)
    observed_dev = abs(obs_stat - mean)
    p_value = float(jnp.mean(deviations >= observed_dev))

    return PPCResult(
        observed=obs_stat,
        predictive_mean=mean,
        predictive_std=std,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
    )


@dataclass
class CoverageCheckResult:
    """Empirical coverage sweep for a conformal classifier.

    Attributes:
        alphas: Array of requested miscoverage levels.
        empirical_coverage: Fraction of test labels inside the
            prediction set at each :math:`\\alpha`.
        set_sizes: Mean prediction-set size at each :math:`\\alpha`.
        max_deviation: Worst-case absolute deviation of empirical
            coverage from the nominal :math:`1 - \\alpha` line.
    """

    alphas: jax.Array
    empirical_coverage: jax.Array
    set_sizes: jax.Array
    max_deviation: float


def coverage_calibration_check(
    conformal_factory: Callable,
    probs_cal: jax.Array,
    labels_cal: jax.Array,
    probs_test: jax.Array,
    labels_test: jax.Array,
    alphas: list[float] | None = None,
) -> CoverageCheckResult:
    """Audit conformal-predictor coverage across a grid of :math:`\\alpha`.

    For each :math:`\\alpha \\in` ``alphas``, fits a fresh conformal
    classifier on ``(probs_cal, labels_cal)`` and measures the
    empirical coverage and mean set size on
    ``(probs_test, labels_test)``. Useful for verifying that the
    classifier actually delivers its claimed coverage guarantee and
    for choosing a practical operating point.

    Args:
        conformal_factory: Callable ``alpha -> ConformalClassifier``.
        probs_cal, labels_cal: Calibration set.
        probs_test, labels_test: Test set.
        alphas: Miscoverage levels to sweep. Defaults to
            ``[0.05, 0.1, 0.15, 0.2, 0.25, 0.3]``.
    """
    if alphas is None:
        alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    alphas_arr = jnp.asarray(alphas, dtype=jnp.float32)

    coverages = []
    sizes = []
    for a in alphas:
        wrapper = conformal_factory(float(a)).fit(probs_cal, labels_cal)
        coverages.append(float(wrapper.coverage(probs_test, labels_test)))
        sizes.append(float(wrapper.set_size(probs_test)))

    cov_arr = jnp.asarray(coverages, dtype=jnp.float32)
    size_arr = jnp.asarray(sizes, dtype=jnp.float32)
    nominal = 1.0 - alphas_arr
    max_dev = float(jnp.max(jnp.abs(cov_arr - nominal)))

    return CoverageCheckResult(
        alphas=alphas_arr,
        empirical_coverage=cov_arr,
        set_sizes=size_arr,
        max_deviation=max_dev,
    )


__all__ = [
    "PPCResult",
    "statistic_mean_norm",
    "statistic_cosine_to_reference",
    "posterior_predictive_check",
    "CoverageCheckResult",
    "coverage_calibration_check",
]
