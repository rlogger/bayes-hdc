# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Uncertainty quantification and calibration for HDC classifiers.

This module ships two wrappers that convert any classifier producing
pre-softmax scores or class probabilities into an uncertainty-aware one:

- :class:`TemperatureCalibrator` — a one-parameter post-hoc calibrator
  that rescales logits by a learned temperature :math:`T > 0` before the
  softmax, fitted by minimising negative log-likelihood on a held-out
  validation set. This is the standard baseline from Guo et al. (2017).

- :class:`ConformalClassifier` — a split-conformal wrapper that returns
  prediction *sets* with a guaranteed marginal coverage of
  :math:`1 - \\alpha`. Uses the Adaptive Prediction Sets (APS)
  nonconformity score from Romano et al. (2020), which produces
  class-balanced sets and handles multi-class natively.

Both are JAX pytrees: ``jit``, ``vmap``, and ``grad`` compose through
them without special handling. They wrap any classifier that exposes raw
scores — :class:`~bayes_hdc.models.CentroidClassifier.similarity`,
:class:`~bayes_hdc.models.RegularizedLSClassifier.predict`, or a
user-supplied model — so existing pipelines get calibration without
retraining.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from bayes_hdc._compat import register_dataclass
from bayes_hdc.constants import EPS

# ----------------------------------------------------------------------
# Temperature scaling
# ----------------------------------------------------------------------


@register_dataclass
@dataclass
class TemperatureCalibrator:
    r"""Temperature scaling for post-hoc classifier calibration.

    Rescales pre-softmax scores (logits) by a single positive scalar
    :math:`T` before applying the softmax:

    .. math::
        p_{\text{cal}}(y \mid x) = \mathrm{softmax}(z / T)_y

    :math:`T` is fitted by minimising negative log-likelihood on a
    held-out validation set with a few hundred steps of gradient descent.
    The calibrator is accuracy-preserving: :math:`\arg\max_y p(y \mid x)`
    is unchanged.

    Attributes:
        temperature: Positive scalar :math:`T`. Reported as a JAX array to
            stay pytree-compatible.
    """

    temperature: jax.Array

    @staticmethod
    def create(initial_temperature: float = 1.0) -> TemperatureCalibrator:
        """Build an identity calibrator (T = 1). Call :meth:`fit` to learn."""
        return TemperatureCalibrator(temperature=jnp.asarray(float(initial_temperature)))

    def fit(
        self,
        logits: jax.Array,
        labels: jax.Array,
        max_iters: int = 300,
        lr: float = 0.01,
        t_min: float = 0.01,
        t_max: float = 100.0,
    ) -> TemperatureCalibrator:
        """Fit the temperature by minimising NLL.

        Uses L-BFGS in log-space (:math:`T = e^{\\theta}`) with a safety
        clip to ``[t_min, t_max]``. This matches the Guo et al. (2017)
        reference implementation and is robust to the flat NLL landscape
        that arises when raw cosine-similarity logits have small range.

        Args:
            logits: Validation logits of shape ``(n, k)``.
            labels: Validation integer labels of shape ``(n,)``.
            max_iters: BFGS iteration cap (default 300).
            lr: Fallback gradient-descent step size if BFGS is unavailable.
            t_min: Hard lower bound on the fitted temperature.
            t_max: Hard upper bound on the fitted temperature.

        Returns:
            A new :class:`TemperatureCalibrator` with the fitted temperature.
        """
        labels_idx = jnp.asarray(labels, dtype=jnp.int32)
        n = labels_idx.shape[0]

        def nll_log_t(log_t: jax.Array) -> jax.Array:
            t = jnp.exp(log_t[0])
            scaled = logits / t
            log_probs = jax.nn.log_softmax(scaled, axis=-1)
            return -jnp.mean(log_probs[jnp.arange(n), labels_idx])

        init_log_t = jnp.log(jnp.maximum(self.temperature, EPS))
        init = jnp.asarray([init_log_t])

        try:
            from jax.scipy.optimize import minimize

            result = minimize(
                nll_log_t,
                init,
                method="BFGS",
                options={"maxiter": max_iters},
            )
            log_t = result.x[0]
        except Exception:  # pragma: no cover — JAX version without minimize
            log_t = init_log_t
            grad_fn = jax.jit(jax.value_and_grad(lambda lt: nll_log_t(jnp.asarray([lt]))))
            for _ in range(max_iters):
                _, g = grad_fn(log_t)
                g = jnp.clip(g, -1.0, 1.0)
                log_t = log_t - lr * g

        log_t = jnp.clip(log_t, jnp.log(t_min), jnp.log(t_max))
        t = jnp.exp(log_t)
        return TemperatureCalibrator(temperature=t)

    @jax.jit
    def calibrate(self, logits: jax.Array) -> jax.Array:
        """Apply temperature scaling and return calibrated probabilities."""
        return jax.nn.softmax(logits / jnp.maximum(self.temperature, EPS), axis=-1)


# ----------------------------------------------------------------------
# Conformal prediction
# ----------------------------------------------------------------------


@register_dataclass
@dataclass
class ConformalClassifier:
    r"""Split-conformal wrapper — prediction sets with coverage guarantee.

    Given a held-out calibration set of (probability vector, label) pairs,
    :meth:`fit` learns a scalar threshold :math:`\hat{q}` such that
    :meth:`predict_set` returns a boolean mask over classes guaranteeing
    marginal coverage at level :math:`1 - \alpha`:

    .. math::
        \Pr\!\left(Y_{n+1} \in \hat{C}(X_{n+1})\right) \geq 1 - \alpha

    under the standard exchangeability assumption on the calibration and
    test data.

    The nonconformity score is the Adaptive Prediction Sets (APS) score
    from Romano et al. (2020):

    .. math::
        s(x, y) = \sum_{k: p_k(x) \geq p_y(x)} p_k(x)

    — the cumulative probability of classes at least as confident as the
    true one. APS produces class-balanced sets and handles multi-class
    natively, unlike the simpler LAC score.

    Attributes:
        threshold: Scalar quantile :math:`\hat{q}` learned from calibration.
        alpha: Target miscoverage rate :math:`\alpha` (static).
    """

    threshold: jax.Array
    alpha: float = field(metadata=dict(static=True), default=0.1)

    @staticmethod
    def create(alpha: float = 0.1) -> ConformalClassifier:
        """Build an untrained wrapper. Call :meth:`fit` on a calibration set."""
        return ConformalClassifier(threshold=jnp.asarray(1.0), alpha=float(alpha))

    def fit(
        self,
        calibration_probs: jax.Array,
        calibration_labels: jax.Array,
    ) -> ConformalClassifier:
        """Compute conformal threshold from a held-out calibration set.

        Args:
            calibration_probs: Class probabilities of shape ``(n, k)``.
            calibration_labels: Integer labels of shape ``(n,)``.

        Returns:
            A new :class:`ConformalClassifier` with the fitted threshold.
        """
        n = calibration_probs.shape[0]

        # APS score per calibration sample.
        sort_idx = jnp.argsort(-calibration_probs, axis=-1)
        sorted_probs = jnp.take_along_axis(calibration_probs, sort_idx, axis=-1)
        cumsums = jnp.cumsum(sorted_probs, axis=-1)

        # Rank of the true label within the sorted (descending) order.
        label_mask = sort_idx == calibration_labels[:, None]
        ranks = jnp.argmax(label_mask.astype(jnp.int32), axis=-1)

        scores = cumsums[jnp.arange(n), ranks]

        # Finite-sample corrected quantile: ceil((n+1)(1-alpha)) / n.
        q = jnp.clip(jnp.ceil((n + 1) * (1.0 - self.alpha)) / n, 0.0, 1.0)
        threshold = jnp.quantile(scores, q)

        return ConformalClassifier(threshold=threshold, alpha=self.alpha)

    @jax.jit
    def predict_set(self, probs: jax.Array) -> jax.Array:
        """Return a boolean mask of shape ``(n, k)`` — True if class in set."""
        sort_idx = jnp.argsort(-probs, axis=-1)
        inv_sort = jnp.argsort(sort_idx, axis=-1)
        sorted_probs = jnp.take_along_axis(probs, sort_idx, axis=-1)
        cumsums = jnp.cumsum(sorted_probs, axis=-1)

        # APS: include classes while cumulative prob is below threshold.
        include_sorted = cumsums <= self.threshold
        # Always include the top-1 to avoid empty sets.
        include_sorted = include_sorted.at[:, 0].set(True)

        # Scatter back to original class ordering.
        return jnp.take_along_axis(include_sorted, inv_sort, axis=-1)

    @jax.jit
    def coverage(self, probs: jax.Array, labels: jax.Array) -> jax.Array:
        """Fraction of samples whose true label falls in the prediction set.

        Should be close to ``1 - alpha`` on exchangeable test data.
        """
        mask = self.predict_set(probs)
        n = labels.shape[0]
        in_set = mask[jnp.arange(n), labels]
        return jnp.mean(in_set.astype(jnp.float32))

    @jax.jit
    def set_size(self, probs: jax.Array) -> jax.Array:
        """Mean number of classes in each prediction set (sharpness).

        Smaller is sharper. A useful quality signal alongside coverage:
        two classifiers with 90% coverage are not equivalent if one returns
        sets of size 2 and the other returns sets of size 7.
        """
        mask = self.predict_set(probs)
        return jnp.mean(jnp.sum(mask.astype(jnp.float32), axis=-1))


__all__ = [
    "TemperatureCalibrator",
    "ConformalClassifier",
]
