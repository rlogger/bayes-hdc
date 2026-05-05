# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Uncertainty quantification and calibration for HDC classifiers.

This module ships three wrappers that convert any classifier or
regressor into an uncertainty-aware one:

- :class:`TemperatureCalibrator` — a one-parameter post-hoc calibrator
  that rescales logits by a learned temperature :math:`T > 0` before the
  softmax, fitted by minimising negative log-likelihood on a held-out
  validation set. This is the standard baseline from Guo et al. (2017).

- :class:`ConformalClassifier` — a split-conformal wrapper that returns
  prediction *sets* with a guaranteed marginal coverage of
  :math:`1 - \\alpha`. Uses the Adaptive Prediction Sets (APS)
  nonconformity score from Romano et al. (2020), which produces
  class-balanced sets and handles multi-class natively.

- :class:`ConformalRegressor` — a split-conformal wrapper for
  *continuous*-output predictors. Returns symmetric prediction
  intervals :math:`[\\hat y - q, \\hat y + q]` where :math:`q` is the
  appropriate empirical quantile of the calibration absolute residuals,
  with a finite-sample marginal-coverage guarantee
  :math:`\\mathbb{P}(y \\in [\\hat y - q, \\hat y + q]) \\geq 1 - \\alpha`
  on exchangeable data. Pairs naturally with
  :class:`~bayes_hdc.models.HDRegressor`.

All three are JAX pytrees: ``jit``, ``vmap``, and ``grad`` compose
through them without special handling. They wrap any model that
exposes raw scores or predictions — :class:`~bayes_hdc.models.CentroidClassifier.similarity`,
:class:`~bayes_hdc.models.RegularizedLSClassifier.predict`,
:class:`~bayes_hdc.models.HDRegressor.predict`, or a user-supplied
model — so existing pipelines get calibration without retraining.
"""

from __future__ import annotations

import warnings
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

        log_t = init_log_t
        used_fallback = False
        try:
            from jax.scipy.optimize import minimize  # type: ignore[import-not-found]

            result = minimize(
                nll_log_t,
                init,
                method="BFGS",
                options={"maxiter": max_iters},
            )
            candidate = result.x[0]
            # If BFGS produced a non-finite result, fall through to gradient
            # descent rather than propagate NaN. (BFGS itself does not raise
            # on failure-to-converge; it returns a result with non-finite
            # entries or success=False.)
            if jnp.isfinite(candidate):
                log_t = candidate
            else:
                used_fallback = True
        except ImportError:  # pragma: no cover — older JAX without minimize
            used_fallback = True

        if used_fallback:
            warnings.warn(
                "TemperatureCalibrator: jax.scipy.optimize.minimize unavailable "
                "or BFGS produced a non-finite result; falling back to gradient "
                "descent. The fitted temperature is still consistent but may be "
                "less precise — pin a JAX version that ships minimize, or "
                "increase max_iters.",
                RuntimeWarning,
                stacklevel=2,
            )
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


# ----------------------------------------------------------------------
# Conformal regression — split-conformal absolute-residual intervals
# ----------------------------------------------------------------------


@register_dataclass
@dataclass
class ConformalRegressor:
    r"""Split-conformal regression with finite-sample coverage.

    For a regressor :math:`\hat f` and a calibration set
    :math:`\{(x_i, y_i)\}_{i=1}^{n}` exchangeable with the test point,
    the absolute-residual nonconformity score is
    :math:`s_i = |y_i - \hat f(x_i)|`. Sort the calibration scores and
    take the empirical
    :math:`\lceil (n+1)(1-\alpha) \rceil / n` quantile :math:`q`. The
    prediction interval at a new point :math:`x` is
    :math:`[\hat f(x) - q, \hat f(x) + q]`. Lei et al. (2018,
    *Distribution-Free Predictive Inference for Regression*) prove
    that, under exchangeability,

    .. math::
        \mathbb{P}\bigl(y_{\mathrm{test}} \in [\hat f(x) - q, \hat f(x) + q]\bigr)
        \;\geq\; 1 - \alpha,

    independent of the regressor's quality, the data distribution, or
    the dimension. Loss of regressor accuracy widens the intervals; it
    does not break the coverage guarantee.

    For multi-output targets, an interval is produced per output
    dimension by computing one quantile per output column. (A joint
    coverage guarantee at the *vector* level requires Bonferroni or a
    multivariate nonconformity score; this implementation gives
    marginal coverage per output dimension.)

    Concurrent algorithmic work in HDC: Liang et al. (2026)
    *ConformalHDC* (arXiv:2602.21446) develops adaptive nonconformity
    scores tailored to prototype geometry. ``ConformalRegressor`` is
    the simpler absolute-residual variant — sufficient for the
    calibrated-regression use case but composable with any user-
    supplied score.

    Attributes:
        quantile: Empirical quantile of calibration residuals, of
            shape ``(output_dim,)`` (one per output column).
        alpha: Target miscoverage rate; the guaranteed marginal
            coverage is ``1 - alpha``.
        output_dim: Number of regression output dimensions.
        n_calibration: Number of calibration points used to fit the
            quantile.
    """

    quantile: jax.Array  # (output_dim,)
    alpha: float = field(metadata=dict(static=True))
    output_dim: int = field(metadata=dict(static=True))
    n_calibration: int = field(metadata=dict(static=True), default=0)

    @staticmethod
    def create(alpha: float = 0.1, output_dim: int = 1) -> ConformalRegressor:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if output_dim < 1:
            raise ValueError(f"output_dim must be >= 1, got {output_dim}")
        return ConformalRegressor(
            quantile=jnp.zeros((output_dim,)),
            alpha=alpha,
            output_dim=output_dim,
            n_calibration=0,
        )

    def fit(
        self,
        predictions_cal: jax.Array,
        targets_cal: jax.Array,
    ) -> ConformalRegressor:
        """Compute the conformal quantile from calibration residuals.

        Args:
            predictions_cal: Predicted outputs on the calibration set,
                shape ``(n,)``, ``(n, k)``, or scalar-per-row.
            targets_cal: True targets on the calibration set, same
                shape as ``predictions_cal``.

        Returns:
            A fitted ``ConformalRegressor`` whose ``quantile`` field
            holds the empirical
            :math:`\\lceil (n+1)(1-\\alpha) \\rceil / n` residual quantile
            per output column.
        """
        preds = predictions_cal
        targets = targets_cal
        if preds.ndim == 1:
            preds = preds[:, None]
        if targets.ndim == 1:
            targets = targets[:, None]
        if preds.shape != targets.shape:
            raise ValueError(
                f"predictions_cal and targets_cal must have the same shape; "
                f"got {predictions_cal.shape} and {targets_cal.shape}"
            )
        if preds.shape[1] != self.output_dim:
            raise ValueError(
                f"calibration data has {preds.shape[1]} output columns "
                f"but the regressor was created with output_dim={self.output_dim}"
            )

        n = preds.shape[0]
        if n < 2:
            raise ValueError(
                f"Need at least 2 calibration points to fit a conformal quantile; got {n}"
            )

        residuals = jnp.abs(targets - preds)  # (n, k)
        # Per-column quantile at level ⌈(n+1)(1−α)⌉ / n. Clamp to [0, 1]
        # so that absurdly small n behaves sensibly.
        level = jnp.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = jnp.clip(level, 0.0, 1.0)
        q = jnp.quantile(residuals, level, axis=0)  # (k,)

        return ConformalRegressor(
            quantile=q,
            alpha=self.alpha,
            output_dim=self.output_dim,
            n_calibration=int(n),
        )

    @jax.jit
    def predict_interval(
        self,
        predictions: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Return symmetric ``(lower, upper)`` interval bounds.

        Args:
            predictions: Predicted outputs of shape ``(k,)`` or
                ``(n, k)``.

        Returns:
            Tuple ``(lower, upper)`` of the same shape as
            ``predictions``, with
            ``lower = predictions - quantile``,
            ``upper = predictions + quantile``.
        """
        return predictions - self.quantile, predictions + self.quantile

    @jax.jit
    def coverage(
        self,
        predictions: jax.Array,
        targets: jax.Array,
    ) -> jax.Array:
        """Empirical coverage of the produced intervals on a test set.

        Per-output-dim coverage = fraction of calibration points whose
        true target falls inside the corresponding interval. The
        finite-sample guarantee from Lei et al. (2018) ensures the
        expectation is at least ``1 - alpha`` under exchangeability;
        the empirical fraction is a Monte-Carlo estimate of that
        expectation.

        Returns:
            Per-output coverage of shape ``(output_dim,)``.
        """
        preds = predictions if predictions.ndim > 1 else predictions[None, :]
        tgts = targets if targets.ndim > 1 else targets[None, :]
        lower = preds - self.quantile
        upper = preds + self.quantile
        in_interval = (tgts >= lower) & (tgts <= upper)
        return jnp.mean(in_interval.astype(jnp.float32), axis=0)

    @jax.jit
    def interval_width(self) -> jax.Array:
        """Width of the prediction interval (constant under absolute-residual CP).

        Returns:
            Per-output interval width ``2 * quantile`` of shape
            ``(output_dim,)``.
        """
        return 2.0 * self.quantile


__all__ = [
    "TemperatureCalibrator",
    "ConformalClassifier",
    "ConformalRegressor",
]
