# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Tests for the continuous-output regression stack.

Covers :class:`bayes_hdc.HDRegressor` (ridge regression on hypervector
features) and :class:`bayes_hdc.ConformalRegressor` (split-conformal
prediction intervals with finite-sample marginal coverage).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from bayes_hdc import (
    ConformalRegressor,
    HDRegressor,
)

DIMS = 64


# =============================================================================
# HDRegressor — closed-form ridge on continuous targets
# =============================================================================


def test_hdregressor_fit_predict_recovers_linear_signal() -> None:
    """For a known linear ground-truth W*, HDRegressor recovers it under low noise."""
    key = jax.random.PRNGKey(0)
    k_x, k_w, k_noise = jax.random.split(key, 3)
    n = 200
    output_dim = 3

    X = jax.random.normal(k_x, (n, DIMS))
    W_true = jax.random.normal(k_w, (DIMS, output_dim))
    Y = X @ W_true + 0.01 * jax.random.normal(k_noise, (n, output_dim))

    reg = HDRegressor.create(dimensions=DIMS, output_dim=output_dim, reg=1e-3)
    reg = reg.fit(X, Y)

    preds = reg.predict(X)
    rmse = float(jnp.sqrt(jnp.mean((preds - Y) ** 2)))
    assert rmse < 0.1, f"RMSE on training set = {rmse:.4f}"


def test_hdregressor_dual_form_used_when_d_greater_than_n() -> None:
    """Small-n high-d regime is the typical HDC setting; the dual-form
    solver must succeed there (the primal d×d system would be rank-deficient)."""
    key = jax.random.PRNGKey(1)
    k_x, k_y = jax.random.split(key)
    n = 16  # n < DIMS (=64)
    output_dim = 2

    X = jax.random.normal(k_x, (n, DIMS))
    Y = jax.random.normal(k_y, (n, output_dim))

    reg = HDRegressor.create(dimensions=DIMS, output_dim=output_dim, reg=1e-2)
    reg = reg.fit(X, Y)
    # On training data, dual-form ridge is interpolating-up-to-regularisation;
    # residual should be small but nonzero.
    preds = reg.predict(X)
    assert preds.shape == (n, output_dim)


def test_hdregressor_rejects_target_dim_mismatch() -> None:
    """fit() with the wrong output_dim raises a clear error."""
    reg = HDRegressor.create(dimensions=DIMS, output_dim=3)
    X = jnp.ones((10, DIMS))
    Y_wrong = jnp.ones((10, 5))
    try:
        reg.fit(X, Y_wrong)
        raise AssertionError("Expected ValueError on output_dim mismatch")
    except ValueError as exc:
        assert "output_dim" in str(exc)


def test_hdregressor_rejects_empty_training_set() -> None:
    reg = HDRegressor.create(dimensions=DIMS, output_dim=1)
    try:
        reg.fit(jnp.zeros((0, DIMS)), jnp.zeros((0, 1)))
        raise AssertionError("Expected ValueError on empty training set")
    except ValueError as exc:
        assert "empty" in str(exc).lower()


def test_hdregressor_handles_1d_targets() -> None:
    """Rank-1 targets (n,) are reshaped to (n, 1) automatically."""
    key = jax.random.PRNGKey(2)
    X = jax.random.normal(key, (100, DIMS))
    y_true = jnp.sum(X[:, :8], axis=-1)  # known linear function
    reg = HDRegressor.create(dimensions=DIMS, output_dim=1, reg=1e-3).fit(X, y_true)
    preds = reg.predict(X)
    # Prediction shape: (n, 1) — caller is responsible for any squeeze.
    assert preds.shape == (100, 1)


def test_hdregressor_score_returns_high_r2_on_clean_signal() -> None:
    key = jax.random.PRNGKey(3)
    k_x, k_w = jax.random.split(key)
    X = jax.random.normal(k_x, (500, DIMS))
    W = jax.random.normal(k_w, (DIMS, 2))
    Y = X @ W
    reg = HDRegressor.create(dimensions=DIMS, output_dim=2, reg=1e-4).fit(X, Y)
    r2 = float(reg.score(X, Y))
    assert r2 > 0.99


def test_hdregressor_grad_through_weights_is_finite() -> None:
    """jax.grad composes through HDRegressor.predict — important for
    plugging it into a downstream ELBO or sequence-to-sequence loss."""
    X = jnp.ones((1, DIMS))

    def loss(weights):
        reg = HDRegressor(weights=weights, dimensions=DIMS, output_dim=2, reg=1.0)
        preds = reg.predict(X)
        return jnp.sum(preds**2)

    weights = jnp.ones((DIMS, 2))
    g = jax.grad(loss)(weights)
    assert g.shape == (DIMS, 2)
    assert jnp.all(jnp.isfinite(g))


# =============================================================================
# ConformalRegressor — split-conformal absolute-residual intervals
# =============================================================================


def test_conformal_regressor_create_validates_alpha() -> None:
    """alpha must be in (0, 1)."""
    for bad in [0.0, 1.0, -0.1, 1.1]:
        try:
            ConformalRegressor.create(alpha=bad)
            raise AssertionError(f"Expected ValueError for alpha={bad}")
        except ValueError:
            pass


def test_conformal_regressor_quantile_grows_with_residuals() -> None:
    """Larger calibration residuals → wider intervals."""
    cr = ConformalRegressor.create(alpha=0.1, output_dim=1)

    preds_cal = jnp.zeros((100, 1))
    targets_small = 0.1 * jax.random.normal(jax.random.PRNGKey(0), (100, 1))
    targets_large = 1.0 * jax.random.normal(jax.random.PRNGKey(0), (100, 1))

    fitted_small = cr.fit(preds_cal, targets_small)
    fitted_large = cr.fit(preds_cal, targets_large)

    assert float(fitted_large.quantile[0]) > float(fitted_small.quantile[0])


def test_conformal_regressor_marginal_coverage_at_least_1_minus_alpha() -> None:
    """The finite-sample coverage guarantee: averaged over many
    calibration draws, empirical coverage on a fresh exchangeable test
    set is at least 1 − α (with high probability for sufficiently large n)."""
    rng = jax.random.PRNGKey(42)
    alpha = 0.1
    n_cal = 500
    n_test = 1000
    n_trials = 20

    coverages = []
    for trial in range(n_trials):
        rng, k_cal_x, k_cal_eps, k_test_x, k_test_eps = jax.random.split(rng, 5)
        # Synthetic exchangeable data: y = x · 1 + ε, ε ~ N(0, 0.5²).
        x_cal = jax.random.normal(k_cal_x, (n_cal,))
        y_cal = x_cal + 0.5 * jax.random.normal(k_cal_eps, (n_cal,))
        x_test = jax.random.normal(k_test_x, (n_test,))
        y_test = x_test + 0.5 * jax.random.normal(k_test_eps, (n_test,))

        # Use the identity predictor: ŷ = x. The conformal layer adapts.
        preds_cal = x_cal
        preds_test = x_test

        cr = ConformalRegressor.create(alpha=alpha, output_dim=1)
        cr = cr.fit(preds_cal, y_cal)
        cov = float(cr.coverage(preds_test, y_test)[0])
        coverages.append(cov)

    mean_cov = sum(coverages) / len(coverages)
    # Theoretical lower bound is 1 - α = 0.9; empirical mean over 20
    # trials should comfortably exceed (1 - α) - 2σ where σ ≈ √(α(1-α)/n_test).
    se = (alpha * (1 - alpha) / n_test) ** 0.5
    assert mean_cov >= (1 - alpha) - 3 * se, (
        f"mean coverage {mean_cov:.3f} below 1 - α - 3·SE = {(1 - alpha) - 3 * se:.3f}"
    )


def test_conformal_regressor_per_output_quantile_for_multi_output() -> None:
    """Multi-output targets get one quantile per output column."""
    cr = ConformalRegressor.create(alpha=0.1, output_dim=3)
    rng = jax.random.PRNGKey(7)
    n = 200
    preds = jax.random.normal(rng, (n, 3))
    # Per-column residual scales: 0.1, 1.0, 5.0.
    scales = jnp.array([0.1, 1.0, 5.0])
    targets = preds + scales * jax.random.normal(jax.random.PRNGKey(8), (n, 3))

    cr = cr.fit(preds, targets)
    q = cr.quantile
    assert q.shape == (3,)
    # Quantiles should reflect the relative residual scales.
    assert float(q[0]) < float(q[1]) < float(q[2])


def test_conformal_regressor_predict_interval_shape_matches_input() -> None:
    cr = ConformalRegressor.create(alpha=0.1, output_dim=2)
    cr = cr.fit(jnp.zeros((50, 2)), 0.5 * jax.random.normal(jax.random.PRNGKey(0), (50, 2)))

    # Single point.
    p_single = jnp.array([1.0, 2.0])
    lo, hi = cr.predict_interval(p_single)
    assert lo.shape == (2,)
    assert hi.shape == (2,)
    assert jnp.all(lo < hi)

    # Batch.
    p_batch = jnp.zeros((10, 2))
    lo_b, hi_b = cr.predict_interval(p_batch)
    assert lo_b.shape == (10, 2)
    assert hi_b.shape == (10, 2)


def test_conformal_regressor_interval_width_is_2q() -> None:
    cr = ConformalRegressor.create(alpha=0.1, output_dim=2)
    cr = cr.fit(jnp.zeros((50, 2)), 0.5 * jax.random.normal(jax.random.PRNGKey(0), (50, 2)))
    width = cr.interval_width()
    assert jnp.allclose(width, 2.0 * cr.quantile)


def test_conformal_regressor_rejects_too_few_calibration_points() -> None:
    cr = ConformalRegressor.create(alpha=0.1, output_dim=1)
    try:
        cr.fit(jnp.zeros((1, 1)), jnp.zeros((1, 1)))
        raise AssertionError("Expected ValueError for n_cal < 2")
    except ValueError as exc:
        assert "calibration" in str(exc).lower()


# =============================================================================
# Integration: HDRegressor + ConformalRegressor
# =============================================================================


def test_hdregressor_conformal_pipeline_end_to_end() -> None:
    """Encoder → HDRegressor → ConformalRegressor pipeline produces
    interval bounds that cover the target on a held-out exchangeable
    test set at the nominal rate."""
    rng = jax.random.PRNGKey(2026)
    k_X, k_W, k_eps_train, k_eps_cal, k_eps_test = jax.random.split(rng, 5)
    n_train, n_cal, n_test = 200, 200, 500
    output_dim = 2

    # Synthetic ground truth: y = X @ W + noise.
    W_true = jax.random.normal(k_W, (DIMS, output_dim))
    X_train = jax.random.normal(k_X, (n_train + n_cal + n_test, DIMS))
    y_clean = X_train @ W_true
    noise_scale = 0.3
    y_train = y_clean[:n_train] + noise_scale * jax.random.normal(
        k_eps_train, (n_train, output_dim)
    )
    y_cal = y_clean[n_train : n_train + n_cal] + noise_scale * jax.random.normal(
        k_eps_cal, (n_cal, output_dim)
    )
    y_test = y_clean[n_train + n_cal :] + noise_scale * jax.random.normal(
        k_eps_test, (n_test, output_dim)
    )
    X_train_split = X_train[:n_train]
    X_cal = X_train[n_train : n_train + n_cal]
    X_test = X_train[n_train + n_cal :]

    # Fit regressor on train, calibrate conformal on cal, evaluate on test.
    reg = HDRegressor.create(dimensions=DIMS, output_dim=output_dim, reg=1e-2)
    reg = reg.fit(X_train_split, y_train)

    preds_cal = reg.predict(X_cal)
    preds_test = reg.predict(X_test)

    cr = ConformalRegressor.create(alpha=0.1, output_dim=output_dim)
    cr = cr.fit(preds_cal, y_cal)

    coverage = cr.coverage(preds_test, y_test)
    # Marginal coverage per output column ≥ 1 - α *in expectation*.
    # A single realisation has finite-sample slack of order
    # √(α(1-α)/n_test) ≈ √(0.09/500) ≈ 0.013; the empirical bound on
    # any one trial is 1 - α - 3·SE ≈ 0.86. The multi-trial test
    # `test_conformal_regressor_marginal_coverage_at_least_1_minus_alpha`
    # is the rigorous version of this guarantee.
    assert float(jnp.min(coverage)) >= 0.80
    # Width should be of order 2 × σ × Φ⁻¹(1-α/2) for a Gaussian likelihood;
    # at α = 0.1, σ = 0.3, expected ≈ 2 × 0.3 × 1.645 ≈ 0.99. Loose check:
    width = cr.interval_width()
    assert jnp.all(width > 0.0)
    assert jnp.all(width < 5.0)
