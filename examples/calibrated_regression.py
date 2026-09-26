# SPDX-License-Identifier: MIT
# Copyright (c) 2026 R.S.

"""Continuous-output regression on synthetic hypervector features.

Fits ridge regression on a training split and calibrates absolute residuals
on an independent split. For exchangeable calibration/test data this gives
marginal coverage for each output coordinate, not simultaneous coverage.
A separate heuristic abstains when the interval box contains the zero vector;
it has no selective-risk or control-safety guarantee.

Run: python examples/calibrated_regression.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bayes_hdc import (
    MAP,
    ConformalRegressor,
    HDRegressor,
    RandomEncoder,
)

DIMS = 4096
SEED = 2026
N_FEATURES = 8  # tabular feature dimension
N_VALUES = 16  # discretisation levels per feature
N_TRAIN = 600
N_CAL = 300
N_TEST = 500
OUTPUT_DIM = 2  # 2-D continuous target — think x,y action delta
ALPHA = 0.1  # 90 % target coverage


def synthesise_data(key, n: int):
    """Synthetic tabular features + a smooth nonlinear 2-D target.

    Features are integer indices in [0, N_VALUES); the target is a
    smooth function of the indices plus Gaussian observation noise.
    """
    k_x, k_eps = jax.random.split(key)
    indices = jax.random.randint(k_x, (n, N_FEATURES), 0, N_VALUES)
    # Smooth ground-truth target: weighted sums of sin / cos of the
    # feature indices. Linearity in the encoded HDC space depends on
    # whether the encoder captures these index relationships, which
    # RandomEncoder does only approximately — so the regressor will
    # have honest residuals.
    weights_x = jnp.linspace(-1.0, 1.0, N_FEATURES)
    weights_y = jnp.linspace(0.5, -0.5, N_FEATURES)
    target_x = jnp.sum(jnp.sin(0.4 * indices) * weights_x, axis=-1)
    target_y = jnp.sum(jnp.cos(0.3 * indices) * weights_y, axis=-1)
    targets_clean = jnp.stack([target_x, target_y], axis=-1)
    noise = 0.15 * jax.random.normal(k_eps, (n, OUTPUT_DIM))
    return indices, targets_clean + noise


def main() -> None:
    print("Calibrated continuous-output regression with HDC")
    print(f"  d = {DIMS}    α = {ALPHA}    output_dim = {OUTPUT_DIM}\n")

    key = jax.random.PRNGKey(SEED)
    k_enc, k_train, k_cal, k_test = jax.random.split(key, 4)

    # ----------------------------------------------------------------- 1.
    print("[1] Build a RandomEncoder over tabular features.")
    encoder = RandomEncoder.create(
        num_features=N_FEATURES,
        num_values=N_VALUES,
        dimensions=DIMS,
        vsa_model=MAP.create(dimensions=DIMS),
        key=k_enc,
    )
    print(f"      shape: {N_FEATURES} features × {N_VALUES} values → d = {DIMS}")

    # ----------------------------------------------------------------- 2.
    print("\n[2] Synthesise train / calibration / test sets.")
    train_idx, train_targets = synthesise_data(k_train, N_TRAIN)
    cal_idx, cal_targets = synthesise_data(k_cal, N_CAL)
    test_idx, test_targets = synthesise_data(k_test, N_TEST)
    print(f"      n_train = {N_TRAIN}    n_cal = {N_CAL}    n_test = {N_TEST}")

    train_hvs = encoder.encode_batch(train_idx)
    cal_hvs = encoder.encode_batch(cal_idx)
    test_hvs = encoder.encode_batch(test_idx)
    print(f"      encoded train shape: {tuple(train_hvs.shape)}")

    # ----------------------------------------------------------------- 3.
    print("\n[3] Fit HDRegressor (ridge regression on the hypervector features).")
    reg = HDRegressor.create(dimensions=DIMS, output_dim=OUTPUT_DIM, reg=1.0)
    reg = reg.fit(train_hvs, train_targets)
    train_r2 = float(reg.score(train_hvs, train_targets))
    test_r2 = float(reg.score(test_hvs, test_targets))
    print(f"      R² (train) = {train_r2:+.4f}")
    print(f"      R² (test)  = {test_r2:+.4f}")

    test_preds = reg.predict(test_hvs)
    cal_preds = reg.predict(cal_hvs)
    test_rmse = float(jnp.sqrt(jnp.mean((test_preds - test_targets) ** 2)))
    print(f"      RMSE (test) = {test_rmse:.4f}")

    # ----------------------------------------------------------------- 4.
    print(f"\n[4] Calibrate ConformalRegressor at α = {ALPHA}.")
    cr = ConformalRegressor.create(alpha=ALPHA, output_dim=OUTPUT_DIM).fit(cal_preds, cal_targets)
    width = cr.interval_width()
    print(f"      learned per-output quantile: {np.asarray(cr.quantile)}")
    print(f"      interval width (per output): {np.asarray(width)}")
    print(f"      n_calibration: {cr.n_calibration}")

    # ----------------------------------------------------------------- 5.
    print("\n[5] Empirical coverage on held-out test set.")
    coverage = np.asarray(cr.coverage(test_preds, test_targets))
    print(f"      target coverage:    1 - α = {1 - ALPHA:.2f}")
    print(f"      empirical (per dim): {coverage}")
    print(f"      empirical (mean):    {float(coverage.mean()):.3f}")
    print("      Coverage is marginal per output, not simultaneous or conditional on acting.")

    # ----------------------------------------------------------------- 6.
    print("\n[6] Selective abstention demo.")
    # Absolute-residual intervals have constant width per output. This
    # separate decision heuristic uses whether the box contains zero;
    # it is not a ranking of calibrated per-example error risk.
    lo, hi = cr.predict_interval(test_preds)
    pred_norm = jnp.linalg.norm(test_preds, axis=-1)
    # A conformal interval is an axis-aligned box, not a Euclidean ball.
    abstain_mask = jnp.all((lo <= 0.0) & (hi >= 0.0), axis=-1)
    n_abstain = int(jnp.sum(abstain_mask))
    n_act = N_TEST - n_abstain
    print("      rule: abstain when every coordinate interval contains zero")
    print(f"      acted on:  {n_act} / {N_TEST} ({n_act / N_TEST:.1%})")
    print(f"      abstained: {n_abstain} / {N_TEST} ({n_abstain / N_TEST:.1%})")

    if 0 < n_act < N_TEST:
        # This heuristic has no selective-risk guarantee; compare realised
        # errors without assuming that the acted-on subset is more accurate.
        kept = ~abstain_mask
        rmse_acted = float(jnp.sqrt(jnp.mean((test_preds[kept] - test_targets[kept]) ** 2)))
        rmse_abstain = float(
            jnp.sqrt(jnp.mean((test_preds[abstain_mask] - test_targets[abstain_mask]) ** 2))
        )
        print(f"      RMSE on acted points: {rmse_acted:.4f}")
        print(f"      RMSE on abstained:    {rmse_abstain:.4f}")
        # These descriptive ratios divide RMSE by mean prediction magnitude;
        # they are not calibrated risks or per-example relative errors.
        rel_acted = rmse_acted / float(jnp.mean(pred_norm[kept]))
        rel_abstain = rmse_abstain / max(float(jnp.mean(pred_norm[abstain_mask])), 1e-8)
        print(f"      relative err (acted):     {rel_acted:.3f}")
        print(f"      relative err (abstained): {rel_abstain:.3f}")
        print("      Subset errors are descriptive; marginal coverage does not bound them.")

    # ----------------------------------------------------------------- 7.
    print("\nThe regression head and predictive intervals are distinct from the")
    print("decision heuristic. Neither marginal intervals nor this synthetic")
    print("example establish safety or an error bound on accepted actions.")


if __name__ == "__main__":
    main()
