# SPDX-License-Identifier: MIT
# Copyright (c) 2026 R.S.

"""A short tour of distribution-valued hypervectors and calibration.

Independent Gaussian binding has exact first two product moments; the product
itself is not Gaussian. Bundling applies a plug-in normalisation and expected
cosine is an approximation. These representations do not track dependence
created by reusing operands, so this example uses independent operands.

The classifier estimates class moments, not posterior uncertainty in its mean.
Training, temperature fitting, conformal calibration and test evaluation use
independent draws from the same mixture.

Run: python examples/pvsa_quickstart.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from bayes_hdc import (
    BayesianCentroidClassifier,
    ConformalClassifier,
    GaussianHV,
    TemperatureCalibrator,
    bind_gaussian,
    bundle_gaussian,
    expected_cosine_similarity,
    similarity_variance,
)

DIMS = 4096


def main() -> None:
    print(f"bayes-hdc PVSA quick-start  —  D = {DIMS}\n")

    # -------------------------------------------------------------- 1.
    print("[1/7] Construct a Gaussian hypervector")
    key = jax.random.PRNGKey(0)
    x = GaussianHV.random(key, DIMS, var=0.01)
    print(
        f"      x.mu.shape={tuple(x.mu.shape)},  ||x.mu|| = {float(jnp.linalg.norm(x.mu)):.3f},"
        f"  mean var = {float(jnp.mean(x.var)):.4f}"
    )

    # -------------------------------------------------------------- 2.
    print("\n[2/7] bind_gaussian propagates moments exactly")
    y = GaussianHV.random(jax.random.fold_in(key, 1), DIMS, var=0.01)
    z = bind_gaussian(x, y)
    print(
        f"      z.mu = x.mu * y.mu           (by construction)\n"
        f"      z.var = x.mu^2 * y.var + y.mu^2 * x.var + x.var * y.var\n"
        f"      mean(z.var) = {float(jnp.mean(z.var)):.6f}"
    )

    # -------------------------------------------------------------- 3.
    print("\n[3/7] bundle_gaussian: independent variances sum, followed by plug-in normalisation")
    w = GaussianHV.random(jax.random.fold_in(key, 2), DIMS, var=0.01)
    stacked = GaussianHV(
        mu=jnp.stack([x.mu, y.mu, w.mu]),
        var=jnp.stack([x.var, y.var, w.var]),
        dimensions=DIMS,
    )
    bundled = bundle_gaussian(stacked)
    print(
        f"      ||bundled.mu|| = {float(jnp.linalg.norm(bundled.mu)):.3f}  (unit sphere)\n"
        f"      mean(bundled.var) = {float(jnp.mean(bundled.var)):.6f}"
    )

    # -------------------------------------------------------------- 4.
    print("\n[4/7] Approximate cosine mean + exact independent dot-product variance")
    sim = float(expected_cosine_similarity(x, y))
    var_sim = float(similarity_variance(x, y))
    print(f"      approx E[cos(x, y)]    = {sim:+.3f}")
    print(f"      Var[<x, y>]     = {var_sim:.6f}")

    # -------------------------------------------------------------- 5.
    print("\n[5/7] Lift a deterministic HV to PVSA with from_sample(var=0)")
    classical = x.mu  # pretend this came from classical HDC
    as_pvsa = GaussianHV.from_sample(classical)
    print(
        f"      as_pvsa.var all zero? {bool(jnp.all(as_pvsa.var == 0.0))}  "
        "(Dirac — deterministic VSA is the zero-variance limit of PVSA)"
    )

    # -------------------------------------------------------------- 6.
    print("\n[6/7] BayesianCentroidClassifier — per-class Gaussian moments")
    k = 4
    centre_key, train_key, temp_key, cal_key, test_key = jax.random.split(
        jax.random.PRNGKey(100), 5
    )
    centres = jax.random.normal(centre_key, (k, DIMS))
    centres = centres / jnp.linalg.norm(centres, axis=-1, keepdims=True)

    def sample_mixture(sample_key, n):
        label_key, noise_key = jax.random.split(sample_key)
        labels = jax.random.randint(label_key, (n,), 0, k)
        return centres[labels] + 0.05 * jax.random.normal(noise_key, (n, DIMS)), labels

    train_hvs, train_labels = sample_mixture(train_key, 120)
    temp_hvs, temp_labels = sample_mixture(temp_key, 80)
    cal_hvs, cal_labels = sample_mixture(cal_key, 100)
    test_hvs, test_labels = sample_mixture(test_key, 150)
    clf = BayesianCentroidClassifier.create(num_classes=k, dimensions=DIMS).fit(
        train_hvs,
        train_labels,
    )
    probs = clf.predict_proba(train_hvs)
    uncertainty = clf.predict_uncertainty(train_hvs)
    print(
        f"      Train accuracy:         {float(clf.score(train_hvs, train_labels)):.3f}\n"
        f"      Mean top-1 probability: {float(jnp.mean(jnp.max(probs, axis=-1))):.3f}\n"
        f"      Mean per-class sim var: {float(jnp.mean(uncertainty)):.5f}"
    )

    # -------------------------------------------------------------- 7.
    print("\n[7/7] ConformalClassifier — separate held-out calibration and evaluation")
    logits_cal = clf.logits(cal_hvs)
    logits_test = clf.logits(test_hvs)
    calibrator = TemperatureCalibrator.create().fit(
        clf.logits(temp_hvs), temp_labels, max_iters=200
    )
    probs_cal = calibrator.calibrate(logits_cal)
    probs_test = calibrator.calibrate(logits_test)

    conformal = ConformalClassifier.create(alpha=0.1).fit(probs_cal, cal_labels)
    coverage = float(conformal.coverage(probs_test, test_labels))
    set_size = float(conformal.set_size(probs_test))
    print(
        f"      Conformal α = 0.1  →  empirical coverage = {coverage:.3f}  "
        f"(marginal target 0.900 under exchangeability)\n"
        f"      Mean prediction-set size = {set_size:.2f}"
    )

    print(
        "\nThe independent calibration/test draws support marginal coverage; a single run may vary."
    )
    print("See DESIGN.md for the design rationale and BENCHMARKS.md for numbers.")


if __name__ == "__main__":
    main()
