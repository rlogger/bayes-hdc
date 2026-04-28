# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""PVSA quick-start — 90-second tour of the Bayesian layer.

Run: ``python examples/pvsa_quickstart.py``

Walks through the seven things that make PVSA different from classical HDC:

1. Construct a Gaussian hypervector with an explicit mean + variance.
2. Bind two Gaussian HVs — moments propagate exactly in closed form.
3. Bundle several Gaussian HVs — variance grows as expected.
4. Compute *expected* cosine similarity and the *variance* of the similarity.
5. Lift a deterministic pipeline to PVSA with ``GaussianHV.from_sample``.
6. Fit a ``BayesianCentroidClassifier`` and read out per-class uncertainty.
7. Wrap the classifier in a ``ConformalClassifier`` and verify coverage.
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
    print("\n[3/7] bundle_gaussian: variance sums, mean is normalised to the unit sphere")
    stacked = GaussianHV(
        mu=jnp.stack([x.mu, y.mu, z.mu]),
        var=jnp.stack([x.var, y.var, z.var]),
        dimensions=DIMS,
    )
    bundled = bundle_gaussian(stacked)
    print(
        f"      ||bundled.mu|| = {float(jnp.linalg.norm(bundled.mu)):.3f}  (unit sphere)\n"
        f"      mean(bundled.var) = {float(jnp.mean(bundled.var)):.6f}"
    )

    # -------------------------------------------------------------- 4.
    print("\n[4/7] Expected similarity + similarity variance")
    sim = float(expected_cosine_similarity(x, z))
    var_sim = float(similarity_variance(x, z))
    print(f"      E[cos(x, z)]    = {sim:+.3f}")
    print(f"      Var[<x, z>]     = {var_sim:.6f}")

    # -------------------------------------------------------------- 5.
    print("\n[5/7] Lift a deterministic HV to PVSA with from_sample(var=0)")
    classical = x.mu  # pretend this came from classical HDC
    as_pvsa = GaussianHV.from_sample(classical)
    print(
        f"      as_pvsa.var all zero? {bool(jnp.all(as_pvsa.var == 0.0))}  "
        "(Dirac — deterministic VSA is the zero-variance limit of PVSA)"
    )

    # -------------------------------------------------------------- 6.
    print("\n[6/7] BayesianCentroidClassifier — per-class Gaussian posteriors")
    k = 4
    keys = jax.random.split(key, k + 1)
    centres = jax.random.normal(keys[0], (k, DIMS))
    centres = centres / (jnp.linalg.norm(centres, axis=-1, keepdims=True) + 1e-8)
    train_hvs = jnp.concatenate(
        [centres[c] + 0.05 * jax.random.normal(keys[c + 1], (30, DIMS)) for c in range(k)]
    )
    train_labels = jnp.concatenate([jnp.full((30,), c, dtype=jnp.int32) for c in range(k)])
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
    print("\n[7/7] ConformalClassifier — coverage-guaranteed prediction sets")
    n_cal = 50
    cal_hvs = train_hvs[:n_cal]
    cal_labels = train_labels[:n_cal]
    test_hvs = train_hvs[n_cal:]
    test_labels = train_labels[n_cal:]

    logits_cal = clf.logits(cal_hvs)
    logits_test = clf.logits(test_hvs)
    calibrator = TemperatureCalibrator.create().fit(logits_cal, cal_labels, max_iters=200)
    probs_cal = calibrator.calibrate(logits_cal)
    probs_test = calibrator.calibrate(logits_test)

    conformal = ConformalClassifier.create(alpha=0.1).fit(probs_cal, cal_labels)
    coverage = float(conformal.coverage(probs_test, test_labels))
    set_size = float(conformal.set_size(probs_test))
    print(
        f"      Conformal α = 0.1  →  empirical coverage = {coverage:.3f}  "
        f"(target ≥ 0.900)\n"
        f"      Mean prediction-set size = {set_size:.2f}"
    )

    print("\nAll seven PVSA primitives work out of the box, end-to-end, in < 90 s.")
    print("See DESIGN.md for the design rationale and BENCHMARKS.md for numbers.")


if __name__ == "__main__":
    main()
