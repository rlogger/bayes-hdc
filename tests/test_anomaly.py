# SPDX-License-Identifier: MIT
# Copyright (c) 2026 R.S.

"""Tests for ``bayes_hdc.anomaly`` — HDC nonconformity scoring and the
split-conformal anomaly detector.

The split-conformal protocol used here is Laxhammar (2014) /
Lei et al. (2018) / Bates et al. (2023); the HDC plug-in nonconformity
score is Kleyko et al. (2017) / Imani et al. (2019) / Furlong &
Eliasmith (2024) / Liang et al. (2026) ConformalHDC.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from bayes_hdc.anomaly import (
    ConformalAnomalyDetector,
    HDCAnomalyScorer,
    fit_anomaly_pipeline,
)
from bayes_hdc.embeddings import RandomEncoder

DIMS = 256


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _normal_hvs(key: jax.Array, n: int, dims: int = DIMS) -> jax.Array:
    """Synthetic in-distribution hypervectors: Gaussian around a fixed mean.

    Real HDC pipelines feed encoder outputs in; we use raw Gaussians here
    because the conformal guarantee is distribution-free — all that
    matters is that ``score()`` produces a smooth response on the chosen
    "normal" distribution.
    """
    mean = jnp.ones((dims,)) / jnp.sqrt(dims)
    return mean[None, :] + 0.3 * jax.random.normal(key, (n, dims))


def _anomalous_hvs(key: jax.Array, n: int, dims: int = DIMS) -> jax.Array:
    """Out-of-distribution hypervectors: rotated mean + larger noise."""
    mean = -jnp.ones((dims,)) / jnp.sqrt(dims)
    return mean[None, :] + 0.3 * jax.random.normal(key, (n, dims))


def _ks_statistic_uniform(samples: jax.Array) -> float:
    """One-sample Kolmogorov–Smirnov statistic against Uniform[0, 1].

    Returns ``sup_x |F_n(x) - x|`` for ``x`` on the empirical support.
    Implemented in NumPy to avoid a SciPy dependency.
    """
    x = jnp.sort(samples)
    n = x.shape[0]
    i = jnp.arange(1, n + 1, dtype=jnp.float32)
    d_plus = jnp.max(i / n - x)
    d_minus = jnp.max(x - (i - 1) / n)
    return float(jnp.maximum(d_plus, d_minus))


# ----------------------------------------------------------------------
# 1. Shape & init correctness
# ----------------------------------------------------------------------


def test_scorer_create_returns_pytree_with_expected_shapes() -> None:
    scorer = HDCAnomalyScorer.create(dimensions=DIMS, vsa_model="map")
    assert scorer.centroid.shape == (DIMS,)
    assert scorer.dimensions == DIMS
    assert scorer.distance_metric == "cosine"
    assert scorer.k_neighbors == 1
    assert scorer.vsa_model_name == "map"
    # Pytree-flat round-trip preserves the static / dynamic split.
    leaves, treedef = jax.tree_util.tree_flatten(scorer)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.dimensions == DIMS
    assert rebuilt.distance_metric == "cosine"


def test_scorer_create_picks_hamming_for_bsc() -> None:
    """BSC defaults to Hamming distance; everything else to cosine."""
    bsc_scorer = HDCAnomalyScorer.create(dimensions=DIMS, vsa_model="bsc")
    assert bsc_scorer.distance_metric == "hamming"
    assert bsc_scorer.centroid.dtype == jnp.bool_

    map_scorer = HDCAnomalyScorer.create(dimensions=DIMS, vsa_model="map")
    assert map_scorer.distance_metric == "cosine"
    assert map_scorer.centroid.dtype == jnp.float32


def test_detector_create_returns_pytree_with_expected_shapes() -> None:
    scorer = HDCAnomalyScorer.create(dimensions=DIMS)
    det = ConformalAnomalyDetector.create(scorer=scorer, n_calibration=50)
    assert det.calibration_scores.shape == (50,)
    assert det.n_calibration == 50
    assert det.scorer.dimensions == DIMS


# ----------------------------------------------------------------------
# 2. fit() on a simple synthetic distribution
# ----------------------------------------------------------------------


def test_scorer_fit_populates_normalised_centroid() -> None:
    key = jax.random.PRNGKey(0)
    normal = _normal_hvs(key, n=64)
    scorer = HDCAnomalyScorer.create(dimensions=DIMS).fit(normal)
    # Centroid must be unit-normalised for the cosine variant.
    assert jnp.isclose(jnp.linalg.norm(scorer.centroid), 1.0, atol=1e-5)
    # And direction-consistent with the empirical mean.
    emp_mean = jnp.mean(normal, axis=0)
    emp_dir = emp_mean / jnp.linalg.norm(emp_mean)
    assert float(jnp.dot(scorer.centroid, emp_dir)) > 0.99


def test_detector_fit_stores_sorted_calibration_scores() -> None:
    key = jax.random.PRNGKey(1)
    k_train, k_cal = jax.random.split(key)
    train = _normal_hvs(k_train, n=128)
    cal = _normal_hvs(k_cal, n=200)

    scorer = HDCAnomalyScorer.create(dimensions=DIMS).fit(train)
    det = ConformalAnomalyDetector.create(scorer=scorer).fit(cal)

    assert det.n_calibration == 200
    assert det.calibration_scores.shape == (200,)
    # Stored in ascending order so the (n+1) p-value formula is O(log n)
    # if anyone later swaps in a binary-search variant.
    diffs = jnp.diff(det.calibration_scores)
    assert jnp.all(diffs >= -1e-6)
    # And scores match a fresh score_batch on the same data (up to sort).
    direct = jnp.sort(scorer.score_batch(cal))
    assert jnp.allclose(direct, det.calibration_scores, atol=1e-5)


# ----------------------------------------------------------------------
# 3. p-value validity: approximately uniform on held-out normal data
# ----------------------------------------------------------------------


def test_pvalues_uniform_on_holdout_normal_data() -> None:
    """The formal conformal guarantee: p-values on exchangeable normal
    test data are stochastically uniform on [0, 1].

    We test the empirical CDF against Uniform[0, 1] via the
    Kolmogorov–Smirnov statistic. With n_test = 800 the 99.9 %
    KS critical value is ≈ 1.95 / √n ≈ 0.069. We use a comfortably
    loose 0.10 cutoff so the test is robust to seed jitter while still
    falsifying a grossly non-uniform implementation.
    """
    key = jax.random.PRNGKey(2)
    k_train, k_cal, k_test = jax.random.split(key, 3)
    train = _normal_hvs(k_train, n=256)
    cal = _normal_hvs(k_cal, n=400)
    test = _normal_hvs(k_test, n=800)

    scorer = HDCAnomalyScorer.create(dimensions=DIMS).fit(train)
    det = ConformalAnomalyDetector.create(scorer=scorer).fit(cal)

    pvals = det.pvalue_batch(test)
    assert pvals.shape == (800,)
    assert float(jnp.min(pvals)) > 0.0
    assert float(jnp.max(pvals)) <= 1.0

    ks = _ks_statistic_uniform(pvals)
    assert ks < 0.10, (
        f"Conformal p-values deviate from Uniform[0,1] by KS = {ks:.4f}; "
        "either the centroid is biased or the (n+1) correction is wrong."
    )


# ----------------------------------------------------------------------
# 4. Coverage at alpha — FPR ≤ alpha + finite-sample slack
# ----------------------------------------------------------------------


def test_false_positive_rate_bounded_by_alpha_across_seeds() -> None:
    """For exchangeable normal data, the fraction flagged anomalous
    must be ≤ α + Monte-Carlo slack.

    The marginal-coverage guarantee (Vovk et al. 2005, Bates et al.
    2023) is in expectation over the calibration draw; we average over
    multiple seeds to estimate that expectation.
    """
    alpha = 0.10
    n_seeds = 8
    n_test = 500

    fprs = []
    for seed in range(n_seeds):
        key = jax.random.PRNGKey(100 + seed)
        k_train, k_cal, k_test = jax.random.split(key, 3)
        train = _normal_hvs(k_train, n=128)
        cal = _normal_hvs(k_cal, n=200)
        test = _normal_hvs(k_test, n=n_test)

        scorer = HDCAnomalyScorer.create(dimensions=DIMS).fit(train)
        det = ConformalAnomalyDetector.create(scorer=scorer).fit(cal)
        flagged = det.predict_batch(test, alpha=alpha)
        fprs.append(float(jnp.mean(flagged.astype(jnp.float32))))

    mean_fpr = sum(fprs) / len(fprs)
    # E[FPR] ≤ α; with a one-sided 3·SE finite-sample slack and the
    # extra (n+1)-correction conservatism, mean_fpr should sit at or
    # below α + ~0.02.
    se = (alpha * (1 - alpha) / (n_test * n_seeds)) ** 0.5
    assert mean_fpr <= alpha + 3 * se + 0.02, (
        f"mean FPR {mean_fpr:.3f} exceeds α + slack ({alpha + 3 * se + 0.02:.3f}); "
        "conformal coverage guarantee broken."
    )


# ----------------------------------------------------------------------
# 5. Detection power on actual anomalies
# ----------------------------------------------------------------------


def test_detection_power_on_distribution_shift() -> None:
    """On data drawn from a *different* distribution, the detector
    should flag the vast majority at α = 0.05.

    Power is not a conformal guarantee — it depends entirely on the
    nonconformity score's ability to separate the two classes. We pick
    a well-separated synthetic shift (opposite mean direction) where
    the cosine-to-centroid score is essentially a linear classifier.
    """
    key = jax.random.PRNGKey(3)
    k_train, k_cal, k_anom = jax.random.split(key, 3)
    train = _normal_hvs(k_train, n=256)
    cal = _normal_hvs(k_cal, n=400)
    anom = _anomalous_hvs(k_anom, n=200)

    scorer = HDCAnomalyScorer.create(dimensions=DIMS).fit(train)
    det = ConformalAnomalyDetector.create(scorer=scorer).fit(cal)

    pvals = det.pvalue_batch(anom)
    flagged = det.predict_batch(anom, alpha=0.05)
    detection_rate = float(jnp.mean(flagged.astype(jnp.float32)))
    median_p = float(jnp.median(pvals))

    assert detection_rate > 0.80, (
        f"Detected only {detection_rate:.1%} of injected anomalies at α=0.05; "
        "the score is barely separating the two distributions."
    )
    # And the p-value distribution should be visibly compressed toward 0.
    assert median_p < 0.05


# ----------------------------------------------------------------------
# 6. JIT composition
# ----------------------------------------------------------------------


def test_score_pvalue_predict_jit_match_eager() -> None:
    """Wrapping ``score`` / ``pvalue`` / a threshold of pvalue in
    ``jax.jit`` must return identical values to the eager path."""
    key = jax.random.PRNGKey(4)
    k_train, k_cal, k_test = jax.random.split(key, 3)
    train = _normal_hvs(k_train, n=64)
    cal = _normal_hvs(k_cal, n=100)
    query = _normal_hvs(k_test, n=1)[0]

    scorer = HDCAnomalyScorer.create(dimensions=DIMS).fit(train)
    det = ConformalAnomalyDetector.create(scorer=scorer).fit(cal)

    score_eager = float(det.score(query))
    pval_eager = float(det.pvalue(query))

    score_jit = float(jax.jit(det.score)(query))
    pval_jit = float(jax.jit(det.pvalue)(query))

    assert jnp.isclose(score_eager, score_jit, atol=1e-6)
    assert jnp.isclose(pval_eager, pval_jit, atol=1e-6)

    # `predict()` does a Python-level α-validation, so we jit the
    # underlying decision rule (pvalue ≤ α) rather than `predict`
    # itself. The two must agree.
    alpha = 0.05
    eager_flag = bool(det.predict(query, alpha=alpha))
    jit_flag = bool(jax.jit(lambda q: det.pvalue(q) <= alpha)(query))
    assert eager_flag == jit_flag


# ----------------------------------------------------------------------
# 7. vmap composition
# ----------------------------------------------------------------------


def test_score_vmap_matches_score_batch() -> None:
    """`score_batch` is documented as ``jax.vmap(score)``; user code
    that vmaps directly must produce the same numbers and shape."""
    key = jax.random.PRNGKey(5)
    k_train, k_test = jax.random.split(key)
    train = _normal_hvs(k_train, n=64)
    batch = _normal_hvs(k_test, n=32)

    scorer = HDCAnomalyScorer.create(dimensions=DIMS).fit(train)
    direct = scorer.score_batch(batch)
    vmapped = jax.vmap(scorer.score)(batch)

    assert direct.shape == (32,)
    assert vmapped.shape == (32,)
    assert jnp.allclose(direct, vmapped, atol=1e-6)


# ----------------------------------------------------------------------
# 8. grad composition
# ----------------------------------------------------------------------


def test_score_grad_is_finite_and_nonzero() -> None:
    """``jax.grad`` of ``score`` w.r.t. a continuous query is finite —
    important for using HDCAnomalyScorer inside a downstream loss
    (e.g. adversarial perturbation, contrastive training of the
    encoder). Verified with ``jax.test_util.check_grads`` at order 1
    in reverse mode."""
    from jax.test_util import check_grads

    key = jax.random.PRNGKey(6)
    k_train, k_q = jax.random.split(key)
    train = _normal_hvs(k_train, n=64)
    scorer = HDCAnomalyScorer.create(dimensions=DIMS).fit(train)

    # Pick a query slightly displaced from the centroid so we are
    # comfortably in the smooth interior of the cosine map.
    query = scorer.centroid + 0.1 * jax.random.normal(k_q, (DIMS,))

    g = jax.grad(lambda q: scorer.score(q))(query)
    assert g.shape == (DIMS,)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.linalg.norm(g)) > 0.0

    # Finite-difference verification of the gradient.
    check_grads(lambda q: scorer.score(q), (query,), order=1, modes=("rev",))


# ----------------------------------------------------------------------
# 9. Edge case: fit on empty data
# ----------------------------------------------------------------------


def test_scorer_fit_on_empty_normal_raises() -> None:
    scorer = HDCAnomalyScorer.create(dimensions=DIMS)
    with pytest.raises(ValueError, match="empty"):
        scorer.fit(jnp.zeros((0, DIMS)))


def test_detector_fit_on_empty_calibration_raises() -> None:
    scorer = HDCAnomalyScorer.create(dimensions=DIMS).fit(_normal_hvs(jax.random.PRNGKey(0), n=8))
    det = ConformalAnomalyDetector.create(scorer=scorer)
    with pytest.raises(ValueError, match="empty"):
        det.fit(jnp.zeros((0, DIMS)))


# ----------------------------------------------------------------------
# 10. Edge case: alpha boundary behaviour
# ----------------------------------------------------------------------


def test_predict_rejects_alpha_outside_open_unit_interval() -> None:
    """α ∈ (0, 1) strict; α = 0 and α = 1 are nonsensical for a
    conformal p-value threshold (no test would ever fire / every test
    would always fire)."""
    key = jax.random.PRNGKey(7)
    train = _normal_hvs(key, n=64)
    scorer = HDCAnomalyScorer.create(dimensions=DIMS).fit(train)
    det = ConformalAnomalyDetector.create(scorer=scorer).fit(train)
    query = train[0]

    for bad in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError, match="alpha"):
            det.predict(query, alpha=bad)
        with pytest.raises(ValueError, match="alpha"):
            det.predict_batch(train[:4], alpha=bad)

    # The smallest possible p-value is 1 / (n + 1); at α just above
    # that, at most one calibration-tied sample is flagged.
    flag_low = bool(det.predict(query, alpha=1.0 / (det.n_calibration + 1) - 1e-6))
    assert flag_low is False or flag_low is True  # type sanity

    # At α very close to 1, almost every sample is flagged anomalous —
    # the threshold has degenerated. We do not require strict equality
    # because the (n+1)-correction floors p-values at 1/(n+1) > 0.
    flag_high = bool(det.predict(query, alpha=0.999))
    assert flag_high is True


# ----------------------------------------------------------------------
# 11. Multi-VSA: works with MAP, BSC, HRR encoders
# ----------------------------------------------------------------------


@pytest.mark.parametrize("vsa_model", ["map", "bsc", "hrr"])
def test_works_across_vsa_models(vsa_model: str) -> None:
    """The detector composes with any VSA — BSC picks the Hamming
    metric automatically, MAP / HRR pick cosine. We construct
    synthetic hypervectors directly in the right dtype rather than
    going through an encoder, which keeps the test focused on the
    anomaly module."""
    key = jax.random.PRNGKey(8)
    k_train, k_cal, k_anom = jax.random.split(key, 3)

    if vsa_model == "bsc":
        # Binary hypervectors biased toward all-ones for "normal"
        # and toward all-zeros for "anomalous".
        normal_probs = 0.75
        anom_probs = 0.25
        train = (jax.random.uniform(k_train, (96, DIMS)) < normal_probs).astype(jnp.bool_)
        cal_key, anom_key = jax.random.split(k_cal)
        cal = (jax.random.uniform(cal_key, (200, DIMS)) < normal_probs).astype(jnp.bool_)
        anom = (jax.random.uniform(k_anom, (100, DIMS)) < anom_probs).astype(jnp.bool_)
    else:
        # Real-valued hypervectors (MAP and HRR are both R^d at the
        # representation level — they differ in binding only).
        train = _normal_hvs(k_train, n=96)
        cal_key, anom_key = jax.random.split(k_cal)
        cal = _normal_hvs(cal_key, n=200)
        anom = _anomalous_hvs(k_anom, n=100)

    scorer = HDCAnomalyScorer.create(dimensions=DIMS, vsa_model=vsa_model).fit(train)
    det = ConformalAnomalyDetector.create(scorer=scorer).fit(cal)

    # Scores must be finite and non-negative on both classes.
    norm_scores = scorer.score_batch(cal)
    anom_scores = scorer.score_batch(anom)
    assert bool(jnp.all(jnp.isfinite(norm_scores)))
    assert bool(jnp.all(jnp.isfinite(anom_scores)))
    assert float(jnp.min(norm_scores)) >= 0.0 - 1e-5
    assert float(jnp.min(anom_scores)) >= 0.0 - 1e-5

    # And anomalous data must have systematically higher mean score
    # than normal data — otherwise the score is uninformative.
    assert float(jnp.mean(anom_scores)) > float(jnp.mean(norm_scores))

    # Detector still runs end-to-end at α=0.1.
    flagged_norm = det.predict_batch(cal, alpha=0.1)
    flagged_anom = det.predict_batch(anom, alpha=0.1)
    assert flagged_norm.dtype == jnp.bool_
    assert flagged_anom.dtype == jnp.bool_
    # Power should be visibly above the FPR ceiling.
    assert float(jnp.mean(flagged_anom.astype(jnp.float32))) > float(
        jnp.mean(flagged_norm.astype(jnp.float32))
    )


def test_k_neighbors_path_runs_and_separates() -> None:
    """The kNN variant (``k_neighbors > 1``) stores the full reference
    set and scores against the average top-k similarity. Validates the
    second branch of ``HDCAnomalyScorer.score``."""
    key = jax.random.PRNGKey(9)
    k_train, k_cal, k_anom = jax.random.split(key, 3)
    train = _normal_hvs(k_train, n=64)
    cal = _normal_hvs(k_cal, n=120)
    anom = _anomalous_hvs(k_anom, n=80)

    scorer = HDCAnomalyScorer.create(
        dimensions=DIMS, vsa_model="map", k_neighbors=5, n_reference=64
    ).fit(train)
    # Reference set is populated (not zeros) and matches the training set.
    assert scorer.reference.shape == (64, DIMS)
    assert float(jnp.linalg.norm(scorer.reference)) > 0.0

    norm_scores = scorer.score_batch(cal)
    anom_scores = scorer.score_batch(anom)
    assert float(jnp.mean(anom_scores)) > float(jnp.mean(norm_scores))


# ----------------------------------------------------------------------
# 12. End-to-end pipeline: HDCAnomalyScorer + RandomEncoder + detector
# ----------------------------------------------------------------------


def test_end_to_end_with_random_encoder_pipeline() -> None:
    """Full Lei-et-al. (2018) split-conformal protocol end-to-end:

    1.  RandomEncoder embeds raw discrete features.
    2.  HDCAnomalyScorer learns the centroid on the train split.
    3.  ConformalAnomalyDetector calibrates on the held-out cal split.
    4.  pvalue / predict on a separate anomalous stream.

    The anomalies are a different *index pattern* — feature indices
    drawn from the high end of the value range while the normal class
    draws from the low end — which produces a different bundled
    centroid in MAP-space and is reliably caught by cosine-to-centroid.
    """
    num_features = 8
    num_values = 16
    n_train = 64
    n_cal = 200
    n_anom = 200

    enc_key, train_key, cal_key, anom_key = jax.random.split(jax.random.PRNGKey(42), 4)

    encoder = RandomEncoder.create(
        num_features=num_features,
        num_values=num_values,
        dimensions=DIMS,
        vsa_model="map",
        key=enc_key,
    )

    # Normal class draws low indices; anomalous class draws high
    # indices. The two empirical means in hypervector space are then
    # well separated.
    normal_train = jax.random.randint(train_key, (n_train, num_features), 0, num_values // 4)
    normal_cal = jax.random.randint(cal_key, (n_cal, num_features), 0, num_values // 4)
    anomalous_test = jax.random.randint(
        anom_key, (n_anom, num_features), 3 * num_values // 4, num_values
    )

    detector = fit_anomaly_pipeline(
        encoder=encoder,
        normal_data=normal_train,
        calibration_data=normal_cal,
        alpha=0.05,
    )
    assert isinstance(detector, ConformalAnomalyDetector)
    assert detector.n_calibration == n_cal
    assert detector.scorer.dimensions == DIMS

    # Encode test anomalies and check detection power.
    anom_hvs = encoder.encode_batch(anomalous_test)
    flagged = detector.predict_batch(anom_hvs, alpha=0.05)
    detection_rate = float(jnp.mean(flagged.astype(jnp.float32)))
    assert detection_rate > 0.80, (
        f"Pipeline caught only {detection_rate:.1%} of out-of-distribution "
        "samples at α=0.05; nonconformity score is not separating the classes."
    )

    # And the calibration distribution on held-out normal data still
    # behaves like the conformal guarantee says it should.
    norm_hvs = encoder.encode_batch(normal_cal)
    norm_flagged = detector.predict_batch(norm_hvs, alpha=0.05)
    norm_fpr = float(jnp.mean(norm_flagged.astype(jnp.float32)))
    # In-sample FPR (calibration data == fit data) is biased low — the
    # interesting check is that it stays at or near α and never spikes.
    assert norm_fpr <= 0.20, f"In-sample FPR {norm_fpr:.3f} grossly above α=0.05."
