# SPDX-License-Identifier: MIT
# Copyright (c) 2026 R.S.

"""Split-conformal anomaly detection on independent synthetic samples.

Fits a scoring pipeline on proper training data and calibrates on separate
normal observations. For exchangeable calibration/normal test observations,
conformal p-values are super-uniform and control marginal false-positive
probability. They do not bound every realised test-batch FPR, guarantee
anomaly power, or remain valid under arbitrary temporal dependence or shift.

The tutorial compares a fixed threshold to recalibration, then illustrates
refitting with known-normal samples from each current regime. This is an
oracle simulation, not a guarantee for unlabeled streaming deployment.

Reference: Bates, Candes, Lei, Romano and Sesia (2023), Testing for Outliers
with Conformal p-values, Annals of Statistics 51(1):149-178,
https://arxiv.org/abs/2104.08279.

Run: python tutorials/02_anomaly_detection.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bayes_hdc import (
    ConformalAnomalyDetector,
    HDCAnomalyScorer,
    ProjectionEncoder,
    RandomEncoder,
    fit_anomaly_pipeline,
)

DIMS = 4096
SEED = 2026


# ---------------------------------------------------------------------
# Small shared helpers (no plotting dependency — pure ASCII).
# ---------------------------------------------------------------------


def synth_cluster(key: jax.Array, n: int, centre, scale: float = 0.5) -> jax.Array:
    """A tight 2-D Gaussian cluster. Stands in for one "normal" regime.

    Two dimensions keep the example legible; the same code path works
    for arbitrary feature dimensions — only the encoder's ``input_dim``
    changes.
    """
    centre = jnp.asarray(centre, dtype=jnp.float32)
    return jax.random.normal(key, (n, 2)) * scale + centre


def ascii_hist(values, n_bins: int = 10, width: int = 40, lo: float = 0.0, hi: float = 1.0) -> str:
    """Render a horizontal ASCII histogram of ``values`` over ``[lo, hi]``."""
    values = np.asarray(values)
    edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    peak = max(int(counts.max()), 1)
    lines = []
    for i in range(n_bins):
        bar = "#" * int(round(width * counts[i] / peak))
        lines.append(f"  [{edges[i]:.2f}, {edges[i + 1]:.2f})  {bar:<{width}} {counts[i]:>4d}")
    return "\n".join(lines)


# =====================================================================
# 1. Motivation: why HDC for anomaly detection
# =====================================================================


def section_1_motivation() -> None:
    print("=" * 68)
    print("[1] Why HDC for anomaly detection")
    print("=" * 68)
    print(
        "\nA single-pass centroid supplies a normality score. Separate normal\n"
        "calibration data convert that frozen score to finite-rank p-values.\n"
        "Under exchangeability, P(p <= alpha) <= alpha for a normal test point.\n"
        "This is a marginal statement; ties can make p-values conservative.\n"
        "Realised batch FPRs can exceed alpha, and anomaly recall is not guaranteed.\n"
        "The wrapper is model-independent and can also calibrate non-HDC scores.\n"
    )


# =====================================================================
# 2. Simplest setup + ASCII histogram of holdout p-values
# =====================================================================


def section_2_simplest() -> ProjectionEncoder:
    print("=" * 68)
    print("[2] Simplest setup: scorer + conformal detector")
    print("=" * 68)

    key = jax.random.PRNGKey(SEED)
    k_enc, k_data = jax.random.split(key)

    # Encode the 2-D normal cluster into hypervectors. ProjectionEncoder
    # handles continuous features directly (a random Gaussian projection
    # followed by the VSA's sign/normalisation), so the encoded normal
    # points form a tight cluster with a meaningful centroid.
    encoder = ProjectionEncoder.create(input_dim=2, dimensions=DIMS, vsa_model="map", key=k_enc)

    # Three disjoint splits, all drawn from the SAME normal regime:
    #   - train      : fits the scorer's centroid (proper-training split)
    #   - calibration: fits the conformal p-value distribution
    #   - holdout    : fresh normal data; its p-values should be ~uniform
    k_tr, k_cal, k_hold = jax.random.split(k_data, 3)
    train = synth_cluster(k_tr, 400, centre=[2.0, -1.0])
    calib = synth_cluster(k_cal, 300, centre=[2.0, -1.0])
    holdout = synth_cluster(k_hold, 500, centre=[2.0, -1.0])

    print("\n  Split-conformal outlier protocol (Bates et al. 2023):")
    print("    fit scorer on the train split, fit p-values on calibration,")
    print("    evaluate on a disjoint holdout split.\n")

    scorer = HDCAnomalyScorer.create(dimensions=DIMS, vsa_model="map")
    scorer = scorer.fit(encoder.encode_batch(train))
    detector = ConformalAnomalyDetector.create(scorer)
    detector = detector.fit(encoder.encode_batch(calib))
    print(f"    distance_metric  = {detector.scorer.distance_metric}")
    print(f"    n_calibration    = {detector.n_calibration}")

    holdout_p = np.asarray(detector.pvalue_batch(encoder.encode_batch(holdout)))
    print(f"\n  Holdout p-values: mean = {holdout_p.mean():.3f} (uniform -> 0.50),")
    print(f"                     std  = {holdout_p.std():.3f} (uniform -> 0.289)")
    print("\n  Histogram of holdout p-values (descriptive; ties can make it conservative):\n")
    print(ascii_hist(holdout_p, n_bins=10, width=40))

    # Sanity check: a far-away cluster should be flagged with low p-values.
    k_anom = jax.random.PRNGKey(SEED + 99)
    anomalies = synth_cluster(k_anom, 200, centre=[-2.0, 3.0])
    anom_p = np.asarray(detector.pvalue_batch(encoder.encode_batch(anomalies)))
    recall = float((anom_p <= 0.05).mean())
    print("\n  As a sanity check, a shifted cluster at [-2, 3] gets")
    print(f"  mean p-value {anom_p.mean():.3f} and recall {recall:.2f} at alpha = 0.05.")
    print()
    return encoder


# =====================================================================
# 3. Repeated-split empirical false-positive rates
# =====================================================================


def section_3_coverage() -> None:
    print("=" * 68)
    print("[3] Repeated-split empirical false-positive rates (200 splits)")
    print("=" * 68)

    alpha = 0.1
    n_test = 200
    n_splits = 200
    centre = [2.0, -1.0]

    key = jax.random.PRNGKey(SEED + 1)
    k_enc, k_base = jax.random.split(key)
    encoder = ProjectionEncoder.create(input_dim=2, dimensions=DIMS, vsa_model="map", key=k_enc)

    # Fit the scorer once on a fixed training split; only the calibration
    # and test sets are resampled per split. This isolates the conformal
    # layer's contribution to coverage.
    base = synth_cluster(k_base, 400, centre=centre)
    scorer = HDCAnomalyScorer.create(dimensions=DIMS, vsa_model="map")
    scorer = scorer.fit(encoder.encode_batch(base))

    print(f"\n  alpha = {alpha}    test points/split = {n_test}    splits = {n_splits}")
    print("  Each split draws a fresh calibration set and a fresh (still-")
    print("  normal) test set, then measures the empirical false-positive")
    print("  rate of predict_batch(..., alpha).\n")

    fprs = []
    for t in range(n_splits):
        k_cal, k_test = jax.random.split(jax.random.PRNGKey(10_000 + t))
        calib = synth_cluster(k_cal, 200, centre=centre)
        test = synth_cluster(k_test, n_test, centre=centre)
        detector = ConformalAnomalyDetector.create(scorer)
        detector = detector.fit(encoder.encode_batch(calib))
        flags = np.asarray(detector.predict_batch(encoder.encode_batch(test), alpha=alpha))
        fprs.append(float(flags.mean()))

    fprs = np.asarray(fprs)
    # Test flags share a calibration sample and are dependent marginally.
    # Estimate uncertainty across independent calibration/test replications,
    # rather than using a binomial band for all test flags together.
    se_mean = float(fprs.std(ddof=1) / np.sqrt(n_splits))
    print(f"  mean empirical FPR   = {fprs.mean():.4f}   (marginal target <= {alpha})")
    print(f"  per-split FPR std     = {fprs.std(ddof=1):.4f}")
    print(f"  Monte Carlo SE of the mean across splits = {se_mean:.4f}")
    print("  This simulation describes one score/distribution; it is not a proof.")
    print()


# =====================================================================
# 4. Versus a naive max-z-score threshold (no calibration)
# =====================================================================


def section_4_naive_comparison() -> None:
    print("=" * 68)
    print("[4] Conformal vs a naive max-z-score threshold")
    print("=" * 68)

    alpha = 0.05
    centre = [2.0, -1.0]

    key = jax.random.PRNGKey(SEED + 2)
    k_enc, k_train = jax.random.split(key)
    encoder = ProjectionEncoder.create(input_dim=2, dimensions=DIMS, vsa_model="map", key=k_enc)

    train = synth_cluster(k_train, 400, centre=centre)
    train_hv = encoder.encode_batch(train)
    scorer = HDCAnomalyScorer.create(dimensions=DIMS, vsa_model="map").fit(train_hv)

    # The naive baseline: freeze a z-score threshold from the training
    # scores once (mean + 3*std) and never touch it again. This is the
    # illustrative heuristic; it is not claimed to be a literature baseline.
    train_scores = np.asarray(scorer.score_batch(train_hv))
    naive_thr = float(train_scores.mean() + 3.0 * train_scores.std())
    print("\n  Naive rule: flag if nonconformity score > mean + 3*std")
    print(f"  of the training scores (= {naive_thr:.4f}). Fixed forever.")
    print(f"  Conformal rule: predict_batch(..., alpha = {alpha}), which")
    print("  recalibrates on each window's own calibration set.\n")
    print("  Each spread defines a different normal regime. Conformal gets new")
    print("  known-normal calibration data from that regime; the fixed threshold does not.")
    print("  This contrasts recalibration with staleness, not equally adapted methods.\n")

    print("    spread   naive FPR   conformal FPR")
    print("    " + "-" * 38)
    for i, scale in enumerate([0.5, 0.7, 0.9, 1.1]):
        k_cal, k_test = jax.random.split(jax.random.PRNGKey(20_000 + i))
        calib = synth_cluster(k_cal, 300, centre=centre, scale=scale)
        test = synth_cluster(k_test, 300, centre=centre, scale=scale)

        test_scores = np.asarray(scorer.score_batch(encoder.encode_batch(test)))
        naive_fpr = float((test_scores > naive_thr).mean())

        detector = ConformalAnomalyDetector.create(scorer)
        detector = detector.fit(encoder.encode_batch(calib))
        conf_fpr = float(
            np.asarray(detector.predict_batch(encoder.encode_batch(test), alpha=alpha)).mean()
        )
        print(f"    {scale:>5.1f}    {naive_fpr:>7.3f}     {conf_fpr:>9.3f}")

    print("\n  The naive threshold's FPR drifts upward as the spread grows")
    print("  (its fixed cut no longer matches the score distribution),")
    print(f"  while the conformal FPR stays near alpha = {alpha}. Conformal")
    print("  control here relies on calibration/test samples sharing the current regime.")
    print()


# =====================================================================
# 5. Multi-VSA: MAP, BSC, HRR
# =====================================================================


def section_5_multi_vsa() -> None:
    print("=" * 68)
    print("[5] Empirical rates across VSA models (MAP, BSC, HRR)")
    print("=" * 68)

    alpha = 0.1
    n_test = 300
    centre = [2.0, -1.0]
    print(f"\n  alpha = {alpha}. For each VSA model we fit a scorer +")
    print("  conformal detector and measure the holdout FPR on normal data")
    print("  plus the recall on a shifted anomaly cluster. The detector")
    print("  auto-selects the distance metric: Hamming for BSC, cosine")
    print("  otherwise.\n")

    print("    VSA    metric    holdout FPR   anomaly recall")
    print("    " + "-" * 48)
    for model_index, vsa in enumerate(("map", "bsc", "hrr")):
        key = jax.random.PRNGKey(SEED + model_index)
        k_enc, k_tr, k_cal, k_hold, k_an = jax.random.split(key, 5)
        encoder = ProjectionEncoder.create(input_dim=2, dimensions=DIMS, vsa_model=vsa, key=k_enc)

        train = synth_cluster(k_tr, 400, centre=centre)
        calib = synth_cluster(k_cal, 300, centre=centre)
        holdout = synth_cluster(k_hold, n_test, centre=centre)
        anomalies = synth_cluster(k_an, 200, centre=[-2.0, 3.0])

        scorer = HDCAnomalyScorer.create(dimensions=DIMS, vsa_model=vsa)
        scorer = scorer.fit(encoder.encode_batch(train))
        detector = ConformalAnomalyDetector.create(scorer)
        detector = detector.fit(encoder.encode_batch(calib))

        fpr = float(
            np.asarray(detector.predict_batch(encoder.encode_batch(holdout), alpha=alpha)).mean()
        )
        recall = float(
            np.asarray(detector.predict_batch(encoder.encode_batch(anomalies), alpha=alpha)).mean()
        )
        print(
            f"    {vsa:<5}  {detector.scorer.distance_metric:<8}  {fpr:>9.3f}     {recall:>11.3f}"
        )

    print(f"\n  Marginal FPR target is <= {alpha}; realised rates and recall vary.")
    print("  The same")
    print("  conformal wrapper is VSA-agnostic; only the nonconformity")
    print("  metric changes underneath it.")
    print()


# =====================================================================
# 6. Tabular fraud-style demo with the one-liner pipeline
# =====================================================================


def section_6_fraud() -> None:
    print("=" * 68)
    print("[6] Tabular fraud-style demo (scaler -> KBins -> RandomEncoder)")
    print("=" * 68)

    try:
        from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
    except ImportError:
        print("\n  scikit-learn is not installed; skipping the tabular demo.")
        print("  Install it with `pip install scikit-learn` to run section 6.")
        print()
        return

    rng = np.random.default_rng(SEED)
    n_feat = 6
    n_bins = 12
    alpha = 0.05

    # Correlated "normal" transactions: a shared covariance structure
    # across the six features. Fraud is the same shape, mean-shifted.
    cov = np.eye(n_feat) * 0.6 + 0.4
    chol = np.linalg.cholesky(cov)
    base_mean = np.array([1.0, 2.0, 0.5, -1.0, 3.0, 0.0])
    normal = rng.standard_normal((1600, n_feat)) @ chol.T + base_mean
    fraud = rng.standard_normal((200, n_feat)) @ chol.T + base_mean + 3.5

    n_train, n_cal, n_test = 600, 500, 500
    train = normal[:n_train]
    calib = normal[n_train : n_train + n_cal]
    test_normal = normal[n_train + n_cal : n_train + n_cal + n_test]

    print(f"\n  {n_feat} correlated features; normal vs a +3.5-sigma mean-shifted")
    print(f"  fraud class. n_train = {n_train}, n_cal = {n_cal}, alpha = {alpha}.")
    print("\n  Preprocess: StandardScaler -> KBinsDiscretizer (uniform bins,")
    print(f"  {n_bins} per feature) -> RandomEncoder, then the one-liner")
    print("  fit_anomaly_pipeline does scorer + conformal calibration.\n")

    scaler = StandardScaler().fit(train)
    discretiser = KBinsDiscretizer(
        n_bins=n_bins, encode="ordinal", strategy="uniform", subsample=None
    ).fit(scaler.transform(train))

    def discretise(x: np.ndarray) -> jax.Array:
        binned = discretiser.transform(scaler.transform(x))
        binned = np.clip(binned, 0, n_bins - 1).astype(np.int32)
        return jnp.asarray(binned)

    encoder = RandomEncoder.create(
        num_features=n_feat,
        num_values=n_bins,
        dimensions=DIMS,
        vsa_model="map",
        key=jax.random.PRNGKey(SEED + 3),
    )

    detector = fit_anomaly_pipeline(
        encoder,
        discretise(train),
        discretise(calib),
        alpha=alpha,
    )

    # Encode the evaluation sets and predict. Anomaly = positive class.
    normal_hv = encoder.encode_batch(discretise(test_normal))
    fraud_hv = encoder.encode_batch(discretise(fraud))
    flags_normal = np.asarray(detector.predict_batch(normal_hv, alpha=alpha))
    flags_fraud = np.asarray(detector.predict_batch(fraud_hv, alpha=alpha))

    tp = int(flags_fraud.sum())
    fp = int(flags_normal.sum())
    fn = int((~flags_fraud).sum())
    tn = int((~flags_normal).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    # Coverage here = 1 - empirical FPR on the normal test set; the
    # conformal guarantee targets FPR <= alpha, i.e. coverage >= 1 - alpha.
    empirical_fpr = fp / (fp + tn) if (fp + tn) else 0.0

    print(f"    confusion: TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"    precision  = {precision:.3f}")
    print(f"    recall     = {recall:.3f}")
    print(f"    F1         = {f1:.3f}")
    print(f"    normal FPR = {empirical_fpr:.3f}   (conformal target <= {alpha})")
    print(f"    coverage   = {1.0 - empirical_fpr:.3f}   (target >= {1.0 - alpha:.2f})")
    print("\n  Fraud is recovered at high recall while the false-positive")
    print("  rate on genuine transactions stays close to the conformal")
    print(f"  budget of alpha = {alpha} (it is a single finite-sample draw,")
    print("  so the realised FPR can exceed alpha; test flags share calibration data).")
    print()


# =====================================================================
# 7. Streaming twist: re-fit the detector on a drifting window
# =====================================================================


def section_7_streaming() -> None:
    print("=" * 68)
    print("[7] Streaming twist: tracking a drifting normal regime")
    print("=" * 68)

    # Design note (honest): StreamingBayesianHDC in the library is a
    # *classifier* — it keeps an EMA mean/variance per class for
    # supervised prediction, not a one-class nonconformity score, so it
    # does not drop into ConformalAnomalyDetector without writing a
    # custom scorer. Rather than force that, this section uses the clean,
    # composable approach: re-fit the scorer + conformal detector on a
    # recent window of the (still-normal) stream. Both classes are cheap
    # to rebuild because fit() is a single bundle + sort, so a sliding-
    # window refit is illustrated here using known-normal oracle data.

    key = jax.random.PRNGKey(SEED + 4)
    k_enc, k_base = jax.random.split(key)
    encoder = ProjectionEncoder.create(input_dim=2, dimensions=DIMS, vsa_model="map", key=k_enc)

    # The normal regime ROTATES around the origin over time. With a tight
    # cluster off the origin, that angular drift moves the encoded
    # centroid in cosine space — exactly the regime where a stale
    # detector starts crying wolf.
    print("\n  The normal mean rotates around the origin over time. A stale")
    print("  detector (fit once at t=0) and a window-refit detector (re-fit")
    print("  on the most recent normal window) are both scored on fresh")
    print("  normal data from the current regime. A mean p-value is only descriptive;")
    print("  it is not a probability of normality or a calibration diagnostic by itself.\n")

    base = synth_cluster(k_base, 400, centre=[3.0, 0.0], scale=0.3)
    stale_scorer = HDCAnomalyScorer.create(dimensions=DIMS, vsa_model="map")
    stale_scorer = stale_scorer.fit(encoder.encode_batch(base[:200]))
    stale_detector = ConformalAnomalyDetector.create(stale_scorer)
    stale_detector = stale_detector.fit(encoder.encode_batch(base[200:]))

    trajectory = [[3.0, 0.0], [2.0, 2.0], [0.0, 3.0], [-3.0, 0.0], [0.0, -3.0]]
    print("    regime mean      stale mean-p   window-refit mean-p")
    print("    " + "-" * 52)
    for i, centre in enumerate(trajectory):
        current = synth_cluster(jax.random.PRNGKey(30_000 + i), 300, centre=centre, scale=0.3)

        # Stale detector: never updated since t=0.
        stale_p = float(
            np.asarray(stale_detector.pvalue_batch(encoder.encode_batch(current[150:]))).mean()
        )

        # Window refit: rebuild scorer + detector from the recent window.
        win_scorer = HDCAnomalyScorer.create(dimensions=DIMS, vsa_model="map")
        win_scorer = win_scorer.fit(encoder.encode_batch(current[:75]))
        win_detector = ConformalAnomalyDetector.create(win_scorer)
        win_detector = win_detector.fit(encoder.encode_batch(current[75:150]))
        refit_p = float(
            np.asarray(win_detector.pvalue_batch(encoder.encode_batch(current[150:]))).mean()
        )

        print(f"    {str(centre):<14}   {stale_p:>10.3f}   {refit_p:>17.3f}")

    print("\n  Once the regime rotates away from t=0, the stale detector's")
    print("  mean p-value collapses toward 0 — it now flags the new normal")
    print("  as anomalous. Re-fitting on the recent window restores mean")
    print("  p-value near 0.5: the detector tracks the shift and the")
    print("  current-regime calibration/test draws are exchangeable in this simulation.")
    print("  Arbitrary drift, overlapping windows and unlabelled refits need different analysis.")
    print()


# =====================================================================
# 8. Where next
# =====================================================================


def section_8_where_next() -> None:
    print("=" * 68)
    print("[8] Where next")
    print("=" * 68)
    print(
        "\n"
        "  Applied, domain-specific worked examples:\n\n"
        "    examples/anomaly_detection_sensors.py\n"
        "        Industrial condition monitoring: windowed multi-channel\n"
        "        sensor traces, FFT features, ProjectionEncoder, and a\n"
        "        running p-value sequence showing the 'cliff drop' on\n"
        "        anomaly windows.\n\n"
        "    examples/anomaly_detection_intrusion.py\n"
        "        Network-intrusion-style tabular detection end-to-end\n"
        "        (if present in your checkout; otherwise the sensors\n"
        "        example covers the same pipeline shape).\n\n"
        "  API reference for the classes used here:\n\n"
        "    HDCAnomalyScorer        — distance-metric-aware nonconformity\n"
        "                              score (cosine / Hamming, k-NN).\n"
        "    ConformalAnomalyDetector — split-conformal p-values + FPR-\n"
        "                              bounded predict().\n"
        "    fit_anomaly_pipeline    — encoder -> scorer -> detector in\n"
        "                              one call.\n\n"
        "  Full docs: https://rlogger.github.io/bayes-hdc\n"
    )


# =====================================================================
# Driver
# =====================================================================


def main() -> None:
    print("\nBayes-HDC tutorial 02 — calibrated anomaly detection\n")
    section_1_motivation()
    section_2_simplest()
    section_3_coverage()
    section_4_naive_comparison()
    section_5_multi_vsa()
    section_6_fraud()
    section_7_streaming()
    section_8_where_next()
    print("Tutorial 02 complete. Next: 03_sequences.py.\n")


if __name__ == "__main__":
    main()
