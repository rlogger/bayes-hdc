# SPDX-License-Identifier: MIT
# Copyright (c) 2026 R.S.

"""Synthetic network-style anomaly detection with split-conformal p-values.

Train the scaler, discretiser and score on normal training flows; calibrate
on independent normal flows drawn from the same fixed mixture. The guarantee
is a marginal false-positive probability <= alpha under exchangeability,
not a bound on every realised test-batch rate or arbitrary traffic shift.
The generated features are illustrative and are not NSL-KDD/CICIDS data.

Reference: Bates, Candes, Lei, Romano and Sesia (2023), Testing for Outliers
with Conformal p-values, Annals of Statistics 51(1):149-178,
https://arxiv.org/abs/2104.08279.

Run: python examples/anomaly_detection_intrusion.py
"""

from __future__ import annotations

import jax
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

from bayes_hdc import MAP, RandomEncoder, fit_anomaly_pipeline

# ----------------------------------------------------------------------
# Configuration.
# ----------------------------------------------------------------------

DIMS = 8192
SEED = 2026
N_FEATURES = 10  # NSL-KDD-style numeric flow features
N_BINS = 16  # discretisation levels per feature → RandomEncoder values

# n_calib >= 1/alpha is a practical requirement: the smallest attainable
# conformal p-value is 1 / (n_calib + 1), so n_calib must be large
# enough that the floor sits below alpha or no flow can ever be flagged.
N_TRAIN = 600  # normal-only proper-training split (scorer centroid)
N_CAL = 400  # normal-only calibration split (p-value distribution)
N_TEST_NORMAL = 400  # normal test flows — measures the empirical FPR
N_PER_ATTACK = 120  # test flows per attack family

# alpha = 0.05 — flag a flow when its conformal p-value is <= 0.05. The
# guarantee is a marginal false-positive rate <= alpha on normal
# traffic exchangeable with the calibration split.
ALPHA = 0.05

# Three attack families, NSL-KDD's coarse taxonomy. Each is a distinct
# perturbation of the normal feature distribution.
ATTACK_NAMES = ("dos", "probe", "r2l")


# ----------------------------------------------------------------------
# Synthetic NSL-KDD-style traffic generator.
#
# "Normal" traffic is a mixture of a few correlated Gaussian blobs in a
# 10-D feature space (think: a handful of benign service profiles —
# http, dns, smtp — each a tight cluster with feature correlations).
# Attacks are distinct mean-shifted and/or variance-inflated clusters:
#   * dos   — high-volume floods: large mean shift on the "rate" /
#             "count" features, low variance (very regular).
#   * probe — port / host scans: moderate mean shift spread across many
#             features, inflated variance (scattered exploratory flows).
#   * r2l   — remote-to-local: a near-normal blob with a sharp shift on
#             a couple of features (stealthy — the hard case, low
#             expected recall).
# ----------------------------------------------------------------------


def _normal_traffic(rng: np.random.Generator, n: int, loadings: np.ndarray) -> np.ndarray:
    """Sample ``n`` benign flows as a mixture of correlated Gaussians."""
    # Three benign service profiles, each a correlated Gaussian blob.
    means = np.array(
        [
            [0.2, -0.4, 0.1, 0.5, -0.2, 0.3, -0.1, 0.0, 0.2, -0.3],
            [-0.5, 0.3, -0.2, -0.1, 0.4, -0.3, 0.2, 0.1, -0.4, 0.2],
            [0.4, 0.5, 0.3, -0.4, -0.5, 0.1, 0.4, -0.2, 0.1, 0.5],
        ],
        dtype=np.float64,
    )
    weights = np.array([0.5, 0.3, 0.2])
    # A shared low-rank correlation structure across features so the
    # blobs are not axis-aligned — encoded HDC similarity then has to
    # cope with genuine feature dependence, as in real flow records.

    comp = rng.choice(len(weights), size=n, p=weights)
    latent = rng.standard_normal((n, 3))
    base = means[comp] + latent @ loadings.T
    jitter = 0.18 * rng.standard_normal((n, N_FEATURES))
    return (base + jitter).astype(np.float32)


def _attack_traffic(
    rng: np.random.Generator, n: int, family: str, loadings: np.ndarray
) -> np.ndarray:
    """Sample ``n`` malicious flows for one attack family."""
    base = _normal_traffic(rng, n, loadings)
    shift = np.zeros((1, N_FEATURES), dtype=np.float32)
    scale = np.ones((1, N_FEATURES), dtype=np.float32)

    if family == "dos":
        # Volumetric flood: large, regular shift on a few "rate" features.
        shift[0, [0, 3, 5]] = np.array([3.2, 3.8, 2.6], dtype=np.float32)
        scale[0, [0, 3, 5]] = 0.25  # very low variance — machine-regular
    elif family == "probe":
        # Scan: moderate shift smeared across many features, high spread.
        shift[0, :] = 1.4
        scale[0, :] = 2.2  # inflated variance — scattered exploration
    elif family == "r2l":
        # Stealthy remote-to-local: near-normal except two sharp features.
        shift[0, [2, 7]] = np.array([2.4, -2.2], dtype=np.float32)
        scale[0, [2, 7]] = 0.6
    else:  # pragma: no cover - guarded by the ATTACK_NAMES tuple
        raise ValueError(f"unknown attack family: {family!r}")

    return (base * scale + shift).astype(np.float32)


def synthesise_dataset(
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build all splits.

    Returns ``(train, calib, test_feats, test_labels, attack_ids)``:

    * ``train``  — ``(N_TRAIN, N_FEATURES)`` normal-only flows.
    * ``calib``  — ``(N_CAL, N_FEATURES)`` normal-only flows.
    * ``test_feats`` — normal + attack flows, shuffled.
    * ``test_labels`` — 0 = normal, 1 = attack.
    * ``attack_ids`` — -1 for normal, else index into ``ATTACK_NAMES``.
    """
    rng = np.random.default_rng(seed)

    # Freeze the population covariance before drawing independent splits.
    loadings = rng.standard_normal((N_FEATURES, 3)) * 0.35
    train = _normal_traffic(rng, N_TRAIN, loadings)
    calib = _normal_traffic(rng, N_CAL, loadings)

    test_chunks = [_normal_traffic(rng, N_TEST_NORMAL, loadings)]
    labels = [np.zeros(N_TEST_NORMAL, dtype=np.int32)]
    attack_ids = [np.full(N_TEST_NORMAL, -1, dtype=np.int32)]

    for a_id, family in enumerate(ATTACK_NAMES):
        test_chunks.append(_attack_traffic(rng, N_PER_ATTACK, family, loadings))
        labels.append(np.ones(N_PER_ATTACK, dtype=np.int32))
        attack_ids.append(np.full(N_PER_ATTACK, a_id, dtype=np.int32))

    test_feats = np.concatenate(test_chunks, axis=0)
    test_labels = np.concatenate(labels, axis=0)
    test_attack = np.concatenate(attack_ids, axis=0)

    perm = rng.permutation(test_feats.shape[0])
    return train, calib, test_feats[perm], test_labels[perm], test_attack[perm]


# ----------------------------------------------------------------------
# Preprocessing: StandardScaler -> KBinsDiscretizer (ordinal indices).
# Both are fit on the normal-training split only; calibration and test
# splits are transformed with the same statistics so the splits stay
# exchangeable under the null (the prerequisite for the conformal
# guarantee).
# ----------------------------------------------------------------------


def fit_preprocessor(train: np.ndarray) -> tuple[StandardScaler, KBinsDiscretizer]:
    """Fit the scaler + discretiser on normal training flows."""
    scaler = StandardScaler().fit(train)
    discretiser = KBinsDiscretizer(
        n_bins=N_BINS,
        encode="ordinal",  # integer bin indices, not one-hot
        strategy="quantile",  # equal-mass bins on the normal distribution
        subsample=None,
    )
    discretiser.fit(scaler.transform(train))
    return scaler, discretiser


def to_indices(
    feats: np.ndarray,
    scaler: StandardScaler,
    discretiser: KBinsDiscretizer,
) -> np.ndarray:
    """Standardise then discretise into ``(n, N_FEATURES)`` int indices."""
    standardised = scaler.transform(feats)
    indices = discretiser.transform(standardised)
    # Quantile bins are learned on the normal split; out-of-distribution
    # attack values can fall outside the outermost edges. KBinsDiscretizer
    # already clips to [0, n_bins - 1], but cast to int32 for the encoder.
    return np.clip(indices, 0, N_BINS - 1).astype(np.int32)


# ----------------------------------------------------------------------
# Reporting helpers.
# ----------------------------------------------------------------------


def _histogram(values: np.ndarray, n_bins: int = 10, width: int = 40) -> None:
    """ASCII histogram of values in [0, 1] — used for the p-value dist."""
    counts, edges = np.histogram(values, bins=n_bins, range=(0.0, 1.0))
    peak = max(int(counts.max()), 1)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        bar = "#" * int(round(width * counts[b] / peak))
        print(f"      [{lo:.2f}, {hi:.2f}) | {counts[b]:>4d} | {bar}")


def main() -> None:
    print("Network-intrusion anomaly detection — HDC + split-conformal p-values")
    print(f"  d = {DIMS}    features = {N_FEATURES}    bins = {N_BINS}    alpha = {ALPHA}\n")

    key = jax.random.PRNGKey(SEED)

    # ----------------------------------------------------------------- 1.
    print("[1] Synthesise NSL-KDD-style flow records (normal + attacks).")
    train, calib, test_feats, test_labels, test_attack = synthesise_dataset(SEED)
    n_attack = int((test_labels == 1).sum())
    n_normal_test = int((test_labels == 0).sum())
    print(
        f"      train (normal) = {train.shape}\n"
        f"      calib (normal) = {calib.shape}\n"
        f"      test  (mixed)  = {test_feats.shape}    "
        f"({n_normal_test} normal / {n_attack} attack)"
    )
    for a_id, family in enumerate(ATTACK_NAMES):
        print(f"        {family:>6s}: {int((test_attack == a_id).sum())} flows")

    # ----------------------------------------------------------------- 2.
    print("\n[2] StandardScaler -> KBinsDiscretizer (fit on normal train only).")
    scaler, discretiser = fit_preprocessor(train)
    train_idx = to_indices(train, scaler, discretiser)
    calib_idx = to_indices(calib, scaler, discretiser)
    test_idx = to_indices(test_feats, scaler, discretiser)
    print(
        f"      ordinal index range: [{int(train_idx.min())}, "
        f"{int(train_idx.max())}]   shape: {train_idx.shape}"
    )

    # ----------------------------------------------------------------- 3.
    print("\n[3] Build RandomEncoder over the discretised features.")
    encoder = RandomEncoder.create(
        num_features=N_FEATURES,
        num_values=N_BINS,
        dimensions=DIMS,
        vsa_model=MAP.create(dimensions=DIMS),
        key=key,
    )
    print(f"      {N_FEATURES} features x {N_BINS} values -> d = {DIMS} (MAP)")

    # ----------------------------------------------------------------- 4.
    print("\n[4] Fit the split-conformal anomaly detector (normal-only).")
    # fit_anomaly_pipeline encodes both splits, fits the HDCAnomalyScorer
    # centroid on the proper-training split, and calibrates the
    # ConformalAnomalyDetector p-value distribution on the disjoint
    # calibration split. Imported from bayes_hdc — never defined here.
    detector = fit_anomaly_pipeline(
        encoder,
        normal_data=train_idx,
        calibration_data=calib_idx,
        alpha=ALPHA,
    )
    p_floor = 1.0 / (detector.n_calibration + 1)
    print(
        f"      scorer: {detector.scorer.distance_metric} distance, "
        f"centroid d = {detector.scorer.dimensions}\n"
        f"      n_calibration = {detector.n_calibration}    "
        f"p-value floor = 1/(n+1) = {p_floor:.5f}"
    )
    if p_floor > ALPHA:
        print(
            f"      warning: p-value floor {p_floor:.5f} > alpha {ALPHA}; "
            "increase N_CAL so a flow can be flagged at all."
        )

    # ----------------------------------------------------------------- 5.
    print("\n[5] Calibration self-ranks (descriptive only).")
    # Self-scored calibration p-values are a sanity check, not the
    # guarantee. Ranking the same observations used for calibration is
    # largely tautological. Fresh exchangeable normal points have
    # super-uniform p-values; ties can make their distribution conservative.
    calib_hvs = encoder.encode_batch(calib_idx)
    calib_p = np.asarray(detector.pvalue_batch(calib_hvs))
    _histogram(calib_p)
    print(
        f"      mean = {calib_p.mean():.3f}   "
        f"median = {np.median(calib_p):.3f}   "
        f"(uniform reference: mean ~ 0.5)"
    )

    # ----------------------------------------------------------------- 6.
    print("\n[6] Test-set p-values, flags, and the observed FPR on NORMAL traffic.")
    test_hvs = encoder.encode_batch(test_idx)
    test_p = np.asarray(detector.pvalue_batch(test_hvs))
    test_flags = np.asarray(detector.predict_batch(test_hvs, alpha=ALPHA))

    normal_mask = test_labels == 0
    attack_mask = test_labels == 1
    fpr = float(test_flags[normal_mask].mean())
    print(
        f"      target FPR (= alpha)     : {ALPHA:.3f}\n"
        f"      observed FPR on normal   : {fpr:.3f}   "
        f"({int(test_flags[normal_mask].sum())} / {n_normal_test} flagged)"
    )
    print("      Marginal control does not bound every realised test-batch FPR.")
    overall_recall = float(test_flags[attack_mask].mean())
    print(
        f"      overall attack recall    : {overall_recall:.3f}   "
        f"({int(test_flags[attack_mask].sum())} / {n_attack} caught)"
    )

    # ----------------------------------------------------------------- 7.
    print("\n[7] Detection power (recall at alpha) per attack family.")
    print("      family | flows | recall | median p | mean p")
    print("      ------ + ----- + ------ + -------- + ------")
    for a_id, family in enumerate(ATTACK_NAMES):
        fam_mask = test_attack == a_id
        n_fam = int(fam_mask.sum())
        recall = float(test_flags[fam_mask].mean())
        med_p = float(np.median(test_p[fam_mask]))
        mean_p = float(test_p[fam_mask].mean())
        print(f"      {family:>6s} | {n_fam:>5d} | {recall:>6.3f} | {med_p:>8.5f} | {mean_p:>6.4f}")

    # ----------------------------------------------------------------- 8.
    print(
        "\nThese are synthetic flow records, not an intrusion benchmark.\n"
        "With the frozen scorer, independent calibration, and exchangeable\n"
        "normal test flows, p-values control marginal false-positive probability.\n"
        "A realised test-batch FPR can exceed alpha. Real traffic needs a\n"
        "time/group-aware evaluation and separate checks for distribution shift."
    )


if __name__ == "__main__":
    main()
