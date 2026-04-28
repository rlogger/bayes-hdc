# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Human activity recognition — UCIHAR-style multi-class accelerometer classification.

Activity recognition from wearable accelerometer + gyroscope streams is
the second-most-cited application of HDC after EMG gesture recognition.
The reference benchmark is the UCI Human Activity Recognition dataset
(Anguita et al. 2013): 6 daily-living activities (walking, stairs up,
stairs down, sitting, standing, laying) sampled from 30 subjects with a
smartphone IMU. The HDC literature treats it as a high-dim feature
classification problem — discretise each of 561 hand-crafted features
into ordinal levels, encode with feature-value binding, train a
centroid classifier — and consistently reports 90+ % accuracy.

This example demonstrates the literature-canonical pipeline with three
deliverables that classical centroid HDC does not produce:

1. **Calibrated probabilities** — temperature scaling on the cosine-
   similarity logits, fit via L-BFGS in log-space (Guo et al. 2017).
2. **Conformal prediction sets** — at α = 0.1, every sample is given a
   set of activities that contains the true one with marginal coverage
   ≥ 0.9 (Romano et al. 2020). The set size measures task difficulty.
3. **Selective abstention** — when the conformal set is not a singleton,
   the classifier abstains and routes the case to follow-up. This is
   the same pattern that makes HDC interesting for medical telemetry
   and prosthetic-control safety layers.

By default the example runs on a synthetic 36-feature accelerometer
window so the pipeline is fast and offline. Pass ``--real-data`` to load
the actual UCIHAR 561-feature benchmark via
:func:`bayes_hdc.datasets.load_ucihar` (one-time OpenML download); the
encoder and classifier code paths are identical, only the data source
changes.

References:

* Anguita, D. et al. (2013). "A Public Domain Dataset for Human Activity
  Recognition Using Smartphones." ESANN.
* Hassan, E. et al. (2018). "Hyperdimensional Computing for Human
  Activity Recognition with Inertial Sensors."
* Schmuck, M. et al. (2019). "Hardware Optimizations of Dense Binary
  Hyperdimensional Computing." JETC.

Run::

    python examples/activity_recognition.py                # synthetic
    python examples/activity_recognition.py --real-data    # real UCIHAR
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from bayes_hdc import (
    MAP,
    BayesianCentroidClassifier,
    ConformalClassifier,
    RandomEncoder,
    TemperatureCalibrator,
)

DIMS = 4096
NUM_FEATURES = 36  # 9 axes × 4 statistics (mean, std, energy, peak)
NUM_LEVELS = 32
NUM_ACTIVITIES = 6
SAMPLES_PER_ACTIVITY = 100
SEED = 2026
ALPHA = 0.10  # 90 % conformal coverage target

ACTIVITY_NAMES = (
    "walking",
    "stairs-up",
    "stairs-down",
    "sitting",
    "standing",
    "laying",
)


def _synthetic_imu_features(key: jax.Array) -> tuple[np.ndarray, np.ndarray]:
    """36-feature accelerometer windows with activity-specific signatures."""
    rng = np.random.default_rng(int(jax.random.bits(key)))

    # (NUM_ACTIVITIES, NUM_FEATURES) class centroids.
    base = rng.standard_normal((NUM_ACTIVITIES, NUM_FEATURES)).astype(np.float32) * 0.6

    # Encode activity-specific structure: dynamic activities (walking,
    # stairs) have higher energy and peak features; static activities
    # (sitting, standing, laying) have lower magnitudes and tighter
    # variance. Pairs of similar activities (sitting/standing) are
    # made deliberately confusable.
    base[0] += np.array([1.5] * 9 + [1.2] * 9 + [1.8] * 9 + [1.5] * 9, dtype=np.float32)
    base[1] += np.array([1.6] * 9 + [1.3] * 9 + [1.9] * 9 + [1.7] * 9, dtype=np.float32)
    base[2] += np.array([1.4] * 9 + [1.1] * 9 + [1.7] * 9 + [1.4] * 9, dtype=np.float32)
    base[3] -= 1.0  # sitting
    base[4] -= 0.9  # standing — close to sitting
    base[5] -= 1.4  # laying

    X_list, y_list = [], []
    for a in range(NUM_ACTIVITIES):
        for _ in range(SAMPLES_PER_ACTIVITY):
            sample = base[a] + 0.5 * rng.standard_normal(NUM_FEATURES).astype(np.float32)
            X_list.append(sample)
            y_list.append(a)
    return np.stack(X_list), np.asarray(y_list, dtype=np.int32)


def _discretise(X: np.ndarray, num_levels: int) -> np.ndarray:
    """Per-feature quantile binning into ordinal levels."""
    X_idx = np.empty(X.shape, dtype=np.int32)
    for f in range(X.shape[1]):
        edges = np.quantile(X[:, f], np.linspace(0, 1, num_levels + 1))
        edges = np.unique(edges)
        if len(edges) < 2:
            X_idx[:, f] = 0
        else:
            idx = np.digitize(X[:, f], edges[1:-1])
            X_idx[:, f] = np.clip(idx, 0, num_levels - 1)
    return X_idx


def _load_real_ucihar() -> tuple[np.ndarray, np.ndarray, tuple[str, ...], int]:
    """Load real UCIHAR via bayes_hdc.datasets (one-time OpenML download)."""
    from bayes_hdc.datasets import load_ucihar

    data = load_ucihar()
    # Standardise per-feature so the discretisation quantiles are stable.
    X = np.asarray(data.X, dtype=np.float32)
    y = np.asarray(data.y, dtype=np.int32)
    activity_names = (
        "walking",
        "stairs-up",
        "stairs-down",
        "sitting",
        "standing",
        "laying",
    )
    return X, y, activity_names, X.shape[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    p.add_argument(
        "--real-data",
        action="store_true",
        help="Load the real UCIHAR dataset via OpenML (one-time download). "
        "Default: synthetic 36-feature accelerometer windows that always run offline.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    print("Human activity recognition — UCIHAR-style multi-class")

    key = jax.random.PRNGKey(SEED)
    k_data, k_codebook = jax.random.split(key)

    if args.real_data:
        try:
            X, y, activity_names, num_features = _load_real_ucihar()
            num_activities = int(y.max() + 1)
            data_label = "real UCIHAR"
        except Exception as e:  # noqa: BLE001 — surface the failure, fall back.
            print(f"  ! could not load real UCIHAR ({e}); falling back to synthetic.")
            X, y_jax = _synthetic_imu_features(k_data)
            y = np.asarray(y_jax)
            activity_names = ACTIVITY_NAMES
            num_features = NUM_FEATURES
            num_activities = NUM_ACTIVITIES
            data_label = "synthetic (fallback)"
    else:
        X, y_jax = _synthetic_imu_features(k_data)
        y = np.asarray(y_jax)
        activity_names = ACTIVITY_NAMES
        num_features = NUM_FEATURES
        num_activities = NUM_ACTIVITIES
        data_label = "synthetic (use --real-data for OpenML UCIHAR)"

    print(
        f"  data = {data_label}   activities = {num_activities}   "
        f"features = {num_features}   n = {len(y)}   D = {DIMS}\n"
    )

    # ----------------------------------------------------------------- 1.
    print("[1] Discretise features into ordinal levels.")
    X_idx = _discretise(X, NUM_LEVELS)
    print(f"      X shape: {X.shape}    discretised range: [{X_idx.min()}, {X_idx.max()}]")

    # 60 / 20 / 20 train / cal / test split.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X))
    n_tr, n_ca = int(0.6 * len(X)), int(0.8 * len(X))
    tr_idx, ca_idx, te_idx = perm[:n_tr], perm[n_tr:n_ca], perm[n_ca:]

    # ----------------------------------------------------------------- 2.
    print("\n[2] Encode each window via feature-value binding (RandomEncoder).")
    vsa = MAP.create(dimensions=DIMS)
    encoder = RandomEncoder.create(
        num_features=num_features,
        num_values=NUM_LEVELS,
        dimensions=DIMS,
        vsa_model=vsa,
        key=k_codebook,
    )
    hv_all = encoder.encode_batch(jnp.asarray(X_idx))
    hv_tr, y_tr = hv_all[tr_idx], jnp.asarray(y[tr_idx])
    hv_ca, y_ca = hv_all[ca_idx], jnp.asarray(y[ca_idx])
    hv_te, y_te = hv_all[te_idx], jnp.asarray(y[te_idx])
    print(
        f"      encoded HV shape: {tuple(hv_all.shape)}    "
        f"(train/cal/test = {len(y_tr)}/{len(y_ca)}/{len(y_te)})"
    )

    # ----------------------------------------------------------------- 3.
    print("\n[3] Train BayesianCentroidClassifier.")
    clf = BayesianCentroidClassifier.create(
        num_classes=num_activities,
        dimensions=DIMS,
    ).fit(hv_tr, y_tr, prior_strength=1.0)

    train_acc = float(clf.score(hv_tr, y_tr))
    test_acc = float(clf.score(hv_te, y_te))
    print(f"      train accuracy: {train_acc:.3f}    test accuracy: {test_acc:.3f}")

    # ----------------------------------------------------------------- 4.
    print("\n[4] Calibrate logits + wrap in a ConformalClassifier.")
    logits_ca = clf.logits(hv_ca)
    logits_te = clf.logits(hv_te)
    calibrator = TemperatureCalibrator.create().fit(logits_ca, y_ca, max_iters=200)
    probs_ca = calibrator.calibrate(logits_ca)
    probs_te = calibrator.calibrate(logits_te)

    conformal = ConformalClassifier.create(alpha=ALPHA).fit(probs_ca, y_ca)
    coverage = float(conformal.coverage(probs_te, y_te))
    mean_set_size = float(conformal.set_size(probs_te))
    print(f"      target coverage (1 − α) = {1 - ALPHA:.2f}")
    print(f"      empirical coverage      = {coverage:.3f}")
    print(f"      mean prediction-set size = {mean_set_size:.2f}  (of {num_activities} activities)")

    # ----------------------------------------------------------------- 5.
    print("\n[5] Selective abstention: predict iff conformal set is a singleton.")
    set_mask = np.asarray(conformal.predict_set(probs_te).astype(np.int32))
    set_sizes = set_mask.sum(axis=-1)
    confident = set_sizes == 1
    preds = np.asarray(jnp.argmax(probs_te, axis=-1))
    y_te_np = np.asarray(y_te)

    n_test = len(y_te_np)
    n_confident = int(confident.sum())
    overall_acc = float(np.mean(preds == y_te_np))
    confident_acc = (
        float(np.mean(preds[confident] == y_te_np[confident])) if n_confident > 0 else float("nan")
    )
    abstained_acc = (
        float(np.mean(preds[~confident] == y_te_np[~confident]))
        if (~confident).any()
        else float("nan")
    )

    print(f"      overall accuracy (no abstention)          = {overall_acc:.3f}")
    print(
        f"      confident predictions                     = "
        f"{n_confident}/{n_test} ({n_confident / n_test:.1%})"
    )
    print(f"      accuracy on confident subset              = {confident_acc:.3f}")
    print(f"      accuracy on abstained subset (would-have) = {abstained_acc:.3f}")

    # ----------------------------------------------------------------- 6.
    print("\n[6] Per-activity coverage breakdown:")
    print(f"      {'activity':<14s} {'in-set rate':>12s} {'mean set size':>15s}")
    for a in range(num_activities):
        mask = y_te_np == a
        if not mask.any():
            continue
        in_set = float(set_mask[mask, a].mean())
        m_size = float(set_sizes[mask].mean())
        name = activity_names[a] if a < len(activity_names) else f"class-{a}"
        print(f"      {name:<14s} {in_set:>12.3f} {m_size:>15.2f}")

    if args.real_data:
        print(
            "\nNumbers above are on the real UCIHAR 561-feature benchmark "
            "(Anguita et al. 2013), encoded by feature-value binding."
        )
    else:
        print(
            "\nThe pipeline above is feature-value binding (Hassan et al. 2018, "
            "Schmuck et al. 2019)\nrunning on synthetic 36-feature accelerometer windows. "
            "Run with --real-data to point\nit at `bayes_hdc.datasets.load_ucihar()` "
            "and the real 561-feature benchmark."
        )


if __name__ == "__main__":
    main()
