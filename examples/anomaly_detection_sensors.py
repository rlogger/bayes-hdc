# SPDX-License-Identifier: MIT
# Copyright (c) 2026 R.S.

"""Industrial-sensor anomaly detection with HDC + split-conformal p-values.

In a real deployment, swap the generator for your SCADA / OPC-UA /
vibration-sensor loader. The pipeline assumes short windowed
multi-channel sensor traces with stationary "normal" statistics during
the calibration phase -- the standard condition-monitoring assumption,
and the prerequisite for the distribution-free conformal anomaly
guarantee.

End-to-end pipeline (all library, no hand-rolled detector):

    raw 8-channel windows
        -> FFT + per-channel summary-stat features (~40 / window)
        -> sklearn StandardScaler  (fit on normal training data)
        -> bayes_hdc.ProjectionEncoder  (MAP VSA, random projection)
        -> bayes_hdc.fit_anomaly_pipeline(...)
             == HDCAnomalyScorer.fit(normal)  +  ConformalAnomalyDetector.fit(calib)

The detector is imported from :mod:`bayes_hdc`; this example does not
define one locally. The score is the cosine-distance-to-centroid
nonconformity measure (Furlong & Eliasmith 2024), conformalised with
the split-conformal protocol (Lei et al. 2018; Laxhammar 2014;
Bates et al. 2023) into a p-value with a finite-sample false-positive
guarantee. Anomalous windows pin near the conformal p-value floor
1 / (n_calib + 1); normal windows spread across (0, 1].

References:
  * Lei, G'Sell, Rinaldo, Tibshirani, Wasserman (2018) JASA 113(523)
    -- split-conformal predictive inference.
  * Laxhammar (2014) Conformal Anomaly Detection; Bates, Candes, Lei,
    Romano (2023) Testing for Outliers with Conformal p-Values
    -- conformal anomaly / outlier p-values.
  * Furlong & Eliasmith (2024) Probabilistic Hyperdimensional Computing
    -- the HDC nonconformity score.
  * Liang et al. (2026) ConformalHDC -- the conformalisation choice.

Run::

    python examples/anomaly_detection_sensors.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.preprocessing import StandardScaler

from bayes_hdc import ProjectionEncoder, fit_anomaly_pipeline

# ----------------------------------------------------------------------
# Configuration.
# ----------------------------------------------------------------------

NUM_CHANNELS = 8
WINDOW_LENGTH = 64
# Per-channel mean + std (16) plus per-channel coarse log-FFT energy
# (3 bins x 8 channels = 24). 40 features per window. Per-channel bins
# (not channel-averaged) preserve a single-channel frequency drift.
NUM_FFT_BINS_PER_CHANNEL = 3
NUM_STAT_FEATURES = 2 * NUM_CHANNELS
NUM_SPEC_FEATURES = NUM_FFT_BINS_PER_CHANNEL * NUM_CHANNELS
NUM_FEATURES = NUM_STAT_FEATURES + NUM_SPEC_FEATURES  # 40
DIMS = 8192

# n_calib >= 1/alpha is a practical requirement: the smallest attainable
# conformal p-value is the floor 1 / (n_calib + 1).
NUM_NORMAL_TRAIN = 240
NUM_NORMAL_CALIB = 200
NUM_NORMAL_TEST = 60
NUM_ANOMALY_TEST = 20

# alpha = 0.01 -- industrial monitoring tolerates ~1 % steady-state
# false-positive rate. The conformal guarantee is on calibration-test
# exchangeability of the normal component; anomalies legitimately
# receive low p-values.
ALPHA = 0.01
SEED = 2026


# ----------------------------------------------------------------------
# Synthetic sensor generator. Mimics rotating-machinery vibration: a
# shared carrier frequency across channels with correlated background
# noise; anomalies are amplitude + frequency deviations in one channel.
# ----------------------------------------------------------------------


def _generate_window(rng: np.random.Generator, anomaly: bool) -> np.ndarray:
    """One ``(NUM_CHANNELS, WINDOW_LENGTH)`` sensor window."""
    t = np.arange(WINDOW_LENGTH) / WINDOW_LENGTH

    # Shared carrier: 6 cycles per window on every channel, small
    # per-channel phase offset (a rotating element seen by every sensor).
    carrier_freq = 6.0
    phases = rng.uniform(0.0, 2 * np.pi, size=NUM_CHANNELS)
    carrier = 0.30 * np.sin(2 * np.pi * carrier_freq * t[None, :] + phases[:, None])

    # Correlated background: a single latent factor mixed into every
    # channel plus independent per-channel jitter.
    latent = rng.standard_normal(WINDOW_LENGTH)
    mix = rng.uniform(0.4, 0.9, size=NUM_CHANNELS)
    background = 0.10 * (mix[:, None] * latent[None, :])
    background += 0.05 * rng.standard_normal((NUM_CHANNELS, WINDOW_LENGTH))

    window = (carrier + background).astype(np.float32)

    if anomaly:
        # Single-channel amplitude jump (~10x carrier) plus frequency
        # drift to 17 cycles / window -- the bearing-spall / rotor-
        # imbalance failure signature: localised, narrowband, energy-
        # elevated.
        ch = int(rng.integers(NUM_CHANNELS))
        anom_phase = rng.uniform(0.0, 2 * np.pi)
        window[ch] += (3.5 * np.sin(2 * np.pi * 17.0 * t + anom_phase)).astype(np.float32)

    return window


def synthesise_dataset(
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(train_windows, calib_windows, test_windows, test_labels)``.

    Labels: 0 = normal, 1 = true anomaly. The test split is shuffled so
    anomaly indices are not contiguous.
    """
    rng = np.random.default_rng(seed)
    train = np.stack([_generate_window(rng, False) for _ in range(NUM_NORMAL_TRAIN)])
    calib = np.stack([_generate_window(rng, False) for _ in range(NUM_NORMAL_CALIB)])
    test_windows = [_generate_window(rng, False) for _ in range(NUM_NORMAL_TEST)]
    test_windows += [_generate_window(rng, True) for _ in range(NUM_ANOMALY_TEST)]
    test_labels = np.array([0] * NUM_NORMAL_TEST + [1] * NUM_ANOMALY_TEST, dtype=np.int32)
    perm = rng.permutation(len(test_labels))
    return train, calib, np.stack(test_windows)[perm], test_labels[perm]


# ----------------------------------------------------------------------
# Feature extraction. Cheap, interpretable, ~40-dim. Deployment can swap
# in mel-cepstrum, wavelet packets, learned encoders, etc.
# ----------------------------------------------------------------------


def _per_channel_stats(window: np.ndarray) -> np.ndarray:
    return np.concatenate([window.mean(axis=-1), window.std(axis=-1)], axis=-1)


def _spectral_summary(window: np.ndarray) -> np.ndarray:
    """Per-channel log-FFT energy in ``NUM_FFT_BINS_PER_CHANNEL`` bins."""
    n_freq = WINDOW_LENGTH // 2 + 1
    edges = np.linspace(0, n_freq, NUM_FFT_BINS_PER_CHANNEL + 1, dtype=int)
    out = np.empty((NUM_CHANNELS, NUM_FFT_BINS_PER_CHANNEL), dtype=np.float32)
    for c in range(NUM_CHANNELS):
        spectrum = np.abs(np.fft.rfft(window[c]))
        for b in range(NUM_FFT_BINS_PER_CHANNEL):
            out[c, b] = spectrum[edges[b] : edges[b + 1]].mean()
    return np.log1p(out).reshape(-1)


def extract_features(windows: np.ndarray) -> np.ndarray:
    """``(N, NUM_FEATURES)`` feature matrix from ``(N, C, T)`` windows."""
    feats = np.empty((windows.shape[0], NUM_FEATURES), dtype=np.float32)
    for i, w in enumerate(windows):
        feats[i, :NUM_STAT_FEATURES] = _per_channel_stats(w)
        feats[i, NUM_STAT_FEATURES:] = _spectral_summary(w)
    return feats


# ----------------------------------------------------------------------
# Reporting helpers.
# ----------------------------------------------------------------------


def _print_flag_table(p_values: np.ndarray, flags: np.ndarray, labels: np.ndarray) -> None:
    print("    idx |     p-value | flagged | truth")
    print("    --- + ----------- + ------- + -----")
    for i, (p, f, y) in enumerate(zip(p_values, flags, labels)):
        marker = "  TP" if f and y else "  FP" if f else "  FN" if y else ""
        print(
            f"    {i:>3d} | {p:>11.5f} | {'YES' if f else ' no':>7s} | "
            f"{'anom' if y else 'norm':>4s}{marker}"
        )


def _precision_recall(flags: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    tp = int(((flags == 1) & (labels == 1)).sum())
    fp = int(((flags == 1) & (labels == 0)).sum())
    fn = int(((flags == 0) & (labels == 1)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return p, r, (2 * p * r) / max(p + r, 1e-12)


def main() -> None:
    print("Industrial-sensor anomaly detection -- HDC + split-conformal")
    print(
        f"  channels = {NUM_CHANNELS}   window length = {WINDOW_LENGTH}   "
        f"features = {NUM_FEATURES}   D = {DIMS}   alpha = {ALPHA}\n"
    )

    # ----------------------------------------------------------------- 1.
    print("[1] Synthesise multi-channel sensor windows.")
    train_w, calib_w, test_w, test_y = synthesise_dataset(SEED)
    print(
        f"      train (normal)  = {train_w.shape}\n"
        f"      calib (normal)  = {calib_w.shape}\n"
        f"      test  (mixed)   = {test_w.shape}    "
        f"({int((test_y == 0).sum())} normal / {int((test_y == 1).sum())} anomaly)"
    )

    # ----------------------------------------------------------------- 2.
    print("\n[2] Extract features (per-channel mean/std + coarse FFT bins).")
    train_f = extract_features(train_w)
    calib_f = extract_features(calib_w)
    test_f = extract_features(test_w)
    print(
        f"      feature shapes: train={train_f.shape}  calib={calib_f.shape}  test={test_f.shape}"
    )

    # ----------------------------------------------------------------- 3.
    print("\n[3] Standardise on training statistics (sklearn StandardScaler).")
    scaler = StandardScaler().fit(train_f)
    train_s = scaler.transform(train_f).astype(np.float32)
    calib_s = scaler.transform(calib_f).astype(np.float32)
    test_s = scaler.transform(test_f).astype(np.float32)

    # ----------------------------------------------------------------- 4.
    print("\n[4] Build the ProjectionEncoder (MAP VSA, random projection).")
    encoder = ProjectionEncoder.create(
        input_dim=NUM_FEATURES,
        dimensions=DIMS,
        vsa_model="map",
        key=jax.random.PRNGKey(SEED),
    )
    print(
        f"      projection matrix = {tuple(encoder.projection_matrix.shape)}   "
        f"vsa = {encoder.vsa_model_name}"
    )

    # ----------------------------------------------------------------- 5.
    # One library call does the whole split-conformal protocol:
    #   - encode_batch(normal)  -> HDCAnomalyScorer.fit  (centroid)
    #   - encode_batch(calib)   -> ConformalAnomalyDetector.fit
    #                              (empirical nonconformity distribution)
    # `normal` and `calib` are disjoint normal splits, so the marginal
    # FPR <= alpha guarantee holds.
    print("\n[5] Fit the conformal anomaly pipeline on normal-only data.")
    detector = fit_anomaly_pipeline(
        encoder,
        jnp.asarray(train_s),
        jnp.asarray(calib_s),
        alpha=ALPHA,
    )
    n_calib = detector.n_calibration
    cal = np.asarray(detector.calibration_scores)
    print(
        f"      n_calib = {n_calib}    "
        f"calib-score range = [{cal.min():.4f}, {cal.max():.4f}]    "
        f"p-value floor = {1.0 / (n_calib + 1):.5f}"
    )

    # ----------------------------------------------------------------- 6.
    print("\n[6] Score test windows and produce conformal p-values.\n")
    test_hv = encoder.encode_batch(jnp.asarray(test_s))
    p_vals = np.asarray(detector.pvalue_batch(test_hv))
    flags = np.asarray(detector.predict_batch(test_hv, alpha=ALPHA)).astype(np.int32)
    truth = np.asarray(test_y)

    _print_flag_table(p_vals, flags, truth)

    precision, recall, f1 = _precision_recall(flags, truth)
    n_anom = int((truth == 1).sum())
    n_flagged = int(flags.sum())
    n_flagged_anom = int(((flags == 1) & (truth == 1)).sum())
    n_flagged_norm = int(((flags == 1) & (truth == 0)).sum())
    print(
        f"\n  summary (alpha = {ALPHA})\n"
        f"    windows total       : {len(truth)}\n"
        f"    windows true anomaly: {n_anom}\n"
        f"    windows flagged     : {n_flagged}   "
        f"(anom={n_flagged_anom}, norm={n_flagged_norm})\n"
        f"    precision           : {precision:.3f}\n"
        f"    recall              : {recall:.3f}\n"
        f"    F1                  : {f1:.3f}"
    )

    # ----------------------------------------------------------------- 7.
    # Running p-value series in time order. The conformal guarantee says:
    # on normal windows, p-values are sub-uniform on (0, 1]; on anomalies,
    # p-values "cliff" toward the floor. Walking the sequence makes that
    # visible to the operator.
    print("\n[7] Running conformal p-value series (chronological order).")
    print("    window   |  p-value   |  truth")
    print("    -------- + ---------- + ------")
    for i, (p, y) in enumerate(zip(p_vals, truth)):
        label = "ANOMALY" if y else "normal "
        bar_len = int(min(40, max(1, round(p * 40))))
        bar = "#" * bar_len
        print(f"    win {i:>3d}  | {p:>10.5f} | {label}  {bar}")

    print(
        "\n    Read the column: p-values for normal windows spread across "
        f"(0, 1]; anomalies pin at the floor 1 / (n_calib + 1) = "
        f"{1.0 / (n_calib + 1):.5f}."
    )


if __name__ == "__main__":
    main()
