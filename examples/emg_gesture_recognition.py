# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""EMG gesture recognition — the canonical modern HDC application.

Hand-gesture recognition from surface electromyography (sEMG) is the
single most-cited application of HDC since Rahimi et al. (2016). Multi-
channel EMG is a natural fit for the channel-position-binding encoder:
each electrode contributes a hypervector that is bound with a
discretised level of its measured amplitude, and the per-channel
bindings are bundled into a single window-level hypervector that
classifies the gesture. The pipeline is fast, low-power, and runs on
microcontrollers — exactly what the prosthetics, wearables, and
neuromorphic-edge communities want.

This example demonstrates the literature-canonical pipeline:

1. Window the EMG stream (here: 200-sample windows across 8 channels).
2. Compute a per-channel summary statistic (RMS amplitude).
3. Discretise each channel's RMS into ``L`` ordinal levels.
4. Encode the window: bundle over channels of
   ``bind(channel_hv[c], level_hv[level_c])``.
5. Train a :class:`~bayes_hdc.BayesianCentroidClassifier` and report
   per-gesture accuracy, calibrated probabilities, and posterior variance.

The data here is synthetic so the example runs deterministically and
without a network. The encoding pipeline and classifier are
literature-faithful — pointing
:func:`bayes_hdc.datasets.load_emg` at the same pipeline runs on the
real Rahimi-et-al EMG dataset (``EMG_data_for_gestures`` on OpenML).

References:

* Rahimi, A. et al. (2016). "Hyperdimensional Biosignal Processing."
* Burrello, A. et al. (2018). "Laelaps: Energy-Efficient Seizure Detection."
* Hersche, M. et al. (2019). "Exploring Embedding Methods in Binary
  Hyperdimensional Computing."

Run::

    python examples/emg_gesture_recognition.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bayes_hdc import (
    MAP,
    BayesianCentroidClassifier,
    RandomEncoder,
    TemperatureCalibrator,
)

DIMS = 4096
NUM_CHANNELS = 8
NUM_LEVELS = 16
NUM_GESTURES = 4
WINDOWS_PER_GESTURE = 80
WINDOW_LENGTH = 200
SEED = 2026

GESTURE_NAMES = ("rest", "fist", "open-hand", "wave-out")


def _synthetic_emg(key: jax.Array) -> tuple[np.ndarray, np.ndarray]:
    """Synthesise multi-channel EMG-like windows with gesture-specific profiles.

    Each gesture activates a different subset of channels at a different
    amplitude. The resulting RMS-per-channel signature is what the HDC
    encoder picks up — exactly the structure real sEMG exhibits.
    """
    rng = np.random.default_rng(int(jax.random.bits(key)))

    # Gesture-specific amplitude profile per channel (NUM_GESTURES, NUM_CHANNELS).
    # Profiles deliberately overlap: real sEMG has crosstalk between adjacent
    # electrodes and inter-subject variability, so a perfectly separable
    # 4-class signature would be unrealistic.
    profiles = np.array(
        [
            [0.20, 0.20, 0.18, 0.18, 0.20, 0.18, 0.20, 0.20],  # rest
            [0.55, 0.50, 0.30, 0.25, 0.30, 0.30, 0.45, 0.50],  # fist
            [0.30, 0.25, 0.55, 0.50, 0.50, 0.55, 0.30, 0.25],  # open-hand
            [0.40, 0.30, 0.30, 0.55, 0.50, 0.30, 0.30, 0.40],  # wave-out
        ],
        dtype=np.float32,
    )

    X_list, y_list = [], []
    for g in range(NUM_GESTURES):
        amplitudes = profiles[g]
        for _ in range(WINDOWS_PER_GESTURE):
            # Per-window amplitude jitter (subject / electrode-placement noise)
            # plus per-sample Gaussian noise (background motor-unit firing).
            jitter = 1.0 + 0.30 * rng.standard_normal(NUM_CHANNELS).astype(np.float32)
            jitter = np.clip(jitter, 0.4, 1.6)
            window = (
                amplitudes[:, None]
                * jitter[:, None]
                * rng.standard_normal((NUM_CHANNELS, WINDOW_LENGTH)).astype(np.float32)
            )
            # Background noise floor.
            window = window + 0.05 * rng.standard_normal(window.shape).astype(np.float32)
            X_list.append(window)
            y_list.append(g)

    return np.stack(X_list), np.asarray(y_list, dtype=np.int32)


def _rms_per_channel(windows: np.ndarray) -> np.ndarray:
    """RMS amplitude per channel for each window. Shape: (N, C)."""
    return np.sqrt(np.mean(windows**2, axis=-1))


def _discretise(rms: np.ndarray, num_levels: int) -> np.ndarray:
    """Map per-channel RMS to ordinal levels [0, num_levels)."""
    edges = np.linspace(rms.min(), rms.max() + 1e-6, num_levels + 1)
    levels = np.digitize(rms, edges[1:-1])
    return np.clip(levels, 0, num_levels - 1).astype(np.int32)


def main() -> None:
    print("EMG gesture recognition — the canonical modern HDC application")
    print(
        f"  channels = {NUM_CHANNELS}   gestures = {NUM_GESTURES}   "
        f"windows/gesture = {WINDOWS_PER_GESTURE}   D = {DIMS}\n"
    )

    key = jax.random.PRNGKey(SEED)
    k_data, k_codebook = jax.random.split(key)

    # ----------------------------------------------------------------- 1.
    print("[1] Window the EMG stream and compute RMS per channel.")
    X, y = _synthetic_emg(k_data)
    rms = _rms_per_channel(X)
    print(f"      windows shape: {X.shape}    RMS shape: {rms.shape}")

    # ----------------------------------------------------------------- 2.
    print("\n[2] Discretise each channel's RMS into ordinal levels.")
    indices = _discretise(rms, NUM_LEVELS)
    print(f"      level indices shape: {indices.shape}   range: [{indices.min()}, {indices.max()}]")

    # 60 / 20 / 20 train / cal / test split, stratified-ish (round-robin).
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(indices))
    n = len(perm)
    n_tr, n_ca = int(0.6 * n), int(0.8 * n)
    tr_idx, ca_idx, te_idx = perm[:n_tr], perm[n_tr:n_ca], perm[n_ca:]

    # ----------------------------------------------------------------- 3.
    print("\n[3] Encode each window via channel-position binding + bundle.")
    vsa = MAP.create(dimensions=DIMS)
    encoder = RandomEncoder.create(
        num_features=NUM_CHANNELS,
        num_values=NUM_LEVELS,
        dimensions=DIMS,
        vsa_model=vsa,
        key=k_codebook,
    )
    hv_all = encoder.encode_batch(jnp.asarray(indices))
    hv_tr, y_tr = hv_all[tr_idx], jnp.asarray(y[tr_idx])
    hv_ca, y_ca = hv_all[ca_idx], jnp.asarray(y[ca_idx])
    hv_te, y_te = hv_all[te_idx], jnp.asarray(y[te_idx])
    print(
        f"      encoded HV shape: {tuple(hv_all.shape)}    "
        f"(train/cal/test = {len(y_tr)}/{len(y_ca)}/{len(y_te)})"
    )

    # ----------------------------------------------------------------- 4.
    print("\n[4] Train BayesianCentroidClassifier — per-gesture Gaussian posteriors.")
    clf = BayesianCentroidClassifier.create(
        num_classes=NUM_GESTURES,
        dimensions=DIMS,
    ).fit(hv_tr, y_tr, prior_strength=1.0)

    train_acc = float(clf.score(hv_tr, y_tr))
    test_acc = float(clf.score(hv_te, y_te))
    per_class_uncertainty = np.asarray(jnp.mean(clf.predict_uncertainty(hv_te), axis=0))
    print(f"      train accuracy: {train_acc:.3f}    test accuracy: {test_acc:.3f}")
    print("      mean posterior similarity-variance per gesture (test set):")
    for g in range(NUM_GESTURES):
        print(f"        {GESTURE_NAMES[g]:<10s}  {per_class_uncertainty[g]:.5f}")

    # ----------------------------------------------------------------- 5.
    print("\n[5] Calibrate probabilities (Guo et al. 2017).")
    logits_ca = clf.logits(hv_ca)
    logits_te = clf.logits(hv_te)
    calibrator = TemperatureCalibrator.create().fit(logits_ca, y_ca, max_iters=200)
    probs_te = calibrator.calibrate(logits_te)
    preds = np.asarray(jnp.argmax(probs_te, axis=-1))
    confidence = np.asarray(jnp.max(probs_te, axis=-1))
    print(f"      fitted temperature: {float(calibrator.temperature):.4f}")
    print(f"      mean top-1 calibrated probability: {confidence.mean():.3f}")

    # ----------------------------------------------------------------- 6.
    print("\n[6] Per-gesture confusion (calibrated argmax):")
    print("      " + " ".join(f"{n:>10s}" for n in GESTURE_NAMES))
    for true_g in range(NUM_GESTURES):
        mask = np.asarray(y_te) == true_g
        if not mask.any():
            continue
        row = []
        for pred_g in range(NUM_GESTURES):
            count = int(((preds == pred_g) & mask).sum())
            row.append(f"{count:>10d}")
        print(f"  {GESTURE_NAMES[true_g]:<6s}" + " ".join(row))

    print(
        "\nThe encoder used here is the literature-canonical "
        "channel-position binding from Rahimi et al. 2016. The same pipeline,"
        "\npointed at `bayes_hdc.datasets.load_emg()`, runs on the real "
        "EMG_data_for_gestures benchmark (one-time OpenML download)."
    )


if __name__ == "__main__":
    main()
