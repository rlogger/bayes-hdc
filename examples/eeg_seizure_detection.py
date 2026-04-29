# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""EEG seizure detection with HDC.

Multi-channel intracranial / scalp EEG seizure detection is the third-
most-cited application cluster in Kleyko et al. (2023) Part II Table 10
after EMG gesture recognition (Table 9) and behavioural / activity
recognition (Table 8). The reference HDC pipeline is a sliding window
of multi-channel EEG, a symbolic per-channel encoding of band-power or
local-binary-pattern features, channel-position binding, bundle, and
classify (Burrello, Schindler, Benini & Rahimi 2018-2021; Asgarinejad
et al. 2020).

This example demonstrates the literature-canonical pipeline:

1. window the EEG stream (here: 256-sample windows across 8 channels);
2. compute a per-channel summary statistic (band-power proxy: log-RMS);
3. discretise each channel's statistic into ``L`` ordinal levels;
4. encode the window: bundle over channels of
   ``bind(channel_hv[c], level_hv[level_c])``;
5. train a :class:`~bayes_hdc.BayesianCentroidClassifier` and report
   detection accuracy, calibrated probabilities, and conformal
   prediction sets at α = 0.1 (target 90 % marginal coverage).

The data here is synthetic so the example runs deterministically and
without a network. Two classes are simulated to mimic the seizure-
detection structure of real iEEG: an "interictal" baseline with
broadband low-amplitude activity, and an "ictal" class with
high-amplitude rhythmic spikes concentrated in a subset of channels —
the structure all of the cited iEEG-HDC papers exploit. The encoding
pipeline and classifier are literature-faithful.

References:

* Burrello, A., Cavigelli, L., Schindler, K., Benini, L., Rahimi, A.
  (2018-2021). Multiple papers on the hyperdimensional EEG seizure
  detection pipeline; see Kleyko et al. 2023 Part II Table 10 refs
  [200]-[204] for the line.
* Asgarinejad, F. et al. (2020). Detection of epileptic seizures from
  iEEG signals with hyperdimensional computing.
* Kleyko, D. et al. (2023). A Survey on HDC aka VSA, Part II.
  ACM Computing Surveys 55(9): Article 175.

Run::

    python examples/eeg_seizure_detection.py
"""

from __future__ import annotations

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
NUM_CHANNELS = 8
NUM_LEVELS = 16
WINDOW_LENGTH = 256
WINDOWS_PER_CLASS = 120  # 240 total
SEED = 2026
ALPHA = 0.10  # 90 % conformal coverage target

CLASS_NAMES = ("interictal", "ictal")


def _synthetic_eeg(key: jax.Array) -> tuple[np.ndarray, np.ndarray]:
    """Synthesise multi-channel EEG-like windows with a binary class label.

    Interictal baseline: broadband Gaussian noise with low per-channel
    amplitude. Ictal spike: a subset of channels (4 of 8) develops a
    high-amplitude rhythmic component (a 6 Hz-equivalent sinusoid for
    256-sample windows) plus elevated background noise. The resulting
    log-RMS-per-channel signature is the structure HDC iEEG pipelines
    pick up; it mirrors the "rhythmic high-amplitude activity localised
    to a seizure focus" pattern reported in the cited line.
    """
    rng = np.random.default_rng(int(jax.random.bits(key)))

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    # Interictal: broadband noise, all channels. Per-window background-
    # amplitude jitter mimics real EEG: the difficulty for the
    # classifier is that ~10 % of interictal windows have transient
    # high-amplitude artefacts that look superficially seizure-like.
    for _ in range(WINDOWS_PER_CLASS):
        background = 0.30 + 0.10 * rng.standard_normal()
        window = background * rng.standard_normal((NUM_CHANNELS, WINDOW_LENGTH)).astype(np.float32)
        # Occasional artefact channel that elevates RMS without rhythmicity.
        if rng.uniform() < 0.10:
            artefact_ch = int(rng.integers(0, NUM_CHANNELS))
            window[artefact_ch] *= 2.0
        X_list.append(window)
        y_list.append(0)

    # Ictal: a subset of channels (3-4 of 8) develops a rhythmic spike with
    # variable amplitude. Some windows have weak seizure activity (so the
    # classifier doesn't get a 100 % free pass); some have strong activity
    # in many channels.
    t = np.arange(WINDOW_LENGTH) / WINDOW_LENGTH
    candidate_seizure = np.array([0, 1, 2, 6, 7])
    for _ in range(WINDOWS_PER_CLASS):
        background = 0.30 + 0.10 * rng.standard_normal()
        window = background * rng.standard_normal((NUM_CHANNELS, WINDOW_LENGTH)).astype(np.float32)
        # Pick 3-4 of the 5 candidate seizure channels each window.
        n_active = int(rng.integers(3, 5))
        sz_chs = rng.choice(candidate_seizure, size=n_active, replace=False)
        for ch in sz_chs:
            phase = rng.uniform(0, 2 * np.pi)
            # Amplitude jitter — high-mean amplitude for the rhythmic spike,
            # with enough variation that the weak end of the distribution
            # is genuinely difficult.
            amp = 0.85 + 0.25 * rng.standard_normal()
            amp = max(amp, 0.30)
            window[ch] += (amp * np.sin(2 * np.pi * 6 * t + phase)).astype(np.float32)
        X_list.append(window)
        y_list.append(1)

    return np.stack(X_list), np.asarray(y_list, dtype=np.int32)


def _log_rms_per_channel(windows: np.ndarray) -> np.ndarray:
    """Log-RMS power per channel for each window. Shape: (N, C).

    log-RMS is a band-power proxy widely used in EEG-HDC pipelines as a
    cheap, noise-robust per-channel summary.
    """
    rms = np.sqrt(np.mean(windows**2, axis=-1))
    return np.log1p(rms)


def _discretise(features: np.ndarray, num_levels: int) -> np.ndarray:
    """Quantile-bin per-channel features into ordinal levels [0, num_levels)."""
    out = np.empty(features.shape, dtype=np.int32)
    for c in range(features.shape[1]):
        edges = np.quantile(features[:, c], np.linspace(0, 1, num_levels + 1))
        edges = np.unique(edges)
        if len(edges) < 2:
            out[:, c] = 0
        else:
            idx = np.digitize(features[:, c], edges[1:-1])
            out[:, c] = np.clip(idx, 0, num_levels - 1)
    return out


def main() -> None:
    print("EEG seizure detection — Burrello-line iEEG HDC pipeline")
    print(
        f"  channels = {NUM_CHANNELS}   classes = {len(CLASS_NAMES)}   "
        f"windows/class = {WINDOWS_PER_CLASS}   D = {DIMS}\n"
    )

    key = jax.random.PRNGKey(SEED)
    k_data, k_codebook = jax.random.split(key)

    # ----------------------------------------------------------------- 1.
    print("[1] Window the EEG stream and compute log-RMS per channel.")
    X, y = _synthetic_eeg(k_data)
    rms = _log_rms_per_channel(X)
    print(f"      windows shape: {X.shape}    log-RMS shape: {rms.shape}")

    # ----------------------------------------------------------------- 2.
    print("\n[2] Discretise each channel's log-RMS into ordinal levels.")
    indices = _discretise(rms, NUM_LEVELS)
    print(f"      level indices shape: {indices.shape}   range: [{indices.min()}, {indices.max()}]")

    # 60 / 20 / 20 train / cal / test split.
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
    print("\n[4] Train BayesianCentroidClassifier — per-class Gaussian posteriors.")
    clf = BayesianCentroidClassifier.create(
        num_classes=len(CLASS_NAMES),
        dimensions=DIMS,
    ).fit(hv_tr, y_tr, prior_strength=1.0)

    train_acc = float(clf.score(hv_tr, y_tr))
    test_acc = float(clf.score(hv_te, y_te))
    print(f"      train accuracy: {train_acc:.3f}    test accuracy: {test_acc:.3f}")

    # ----------------------------------------------------------------- 5.
    print("\n[5] Calibrate logits + wrap in a ConformalClassifier (α = 0.10).")
    logits_ca = clf.logits(hv_ca)
    logits_te = clf.logits(hv_te)
    calibrator = TemperatureCalibrator.create().fit(logits_ca, y_ca, max_iters=200)
    probs_ca = calibrator.calibrate(logits_ca)
    probs_te = calibrator.calibrate(logits_te)

    conformal = ConformalClassifier.create(alpha=ALPHA).fit(probs_ca, y_ca)
    coverage = float(conformal.coverage(probs_te, y_te))
    mean_set_size = float(conformal.set_size(probs_te))
    preds = np.asarray(jnp.argmax(probs_te, axis=-1))
    y_te_np = np.asarray(y_te)
    print(f"      fitted temperature: {float(calibrator.temperature):.4f}")
    print(f"      target coverage: {1 - ALPHA:.2f}    empirical: {coverage:.3f}")
    print(f"      mean prediction-set size: {mean_set_size:.2f}  (of 2 classes)")

    # ----------------------------------------------------------------- 6.
    print("\n[6] Per-class detection performance:")
    print("      " + " ".join(f"{n:>12s}" for n in CLASS_NAMES))
    for true_c in range(len(CLASS_NAMES)):
        mask = y_te_np == true_c
        if not mask.any():
            continue
        row = []
        for pred_c in range(len(CLASS_NAMES)):
            count = int(((preds == pred_c) & mask).sum())
            row.append(f"{count:>12d}")
        print(f"  {CLASS_NAMES[true_c]:<12s}" + " ".join(row))

    # Sensitivity / specificity (the canonical seizure-detection report).
    tp = int(((preds == 1) & (y_te_np == 1)).sum())
    fn = int(((preds == 0) & (y_te_np == 1)).sum())
    fp = int(((preds == 1) & (y_te_np == 0)).sum())
    tn = int(((preds == 0) & (y_te_np == 0)).sum())
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    print(
        f"\n      sensitivity (true-positive rate)  = {sensitivity:.3f}    "
        f"({tp} / {tp + fn} ictal windows detected)"
    )
    print(
        f"      specificity (true-negative rate)  = {specificity:.3f}    "
        f"({tn} / {tn + fp} interictal windows correctly identified)"
    )

    print(
        "\nThe pipeline above is per-channel log-RMS + ordinal levels +"
        "\nchannel-value binding + bundle, the structure shared by the"
        "\nBurrello-Schindler-Benini-Rahimi 2018-2021 iEEG seizure-detection"
        "\nline. For a real benchmark, swap the synthetic data for a CHB-MIT"
        "\nor Bonn-University EEG corpus loader; the encoder and classifier"
        "\ncode paths above are unchanged."
    )


if __name__ == "__main__":
    main()
