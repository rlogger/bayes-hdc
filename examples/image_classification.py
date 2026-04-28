# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Image classification with classical HDC.

The reference HDC pipeline for vision (Imani et al. 2017,
Hassan et al. 2018, Hersche et al. 2019) is:

1. encode each image as a single hypervector via random projection;
2. bundle per-class training hypervectors into a class prototype;
3. classify by cosine similarity to each prototype.

This example runs that pipeline on the sklearn-bundled 8x8 digits
dataset by default (no network needed) and on real 28x28 MNIST when
called with ``--real-data`` (one-time OpenML download via
:func:`bayes_hdc.datasets.load_mnist`).

Three classical-HDC classifiers are compared on the same encoded
hypervectors:

- :class:`~bayes_hdc.CentroidClassifier` — one-shot bundling of training
  hypervectors per class (Kanerva 2009, Rahimi et al. 2016). The
  fastest path; no iteration.
- :class:`~bayes_hdc.AdaptiveHDC` — iterative prototype refinement with
  misclassification-driven updates (Imani et al. 2017 "VoiceHD",
  generalised). A few epochs noticeably improve test accuracy.
- :class:`~bayes_hdc.RegularizedLSClassifier` — closed-form ridge
  regression in hypervector space; auto-selects primal vs. dual form.

For the same task with calibrated probabilities and conformal
prediction sets, swap ``CentroidClassifier`` for
``BayesianCentroidClassifier`` — see ``examples/activity_recognition.py``.

Run::

    python examples/image_classification.py                # 8x8 digits
    python examples/image_classification.py --real-data    # 28x28 MNIST
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import load_digits

from bayes_hdc import (
    MAP,
    AdaptiveHDC,
    CentroidClassifier,
    ProjectionEncoder,
    RegularizedLSClassifier,
)

DIMS = 10_000
SEED = 2026
TEST_FRAC = 0.3


def _load_sklearn_digits() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, str]:
    """Bundled sklearn 8x8 digits (1797 samples, 10 classes). Always offline."""
    digits = load_digits()
    X = (digits.data / 16.0).astype(np.float32)  # normalise to [0, 1]
    y = digits.target.astype(np.int32)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X))
    n_tr = int((1 - TEST_FRAC) * len(X))
    tr_idx, te_idx = perm[:n_tr], perm[n_tr:]
    return X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], 10, "sklearn digits 8x8 (1797 samples)"


def _load_real_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, str]:
    """Real MNIST 28x28 via OpenML (one-time download, ~12 MB)."""
    from bayes_hdc.datasets import load_mnist

    data = load_mnist(subsample=10_000)
    return (
        np.asarray(data.X_train, dtype=np.float32),
        np.asarray(data.y_train, dtype=np.int32),
        np.asarray(data.X_test, dtype=np.float32),
        np.asarray(data.y_test, dtype=np.int32),
        10,
        "MNIST 28x28 (subsampled to 10000)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Load real MNIST 28x28 via OpenML (one-time download). "
        "Default: sklearn-bundled 8x8 digits, always offline.",
    )
    args = parser.parse_args()

    if args.real_data:
        try:
            X_tr, y_tr, X_te, y_te, n_classes, label = _load_real_mnist()
        except Exception as e:  # noqa: BLE001 — surface failure, fall back.
            print(f"  ! could not load real MNIST ({e}); falling back to sklearn digits.")
            X_tr, y_tr, X_te, y_te, n_classes, label = _load_sklearn_digits()
    else:
        X_tr, y_tr, X_te, y_te, n_classes, label = _load_sklearn_digits()

    print("Image classification with classical HDC")
    print(
        f"  data = {label}   classes = {n_classes}   "
        f"train / test = {len(y_tr)} / {len(y_te)}   D = {DIMS}\n"
    )

    key = jax.random.PRNGKey(SEED)
    vsa = MAP.create(dimensions=DIMS)

    # ----------------------------------------------------------------- 1.
    print("[1] Encode every image as a single hypervector via random projection.")
    encoder = ProjectionEncoder.create(
        input_dim=X_tr.shape[1],
        dimensions=DIMS,
        vsa_model=vsa,
        key=key,
    )
    hv_tr = encoder.encode_batch(jnp.asarray(X_tr))
    hv_te = encoder.encode_batch(jnp.asarray(X_te))
    print(f"      encoded HV shape: train {tuple(hv_tr.shape)}    test {tuple(hv_te.shape)}")

    y_tr_jax = jnp.asarray(y_tr)
    y_te_jax = jnp.asarray(y_te)

    # ----------------------------------------------------------------- 2.
    print("\n[2] CentroidClassifier — one-shot bundling per class.")
    centroid = CentroidClassifier.create(
        num_classes=n_classes,
        dimensions=DIMS,
        vsa_model=vsa,
    ).fit(hv_tr, y_tr_jax)
    centroid_acc = float(centroid.score(hv_te, y_te_jax))
    print(f"      test accuracy = {centroid_acc:.3f}")

    # ----------------------------------------------------------------- 3.
    print("\n[3] AdaptiveHDC — 5 epochs of misclassification-driven refinement.")
    adaptive = AdaptiveHDC.create(
        num_classes=n_classes,
        dimensions=DIMS,
        vsa_model=vsa,
    ).fit(hv_tr, y_tr_jax, epochs=5)
    adaptive_acc = float(adaptive.score(hv_te, y_te_jax))
    delta = adaptive_acc - centroid_acc
    sign = "+" if delta >= 0 else ""
    print(f"      test accuracy = {adaptive_acc:.3f}    ({sign}{delta * 100:.1f} pp vs. centroid)")

    # ----------------------------------------------------------------- 4.
    print("\n[4] RegularizedLSClassifier — closed-form ridge in HV space.")
    rls = RegularizedLSClassifier.create(
        dimensions=DIMS,
        num_classes=n_classes,
        reg=1.0,
    ).fit(hv_tr, y_tr_jax)
    rls_preds = jnp.argmax(hv_te @ rls.weights, axis=-1)
    rls_acc = float(jnp.mean(rls_preds == y_te_jax))
    print(f"      test accuracy = {rls_acc:.3f}")

    # ----------------------------------------------------------------- 5.
    print("\n[5] Per-class confusion (best classifier):")
    best_name, best_preds, best_acc = max(
        [
            ("CentroidClassifier", np.asarray(centroid.predict(hv_te)), centroid_acc),
            ("AdaptiveHDC", np.asarray(adaptive.predict(hv_te)), adaptive_acc),
            ("RegularizedLSClassifier", np.asarray(rls_preds), rls_acc),
        ],
        key=lambda t: t[2],
    )
    print(f"      best = {best_name} ({best_acc:.3f})")
    y_te_np = np.asarray(y_te)
    print("      true \\ predicted →")
    header = "       " + " ".join(f"{c:>4d}" for c in range(n_classes))
    print(header)
    for true_c in range(n_classes):
        row_mask = y_te_np == true_c
        if not row_mask.any():
            continue
        cells = [int(((best_preds == pred_c) & row_mask).sum()) for pred_c in range(n_classes)]
        print(f"  {true_c:>3d} | " + " ".join(f"{v:>4d}" for v in cells))

    print(
        "\nClassical HDC vision pipeline (Imani et al. 2017, Hassan et al. 2018):"
        "\n  random projection → bundle → centroid / adaptive / ridge classifier."
        "\nFor calibrated probabilities and conformal prediction sets on top of"
        "\nthe same encoder, see examples/activity_recognition.py."
    )


if __name__ == "__main__":
    main()
