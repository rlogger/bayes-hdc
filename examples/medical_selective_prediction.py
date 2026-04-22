# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Selective classification for high-stakes medical decisions.

A selective classifier either makes a prediction or abstains. In medical
settings this is the right default: misclassifying a tumour as benign has
much higher cost than reporting "uncertain, send for biopsy".

This example runs on the UCI Breast Cancer Wisconsin Diagnostic dataset
(569 tumours, 30 features, binary benign vs malignant). It trains a PVSA
pipeline (random codebook + ridge regression on hypervectors + temperature
scaling + conformal prediction) and uses the conformal set size as the
abstention gate:

- If the conformal prediction set has size 1 at :math:`\\alpha = 0.05`, the
  classifier *predicts* with marginal coverage ≥ 95 %.
- If the set has size 2 (both classes), the classifier *abstains*.

Reported metrics:

- **Overall coverage** — fraction of test samples given a confident prediction.
- **Accuracy on predicted subset** — should be much higher than the
  overall accuracy, because the hard cases are abstained on.
- **Overall accuracy under abstention** — abstentions count as wrong.

Run::

    python examples/medical_selective_prediction.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

from bayes_hdc import (
    MAP,
    ConformalClassifier,
    RandomEncoder,
    RegularizedLSClassifier,
    TemperatureCalibrator,
)

DIMS = 4096
LEVELS = 32
SEED = 42
ALPHA = 0.05  # tight coverage: 95 %


def main() -> None:
    print("Selective classification — UCI Breast Cancer Wisconsin Diagnostic\n")

    data = load_breast_cancer()
    X = np.asarray(data.data, dtype=np.float32)
    y = np.asarray(data.target, dtype=np.int32)  # 0 = malignant, 1 = benign
    class_names = list(data.target_names)

    # 60 / 20 / 20 train / cal / test split, stratified.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )
    X_tr, X_ca, y_tr, y_ca = train_test_split(
        X_tr,
        y_tr,
        test_size=0.25,
        random_state=SEED,
        stratify=y_tr,
    )

    # Discretise + encode.
    disc = KBinsDiscretizer(
        n_bins=LEVELS,
        encode="ordinal",
        strategy="quantile",
    )
    X_tr_idx = disc.fit_transform(X_tr).astype(np.int32)
    X_ca_idx = np.clip(disc.transform(X_ca), 0, LEVELS - 1).astype(np.int32)
    X_te_idx = np.clip(disc.transform(X_te), 0, LEVELS - 1).astype(np.int32)

    key = jax.random.PRNGKey(SEED)
    vsa = MAP.create(dimensions=DIMS)
    enc = RandomEncoder.create(
        num_features=X.shape[1],
        num_values=LEVELS,
        dimensions=DIMS,
        vsa_model=vsa,
        key=key,
    )
    hv_tr = enc.encode_batch(jnp.asarray(X_tr_idx))
    hv_ca = enc.encode_batch(jnp.asarray(X_ca_idx))
    hv_te = enc.encode_batch(jnp.asarray(X_te_idx))

    # Classifier, calibrator, conformal wrap.
    clf = RegularizedLSClassifier.create(
        dimensions=DIMS,
        num_classes=2,
        reg=1.0,
    ).fit(hv_tr, jnp.asarray(y_tr))
    logits_ca = hv_ca @ clf.weights
    logits_te = hv_te @ clf.weights
    calibrator = TemperatureCalibrator.create().fit(
        logits_ca,
        jnp.asarray(y_ca),
        max_iters=200,
    )
    probs_ca = calibrator.calibrate(logits_ca)
    probs_te = calibrator.calibrate(logits_te)
    conformal = ConformalClassifier.create(alpha=ALPHA).fit(
        probs_ca,
        jnp.asarray(y_ca),
    )

    # Selective decision: predict iff the conformal set has size 1.
    set_mask = np.asarray(conformal.predict_set(probs_te).astype(np.int32))
    set_sizes = set_mask.sum(axis=-1)
    confident = set_sizes == 1

    preds = np.asarray(jnp.argmax(probs_te, axis=-1))

    # Metrics.
    n_test = len(y_te)
    n_confident = int(confident.sum())
    overall_acc = float(np.mean(preds == y_te))
    coverage_pct = n_confident / n_test
    if n_confident > 0:
        confident_acc = float(np.mean(preds[confident] == y_te[confident]))
    else:
        confident_acc = float("nan")
    abstained_acc = (
        float(np.mean(preds[~confident] == y_te[~confident]))
        if (~confident).any()
        else float("nan")
    )

    # "Under abstention" accuracy: predictions on abstentions count as wrong.
    correct_under_abs = (preds == y_te) & confident
    accuracy_under_abs = float(np.mean(correct_under_abs))

    empirical_coverage = float(conformal.coverage(probs_te, jnp.asarray(y_te)))
    mean_set_size = float(conformal.set_size(probs_te))

    print("Dataset:")
    print(f"  classes           = {class_names}")
    print(f"  n_train / cal / test = {len(y_tr)} / {len(y_ca)} / {n_test}\n")

    print(f"Target coverage (1 − α)   = {1 - ALPHA:.2f}")
    print(f"Empirical coverage        = {empirical_coverage:.3f}")
    print(f"Mean prediction-set size  = {mean_set_size:.2f}")
    print(f"Fitted temperature T      = {float(calibrator.temperature):.4f}\n")

    print("Overall classification (no abstention):")
    print(f"  test accuracy            = {overall_acc:.3f}")
    print()

    print("Selective classification via conformal set size:")
    print(f"  confident predictions    = {n_confident}/{n_test} ({coverage_pct:.1%})")
    print(f"  accuracy on confident    = {confident_acc:.3f}  ← the one that matters")
    print(f"  accuracy on abstained    = {abstained_acc:.3f}")
    print(f"  accuracy under abstention= {accuracy_under_abs:.3f}")
    print()

    if confident_acc > overall_acc:
        gain = 100 * (confident_acc - overall_acc)
        print(f"→ abstention buys +{gain:.1f}pp accuracy on the confident subset,")
        print(f"  at the cost of {(1 - coverage_pct):.1%} of cases routed to follow-up.")
    else:
        print("→ on this split, abstention does not improve confident-subset accuracy.")


if __name__ == "__main__":
    main()
