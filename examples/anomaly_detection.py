# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Out-of-distribution detection via PVSA posterior Mahalanobis distance.

A PVSA classifier stores a :class:`~bayes_hdc.GaussianHV` posterior per class
(mean ``mu_c`` and per-dimension variance ``var_c``), which gives two
complementary OOD signals:

1. **Max predicted probability** — the standard MSP baseline (Hendrycks &
   Gimpel 2017). Low MSP suggests the classifier is unsure which class it
   belongs to.
2. **Minimum per-class Mahalanobis distance** — exclusive to PVSA. For
   each query ``x`` and class ``c`` we compute

       d_c(x) = sum_i (x_i - mu_c,i)^2 / (var_c,i + eps)

   which is the Mahalanobis distance under the (diagonal) class posterior.
   Taking the minimum over classes asks: "is there *any* class whose
   learned distribution explains this query?" High min-distance indicates
   the query is far from all training clusters — an OOD signal that the
   deterministic MSP cannot match because it ignores ``var_c``.

This example trains a :class:`BayesianCentroidClassifier` on 8 digit classes
(0-7 in the UCI digits dataset), then scores the held-out classes 8 and 9 as
out-of-distribution. We report AUROC for each signal and their combination.

Run::

    python examples/anomaly_detection.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer

from bayes_hdc import (
    MAP,
    BayesianCentroidClassifier,
    RandomEncoder,
)

DIMS = 4096
LEVELS = 16
SEED = 42


def main() -> None:
    print("PVSA anomaly detection — train on digits 0–7, score 8–9 as OOD\n")

    # Load UCI digits; split into in-distribution (classes 0–7) and OOD (8–9).
    data = load_digits()
    X = np.asarray(data.data, dtype=np.float32)
    y = np.asarray(data.target, dtype=np.int32)

    id_mask = y < 8
    X_id, y_id = X[id_mask], y[id_mask]
    X_ood = X[~id_mask]

    # Discretise pixel intensities into 16 ordinal levels.
    disc = KBinsDiscretizer(
        n_bins=LEVELS, encode="ordinal", strategy="quantile",
    )
    X_id_idx = disc.fit_transform(X_id).astype(np.int32)
    X_ood_idx = np.clip(disc.transform(X_ood), 0, LEVELS - 1).astype(np.int32)

    # Random codebook, encode training + OOD samples.
    key = jax.random.PRNGKey(SEED)
    vsa = MAP.create(dimensions=DIMS)
    encoder = RandomEncoder.create(
        num_features=X_id.shape[1],
        num_values=LEVELS,
        dimensions=DIMS,
        vsa_model=vsa,
        key=key,
    )
    hv_id = encoder.encode_batch(jnp.asarray(X_id_idx))
    hv_ood = encoder.encode_batch(jnp.asarray(X_ood_idx))

    # Split ID into 70/30 train/test for in-dist evaluation.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(hv_id.shape[0])
    n_tr = int(0.7 * len(perm))
    tr_idx, te_idx = perm[:n_tr], perm[n_tr:]
    hv_tr, y_tr = hv_id[tr_idx], jnp.asarray(y_id[tr_idx])
    hv_te_id, y_te_id = hv_id[te_idx], jnp.asarray(y_id[te_idx])

    clf = BayesianCentroidClassifier.create(
        num_classes=8, dimensions=DIMS,
    ).fit(hv_tr, y_tr, prior_strength=1.0)

    acc = float(clf.score(hv_te_id, y_te_id))
    print(f"Accuracy on in-distribution test set (classes 0–7): {acc:.3f}\n")

    # ============================================================== OOD scores
    # Signal 1: max predicted probability (MSP baseline).
    probs_id = clf.predict_proba(hv_te_id)
    probs_ood = clf.predict_proba(hv_ood)
    msp_id = np.asarray(jnp.max(probs_id, axis=-1))
    msp_ood = np.asarray(jnp.max(probs_ood, axis=-1))

    # Signal 2: min per-class Mahalanobis distance (PVSA-exclusive).
    # d_c(x) = sum_i (x_i - mu_c,i)^2 / (var_c,i + eps)
    def mahal_per_class(hv: jax.Array) -> jax.Array:
        mu, var = clf.mu, clf.var  # shapes (K, D)
        # (N, K, D): broadcast diff per sample per class.
        diff = hv[:, None, :] - mu[None, :, :]
        d = jnp.sum((diff ** 2) / (var[None, :, :] + 1e-6), axis=-1)
        return d  # (N, K)

    mahal_id = np.asarray(mahal_per_class(hv_te_id))
    mahal_ood = np.asarray(mahal_per_class(hv_ood))
    min_mahal_id = mahal_id.min(axis=-1)
    min_mahal_ood = mahal_ood.min(axis=-1)

    # Labels for AUROC: 1 = OOD.
    y_auroc = np.concatenate(
        [np.zeros(len(msp_id)), np.ones(len(msp_ood))]
    )
    # MSP: lower on OOD → negate so higher = more OOD.
    scores_msp = np.concatenate([-msp_id, -msp_ood])
    # Mahalanobis: higher on OOD already.
    scores_mahal = np.concatenate([min_mahal_id, min_mahal_ood])

    auroc_msp = float(roc_auc_score(y_auroc, scores_msp))
    auroc_mahal = float(roc_auc_score(y_auroc, scores_mahal))

    # Combine MSP + Mahalanobis into a single score.
    # Normalise each to [0, 1] then sum.
    def _norm(x: np.ndarray) -> np.ndarray:
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    combined = _norm(scores_msp) + _norm(scores_mahal)
    auroc_combined = float(roc_auc_score(y_auroc, combined))

    print("OOD detection AUROC (higher = better):")
    print(f"  MSP baseline (Hendrycks & Gimpel 2017):        {auroc_msp:.3f}")
    print(f"  PVSA min posterior Mahalanobis (this library): {auroc_mahal:.3f}")
    print(f"  Combined (MSP + PVSA Mahalanobis):             {auroc_combined:.3f}\n")

    # Report per-distribution mean scores.
    print("Mean scores per distribution:")
    print(f"  ID  : MSP = {msp_id.mean():.3f}   min_mahal = {min_mahal_id.mean():.1f}")
    print(f"  OOD : MSP = {msp_ood.mean():.3f}   min_mahal = {min_mahal_ood.mean():.1f}\n")

    if auroc_combined > max(auroc_msp, auroc_mahal):
        print("→ combining MSP with PVSA Mahalanobis distance beats either alone.")
    elif auroc_mahal > auroc_msp:
        print("→ PVSA Mahalanobis distance is the stronger OOD signal on this task.")
    else:
        print(
            "→ MSP is the stronger signal here; PVSA Mahalanobis adds complementary info."
        )


if __name__ == "__main__":
    main()
