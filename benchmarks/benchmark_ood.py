#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Out-of-distribution detection: PVSA vs classical HDC.

Setup
-----

- Train a classifier on :math:`K - 1` in-distribution (ID) classes.
- At test time, score each sample against the classifier and ask:
  "Is this sample from the held-out (OOD) class?"
- Report AUROC for ID vs OOD discrimination.

Scoring functions under comparison
----------------------------------

1. **Max softmax probability (MSP)** — the standard baseline
   (Hendrycks & Gimpel, 2017). Available on both libraries; the score
   is :math:`\\max_c p(c \\mid x)`. Higher = more confident the sample is ID.

2. **Negative conformal set size (PVSA-only)** — uses
   ``ConformalClassifier`` fitted on an ID calibration set. A sample is
   scored by the number of classes the conformal procedure includes at
   :math:`1 - \\alpha` coverage: ID samples get small sets (1), OOD
   samples tend to get large sets (set-size grows when no class is a
   confident match). The score is ``-set_size`` so larger is more-ID,
   matching the MSP convention.

This is a PVSA-native OOD signal — it cannot be computed without the
conformal machinery that ``bayes-hdc`` ships and TorchHD does not.

Results are written to ``benchmarks/benchmark_ood_results.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import load_digits, load_wine
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

from bayes_hdc import (
    MAP,
    AdaptiveHDC,
    ConformalClassifier,
    RandomEncoder,
    TemperatureCalibrator,
)

DEFAULT_DIMENSIONS = 10_000
DEFAULT_LEVELS = 64
DEFAULT_SEED = 42
CAL_FRACTION = 0.3


@dataclass
class OODResult:
    dataset: str
    ood_class: int
    n_id_train: int
    n_id_test: int
    n_ood_test: int
    auroc_msp: float  # baseline, both libraries
    auroc_msp_calibrated: float  # after temperature scaling
    auroc_conformal_set_size: float  # PVSA-only
    fit_ms: float


def _holdout_ood(
    X: np.ndarray, y: np.ndarray, ood_class: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split X, y into (X_id, y_id, X_ood).  y_id keeps original labels."""
    ood_mask = y == ood_class
    return X[~ood_mask], y[~ood_mask], X[ood_mask]


def _encode(X: np.ndarray, disc: KBinsDiscretizer, enc: RandomEncoder) -> jax.Array:
    X_idx = np.clip(disc.transform(X), 0, enc.num_values - 1).astype(np.int32)
    return enc.encode_batch(jnp.asarray(X_idx))


def _run_one(
    dataset: str,
    X: np.ndarray,
    y: np.ndarray,
    ood_class: int,
    dimensions: int,
    levels: int,
    seed: int,
    alpha: float,
    epochs: int,
) -> OODResult:
    X_id, y_id, X_ood = _holdout_ood(X, y, ood_class)

    # Remap labels to contiguous 0..K-2 for the classifier.
    unique_ids = np.sort(np.unique(y_id))
    label_map = {int(old): i for i, old in enumerate(unique_ids)}
    y_id_remapped = np.asarray([label_map[int(v)] for v in y_id], dtype=np.int32)
    n_classes = len(unique_ids)

    X_train, X_test, y_train, y_test = train_test_split(
        X_id,
        y_id_remapped,
        test_size=0.3,
        random_state=seed,
        stratify=y_id_remapped,
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train,
        y_train,
        test_size=CAL_FRACTION,
        random_state=seed,
        stratify=y_train,
    )

    # Fit discretiser on training ID data only.
    disc = KBinsDiscretizer(n_bins=levels, encode="ordinal", strategy="quantile")
    disc.fit(X_train)

    key = jax.random.PRNGKey(seed)
    k_enc, k_clf = jax.random.split(key)
    vsa = MAP.create(dimensions=dimensions)
    enc = RandomEncoder.create(
        num_features=X_train.shape[1],
        num_values=levels,
        dimensions=dimensions,
        vsa_model=vsa,
        key=k_enc,
    )

    hv_tr = _encode(X_train, disc, enc)
    hv_ca = _encode(X_cal, disc, enc)
    hv_te = _encode(X_test, disc, enc)
    hv_ood = _encode(X_ood, disc, enc)

    t0 = time.perf_counter()
    clf = AdaptiveHDC.create(
        num_classes=n_classes,
        dimensions=dimensions,
        vsa_model=vsa,
        key=k_clf,
    ).fit(hv_tr, jnp.asarray(y_train), epochs=epochs)
    jax.block_until_ready(clf.prototypes)
    fit_ms = (time.perf_counter() - t0) * 1000

    # Raw logits and probabilities for all splits.
    @jax.jit
    def _logits(hv: jax.Array, p: jax.Array) -> jax.Array:
        return hv @ p.T

    logits_ca = _logits(hv_ca, clf.prototypes)
    logits_id_test = _logits(hv_te, clf.prototypes)
    logits_ood = _logits(hv_ood, clf.prototypes)

    # Temperature scaling on calibration set.
    calibrator = TemperatureCalibrator.create().fit(
        logits_ca, jnp.asarray(y_cal), max_iters=500, lr=0.05
    )

    probs_id_test_raw = jax.nn.softmax(logits_id_test, axis=-1)
    probs_ood_raw = jax.nn.softmax(logits_ood, axis=-1)
    probs_id_test_cal = calibrator.calibrate(logits_id_test)
    probs_ood_cal = calibrator.calibrate(logits_ood)
    probs_ca_cal = calibrator.calibrate(logits_ca)

    # AUROC for: OOD labeled 1, ID labeled 0; score = MSP.
    # Higher MSP = more confidently ID. We compute AUROC with the
    # OOD as the "positive" class and use negative MSP as the score
    # so that larger scores → more OOD.
    y_auroc = np.concatenate(
        [np.zeros(probs_id_test_raw.shape[0]), np.ones(probs_ood_raw.shape[0])]
    )
    msp_raw = np.concatenate(
        [
            -np.asarray(jnp.max(probs_id_test_raw, axis=-1)),
            -np.asarray(jnp.max(probs_ood_raw, axis=-1)),
        ]
    )
    msp_cal = np.concatenate(
        [
            -np.asarray(jnp.max(probs_id_test_cal, axis=-1)),
            -np.asarray(jnp.max(probs_ood_cal, axis=-1)),
        ]
    )
    auroc_msp = float(roc_auc_score(y_auroc, msp_raw))
    auroc_msp_cal = float(roc_auc_score(y_auroc, msp_cal))

    # Conformal set size as OOD score (PVSA-only).
    conformal = ConformalClassifier.create(alpha=alpha).fit(probs_ca_cal, jnp.asarray(y_cal))
    set_mask_id = conformal.predict_set(probs_id_test_cal)
    set_mask_ood = conformal.predict_set(probs_ood_cal)
    set_sizes_id = np.asarray(jnp.sum(set_mask_id.astype(jnp.int32), axis=-1))
    set_sizes_ood = np.asarray(jnp.sum(set_mask_ood.astype(jnp.int32), axis=-1))
    conformal_scores = np.concatenate([set_sizes_id, set_sizes_ood]).astype(np.float64)
    # Larger set_size = more OOD; that matches y_auroc convention.
    auroc_conformal = float(roc_auc_score(y_auroc, conformal_scores))

    return OODResult(
        dataset=dataset,
        ood_class=ood_class,
        n_id_train=X_train.shape[0],
        n_id_test=X_test.shape[0],
        n_ood_test=X_ood.shape[0],
        auroc_msp=auroc_msp,
        auroc_msp_calibrated=auroc_msp_cal,
        auroc_conformal_set_size=auroc_conformal,
        fit_ms=fit_ms,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    ap.add_argument("--levels", type=int, default=DEFAULT_LEVELS)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/benchmark_ood_results.json"),
    )
    args = ap.parse_args()

    print("PVSA OOD detection benchmark — holdout one class at a time")
    print("=" * 72)

    configs = [
        ("digits", load_digits),
        ("wine", load_wine),
    ]

    results: list[dict] = []
    for name, loader in configs:
        data = loader()
        X = np.asarray(data.data, dtype=np.float32)
        y = np.asarray(data.target, dtype=np.int32)
        n_classes = int(y.max() + 1)

        print(f"\n{name}  (n={X.shape[0]}, d_in={X.shape[1]}, K={n_classes})")
        dataset_aurocs: list[OODResult] = []
        for ood_class in range(n_classes):
            res = _run_one(
                dataset=name,
                X=X,
                y=y,
                ood_class=ood_class,
                dimensions=args.dimensions,
                levels=args.levels,
                seed=args.seed,
                alpha=args.alpha,
                epochs=args.epochs,
            )
            dataset_aurocs.append(res)
            print(
                f"  holdout class {ood_class:2d}:  "
                f"MSP auroc={res.auroc_msp:.3f}  "
                f"MSP+T auroc={res.auroc_msp_calibrated:.3f}  "
                f"conformal auroc={res.auroc_conformal_set_size:.3f}"
            )
            results.append(asdict(res))

        # Per-dataset means.
        mean_msp = float(np.mean([r.auroc_msp for r in dataset_aurocs]))
        mean_msp_cal = float(np.mean([r.auroc_msp_calibrated for r in dataset_aurocs]))
        mean_conf = float(np.mean([r.auroc_conformal_set_size for r in dataset_aurocs]))
        print(
            f"  [{name} mean]  MSP={mean_msp:.3f}  "
            f"MSP+T={mean_msp_cal:.3f}  conformal={mean_conf:.3f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\n→ wrote results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
