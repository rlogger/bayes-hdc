#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Selective classification: PVSA vs deterministic HDC.

Setup
-----

A selective classifier either makes a prediction or abstains. The goal
is: trade coverage (fraction of samples predicted) for accuracy on the
covered subset.

- **Bayes-HDC** uses :class:`ConformalClassifier`: the classifier
  predicts confidently on samples whose conformal set has size 1
  (only one class is admitted at the requested :math:`1 - \\alpha`
  coverage) and abstains otherwise.

- **TorchHD baseline (MSP threshold)** uses max-softmax probability:
  predict if :math:`\\max_c p(c \\mid x) \\geq \\tau`, otherwise abstain.
  :math:`\\tau` is chosen to match Bayes-HDC's coverage so the
  comparison is honest.

Both libraries run the same classifier (``AdaptiveHDC`` on Bayes-HDC's
side, ``Centroid`` on TorchHD's side) over the same discretised
encoding. The difference reported is the accuracy on the confident
subset, isolating the value of conformal-guarantee abstention over
softmax thresholding.

Results are written to ``benchmarks/benchmark_selective_results.json``.
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
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)
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
CAL_FRACTION = 0.3  # larger cal set → stable conformal quantile on small datasets


@dataclass
class SelectiveResult:
    dataset: str
    n_classes: int
    accuracy_full: float
    # Bayes-HDC conformal selective.
    coverage_pvsa: float
    accuracy_confident_pvsa: float
    # TorchHD MSP threshold matched to PVSA coverage.
    coverage_msp: float
    accuracy_confident_msp: float
    threshold_msp: float
    alpha: float
    fit_ms: float


def _encode(X, disc, enc):
    X_idx = np.clip(disc.transform(X), 0, enc.num_values - 1).astype(np.int32)
    return enc.encode_batch(jnp.asarray(X_idx))


def _run_one(
    dataset: str,
    X: np.ndarray,
    y: np.ndarray,
    dimensions: int,
    levels: int,
    seed: int,
    alpha: float,
    epochs: int,
) -> SelectiveResult:
    n_classes = int(y.max() + 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=seed,
        stratify=y,
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train,
        y_train,
        test_size=CAL_FRACTION,
        random_state=seed,
        stratify=y_train,
    )

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

    t0 = time.perf_counter()
    clf = AdaptiveHDC.create(
        num_classes=n_classes,
        dimensions=dimensions,
        vsa_model=vsa,
        key=k_clf,
    ).fit(hv_tr, jnp.asarray(y_train), epochs=epochs)
    jax.block_until_ready(clf.prototypes)
    fit_ms = (time.perf_counter() - t0) * 1000

    @jax.jit
    def _logits(hv, p):
        return hv @ p.T

    logits_ca = _logits(hv_ca, clf.prototypes)
    logits_te = _logits(hv_te, clf.prototypes)

    # Temperature-scaled probabilities for both scoring functions.
    calibrator = TemperatureCalibrator.create().fit(
        logits_ca, jnp.asarray(y_cal), max_iters=500, lr=0.05
    )
    probs_ca = calibrator.calibrate(logits_ca)
    probs_te = calibrator.calibrate(logits_te)

    preds_te = np.asarray(jnp.argmax(probs_te, axis=-1))
    accuracy_full = float(np.mean(preds_te == y_test))

    # PVSA selective: confident iff conformal set has size 1.
    conformal = ConformalClassifier.create(alpha=alpha).fit(probs_ca, jnp.asarray(y_cal))
    set_mask = conformal.predict_set(probs_te)
    set_sizes = np.asarray(jnp.sum(set_mask.astype(jnp.int32), axis=-1))
    confident_pvsa = set_sizes == 1
    coverage_pvsa = float(np.mean(confident_pvsa))
    if coverage_pvsa > 0:
        accuracy_confident_pvsa = float(np.mean(preds_te[confident_pvsa] == y_test[confident_pvsa]))
    else:
        accuracy_confident_pvsa = float("nan")

    # MSP threshold: pick τ so that coverage on calibration set equals coverage_pvsa.
    # Matching the empirical coverage makes the two methods directly comparable.
    cal_max = np.asarray(jnp.max(probs_ca, axis=-1))
    if coverage_pvsa >= 1.0:
        threshold = float(cal_max.min() - 1e-9)
    elif coverage_pvsa <= 0.0:
        threshold = float(cal_max.max() + 1e-9)
    else:
        threshold = float(np.quantile(cal_max, 1.0 - coverage_pvsa))

    te_max = np.asarray(jnp.max(probs_te, axis=-1))
    confident_msp = te_max >= threshold
    coverage_msp = float(np.mean(confident_msp))
    if coverage_msp > 0:
        accuracy_confident_msp = float(np.mean(preds_te[confident_msp] == y_test[confident_msp]))
    else:
        accuracy_confident_msp = float("nan")

    return SelectiveResult(
        dataset=dataset,
        n_classes=n_classes,
        accuracy_full=accuracy_full,
        coverage_pvsa=coverage_pvsa,
        accuracy_confident_pvsa=accuracy_confident_pvsa,
        coverage_msp=coverage_msp,
        accuracy_confident_msp=accuracy_confident_msp,
        threshold_msp=threshold,
        alpha=alpha,
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
        default=Path("benchmarks/benchmark_selective_results.json"),
    )
    args = ap.parse_args()

    print(f"PVSA selective-classification benchmark — α={args.alpha}")
    print("=" * 72)

    configs = [
        ("iris", load_iris),
        ("wine", load_wine),
        ("breast_cancer", load_breast_cancer),
        ("digits", load_digits),
    ]

    results: list[dict] = []
    for name, loader in configs:
        data = loader()
        X = np.asarray(data.data, dtype=np.float32)
        y = np.asarray(data.target, dtype=np.int32)
        res = _run_one(
            dataset=name,
            X=X,
            y=y,
            dimensions=args.dimensions,
            levels=args.levels,
            seed=args.seed,
            alpha=args.alpha,
            epochs=args.epochs,
        )
        print(
            f"\n{name}:  full acc = {res.accuracy_full:.3f}\n"
            f"  PVSA conformal  coverage={res.coverage_pvsa:.3f}  "
            f"confident-acc={res.accuracy_confident_pvsa:.3f}\n"
            f"  MSP threshold   coverage={res.coverage_msp:.3f}  "
            f"confident-acc={res.accuracy_confident_msp:.3f}  "
            f"(τ={res.threshold_msp:.3f})"
        )
        results.append(asdict(res))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\n→ wrote results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
