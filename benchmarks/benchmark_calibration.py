#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Calibration benchmarks: Bayes-HDC vs TorchHD on real datasets.

Pipeline (both libraries, identical):

1. Standardise features, then discretise into ``n_levels`` bins
   (VoiceHD-style preprocessing; Imani et al., 2017).
2. Encode with a random codebook of shape
   ``(n_features, n_levels, D)`` and bundle per sample.
3. Train ``AdaptiveHDC`` (Bayes-HDC) or TorchHD's ``Centroid`` with
   iterative refinement — two epochs of misclassification-driven
   prototype updates after the initial centroid pass.
4. Hold out 20% of the training set as a calibration set for
   temperature scaling and conformal prediction.
5. Report accuracy, ECE, MCE, Brier, NLL, and sharpness on the test
   half before and after calibration; report Bayes-HDC conformal
   coverage and mean set size.

MNIST is loaded via OpenML and encoded through a random projection
(images are 784-dim raw).

Results are written to ``benchmarks/benchmark_calibration_results.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import (
    fetch_openml,
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

from bayes_hdc import (
    MAP,
    AdaptiveHDC,
    ConformalClassifier,
    ProjectionEncoder,
    RandomEncoder,
    TemperatureCalibrator,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    negative_log_likelihood,
    sharpness,
)

DEFAULT_DIMENSIONS = 10_000
DEFAULT_LEVELS = 64
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 2
CAL_FRACTION = 0.3


# ----------------------------------------------------------------------
# Dataset loading
# ----------------------------------------------------------------------


@dataclass
class DatasetSpec:
    name: str
    n_samples: int = 0
    n_features: int = 0
    n_classes: int = 0
    encoding: str = "tabular"  # "tabular" or "projection"


def _load_tabular(name: str, loader: Any) -> tuple[DatasetSpec, np.ndarray, np.ndarray]:
    data = loader()
    n_cls = int(data.target.max() + 1)
    spec = DatasetSpec(
        name=name,
        n_samples=data.data.shape[0],
        n_features=data.data.shape[1],
        n_classes=n_cls,
        encoding="tabular",
    )
    return spec, np.asarray(data.data, dtype=np.float32), np.asarray(data.target)


def _load_mnist(subsample: int | None = 10_000) -> tuple[DatasetSpec, np.ndarray, np.ndarray]:
    """Fetch MNIST from OpenML, optionally subsample for speed."""
    print("  fetching MNIST from OpenML...", flush=True)
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int32)
    if subsample is not None and subsample < X.shape[0]:
        rng = np.random.default_rng(DEFAULT_SEED)
        idx = rng.permutation(X.shape[0])[:subsample]
        X, y = X[idx], y[idx]
    spec = DatasetSpec(
        name="mnist",
        n_samples=X.shape[0],
        n_features=X.shape[1],
        n_classes=10,
        encoding="projection",
    )
    return spec, X, y


def _load_datasets(seed: int) -> list[tuple[DatasetSpec, np.ndarray, np.ndarray]]:
    """Return the full dataset list for the benchmark."""
    datasets: list[tuple[DatasetSpec, np.ndarray, np.ndarray]] = []

    datasets.append(_load_tabular("iris", load_iris))
    datasets.append(_load_tabular("wine", load_wine))
    datasets.append(_load_tabular("breast_cancer", load_breast_cancer))
    datasets.append(_load_tabular("digits", load_digits))

    try:
        datasets.append(_load_mnist(subsample=10_000))
    except Exception as e:  # pragma: no cover — network-dependent
        print(f"  [mnist unavailable: {e}]")

    return datasets


# ----------------------------------------------------------------------
# Metric helpers
# ----------------------------------------------------------------------


@dataclass
class Metrics:
    accuracy: float
    ece: float
    mce: float
    brier: float
    nll: float
    sharpness_: float


def _compute_metrics(probs: jax.Array, labels: jax.Array, n_classes: int) -> Metrics:
    return Metrics(
        accuracy=float(jnp.mean(jnp.argmax(probs, axis=-1) == labels)),
        ece=float(expected_calibration_error(probs, labels, n_bins=15)),
        mce=float(maximum_calibration_error(probs, labels, n_bins=15)),
        brier=float(brier_score(probs, labels, n_classes=n_classes)),
        nll=float(negative_log_likelihood(probs, labels)),
        sharpness_=float(sharpness(probs)),
    )


# ----------------------------------------------------------------------
# Pipelines
# ----------------------------------------------------------------------


@dataclass
class BayesHDCResult:
    raw: Metrics
    calibrated: Metrics
    conformal_alpha: float
    conformal_coverage: float
    conformal_set_size: float
    temperature: float
    train_ms: float
    infer_ms: float


def _encode_bayes_hdc(
    spec: DatasetSpec,
    X: np.ndarray,
    dimensions: int,
    levels: int,
    vsa: MAP,
    key: jax.Array,
) -> jax.Array:
    """Encode features to hypervectors with the standard HDC pipeline."""
    if spec.encoding == "projection":
        enc = ProjectionEncoder.create(
            input_dim=X.shape[1],
            dimensions=dimensions,
            vsa_model=vsa,
            key=key,
        )
        return enc.encode_batch(jnp.asarray(X, dtype=jnp.float32))

    # Tabular: discretise into `levels` bins, encode via RandomEncoder.
    disc = KBinsDiscretizer(n_bins=levels, encode="ordinal", strategy="quantile")
    X_idx = disc.fit_transform(X).astype(np.int32)
    enc = RandomEncoder.create(
        num_features=X.shape[1],
        num_values=levels,
        dimensions=dimensions,
        vsa_model=vsa,
        key=key,
    )
    return enc.encode_batch(jnp.asarray(X_idx))


def _run_bayes_hdc(
    spec: DatasetSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dimensions: int,
    levels: int,
    epochs: int,
    seed: int,
    alpha: float,
) -> BayesHDCResult:
    key = jax.random.PRNGKey(seed)
    k_enc, k_clf = jax.random.split(key)

    vsa = MAP.create(dimensions=dimensions)

    # Fit the discretiser on training data only, then apply to all splits.
    if spec.encoding == "tabular":
        disc = KBinsDiscretizer(n_bins=levels, encode="ordinal", strategy="quantile")
        X_train_idx = disc.fit_transform(X_train).astype(np.int32)
        X_cal_idx = np.clip(disc.transform(X_cal), 0, levels - 1).astype(np.int32)
        X_test_idx = np.clip(disc.transform(X_test), 0, levels - 1).astype(np.int32)

        enc = RandomEncoder.create(
            num_features=X_train.shape[1],
            num_values=levels,
            dimensions=dimensions,
            vsa_model=vsa,
            key=k_enc,
        )
        hv_tr = enc.encode_batch(jnp.asarray(X_train_idx))
        hv_ca = enc.encode_batch(jnp.asarray(X_cal_idx))
        hv_te = enc.encode_batch(jnp.asarray(X_test_idx))
    else:
        enc = ProjectionEncoder.create(
            input_dim=X_train.shape[1],
            dimensions=dimensions,
            vsa_model=vsa,
            key=k_enc,
        )
        hv_tr = enc.encode_batch(jnp.asarray(X_train, dtype=jnp.float32))
        hv_ca = enc.encode_batch(jnp.asarray(X_cal, dtype=jnp.float32))
        hv_te = enc.encode_batch(jnp.asarray(X_test, dtype=jnp.float32))

    ytr = jnp.asarray(y_train, dtype=jnp.int32)
    yca = jnp.asarray(y_cal, dtype=jnp.int32)
    yte = jnp.asarray(y_test, dtype=jnp.int32)

    # AdaptiveHDC: centroid init + iterative refinement.
    t0 = time.perf_counter()
    clf = AdaptiveHDC.create(
        num_classes=spec.n_classes,
        dimensions=dimensions,
        vsa_model=vsa,
        key=k_clf,
    ).fit(hv_tr, ytr, epochs=epochs)
    jax.block_until_ready(clf.prototypes)
    train_ms = (time.perf_counter() - t0) * 1000

    # Get logits (similarity scores) for each split.
    @jax.jit
    def _logits(hv_batch: jax.Array, prototypes: jax.Array) -> jax.Array:
        return hv_batch @ prototypes.T

    t0 = time.perf_counter()
    logits_te = _logits(hv_te, clf.prototypes)
    jax.block_until_ready(logits_te)
    infer_ms = (time.perf_counter() - t0) * 1000

    logits_ca = _logits(hv_ca, clf.prototypes)

    raw_probs = jax.nn.softmax(logits_te, axis=-1)
    raw_metrics = _compute_metrics(raw_probs, yte, spec.n_classes)

    # Temperature scaling on calibration set.
    calibrator = TemperatureCalibrator.create().fit(logits_ca, yca, max_iters=500, lr=0.05)
    probs_te_cal = calibrator.calibrate(logits_te)
    cal_metrics = _compute_metrics(probs_te_cal, yte, spec.n_classes)

    # Conformal prediction on calibrated probabilities.
    probs_ca_cal = calibrator.calibrate(logits_ca)
    conformal = ConformalClassifier.create(alpha=alpha).fit(probs_ca_cal, yca)
    coverage = float(conformal.coverage(probs_te_cal, yte))
    set_size = float(conformal.set_size(probs_te_cal))

    return BayesHDCResult(
        raw=raw_metrics,
        calibrated=cal_metrics,
        conformal_alpha=alpha,
        conformal_coverage=coverage,
        conformal_set_size=set_size,
        temperature=float(calibrator.temperature),
        train_ms=train_ms,
        infer_ms=infer_ms,
    )


@dataclass
class TorchHDResult:
    raw: Metrics
    calibrated: Metrics
    temperature: float
    train_ms: float
    infer_ms: float


def _run_torchhd(
    spec: DatasetSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dimensions: int,
    levels: int,
    epochs: int,
    seed: int,
) -> TorchHDResult | None:
    try:
        import torch
        import torchhd
        from torchhd import embeddings
        from torchhd.classifiers import Centroid
    except Exception as e:  # pragma: no cover
        print(f"  [torchhd unavailable: {e}]")
        return None

    torch.manual_seed(seed)

    if spec.encoding == "tabular":
        disc = KBinsDiscretizer(n_bins=levels, encode="ordinal", strategy="quantile")
        X_train_idx = disc.fit_transform(X_train).astype(np.int64)
        X_cal_idx = np.clip(disc.transform(X_cal), 0, levels - 1).astype(np.int64)
        X_test_idx = np.clip(disc.transform(X_test), 0, levels - 1).astype(np.int64)

        pos = embeddings.Random(spec.n_features, dimensions)
        val = embeddings.Random(levels, dimensions)

        def encode(X_idx: np.ndarray) -> torch.Tensor:
            X_t = torch.from_numpy(X_idx)
            pos_hv = pos.weight  # (n_features, D)
            val_hv = val(X_t)  # (batch, n_features, D)
            bound = torchhd.bind(pos_hv.unsqueeze(0), val_hv)
            return torchhd.multiset(bound)  # (batch, D)

        with torch.no_grad():
            hv_tr = encode(X_train_idx)
            hv_ca = encode(X_cal_idx)
            hv_te = encode(X_test_idx)
    else:
        enc = embeddings.Projection(in_features=spec.n_features, out_features=dimensions)
        with torch.no_grad():
            hv_tr = enc(torch.from_numpy(X_train))
            hv_ca = enc(torch.from_numpy(X_cal))
            hv_te = enc(torch.from_numpy(X_test))

    ytr = torch.from_numpy(np.asarray(y_train, dtype=np.int64))

    clf = Centroid(in_features=dimensions, out_features=spec.n_classes)

    t0 = time.perf_counter()
    with torch.no_grad():
        clf.add(hv_tr, ytr)
        # Two epochs of online refinement — matches AdaptiveHDC's pipeline.
        for _ in range(epochs):
            for i in range(hv_tr.shape[0]):
                x = hv_tr[i : i + 1]
                y = ytr[i : i + 1]
                logit = clf(x)
                pred = logit.argmax(dim=-1)
                if int(pred) != int(y):
                    clf.weight.data[y] += x.squeeze(0)
                    clf.weight.data[pred] -= x.squeeze(0)
    train_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    with torch.no_grad():
        logits_te = clf(hv_te)
        logits_ca = clf(hv_ca)
    infer_ms = (time.perf_counter() - t0) * 1000

    yte_j = jnp.asarray(y_test, dtype=jnp.int32)
    yca_j = jnp.asarray(y_cal, dtype=jnp.int32)
    logits_te_j = jnp.asarray(logits_te.numpy())
    logits_ca_j = jnp.asarray(logits_ca.numpy())

    raw_probs = jax.nn.softmax(logits_te_j, axis=-1)
    raw_metrics = _compute_metrics(raw_probs, yte_j, spec.n_classes)

    calibrator = TemperatureCalibrator.create().fit(logits_ca_j, yca_j, max_iters=500, lr=0.05)
    probs_te_cal = calibrator.calibrate(logits_te_j)
    cal_metrics = _compute_metrics(probs_te_cal, yte_j, spec.n_classes)

    return TorchHDResult(
        raw=raw_metrics,
        calibrated=cal_metrics,
        temperature=float(calibrator.temperature),
        train_ms=train_ms,
        infer_ms=infer_ms,
    )


# ----------------------------------------------------------------------
# Printing
# ----------------------------------------------------------------------


def _fmt(m: Metrics) -> str:
    return (
        f"acc={m.accuracy:.3f}  ECE={m.ece:.3f}  MCE={m.mce:.3f}  "
        f"Brier={m.brier:.3f}  NLL={m.nll:.3f}  sharp={m.sharpness_:.3f}"
    )


def _print_report(spec: DatasetSpec, bh: BayesHDCResult, th: TorchHDResult | None) -> None:
    header = (
        f"\n=== {spec.name}  (n={spec.n_samples}, "
        f"d_in={spec.n_features}, classes={spec.n_classes}, enc={spec.encoding}) ==="
    )
    print(header)
    print(f"  bayes-hdc  raw        {_fmt(bh.raw)}")
    print(f"  bayes-hdc  + temp     {_fmt(bh.calibrated)}")
    if th is not None:
        print(f"  torchhd    raw        {_fmt(th.raw)}")
        print(f"  torchhd    + temp     {_fmt(th.calibrated)}")
    print(
        f"  bayes-hdc conformal α={bh.conformal_alpha:.2f}  "
        f"coverage={bh.conformal_coverage:.3f}  set_size={bh.conformal_set_size:.2f}"
    )
    print(f"  T (bayes-hdc) = {bh.temperature:.3f}", end="")
    if th is not None:
        print(f"    T (torchhd) = {th.temperature:.3f}")
    else:
        print()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    ap.add_argument("--levels", type=int, default=DEFAULT_LEVELS)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/benchmark_calibration_results.json"),
    )
    ap.add_argument("--skip-torchhd", action="store_true")
    args = ap.parse_args()

    print(
        f"PVSA calibration benchmark — D={args.dimensions}, levels={args.levels}, "
        f"epochs={args.epochs}, seed={args.seed}"
    )
    print("=" * 72)

    results: list[dict[str, Any]] = []

    for spec, X, y in _load_datasets(args.seed):
        X = StandardScaler().fit_transform(X) if spec.encoding == "projection" else X

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=args.seed,
            stratify=y,
        )
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_train,
            y_train,
            test_size=CAL_FRACTION,
            random_state=args.seed,
            stratify=y_train,
        )

        bh = _run_bayes_hdc(
            spec,
            X_train,
            y_train,
            X_cal,
            y_cal,
            X_test,
            y_test,
            dimensions=args.dimensions,
            levels=args.levels,
            epochs=args.epochs,
            seed=args.seed,
            alpha=args.alpha,
        )
        th = (
            None
            if args.skip_torchhd
            else _run_torchhd(
                spec,
                X_train,
                y_train,
                X_cal,
                y_cal,
                X_test,
                y_test,
                dimensions=args.dimensions,
                levels=args.levels,
                epochs=args.epochs,
                seed=args.seed,
            )
        )

        _print_report(spec, bh, th)
        results.append(
            {
                "dataset": asdict(spec),
                "bayes_hdc": {
                    "raw": asdict(bh.raw),
                    "calibrated": asdict(bh.calibrated),
                    "temperature": bh.temperature,
                    "conformal": {
                        "alpha": bh.conformal_alpha,
                        "coverage": bh.conformal_coverage,
                        "set_size": bh.conformal_set_size,
                    },
                    "train_ms": bh.train_ms,
                    "infer_ms": bh.infer_ms,
                },
                "torchhd": None
                if th is None
                else {
                    "raw": asdict(th.raw),
                    "calibrated": asdict(th.calibrated),
                    "temperature": th.temperature,
                    "train_ms": th.train_ms,
                    "infer_ms": th.infer_ms,
                },
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\n→ wrote results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
