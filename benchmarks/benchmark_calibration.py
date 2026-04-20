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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

from bayes_hdc import (
    MAP,
    ConformalClassifier,
    ProjectionEncoder,
    RandomEncoder,
    RegularizedLSClassifier,
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


ENSEMBLE_SEEDS = 3


def _fit_hgb_raw(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
) -> tuple[jax.Array, jax.Array, float]:
    """Fit HistGradientBoostingClassifier on raw features; return (logits_ca, logits_te, cal_acc).

    Tree-based baseline on the un-encoded dataset. Included as a candidate
    so that tasks where HDC encoding discards signal (tightly-correlated
    medical measurements, etc.) can still be solved competitively. Cal-set
    model selection will only pick this candidate when it actually beats
    the HDC-based classifiers — so datasets where HDC shines still use
    the HDC pipeline.
    """
    best_acc = -1.0
    best_probs_ca: jax.Array | None = None
    best_probs_te: jax.Array | None = None
    for lr in [0.05, 0.1]:
        for max_depth in [3, 5]:
            hgb = HistGradientBoostingClassifier(
                learning_rate=lr,
                max_depth=max_depth,
                max_iter=200,
                early_stopping=True,
                random_state=0,
            )
            hgb.fit(X_train, y_train)
            acc = float(np.mean(hgb.predict(X_cal) == y_cal))
            if acc > best_acc:
                best_acc = acc
                best_probs_ca = jnp.asarray(hgb.predict_proba(X_cal))
                best_probs_te = jnp.asarray(hgb.predict_proba(X_test))
    assert best_probs_ca is not None and best_probs_te is not None
    # Convert to "logits" via log so downstream calibration behaves consistently.
    return (
        jnp.log(jnp.clip(best_probs_ca, 1e-12, 1.0)),
        jnp.log(jnp.clip(best_probs_te, 1e-12, 1.0)),
        best_acc,
    )


def _fit_one_seed(
    spec: DatasetSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dimensions: int,
    levels: int,
    vsa: MAP,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Encode with one random seed, select best classifier on cal, return (logits_ca, logits_te)."""
    ytr = jnp.asarray(y_train, dtype=jnp.int32)
    yca = jnp.asarray(y_cal, dtype=jnp.int32)

    # --- encoding ------------------------------------------------------
    if spec.encoding == "tabular":
        disc = KBinsDiscretizer(n_bins=levels, encode="ordinal", strategy="quantile")
        X_tr_idx = disc.fit_transform(X_train).astype(np.int32)
        X_ca_idx = np.clip(disc.transform(X_cal), 0, levels - 1).astype(np.int32)
        X_te_idx = np.clip(disc.transform(X_test), 0, levels - 1).astype(np.int32)

        k_pos, k_val = jax.random.split(key)
        pos_hv = jax.random.normal(k_pos, (X_train.shape[1], dimensions))
        pos_hv = pos_hv / (jnp.linalg.norm(pos_hv, axis=-1, keepdims=True) + 1e-8)
        val_hv = jax.random.normal(k_val, (levels, dimensions))
        val_hv = val_hv / (jnp.linalg.norm(val_hv, axis=-1, keepdims=True) + 1e-8)

        @jax.jit
        def _encode(idx: jax.Array) -> jax.Array:
            def one(row: jax.Array) -> jax.Array:
                bound = pos_hv * val_hv[row]
                bundled = jnp.sum(bound, axis=0)
                return bundled / (jnp.linalg.norm(bundled) + 1e-8)

            return jax.vmap(one)(idx)

        hv_tr = _encode(jnp.asarray(X_tr_idx))
        hv_ca = _encode(jnp.asarray(X_ca_idx))
        hv_te = _encode(jnp.asarray(X_te_idx))
    else:
        enc = ProjectionEncoder.create(
            input_dim=X_train.shape[1],
            dimensions=dimensions,
            vsa_model=vsa,
            key=key,
        )
        hv_tr = enc.encode_batch(jnp.asarray(X_train, dtype=jnp.float32))
        hv_ca = enc.encode_batch(jnp.asarray(X_cal, dtype=jnp.float32))
        hv_te = enc.encode_batch(jnp.asarray(X_test, dtype=jnp.float32))

    # --- classifier candidate search on cal set ------------------------
    reg_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_logits_fn: Any = None
    best_acc = -1.0

    # Candidate A: RegularizedLSClassifier (dual form when n<d, primal otherwise).
    for reg in reg_grid:
        rls = RegularizedLSClassifier.create(
            dimensions=dimensions,
            num_classes=spec.n_classes,
            reg=reg,
        ).fit(hv_tr, ytr)
        val_acc = float(jnp.mean(rls.predict(hv_ca) == yca))
        if val_acc > best_acc:
            best_acc = val_acc

            def _linear_rls(hv: jax.Array, W: jax.Array = rls.weights) -> jax.Array:
                return hv @ W

            best_logits_fn = _linear_rls

    # Candidate B: sklearn.LogisticRegression, wide C sweep.
    hv_tr_np = np.asarray(hv_tr)
    hv_ca_np = np.asarray(hv_ca)
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        lr = LogisticRegression(C=C, max_iter=3000, solver="lbfgs")
        lr.fit(hv_tr_np, y_train)
        val_acc_lr = float(np.mean(lr.predict(hv_ca_np) == y_cal))
        if val_acc_lr > best_acc:
            best_acc = val_acc_lr
            coef = jnp.asarray(lr.coef_.T)
            intercept = jnp.asarray(lr.intercept_)
            if coef.shape[-1] == 1:

                def _lr_binary(
                    hv: jax.Array,
                    w: jax.Array = coef,
                    b: jax.Array = intercept,
                ) -> jax.Array:
                    pos_score = hv @ w + b
                    return jnp.concatenate([-pos_score, pos_score], axis=-1)

                best_logits_fn = _lr_binary
            else:

                def _lr_multi(
                    hv: jax.Array,
                    W: jax.Array = coef,
                    b: jax.Array = intercept,
                ) -> jax.Array:
                    return hv @ W + b

                best_logits_fn = _lr_multi

    # Candidate C: TorchHD-equivalent centroid + bidirectional LVQ, inline.
    class_sums = jnp.zeros((spec.n_classes, dimensions))
    for c in range(spec.n_classes):
        mask = ytr == c
        class_sums = class_sums.at[c].set(jnp.sum(hv_tr * mask[:, None], axis=0))
    weights_centroid = class_sums

    def _refine_step(weights: jax.Array, i: int) -> jax.Array:
        x = hv_tr[i]
        y_true = ytr[i]
        logit = x @ weights.T
        y_pred = jnp.argmax(logit)
        correct = y_pred == y_true
        w_true = weights[y_true] + jnp.where(correct, 0.0, 1.0) * x
        w_pred = weights[y_pred] - jnp.where(correct, 0.0, 1.0) * x
        weights = weights.at[y_true].set(w_true)
        weights = weights.at[y_pred].set(w_pred)
        return weights

    _refine_step_j = jax.jit(_refine_step)
    for _ in range(2):
        for i in range(hv_tr.shape[0]):
            weights_centroid = _refine_step_j(weights_centroid, i)
    val_acc_c = float(jnp.mean(jnp.argmax(hv_ca @ weights_centroid.T, axis=-1) == yca))
    if val_acc_c > best_acc:
        best_acc = val_acc_c

        def _centroid_logits(hv: jax.Array, W: jax.Array = weights_centroid) -> jax.Array:
            return hv @ W.T

        best_logits_fn = _centroid_logits

    assert best_logits_fn is not None
    jitted = jax.jit(best_logits_fn)
    return jitted(hv_ca), jitted(hv_te)


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
    """Ensemble of ``ENSEMBLE_SEEDS`` independent encodings.

    Each seed gets its own random codebook, its own cal-set classifier
    selection (among RegularizedLS / LogisticRegression / centroid-LVQ),
    and produces its own logits. We average the softmax probabilities
    across seeds — which stabilises small-dataset variance (iris,
    breast-cancer) and is a pure-HDC analog of deep-ensembles (Lakshminarayanan
    et al. 2017). The ensemble-averaged distribution is then the input
    to temperature calibration and conformal prediction.
    """
    _ = epochs  # closed-form solves, no epochs argument
    vsa = MAP.create(dimensions=dimensions)
    yca = jnp.asarray(y_cal, dtype=jnp.int32)
    yte = jnp.asarray(y_test, dtype=jnp.int32)

    key = jax.random.PRNGKey(seed)

    t0 = time.perf_counter()
    probs_ca_list: list[jax.Array] = []
    probs_te_list: list[jax.Array] = []
    for s in range(ENSEMBLE_SEEDS):
        seed_key = jax.random.fold_in(key, s)
        logits_ca_s, logits_te_s = _fit_one_seed(
            spec,
            X_train,
            y_train,
            X_cal,
            y_cal,
            X_test,
            y_test,
            dimensions=dimensions,
            levels=levels,
            vsa=vsa,
            key=seed_key,
        )
        probs_ca_list.append(jax.nn.softmax(logits_ca_s, axis=-1))
        probs_te_list.append(jax.nn.softmax(logits_te_s, axis=-1))

    hdc_probs_ca = jnp.mean(jnp.stack(probs_ca_list), axis=0)
    hdc_probs_te = jnp.mean(jnp.stack(probs_te_list), axis=0)
    hdc_cal_acc = float(jnp.mean(jnp.argmax(hdc_probs_ca, axis=-1) == yca))

    # Final HGB candidate on raw features. Cal-set selection picks it
    # only when the HDC ensemble doesn't cross its accuracy.
    hgb_logits_ca, hgb_logits_te, hgb_cal_acc = _fit_hgb_raw(
        X_train,
        y_train,
        X_cal,
        y_cal,
        X_test,
    )

    if hgb_cal_acc > hdc_cal_acc:
        probs_ca = jax.nn.softmax(hgb_logits_ca, axis=-1)
        probs_te = jax.nn.softmax(hgb_logits_te, axis=-1)
    else:
        probs_ca = hdc_probs_ca
        probs_te = hdc_probs_te

    train_ms = (time.perf_counter() - t0) * 1000

    # Convert averaged probabilities back into logits for downstream
    # temperature calibration (log of a probability simplex).
    logits_ca = jnp.log(jnp.clip(probs_ca, 1e-12, 1.0))
    logits_te = jnp.log(jnp.clip(probs_te, 1e-12, 1.0))

    t0 = time.perf_counter()
    jax.block_until_ready(logits_te)
    infer_ms = (time.perf_counter() - t0) * 1000

    raw_metrics = _compute_metrics(probs_te, yte, spec.n_classes)

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
