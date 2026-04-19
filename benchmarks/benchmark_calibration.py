#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Calibration benchmarks: Bayes-HDC vs TorchHD on real datasets.

For each dataset we:

1. Encode features with a random projection into ``D``-dimensional
   hypervectors (the same procedure for both libraries).
2. Train a single-pass Centroid classifier on the training half.
3. Use 20% of the training set as a held-out calibration set for
   temperature scaling and conformal prediction.
4. Compute accuracy, Expected Calibration Error (ECE), and Brier score
   on the test half before and after temperature scaling.
5. Report Bayes-HDC conformal coverage and mean set size (this is the
   capability TorchHD does not provide).

This is the empirical back-end of the Bayesian contribution: TorchHD
plus post-hoc temperature scaling is the strongest fair baseline.

Output: a printable table plus a JSON dump to
``benchmarks/benchmark_calibration_results.json``.

Install:

    pip install -e ".[benchmark,examples]"   # scikit-learn + torchhd

Run:

    python benchmarks/benchmark_calibration.py
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
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_classification,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bayes_hdc import (
    MAP,
    CentroidClassifier,
    ConformalClassifier,
    ProjectionEncoder,
    TemperatureCalibrator,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    negative_log_likelihood,
    sharpness,
)

DEFAULT_DIMENSIONS = 4096
DEFAULT_SEED = 42
CAL_FRACTION = 0.2  # fraction of training set held out for calibration


# ----------------------------------------------------------------------
# Dataset registry
# ----------------------------------------------------------------------


@dataclass
class DatasetSpec:
    name: str
    loader: Any
    n_samples: int = 0
    n_features: int = 0
    n_classes: int = 0


def _load_datasets(seed: int) -> list[tuple[DatasetSpec, np.ndarray, np.ndarray]]:
    """Load the benchmark datasets and return (spec, X, y) tuples."""
    datasets: list[tuple[DatasetSpec, np.ndarray, np.ndarray]] = []

    for name, loader in [
        ("iris", load_iris),
        ("wine", load_wine),
        ("breast_cancer", load_breast_cancer),
        ("digits", load_digits),
    ]:
        data = loader()
        n_cls = (
            len(data.target_names)
            if hasattr(data, "target_names")
            else int(data.target.max() + 1)
        )
        datasets.append(
            (
                DatasetSpec(
                    name=name,
                    loader=loader,
                    n_samples=data.data.shape[0],
                    n_features=data.data.shape[1],
                    n_classes=n_cls,
                ),
                data.data,
                data.target,
            )
        )

    # A larger synthetic dataset with a harder, noisier decision boundary.
    Xs, ys = make_classification(
        n_samples=4000,
        n_features=30,
        n_informative=15,
        n_redundant=5,
        n_classes=5,
        n_clusters_per_class=2,
        flip_y=0.05,
        random_state=seed,
    )
    datasets.append(
        (
            DatasetSpec(name="synthetic", loader=None, n_samples=4000, n_features=30, n_classes=5),
            Xs,
            ys,
        )
    )

    return datasets


# ----------------------------------------------------------------------
# Bayes-HDC pipeline
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


@dataclass
class BayesHDCResult:
    accuracy: float
    raw: Metrics
    calibrated: Metrics
    conformal_alpha: float
    conformal_coverage: float
    conformal_set_size: float
    temperature: float
    train_ms: float
    infer_ms: float
    cal_ms: float
    conformal_ms: float


def _run_bayes_hdc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    dimensions: int,
    seed: int,
    alpha: float = 0.1,
) -> BayesHDCResult:
    key = jax.random.PRNGKey(seed)
    key_enc, key_clf = jax.random.split(key)

    vsa = MAP.create(dimensions=dimensions)
    encoder = ProjectionEncoder.create(
        input_dim=X_train.shape[1],
        dimensions=dimensions,
        vsa_model=vsa,
        key=key_enc,
    )

    # Encode (deterministic cast to float32 for JAX).
    Xtr = jnp.asarray(X_train, dtype=jnp.float32)
    Xca = jnp.asarray(X_cal, dtype=jnp.float32)
    Xte = jnp.asarray(X_test, dtype=jnp.float32)
    ytr = jnp.asarray(y_train, dtype=jnp.int32)
    yca = jnp.asarray(y_cal, dtype=jnp.int32)
    yte = jnp.asarray(y_test, dtype=jnp.int32)

    hv_tr = encoder.encode_batch(Xtr)
    hv_ca = encoder.encode_batch(Xca)
    hv_te = encoder.encode_batch(Xte)
    jax.block_until_ready(hv_te)

    # Train
    t0 = time.perf_counter()
    clf = CentroidClassifier.create(
        num_classes=n_classes, dimensions=dimensions, vsa_model=vsa, key=key_clf,
    ).fit(hv_tr, ytr)
    jax.block_until_ready(clf.prototypes)
    train_ms = (time.perf_counter() - t0) * 1000

    # Raw inference (logits = similarity scores; softmax for probs).
    t0 = time.perf_counter()
    probs_test_raw = jax.jit(clf.predict_proba)(hv_te)
    jax.block_until_ready(probs_test_raw)
    infer_ms = (time.perf_counter() - t0) * 1000

    raw_metrics = _compute_metrics(probs_test_raw, yte, n_classes)

    # Temperature calibration on the held-out calibration set.
    logits_cal = jax.vmap(clf.similarity)(hv_ca)
    logits_test = jax.vmap(clf.similarity)(hv_te)

    t0 = time.perf_counter()
    calibrator = TemperatureCalibrator.create().fit(logits_cal, yca, max_iters=500, lr=0.05)
    jax.block_until_ready(calibrator.temperature)
    cal_ms = (time.perf_counter() - t0) * 1000

    probs_test_cal = calibrator.calibrate(logits_test)
    cal_metrics = _compute_metrics(probs_test_cal, yte, n_classes)

    # Conformal prediction on the calibration set.
    probs_cal = calibrator.calibrate(logits_cal)
    t0 = time.perf_counter()
    conformal = ConformalClassifier.create(alpha=alpha).fit(probs_cal, yca)
    jax.block_until_ready(conformal.threshold)
    conformal_ms = (time.perf_counter() - t0) * 1000

    coverage = float(conformal.coverage(probs_test_cal, yte))
    set_size = float(conformal.set_size(probs_test_cal))

    return BayesHDCResult(
        accuracy=raw_metrics.accuracy,
        raw=raw_metrics,
        calibrated=cal_metrics,
        conformal_alpha=alpha,
        conformal_coverage=coverage,
        conformal_set_size=set_size,
        temperature=float(calibrator.temperature),
        train_ms=train_ms,
        infer_ms=infer_ms,
        cal_ms=cal_ms,
        conformal_ms=conformal_ms,
    )


# ----------------------------------------------------------------------
# TorchHD pipeline
# ----------------------------------------------------------------------


@dataclass
class TorchHDResult:
    accuracy: float
    raw: Metrics
    calibrated: Metrics
    temperature: float
    train_ms: float
    infer_ms: float
    cal_ms: float


def _run_torchhd(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    dimensions: int,
    seed: int,
) -> TorchHDResult | None:
    try:
        import torch
        from torchhd import embeddings
        from torchhd.classifiers import Centroid
    except Exception as e:  # pragma: no cover — reported, not tested
        print(f"  [torchhd unavailable: {e}]")
        return None

    torch.manual_seed(seed)
    device = torch.device("cpu")

    Xtr = torch.from_numpy(np.ascontiguousarray(X_train, dtype=np.float32))
    Xca = torch.from_numpy(np.ascontiguousarray(X_cal, dtype=np.float32))
    Xte = torch.from_numpy(np.ascontiguousarray(X_test, dtype=np.float32))
    ytr = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.int64))
    torch.from_numpy(np.ascontiguousarray(y_cal, dtype=np.int64))
    torch.from_numpy(np.ascontiguousarray(y_test, dtype=np.int64))

    # Random-projection encoder.
    enc = embeddings.Projection(in_features=X_train.shape[1], out_features=dimensions).to(device)
    with torch.no_grad():
        hv_tr = enc(Xtr)
        hv_ca = enc(Xca)
        hv_te = enc(Xte)

    clf = Centroid(in_features=dimensions, out_features=n_classes).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        clf.add(hv_tr, ytr)
    train_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    with torch.no_grad():
        logits_test = clf(hv_te)
    infer_ms = (time.perf_counter() - t0) * 1000

    probs_test_raw = torch.softmax(logits_test, dim=-1).numpy()
    probs_test_raw_j = jnp.asarray(probs_test_raw)
    yte_j = jnp.asarray(y_test, dtype=jnp.int32)
    raw_metrics = _compute_metrics(probs_test_raw_j, yte_j, n_classes)

    # Temperature scaling on calibration set (bayes-hdc's calibrator
    # — TorchHD does not ship one, so we use ours; this is a fair
    # comparison since temperature scaling is the classic baseline).
    with torch.no_grad():
        logits_cal = clf(hv_ca)

    t0 = time.perf_counter()
    logits_cal_j = jnp.asarray(logits_cal.numpy())
    yca_j = jnp.asarray(y_cal, dtype=jnp.int32)
    calibrator = TemperatureCalibrator.create().fit(logits_cal_j, yca_j, max_iters=500, lr=0.05)
    cal_ms = (time.perf_counter() - t0) * 1000

    logits_test_j = jnp.asarray(logits_test.numpy())
    probs_test_cal = calibrator.calibrate(logits_test_j)
    cal_metrics = _compute_metrics(probs_test_cal, yte_j, n_classes)

    return TorchHDResult(
        accuracy=raw_metrics.accuracy,
        raw=raw_metrics,
        calibrated=cal_metrics,
        temperature=float(calibrator.temperature),
        train_ms=train_ms,
        infer_ms=infer_ms,
        cal_ms=cal_ms,
    )


# ----------------------------------------------------------------------
# Pretty printing
# ----------------------------------------------------------------------


def _fmt_row(label: str, m: Metrics) -> str:
    return (
        f"{label:<28} acc={m.accuracy:.3f}  ECE={m.ece:.3f}  MCE={m.mce:.3f}  "
        f"Brier={m.brier:.3f}  NLL={m.nll:.3f}  sharp={m.sharpness_:.3f}"
    )


def _print_dataset_report(
    spec: DatasetSpec,
    bh: BayesHDCResult,
    th: TorchHDResult | None,
) -> None:
    header = (
        f"\n=== {spec.name}  (n={spec.n_samples}, "
        f"d_in={spec.n_features}, classes={spec.n_classes}) ==="
    )
    print(header)
    print(_fmt_row("bayes-hdc (raw)", bh.raw))
    print(_fmt_row("bayes-hdc + temp scaling", bh.calibrated))
    if th is not None:
        print(_fmt_row("torchhd (raw)", th.raw))
        print(_fmt_row("torchhd + temp scaling", th.calibrated))
    print(
        f"{'conformal (α=' + f'{bh.conformal_alpha:.2f}' + ')':<28} "
        f"coverage={bh.conformal_coverage:.3f}  set_size={bh.conformal_set_size:.2f}"
    )
    print(f"  bayes-hdc T = {bh.temperature:.3f}", end="")
    if th is not None:
        print(f"    torchhd T = {th.temperature:.3f}")
    else:
        print()


def _improvement_row(raw: Metrics, cal: Metrics, label: str) -> str:
    delta_ece = raw.ece - cal.ece
    delta_brier = raw.brier - cal.brier
    return f"{label:<28} ΔECE = {delta_ece:+.3f}    ΔBrier = {delta_brier:+.3f}"


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--alpha", type=float, default=0.1, help="Conformal miscoverage")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/benchmark_calibration_results.json"),
    )
    ap.add_argument("--skip-torchhd", action="store_true")
    args = ap.parse_args()

    print(f"Bayes-HDC calibration benchmark — D={args.dimensions}, seed={args.seed}")
    print(f"{'=' * 72}")

    results: list[dict[str, Any]] = []

    for spec, X, y in _load_datasets(args.seed):
        # Standardise features before projection — puts every dataset on
        # comparable scale so the projection encoder is meaningful.
        X = StandardScaler().fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=args.seed, stratify=y,
        )
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_train, y_train, test_size=CAL_FRACTION,
            random_state=args.seed, stratify=y_train,
        )

        n_classes = int(np.max(y) + 1)
        spec.n_classes = n_classes

        bh = _run_bayes_hdc(
            X_train, y_train, X_cal, y_cal, X_test, y_test,
            n_classes=n_classes,
            dimensions=args.dimensions,
            seed=args.seed,
            alpha=args.alpha,
        )
        th = (
            None if args.skip_torchhd
            else _run_torchhd(
                X_train, y_train, X_cal, y_cal, X_test, y_test,
                n_classes=n_classes,
                dimensions=args.dimensions,
                seed=args.seed,
            )
        )

        _print_dataset_report(spec, bh, th)

        entry: dict[str, Any] = {
            "dataset": asdict(spec) | {"loader": None},  # loader is not JSON-serialisable
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
                "cal_ms": bh.cal_ms,
                "conformal_ms": bh.conformal_ms,
            },
            "torchhd": None if th is None else {
                "raw": asdict(th.raw),
                "calibrated": asdict(th.calibrated),
                "temperature": th.temperature,
                "train_ms": th.train_ms,
                "infer_ms": th.infer_ms,
                "cal_ms": th.cal_ms,
            },
        }
        results.append(entry)

    # Summary table: per-dataset ECE / Brier before and after calibration.
    print(f"\n{'=' * 72}")
    print("Summary — effect of temperature scaling on ECE and Brier")
    print(f"{'=' * 72}")
    for entry in results:
        name = entry["dataset"]["name"]
        bh_raw = Metrics(**entry["bayes_hdc"]["raw"])
        bh_cal = Metrics(**entry["bayes_hdc"]["calibrated"])
        print(f"\n{name}:")
        print("  " + _improvement_row(bh_raw, bh_cal, "bayes-hdc"))
        if entry["torchhd"] is not None:
            th_raw = Metrics(**entry["torchhd"]["raw"])
            th_cal = Metrics(**entry["torchhd"]["calibrated"])
            print("  " + _improvement_row(th_raw, th_cal, "torchhd"))

    # Write JSON dump.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # loader isn't serialisable; remove it.
    clean = []
    for entry in results:
        d = dict(entry)
        d["dataset"] = {k: v for k, v in d["dataset"].items() if k != "loader"}
        clean.append(d)
    args.output.write_text(json.dumps(clean, indent=2))
    print(f"\n→ wrote results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
