#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 bayes-hdc contributors
"""Matched centroid calibration benchmark on shared HDC representations.

Both backends receive exactly the same encoded observations and use normalized
class sums with cosine logits. This compares classifier implementations and
calibration, not independently optimized pipelines or end-to-end encoder speed.
Preprocessing sees proper training data only. Temperature fitting and conformal
calibration use separate held-out subsets. Test labels are used only for metrics.
The four offline datasets run by default; MNIST downloads require --include-mnist.
Saved probabilities and labels are the source for every generated figure.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

from bayes_hdc import (
    MAP,
    CentroidClassifier,
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

if __package__:
    from ._common import provenance, standardize_train, write_json
else:
    from _common import provenance, standardize_train, write_json

DEFAULT_DIMENSIONS = 10_000
DEFAULT_LEVELS = 64
DEFAULT_SEED = 42


@dataclass
class DatasetSpec:
    name: str
    n_samples: int
    n_features: int
    n_classes: int
    encoding: str = "tabular"


@dataclass
class Metrics:
    accuracy: float
    ece: float
    mce: float
    brier: float
    nll: float
    sharpness_: float


def _compute_metrics(probs, labels, n_classes):
    return Metrics(
        accuracy=float(jnp.mean(jnp.argmax(probs, axis=-1) == labels)),
        ece=float(expected_calibration_error(probs, labels, n_bins=15)),
        mce=float(maximum_calibration_error(probs, labels, n_bins=15)),
        brier=float(brier_score(probs, labels, n_classes=n_classes)),
        nll=float(negative_log_likelihood(probs, labels)),
        sharpness_=float(sharpness(probs)),
    )


def _load_datasets(seed, include_mnist=False):
    for name, loader in (
        ("iris", load_iris),
        ("wine", load_wine),
        ("breast_cancer", load_breast_cancer),
        ("digits", load_digits),
    ):
        data = loader()
        X, y = np.asarray(data.data, dtype=np.float32), np.asarray(data.target, dtype=np.int32)
        yield DatasetSpec(name, len(y), X.shape[1], len(np.unique(y))), X, y
    if include_mnist:
        data = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
        idx = np.random.default_rng(seed).permutation(len(data.target))[:10_000]
        X = np.asarray(data.data[idx], dtype=np.float32) / 255.0
        y = np.asarray(data.target[idx], dtype=np.int32)
        yield DatasetSpec("mnist", len(y), X.shape[1], 10, "projection"), X, y


def split_indices(y, seed):
    """60/10/10/20 fit/temperature/conformal/test, with auditable row IDs.

    Holdouts are split without conditioning on their labels. This avoids claiming
    ordinary exchangeable split-conformal coverage for class-count-conditioned
    partitions. Small datasets may have noisy metrics or lack a training class.
    """
    fit, held = train_test_split(np.arange(len(y)), test_size=0.4, random_state=seed)
    temp_conf, test = train_test_split(held, test_size=0.5, random_state=seed + 1)
    temp, conf = train_test_split(temp_conf, test_size=0.5, random_state=seed + 2)
    if len(temp) < 2 or len(conf) < 2:
        raise ValueError("Need at least two temperature and conformal observations")
    if len(np.unique(y[fit])) != len(np.unique(y)):
        raise ValueError("Training split lacks a class; use more observations or another seed")
    return fit, temp, conf, test


def encode_splits(spec, splits, dimensions, levels, seed):
    """Fit a single encoder/preprocessor on the proper training split."""
    key = jax.random.PRNGKey(seed)
    if spec.encoding == "projection":
        splits = standardize_train(*splits)
        enc = ProjectionEncoder.create(spec.n_features, dimensions, vsa_model="map", key=key)
        return tuple(enc.encode_batch(jnp.asarray(x)) for x in splits)
    disc = KBinsDiscretizer(n_bins=levels, encode="ordinal", strategy="quantile", random_state=seed)
    disc.fit(splits[0])
    enc = RandomEncoder.create(spec.n_features, levels, dimensions, MAP.create(dimensions), key)
    return tuple(enc.encode_batch(jnp.asarray(disc.transform(x).astype(np.int32))) for x in splits)


def _jax_centroid(hvs, yfit, n_classes):
    train, temp, conf, test = hvs
    jax.block_until_ready(hvs)
    model = CentroidClassifier.create(n_classes, train.shape[1])
    start = time.perf_counter()
    model = model.fit(train, jnp.asarray(yfit))
    jax.block_until_ready(model.prototypes)
    fit_ms = (time.perf_counter() - start) * 1000
    logits_fn = jax.jit(jax.vmap(model.similarity))
    # Warm up the exact test shape before measuring steady-state scoring.
    jax.block_until_ready(logits_fn(test))
    start = time.perf_counter()
    logits_test = jax.block_until_ready(logits_fn(test))
    infer_ms = (time.perf_counter() - start) * 1000
    return (logits_fn(temp), logits_fn(conf), logits_test), fit_ms, infer_ms


def _torch_centroid(hvs, yfit, n_classes):
    try:
        import torch
        from torchhd.models import Centroid
    except ImportError:
        return None
    # Input transfer and encoding are excluded from both classifier timers.
    train, temp, conf, test = [torch.from_numpy(np.asarray(x).copy()) for x in hvs]
    labels = torch.as_tensor(yfit, dtype=torch.long)
    model = Centroid(train.shape[1], n_classes)
    with torch.no_grad():
        start = time.perf_counter()
        model.add(train, labels)
        model.normalize()
        fit_ms = (time.perf_counter() - start) * 1000
        model(test, dot=False)
        start = time.perf_counter()
        logits_test = model(test, dot=False)
        infer_ms = (time.perf_counter() - start) * 1000
        logits = [model(temp, dot=False), model(conf, dot=False), logits_test]
    return tuple(jnp.asarray(x.numpy()) for x in logits), fit_ms, infer_ms


def summarize(logits, labels, n_classes, alpha, fit_ms, infer_ms):
    logits_temp, logits_conf, logits_test = logits
    ytemp, yconf, ytest = map(jnp.asarray, labels)
    calibrator = TemperatureCalibrator.create().fit(logits_temp, ytemp, max_iters=500, lr=0.05)
    probs_raw = jax.nn.softmax(logits_test, axis=-1)
    probs_conf = calibrator.calibrate(logits_conf)
    probs_test = calibrator.calibrate(logits_test)
    conformal = ConformalClassifier.create(alpha=alpha).fit(probs_conf, yconf)
    return {
        "model": "cosine_centroid",
        "raw": asdict(_compute_metrics(probs_raw, ytest, n_classes)),
        "calibrated": asdict(_compute_metrics(probs_test, ytest, n_classes)),
        "temperature": float(calibrator.temperature),
        "conformal": {
            "alpha": alpha,
            "coverage": float(conformal.coverage(probs_test, ytest)),
            "set_size": float(conformal.set_size(probs_test)),
        },
        "train_ms": fit_ms,
        "infer_ms": infer_ms,
        "predictions": {
            "probs_cal": np.asarray(probs_conf).tolist(),
            "y_cal": yconf.tolist(),
            "probs_test": np.asarray(probs_test).tolist(),
            "y_test": ytest.tolist(),
        },
    }


def run_dataset(
    spec,
    X,
    y,
    dimensions=DEFAULT_DIMENSIONS,
    levels=DEFAULT_LEVELS,
    seed=DEFAULT_SEED,
    alpha=0.1,
    skip_torchhd=False,
):
    ids = split_indices(y, seed)
    # CPU is explicit for both libraries; defaults on GPU hosts cannot skew timings.
    with jax.default_device(jax.devices("cpu")[0]):
        hvs = encode_splits(spec, tuple(X[idx] for idx in ids), dimensions, levels, seed)
        logits, fit_ms, infer_ms = _jax_centroid(hvs, y[ids[0]], spec.n_classes)
        bh = summarize(logits, [y[i] for i in ids[1:]], spec.n_classes, alpha, fit_ms, infer_ms)
        th = None
        if not skip_torchhd:
            result = _torch_centroid(hvs, y[ids[0]], spec.n_classes)
            if result is not None:
                t_logits, t_fit, t_infer = result
                # Fail loudly if a future backend changes the supposedly matched workload.
                for a, b in zip(logits, t_logits):
                    np.testing.assert_allclose(a, b, atol=2e-5, rtol=2e-5)
                th = summarize(
                    t_logits, [y[i] for i in ids[1:]], spec.n_classes, alpha, t_fit, t_infer
                )
    return {
        "dataset": asdict(spec),
        "bayes_hdc": bh,
        "torchhd": th,
        "split_indices": dict(
            zip(("fit", "temperature", "conformal", "test"), [idx.tolist() for idx in ids])
        ),
        "config": {
            "dimensions": dimensions,
            "levels": levels,
            "seed": seed,
            "protocol": "matched-centroid-v2",
            "device": "cpu",
            "timing": "fit includes compilation; inference warm; encoding excluded",
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    ap.add_argument("--levels", type=int, default=DEFAULT_LEVELS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument(
        "--output", type=Path, default=Path("benchmarks/benchmark_calibration_results.json")
    )
    ap.add_argument("--skip-torchhd", action="store_true")
    ap.add_argument("--include-mnist", action="store_true")
    args = ap.parse_args()
    results = []
    for spec, X, y in _load_datasets(args.seed, args.include_mnist):
        row = run_dataset(
            spec, X, y, args.dimensions, args.levels, args.seed, args.alpha, args.skip_torchhd
        )
        row["provenance"] = provenance()
        results.append(row)
        print(f"{spec.name}: {row['bayes_hdc']['raw']} conformal={row['bayes_hdc']['conformal']}")
    write_json(args.output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
