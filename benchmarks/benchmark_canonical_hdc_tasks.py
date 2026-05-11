#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Bayes-HDC on the canonical HDC benchmark datasets, with calibration
and conformal coverage columns.

The library's existing ``benchmark_calibration.py`` reports calibration
+ coverage on iris / wine / breast-cancer / digits / MNIST — useful as
smoke checks, not as HDC-canonical numbers. This script targets the
datasets actually used in the HDC literature:

  ISOLET    — 26-class spoken-letter recognition, 617 features
              (Fanty & Cole 1990; canonical since Rahimi et al. 2016)
  UCI-HAR   — 6-class daily-living activity recognition, 561 features
              (Anguita et al. 2013)
  EMG       — multi-class hand-gesture EMG (Rahimi et al. 2016 HDC
              benchmark family)
  European
   Languages — 21-class character-trigram language ID
              (Joshi, Halseth, Kanerva 2016 — the headline HDC
              language-ID task)

For each dataset, the bayes-hdc pipeline runs:

  feature standardisation → KBinsDiscretizer →
  RandomEncoder (d=10000) → CentroidClassifier with LVQ refinement →
  TemperatureCalibrator (calibration set) →
  ConformalClassifier (α = 0.1)

and reports:

  accuracy, ECE (raw), ECE (post-temperature), Brier, NLL,
  conformal coverage at α=0.1, mean conformal set size

Network-gated: the canonical datasets are fetched via OpenML through
:mod:`bayes_hdc.datasets`. If a fetch fails (no network, OpenML down,
dataset renamed), that dataset is recorded with status ``unavailable``
and the others run unaffected. Re-run on a host with network access to
backfill.

Run::

    python benchmarks/benchmark_canonical_hdc_tasks.py

Writes ``benchmarks/canonical_hdc_results.json`` (gitignored; local
hardware varies).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

from bayes_hdc import (
    MAP,
    CentroidClassifier,
    ConformalClassifier,
    RandomEncoder,
    TemperatureCalibrator,
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
)

DIMENSIONS = 10_000
N_LEVELS = 64
SEED = 42
CAL_FRACTION = 0.3
ALPHA = 0.1


# ----------------------------------------------------------------------
# Dataset registry
# ----------------------------------------------------------------------


@dataclass
class DatasetSpec:
    name: str
    loader: Callable[[], Any]
    description: str


def _load_isolet():
    from bayes_hdc.datasets.loaders import load_isolet

    return load_isolet(test_size=0.25, random_state=SEED)


def _load_ucihar():
    from bayes_hdc.datasets.loaders import load_ucihar

    return load_ucihar(test_size=0.25, random_state=SEED)


def _load_emg():
    from bayes_hdc.datasets.loaders import load_emg

    return load_emg(test_size=0.25, random_state=SEED)


def _load_european_languages():
    from bayes_hdc.datasets.loaders import load_european_languages

    return load_european_languages(test_size=0.25, random_state=SEED)


REGISTRY = [
    DatasetSpec("isolet", _load_isolet, "ISOLET 26-class spoken-letter (Fanty & Cole 1990)"),
    DatasetSpec(
        "ucihar", _load_ucihar, "UCI-HAR 6-class activity recognition (Anguita et al. 2013)"
    ),
    DatasetSpec("emg", _load_emg, "EMG hand-gesture (Rahimi et al. 2016 HDC benchmark)"),
    DatasetSpec(
        "european_languages",
        _load_european_languages,
        "European Languages 21-class trigram (Joshi-Halseth-Kanerva 2016)",
    ),
]


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------


@dataclass
class TaskResult:
    name: str
    status: str
    description: str
    n_samples: int = 0
    n_features: int = 0
    n_classes: int = 0
    accuracy: float = float("nan")
    ece_raw: float = float("nan")
    ece_calibrated: float = float("nan")
    brier: float = float("nan")
    nll: float = float("nan")
    conformal_coverage: float = float("nan")
    conformal_set_size: float = float("nan")
    runtime_s: float = float("nan")
    error: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


def _run_one(spec: DatasetSpec) -> TaskResult:
    t0 = time.perf_counter()
    print(f"\n[{spec.name}] {spec.description}")
    print("  fetching ...", end=" ", flush=True)
    try:
        dataset = spec.loader()
    except Exception as exc:  # noqa: BLE001 — we want to keep the run alive on any fetch failure
        print(f"FAILED ({type(exc).__name__})")
        return TaskResult(
            name=spec.name,
            status="unavailable",
            description=spec.description,
            error=f"{type(exc).__name__}: {exc}",
        )
    print(f"ok ({dataset.x_train.shape[0]} train + {dataset.x_test.shape[0]} test)")

    X_train, y_train = np.asarray(dataset.x_train), np.asarray(dataset.y_train).astype(np.int64)
    X_test, y_test = np.asarray(dataset.x_test), np.asarray(dataset.y_test).astype(np.int64)
    n_classes = int(max(int(y_train.max()), int(y_test.max())) + 1)
    print(f"  shape: train={X_train.shape}, test={X_test.shape}, classes={n_classes}")

    # Re-label as contiguous integers if needed.
    classes = np.unique(np.concatenate([y_train, y_test]))
    class_index = {int(c): i for i, c in enumerate(classes)}
    y_train = np.array([class_index[int(c)] for c in y_train])
    y_test = np.array([class_index[int(c)] for c in y_test])
    n_classes = len(classes)

    # Hold out calibration set.
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_train, y_train, test_size=CAL_FRACTION, random_state=SEED, stratify=y_train
    )

    # Standardise + discretise — the canonical HDC preprocessing.
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_cal_s = scaler.transform(X_cal)
    X_test_s = scaler.transform(X_test)

    # KBinsDiscretizer can fail on some datasets if there are constant
    # columns; guard with a try / fallback to clipping into [0, N_LEVELS-1].
    try:
        disc = KBinsDiscretizer(n_bins=N_LEVELS, encode="ordinal", strategy="uniform").fit(X_tr_s)
        X_tr_d = disc.transform(X_tr_s).astype(np.int32)
        X_cal_d = disc.transform(X_cal_s).astype(np.int32)
        X_test_d = disc.transform(X_test_s).astype(np.int32)
    except ValueError as exc:
        # Fall back to a per-column uniform quantisation.
        print(f"  KBinsDiscretizer failed: {exc}; using uniform clip fallback")
        lo, hi = X_tr_s.min(axis=0), X_tr_s.max(axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)

        def _q(X):
            normed = (X - lo[None, :]) / span[None, :]
            return np.clip((normed * N_LEVELS).astype(np.int32), 0, N_LEVELS - 1)

        X_tr_d, X_cal_d, X_test_d = _q(X_tr_s), _q(X_cal_s), _q(X_test_s)

    n_features = X_tr_d.shape[1]
    key = jax.random.PRNGKey(SEED)

    # RandomEncoder + Centroid classifier.
    encoder = RandomEncoder.create(
        num_features=n_features,
        num_values=N_LEVELS,
        dimensions=DIMENSIONS,
        vsa_model=MAP.create(dimensions=DIMENSIONS),
        key=key,
    )
    enc_tr = encoder.encode_batch(jnp.asarray(X_tr_d))
    enc_cal = encoder.encode_batch(jnp.asarray(X_cal_d))
    enc_test = encoder.encode_batch(jnp.asarray(X_test_d))

    clf = CentroidClassifier.create(
        num_classes=n_classes, dimensions=DIMENSIONS, vsa_model=MAP.create(dimensions=DIMENSIONS)
    ).fit(enc_tr, jnp.asarray(y_tr))

    # Per-sample logits for the calibration + conformal pipeline.
    logits_cal = jax.vmap(clf.similarity)(enc_cal)
    logits_test = jax.vmap(clf.similarity)(enc_test)

    # Raw accuracy.
    preds = jnp.argmax(logits_test, axis=-1)
    accuracy = float(jnp.mean(preds == jnp.asarray(y_test)))

    # Raw probs → ECE / Brier / NLL.
    probs_test_raw = jax.nn.softmax(logits_test, axis=-1)
    ece_raw = float(expected_calibration_error(probs_test_raw, jnp.asarray(y_test)))

    # Temperature scaling on the calibration set.
    calibrator = TemperatureCalibrator.create().fit(logits_cal, jnp.asarray(y_cal))
    probs_cal = calibrator.calibrate(logits_cal)
    probs_test = calibrator.calibrate(logits_test)
    ece_cal = float(expected_calibration_error(probs_test, jnp.asarray(y_test)))
    brier = float(brier_score(probs_test, jnp.asarray(y_test)))
    nll = float(negative_log_likelihood(probs_test, jnp.asarray(y_test)))

    # Conformal layer.
    conformal = ConformalClassifier.create(alpha=ALPHA).fit(probs_cal, jnp.asarray(y_cal))
    sets = conformal.predict_set(probs_test)
    cov = float(jnp.mean(sets[jnp.arange(len(y_test)), jnp.asarray(y_test)]))
    set_size = float(jnp.mean(jnp.sum(sets.astype(jnp.float32), axis=-1)))

    runtime = time.perf_counter() - t0
    print(
        f"  acc={accuracy:.3f}  ECE_raw={ece_raw:.3f}  ECE_T={ece_cal:.3f}  "
        f"cov={cov:.3f}  set_size={set_size:.2f}  ({runtime:.1f}s)"
    )

    return TaskResult(
        name=spec.name,
        status="ok",
        description=spec.description,
        n_samples=int(X_train.shape[0]) + int(X_test.shape[0]),
        n_features=int(n_features),
        n_classes=int(n_classes),
        accuracy=accuracy,
        ece_raw=ece_raw,
        ece_calibrated=ece_cal,
        brier=brier,
        nll=nll,
        conformal_coverage=cov,
        conformal_set_size=set_size,
        runtime_s=runtime,
        extras={"temperature": float(calibrator.temperature)},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="comma-separated subset (e.g. 'isolet,ucihar') — defaults to all",
    )
    args = parser.parse_args()

    selected = set(args.only.split(",")) if args.only else None
    tasks = [s for s in REGISTRY if selected is None or s.name in selected]

    print("=" * 78)
    print("Bayes-HDC on canonical HDC benchmarks — calibration + coverage")
    print(f"d = {DIMENSIONS}    n_levels = {N_LEVELS}    α = {ALPHA}    seed = {SEED}")
    print("=" * 78)

    results = [_run_one(spec) for spec in tasks]

    print("\n" + "=" * 78)
    print("Summary")
    print("=" * 78)
    print(f"{'Task':<22} {'status':<12} {'acc':>6} {'ECE_T':>7} {'cov':>6} {'set':>6}")
    print("-" * 78)
    for r in results:
        if r.status == "ok":
            print(
                f"{r.name:<22} {r.status:<12} {r.accuracy:>6.3f} {r.ece_calibrated:>7.3f}"
                f" {r.conformal_coverage:>6.3f} {r.conformal_set_size:>6.2f}"
            )
        else:
            print(f"{r.name:<22} {r.status:<12} (error: {r.error.splitlines()[0][:40]})")

    out_path = Path(__file__).parent / "canonical_hdc_results.json"
    out_path.write_text(
        json.dumps(
            {
                "config": {
                    "dimensions": DIMENSIONS,
                    "n_levels": N_LEVELS,
                    "seed": SEED,
                    "alpha": ALPHA,
                    "cal_fraction": CAL_FRACTION,
                },
                "results": [asdict(r) for r in results],
            },
            indent=2,
            default=lambda o: float(o) if isinstance(o, (jnp.ndarray, np.ndarray)) else str(o),
        )
    )
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
