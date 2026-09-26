#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 R.S.
"""Canonical-data comparison of tuned random-feature centroid classifiers.

Compares bayes-hdc with TorchHD's Centroid model using explicitly constructed,
unit-normalized cosine random Fourier features. Both use the same RBF bandwidth
grid and model-selection budget; their framework-specific PRNG draws differ.
This is a matched random-feature centroid comparison, not a SOTA HDC comparison.
Temperature scaling and conformal prediction are generic post-hoc procedures
that can also wrap deterministic baselines. This script reports them for the
bayes-hdc model and reports point accuracy for the TorchHD reference.

ISOLET uses TorchHD's canonical 6238/1559 split. UCI-HAR uses the official
subject-disjoint 7352/2947 split. EMG uses label-pure 256-sample windows from the
original dataset.mat with a stratified 70/30 window split; it is not a subject-
or session-held-out evaluation.

Each training pool is split 70% proper training, 15% model selection and 15%
calibration, without label stratification. Preprocessing is fitted only on
proper training. Each framework selects gamma on model selection. Temperature
is fitted on calibration data solely for the separately reported temperature-
scaled probabilities/ECE. Conformal calibration and test sets use RAW model
probabilities, so their score function is independent of temperature fitting
even though the two post-hoc procedures share calibration rows. Do not interpret
this as conformal coverage for the temperature-scaled probabilities.

Coverage additionally requires exchangeable calibration/test scores, which
subject-disjoint or temporally grouped data alone do not establish. Test data
is reserved for final evaluation. Per-seed splits, predictions, probabilities,
calibration parameters and processed-data hashes are retained with mean/std
summaries for independent recomputation.

    uv run --with scikit-learn --with torch --with torch-hd --with gdown \
        python benchmarks/benchmark_canonical.py

Writes benchmarks/canonical_results.json (gitignored).
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from bayes_hdc import expected_calibration_error
from bayes_hdc.sklearn import HDClassifier
from bayes_hdc.uncertainty import ConformalClassifier, TemperatureCalibrator

if __package__:
    from ._common import provenance, standardize_train, write_json
else:
    from _common import provenance, standardize_train, write_json

DIMS = 10000
ALPHA = 0.1
SEEDS = [0, 1, 2, 3, 4]
DATA_ROOT = os.path.expanduser("~/.cache/bayes_hdc_data")


# --------------------------------------------------------------------------
# Data loading (via TorchHD loaders -> numpy)
# --------------------------------------------------------------------------
def load_isolet():
    """ISOLET: 617 spoken-letter features, 26 classes, canonical 6238/1559 split."""
    from torchhd.datasets import ISOLET

    tr = ISOLET(DATA_ROOT, train=True, download=True)
    te = ISOLET(DATA_ROOT, train=False, download=True)
    Xtr = np.stack([tr[i][0].numpy() for i in range(len(tr))]).astype(np.float32)
    ytr = np.array([int(tr[i][1]) for i in range(len(tr))], dtype=np.int64)
    Xte = np.stack([te[i][0].numpy() for i in range(len(te))]).astype(np.float32)
    yte = np.array([int(te[i][1]) for i in range(len(te))], dtype=np.int64)
    return "ISOLET", Xtr, ytr, Xte, yte


def load_emg():
    """EMG hand gestures (Rahimi et al. 2016a): 4-channel windows, 5 classes.

    Uses bayes_hdc's own loader, which fetches the original authors'
    dataset.mat and cuts label-pure 256-sample windows; stratified 70/30
    split (no canonical split ships with the dataset).
    """
    from bayes_hdc.datasets import load_emg as _load

    ds = _load()
    return "EMG", ds.X_train, ds.y_train.astype(np.int64), ds.X_test, ds.y_test.astype(np.int64)


def load_ucihar():
    """UCI-HAR (Anguita et al. 2013): 6-class activity recognition, 561 features."""
    from bayes_hdc.datasets import load_ucihar as _load

    ds = _load()
    return "UCI-HAR", ds.X_train, ds.y_train.astype(np.int64), ds.X_test, ds.y_test.astype(np.int64)


# --------------------------------------------------------------------------
# TorchHD centroid on explicit RBF random Fourier features, same tuning budget
# --------------------------------------------------------------------------
def _torchhd_fit_eval(Xtr, ytr, Xev, n_classes, seed, scale, dims=DIMS):
    """Train a TorchHD centroid over RBF random Fourier features; return preds.

    Scaling the standardized inputs by sqrt(2*gamma) makes these cosine
    random Fourier features approximate the RBF kernel at bandwidth gamma,
    the same family bayes-hdc's KernelEncoder searches over.
    """
    import torch
    from torchhd.models import Centroid

    torch.manual_seed(seed)
    frequencies = torch.randn(Xtr.shape[1], dims)
    phases = torch.rand(dims) * (2 * torch.pi)

    def encode(X):
        hv = np.sqrt(2.0 / dims) * torch.cos(torch.as_tensor(X * scale) @ frequencies + phases)
        return hv / (torch.linalg.vector_norm(hv, dim=-1, keepdim=True) + 1e-8)

    with torch.no_grad():
        tr_hv = encode(Xtr)
        ev_hv = encode(Xev)
        model = Centroid(dims, n_classes)
        model.add(tr_hv, torch.as_tensor(ytr))
        model.normalize()  # required, else the majority class dominates
        return model(ev_hv).argmax(1).numpy()


def torchhd_centroid_accuracy(
    Xtr, ytr, Xsel, ysel, Xte, yte, n_classes, seed, dims=DIMS, *, return_details=False
):
    """Tuned TorchHD reference: bandwidth picked on the model-selection split."""
    best_scale, best_gamma, best_acc = None, None, -1.0
    for g in GAMMA_GRID:
        scale = float(np.sqrt(2.0 * g))
        preds = _torchhd_fit_eval(Xtr, ytr, Xsel, n_classes, seed, scale, dims)
        acc = float((preds == ysel).mean())
        if acc > best_acc:
            best_scale, best_gamma, best_acc = scale, g, acc
    preds = _torchhd_fit_eval(Xtr, ytr, Xte, n_classes, seed, best_scale, dims)
    accuracy = float((preds == yte).mean())
    if return_details:
        return {
            "accuracy": accuracy,
            "gamma": best_gamma,
            "selection_accuracy": best_acc,
            "test_predictions": preds.tolist(),
        }
    return accuracy


# --------------------------------------------------------------------------
# One dataset, one seed
# --------------------------------------------------------------------------
# RBF bandwidths searched on the model-selection split (never on test,
# never on the conformal-calibration slice). Both libraries get the same
# grid and the same RBF feature distribution. PRNG realizations differ across
# frameworks; this compares tuned random-feature centroids, not SOTA HDC training.
GAMMA_GRID = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.5]


def select_gamma(Xtr, ytr, Xval, yval, seed):
    """Pick the RFF bandwidth with the best model-selection-split accuracy."""
    best_g, best_acc = GAMMA_GRID[0], -1.0
    for g in GAMMA_GRID:
        clf = HDClassifier(dimensions=DIMS, encoder="kernel", gamma=g, random_state=seed).fit(
            Xtr, ytr
        )
        acc = float((clf.predict(Xval) == yval).mean())
        if acc > best_acc:
            best_g, best_acc = g, acc
    return best_g


def run_once(Xtr_full, ytr_full, Xte, yte, n_classes, seed):
    from sklearn.model_selection import train_test_split

    # Save indices in their actual fitting order, relative to the supplied pool.
    fit_idx, hold_idx = train_test_split(np.arange(len(ytr_full)), test_size=0.3, random_state=seed)
    selection_idx, calibration_idx = train_test_split(
        hold_idx, test_size=0.5, random_state=seed + 1
    )
    Xtr, ytr = Xtr_full[fit_idx], ytr_full[fit_idx]
    Xsel, ysel = Xtr_full[selection_idx], ytr_full[selection_idx]
    Xcal, ycal = Xtr_full[calibration_idx], ytr_full[calibration_idx]
    # Fit preprocessing only after separating model selection and calibration.
    Xtr, Xsel, Xcal, Xte_s = standardize_train(Xtr, Xsel, Xcal, Xte)

    gamma = select_gamma(Xtr, ytr, Xsel, ysel, seed)
    clf = HDClassifier(dimensions=DIMS, encoder="kernel", gamma=gamma, random_state=seed).fit(
        Xtr, ytr
    )
    proba_te = np.asarray(clf.predict_proba(Xte_s))
    predictions_te = np.asarray(clf.predict(Xte_s))
    acc = float((predictions_te == yte).mean())
    ece_raw = float(expected_calibration_error(jnp.asarray(proba_te), jnp.asarray(yte)))

    proba_cal = np.asarray(clf.predict_proba(Xcal))
    logits_cal = jnp.log(proba_cal + 1e-9)
    logits_te = jnp.log(proba_te + 1e-9)
    temp = TemperatureCalibrator.create().fit(logits_cal, jnp.asarray(ycal))
    proba_cal_temp = np.asarray(temp.calibrate(logits_cal))
    proba_te_cal = np.asarray(temp.calibrate(logits_te))
    ece_cal = float(expected_calibration_error(jnp.asarray(proba_te_cal), jnp.asarray(yte)))

    # Raw APS scores are untouched by the temperature fitted above.
    conf = ConformalClassifier.create(alpha=ALPHA).fit(jnp.asarray(proba_cal), jnp.asarray(ycal))
    sets = np.asarray(conf.predict_set(jnp.asarray(proba_te)))
    coverage = float(sets[np.arange(len(yte)), yte].mean())
    set_size = float(sets.sum(1).mean())

    th_details = None
    try:
        th_details = torchhd_centroid_accuracy(
            Xtr, ytr, Xsel, ysel, Xte_s, yte, n_classes, seed, dims=DIMS, return_details=True
        )
        th_acc = th_details["accuracy"]
    except Exception as e:  # noqa: BLE001
        th_acc = {"error": f"{type(e).__name__}: {e}"}

    return {
        "seed": int(seed),
        "split_indices": {
            "index_reference": (
                "fit/selection/calibration index the supplied training pool; "
                "test indexes the supplied test array"
            ),
            "fit": fit_idx.tolist(),
            "model_selection": selection_idx.tolist(),
            "calibration": calibration_idx.tolist(),
            "test": np.arange(len(yte)).tolist(),
        },
        "labels": {
            "fit": ytr.tolist(),
            "model_selection": ysel.tolist(),
            "calibration": ycal.tolist(),
            "test": yte.tolist(),
        },
        "class_order": clf.classes_.tolist(),
        "predictions": {
            "calibration_raw": clf.classes_[proba_cal.argmax(axis=1)].tolist(),
            "test_raw": predictions_te.tolist(),
            "calibration_temperature_only": clf.classes_[proba_cal_temp.argmax(axis=1)].tolist(),
            "test_temperature_only": clf.classes_[proba_te_cal.argmax(axis=1)].tolist(),
        },
        "probabilities": {
            "calibration_raw": proba_cal.tolist(),
            "test_raw": proba_te.tolist(),
            "calibration_temperature_only": proba_cal_temp.tolist(),
            "test_temperature_only": proba_te_cal.tolist(),
        },
        "temperature": float(temp.temperature),
        "q_hat": float(conf.threshold),
        "q_hat_is_infinite": bool(jnp.isinf(conf.threshold)),
        "conformal_score_input": "raw probabilities; independent of fitted temperature",
        "test_prediction_sets": sets.tolist(),
        "torchhd_details": th_details,
        "hd_acc": acc,
        "ece_raw": ece_raw,
        "ece_cal": ece_cal,
        "coverage": coverage,
        "set_size": set_size,
        "torchhd_acc": th_acc,
        "gamma": gamma,
    }


def _array_fingerprint(values):
    """Hash processed array dtype, shape and C-order bytes, preserving row order."""
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return {"sha256": digest.hexdigest(), "dtype": array.dtype.str, "shape": list(array.shape)}


def aggregate(name, Xtr, ytr, Xte, yte):
    n_classes = int(max(ytr.max(), yte.max()) + 1)
    runs = [run_once(Xtr, ytr, Xte, yte, n_classes, s) for s in SEEDS]

    def ms(key):
        vals = [r[key] for r in runs]
        return float(np.mean(vals)), float(np.std(vals))

    th = [r["torchhd_acc"] for r in runs if isinstance(r["torchhd_acc"], float)]
    th_mean = float(np.mean(th)) if len(th) == len(runs) else None
    th_std = float(np.std(th)) if len(th) == len(runs) else None

    acc_m, acc_s = ms("hd_acc")
    er_m, _ = ms("ece_raw")
    ec_m, _ = ms("ece_cal")
    cov_m, _ = ms("coverage")
    sz_m, _ = ms("set_size")
    row = {
        "dataset": name,
        "data_fingerprints": {
            "X_train_pool": _array_fingerprint(Xtr),
            "y_train_pool": _array_fingerprint(ytr),
            "X_test": _array_fingerprint(Xte),
            "y_test": _array_fingerprint(yte),
        },
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "features": int(Xtr.shape[1]),
        "classes": n_classes,
        "hd_acc_mean": round(acc_m, 4),
        "hd_acc_std": round(acc_s, 4),
        "ece_raw_mean": round(er_m, 4),
        "ece_cal_mean": round(ec_m, 4),
        "coverage_mean": round(cov_m, 4),
        "set_size_mean": round(sz_m, 3),
        "torchhd_acc_mean": None if th_mean is None else round(th_mean, 4),
        "torchhd_acc_std": None if th_std is None else round(th_std, 4),
        "gammas": [r["gamma"] for r in runs],
        "runs": runs,
        "torchhd_successful_runs": len(th),
    }
    print(
        f"[{name:7s}] HD acc={acc_m:.3f}+/-{acc_s:.3f}  "
        f"ECE {er_m:.3f}->{ec_m:.3f}  cov={cov_m:.3f} |C|={sz_m:.2f}  "
        f"TorchHD={row['torchhd_acc_mean']}"
    )
    return row


def main():
    import argparse

    global DIMS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, default=DIMS)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "canonical_results.json"
    )
    args = parser.parse_args()
    DIMS = args.dimensions
    print(f"canonical HDC benchmark (d={DIMS}, alpha={ALPHA}, seeds={SEEDS})")
    t0 = time.perf_counter()
    rows = []
    for loader in (load_isolet, load_ucihar, load_emg):
        name, Xtr, ytr, Xte, yte = loader()
        rows.append(aggregate(name, Xtr, ytr, Xte, yte))
    out = {
        "config": {
            "dimensions": DIMS,
            "protocol": "training-only-preprocessing-raw-conformal-v3",
            "conformal_input": "raw model probabilities, never temperature-scaled",
            "temperature_input": "same calibration rows, separate post-hoc ECE evaluation",
            "alpha": ALPHA,
            "seeds": SEEDS,
            "gamma_grid": GAMMA_GRID,
            "split": "train pool -> 70% fit / 15% model-selection / 15% calibration",
        },
        "provenance": provenance(),
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "results": rows,
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    p = args.output
    write_json(p, out)
    print(f"\nwrote {p} ({out['runtime_s']}s)")


if __name__ == "__main__":
    main()
