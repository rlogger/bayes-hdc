#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 R.S.
"""Canonical HDC benchmark: ISOLET, UCI-HAR, and EMG hand gestures — the
datasets the HDC literature reports on (Rahimi et al. 2016; Imani et al.
2017; Anguita et al. 2013). Compares bayes-hdc against a faithful TorchHD
centroid (Sinusoid embedding, the matching random-Fourier encoder) on
identical splits, and reports the calibration and conformal-coverage metrics
that the deterministic baseline does not provide.

ISOLET is fetched with TorchHD's dataset loader (canonical 6238/1559 split);
UCI-HAR uses the official subject-disjoint 7352/2947 split via
bayes_hdc.datasets; EMG uses the original Rahimi dataset.mat with a
stratified 70/30 window split (no canonical split ships with it).

Protocol: the training pool is split three ways (70% fit / 15%
model-selection / 15% conformal+temperature calibration), all stratified.
BOTH encoders get the same RBF-bandwidth search on the model-selection
split — bayes-hdc via KernelEncoder's gamma, TorchHD via the equivalent
input-scaling of its Sinusoid embedding — and the conformal/temperature
calibration uses data disjoint from everything else, so the finite-sample
coverage guarantee is intact. Test data is never touched before the final
evaluation. Runs over several seeds (the seed controls the random codebook)
and reports mean +/- std.

    uv run --with scikit-learn --with torch --with torch-hd --with gdown \
        python benchmarks/benchmark_canonical.py

Writes benchmarks/canonical_results.json (gitignored).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from bayes_hdc import expected_calibration_error
from bayes_hdc.sklearn import HDClassifier
from bayes_hdc.uncertainty import ConformalClassifier, TemperatureCalibrator

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
# TorchHD reference (centroid over Sinusoid, same tuning budget)
# --------------------------------------------------------------------------
def _torchhd_fit_eval(Xtr, ytr, Xev, n_classes, seed, scale, dims=DIMS):
    """Train a TorchHD Sinusoid+Centroid at the given input scale; return preds.

    Scaling the standardized inputs by sqrt(2*gamma) makes the Sinusoid
    random-Fourier embedding approximate the RBF kernel at bandwidth gamma,
    the same family bayes-hdc's KernelEncoder searches over.
    """
    import torch
    from torchhd import embeddings
    from torchhd.models import Centroid

    torch.manual_seed(seed)
    enc = embeddings.Sinusoid(Xtr.shape[1], dims)
    with torch.no_grad():
        tr_hv = enc(torch.as_tensor(Xtr * scale))
        ev_hv = enc(torch.as_tensor(Xev * scale))
        model = Centroid(dims, n_classes)
        model.add(tr_hv, torch.as_tensor(ytr))
        model.normalize()  # required, else the majority class dominates
        return model(ev_hv).argmax(1).numpy()


def torchhd_centroid_accuracy(Xtr, ytr, Xsel, ysel, Xte, yte, n_classes, seed, dims=DIMS):
    """Tuned TorchHD reference: bandwidth picked on the model-selection split."""
    best_scale, best_acc = None, -1.0
    for g in GAMMA_GRID:
        scale = float(np.sqrt(2.0 * g))
        preds = _torchhd_fit_eval(Xtr, ytr, Xsel, n_classes, seed, scale, dims)
        acc = float((preds == ysel).mean())
        if acc > best_acc:
            best_scale, best_acc = scale, acc
    preds = _torchhd_fit_eval(Xtr, ytr, Xte, n_classes, seed, best_scale, dims)
    return float((preds == yte).mean())


# --------------------------------------------------------------------------
# One dataset, one seed
# --------------------------------------------------------------------------
# RBF bandwidths searched on the model-selection split (never on test,
# never on the conformal-calibration slice). Both libraries get the same
# grid; gamma = 0.5 corresponds to input scale 1.0, i.e. TorchHD's
# untuned Sinusoid default, so neither library's natural operating point
# is excluded from the search.
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
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(Xtr_full)
    Xtr_full_s = scaler.transform(Xtr_full).astype(np.float32)
    Xte_s = scaler.transform(Xte).astype(np.float32)
    # Three-way split of the training pool: fit / model-selection /
    # conformal+temperature calibration. Keeping the selection and
    # calibration slices disjoint preserves the conformal coverage
    # guarantee; test data is never touched before final evaluation.
    Xtr, Xhold, ytr, yhold = train_test_split(
        Xtr_full_s, ytr_full, test_size=0.3, random_state=seed, stratify=ytr_full
    )
    Xsel, Xcal, ysel, ycal = train_test_split(
        Xhold, yhold, test_size=0.5, random_state=seed, stratify=yhold
    )

    gamma = select_gamma(Xtr, ytr, Xsel, ysel, seed)
    clf = HDClassifier(dimensions=DIMS, encoder="kernel", gamma=gamma, random_state=seed).fit(
        Xtr, ytr
    )
    proba_te = np.asarray(clf.predict_proba(Xte_s))
    acc = float((clf.predict(Xte_s) == yte).mean())
    ece_raw = float(expected_calibration_error(jnp.asarray(proba_te), jnp.asarray(yte)))

    logits_cal = jnp.log(np.asarray(clf.predict_proba(Xcal)) + 1e-9)
    logits_te = jnp.log(proba_te + 1e-9)
    temp = TemperatureCalibrator.create().fit(logits_cal, jnp.asarray(ycal))
    proba_te_cal = np.asarray(temp.calibrate(logits_te))
    ece_cal = float(expected_calibration_error(jnp.asarray(proba_te_cal), jnp.asarray(yte)))

    proba_cal = np.asarray(clf.predict_proba(Xcal))
    conf = ConformalClassifier.create(alpha=ALPHA).fit(jnp.asarray(proba_cal), jnp.asarray(ycal))
    sets = np.asarray(conf.predict_set(jnp.asarray(proba_te)))
    coverage = float(sets[np.arange(len(yte)), yte].mean())
    set_size = float(sets.sum(1).mean())

    th_acc = None
    try:
        th_acc = torchhd_centroid_accuracy(Xtr, ytr, Xsel, ysel, Xte_s, yte, n_classes, seed)
    except Exception as e:  # noqa: BLE001
        th_acc = {"error": f"{type(e).__name__}: {e}"}

    return {
        "hd_acc": acc,
        "ece_raw": ece_raw,
        "ece_cal": ece_cal,
        "coverage": coverage,
        "set_size": set_size,
        "torchhd_acc": th_acc,
        "gamma": gamma,
    }


def aggregate(name, Xtr, ytr, Xte, yte):
    n_classes = int(max(ytr.max(), yte.max()) + 1)
    runs = [run_once(Xtr, ytr, Xte, yte, n_classes, s) for s in SEEDS]

    def ms(key):
        vals = [r[key] for r in runs]
        return float(np.mean(vals)), float(np.std(vals))

    th = [r["torchhd_acc"] for r in runs if isinstance(r["torchhd_acc"], float)]
    th_mean = float(np.mean(th)) if th else None
    th_std = float(np.std(th)) if th else None

    acc_m, acc_s = ms("hd_acc")
    er_m, _ = ms("ece_raw")
    ec_m, _ = ms("ece_cal")
    cov_m, _ = ms("coverage")
    sz_m, _ = ms("set_size")
    row = {
        "dataset": name,
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
    }
    print(
        f"[{name:7s}] HD acc={acc_m:.3f}+/-{acc_s:.3f}  "
        f"ECE {er_m:.3f}->{ec_m:.3f}  cov={cov_m:.3f} |C|={sz_m:.2f}  "
        f"TorchHD={row['torchhd_acc_mean']}"
    )
    return row


def provenance():
    """Environment provenance, recorded into the results JSON."""
    import platform
    import subprocess
    import sys

    import jax

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            check=False,
        ).stdout.strip()
    except OSError:
        commit = "unknown"
    try:
        import torchhd

        torchhd_v = torchhd.__version__
    except Exception:  # noqa: BLE001
        torchhd_v = None
    return {
        "commit": commit or "unknown",
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "torchhd": torchhd_v,
        "platform": platform.platform(),
    }


def main():
    print(f"canonical HDC benchmark (d={DIMS}, alpha={ALPHA}, seeds={SEEDS})")
    t0 = time.perf_counter()
    rows = []
    for loader in (load_isolet, load_ucihar, load_emg):
        name, Xtr, ytr, Xte, yte = loader()
        rows.append(aggregate(name, Xtr, ytr, Xte, yte))
    out = {
        "config": {
            "dimensions": DIMS,
            "alpha": ALPHA,
            "seeds": SEEDS,
            "gamma_grid": GAMMA_GRID,
            "split": "train pool -> 70% fit / 15% model-selection / 15% calibration",
        },
        "provenance": provenance(),
        "results": rows,
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    p = Path(__file__).parent / "canonical_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p} ({out['runtime_s']}s)")


if __name__ == "__main__":
    main()
