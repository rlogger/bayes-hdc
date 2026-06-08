#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 R.S.
"""Reproducible benchmark for the paper: classification (accuracy + ECE +
conformal coverage) and one-class anomaly detection (AUROC + FPR@alpha) for
bayes-hdc against scikit-learn baselines and, when installed, TorchHD.

Datasets are scikit-learn built-ins (no network): digits, breast_cancer, wine.
Writes benchmarks/paper_results.json. Run:

    uv run --with scikit-learn [--with torch --with torch-hd] \
        python benchmarks/benchmark_paper.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from bayes_hdc import expected_calibration_error
from bayes_hdc.sklearn import HDAnomalyDetector, HDClassifier
from bayes_hdc.uncertainty import ConformalClassifier, TemperatureCalibrator

SEED = 0
DIMS = 10000
ALPHA = 0.1
DATASETS = {
    "digits": load_digits,
    "breast_cancer": load_breast_cancer,
    "wine": load_wine,
}


def _try_torchhd_accuracy(Xtr, ytr, Xte, yte, n_classes, dims=DIMS):
    """TorchHD centroid (RecordEncoder + Centroid) accuracy, or None."""
    try:
        import torch
        import torchhd
        from torchhd import embeddings
        from torchhd.models import Centroid
    except Exception:
        return None
    try:
        torch.manual_seed(SEED)
        d = dims
        n_feat = Xtr.shape[1]
        # Level-encode standardized features into [0, n_levels) ids, project.
        n_levels = 100
        lo = Xtr.min(0, keepdims=True)
        hi = Xtr.max(0, keepdims=True)
        span = np.where(hi > lo, hi - lo, 1.0)

        def to_levels(X):
            z = np.clip((X - lo) / span, 0, 1)
            return torch.as_tensor((z * (n_levels - 1)).astype(np.int64))

        levels = embeddings.Level(n_levels, d)
        feats = embeddings.Random(n_feat, d)

        def encode(ids):
            # bundle over features of (level[value] * feat[i])
            sample_hv = levels(ids) * feats.weight.unsqueeze(0)
            return torchhd.multiset(sample_hv)

        with torch.no_grad():
            tr_hv = encode(to_levels(Xtr))
            te_hv = encode(to_levels(Xte))
            model = Centroid(d, n_classes)
            model.add(tr_hv, torch.as_tensor(ytr.astype(np.int64)))
            model.normalize()  # standard TorchHD step; without it majority class dominates
            preds = model(te_hv).argmax(1).numpy()
        return float((preds == yte).mean())
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def classification_benchmark():
    rows = []
    for name, loader in DATASETS.items():
        data = loader()
        X = StandardScaler().fit_transform(data.data.astype(np.float32))
        y = data.target.astype(np.int64)
        n_classes = int(y.max() + 1)
        Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, random_state=SEED, stratify=y)
        Xcal, Xte, ycal, yte = train_test_split(
            Xtmp, ytmp, test_size=0.5, random_state=SEED, stratify=ytmp
        )

        clf = HDClassifier(dimensions=DIMS, random_state=SEED).fit(Xtr, ytr)
        proba_te = np.asarray(clf.predict_proba(Xte))
        acc = float((clf.predict(Xte) == yte).mean())
        ece_raw = float(expected_calibration_error(jnp.asarray(proba_te), jnp.asarray(yte)))

        # The library's calibration story: fit a temperature on the
        # calibration split, then re-measure ECE on test. We recover the
        # pre-softmax similarity logits by log of the reported probabilities.
        logits_cal = jnp.log(np.asarray(clf.predict_proba(Xcal)) + 1e-9)
        logits_te = jnp.log(proba_te + 1e-9)
        temp = TemperatureCalibrator.create().fit(logits_cal, jnp.asarray(ycal))
        proba_te_cal = np.asarray(temp.calibrate(logits_te))
        ece_cal = float(expected_calibration_error(jnp.asarray(proba_te_cal), jnp.asarray(yte)))

        # Conformal coverage at alpha using the calibration split.
        proba_cal = np.asarray(clf.predict_proba(Xcal))
        conf = ConformalClassifier.create(alpha=ALPHA).fit(
            jnp.asarray(proba_cal), jnp.asarray(ycal)
        )
        sets = np.asarray(conf.predict_set(jnp.asarray(proba_te)))
        coverage = float(sets[np.arange(len(yte)), yte].mean())
        set_size = float(sets.sum(1).mean())

        # Fair reference baselines on identical splits.
        logreg = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
        logreg_acc = float((logreg.predict(Xte) == yte).mean())
        torchhd_acc = _try_torchhd_accuracy(Xtr, ytr, Xte, yte, n_classes)

        rows.append(
            {
                "dataset": name,
                "n": int(X.shape[0]),
                "features": int(X.shape[1]),
                "classes": n_classes,
                "hd_accuracy": round(acc, 4),
                "hd_ece_raw": round(ece_raw, 4),
                "hd_ece_calibrated": round(ece_cal, 4),
                "conformal_coverage": round(coverage, 4),
                "conformal_set_size": round(set_size, 3),
                "logreg_accuracy": round(logreg_acc, 4),
                "torchhd_accuracy": torchhd_acc,
            }
        )
        print(
            f"[cls] {name:14s} acc={acc:.3f} ece {ece_raw:.3f}->{ece_cal:.3f} "
            f"cov={coverage:.3f} |C|={set_size:.2f} logreg={logreg_acc:.3f} "
            f"torchhd={torchhd_acc}"
        )
    return rows


def anomaly_benchmark():
    rows = []
    for name, loader in DATASETS.items():
        data = loader()
        X = StandardScaler().fit_transform(data.data.astype(np.float32))
        y = data.target.astype(np.int64)
        normal_cls = int(np.bincount(y).argmax())
        is_norm = y == normal_cls

        Xn = X[is_norm]
        Xa = X[~is_norm]
        Xn_tr, Xn_te = train_test_split(Xn, test_size=0.4, random_state=SEED)
        # Test set: held-out normal (label 0) + all anomalies (label 1).
        X_test = np.vstack([Xn_te, Xa])
        y_test = np.concatenate([np.zeros(len(Xn_te)), np.ones(len(Xa))]).astype(int)

        # bayes-hdc: higher p-value = more normal, so anomaly score = -pvalue.
        det = HDAnomalyDetector(alpha=ALPHA, dimensions=DIMS, random_state=SEED).fit(Xn_tr)
        pv = det.pvalue(X_test)
        hd_auroc = float(roc_auc_score(y_test, -pv))
        hd_fpr = float((det.predict(Xn_te) == -1).mean())  # FP rate on normal

        # sklearn baselines: decision_function higher = more normal → negate.
        baselines = {}
        for bname, model in {
            "IsolationForest": IsolationForest(random_state=SEED),
            "LOF": LocalOutlierFactor(novelty=True),
            "OneClassSVM": OneClassSVM(gamma="scale"),
        }.items():
            try:
                model.fit(Xn_tr)
                score = -model.decision_function(X_test)
                baselines[bname] = round(float(roc_auc_score(y_test, score)), 4)
            except Exception as e:  # noqa: BLE001
                baselines[bname] = f"err:{type(e).__name__}"

        rows.append(
            {
                "dataset": name,
                "normal_class": normal_cls,
                "n_normal_train": int(len(Xn_tr)),
                "n_test": int(len(y_test)),
                "n_anomalies": int((~is_norm).sum()),
                "hd_auroc": round(hd_auroc, 4),
                "hd_fpr_at_alpha": round(hd_fpr, 4),
                "baselines_auroc": baselines,
            }
        )
        print(f"[anom] {name:14s} HD_AUROC={hd_auroc:.3f} fpr@a={hd_fpr:.3f} baselines={baselines}")
    return rows


def main():
    print(f"bayes-hdc paper benchmark (d={DIMS}, alpha={ALPHA}, seed={SEED})")
    print(f"jax backend: {jax.default_backend()}")
    t0 = time.perf_counter()
    cls = classification_benchmark()
    anom = anomaly_benchmark()
    out = {
        "config": {"dimensions": DIMS, "alpha": ALPHA, "seed": SEED},
        "classification": cls,
        "anomaly": anom,
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    p = Path(__file__).parent / "paper_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p} ({out['runtime_s']}s)")


if __name__ == "__main__":
    main()
