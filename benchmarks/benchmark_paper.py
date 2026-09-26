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
from sklearn.svm import OneClassSVM

from bayes_hdc import ProjectionEncoder, expected_calibration_error, fit_anomaly_pipeline
from bayes_hdc.sklearn import HDClassifier
from bayes_hdc.uncertainty import ConformalClassifier, TemperatureCalibrator

if __package__:
    from ._common import provenance, standardize_train, write_json
else:
    from _common import provenance, standardize_train, write_json

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
    except ImportError:
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
        X = data.data.astype(np.float32)
        y = data.target.astype(np.int64)
        n_classes = int(y.max() + 1)
        Xtr, Xtmp, ytr, ytmp = train_test_split(
            X,
            y,
            test_size=0.4,
            random_state=SEED,
        )
        Xcal, Xte, ycal, yte = train_test_split(
            Xtmp,
            ytmp,
            test_size=0.5,
            random_state=SEED,
        )

        Xtr, Xcal, Xte = standardize_train(Xtr, Xcal, Xte)
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
        torchhd_acc = _try_torchhd_accuracy(Xtr, ytr, Xte, yte, n_classes, dims=DIMS)

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


ANOM_SEEDS = [0, 1, 2, 3, 4]


def anomaly_benchmark():
    """One-class anomaly detection over ANOM_SEEDS; the seed controls the
    train/test split, the random codebook, and the IsolationForest."""
    rows = []
    for name, loader in DATASETS.items():
        data = loader()
        X = data.data.astype(np.float32)
        y = data.target.astype(np.int64)
        normal_cls = int(np.bincount(y).argmax())
        is_norm = y == normal_cls
        Xn = X[is_norm]
        Xa = X[~is_norm]

        per_seed = {
            "hd_auroc": [],
            "hd_fpr": [],
            "IsolationForest": [],
            "LOF": [],
            "OneClassSVM": [],
        }
        for seed in ANOM_SEEDS:
            Xn_tr, Xn_te = train_test_split(Xn, test_size=0.4, random_state=seed)
            # Test set: held-out normal (label 0) + all anomalies (label 1).
            X_test = np.vstack([Xn_te, Xa])
            y_test = np.concatenate([np.zeros(len(Xn_te)), np.ones(len(Xa))]).astype(int)

            # Hold out calibration BEFORE learning any preprocessing.
            Xfit, Xcal = train_test_split(Xn_tr, test_size=0.3, random_state=seed)
            Xfit, Xcal, X_test, Xn_te = standardize_train(Xfit, Xcal, X_test, Xn_te)
            encoder = ProjectionEncoder.create(
                input_dim=Xfit.shape[1], dimensions=DIMS, key=jax.random.PRNGKey(seed)
            )
            det = fit_anomaly_pipeline(encoder, jnp.asarray(Xfit), jnp.asarray(Xcal), alpha=ALPHA)
            test_hv = encoder.encode_batch(jnp.asarray(X_test))
            # Rank with the raw score; p-value discretization creates avoidable ties.
            scores = np.asarray(det.scorer.score_batch(test_hv))
            per_seed["hd_auroc"].append(float(roc_auc_score(y_test, scores)))
            normal_hv = encoder.encode_batch(jnp.asarray(Xn_te))
            per_seed["hd_fpr"].append(
                float(np.asarray(det.predict_batch(normal_hv, alpha=ALPHA)).mean())
            )

            # sklearn baselines: decision_function higher = more normal → negate.
            for bname, model in {
                "IsolationForest": IsolationForest(random_state=seed),
                "LOF": LocalOutlierFactor(novelty=True),
                "OneClassSVM": OneClassSVM(gamma="scale"),
            }.items():
                model.fit(Xfit)
                score = -model.decision_function(X_test)
                per_seed[bname].append(float(roc_auc_score(y_test, score)))

        def ms(key):
            return round(float(np.mean(per_seed[key])), 4), round(float(np.std(per_seed[key])), 4)

        hd_m, hd_s = ms("hd_auroc")
        fpr_m, fpr_s = ms("hd_fpr")
        rows.append(
            {
                "dataset": name,
                "normal_class": normal_cls,
                "n_anomalies": int((~is_norm).sum()),
                "seeds": ANOM_SEEDS,
                "hd_auroc_mean": hd_m,
                "hd_auroc_std": hd_s,
                "hd_fpr_at_alpha_mean": fpr_m,
                "hd_fpr_at_alpha_std": fpr_s,
                "baselines_auroc": {
                    b: {"mean": ms(b)[0], "std": ms(b)[1]}
                    for b in ("IsolationForest", "LOF", "OneClassSVM")
                },
            }
        )
        print(
            f"[anom] {name:14s} HD={hd_m:.3f}+/-{hd_s:.3f} fpr@a={fpr_m:.3f} "
            + " ".join(f"{b}={ms(b)[0]:.3f}" for b in ("IsolationForest", "LOF", "OneClassSVM"))
        )
    return rows


def main():
    import argparse

    global DIMS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, default=DIMS)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "paper_results.json")
    args = parser.parse_args()
    DIMS = args.dimensions
    print(f"bayes-hdc paper benchmark (d={DIMS}, alpha={ALPHA}, seed={SEED})")
    print(f"jax backend: {jax.default_backend()}")
    t0 = time.perf_counter()
    cls = classification_benchmark()
    anom = anomaly_benchmark()
    out = {
        "config": {
            "dimensions": DIMS,
            "alpha": ALPHA,
            "classification_seed": SEED,
            "anomaly_seeds": ANOM_SEEDS,
            "protocol": "training-only-preprocessing-v2",
            "comparison": "different encoders; not a matched implementation or SOTA comparison",
        },
        "provenance": provenance(),
        "classification": cls,
        "anomaly": anom,
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    p = args.output
    write_json(p, out)
    print(f"\nwrote {p} ({out['runtime_s']}s)")


if __name__ == "__main__":
    main()
