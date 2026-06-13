# SPDX-License-Identifier: MIT
# Copyright (c) 2026 R.S.

"""Tests for the scikit-learn-compatible estimators in bayes_hdc.sklearn."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.datasets import make_blobs, make_classification  # noqa: E402
from sklearn.exceptions import NotFittedError  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from bayes_hdc.sklearn import HDAnomalyDetector, HDClassifier  # noqa: E402

# ----------------------------------------------------------------------
# HDClassifier
# ----------------------------------------------------------------------


def test_classifier_fit_predict_shapes_and_labels():
    X, y = make_classification(
        n_samples=200, n_features=10, n_classes=3, n_informative=6, random_state=0
    )
    clf = HDClassifier(dimensions=2000, random_state=0).fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (200,)
    # predictions are drawn from the trained label set
    assert set(np.unique(preds)).issubset(set(np.unique(y)))
    assert clf.n_features_in_ == 10


def test_classifier_predict_proba_is_a_distribution():
    X, y = make_classification(
        n_samples=150, n_features=8, n_classes=3, n_informative=5, random_state=1
    )
    clf = HDClassifier(dimensions=2000, random_state=1).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (150, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    assert (proba >= 0).all()


def test_classifier_learns_above_chance():
    X, y = make_classification(
        n_samples=400, n_features=12, n_classes=2, n_informative=8, random_state=2
    )
    clf = HDClassifier(dimensions=4000, random_state=2).fit(X, y)
    acc = (clf.predict(X) == y).mean()
    assert acc > 0.7, f"train accuracy {acc:.2f} not above chance"


def test_classifier_preserves_string_labels():
    X, y_int = make_classification(
        n_samples=120, n_features=6, n_classes=2, n_informative=4, random_state=3
    )
    y = np.array(["cat", "dog"])[y_int]
    clf = HDClassifier(dimensions=2000, random_state=3).fit(X, y)
    preds = clf.predict(X)
    assert set(np.unique(preds)).issubset({"cat", "dog"})


def test_classifier_in_sklearn_pipeline_and_cv():
    X, y = make_classification(
        n_samples=200, n_features=10, n_classes=2, n_informative=6, random_state=4
    )
    pipe = make_pipeline(StandardScaler(), HDClassifier(dimensions=2000, random_state=4))
    scores = cross_val_score(pipe, X, y, cv=3)
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()


def test_classifier_get_set_params_roundtrip():
    clf = HDClassifier(dimensions=3000, random_state=7)
    params = clf.get_params()
    assert params["dimensions"] == 3000
    clf.set_params(dimensions=5000)
    assert clf.get_params()["dimensions"] == 5000


def test_classifier_kernel_encoder_fits_and_predicts():
    X, y = make_classification(
        n_samples=300, n_features=12, n_classes=3, n_informative=8, random_state=5
    )
    X = StandardScaler().fit_transform(X)  # RFF bandwidth assumes scaled features
    clf = HDClassifier(dimensions=4000, encoder="kernel", gamma=0.03, random_state=5).fit(X, y)
    acc = (clf.predict(X) == y).mean()
    assert acc > 0.6, f"kernel-encoder train accuracy {acc:.2f} not above chance (3 classes)"
    proba = clf.predict_proba(X)
    assert proba.shape == (300, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)


def test_classifier_invalid_encoder_raises():
    X, y = make_classification(n_samples=40, n_features=6, random_state=6)
    with pytest.raises(ValueError, match="encoder"):
        HDClassifier(dimensions=1000, encoder="nope").fit(X, y)


def test_classifier_kernel_gamma_tunable_in_gridsearch():
    from sklearn.model_selection import GridSearchCV

    X, y = make_classification(
        n_samples=200, n_features=10, n_classes=2, n_informative=6, random_state=8
    )
    grid = GridSearchCV(
        HDClassifier(dimensions=2000, encoder="kernel", random_state=8),
        {"gamma": [0.003, 0.03]},
        cv=3,
    ).fit(X, y)
    assert grid.best_params_["gamma"] in {0.003, 0.03}


# ----------------------------------------------------------------------
# HDAnomalyDetector
# ----------------------------------------------------------------------


def test_anomaly_predict_outlier_convention():
    # Normal cluster + clearly separated outliers.
    X_norm, _ = make_blobs(n_samples=300, n_features=8, centers=1, cluster_std=1.0, random_state=0)
    rng = np.random.default_rng(0)
    X_out = rng.normal(loc=12.0, scale=1.0, size=(40, 8))

    det = HDAnomalyDetector(alpha=0.05, dimensions=4000, random_state=0).fit(X_norm)
    pred_norm = det.predict(X_norm)
    pred_out = det.predict(X_out)

    # sklearn convention: +1 inlier, -1 outlier
    assert set(np.unique(np.concatenate([pred_norm, pred_out]))).issubset({-1, 1})
    # most outliers flagged
    assert (pred_out == -1).mean() > 0.8
    # false-positive rate on normal data near alpha (finite-sample slack)
    assert (pred_norm == -1).mean() < 0.15


def test_anomaly_pvalue_range_and_orientation():
    X_norm, _ = make_blobs(n_samples=200, n_features=6, centers=1, cluster_std=1.0, random_state=1)
    rng = np.random.default_rng(1)
    X_out = rng.normal(loc=15.0, scale=1.0, size=(30, 6))
    det = HDAnomalyDetector(alpha=0.05, dimensions=3000, random_state=1).fit(X_norm)
    p_norm = det.pvalue(X_norm)
    p_out = det.pvalue(X_out)
    assert (p_norm >= 0).all() and (p_norm <= 1.0 + 1e-6).all()
    # anomalies get smaller p-values than normal points on average
    assert p_out.mean() < p_norm.mean()


def test_anomaly_score_samples_higher_is_more_normal():
    X_norm, _ = make_blobs(n_samples=200, n_features=6, centers=1, cluster_std=1.0, random_state=2)
    rng = np.random.default_rng(2)
    X_out = rng.normal(loc=15.0, scale=1.0, size=(30, 6))
    det = HDAnomalyDetector(dimensions=3000, random_state=2).fit(X_norm)
    assert det.score_samples(X_norm).mean() > det.score_samples(X_out).mean()


def test_anomaly_decision_function_sign_matches_predict():
    X_norm, _ = make_blobs(n_samples=200, n_features=6, centers=1, cluster_std=1.0, random_state=3)
    det = HDAnomalyDetector(alpha=0.1, dimensions=3000, random_state=3).fit(X_norm)
    df = det.decision_function(X_norm)
    pred = det.predict(X_norm)
    # decision_function >= 0  <=>  predicted inlier (+1)
    assert np.all((df >= 0) == (pred == 1))


def test_anomaly_get_params_roundtrip():
    det = HDAnomalyDetector(alpha=0.05, dimensions=2000)
    assert det.get_params()["alpha"] == 0.05
    det.set_params(alpha=0.01)
    assert det.get_params()["alpha"] == 0.01


def test_anomaly_warns_when_alpha_below_resolution_floor():
    # n=20, calibration_fraction=0.3 -> n_cal=6 -> floor = 1/7 ~= 0.143.
    # alpha=0.05 is below the floor, so no point can ever be flagged.
    X_norm, _ = make_blobs(n_samples=20, n_features=4, centers=1, random_state=0)
    rng = np.random.default_rng(0)
    X_out = rng.normal(loc=15.0, size=(10, 4))
    with pytest.warns(UserWarning, match="conformal resolution floor"):
        det = HDAnomalyDetector(alpha=0.05, dimensions=2000, random_state=0).fit(X_norm)
    # The warning's premise: even extreme outliers cannot be flagged at this alpha.
    assert (det.predict(X_out) == -1).sum() == 0


def test_anomaly_no_warning_when_alpha_attainable():
    X_norm, _ = make_blobs(n_samples=300, n_features=4, centers=1, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any UserWarning would fail the test
        HDAnomalyDetector(alpha=0.05, dimensions=2000, random_state=0).fit(X_norm)


def test_estimators_raise_not_fitted_before_fit():
    clf = HDClassifier(dimensions=1000)
    det = HDAnomalyDetector(dimensions=1000)
    X = np.zeros((3, 4), dtype=np.float32)
    with pytest.raises(NotFittedError):
        clf.predict(X)
    with pytest.raises(NotFittedError):
        clf.predict_proba(X)
    with pytest.raises(NotFittedError):
        det.predict(X)
    with pytest.raises(NotFittedError):
        det.pvalue(X)
