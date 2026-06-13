# SPDX-License-Identifier: MIT
# Copyright (c) 2026 R.S.

"""scikit-learn-compatible estimators for bayes-hdc.

Thin wrappers that let the whole scikit-learn ecosystem — pipelines,
``cross_val_score``, ``GridSearchCV``, ``OutlierMixin`` tooling — drive
hyperdimensional-computing models with the usual ``fit`` / ``predict`` /
``predict_proba`` / ``score_samples`` API:

    from bayes_hdc.sklearn import HDClassifier, HDAnomalyDetector

    clf = HDClassifier(dimensions=10_000).fit(X_train, y_train)
    clf.predict(X_test)                 # labels
    clf.predict_proba(X_test)           # softmax over class similarities

    det = HDAnomalyDetector(alpha=0.05).fit(X_normal)
    det.predict(X_test)                 # +1 inlier / -1 outlier (sklearn convention)
    det.score_samples(X_test)           # higher = more normal
    det.pvalue(X_test)                  # split-conformal p-values

scikit-learn is an optional dependency (``pip install bayes-hdc[examples]``
or ``pip install scikit-learn``). Importing this module without it raises a
clear error; the rest of bayes-hdc has no scikit-learn dependency.

The continuous feature matrix ``X`` is encoded with a random-projection
``ProjectionEncoder`` by default, or a random-Fourier ``KernelEncoder``
(``encoder="kernel"``) that approximates an RBF kernel and is usually more
accurate on real-valued features; no manual discretisation is needed. The
anomaly detector wraps :class:`~bayes_hdc.ConformalAnomalyDetector`, so its
``predict`` inherits the finite-sample false-positive guarantee at the
chosen ``alpha``.
"""

from __future__ import annotations

import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

try:
    from sklearn.base import BaseEstimator, ClassifierMixin, OutlierMixin
    from sklearn.utils.validation import check_is_fitted
except ImportError as exc:  # pragma: no cover - exercised only without sklearn
    raise ImportError(
        "bayes_hdc.sklearn requires scikit-learn. Install it with "
        "`pip install scikit-learn` or `pip install bayes-hdc[examples]`."
    ) from exc

from bayes_hdc import MAP, KernelEncoder, ProjectionEncoder
from bayes_hdc.anomaly import ConformalAnomalyDetector, HDCAnomalyScorer
from bayes_hdc.models import CentroidClassifier

__all__ = ["HDClassifier", "HDAnomalyDetector"]


def _as_f32(X: Any) -> np.ndarray:
    return np.asarray(X, dtype=np.float32)


class HDClassifier(ClassifierMixin, BaseEstimator):
    """Hyperdimensional classifier with a scikit-learn API.

    Encodes continuous features with a random-projection or random-Fourier
    (RBF-kernel) encoder and classifies with a centroid (prototype) model.
    Drop-in for any scikit-learn pipeline or model-selection tool.

    Parameters
    ----------
    dimensions : int, default=10000
        Hypervector dimensionality.
    encoder : {"projection", "kernel"}, default="projection"
        Feature encoder. ``"projection"`` is a plain random projection;
        ``"kernel"`` is a random-Fourier-features encoder that approximates
        an RBF kernel and is typically more accurate on real-valued features
        (it mirrors TorchHD's ``Sinusoid`` embedding). With ``"kernel"`` the
        ``gamma`` bandwidth matters and is worth tuning via ``GridSearchCV``.
    gamma : float, default=0.01
        RBF bandwidth for ``encoder="kernel"`` (ignored otherwise).
    vsa_model : {"map"}, default="map"
        VSA backend for the encoder/prototype space. (MAP is the
        real-valued default; the prototype classifier assumes it.)
    random_state : int, default=0
        Seed for the random projection.
    """

    def __init__(
        self,
        dimensions: int = 10000,
        encoder: str = "projection",
        gamma: float = 0.01,
        vsa_model: str = "map",
        random_state: int = 0,
    ) -> None:
        self.dimensions = dimensions
        self.encoder = encoder
        self.gamma = gamma
        self.vsa_model = vsa_model
        self.random_state = random_state

    def _make_encoder(self, key: jax.Array) -> Any:
        model = MAP.create(dimensions=self.dimensions)
        if self.encoder == "kernel":
            return KernelEncoder.create(
                input_dim=self.n_features_in_,
                dimensions=self.dimensions,
                gamma=self.gamma,
                vsa_model=model,
                key=key,
            )
        if self.encoder == "projection":
            return ProjectionEncoder.create(
                input_dim=self.n_features_in_,
                dimensions=self.dimensions,
                vsa_model=model,
                key=key,
            )
        raise ValueError(f"encoder must be 'projection' or 'kernel', got {self.encoder!r}")

    def fit(self, X: Any, y: Any) -> HDClassifier:
        X = _as_f32(X)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        key = jax.random.PRNGKey(int(self.random_state))
        self.encoder_ = self._make_encoder(key)
        hvs = self.encoder_.encode_batch(jnp.asarray(X))
        self.classifier_ = CentroidClassifier.create(
            num_classes=len(self.classes_),
            dimensions=self.dimensions,
            vsa_model=MAP.create(dimensions=self.dimensions),
        ).fit(hvs, jnp.asarray(y_idx))
        return self

    def _encode(self, X: Any) -> jax.Array:
        return self.encoder_.encode_batch(jnp.asarray(_as_f32(X)))

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "classifier_")
        idx = np.asarray(self.classifier_.predict(self._encode(X)))
        return self.classes_[idx]

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "classifier_")
        return np.asarray(self.classifier_.predict_proba(self._encode(X)))


class HDAnomalyDetector(OutlierMixin, BaseEstimator):
    """Calibrated one-class anomaly detector with a scikit-learn API.

    Wraps :class:`~bayes_hdc.ConformalAnomalyDetector`: ``fit`` on normal
    data, then ``predict`` flags outliers at a guaranteed false-positive
    rate ``alpha`` (finite-sample, distribution-free, under
    exchangeability). Follows scikit-learn's outlier conventions:
    ``predict`` returns +1 for inliers and -1 for outliers, and
    ``score_samples`` returns higher values for more-normal points.

    Parameters
    ----------
    alpha : float, default=0.05
        Target false-positive rate; ``predict`` flags a point as an
        outlier when its conformal p-value is below ``alpha``.
    dimensions : int, default=10000
        Hypervector dimensionality.
    distance_metric : str or None, default=None
        Nonconformity score for :class:`HDCAnomalyScorer` (defaults to
        cosine distance to the centroid).
    k_neighbors : int, default=1
        k for the k-NN-mean variant of the score.
    calibration_fraction : float, default=0.3
        Fraction of the fit data held out to calibrate the conformal
        p-values. The rest fits the scorer's normal region.
    random_state : int, default=0
        Seed for the random projection and the calibration split.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        dimensions: int = 10000,
        distance_metric: str | None = None,
        k_neighbors: int = 1,
        calibration_fraction: float = 0.3,
        random_state: int = 0,
    ) -> None:
        self.alpha = alpha
        self.dimensions = dimensions
        self.distance_metric = distance_metric
        self.k_neighbors = k_neighbors
        self.calibration_fraction = calibration_fraction
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> HDAnomalyDetector:
        X = _as_f32(X)
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(int(self.random_state))
        n = X.shape[0]
        perm = rng.permutation(n)
        n_cal = max(2, int(round(self.calibration_fraction * n)))
        cal_idx, fit_idx = perm[:n_cal], perm[n_cal:]
        if fit_idx.size == 0:  # tiny inputs: reuse the calibration split
            fit_idx = cal_idx

        key = jax.random.PRNGKey(int(self.random_state))
        self.encoder_ = ProjectionEncoder.create(
            input_dim=self.n_features_in_,
            dimensions=self.dimensions,
            vsa_model=MAP.create(dimensions=self.dimensions),
            key=key,
        )
        normal_hvs = self.encoder_.encode_batch(jnp.asarray(X[fit_idx]))
        cal_hvs = self.encoder_.encode_batch(jnp.asarray(X[cal_idx]))
        scorer = HDCAnomalyScorer.create(
            dimensions=self.dimensions,
            vsa_model="map",
            distance_metric=self.distance_metric,
            k_neighbors=self.k_neighbors,
        ).fit(normal_hvs)
        self.detector_ = ConformalAnomalyDetector.create(scorer).fit(cal_hvs)

        # Conformal p-values live on the grid {1/(n_cal+1), ..., 1}, so the
        # smallest attainable p-value is 1/(n_cal+1). If ``alpha`` is below that
        # floor, ``predict`` can never flag *any* point as an outlier (no matter
        # how extreme) and the detector silently returns all-inliers. Warn loudly
        # rather than letting users mistake "0 detections" for "0 anomalies".
        n_cal = int(self.detector_.n_calibration)
        floor = 1.0 / (n_cal + 1.0)
        if self.alpha < floor:
            warnings.warn(
                f"alpha={self.alpha:g} is below the conformal resolution floor "
                f"1/(n_calibration+1)={floor:g} (n_calibration={n_cal}). No point "
                f"can be flagged as an outlier at this alpha. Increase the amount "
                f"of fit data, raise calibration_fraction, or use a larger alpha "
                f"(alpha >= {floor:g}).",
                UserWarning,
                stacklevel=2,
            )
        return self

    def _encode(self, X: Any) -> jax.Array:
        return self.encoder_.encode_batch(jnp.asarray(_as_f32(X)))

    def pvalue(self, X: Any) -> np.ndarray:
        """Split-conformal p-values; small = anomalous."""
        check_is_fitted(self, "detector_")
        return np.asarray(self.detector_.pvalue_batch(self._encode(X)))

    def predict(self, X: Any) -> np.ndarray:
        """+1 for inliers, -1 for outliers (scikit-learn convention)."""
        pvals = self.pvalue(X)
        return np.where(pvals <= self.alpha, -1, 1).astype(int)

    def score_samples(self, X: Any) -> np.ndarray:
        """Higher = more normal (scikit-learn convention).

        Returns the conformal p-value, which is already oriented so that
        larger values are more in-distribution.
        """
        return self.pvalue(X)

    def decision_function(self, X: Any) -> np.ndarray:
        """Signed margin: ``pvalue - alpha`` (>= 0 inlier, < 0 outlier)."""
        return self.pvalue(X) - self.alpha
