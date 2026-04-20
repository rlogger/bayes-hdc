# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Loaders for the HDC benchmark suite.

Sklearn-backed loaders (``load_iris``, ``load_wine``, ...) ship with
the library and work offline. OpenML-backed loaders (``load_isolet``,
``load_ucihar``, ``load_mnist``, ``load_fashion_mnist``) download on
first use and cache in the user's scikit-learn home; downstream calls
reuse the cache.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from bayes_hdc.datasets.base import HDCDataset

DEFAULT_TEST_SIZE = 0.3
DEFAULT_SEED = 42


def _import_sklearn() -> tuple[Any, Any]:
    """Return (sklearn.datasets, sklearn.model_selection) with a friendly error."""
    try:
        from sklearn import datasets as sk_datasets
        from sklearn import model_selection as sk_model_selection
    except ImportError as e:  # pragma: no cover — user-environment-dependent
        raise ImportError(
            "bayes_hdc.datasets requires scikit-learn. Install with "
            "`pip install bayes-hdc[datasets]` or `pip install scikit-learn`."
        ) from e
    return sk_datasets, sk_model_selection


def _stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, sk_model_selection = _import_sklearn()
    return sk_model_selection.train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def _normalise_labels(y: np.ndarray) -> np.ndarray:
    """Map string / arbitrary-int labels to contiguous ``int32`` indices."""
    if y.dtype.kind in {"U", "O", "S"}:
        uniq = sorted(np.unique(y).tolist())
        mapping: dict[Any, int] = {v: i for i, v in enumerate(uniq)}
        return np.asarray([mapping[v] for v in y], dtype=np.int32)
    y_int: np.ndarray = np.asarray(y, dtype=np.int32)
    # Remap to 0..K-1 in case the raw labels are not contiguous.
    uniq_arr = np.unique(y_int)
    if not np.array_equal(uniq_arr, np.arange(len(uniq_arr))):
        int_mapping: dict[int, int] = {int(v): i for i, v in enumerate(uniq_arr)}
        y_int = np.asarray([int_mapping[int(v)] for v in y_int], dtype=np.int32)
    return y_int


def _build(
    name: str,
    description: str,
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
    classes: tuple[str, ...] | None = None,
) -> HDCDataset:
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = _normalise_labels(y)
    X_tr, X_te, y_tr, y_te = _stratified_split(X, y, test_size, random_state)
    return HDCDataset(
        name=name,
        X=X,
        y=y,
        X_train=X_tr,
        y_train=y_tr,
        X_test=X_te,
        y_test=y_te,
        n_classes=int(y.max() + 1),
        n_features=int(X.shape[1]),
        description=description,
        classes=classes,
    )


# ----------------------------------------------------------------------
# sklearn built-in loaders (offline, ship with the library)
# ----------------------------------------------------------------------


def _from_sklearn_builtin(
    sklearn_fn: Callable[[], Any],
    name: str,
    description: str,
    test_size: float,
    random_state: int,
) -> HDCDataset:
    data = sklearn_fn()
    X = np.asarray(data.data, dtype=np.float32)
    y = np.asarray(data.target)
    classes = tuple(str(c) for c in data.target_names) if hasattr(data, "target_names") else None
    return _build(name, description, X, y, test_size, random_state, classes=classes)


def load_iris(
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_SEED,
) -> HDCDataset:
    """Fisher's iris — 3-class flower taxonomy, 4 features, 150 samples."""
    sk_datasets, _ = _import_sklearn()
    return _from_sklearn_builtin(
        sk_datasets.load_iris,
        "iris",
        "Fisher's iris — 3-class flower taxonomy, 4 features, 150 samples.",
        test_size,
        random_state,
    )


def load_wine(
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_SEED,
) -> HDCDataset:
    """UCI Wine — 3-class cultivar classification, 13 chemical features, 178 samples."""
    sk_datasets, _ = _import_sklearn()
    return _from_sklearn_builtin(
        sk_datasets.load_wine,
        "wine",
        "UCI Wine — 3-class cultivar classification, 13 features, 178 samples.",
        test_size,
        random_state,
    )


def load_breast_cancer(
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_SEED,
) -> HDCDataset:
    """UCI Wisconsin Breast Cancer Diagnostic — binary, 30 features, 569 samples."""
    sk_datasets, _ = _import_sklearn()
    return _from_sklearn_builtin(
        sk_datasets.load_breast_cancer,
        "breast_cancer",
        "UCI Breast Cancer Wisconsin Diagnostic — binary, 30 features, 569 samples.",
        test_size,
        random_state,
    )


def load_digits(
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_SEED,
) -> HDCDataset:
    """UCI Optical Digits — 10-class, 64 pixel features, 1 797 samples."""
    sk_datasets, _ = _import_sklearn()
    return _from_sklearn_builtin(
        sk_datasets.load_digits,
        "digits",
        "UCI Optical Digits — 10-class, 64 pixel features, 1797 samples.",
        test_size,
        random_state,
    )


# ----------------------------------------------------------------------
# OpenML-backed loaders (download + cache in sklearn home)
# ----------------------------------------------------------------------


def _fetch_openml_cached(
    openml_name: str | int,
    version: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    sk_datasets, _ = _import_sklearn()
    bunch = sk_datasets.fetch_openml(
        openml_name,
        version=version,
        as_frame=False,
        parser="auto",
    )
    return bunch.data, bunch.target


def load_mnist(
    subsample: int | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_SEED,
) -> HDCDataset:
    """MNIST handwritten digits — 10-class, 784 pixel features.

    Loaded via OpenML (``mnist_784``); cached on first use.
    """
    X, y = _fetch_openml_cached("mnist_784", version=1)
    X = np.asarray(X, dtype=np.float32) / 255.0
    y = np.asarray(y, dtype=np.int32)
    if subsample is not None and subsample < X.shape[0]:
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(X.shape[0])[:subsample]
        X, y = X[idx], y[idx]
    return _build(
        "mnist",
        "MNIST handwritten digits — 10-class, 784 features, 70 000 samples.",
        X,
        y,
        test_size,
        random_state,
    )


def load_fashion_mnist(
    subsample: int | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_SEED,
) -> HDCDataset:
    """Fashion-MNIST — 10-class clothing images, 784 pixel features.

    Loaded via OpenML; cached on first use. Xiao, Rasul, Vollgraf
    (2017).
    """
    X, y = _fetch_openml_cached("Fashion-MNIST", version=1)
    X = np.asarray(X, dtype=np.float32) / 255.0
    y = np.asarray(y, dtype=np.int32)
    if subsample is not None and subsample < X.shape[0]:
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(X.shape[0])[:subsample]
        X, y = X[idx], y[idx]
    classes = (
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    )
    return _build(
        "fashion_mnist",
        "Fashion-MNIST — 10-class clothing, 784 features, 70 000 samples (Xiao et al. 2017).",
        X,
        y,
        test_size,
        random_state,
        classes=classes,
    )


def load_isolet(
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_SEED,
) -> HDCDataset:
    """ISOLET — 26-class spoken-letter recognition, 617 features.

    Fanty & Cole (1990). The canonical HDC benchmark since Rahimi,
    Kanerva, Rabaey (2016).
    """
    X, y = _fetch_openml_cached("isolet", version=1)
    return _build(
        "isolet",
        "ISOLET — 26-class spoken-letter recognition, 617 features, 7 797 samples "
        "(Fanty & Cole 1990; Rahimi et al. 2016 HDC benchmark).",
        np.asarray(X),
        np.asarray(y),
        test_size,
        random_state,
    )


def load_ucihar(
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_SEED,
) -> HDCDataset:
    """UCI Human Activity Recognition — 6-class, 561 features.

    Anguita, Ghio, Oneto, Parra, Reyes-Ortiz (2013). Standard HDC
    benchmark.
    """
    X, y = _fetch_openml_cached("UCI-HAR", version=1)
    return _build(
        "ucihar",
        "UCI Human Activity Recognition — 6-class, 561 features (Anguita et al. 2013).",
        np.asarray(X),
        np.asarray(y),
        test_size,
        random_state,
    )


__all__ = [
    "load_iris",
    "load_wine",
    "load_breast_cancer",
    "load_digits",
    "load_mnist",
    "load_fashion_mnist",
    "load_isolet",
    "load_ucihar",
]
