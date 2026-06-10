# SPDX-License-Identifier: MIT
# Copyright (c) 2026 R.S.

"""Loaders for the HDC benchmark suite.

Sklearn-backed loaders (``load_iris``, ``load_wine``, ...) ship with
the library and work offline. OpenML-backed loaders (``load_isolet``,
``load_mnist``, ``load_fashion_mnist``) download on first use and cache
in the user's scikit-learn home. ``load_ucihar`` and ``load_emg`` fetch
the official UCI archive and the original authors' ``dataset.mat``
respectively, cached under ``~/.cache/bayes_hdc``.
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
    # parser="liac-arff" works without the optional pandas/pyarrow stack;
    # parser="auto" fails on hosts that lack them.
    bunch = sk_datasets.fetch_openml(
        openml_name,
        version=version,
        as_frame=False,
        parser="liac-arff",
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
    Kanerva, Rabaey (2016). OpenML pools all 7 797 utterances, so this
    loader uses a stratified random split; for the canonical
    isolet1-4/isolet5 (6 238/1 559) split, fetch from the UCI archive
    or use TorchHD's ``ISOLET`` dataset class.
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


_UCIHAR_URL = (
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
)


def load_ucihar(
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_SEED,
) -> HDCDataset:
    """UCI Human Activity Recognition — 6-class, 561 features, 10 299 samples.

    Anguita, Ghio, Oneto, Parra, Reyes-Ortiz (2013). Standard HDC
    benchmark. Downloads the official UCI archive on first use and keeps
    the **canonical subject-disjoint split** (7 352 train / 2 947 test —
    no subject appears in both), so accuracies are comparable to the
    literature. ``test_size`` and ``random_state`` are ignored: the
    official partition is always used.
    """
    import io
    import urllib.request
    import zipfile

    del test_size, random_state  # official subject-disjoint split is fixed
    cached = _cache_dir() / "ucihar_official.npz"
    if not cached.exists():
        outer_path = _cache_dir() / "ucihar.zip"
        if not outer_path.exists():
            urllib.request.urlretrieve(_UCIHAR_URL, outer_path)  # noqa: S310 — fixed https URL
        with zipfile.ZipFile(outer_path) as outer:
            inner_bytes = outer.read("UCI HAR Dataset.zip")
        with zipfile.ZipFile(io.BytesIO(inner_bytes)) as z:

            def read_txt(name: str) -> np.ndarray:
                with z.open(f"UCI HAR Dataset/{name}") as fh:
                    return np.loadtxt(fh)

            np.savez_compressed(
                cached,
                X_train=read_txt("train/X_train.txt").astype(np.float32),
                y_train=read_txt("train/y_train.txt").astype(np.int32),
                X_test=read_txt("test/X_test.txt").astype(np.float32),
                y_test=read_txt("test/y_test.txt").astype(np.int32),
            )
    arr = np.load(cached)
    X_tr, X_te = arr["X_train"], arr["X_test"]
    n_tr = X_tr.shape[0]
    y_all = _normalise_labels(np.concatenate([arr["y_train"], arr["y_test"]]))
    y_tr, y_te = y_all[:n_tr], y_all[n_tr:]
    X = np.vstack([X_tr, X_te])
    y = y_all
    return HDCDataset(
        name="ucihar",
        X=X,
        y=y,
        X_train=X_tr,
        y_train=y_tr,
        X_test=X_te,
        y_test=y_te,
        n_classes=int(y.max() + 1),
        n_features=int(X.shape[1]),
        description=(
            "UCI Human Activity Recognition — 6-class, 561 features; official "
            "subject-disjoint 7 352/2 947 split (Anguita et al. 2013)."
        ),
        classes=("walking", "walking_up", "walking_down", "sitting", "standing", "laying"),
    )


_EMG_URL = "https://raw.githubusercontent.com/abbas-rahimi/HDC-EMG/master/dataset.mat"


def _cache_dir() -> Any:
    import pathlib

    d = pathlib.Path.home() / ".cache" / "bayes_hdc"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_emg(
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_SEED,
    window: int = 256,
    subjects: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> HDCDataset:
    """EMG hand gestures — 5-class, 4-channel EMG (Rahimi et al. 2016).

    Downloads ``dataset.mat`` from the original authors' repository
    (``abbas-rahimi/HDC-EMG``) on first use and caches it under
    ``~/.cache/bayes_hdc``. Each subject's 4-channel stream is cut into
    non-overlapping label-pure windows of ``window`` samples, flattened
    to a ``4 * window`` feature vector; windows spanning a gesture
    transition are dropped. Classes: closed hand at rest plus four
    gestures.
    """
    import urllib.request

    from scipy.io import loadmat

    path = _cache_dir() / "rahimi_emg_dataset.mat"
    if not path.exists():
        urllib.request.urlretrieve(_EMG_URL, path)  # noqa: S310 — fixed https URL
    mat = loadmat(str(path))

    feats, labels = [], []
    for s in subjects:
        sig = np.asarray(mat[f"COMPLETE_{s}"], dtype=np.float32)
        lab = np.asarray(mat[f"LABEL_{s}"]).ravel()
        n_win = sig.shape[0] // window
        for w in range(n_win):
            seg_lab = lab[w * window : (w + 1) * window]
            if (seg_lab != seg_lab[0]).any():
                continue  # window spans a gesture transition
            feats.append(sig[w * window : (w + 1) * window].reshape(-1))
            labels.append(seg_lab[0])
    X = np.stack(feats)
    y = np.asarray(labels)
    return _build(
        "emg",
        "EMG hand gestures — 5-class, 4-channel EMG, label-pure windows "
        f"of {window} samples (Rahimi et al. 2016).",
        X,
        y,
        test_size,
        random_state,
    )


def load_pamap2(
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_SEED,
    subsample: int | None = 20_000,
) -> HDCDataset:
    """PAMAP2 — physical activity monitoring, 12+ classes (Reiss & Stricker 2012).

    PAMAP2 is not hosted on OpenML and the raw UCI archive is ~650 MB,
    which this library will not download silently. Fetch it yourself from
    https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
    and build an :class:`HDCDataset` from the extracted protocol files.
    """
    raise ValueError(
        "PAMAP2 is not available on OpenML and the raw UCI archive is ~650 MB, "
        "so bayes-hdc does not download it automatically. Fetch it from "
        "https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring "
        "and construct an HDCDataset from the protocol files directly."
    )


def load_european_languages(
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_SEED,
) -> HDCDataset:
    """European Languages — 21-class n-gram classification.

    The canonical HDC language-identification benchmark since Joshi,
    Halseth, Kanerva (2016). Attempts OpenML's ``European-Languages-Full``
    or similar name; raises ``ValueError`` if the fetch fails so users
    get a clear error rather than silent fallback.
    """
    try:
        X, y = _fetch_openml_cached("European-Languages", version=1)
    except Exception as e:  # pragma: no cover — network-dependent
        raise ValueError(
            "European Languages dataset is not reliably available on OpenML "
            "under a stable name. Either fetch it manually from the Joshi et "
            "al. (2016) companion site and pass features to HDCDataset(...), "
            "or pin a specific OpenML dataset_id you've verified works in "
            "your environment."
        ) from e
    return _build(
        "european_languages",
        "European Languages — 21-class n-gram classification (Joshi, Halseth, "
        "Kanerva 2016); canonical HDC language-ID benchmark.",
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
    "load_emg",
    "load_pamap2",
    "load_european_languages",
]
