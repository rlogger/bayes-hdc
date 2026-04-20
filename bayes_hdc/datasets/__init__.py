# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Standard benchmark datasets for HDC evaluation.

This submodule provides a uniform loader API for the datasets the HDC
literature uses most often. Every loader returns an
:class:`HDCDataset` — a lightweight dataclass with ``X``, ``y``,
``X_train``, ``y_train``, ``X_test``, ``y_test``, ``n_classes``,
``n_features``, and descriptive metadata — so downstream benchmark code
is data-source-agnostic.

Datasets currently supported:

- **iris / wine / breast_cancer / digits** — sklearn built-ins, ship
  with the library (no network required). Fast smoke tests.
- **mnist / fashion_mnist** — loaded via scikit-learn's OpenML
  fetcher, cached in the user's scikit-learn home.
- **isolet** — 26-class spoken-letter recognition (Fanty & Cole, 1990);
  the canonical HDC benchmark since Rahimi et al. (2016). 617 features.
- **ucihar** — Human Activity Recognition using Smartphones (Anguita
  et al., 2013); 6-class, 561 features.

All OpenML-backed loaders support a ``subsample`` argument for faster
iteration and a ``random_state`` / ``test_size`` for reproducible
splits. The default split is 70 / 30 stratified.

Example::

    from bayes_hdc.datasets import load_isolet, load_mnist

    ds = load_isolet()
    print(ds)  # HDCDataset(name='isolet', n_samples=7797, ...)
    X_train, y_train = ds.X_train, ds.y_train
"""

from bayes_hdc.datasets.base import HDCDataset
from bayes_hdc.datasets.loaders import (
    load_breast_cancer,
    load_digits,
    load_emg,
    load_european_languages,
    load_fashion_mnist,
    load_iris,
    load_isolet,
    load_mnist,
    load_pamap2,
    load_ucihar,
    load_wine,
)

# Registry of every loader in this module, used by :func:`load` for
# name-based dispatch (e.g. in benchmark harnesses).
ALL_DATASETS: dict[str, object] = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
    "mnist": load_mnist,
    "fashion_mnist": load_fashion_mnist,
    "isolet": load_isolet,
    "ucihar": load_ucihar,
    "emg": load_emg,
    "pamap2": load_pamap2,
    "european_languages": load_european_languages,
}


def load(name: str, **kwargs: object) -> HDCDataset:
    """Dispatch to the named dataset loader.

    Args:
        name: Dataset key — one of the entries in :data:`ALL_DATASETS`.
        **kwargs: Passed through to the underlying loader
            (e.g. ``subsample``, ``test_size``, ``random_state``).

    Raises:
        ValueError: If ``name`` is not a registered dataset.
    """
    if name not in ALL_DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; available: {sorted(ALL_DATASETS.keys())}")
    loader = ALL_DATASETS[name]
    return loader(**kwargs)  # type: ignore[operator]


__all__ = [
    "HDCDataset",
    "ALL_DATASETS",
    "load",
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
