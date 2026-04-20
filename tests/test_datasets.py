# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Tests for the bayes_hdc.datasets submodule.

Sklearn-backed loaders are exercised fully. OpenML-backed loaders are
checked for module wiring / dispatch only; a ``@pytest.mark.network``
integration test covers the download path and is skipped by default.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayes_hdc.datasets import (
    ALL_DATASETS,
    HDCDataset,
    load,
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)


@pytest.mark.parametrize(
    "loader, name, expected_classes, expected_features, expected_samples",
    [
        (load_iris, "iris", 3, 4, 150),
        (load_wine, "wine", 3, 13, 178),
        (load_breast_cancer, "breast_cancer", 2, 30, 569),
        (load_digits, "digits", 10, 64, 1797),
    ],
)
def test_sklearn_loaders_produce_dataset_with_correct_shape(
    loader,
    name: str,
    expected_classes: int,
    expected_features: int,
    expected_samples: int,
) -> None:
    ds = loader()
    assert isinstance(ds, HDCDataset)
    assert ds.name == name
    assert ds.n_classes == expected_classes
    assert ds.n_features == expected_features
    assert ds.n_samples == expected_samples
    assert ds.X.shape == (expected_samples, expected_features)
    assert ds.y.shape == (expected_samples,)


def test_sklearn_loader_produces_contiguous_float32() -> None:
    ds = load_iris()
    assert ds.X.dtype == np.float32
    assert ds.X_train.dtype == np.float32
    assert ds.X.flags["C_CONTIGUOUS"]


def test_labels_are_contiguous_int32_starting_at_zero() -> None:
    ds = load_wine()
    assert ds.y.dtype == np.int32
    assert int(ds.y.min()) == 0
    assert int(ds.y.max()) == ds.n_classes - 1
    # Labels cover 0..K-1 fully.
    assert set(np.unique(ds.y).tolist()) == set(range(ds.n_classes))


def test_train_test_split_is_stratified_and_disjoint() -> None:
    ds = load_digits()
    # Split sums to full size.
    assert ds.n_train + ds.n_test == ds.n_samples
    # Every class represented in both halves (stratified).
    assert set(np.unique(ds.y_train)) == set(range(ds.n_classes))
    assert set(np.unique(ds.y_test)) == set(range(ds.n_classes))


def test_test_size_default_is_30_percent() -> None:
    ds = load_iris()
    # allow ±1 sample due to integer split arithmetic
    assert abs(ds.n_test - int(0.3 * ds.n_samples)) <= 2


def test_custom_test_size_and_random_state() -> None:
    ds_a = load_wine(test_size=0.2, random_state=7)
    ds_b = load_wine(test_size=0.2, random_state=7)
    ds_c = load_wine(test_size=0.2, random_state=123)
    np.testing.assert_array_equal(ds_a.X_train, ds_b.X_train)
    # Different seed → different split (with high probability).
    assert not np.array_equal(ds_a.X_train, ds_c.X_train)


def test_classes_attribute_when_sklearn_provides_names() -> None:
    ds = load_iris()
    assert ds.classes is not None
    assert len(ds.classes) == 3


def test_repr_includes_key_fields() -> None:
    ds = load_iris()
    s = repr(ds)
    assert "iris" in s
    assert "n_samples=150" in s
    assert "n_classes=3" in s


def test_dispatch_by_name_returns_same_dataset() -> None:
    direct = load_iris()
    via_dispatch = load("iris")
    assert direct.name == via_dispatch.name
    assert direct.n_samples == via_dispatch.n_samples


def test_dispatch_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        load("nonexistent_dataset_xyz")


def test_all_datasets_registry_covers_every_exported_loader() -> None:
    # Four sklearn-backed + four OpenML-backed = 8 entries.
    assert len(ALL_DATASETS) == 8
    for name in [
        "iris",
        "wine",
        "breast_cancer",
        "digits",
        "mnist",
        "fashion_mnist",
        "isolet",
        "ucihar",
    ]:
        assert name in ALL_DATASETS


def test_dispatch_accepts_kwargs() -> None:
    ds = load("iris", test_size=0.25, random_state=0)
    assert abs(ds.n_test - int(0.25 * ds.n_samples)) <= 2


# ----------------------------------------------------------------------
# Network-gated integration test (skipped by default).
# ----------------------------------------------------------------------


@pytest.mark.network
def test_openml_loader_smoketest_mnist() -> None:  # pragma: no cover - network
    """End-to-end fetch for MNIST via OpenML. Run manually with `-m network`."""
    from bayes_hdc.datasets import load_mnist

    ds = load_mnist(subsample=200, random_state=0)
    assert ds.name == "mnist"
    assert ds.n_classes == 10
    assert ds.n_features == 784
    assert ds.n_samples == 200
