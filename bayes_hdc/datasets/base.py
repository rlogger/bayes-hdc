# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""``HDCDataset`` — uniform container for HDC benchmark datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HDCDataset:
    """Standardised container for an HDC benchmark dataset.

    All fields are plain NumPy arrays so the container works with any
    downstream HDC pipeline, not just this library. The split fields
    (``X_train`` / ``X_test`` / ``y_train`` / ``y_test``) are produced by
    a stratified 70/30 split by default; the full ``X`` / ``y`` arrays
    are also retained so callers can re-split or cross-validate.

    Attributes:
        name: Short identifier (e.g. ``"isolet"``).
        X: All features, shape ``(n_samples, n_features)``, ``float32``.
        y: All integer labels, shape ``(n_samples,)``, ``int32``.
        X_train, y_train: Training split.
        X_test, y_test: Test split.
        n_classes: Number of distinct labels.
        n_features: Feature dimension.
        description: Human-readable one-line description + citation.
        classes: Optional tuple of class-name strings (e.g. for labelled
            spoken letters ``("a", "b", ...)``). ``None`` if labels are
            just integers.
    """

    name: str
    X: np.ndarray
    y: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_classes: int
    n_features: int
    description: str = ""
    classes: tuple[str, ...] | None = None

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_train(self) -> int:
        return int(self.X_train.shape[0])

    @property
    def n_test(self) -> int:
        return int(self.X_test.shape[0])

    def __repr__(self) -> str:
        return (
            f"HDCDataset(name={self.name!r}, "
            f"n_samples={self.n_samples}, "
            f"n_features={self.n_features}, "
            f"n_classes={self.n_classes}, "
            f"train/test={self.n_train}/{self.n_test})"
        )


__all__ = ["HDCDataset"]
