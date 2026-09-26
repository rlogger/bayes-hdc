# SPDX-License-Identifier: MIT
# Copyright (c) 2026 R.S.

"""Internal validation that preserves tracing of valid JAX inputs.

Shapes and dtypes are checked while tracing. Value checks run eagerly;
callers compiling whole pipelines must validate data before compilation.
"""

import math
import numbers
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value}")


def finite_scalar(value: Any, name: str, *, minimum: float = 0.0, strict: bool = False) -> None:
    if isinstance(value, jax.core.Tracer):
        return
    valid = math.isfinite(float(value))
    valid = valid and (float(value) > minimum if strict else float(value) >= minimum)
    if not valid:
        relation = ">" if strict else ">="
        raise ValueError(f"{name} must be finite and {relation} {minimum}; got {value}")


def finite_array(value: Any, name: str) -> None:
    if not isinstance(value, jax.core.Tracer) and not np.all(np.isfinite(np.asarray(value))):
        raise ValueError(f"{name} must contain only finite values")


def labels_array(labels: Any, n: int, num_classes: int, name: str = "labels") -> jax.Array:
    # Inspect host values before JAX can narrow int64/uint64 to int32.
    if not isinstance(labels, jax.core.Tracer):
        host = np.asarray(labels)
        if host.shape != (n,) or not np.issubdtype(host.dtype, np.integer):
            raise ValueError(f"{name} must contain integer class indices with shape ({n},)")
        if np.any((host < 0) | (host >= num_classes)):
            raise ValueError(f"{name} must be in [0, {num_classes})")
    labels = jnp.asarray(labels)
    if labels.shape != (n,) or not jnp.issubdtype(labels.dtype, jnp.integer):
        raise ValueError(f"{name} must contain integer class indices with shape ({n},)")
    return labels


def training_arrays(
    hvs: Any, labels: Any, dimensions: int, num_classes: int
) -> tuple[jax.Array, jax.Array]:
    hvs = jnp.asarray(hvs)
    if hvs.ndim != 2 or hvs.shape[1] != dimensions:
        raise ValueError(f"train_hvs must have shape (n, {dimensions})")
    if hvs.shape[0] == 0:
        raise ValueError("training data is empty")
    finite_array(hvs, "train_hvs")
    return hvs, labels_array(labels, hvs.shape[0], num_classes, "train_labels")


def probabilities_array(probs: Any, name: str = "probs") -> jax.Array:
    probs = jnp.asarray(probs)
    if probs.ndim not in (1, 2) or probs.shape[-1] == 0:
        raise ValueError(f"{name} must have shape (k,) or (n, k), with k >= 1")
    if jnp.iscomplexobj(probs):
        raise ValueError(f"{name} must contain real probabilities")
    if not jnp.issubdtype(probs.dtype, jnp.floating):
        probs = probs.astype(jnp.float32)
    if not isinstance(probs, jax.core.Tracer):
        host = np.asarray(probs)
        if not np.all(np.isfinite(host)) or np.any((host < 0) | (host > 1)):
            raise ValueError(f"{name} must contain finite probabilities in [0, 1]")
        if not np.allclose(host.sum(axis=-1), 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError(f"{name} rows must sum to one")
    return probs


def sample_label(
    sample: Any, label: Any, dimensions: int, num_classes: int
) -> tuple[jax.Array, jax.Array]:
    sample = jnp.asarray(sample)
    if sample.shape != (dimensions,):
        raise ValueError(f"sample must have shape ({dimensions},)")
    finite_array(sample, "sample")
    label = label if isinstance(label, jax.core.Tracer) else np.asarray(label)
    if label.ndim != 0:
        raise ValueError("label must be a scalar integer")
    label = labels_array(label[None], 1, num_classes, "label")[0]
    return sample, label
