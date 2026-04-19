# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Performance regression tests for Bayes-HDC models."""

import time

import jax
import pytest

from bayes_hdc import MAP, AdaptiveHDC, CentroidClassifier


@pytest.mark.benchmark
def test_centroid_training_speed():
    """Ensure CentroidClassifier training completes in reasonable time."""
    N, D, C = 10000, 2000, 50
    key = jax.random.PRNGKey(42)

    model = MAP.create(dimensions=D)
    classifier = CentroidClassifier.create(num_classes=C, dimensions=D, vsa_model=model)

    train_hvs = jax.random.normal(key, (N, D))
    train_labels = jax.random.randint(jax.random.split(key)[1], (N,), 0, C)

    classifier = classifier.fit(train_hvs[:100], train_labels[:100])

    start = time.perf_counter()
    classifier = classifier.fit(train_hvs, train_labels)
    classifier.prototypes.block_until_ready()
    end = time.perf_counter()

    duration = end - start
    assert duration < 2.0, f"Training too slow ({duration:.4f}s)"


@pytest.mark.benchmark
def test_adaptive_training_speed():
    """Ensure AdaptiveHDC training completes in reasonable time."""
    N, D, C = 500, 1000, 10
    key = jax.random.PRNGKey(42)

    model = MAP.create(dimensions=D)
    classifier = AdaptiveHDC.create(num_classes=C, dimensions=D, vsa_model=model)

    train_hvs = jax.random.normal(key, (N, D))
    train_labels = jax.random.randint(jax.random.split(key)[1], (N,), 0, C)

    classifier = classifier.fit(train_hvs[:10], train_labels[:10], epochs=1)

    start = time.perf_counter()
    classifier = classifier.fit(train_hvs, train_labels, epochs=2)
    classifier.prototypes.block_until_ready()
    end = time.perf_counter()

    duration = end - start
    assert duration < 10.0, f"Adaptive training too slow ({duration:.2f}s)"
