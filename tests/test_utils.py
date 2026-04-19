# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Tests for utility functions."""

import jax
import jax.numpy as jnp

from jax_hdc import utils


class TestNormalize:
    def test_normalize_1d(self):
        x = jnp.array([3.0, 4.0])
        normalized = utils.normalize(x)
        norm = jnp.linalg.norm(normalized)
        assert jnp.allclose(norm, 1.0)

    def test_normalize_2d(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (10, 100))
        normalized = utils.normalize(x, axis=-1)
        norms = jnp.linalg.norm(normalized, axis=-1)
        assert jnp.allclose(norms, 1.0)

    def test_normalize_custom_axis(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (10, 100))
        normalized = utils.normalize(x, axis=0)
        norms = jnp.linalg.norm(normalized, axis=0)
        assert jnp.allclose(norms, 1.0)

    def test_normalize_zero_vector(self):
        x = jnp.zeros(10)
        normalized = utils.normalize(x, eps=1e-8)
        assert not jnp.any(jnp.isnan(normalized))


class TestBenchmarkFunction:
    def test_benchmark_simple_function(self):
        def add_arrays(x, y):
            return x + y

        x = jnp.ones(1000)
        y = jnp.ones(1000)

        stats = utils.benchmark_function(add_arrays, x, y, num_trials=10, warmup=2)

        assert "mean_ms" in stats
        assert "std_ms" in stats
        assert "min_ms" in stats
        assert "max_ms" in stats
        assert "median_ms" in stats
        assert stats["num_trials"] == 10
        assert stats["mean_ms"] >= 0
        assert stats["max_ms"] >= stats["min_ms"]

    def test_benchmark_with_kwargs(self):
        def multiply_with_factor(x, factor=2.0):
            return x * factor

        x = jnp.ones(1000)

        stats = utils.benchmark_function(multiply_with_factor, x, factor=3.0, num_trials=5)

        assert stats["num_trials"] == 5
        assert stats["mean_ms"] >= 0
