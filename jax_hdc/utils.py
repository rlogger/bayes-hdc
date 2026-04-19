# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Utility functions for JAX-HDC."""

import time
from typing import Any, Callable, Union

import jax
import jax.numpy as jnp


def normalize(x: jax.Array, axis: int = -1, eps: float = 1e-8) -> jax.Array:
    """Normalize vectors to unit length.

    Args:
        x: Input array
        axis: Axis along which to normalize (default: -1)
        eps: Small constant to avoid division by zero
    """
    norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)


def benchmark_function(
    fn: Callable[..., Any], *args: Any, num_trials: int = 100, warmup: int = 10, **kwargs: Any
) -> dict[str, Union[float, int]]:
    """Benchmark a JAX function with proper warmup and async handling.

    Args:
        fn: Function to benchmark
        *args: Positional arguments to fn
        num_trials: Number of trials to run
        warmup: Number of warmup trials
        **kwargs: Keyword arguments to fn

    Returns:
        Dictionary with timing statistics (mean, std, min, max, median in ms)
    """
    for _ in range(warmup):
        result = fn(*args, **kwargs)
        jax.block_until_ready(result)

    times_list: list[float] = []
    for _ in range(num_trials):
        start = time.time()
        result = fn(*args, **kwargs)
        jax.block_until_ready(result)
        end = time.time()
        times_list.append((end - start) * 1000)

    times = jnp.array(times_list)

    return {
        "mean_ms": float(jnp.mean(times)),
        "std_ms": float(jnp.std(times)),
        "min_ms": float(jnp.min(times)),
        "max_ms": float(jnp.max(times)),
        "median_ms": float(jnp.median(times)),
        "num_trials": num_trials,
    }


__all__ = [
    "normalize",
    "benchmark_function",
]
