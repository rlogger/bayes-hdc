#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""
Performance benchmarks for JAX-HDC core operations.

Reproducible methodology:
- Warmup: 10 iterations before timing
- Trials: 100 iterations per benchmark
- Report: mean, std, min, max (ms)
- Dimensions: 10,000 (standard HDC size)

Usage:
    python benchmarks/benchmark_operations.py

Or with pytest (for CI):
    pytest benchmarks/ -v --benchmark-only
"""

import jax

from jax_hdc import MAP
from jax_hdc import functional as F
from jax_hdc.embeddings import RandomEncoder
from jax_hdc.utils import benchmark_function


def run_benchmarks():
    """Run all operation benchmarks and print results."""
    key = jax.random.PRNGKey(42)
    dim = 10000

    print("=" * 60)
    print("JAX-HDC Performance Benchmarks")
    print("Dimensions: 10,000 | Warmup: 10 | Trials: 100")
    print("=" * 60)

    # BSC operations
    x_bsc = jax.random.bernoulli(key, 0.5, (dim,))
    y_bsc = jax.random.bernoulli(jax.random.split(key)[1], 0.5, (dim,))
    vectors_bsc = jax.random.bernoulli(key, 0.5, (10, dim))

    stats = benchmark_function(F.bind_bsc, x_bsc, y_bsc)
    print(f"\nBSC bind:     {stats['mean_ms']:.3f} ± {stats['std_ms']:.3f} ms")

    stats = benchmark_function(F.bundle_bsc, vectors_bsc, 0)
    print(f"BSC bundle:   {stats['mean_ms']:.3f} ± {stats['std_ms']:.3f} ms")

    stats = benchmark_function(F.hamming_similarity, x_bsc, y_bsc)
    print(f"BSC similarity: {stats['mean_ms']:.3f} ± {stats['std_ms']:.3f} ms")

    # MAP operations
    model = MAP.create(dimensions=dim)
    x_map = model.random(key, (dim,))
    y_map = model.random(jax.random.split(key)[1], (dim,))
    vectors_map = model.random(key, (10, dim))

    stats = benchmark_function(F.bind_map, x_map, y_map)
    print(f"\nMAP bind:     {stats['mean_ms']:.3f} ± {stats['std_ms']:.3f} ms")

    stats = benchmark_function(F.bundle_map, vectors_map, 0)
    print(f"MAP bundle:   {stats['mean_ms']:.3f} ± {stats['std_ms']:.3f} ms")

    stats = benchmark_function(F.cosine_similarity, x_map, y_map)
    print(f"MAP similarity: {stats['mean_ms']:.3f} ± {stats['std_ms']:.3f} ms")

    # Encoding
    encoder = RandomEncoder.create(
        num_features=20,
        num_values=10,
        dimensions=dim,
        vsa_model=model,
        key=key,
    )
    data = jax.random.randint(key, (100, 20), 0, 10)

    stats = benchmark_function(encoder.encode_batch, data)
    print(
        f"\nRandomEncoder.encode_batch (100x20): {stats['mean_ms']:.3f} ± {stats['std_ms']:.3f} ms"
    )

    print("\n" + "=" * 60)
    print("Benchmark complete. Run on same hardware for reproducibility.")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmarks()
