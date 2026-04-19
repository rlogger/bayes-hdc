#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""
Benchmark JAX-HDC vs TorchHD on equivalent operations.

Methodology (reproducible):
- Dimensions: 10,000
- Warmup: 20 iterations (JAX JIT + PyTorch cache)
- Trials: 200 per operation
- Device: CPU (both libraries) for fair comparison
- Results: mean ± std (ms)

Usage:
    pip install -e ".[benchmark]"
    python benchmarks/benchmark_compare.py

Output: Prints table and optionally saves results to benchmark_results.json
"""

import json
import os
import sys
import time
from typing import Callable


def benchmark_jax(
    fn: Callable, *args: object, warmup: int = 20, trials: int = 200, **kwargs: object
) -> tuple[float, float]:
    """Benchmark a JAX function. Returns (mean_ms, std_ms)."""
    import jax

    for _ in range(warmup):
        out = fn(*args, **kwargs)
        jax.block_until_ready(out)

    times: list[float] = []
    for _ in range(trials):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        jax.block_until_ready(out)
        times.append((time.perf_counter() - t0) * 1000)

    mean = sum(times) / len(times)
    var = sum((t - mean) ** 2 for t in times) / len(times)
    std = var**0.5
    return mean, std


def benchmark_torch(
    fn: Callable, *args: object, warmup: int = 20, trials: int = 200, **kwargs: object
) -> tuple[float, float]:
    """Benchmark a PyTorch/TorchHD function. Returns (mean_ms, std_ms)."""
    import torch

    for _ in range(warmup):
        fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    mean = sum(times) / len(times)
    var = sum((t - mean) ** 2 for t in times) / len(times)
    std = var**0.5
    return mean, std


def run_jax_hdc_benchmarks(
    dim: int = 10000, warmup: int = 20, trials: int = 200
) -> dict[str, float]:
    """Run JAX-HDC benchmarks."""
    import jax

    from jax_hdc import MAP
    from jax_hdc import functional as F
    from jax_hdc.embeddings import RandomEncoder

    key = jax.random.PRNGKey(42)
    model = MAP.create(dimensions=dim)

    x = model.random(key, (dim,))
    y = model.random(jax.random.split(key)[1], (dim,))
    vectors = model.random(key, (10, dim))

    encoder = RandomEncoder.create(
        num_features=20, num_values=10, dimensions=dim, vsa_model=model, key=key
    )
    data = jax.random.randint(key, (100, 20), 0, 10)

    results: dict[str, tuple[float, float]] = {}

    mean, std = benchmark_jax(F.bind_map, x, y, warmup=warmup, trials=trials)
    results["MAP bind (2 HVs)"] = (mean, std)

    mean, std = benchmark_jax(F.bundle_map, vectors, 0, warmup=warmup, trials=trials)
    results["MAP bundle (10 HVs)"] = (mean, std)

    mean, std = benchmark_jax(F.cosine_similarity, x, y, warmup=warmup, trials=trials)
    results["Cosine similarity"] = (mean, std)

    mean, std = benchmark_jax(encoder.encode_batch, data, warmup=warmup, trials=trials)
    results["RandomEncoder (100×20)"] = (mean, std)

    return results


def run_torchhd_benchmarks(
    dim: int = 10000, warmup: int = 20, trials: int = 200
) -> dict[str, float]:
    """Run TorchHD benchmarks. Equivalent operations."""
    import torch
    import torchhd
    from torchhd import embeddings

    device = torch.device("cpu")  # CPU for fair comparison with JAX default
    torch.manual_seed(42)

    # MAP-style: torchhd uses +1/-1 by default (MAP)
    x = torchhd.random(1, dim, device=device).squeeze(0)
    y = torchhd.random(1, dim, device=device).squeeze(0)
    vectors = torchhd.random(10, dim, device=device)

    # TorchHD Random(num_embeddings, embedding_dim): 20 features × 10 values = 200 embeddings
    enc = embeddings.Random(200, dim, vsa="MAP", device=device)
    # data indices: flatten feature*10 + value for each of 100×20
    data = torch.randint(0, 10, (100, 20), device=device)

    results: dict[str, tuple[float, float]] = {}

    def bind_two():
        with torch.no_grad():
            return torchhd.bind(x, y)

    mean, std = benchmark_torch(bind_two, warmup=warmup, trials=trials)
    results["MAP bind (2 HVs)"] = (mean, std)

    def bundle_many():
        with torch.no_grad():
            return torchhd.multiset(vectors)

    mean, std = benchmark_torch(bundle_many, warmup=warmup, trials=trials)
    results["MAP bundle (10 HVs)"] = (mean, std)

    def cos_sim():
        with torch.no_grad():
            return torchhd.cosine_similarity(x, y.unsqueeze(0)).item()

    mean, std = benchmark_torch(cos_sim, warmup=warmup, trials=trials)
    results["Cosine similarity"] = (mean, std)

    def encode_batch():
        with torch.no_grad():
            # indices: (100, 20) - feature_id*10 + value for each position
            indices = torch.arange(20, device=device).unsqueeze(0) * 10 + data
            # enc(indices) -> (100, 20, dim), multiset along features -> (100, dim)
            sampled = enc(indices)
            return torchhd.multiset(sampled)

    mean, std = benchmark_torch(encode_batch, warmup=warmup, trials=trials)
    results["RandomEncoder (100×20)"] = (mean, std)

    return results


def main() -> int:
    dim = 10000
    warmup = 20
    trials = 200

    print("=" * 70)
    print("JAX-HDC vs TorchHD Performance Comparison")
    print("=" * 70)
    print(f"Dimensions: {dim} | Warmup: {warmup} | Trials: {trials}")
    print("Device: CPU (both libraries)")
    print("=" * 70)

    # JAX-HDC
    print("\nRunning JAX-HDC benchmarks...")
    try:
        jax_results = run_jax_hdc_benchmarks(dim=dim, warmup=warmup, trials=trials)
    except Exception as e:
        print(f"JAX-HDC benchmark failed: {e}")
        return 1

    # TorchHD
    print("Running TorchHD benchmarks...")
    try:
        torch_results = run_torchhd_benchmarks(dim=dim, warmup=warmup, trials=trials)
    except ImportError as e:
        print(f"TorchHD not installed: {e}")
        print("Install with: pip install torch torch-hd")
        return 1
    except Exception as e:
        print(f"TorchHD benchmark failed: {e}")
        return 1

    # Comparison table
    ops = ["MAP bind (2 HVs)", "MAP bundle (10 HVs)", "Cosine similarity", "RandomEncoder (100×20)"]
    print("\n" + "-" * 70)
    print(f"{'Operation':<30} {'JAX-HDC (ms)':<18} {'TorchHD (ms)':<18} {'Speedup':<10}")
    print("-" * 70)

    report: dict[str, dict] = {}
    for op in ops:
        jax_mean, jax_std = jax_results[op]
        torch_mean, torch_std = torch_results[op]
        speedup = torch_mean / jax_mean if jax_mean > 0 else 0
        report[op] = {
            "jax_hdc_ms": round(jax_mean, 4),
            "jax_hdc_std_ms": round(jax_std, 4),
            "torchhd_ms": round(torch_mean, 4),
            "torchhd_std_ms": round(torch_std, 4),
            "speedup": round(speedup, 2),
        }
        print(
            f"{op:<30} {jax_mean:>8.3f} ± {jax_std:.3f}"
            f"   {torch_mean:>8.3f} ± {torch_std:.3f}   {speedup:.2f}x"
        )

    print("-" * 70)
    print("\nNote: Results vary by hardware. Run on your machine for local numbers.")
    print("Methodology: same dimensions, warmup, trials; CPU mode.")
    print("=" * 70)

    # Save for README/paper
    out_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "dimensions": dim,
                "warmup": warmup,
                "trials": trials,
                "device": "CPU",
                "operations": report,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
