#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 R.S.

"""Sequence-encoding capacity benchmark: flat vs. hierarchical.

Sweeps sequence length T at fixed dimension d=4096 and codebook size
n_codebook=256, measuring per-position retrieval accuracy under cleanup
for both ``Sequence`` and ``HierarchicalSequence``. Each datapoint
averages over multiple random seeds so the curves are not dominated by
seed jitter at the crossover.

The hierarchical representation stores chunk vectors in addition to its top
vector, whereas the flat representation stores one vector. Report this storage
cost alongside accuracy; gains do not establish greater fixed-memory capacity.
No asymptotic capacity law is inferred from this finite empirical sweep.

Run::

    python benchmarks/benchmark_sequence_capacity.py

The script writes ``benchmarks/sequence_capacity_results.json`` (also
gitignored) and prints a table to stdout.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import jax
import jax.numpy as jnp

from bayes_hdc import HierarchicalSequence, Sequence

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

DIMS = 4096
N_CODEBOOK = 256
CHUNK_SIZE = 16
T_VALUES = (16, 32, 64, 128, 200, 300, 400, 600, 800)
N_SEEDS = 3  # average over this many random codebook + item-id draws


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_codebook(key: jax.Array, n: int, d: int) -> jax.Array:
    v = jax.random.normal(key, (n, d))
    return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def _retrieval_accuracy(
    sequence_get_fn,
    n_items: int,
    codebook: jax.Array,
    item_ids: jax.Array,
) -> float:
    correct = 0
    for i in range(n_items):
        recovered = sequence_get_fn(int(i))
        sims = codebook @ recovered
        if int(jnp.argmax(sims)) == int(item_ids[i]):
            correct += 1
    return correct / n_items


def _measure_one_seed(
    seed: int,
    T: int,
    d: int,
    n_codebook: int,
    chunk_size: int,
) -> tuple[float, float]:
    key = jax.random.PRNGKey(seed)
    cb_key, item_key = jax.random.split(key)
    codebook = _make_codebook(cb_key, n=n_codebook, d=d)
    item_ids = jax.random.randint(item_key, (T,), 0, n_codebook)
    items = codebook[item_ids]

    flat = Sequence.from_vectors(items)
    flat_acc = _retrieval_accuracy(flat.get, T, codebook, item_ids)

    hs = HierarchicalSequence.from_vectors(items, chunk_size=chunk_size)
    hs_acc = _retrieval_accuracy(hs.get, T, codebook, item_ids)

    return flat_acc, hs_acc


# ----------------------------------------------------------------------
# Main sweep
# ----------------------------------------------------------------------


def main() -> int:
    print("=" * 78)
    print("Sequence-encoding capacity: flat Sequence vs HierarchicalSequence")
    print(f"d = {DIMS}    n_codebook = {N_CODEBOOK}    chunk_size = {CHUNK_SIZE}")
    print(f"seeds per T  = {N_SEEDS}    T values = {T_VALUES}")
    print("=" * 78)
    print(f"\n{'T':>5}  {'flat acc':>12}  {'hier acc':>12}  {'gain':>10}")
    print("-" * 50)

    results: list[dict[str, Any]] = []
    for T in T_VALUES:
        flat_accs = []
        hs_accs = []
        for seed in range(N_SEEDS):
            f, h = _measure_one_seed(seed, T, DIMS, N_CODEBOOK, CHUNK_SIZE)
            flat_accs.append(f)
            hs_accs.append(h)
        flat_mean = sum(flat_accs) / len(flat_accs)
        hs_mean = sum(hs_accs) / len(hs_accs)
        gain = hs_mean - flat_mean
        marker = " ✓" if gain > 0.05 else ""
        print(f"{T:>5}  {flat_mean:>12.3f}  {hs_mean:>12.3f}  {gain:+10.3f}{marker}")
        results.append(
            {
                "T": T,
                "flat_accuracy_mean": round(flat_mean, 4),
                "hier_accuracy_mean": round(hs_mean, 4),
                "flat_accuracy_per_seed": [round(x, 4) for x in flat_accs],
                "hier_accuracy_per_seed": [round(x, 4) for x in hs_accs],
                "gain": round(gain, 4),
                "flat_stored_scalars": DIMS,
                "hierarchical_stored_scalars": (1 + (T + CHUNK_SIZE - 1) // CHUNK_SIZE) * DIMS,
            }
        )

    print("-" * 50)
    print("\nGains marked ✓ are where HierarchicalSequence wins by ≥ 5 pp.")
    print("Compare accuracy together with stored scalar counts; memory budgets differ.")

    out_path = os.path.join(os.path.dirname(__file__), "sequence_capacity_results.json")
    with open(out_path, "w") as fh:
        json.dump(
            {
                "dimensions": DIMS,
                "n_codebook": N_CODEBOOK,
                "chunk_size": CHUNK_SIZE,
                "n_seeds": N_SEEDS,
                "T_values": list(T_VALUES),
                "results": results,
            },
            fh,
            indent=2,
        )
    print(f"\nResults saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
