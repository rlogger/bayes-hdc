# SPDX-License-Identifier: MIT
# Copyright (c) 2026 bayes-hdc contributors
#
# tutorials/03_sequences.py
# ─────────────────────────────────────────────────────────────────────────────
# Sequence encoding in bayes-hdc: flat vs. hierarchical
#
# Run:
#   pip install -e ".[dev]"
#   python tutorials/03_sequences.py
#
# No GPU required — finishes in a few seconds on a laptop CPU.
# ─────────────────────────────────────────────────────────────────────────────

# ── 0.  Imports ───────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

from bayes_hdc.structures import HierarchicalSequence, Sequence
from bayes_hdc.vsa import MAP

# Fix the global random key so the tutorial is reproducible.
KEY = jax.random.PRNGKey(0)

# ── 1.  What is a Sequence? ───────────────────────────────────────────────────
#
# HDC encodes ordered lists by permuting each item according to its position
# and bundling all the results into a single hypervector:
#
#   S = P^{T-1}(x_0) + P^{T-2}(x_1) + ... + P^0(x_{T-1})
#
# where P is a cyclic shift.  Retrieval of position i means un-permuting S by
# the right amount, then cleaning up with argmax cosine similarity against the
# item codebook.
#
# `Sequence` in bayes_hdc.structures does exactly this.

# ── 2.  Build a codebook and a short sequence ─────────────────────────────────
#
# Hyperparameters.  d=4096 is large enough for the capacity story to be clear,
# small enough for a tutorial to run in a few seconds on CPU.
D = 4096  # hypervector dimension
N_ITEMS = 256  # size of the item codebook
T_SHORT = 32  # sequence length for the warm-up demo
CHUNK = 16  # chunk size used by HierarchicalSequence

vsa = MAP.create(D)  # MAP VSA: element-wise multiply binding, normalised-sum bundle

# Draw N_ITEMS random unit hypervectors as the codebook.
key, subkey = jax.random.split(KEY)
codebook = vsa.random(subkey, shape=(N_ITEMS, D))  # (N_ITEMS, D)

# Draw T_SHORT item indices at random, then look up the corresponding HVs.
key, subkey = jax.random.split(key)
indices = jax.random.randint(subkey, shape=(T_SHORT,), minval=0, maxval=N_ITEMS)
items = codebook[indices]  # (T_SHORT, D)

# ── 3.  Encode and retrieve — flat Sequence ───────────────────────────────────

seq_flat = Sequence.from_vectors(items)

print(f"Encoded {T_SHORT} items into a single {D}-d hypervector (flat Sequence).")

# Retrieve every position and check against the original codebook.
correct = 0
for i in range(T_SHORT):
    retrieved_hv = seq_flat.get(i)  # one hypervector
    sims = vsa.similarity(retrieved_hv, codebook)  # cosine to every codebook item
    predicted_idx = int(jnp.argmax(sims))
    if predicted_idx == int(indices[i]):
        correct += 1

acc_flat_short = correct / T_SHORT
print(f"  Retrieval accuracy at T={T_SHORT}: {acc_flat_short:.3f}  (expect ~1.0)\n")

# ── 4.  Encode and retrieve — HierarchicalSequence ────────────────────────────
#
# HierarchicalSequence divides the sequence into non-overlapping chunks of
# length `chunk_size`.  Within each chunk it applies the same permute-and-
# bundle scheme as Sequence; then it encodes the chunk hypervectors into a
# second-level Sequence.
#
# The key insight: at retrieval time it projects the noisy outer un-permute
# onto the *cached* chunk codebook before the inner un-permute.  This prunes
# cross-chunk noise first, so only within-chunk noise (from C-1 items, not
# T-1) reaches the item level.  See BENCHMARKS.md and the class docstring for
# the full derivation.

seq_hier = HierarchicalSequence.from_vectors(items, chunk_size=CHUNK)

print(f"Encoded {T_SHORT} items into a HierarchicalSequence (chunk_size={CHUNK}).")

correct_h = 0
for i in range(T_SHORT):
    retrieved_hv = seq_hier.get(i)
    sims = vsa.similarity(retrieved_hv, codebook)
    predicted_idx = int(jnp.argmax(sims))
    if predicted_idx == int(indices[i]):
        correct_h += 1

acc_hier_short = correct_h / T_SHORT
print(f"  Retrieval accuracy at T={T_SHORT}: {acc_hier_short:.3f}  (expect ~1.0)\n")

# ── 5.  Capacity comparison across sequence lengths ───────────────────────────
#
# The flat representation bundles T terms; its per-item SNR scales as 1/sqrt(T)
# (Plate 2003 §6.2), so retrieval degrades past T~200 for d=4096.
#
# The hierarchical variant keeps both layers at O(sqrt(T)) items.  Its SNR is
# dominated by 1/sqrt(chunk_size) regardless of how many chunks there are —
# hence perfect retrieval at T=800 where flat has fallen to ~31%.
# (Full table in BENCHMARKS.md; we sweep a subset here for speed.)

T_VALUES = [32, 64, 128, 200, 300, 400]
N_SEEDS = 2  # BENCHMARKS.md uses 3; 2 keeps the tutorial fast

print(f"{'T':>5}  {'flat acc':>10}  {'hier acc':>10}  {'gain':>6}")
print("-" * 38)

for T in T_VALUES:
    acc_flat_seeds, acc_hier_seeds = [], []

    for seed in range(N_SEEDS):
        rng = jax.random.PRNGKey(seed + 10)

        rng, sk1 = jax.random.split(rng)
        cb = vsa.random(sk1, shape=(N_ITEMS, D))

        rng, sk2 = jax.random.split(rng)
        idx = jax.random.randint(sk2, shape=(T,), minval=0, maxval=N_ITEMS)
        it = cb[idx]

        # Flat
        sf = Sequence.from_vectors(it)
        hits = sum(int(jnp.argmax(vsa.similarity(sf.get(i), cb))) == int(idx[i]) for i in range(T))
        acc_flat_seeds.append(hits / T)

        # Hierarchical
        sh = HierarchicalSequence.from_vectors(it, chunk_size=CHUNK)
        hits_h = sum(
            int(jnp.argmax(vsa.similarity(sh.get(i), cb))) == int(idx[i]) for i in range(T)
        )
        acc_hier_seeds.append(hits_h / T)

    af = sum(acc_flat_seeds) / N_SEEDS
    ah = sum(acc_hier_seeds) / N_SEEDS
    print(f"{T:>5}  {af:>10.3f}  {ah:>10.3f}  {ah - af:>+6.3f}")

print()
print("Flat encoding degrades past T~200; hierarchical stays near-perfect.")
print("See BENCHMARKS.md for the full sweep up to T=800.")

# ── 6.  Key takeaways ─────────────────────────────────────────────────────────
#
# • Sequence.from_vectors(items)                        — flat encoding
# • HierarchicalSequence.from_vectors(items, chunk_size=C) — two-level encoding
# • seq.get(i)                                          — retrieve HV at index i
# • cleanup via argmax cosine against the item codebook — symbolic recovery
#
# For T <= ~100 both representations are perfect.
# For T > ~200 use HierarchicalSequence.
# Both are JAX pytrees: jit, vmap, and grad compose with them.

print("\nDone.")
