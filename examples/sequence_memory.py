# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Position-addressable sequence memory with HDC.

Encodes a sequence of tokens as a single fixed-size hypervector by binding
each token to its ordinal position (via cyclic permutation) and bundling the
result:

    seq_hv = sum_i permute(token_hv[tok_i], i)

At query time, the token at position ``i`` is recovered by permuting the
bundled HV back by ``-i`` and running :func:`cleanup` against the codebook.
Uncertainty in the retrieval is quantified by the gap between the top-1 and
top-2 similarities (``retrieval_confidence``): large gaps mean the recovered
symbol is unambiguous, narrow gaps mean multiple tokens are plausible.

The example encodes a 12-token sentence and retrieves each position in turn,
reporting the recovered token, its cosine similarity to the ground truth, and
the retrieval confidence gap.

Run::

    python examples/sequence_memory.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from bayes_hdc import (
    MAP,
    RandomEncoder,
    cosine_similarity,
    permute,
    retrieval_confidence,
)

DIMS = 10_000
SEED = 42

SENTENCE = "the quick brown fox jumps over the lazy dog in the park".split()


def main() -> None:
    print("Sequence memory — encode and retrieve position-by-position\n")
    print(f"Dimensions: {DIMS}")
    print(f"Sequence:   {' '.join(SENTENCE)}  ({len(SENTENCE)} tokens)\n")

    # Build a codebook with one hypervector per unique word.
    vocab = sorted(set(SENTENCE))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    indices = jnp.asarray([word_to_idx[w] for w in SENTENCE], dtype=jnp.int32)

    key = jax.random.PRNGKey(SEED)
    vsa = MAP.create(dimensions=DIMS)
    # Single-feature RandomEncoder gives us the codebook we need.
    enc = RandomEncoder.create(
        num_features=1,
        num_values=len(vocab),
        dimensions=DIMS,
        vsa_model=vsa,
        key=key,
    )
    codebook = enc.codebook[0]  # shape (|vocab|, D)

    # Encode: bundle( permute(codebook[idx_i], i)  for each i )
    @jax.jit
    def encode_sequence(indices_: jax.Array) -> jax.Array:
        def one(i: jax.Array, idx: jax.Array) -> jax.Array:
            return jnp.roll(codebook[idx], i.astype(jnp.int32))

        positioned = jax.vmap(one)(jnp.arange(len(indices_)), indices_)
        return jnp.sum(positioned, axis=0)

    seq_hv = encode_sequence(indices)
    print(
        f"Encoded sequence HV: shape={tuple(seq_hv.shape)}, "
        f"L2 norm={float(jnp.linalg.norm(seq_hv)):.2f}\n"
    )

    # Retrieve: for each position i, un-permute then cleanup against codebook.
    print("Retrieval per position:")
    print("=" * 72)
    print(f"{'pos':<4} {'truth':<8} {'retrieved':<10} {'cos sim':>8} {'conf gap':>10}  ok?")
    print("-" * 72)
    correct = 0
    sims_fn = jax.vmap(cosine_similarity, in_axes=(None, 0))
    for i, true_word in enumerate(SENTENCE):
        recovered_hv = permute(seq_hv, -i)
        # argmax over cosine similarity to every codebook entry.
        sims = sims_fn(recovered_hv, codebook)
        best_idx = int(jnp.argmax(sims))
        retrieved = vocab[best_idx]

        sim = float(sims[best_idx])
        gap = float(retrieval_confidence(recovered_hv, codebook))
        ok = "✓" if retrieved == true_word else "✗"
        if ok == "✓":
            correct += 1
        print(f"{i:<4} {true_word:<8} {retrieved:<10} {sim:>8.3f} {gap:>10.3f}   {ok}")
    print("=" * 72)
    print(f"\nRetrieval accuracy: {correct}/{len(SENTENCE)} = {correct / len(SENTENCE):.1%}")

    print("\nInterpretation:")
    print("  - High cos sim ≈ the recovered HV is close to a single codebook entry.")
    print("  - Large conf gap ≈ the top candidate clearly beats the runners-up.")
    print("  - Repeated tokens (e.g. 'the' at positions 0 and 6) are disambiguated by")
    print("    the position-bound encoding — each occurrence has a different bound HV.")


if __name__ == "__main__":
    main()
