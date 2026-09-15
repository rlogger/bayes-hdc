# SPDX-License-Identifier: MIT
# Copyright (c) 2026 bayes-hdc contributors

"""Store and retrieve a short sequence with FHRR hypervectors.

Fourier Holographic Reduced Representations (FHRR) use complex-valued
hypervectors whose coordinates are unit phasors. Binding multiplies phasors
(adding their phases), inversion takes the complex conjugate, and bundling
forms a normalized superposition.

This example encodes ``RED -> GREEN -> BLUE`` in one hypervector. Permutations
of a random position vector act as position roles; binding associates each
role with a token. Unbinding a position role produces a noisy token that is
cleaned up by similarity search against the token codebook.

Run::

    python examples/fhrr_demo.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from bayes_hdc import FHRR, permute

DIMENSIONS = 2_048
SEED = 11
TOKENS = ("RED", "GREEN", "BLUE")


def main() -> None:
    """Encode the token sequence and retrieve every position."""
    model = FHRR.create(dimensions=DIMENSIONS)
    token_key, position_key = jax.random.split(jax.random.PRNGKey(SEED))

    token_hvs = model.random(token_key, (len(TOKENS), DIMENSIONS))
    position_seed = model.random(position_key, (DIMENSIONS,))
    position_hvs = jnp.stack([permute(position_seed, i) for i in range(len(TOKENS))])

    # Each bound entry associates a position role with one token. Bundling the
    # entries stores the whole sequence in a single fixed-size hypervector.
    entries = jax.vmap(model.bind)(position_hvs, token_hvs)
    sequence_hv = model.bundle(entries, axis=0)

    print("FHRR sequence memory")
    print(f"Dimensions: {DIMENSIONS}")
    print(f"Token dtype: {token_hvs.dtype}")
    print(f"Mean coordinate magnitude: {float(jnp.mean(jnp.abs(token_hvs))):.3f}")
    print("\nposition  expected  retrieved  similarity  margin")
    print("--------  --------  ---------  ----------  ------")

    for i, (expected, position_hv) in enumerate(zip(TOKENS, position_hvs)):
        query = model.bind(sequence_hv, model.inverse(position_hv))
        similarities = jax.vmap(lambda candidate: model.similarity(query, candidate))(token_hvs)
        ranking = jnp.argsort(similarities)[::-1]
        best_index = int(ranking[0])
        best = TOKENS[best_index]
        margin = float(similarities[ranking[0]] - similarities[ranking[1]])

        print(
            f"{i:^8}  {expected:<8}  {best:<9}  "
            f"{float(similarities[best_index]):>10.3f}  {margin:>6.3f}"
        )
        assert best == expected

    print("\nAll positions recovered correctly.")


if __name__ == "__main__":
    main()
