# SPDX-License-Identifier: MIT
# Copyright (c) 2026 R.S.

"""Cosine-softmax associative retrieval in an HDC pipeline.

HopfieldMemory returns a normalised softmax-weighted mixture of stored
patterns. This is attention-style soft cleanup inspired by modern Hopfield
networks, not a reproduction of recurrent dynamics or a calibrated posterior.
For a unique best match its high-inverse-temperature limit is hard cleanup;
ties need not choose a single pattern. The corrupted composite is diagnostic
only; retrieval is evaluated on a noisy stored ROLE vector.

Run: python examples/hopfield_cleanup_hdc.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from bayes_hdc import MAP, HopfieldMemory, bind_map, cleanup, cosine_similarity

DIMS = 4096
SEED = 2026
# Noise as a fraction of the signal's per-dimension magnitude.
# Unit-norm MAP vectors have per-entry scale ~1/sqrt(DIMS).
# Noise uses 30% of that scale; values refer to this synthetic setup.
NOISE_STD = 0.3 / DIMS**0.5


def main() -> None:
    print("Modern Hopfield as a cleanup memory in an HDC pipeline")
    print(f"  D = {DIMS}    noise σ = {NOISE_STD}\n")

    base_key = jax.random.PRNGKey(SEED)
    codebook_key, composite_noise_key, role_noise_key = jax.random.split(base_key, 3)
    model = MAP.create(dimensions=DIMS)

    # ----------------------------------------------------------------- 1.
    print("[1] Codebook of 5 symbols: ROLE, FILLER, FOO, BAR, BAZ.")
    cb_keys = jax.random.split(codebook_key, 5)
    role = model.random(cb_keys[0], (DIMS,))
    filler = model.random(cb_keys[1], (DIMS,))
    foo = model.random(cb_keys[2], (DIMS,))
    bar = model.random(cb_keys[3], (DIMS,))
    baz = model.random(cb_keys[4], (DIMS,))
    codebook = jnp.stack([role, filler, foo, bar, baz])
    labels = ["ROLE", "FILLER", "FOO", "BAR", "BAZ"]

    # ----------------------------------------------------------------- 2.
    print("[2] Composite c = bind(ROLE, FILLER).")
    composite = bind_map(role, filler)
    print(f"      ‖c‖ = {float(jnp.linalg.norm(composite)):.4f}")

    # ----------------------------------------------------------------- 3.
    print(f"\n[3] Corrupt c with Gaussian noise (σ = {NOISE_STD}).")
    noisy = composite + NOISE_STD * jax.random.normal(composite_noise_key, (DIMS,))
    sim_to_clean = float(cosine_similarity(noisy, composite))
    print(f"      cos(noisy, clean) = {sim_to_clean:.4f}")

    # ----------------------------------------------------------------- 4.
    print("\n[4] Build a HopfieldMemory over the codebook (β = 16).")
    mem = HopfieldMemory.create(dimensions=DIMS, beta=16.0)
    for vec in codebook:
        mem = mem.add(vec)
    print(f"      stored {mem.patterns.shape[0]} patterns")

    # The composite is not in the symbol codebook, but ROLE *is*. We
    # demonstrate cleanup on a noisy version of ROLE so both retrievers
    # have a chance to recover it.
    noisy_role = role + NOISE_STD * jax.random.normal(role_noise_key, (DIMS,))
    print(f"      cos(noisy_role, ROLE)    = {float(cosine_similarity(noisy_role, role)):.4f}")

    # ----------------------------------------------------------------- 5.
    print("\n[5] Classical nearest-neighbour cleanup of noisy ROLE.")
    classical, classical_sim = cleanup(noisy_role, codebook, return_similarity=True)
    classical_sim = float(classical_sim)
    classical_recovers = bool(jnp.allclose(classical, role))
    print(f"      sim(classical_out, ROLE) = {float(cosine_similarity(classical, role)):.4f}")
    print(f"      best-match score         = {classical_sim:.4f}")
    print(f"      recovers ROLE exactly?   {classical_recovers}")

    # ----------------------------------------------------------------- 6.
    print("\n[6] Modern Hopfield (softmax-attention) cleanup of noisy ROLE.")
    hopfield_out = mem.retrieve(noisy_role)
    hopfield_sim = float(cosine_similarity(hopfield_out, role))
    print(f"      sim(hopfield_out, ROLE)  = {hopfield_sim:.4f}")
    print(f"      ‖hopfield_out‖           = {float(jnp.linalg.norm(hopfield_out)):.4f}")

    # ----------------------------------------------------------------- 7.
    print("\n[7] Both retrievers project the noisy query toward the stored")
    print("    pattern with the highest cosine similarity. Differences:")
    print("    • Classical cleanup returns ROLE *exactly* (a hard projection).")
    print("    • Hopfield cleanup returns a softmax average — at β=16 it is")
    print(f"      cos = {hopfield_sim:.4f} aligned with ROLE, with mass on the")
    print("      other patterns weighted by similarity. Drop β → 0 to see")
    print("      the limit converge to the codebook centroid; raise β → ∞")
    print("      to recover the classical hard-cleanup behaviour.")

    print("\n[8] Per-pattern softmax weights at the current β (sanity check):")
    sims = jax.vmap(lambda p: cosine_similarity(noisy_role, p))(mem.patterns)
    weights = jax.nn.softmax(mem.beta * sims)
    for label, w, s in zip(labels, list(weights), list(sims)):
        bar_len = max(0, int(40 * float(w)))
        print(f"      {label:<8} sim = {float(s):+.4f}  weight = {float(w):.3f}  {'█' * bar_len}")

    print("\nThe weights are similarity-based attention weights, not calibrated probabilities.")
    print("This demonstration does not establish the capacity bounds of other Hopfield models.")


if __name__ == "__main__":
    main()
