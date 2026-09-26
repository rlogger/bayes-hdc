# SPDX-License-Identifier: MIT
# Copyright (c) 2026 R.S.

"""Illustrative VSA graph matching inspired by Gayler and Levy (2009).

Two 4-cycles have eight isomorphisms, not a unique correspondence. A heuristic
binding/intersection iteration with an added Sinkhorn projection ranks the 16
candidate vertex mappings. This is not an exact reproduction of the paper's
replicator equations. A final assignment is accepted only if it is bijective
and preserves every adjacency; convergence or recovery is not guaranteed.

Run: python examples/gayler_levy_analogy.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bayes_hdc import vector_intersect
from bayes_hdc.functional import bind_map, cosine_similarity

DIMS = 4096
SEED = 2026
N_ITER = 6


def main() -> None:
    print("Gayler & Levy (2009) — VSA-based analogical mapping")
    print(f"  D = {DIMS}    iterations = {N_ITER}\n")

    key = jax.random.PRNGKey(SEED)

    # ----------------------------------------------------------------- 1.
    print("[1] Build atomic vertex hypervectors for both graphs.")
    keys = jax.random.split(key, 8)
    # Bipolar atoms are self-inverse under multiplication, unlike Gaussian MAP draws.
    atoms = [jax.random.rademacher(k, (DIMS,), dtype=jnp.float32) for k in keys]
    a, b, c, d = atoms[:4]
    p, q, r, s = atoms[4:]
    src_names = ["A", "B", "C", "D"]
    tgt_names = ["P", "Q", "R", "S"]
    src_vertices = jnp.stack([a, b, c, d])
    tgt_vertices = jnp.stack([p, q, r, s])

    # Vertex-set bundles.
    V_s = jnp.sum(src_vertices, axis=0) / jnp.linalg.norm(jnp.sum(src_vertices, axis=0))
    V_t = jnp.sum(tgt_vertices, axis=0) / jnp.linalg.norm(jnp.sum(tgt_vertices, axis=0))

    # ----------------------------------------------------------------- 2.
    print("\n[2] Encode the edge sets as bundles of bound vertex pairs.")
    # Source edges: A-B, A-C, B-D, C-D (4-cycle).
    src_edges = jnp.stack(
        [
            bind_map(a, b),
            bind_map(a, c),
            bind_map(b, d),
            bind_map(c, d),
        ]
    )
    # Target edges: P-Q, P-R, Q-S, R-S (same 4-cycle structure).
    tgt_edges = jnp.stack(
        [
            bind_map(p, q),
            bind_map(p, r),
            bind_map(q, s),
            bind_map(r, s),
        ]
    )
    E_s = jnp.sum(src_edges, axis=0) / jnp.linalg.norm(jnp.sum(src_edges, axis=0))
    E_t = jnp.sum(tgt_edges, axis=0) / jnp.linalg.norm(jnp.sum(tgt_edges, axis=0))

    # ----------------------------------------------------------------- 3.
    print("\n[3] Build the 16 candidate-mapping atoms (one per source-target pair).")
    # candidate_atoms[i, j] = bind(src[i], tgt[j])  — the atom representing
    # the mapping "src[i] ↔ tgt[j]".
    candidate_atoms = jnp.stack(
        [bind_map(src, tgt) for src in src_vertices for tgt in tgt_vertices]
    )  # (16, D)
    candidate_labels = [f"{s}↔{t}" for s in src_names for t in tgt_names]
    print(f"      candidate-atom set shape: {tuple(candidate_atoms.shape)}")

    # ----------------------------------------------------------------- 4.
    # Initial mapping bundle: all 16 candidates bundled equally
    # (M_0 = V_s ⊗ V_t = sum over i,j of bind(src_i, tgt_j) up to normalisation).
    M = bind_map(V_s, V_t)
    M = M / (jnp.linalg.norm(M) + 1e-8)

    # Edge-mapping evidence W = E_s ⊗ E_t. Each source-edge × target-edge
    # bind that *agrees* with the correct vertex correspondence reinforces
    # the right mapping atoms.
    W = bind_map(E_s, E_t)
    W = W / (jnp.linalg.norm(W) + 1e-8)

    # ----------------------------------------------------------------- 5.
    print("\n[4] Heuristic iteration: re-weight candidate mappings.")
    print("      At each step we (i) project M onto the candidate atom set,")
    print("      (ii) Sinkhorn-normalise the resulting 4x4 weight matrix so")
    print("          each source vertex and each target vertex receives unit")
    print("          mass — the doubly-stochastic constraint of the assignment")
    print("          problem; this is an illustrative added projection,")
    print("      (iii) rebuild M as the weighted bundle.")
    print()
    print("      iteration  top-3 candidates    (cosine similarity)")
    print("      ---------  -----------------------------------------")

    def _sinkhorn_4x4(weights: jnp.ndarray, n_iter: int = 8) -> jnp.ndarray:
        # Doubly-stochastic projection of a 4x4 non-negative weight matrix
        # via alternating row/column normalisation. This is an extra
        # heuristic, not a reproduction of the cited replicator equations.
        w = jnp.maximum(weights, 0.0) + 1e-8
        for _ in range(n_iter):
            w = w / w.sum(axis=1, keepdims=True)
            w = w / w.sum(axis=0, keepdims=True)
        return w

    for it in range(N_ITER + 1):
        sims = jax.vmap(lambda atom: cosine_similarity(M, atom))(candidate_atoms)
        sims_np = np.asarray(sims)
        top3 = np.argsort(sims_np)[::-1][:3]
        top3_str = "    ".join(f"{candidate_labels[k]} ({sims_np[k]:+.3f})" for k in top3)
        print(f"      {it:>9d}  {top3_str}")
        if it == N_ITER:
            break
        # Update: π = M ⊗ W (re-weighted by edge consistency).
        pi = bind_map(M, W)
        pi = pi / (jnp.linalg.norm(pi) + 1e-8)
        # Vector intersection: keep only candidates present in both M and π.
        M_new = vector_intersect(M, pi, candidate_atoms)
        # Project onto the candidate atom set: per-atom weight = cos(M_new, atom).
        weights_flat = jax.vmap(lambda atom: cosine_similarity(M_new, atom))(candidate_atoms)
        # Sinkhorn-normalise the 4x4 weight matrix to enforce the
        # one-source-to-one-target (doubly-stochastic) constraint.
        weights_2d = jnp.maximum(weights_flat, 0.0).reshape(4, 4)
        weights_2d = _sinkhorn_4x4(weights_2d, n_iter=8)
        weights_flat = weights_2d.reshape(-1)
        # Rebuild M as the weighted bundle of candidate atoms.
        M = jnp.sum(candidate_atoms * weights_flat[:, None], axis=0)
        M = M / (jnp.linalg.norm(M) + 1e-8)

    # ----------------------------------------------------------------- 6.
    print("\n[5] Final vertex correspondence (top 4 by cosine similarity):")
    sims = jax.vmap(lambda atom: cosine_similarity(M, atom))(candidate_atoms)
    sims_np = np.asarray(sims)
    top4 = np.argsort(sims_np)[::-1][:4]
    for rank, k in enumerate(top4, 1):
        print(f"      {rank}. {candidate_labels[k]}    (cos = {sims_np[k]:+.4f})")

    # Check whether the recovered top-4 is a valid permutation.
    src_used = set()
    tgt_used = set()
    valid = True
    for k in top4:
        i, j = k // 4, k % 4
        if i in src_used or j in tgt_used:
            valid = False
            break
        src_used.add(i)
        tgt_used.add(j)

    if valid:
        assignment = {int(k // 4): int(k % 4) for k in top4}
        edges = {(0, 1), (0, 2), (1, 3), (2, 3)}
        mapped_edges = {tuple(sorted((assignment[i], assignment[j]))) for i, j in edges}
        valid = mapped_edges == edges

    if valid:
        mapping = {}
        for k in top4:
            i, j = k // 4, k % 4
            mapping[src_names[i]] = tgt_names[j]
        print(
            "\n[6] ✓ Verified edge-preserving bijection recovered: "
            + ", ".join(f"{a}→{b}" for a, b in sorted(mapping.items()))
        )
    else:
        print(
            "\n[6] ~ Top-4 candidates do not form a verified isomorphism — try"
            "\n    increasing N_ITER or DIMS, or check the symmetric solution"
            "\n    (the 4-cycle has eight valid isomorphisms)."
        )

    print(
        "\nCandidate similarities are heuristic scores, not proof of an isomorphism.\n"
        "The adjacency check above is the success criterion. The reference\n"
        "implementation is at github.com/simondlevy/GraphIsomorphism; this\n"
        "example does not claim identical dynamics or convergence guarantees."
    )


if __name__ == "__main__":
    main()
