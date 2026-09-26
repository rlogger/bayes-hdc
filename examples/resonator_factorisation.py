# SPDX-License-Identifier: MIT
# Copyright (c) 2026 R.S.

"""Stochastic factor search over Gaussian hypervector codebooks.

The public probabilistic_resonator uses softmax index proposals and multiple
restarts, then keeps the best final alignment. It is inspired by resonator
factorisation, but has no specified Bayesian stationary distribution or
Metropolis-Hastings correction. Low temperature makes proposals greedier;
it does not make this an exact implementation of Frady et al. (2020).
Gaussian reciprocal and cosine calculations are approximations. Exact
recovery is checked against the generated index tuple.

Run: python examples/resonator_factorisation.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bayes_hdc import (
    GaussianHV,
    bind_gaussian,
    probabilistic_resonator,
)

DIMS = 2048
SEED = 2026
CODEBOOK_SIZES = (8, 6, 5)  # n_1, n_2, n_3 — three slots
TRUE_INDICES = (3, 2, 4)


def _make_codebook(key: jax.Array, n: int, d: int) -> GaussianHV:
    """A batched ``GaussianHV`` of ``n`` random unit-sphere atomic vectors."""
    mu = jax.random.normal(key, (n, d))
    mu = mu / jnp.linalg.norm(mu, axis=-1, keepdims=True)
    # Nonzero variance illustrates the representation. Some means are close
    # to zero, so reciprocal delta approximations can be inaccurate even
    # at this variance; this demo is not posterior inference.
    var = jnp.full((n, d), 0.001)
    return GaussianHV(mu=mu, var=var, dimensions=d)


def main() -> None:
    print("Resonator-network factorisation")
    print(f"  codebooks = {len(CODEBOOK_SIZES)}    sizes = {CODEBOOK_SIZES}    D = {DIMS}\n")
    print(f"  brute-force search space = {np.prod(CODEBOOK_SIZES)} triples\n")

    key = jax.random.PRNGKey(SEED)
    cb_keys = jax.random.split(key, len(CODEBOOK_SIZES))
    codebooks = [_make_codebook(k, n, DIMS) for k, n in zip(cb_keys, CODEBOOK_SIZES)]

    # ----------------------------------------------------------------- 1.
    print("[1] Build a composite by binding one factor per codebook.")
    factors = [
        GaussianHV(
            mu=cb.mu[i],
            var=cb.var[i],
            dimensions=DIMS,
        )
        for cb, i in zip(codebooks, TRUE_INDICES)
    ]
    composite = factors[0]
    for f in factors[1:]:
        composite = bind_gaussian(composite, f)
    print(f"      true factor indices: {TRUE_INDICES}")
    mu_shape = tuple(composite.mu.shape)
    var_shape = tuple(composite.var.shape)
    print(f"      composite shape:     mu={mu_shape}    var={var_shape}")

    # ----------------------------------------------------------------- 2.
    print("\n[2] Recover the factors via probabilistic_resonator.")
    res_key = jax.random.fold_in(key, 1)
    result = probabilistic_resonator(
        codebooks=codebooks,
        target=composite,
        key=res_key,
        n_restarts=16,
        max_iters=50,
        temperature=0.05,
    )
    recovered = tuple(int(x) for x in result.indices)
    print(f"      recovered indices:   {recovered}")
    print(f"      true indices:        {TRUE_INDICES}")
    print(f"      final alignment:     {float(result.alignment):.4f}")
    print(f"      n_restarts:          {result.n_restarts}    max_iters: {len(result.history)}")

    if recovered == TRUE_INDICES:
        print("\n[3] ✓ Exact factorisation recovered.")
    else:
        # Alignment is descriptive; a different tuple is not exact recovery.
        if float(result.alignment) > 0.6:
            print("\n[3] Different tuple with alignment > 0.6; the recovered factors")
            print("    explain the composite well even though indices differ.")
        else:
            print("\n[3] ✗ Did not recover the factorisation. Try increasing")
            print("    n_restarts, max_iters, or tightening the temperature.")

    # ----------------------------------------------------------------- 4.
    print("\n[4] Alignment trajectory of the best chain (key iterations):")
    history = np.asarray(result.history)
    width = 30
    max_alignment = max(float(history.max()), 1e-6) if history.size > 0 else 1.0
    iterations_to_show = sorted(set(list(range(0, len(history), 5)) + [len(history) - 1]))
    for i in iterations_to_show:
        if i < 0 or i >= len(history):
            continue
        v = float(history[i])
        bar_len = max(0, int(width * v / max_alignment))
        print(f"  iter {i:>3d}: {v:+.4f}  {'█' * bar_len}")

    print("\nThis is a stochastic search heuristic with approximate reciprocal moments.")
    print("Restarts may improve search, but give no recovery or posterior-calibration guarantee.")


if __name__ == "__main__":
    main()
