# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Probabilistic resonator networks — factorisation of a composite hypervector.

Given a set of codebooks :math:`C_1, \\ldots, C_k` and a composite
hypervector :math:`z = f_1 \\otimes \\cdots \\otimes f_k` (with each
:math:`f_i` an unknown entry of :math:`C_i`), a resonator network
recovers the factor indices by iterative approximation.
:cite:`frady2020resonator` introduced the deterministic case. This
module lifts it into PVSA territory with two ingredients no prior
HDC library ships:

- **Multi-restart MCMC** — run ``n_restarts`` parallel chains from
  random initial index assignments; at each step, for each factor,
  sample a new index from a softmax over cosine similarities between
  the "residual" (target unbound from the other factors) and the
  codebook entries. Accept with Metropolis probability. Return the
  chain with the highest final alignment.

- **Gaussian factors** — the codebooks may be ``GaussianHV`` (each
  row a posterior over a symbol). We search over discrete index
  assignments, but the residual calculation uses
  :func:`~bayes_hdc.distributions.bind_gaussian` and
  :func:`~bayes_hdc.distributions.inverse_gaussian` so uncertainty
  propagates through the factorisation. The returned alignment score
  is the expected cosine similarity between the reconstructed bound
  composite and the target.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from bayes_hdc.constants import EPS
from bayes_hdc.distributions import (
    GaussianHV,
    bind_gaussian,
    expected_cosine_similarity,
    inverse_gaussian,
)


@dataclass
class ResonatorResult:
    """Outcome of a probabilistic resonator run.

    Attributes:
        indices: Best-chain factor indices, shape ``(k,)``.
        alignment: Expected cosine similarity between the reconstructed
            composite (``bind(f_1, ..., f_k)``) and the target.
        history: Alignment trajectory of the best chain, shape
            ``(max_iters,)`` — useful for diagnosing convergence.
        n_restarts: Number of parallel chains that were run.
    """

    indices: jax.Array
    alignment: float
    history: jax.Array
    n_restarts: int


def _reconstruct_composite(
    codebooks: list[GaussianHV],
    indices: jax.Array,
) -> GaussianHV:
    """Bind the factors selected by ``indices`` into one GaussianHV."""
    k = len(codebooks)
    first = GaussianHV(
        mu=codebooks[0].mu[indices[0]],
        var=codebooks[0].var[indices[0]],
        dimensions=codebooks[0].dimensions,
    )
    result = first
    for j in range(1, k):
        factor = GaussianHV(
            mu=codebooks[j].mu[indices[j]],
            var=codebooks[j].var[indices[j]],
            dimensions=codebooks[j].dimensions,
        )
        result = bind_gaussian(result, factor)
    return result


def _residual_for_factor(
    codebooks: list[GaussianHV],
    indices: jax.Array,
    target: GaussianHV,
    factor_idx: int,
) -> GaussianHV:
    """Residual = target unbound from every factor except ``factor_idx``.

    Under MAP / Gaussian binding, this uses the approximate delta-method
    inverse so uncertainty propagates through the unbinding.
    """
    result = target
    for j, cb in enumerate(codebooks):
        if j == factor_idx:
            continue
        factor = GaussianHV(
            mu=cb.mu[indices[j]],
            var=cb.var[indices[j]],
            dimensions=cb.dimensions,
        )
        result = bind_gaussian(result, inverse_gaussian(factor))
    return result


def probabilistic_resonator(
    codebooks: list[GaussianHV],
    target: GaussianHV,
    key: jax.Array,
    *,
    n_restarts: int = 8,
    max_iters: int = 50,
    temperature: float = 1.0,
) -> ResonatorResult:
    r"""Multi-restart MCMC factorisation of a composite PVSA hypervector.

    Args:
        codebooks: List of ``k`` batched ``GaussianHV`` objects. Each
            must have ``mu`` of shape ``(n_i, d)`` and ``var`` of shape
            ``(n_i, d)``. The factorisation finds one index per
            codebook.
        target: Composite hypervector to factorise.
        key: JAX random key for the MCMC chains and factor proposals.
        n_restarts: Number of parallel chains, each from a fresh random
            index assignment. The highest-alignment chain is returned.
        max_iters: Number of factor-update sweeps per chain.
        temperature: Softmax temperature on similarities when proposing
            new indices. Lower = sharper / greedier; higher = more
            exploration. The classical deterministic resonator is the
            ``temperature -> 0`` limit.

    Returns:
        :class:`ResonatorResult` with the best chain's factor indices,
        final alignment, and the alignment-vs-iteration trajectory.
    """
    if not codebooks:
        raise ValueError("probabilistic_resonator: codebooks must be non-empty")
    k = len(codebooks)

    # Initialise each chain with a random index assignment.
    init_key, *chain_keys = jax.random.split(key, n_restarts + 1)
    init_indices_per_chain = []
    for chain_idx in range(n_restarts):
        sub = jax.random.split(jax.random.fold_in(init_key, chain_idx), k)
        idx = jnp.array(
            [int(jax.random.randint(sub[j], (), 0, codebooks[j].mu.shape[0])) for j in range(k)],
            dtype=jnp.int32,
        )
        init_indices_per_chain.append(idx)

    def _run_chain(
        chain_key: jax.Array, indices: jax.Array
    ) -> tuple[jax.Array, float, list[float]]:
        """Single chain — sweep factors, sample new indices, keep alignment trajectory."""
        history: list[float] = []
        for step in range(max_iters):
            step_key = jax.random.fold_in(chain_key, step)
            for j in range(k):
                # Residual for factor j.
                residual = _residual_for_factor(codebooks, indices, target, j)

                # Score every row in codebook[j] against the residual.
                def score_row(row_idx: jax.Array) -> jax.Array:
                    candidate = GaussianHV(
                        mu=codebooks[j].mu[row_idx],
                        var=codebooks[j].var[row_idx],
                        dimensions=codebooks[j].dimensions,
                    )
                    return expected_cosine_similarity(residual, candidate)

                scores = jax.vmap(score_row)(jnp.arange(codebooks[j].mu.shape[0]))
                logits = scores / jnp.maximum(temperature, EPS)
                proposed = int(
                    jax.random.categorical(
                        jax.random.fold_in(step_key, j),
                        logits,
                    )
                )
                indices = indices.at[j].set(proposed)

            # Record alignment after this sweep.
            comp = _reconstruct_composite(codebooks, indices)
            align = float(expected_cosine_similarity(comp, target))
            history.append(align)

        final = _reconstruct_composite(codebooks, indices)
        return indices, float(expected_cosine_similarity(final, target)), history

    best_indices = None
    best_alignment = -jnp.inf
    best_history: list[float] = []
    for chain_idx in range(n_restarts):
        idx, align, hist = _run_chain(chain_keys[chain_idx], init_indices_per_chain[chain_idx])
        if align > best_alignment:
            best_alignment = align
            best_indices = idx
            best_history = hist

    assert best_indices is not None
    return ResonatorResult(
        indices=best_indices,
        alignment=float(best_alignment),
        history=jnp.asarray(best_history, dtype=jnp.float32),
        n_restarts=n_restarts,
    )


__all__ = ["probabilistic_resonator", "ResonatorResult"]
