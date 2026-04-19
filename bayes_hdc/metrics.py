# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Statistical metrics and capacity analysis for Hyperdimensional Computing.

Provides tools for analysing the quality of hypervector representations:
signal-to-noise ratio, theoretical capacity bounds, effective dimensionality,
and sparsity.  These are essential for sizing systems, diagnosing retrieval
failures, and tuning dimensionality in real-world deployments.
"""

import jax
import jax.numpy as jnp

from bayes_hdc.constants import EPS


@jax.jit
def bundle_snr(d: int, n: int) -> jax.Array:
    """Expected signal-to-noise ratio after bundling *n* MAP vectors in *d* dims.

    For MAP (real-valued) bundling, the target vector has expected dot d with the
    bundle, while each of the (n-1) interfering vectors contributes noise with
    variance d.  SNR = d / sqrt((n-1) * d) = sqrt(d / (n-1)).

    Args:
        d: Dimensionality
        n: Number of bundled vectors (must be >= 2)

    Returns:
        Expected SNR (higher is better; retrieval is reliable when SNR >> 1)
    """
    return jnp.sqrt(d / jnp.maximum(n - 1, 1).astype(jnp.float32))


@jax.jit
def bundle_capacity(d: int, delta: float = 0.05) -> jax.Array:
    """Maximum number of vectors that can be bundled and still retrieved.

    Returns the largest *n* such that the probability of correct retrieval
    (cosine similarity of the target exceeding all others) stays above
    1 - *delta*.  For MAP with a codebook of size C = n, the approximate
    capacity is proportional to sqrt(d).

    Uses the conservative bound n_max ≈ sqrt(d / (2 * ln(1 / delta))).

    Args:
        d: Dimensionality
        delta: Tolerable error probability (default: 0.05)

    Returns:
        Approximate maximum number of bundled vectors
    """
    return jnp.sqrt(d / (2.0 * jnp.log(1.0 / delta)))


@jax.jit
def effective_dimensions(x: jax.Array) -> jax.Array:
    """Participation ratio measuring how many dimensions carry signal.

    PR = (Σ x_i²)² / Σ x_i⁴

    For a uniform vector PR = d; for a one-hot vector PR = 1.
    Useful for detecting degenerate or collapsed representations.

    Args:
        x: Hypervector of shape (..., d)

    Returns:
        Participation ratio (scalar or batch)
    """
    x2 = x * x
    return jnp.sum(x2, axis=-1) ** 2 / (jnp.sum(x2 * x2, axis=-1) + EPS)


@jax.jit
def sparsity(x: jax.Array, threshold: float = 1e-6) -> jax.Array:
    """Fraction of near-zero elements in a hypervector.

    Args:
        x: Hypervector of shape (..., d)
        threshold: Absolute value below which an element is considered zero

    Returns:
        Sparsity in [0, 1] (1 = all zeros)
    """
    near_zero = jnp.abs(x) < threshold
    return jnp.mean(near_zero.astype(jnp.float32), axis=-1)


@jax.jit
def signal_energy(x: jax.Array) -> jax.Array:
    """L2 energy (squared norm) of a hypervector.

    Useful for monitoring representation magnitude during training or
    encoding.  A near-zero energy indicates a collapsed representation.

    Args:
        x: Hypervector of shape (..., d)

    Returns:
        Squared L2 norm
    """
    return jnp.sum(x * x, axis=-1)


@jax.jit
def saturation(x: jax.Array) -> jax.Array:
    """Fraction of elements near ±1 in a bipolar hypervector.

    Saturation close to 1.0 indicates the representation is fully
    quantised; low saturation indicates an under-committed vector.

    Args:
        x: Bipolar hypervector of shape (..., d)

    Returns:
        Saturation in [0, 1]
    """
    return jnp.mean((jnp.abs(x) > 0.9).astype(jnp.float32), axis=-1)


@jax.jit
def cosine_matrix(vectors: jax.Array) -> jax.Array:
    """Pairwise cosine similarity matrix for a set of hypervectors.

    Useful for checking that a codebook is quasi-orthogonal (off-diagonal
    entries near 0).

    Args:
        vectors: Hypervectors of shape (n, d)

    Returns:
        Similarity matrix of shape (n, n)
    """
    norms = jnp.linalg.norm(vectors, axis=-1, keepdims=True) + EPS
    normed = vectors / norms
    return jnp.clip(normed @ normed.T, -1.0, 1.0)


@jax.jit
def retrieval_confidence(query: jax.Array, codebook: jax.Array) -> jax.Array:
    """Gap between the best and second-best cosine similarity to *codebook*.

    A large positive gap indicates confident retrieval; a gap near zero
    means the query is ambiguous between two or more codebook entries.

    Args:
        query: Query hypervector of shape (d,)
        codebook: Codebook of shape (n, d)

    Returns:
        Confidence gap (best_sim - second_best_sim)
    """
    norms_cb = jnp.linalg.norm(codebook, axis=-1, keepdims=True) + EPS
    normed_cb = codebook / norms_cb
    q_norm = query / (jnp.linalg.norm(query) + EPS)
    sims = normed_cb @ q_norm
    top2 = jax.lax.top_k(sims, k=2)
    return top2[0][0] - top2[0][1]


__all__ = [
    "bundle_snr",
    "bundle_capacity",
    "effective_dimensions",
    "sparsity",
    "signal_energy",
    "saturation",
    "cosine_matrix",
    "retrieval_confidence",
]
