# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Statistical metrics, capacity analysis, and calibration diagnostics.

Two families of tools live here:

1. **Capacity and representation diagnostics**:
   :func:`bundle_snr`, :func:`bundle_capacity`, :func:`effective_dimensions`,
   :func:`sparsity`, :func:`signal_energy`, :func:`saturation`,
   :func:`cosine_matrix`, :func:`retrieval_confidence`.

2. **Calibration metrics for the Bayesian layer**:
   :func:`expected_calibration_error`, :func:`maximum_calibration_error`,
   :func:`brier_score`, :func:`sharpness`, :func:`reliability_curve`,
   :func:`negative_log_likelihood`.  These quantify how well a
   classifier's reported probabilities match empirical accuracies.

Calibration is the empirical back-end of the Bayesian contribution. A
Bayesian hypervector that propagates variance but reports miscalibrated
probabilities is no better than a deterministic softmax classifier.
"""

import functools

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


# ----------------------------------------------------------------------
# Calibration metrics
# ----------------------------------------------------------------------


@functools.partial(jax.jit, static_argnames=("n_bins",))
def expected_calibration_error(
    probs: jax.Array,
    labels: jax.Array,
    n_bins: int = 15,
) -> jax.Array:
    r"""Expected Calibration Error (ECE) over equal-width confidence bins.

    ECE is the empirical gap between confidence and accuracy, averaged over
    confidence bins:

    .. math::
        \mathrm{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{n} \,
        \big| \mathrm{acc}(B_b) - \mathrm{conf}(B_b) \big|

    where :math:`B_b` is the set of samples whose top-1 probability falls in
    bin :math:`b`. An ECE of 0 means the classifier is perfectly calibrated.

    Args:
        probs: Class probabilities of shape ``(n, k)``. Rows must sum to 1.
        labels: Integer class labels of shape ``(n,)``.
        n_bins: Number of equal-width bins on [0, 1].

    Returns:
        Scalar ECE in ``[0, 1]``.
    """
    confidences = jnp.max(probs, axis=-1)
    predictions = jnp.argmax(probs, axis=-1)
    correct = (predictions == labels).astype(jnp.float32)

    bin_idx = jnp.clip(
        jnp.floor(confidences * n_bins).astype(jnp.int32), 0, n_bins - 1
    )

    sums_correct = jax.ops.segment_sum(correct, bin_idx, num_segments=n_bins)
    sums_conf = jax.ops.segment_sum(confidences, bin_idx, num_segments=n_bins)
    counts = jax.ops.segment_sum(jnp.ones_like(confidences), bin_idx, num_segments=n_bins)

    safe_counts = jnp.maximum(counts, 1.0)
    mean_acc = sums_correct / safe_counts
    mean_conf = sums_conf / safe_counts

    n = confidences.shape[0]
    weights = counts / n
    return jnp.sum(weights * jnp.abs(mean_acc - mean_conf))


@functools.partial(jax.jit, static_argnames=("n_bins",))
def maximum_calibration_error(
    probs: jax.Array,
    labels: jax.Array,
    n_bins: int = 15,
) -> jax.Array:
    """Maximum Calibration Error — the worst bin's accuracy-confidence gap.

    Upper bound on the per-prediction miscalibration. Useful when the tail
    of overconfident predictions matters more than the average.
    """
    confidences = jnp.max(probs, axis=-1)
    predictions = jnp.argmax(probs, axis=-1)
    correct = (predictions == labels).astype(jnp.float32)

    bin_idx = jnp.clip(
        jnp.floor(confidences * n_bins).astype(jnp.int32), 0, n_bins - 1
    )

    sums_correct = jax.ops.segment_sum(correct, bin_idx, num_segments=n_bins)
    sums_conf = jax.ops.segment_sum(confidences, bin_idx, num_segments=n_bins)
    counts = jax.ops.segment_sum(jnp.ones_like(confidences), bin_idx, num_segments=n_bins)

    safe_counts = jnp.maximum(counts, 1.0)
    gaps = jnp.abs(sums_correct / safe_counts - sums_conf / safe_counts)
    # Ignore empty bins
    gaps = jnp.where(counts > 0, gaps, 0.0)
    return jnp.max(gaps)


@functools.partial(jax.jit, static_argnames=("n_classes",))
def brier_score(
    probs: jax.Array,
    labels: jax.Array,
    n_classes: int,
) -> jax.Array:
    r"""Multi-class Brier score — mean squared error vs one-hot labels.

    .. math::
        \mathrm{Brier} = \frac{1}{n} \sum_{i=1}^n \sum_{k=1}^K (p_{i,k} - y_{i,k})^2

    Lower is better; perfectly confident correct predictions score 0,
    uniform predictions score :math:`1 - 1/K`.

    Args:
        probs: Class probabilities of shape ``(n, k)``.
        labels: Integer class labels of shape ``(n,)``.
        n_classes: Number of classes ``k``.

    Returns:
        Scalar Brier score in ``[0, 2]``.
    """
    one_hot = jax.nn.one_hot(labels, n_classes)
    return jnp.mean(jnp.sum((probs - one_hot) ** 2, axis=-1))


@jax.jit
def sharpness(probs: jax.Array) -> jax.Array:
    """Mean top-1 confidence.  High sharpness + low ECE is the goal.

    A perfectly calibrated uniform classifier has sharpness :math:`1/K`;
    a deterministic classifier has sharpness 1.
    """
    return jnp.mean(jnp.max(probs, axis=-1))


@jax.jit
def negative_log_likelihood(probs: jax.Array, labels: jax.Array) -> jax.Array:
    r"""Mean negative log-likelihood (a proper scoring rule).

    .. math::
        \mathrm{NLL} = -\frac{1}{n} \sum_i \log p_{i, y_i}

    Used as the objective for temperature scaling.
    """
    n = labels.shape[0]
    picked = probs[jnp.arange(n), labels]
    return -jnp.mean(jnp.log(jnp.maximum(picked, EPS)))


@functools.partial(jax.jit, static_argnames=("n_bins",))
def reliability_curve(
    probs: jax.Array,
    labels: jax.Array,
    n_bins: int = 15,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Per-bin data for plotting a reliability diagram.

    Returns ``(bin_centers, bin_accuracies, bin_confidences, bin_counts)``,
    each of shape ``(n_bins,)``. Empty bins have zeros in the accuracy and
    confidence slots.
    """
    confidences = jnp.max(probs, axis=-1)
    predictions = jnp.argmax(probs, axis=-1)
    correct = (predictions == labels).astype(jnp.float32)

    bin_idx = jnp.clip(
        jnp.floor(confidences * n_bins).astype(jnp.int32), 0, n_bins - 1
    )

    sums_correct = jax.ops.segment_sum(correct, bin_idx, num_segments=n_bins)
    sums_conf = jax.ops.segment_sum(confidences, bin_idx, num_segments=n_bins)
    counts = jax.ops.segment_sum(jnp.ones_like(confidences), bin_idx, num_segments=n_bins)

    safe = jnp.maximum(counts, 1.0)
    mean_acc = jnp.where(counts > 0, sums_correct / safe, 0.0)
    mean_conf = jnp.where(counts > 0, sums_conf / safe, 0.0)

    edges = jnp.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, mean_acc, mean_conf, counts


__all__ = [
    # Capacity / representation diagnostics
    "bundle_snr",
    "bundle_capacity",
    "effective_dimensions",
    "sparsity",
    "signal_energy",
    "saturation",
    "cosine_matrix",
    "retrieval_confidence",
    # Calibration metrics
    "expected_calibration_error",
    "maximum_calibration_error",
    "brier_score",
    "sharpness",
    "negative_log_likelihood",
    "reliability_curve",
]
