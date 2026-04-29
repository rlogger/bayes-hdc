# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Core functional operations for Hyperdimensional Computing.

This module provides the fundamental operations for manipulating hypervectors:
- Binding: Combines two hypervectors into a dissimilar result
- Bundling: Aggregates multiple hypervectors into a similar result
- Permutation: Reorders elements to encode sequences
- Similarity: Measures relatedness between hypervectors
- Composite: Multi-vector bind/bundle, n-grams, sequences, hash tables
"""

import functools
from typing import Callable, Union

import jax
import jax.numpy as jnp

from bayes_hdc.constants import EPS


@jax.jit
def bind_bsc(x: jax.Array, y: jax.Array) -> jax.Array:
    """Bind two hypervectors using XOR for Binary Spatter Codes.

    Binding creates a new hypervector that is dissimilar to both inputs.
    XOR is its own inverse, so unbinding is identical to binding.

    Args:
        x: Binary hypervector of shape (..., d)
        y: Binary hypervector of shape (..., d)

    Returns:
        Bound hypervector of shape (..., d), dissimilar to both x and y

    References
    ----------
    Kanerva, P. (1997). Fully Distributed Representation. In Proc. RWC '97,
    pp. 358-365.
    """
    return jnp.logical_xor(x, y)


def bundle_bsc(
    vectors: jax.Array,
    axis: int = 0,
    key: jax.Array | None = None,
) -> jax.Array:
    """Bundle hypervectors using majority rule for Binary Spatter Codes.

    Bundling creates a new hypervector similar to all inputs by taking
    the majority vote at each dimension.

    For an even number of input vectors, ties at exactly half the count
    are broken according to ``key``:

    * ``key is None`` (default) — deterministic: ties map to ``False``.
      This is fast, JIT-friendly, and matches the historical behaviour
      of this library.
    * ``key`` is a ``jax.random.PRNGKey`` — stochastic: ties are broken
      by independent fair coin flips per component, matching Kanerva
      (1997)'s prescription for an unbiased majority rule under even
      input counts.

    For an odd number of input vectors there is no tie and ``key`` has
    no effect.

    Args:
        vectors: Binary hypervectors of shape with axis containing vectors to bundle
        axis: Axis along which to bundle (default: 0)
        key: Optional ``jax.random.PRNGKey``. When provided, ties are
            broken by stochastic coin flip; when ``None``, ties map
            deterministically to ``False``.

    Returns:
        Bundled hypervector, similar to all inputs

    References
    ----------
    Kanerva, P. (1997). Fully Distributed Representation. In Proc. RWC '97,
    pp. 358-365.
    """
    counts = jnp.sum(vectors, axis=axis)
    shape_size = vectors.shape[axis]
    threshold = shape_size / 2.0
    if key is None:
        return counts > threshold
    # Stochastic tie-break: where counts == threshold, return a random bit.
    is_tie = counts == threshold
    coin = jax.random.bernoulli(key, p=0.5, shape=counts.shape)
    return jnp.where(is_tie, coin, counts > threshold)


@jax.jit
def inverse_bsc(x: jax.Array) -> jax.Array:
    """Compute inverse for BSC (identity since XOR is self-inverse)."""
    return x  # XOR is self-inverse


@jax.jit
def hamming_similarity(x: jax.Array, y: jax.Array) -> jax.Array:
    """Compute normalized Hamming similarity between binary hypervectors.

    Returns the fraction of matching bits between two binary vectors.
    Random vectors have similarity ≈ 0.5.

    Args:
        x: Binary hypervector of shape (..., d)
        y: Binary hypervector of shape (..., d)

    Returns:
        Similarity score in [0, 1], where 1 is identical and 0.5 is random
    """
    matches = jnp.logical_not(jnp.logical_xor(x, y))
    return jnp.mean(matches.astype(jnp.float32), axis=-1)


@jax.jit
def bind_map(x: jax.Array, y: jax.Array) -> jax.Array:
    """Bind two hypervectors using element-wise multiplication for MAP.

    For real-valued vectors (MAP model), binding is element-wise multiplication.
    The result is dissimilar to both inputs.

    Args:
        x: Real-valued hypervector of shape (..., d)
        y: Real-valued hypervector of shape (..., d)

    Returns:
        Bound hypervector of shape (..., d)
    """
    return x * y


def bundle_map(vectors: jax.Array, axis: int = 0) -> jax.Array:
    """Bundle hypervectors using normalized sum for MAP.

    For real-valued vectors, bundling is the normalized sum. The result
    is similar to all inputs (high cosine similarity).

    Args:
        vectors: Real-valued hypervectors with axis containing vectors to bundle
        axis: Axis along which to bundle (default: 0)

    Returns:
        Bundled and normalized hypervector
    """
    summed = jnp.sum(vectors, axis=axis)
    norm = jnp.linalg.norm(summed, axis=-1, keepdims=True)
    return summed / (norm + EPS)


@jax.jit
def inverse_map(x: jax.Array, eps: float = EPS) -> jax.Array:
    """Compute inverse for MAP using element-wise reciprocal.

    For MAP binding (element-wise multiplication), the inverse is
    element-wise reciprocal: bind(bind(x, y), inverse(y)) = x.
    Near-zero elements return 0 (no inverse; bind with 0 destroys information).

    Args:
        x: Real-valued hypervector of shape (..., d)
        eps: Small constant for numerical stability (default: EPS)

    Returns:
        Inverse hypervector
    """
    safe_inv = jnp.where(jnp.abs(x) > eps, 1.0 / x, 0.0)
    return safe_inv


def vector_intersect(
    x: jax.Array,
    y: jax.Array,
    atoms: jax.Array,
) -> jax.Array:
    r"""Holistic vector intersection (Gayler & Levy 2009).

    Given two bundle hypervectors ``x`` and ``y`` and a known atom set
    ``atoms`` of shape ``(N, d)``, return a bundle hypervector that
    contains *only* the atoms present in both ``x`` and ``y``,
    weighted by their joint projection.

    For each atom ``a_i`` we compute ``s_i = max(cos(x, a_i), 0) *
    max(cos(y, a_i), 0)`` — a non-negative joint-membership weight that
    is large when ``a_i`` is similar to both inputs and zero when it is
    absent from either. The output is :math:`\sum_i s_i \cdot a_i`.

    This is the explicit-atom-set realisation of the cleanup-memory
    construction in Gayler & Levy (2009, §"Distributed Implementation"
    Figure 2). It is a soft logical AND on bundles, the primitive that
    makes VSA-based graph isomorphism / analogical mapping possible.

    Args:
        x: First bundle hypervector of shape ``(d,)``.
        y: Second bundle hypervector of shape ``(d,)``.
        atoms: Known atom set of shape ``(N, d)``. The intersection is
            constrained to lie in the span of these atoms.

    Returns:
        Hypervector of shape ``(d,)`` representing ``x ∧ y`` —
        the atoms shared by both bundles, weighted by joint similarity.

    References
    ----------
    Gayler, R. W., Levy, S. D. (2009). A Distributed Basis for Analogical
    Mapping. In Proc. 2nd Int. Conf. on Analogy (ANALOGY-2009),
    pp. 165-174.
    """
    # Per-atom cosine to each input.
    sim_x = jax.vmap(lambda a: cosine_similarity(x, a))(atoms)  # (N,)
    sim_y = jax.vmap(lambda a: cosine_similarity(y, a))(atoms)  # (N,)
    # Joint membership: capped at 0, multiplied. Atoms absent from either
    # input contribute nothing to the result.
    joint = jnp.maximum(sim_x, 0.0) * jnp.maximum(sim_y, 0.0)  # (N,)
    # Bundle of atoms weighted by joint membership.
    return jnp.sum(atoms * joint[:, None], axis=0)


@jax.jit
def transformation_vector(a: jax.Array, b: jax.Array) -> jax.Array:
    r"""Construct a transformation hypervector :math:`T = a^{-1} \star b`.

    The transformation vector encodes "the rule that maps :math:`a` to
    :math:`b`": applying it via :func:`bind_map` recovers :math:`b` from
    :math:`a`, and bundles of transformation vectors over many example
    pairs serve as a learned rule under the Rasmussen-Eliasmith (2011)
    inductive-reasoning recipe and Kanerva's (2010) "Dollar of Mexico"
    analogical-mapping construction.

    For Binary Spatter Codes (where XOR is self-inverse and
    ``inverse_bsc`` is the identity) this collapses to ``bind_bsc(a, b)``;
    for MAP it is ``bind_map(inverse_map(a), b)``.

    Args:
        a: Source hypervector of shape ``(..., d)``.
        b: Target hypervector of shape ``(..., d)``.

    Returns:
        The transformation hypervector ``inverse(a) * b`` (real-valued
        MAP convention; element-wise reciprocal of ``a`` followed by
        element-wise product with ``b``).

    References
    ----------
    Kanerva, P. (2010). What We Mean When We Say "What's the Dollar of
    Mexico?": Prototypes and Mapping in Concept Space.
    Rasmussen, D., Eliasmith, C. (2011). A Neural Model of Rule Generation
    in Inductive Reasoning. Topics in Cognitive Science 3(1): 140-153.
    """
    return bind_map(inverse_map(a), b)


@jax.jit
def cosine_similarity(x: jax.Array, y: jax.Array) -> jax.Array:
    """Compute cosine similarity between real-valued hypervectors.

    Returns the cosine of the angle between two vectors. Random unit vectors
    have similarity ≈ 0.

    Args:
        x: Real-valued hypervector of shape (..., d)
        y: Real-valued hypervector of shape (..., d)

    Returns:
        Similarity score in [-1, 1], where 1 is identical, -1 is opposite,
        and 0 is orthogonal
    """
    x_norm = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + EPS)
    y_norm = y / (jnp.linalg.norm(y, axis=-1, keepdims=True) + EPS)
    return jnp.clip(jnp.sum(x_norm * y_norm, axis=-1), -1.0, 1.0)


@jax.jit
def permute(x: jax.Array, shifts: int = 1) -> jax.Array:
    """Cyclically permute a hypervector to encode sequence information.

    Permutation reorders elements to represent positional or sequential
    information. Cyclic shifts preserve the distribution of values.

    Args:
        x: Hypervector of shape (..., d)
        shifts: Number of positions to shift (default: 1)

    Returns:
        Permuted hypervector of shape (..., d)
    """
    return jnp.roll(x, shifts, axis=-1)


@functools.partial(jax.jit, static_argnames=("return_similarity",))
def cleanup(
    query: jax.Array,
    memory: jax.Array,
    similarity_fn: Callable[[jax.Array, jax.Array], jax.Array] = cosine_similarity,
    return_similarity: bool = False,
) -> Union[jax.Array, tuple[jax.Array, jax.Array]]:
    """Find the most similar vector in memory to the query.

    Cleanup is used to retrieve the closest known hypervector from
    memory, useful for error correction and symbol retrieval after a
    bind / unbind sequence has introduced approximation noise. This is
    the abstract-vector cleanup operation of Kanerva (2009); for the
    spiking-neuron implementation see Stewart, Tang & Eliasmith (2010,
    *Cognitive Systems Research* 12: 84-92), which is out of scope here.
    Resonator networks (Frady et al. 2020) are a related but distinct
    factorisation algorithm built on top of cleanup.

    Args:
        query: Query hypervector of shape (..., d)
        memory: Memory hypervectors of shape (n, d)
        similarity_fn: Function to compute similarity (default: cosine_similarity)
        return_similarity: Whether to return similarity scores (default: False)

    Returns:
        Most similar vector from memory, or (vector, similarity) if return_similarity=True

    References
    ----------
    Kanerva, P. (2009). Hyperdimensional Computing: An Introduction.
    Cognitive Computation 1(2): 139-159.
    """
    similarities = jax.vmap(lambda m: similarity_fn(query, m))(memory)
    best_idx = jnp.argmax(similarities)
    best_vector = memory[best_idx]

    if return_similarity:
        return best_vector, similarities[best_idx]
    return best_vector


# Batch versions for common operations
batch_bind_bsc = jax.vmap(bind_bsc, in_axes=(0, 0))
batch_bind_map = jax.vmap(bind_map, in_axes=(0, 0))
batch_hamming_similarity = jax.vmap(hamming_similarity, in_axes=(0, None))
batch_cosine_similarity = jax.vmap(cosine_similarity, in_axes=(0, None))


@jax.jit
def bind_hrr(x: jax.Array, y: jax.Array) -> jax.Array:
    """Bind two hypervectors using circular convolution for HRR.

    Circular convolution in the spatial domain is equivalent to element-wise
    multiplication in the Fourier domain, making it efficient to compute via
    the FFT. This is the canonical HRR binding of Plate (1995, 1994/2003);
    Jones & Mewhort (2007) is the canonical cognitive-science application
    (the BEAGLE composite holographic lexicon).

    Args:
        x: Real-valued hypervector of shape (..., d)
        y: Real-valued hypervector of shape (..., d)

    Returns:
        Bound hypervector via circular convolution

    References
    ----------
    Plate, T. A. (1995). Holographic Reduced Representations. IEEE
    Transactions on Neural Networks 6(3): 623-641.
    Plate, T. A. (2003). Holographic Reduced Representation: Distributed
    Representation for Cognitive Structures. CSLI Publications.
    Jones, M. N., Mewhort, D. J. K. (2007). Representing Word Meaning
    and Order Information in a Composite Holographic Lexicon.
    Psychological Review 114(1): 1-37.
    """
    x_fft = jnp.fft.fft(x, axis=-1)
    y_fft = jnp.fft.fft(y, axis=-1)
    result_fft = x_fft * y_fft
    return jnp.fft.ifft(result_fft, axis=-1).real


@jax.jit
def inverse_hrr(x: jax.Array) -> jax.Array:
    """Approximate inverse for HRR — Plate's *involution* ``x*``.

    The involution is the vector ``x*`` with ``(x*)_i = x_{(-i) mod d}`` — i.e.
    the first element is preserved and the remaining ``d - 1`` elements are
    reversed. For a length-4 example ``[c_0, c_1, c_2, c_3]`` this returns
    ``[c_0, c_3, c_2, c_1]``, matching Plate (1995, §II.F) verbatim.

    The involution is an *approximate* inverse: ``bind_hrr(x, inverse_hrr(x))``
    is approximately the unit impulse, with the approximation tightening as
    the dimension grows. It differs from the *exact* inverse
    ``F^{-1}(1 / F(x))``, which exists only when no Fourier coefficient of
    ``x`` vanishes; the library uses the involution because it is
    cheap, always defined, and the standard choice in HRR libraries.

    Args:
        x: Real-valued hypervector of shape (..., d)

    Returns:
        The involution ``x*``, shape (..., d).

    References
    ----------
    Plate, T. A. (1995). Holographic Reduced Representations. IEEE
    Transactions on Neural Networks 6(3): 623-641. (See §II.F for the
    involution definition.)
    """
    return jnp.concatenate([x[..., :1], jnp.flip(x[..., 1:], axis=-1)], axis=-1)


# HRR bundle reuses MAP bundle
bundle_hrr = bundle_map


@jax.jit
def bind_cgr(x: jax.Array, y: jax.Array, q: int) -> jax.Array:
    """Bind using modular addition for Cyclic Group Representation.

    Args:
        x: Integer hypervector with values in {0, ..., q-1}, shape (..., d)
        y: Integer hypervector with values in {0, ..., q-1}, shape (..., d)
        q: Size of the cyclic group

    Returns:
        Bound hypervector: (x + y) mod q
    """
    return (x + y) % q


def bundle_cgr(vectors: jax.Array, q: int, axis: int = 0) -> jax.Array:
    """Bundle using component-wise mode for CGR.

    Selects the most frequent value at each dimension.

    Args:
        vectors: Integer hypervectors with values in {0, ..., q-1}
        q: Size of the cyclic group
        axis: Axis along which to bundle (default: 0)

    Returns:
        Bundled hypervector with mode value at each dimension
    """
    one_hot = jax.nn.one_hot(vectors, q)
    counts = jnp.sum(one_hot, axis=axis)
    return jnp.argmax(counts, axis=-1).astype(jnp.int32)


@jax.jit
def inverse_cgr(x: jax.Array, q: int) -> jax.Array:
    """Inverse via modular negation for CGR.

    Args:
        x: Integer hypervector with values in {0, ..., q-1}
        q: Size of the cyclic group

    Returns:
        Inverse: (q - x) mod q
    """
    return (q - x) % q


@jax.jit
def matching_similarity(x: jax.Array, y: jax.Array) -> jax.Array:
    """Fraction of matching elements between integer hypervectors.

    Random vectors with q levels have expected similarity 1/q.

    Args:
        x: Integer hypervector of shape (..., d)
        y: Integer hypervector of shape (..., d)

    Returns:
        Similarity in [0, 1]
    """
    return jnp.mean((x == y).astype(jnp.float32), axis=-1)


# MCR bind reuses CGR bind
bind_mcr = bind_cgr
# MCR inverse reuses CGR inverse
inverse_mcr = inverse_cgr


def bundle_mcr(vectors: jax.Array, q: int, axis: int = 0) -> jax.Array:
    """Bundle using phasor sum and snap-to-grid for MCR.

    Converts indices to complex phasors (q-th roots of unity), sums them,
    and snaps back to the nearest discrete phase level.

    Args:
        vectors: Integer hypervectors with values in {0, ..., q-1}
        q: Number of phase levels
        axis: Axis along which to bundle (default: 0)

    Returns:
        Bundled hypervector with values snapped to nearest phase index
    """
    phases = 2 * jnp.pi * vectors.astype(jnp.float32) / q
    phasors = jnp.exp(1j * phases)
    summed = jnp.sum(phasors, axis=axis)
    result_angles = jnp.angle(summed) % (2 * jnp.pi)
    result_indices = jnp.round(result_angles * q / (2 * jnp.pi)) % q
    return result_indices.astype(jnp.int32)


@jax.jit
def phasor_similarity(x: jax.Array, y: jax.Array, q: int) -> jax.Array:
    """Similarity via phasor inner product for MCR.

    Converts to phasors and computes the real part of the normalized
    inner product. Random vectors have expected similarity ~0.

    Args:
        x: Integer hypervector with values in {0, ..., q-1}
        y: Integer hypervector with values in {0, ..., q-1}
        q: Number of phase levels

    Returns:
        Similarity in [-1, 1]
    """
    phases_x = 2 * jnp.pi * x.astype(jnp.float32) / q
    phases_y = 2 * jnp.pi * y.astype(jnp.float32) / q
    phasors_x = jnp.exp(1j * phases_x)
    phasors_y = jnp.exp(1j * phases_y)
    return jnp.real(jnp.mean(phasors_x * jnp.conj(phasors_y), axis=-1))


@jax.jit
def bind_vtb(x: jax.Array, y: jax.Array) -> jax.Array:
    """Bind using matrix multiplication for VTB.

    Reshapes d-dimensional vectors into sqrt(d) x sqrt(d) matrices
    and multiplies them. Requires d to be a perfect square.

    Args:
        x: Real-valued hypervector of shape (..., d)
        y: Real-valued hypervector of shape (..., d)

    Returns:
        Bound hypervector of shape (..., d)
    """
    d = x.shape[-1]
    n = round(d**0.5)
    X = x.reshape(*x.shape[:-1], n, n)
    Y = y.reshape(*y.shape[:-1], n, n)
    return (X @ Y).reshape(*x.shape[:-1], d)


@jax.jit
def inverse_vtb(x: jax.Array) -> jax.Array:
    """Inverse via matrix pseudoinverse for VTB."""
    d = x.shape[-1]
    n = round(d**0.5)
    X = x.reshape(*x.shape[:-1], n, n)
    return jnp.linalg.pinv(X).reshape(*x.shape[:-1], d)


# VTB bundle reuses MAP bundle
bundle_vtb = bundle_map


# ---------------------------------------------------------------------------
# Similarity: dot product
# ---------------------------------------------------------------------------


@jax.jit
def dot_similarity(x: jax.Array, y: jax.Array) -> jax.Array:
    """Compute dot product similarity between hypervectors.

    Args:
        x: Hypervector of shape (..., d)
        y: Hypervector of shape (..., d)

    Returns:
        Dot product (scalar or batch)
    """
    return jnp.sum(x * y, axis=-1)


# ---------------------------------------------------------------------------
# Negation (bundling inverse)
# ---------------------------------------------------------------------------


@jax.jit
def negative_bsc(x: jax.Array) -> jax.Array:
    """Bundling inverse for BSC (bit flip)."""
    return jnp.logical_not(x)


@jax.jit
def negative_map(x: jax.Array) -> jax.Array:
    """Bundling inverse for MAP (element-wise negation)."""
    return -x


# ---------------------------------------------------------------------------
# Multi-vector operations
# ---------------------------------------------------------------------------


def multibind_map(vectors: jax.Array, axis: int = 0) -> jax.Array:
    """Bind all vectors along an axis via element-wise product (MAP/HRR).

    Generalises :func:`bind_map` to *n* vectors.

    Args:
        vectors: Array with shape (n, ..., d)
        axis: Axis along which to reduce (default: 0)

    Returns:
        Single hypervector equal to vectors[0] * vectors[1] * ...
    """
    return jnp.prod(vectors, axis=axis)


def multibind_bsc(vectors: jax.Array, axis: int = 0) -> jax.Array:
    """Bind all vectors along an axis via cumulative XOR (BSC).

    XOR of n binary vectors: bit is 1 iff an odd number of inputs are 1.

    Args:
        vectors: Boolean array with shape (n, ..., d)
        axis: Axis along which to reduce (default: 0)

    Returns:
        XOR-reduction of all vectors along axis
    """
    counts = jnp.sum(vectors.astype(jnp.int32), axis=axis)
    return (counts % 2) == 1


def cross_product(set_a: jax.Array, set_b: jax.Array, bind_fn: Callable = bind_map) -> jax.Array:
    """Compute the cross product (all pairwise bindings) of two sets.

    Returns an array of shape (n, m, d) where element [i, j] is
    bind(set_a[i], set_b[j]).

    Args:
        set_a: First set of shape (n, d)
        set_b: Second set of shape (m, d)
        bind_fn: Binding function (default: bind_map)
    """
    return jax.vmap(lambda a: jax.vmap(lambda b: bind_fn(a, b))(set_b))(set_a)


# ---------------------------------------------------------------------------
# Composite encodings
# ---------------------------------------------------------------------------


def hash_table(
    keys: jax.Array,
    values: jax.Array,
    bind_fn: Callable = bind_map,
) -> jax.Array:
    """Create a hash-table hypervector by bundling bound (key, value) pairs.

    hash_table = Σ bind(k_i, v_i)

    Args:
        keys: Key hypervectors of shape (n, d)
        values: Value hypervectors of shape (n, d)
        bind_fn: Binding function (default: bind_map)

    Returns:
        Hash-table hypervector of shape (d,)
    """
    pairs = jax.vmap(bind_fn)(keys, values)
    return jnp.sum(pairs, axis=0)


def ngrams(
    vectors: jax.Array,
    n: int = 3,
    bind_fn: Callable = bind_map,
) -> jax.Array:
    """Compute the n-gram representation of a sequence of hypervectors.

    Each n-gram is the binding of n positionally-permuted consecutive vectors,
    then all n-grams are bundled (summed).

    Args:
        vectors: Sequence of shape (m, d)
        n: Size of each n-gram (default: 3)
        bind_fn: Binding function (default: bind_map)

    Returns:
        N-gram hypervector of shape (d,)
    """
    m = vectors.shape[0]
    if m < n:
        raise ValueError(f"Need at least {n} vectors for {n}-grams, got {m}")

    result = jnp.zeros(vectors.shape[-1])
    for start in range(m - n + 1):
        gram = permute(vectors[start], shifts=n - 1)
        for offset in range(1, n):
            gram = bind_fn(gram, permute(vectors[start + offset], shifts=n - 1 - offset))
        result = result + gram
    return result


def bundle_sequence(vectors: jax.Array) -> jax.Array:
    """Encode a sequence by bundling position-permuted vectors.

    sequence = Σ permute(v_i, shifts=m-1-i)

    Preserves order information through positional permutation.

    Args:
        vectors: Sequence of shape (m, d)

    Returns:
        Sequence hypervector of shape (d,)
    """
    m = vectors.shape[0]
    result = jnp.zeros(vectors.shape[-1])
    for i in range(m):
        result = result + permute(vectors[i], shifts=m - 1 - i)
    return result


def bind_sequence(
    vectors: jax.Array,
    bind_fn: Callable = bind_map,
) -> jax.Array:
    """Encode a sequence by binding position-permuted vectors.

    sequence = Π permute(v_i, shifts=m-1-i)

    Binding-based sequences are more noise-resistant than bundle-based
    for short sequences, at the cost of lossy retrieval.

    Args:
        vectors: Sequence of shape (m, d)
        bind_fn: Binding function (default: bind_map)

    Returns:
        Sequence hypervector of shape (d,)
    """
    m = vectors.shape[0]
    result = permute(vectors[0], shifts=m - 1)
    for i in range(1, m):
        result = bind_fn(result, permute(vectors[i], shifts=m - 1 - i))
    return result


def graph_encode(
    edges: jax.Array,
    node_hvs: jax.Array,
    *,
    directed: bool = False,
    bind_fn: Callable = bind_map,
) -> jax.Array:
    """Encode a graph as a single hypervector.

    Each edge (u, v) is encoded as bind(node_hvs[u], permute(node_hvs[v]))
    for directed graphs, or bind(node_hvs[u], node_hvs[v]) for undirected.
    All edge encodings are bundled.

    Args:
        edges: Edge list of shape (num_edges, 2) with node indices
        node_hvs: Node hypervectors of shape (num_nodes, d)
        directed: Whether edges are directed (default: False)
        bind_fn: Binding function (default: bind_map)

    Returns:
        Graph hypervector of shape (d,)
    """

    def encode_edge(edge: jax.Array) -> jax.Array:
        u_hv = node_hvs[edge[0]]
        v_hv = node_hvs[edge[1]]
        if directed:
            return bind_fn(u_hv, permute(v_hv))
        return bind_fn(u_hv, v_hv)

    edge_hvs = jax.vmap(encode_edge)(edges)
    return jnp.sum(edge_hvs, axis=0)


def resonator(
    codebooks: list[jax.Array],
    target: jax.Array,
    *,
    max_iters: int = 100,
    bind_fn: Callable = bind_map,
) -> list[jax.Array]:
    """Resonator network for factorising a composite hypervector.

    Given codebooks C_1 ... C_k and a target = bind(f_1, ..., f_k),
    iteratively estimates each factor f_i.  Uses early stopping when
    estimates converge.

    Args:
        codebooks: List of k codebooks, each of shape (n_i, d)
        target: Target hypervector of shape (d,)
        max_iters: Maximum iterations (default: 100)
        bind_fn: Binding function (default: bind_map)

    Returns:
        List of k estimated factor hypervectors
    """
    k = len(codebooks)
    estimates = [codebooks[i][0] for i in range(k)]

    for _ in range(max_iters):
        new_estimates = []
        for i in range(k):
            other = target
            for j in range(k):
                if j != i:
                    inv = inverse_map(estimates[j])
                    other = bind_fn(other, inv)
            sims = jax.vmap(lambda c: cosine_similarity(other, c))(codebooks[i])
            best = codebooks[i][jnp.argmax(sims)]
            new_estimates.append(best)

        converged = all(bool(jnp.allclose(new_estimates[i], estimates[i])) for i in range(k))
        estimates = new_estimates
        if converged:
            break

    return estimates


# ---------------------------------------------------------------------------
# Additional similarity metrics (inspired by PyBHV)
# ---------------------------------------------------------------------------


@jax.jit
def jaccard_similarity(x: jax.Array, y: jax.Array) -> jax.Array:
    """Jaccard similarity between binary hypervectors.

    |x AND y| / |x OR y|.  Returns 1 for identical vectors, ~0.33 for random.

    Args:
        x: Binary hypervector of shape (..., d)
        y: Binary hypervector of shape (..., d)

    Returns:
        Jaccard index in [0, 1]
    """
    intersection = jnp.sum(jnp.logical_and(x, y).astype(jnp.float32), axis=-1)
    union = jnp.sum(jnp.logical_or(x, y).astype(jnp.float32), axis=-1)
    return intersection / (union + EPS)


@jax.jit
def tversky_similarity(
    x: jax.Array,
    y: jax.Array,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> jax.Array:
    """Tversky similarity index between binary hypervectors.

    Generalises Jaccard (alpha=beta=1) and Dice (alpha=beta=0.5).

    Args:
        x: Binary prototype hypervector of shape (..., d)
        y: Binary variant hypervector of shape (..., d)
        alpha: Weight for x-only elements (default: 1.0)
        beta: Weight for y-only elements (default: 1.0)

    Returns:
        Tversky index in [0, 1]
    """
    x_f = x.astype(jnp.float32)
    y_f = y.astype(jnp.float32)
    intersection = jnp.sum(x_f * y_f, axis=-1)
    x_only = jnp.sum(x_f * (1 - y_f), axis=-1)
    y_only = jnp.sum((1 - x_f) * y_f, axis=-1)
    return intersection / (intersection + alpha * x_only + beta * y_only + EPS)


# ---------------------------------------------------------------------------
# Selection and threshold operations (inspired by PyBHV)
# ---------------------------------------------------------------------------


@jax.jit
def select_bsc(cond: jax.Array, when_true: jax.Array, when_false: jax.Array) -> jax.Array:
    """Element-wise MUX for binary hypervectors.

    Returns when_true where cond is True, when_false otherwise.

    Args:
        cond: Binary mask of shape (..., d)
        when_true: Binary hypervector returned where cond is True
        when_false: Binary hypervector returned where cond is False
    """
    return jnp.where(cond, when_true, when_false)


@jax.jit
def select_map(cond: jax.Array, when_pos: jax.Array, when_neg: jax.Array) -> jax.Array:
    """Element-wise MUX for real-valued hypervectors.

    Selects when_pos where cond > 0, when_neg otherwise.

    Args:
        cond: Real-valued mask of shape (..., d)
        when_pos: Hypervector returned where cond > 0
        when_neg: Hypervector returned where cond <= 0
    """
    return jnp.where(cond > 0, when_pos, when_neg)


def threshold(vectors: jax.Array, t: int) -> jax.Array:
    """Generalised majority: bit is 1 when at least *t* of the input vectors have a 1.

    Equivalent to standard majority when ``t = n // 2 + 1`` (for odd n).

    Args:
        vectors: Binary hypervectors of shape (n, d)
        t: Minimum count for a bit to be set

    Returns:
        Binary hypervector of shape (d,)
    """
    counts = jnp.sum(vectors.astype(jnp.int32), axis=0)
    return counts >= t


def window(vectors: jax.Array, lo: int, hi: int) -> jax.Array:
    """Window vote: bit is 1 when the count of 1s is in [lo, hi].

    Useful for agreement / disagreement filters.

    Args:
        vectors: Binary hypervectors of shape (n, d)
        lo: Minimum count (inclusive)
        hi: Maximum count (inclusive)

    Returns:
        Binary hypervector of shape (d,)
    """
    counts = jnp.sum(vectors.astype(jnp.int32), axis=0)
    return (counts >= lo) & (counts <= hi)


# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------


def flip_fraction(key: jax.Array, x: jax.Array, fraction: float = 0.1) -> jax.Array:
    """Randomly flip a fraction of bits in a binary hypervector.

    Useful for generating noisy variants at a controlled Hamming distance.

    Args:
        key: JAX PRNG key
        x: Binary hypervector of shape (..., d)
        fraction: Fraction of bits to flip, in [0, 1]

    Returns:
        Noisy binary hypervector
    """
    mask = jax.random.bernoulli(key, fraction, shape=x.shape)
    return jnp.logical_xor(x, mask)


def add_noise_map(key: jax.Array, x: jax.Array, noise_level: float = 0.1) -> jax.Array:
    """Add Gaussian noise to a real-valued hypervector and re-normalise.

    Args:
        key: JAX PRNG key
        x: Real-valued hypervector of shape (..., d)
        noise_level: Standard deviation of the noise

    Returns:
        Noisy normalised hypervector
    """
    noisy = x + noise_level * jax.random.normal(key, shape=x.shape)
    norm = jnp.linalg.norm(noisy, axis=-1, keepdims=True)
    return noisy / (norm + EPS)


# ---------------------------------------------------------------------------
# Quantisation
# ---------------------------------------------------------------------------


@jax.jit
def fractional_power(x: jax.Array, p: float) -> jax.Array:
    """Raise a MAP hypervector to a fractional power.

    Computes sign(x) * |x|^p element-wise.  This smoothly interpolates
    between the zero vector (p -> 0) and x itself (p = 1), and can
    extrapolate beyond (p > 1).  Widely used for encoding continuous
    attributes: bind(role, fractional_power(filler, value)) produces
    representations that vary smoothly with *value*.

    Args:
        x: Real-valued hypervector of shape (..., d)
        p: Exponent (typically in [0, 2])

    Returns:
        Hypervector of shape (..., d)
    """
    return jnp.sign(x) * jnp.abs(x) ** p


@jax.jit
def soft_quantize(x: jax.Array) -> jax.Array:
    """Apply tanh for soft bipolar quantisation."""
    return jnp.tanh(x)


@jax.jit
def hard_quantize(x: jax.Array) -> jax.Array:
    """Snap each element to +1 or -1 (sign function, 0 maps to -1)."""
    return jnp.where(x > 0, 1.0, -1.0)


__all__ = [
    # BSC operations
    "bind_bsc",
    "bundle_bsc",
    "inverse_bsc",
    "negative_bsc",
    "hamming_similarity",
    # MAP operations
    "bind_map",
    "bundle_map",
    "inverse_map",
    "negative_map",
    "cosine_similarity",
    "dot_similarity",
    # HRR operations
    "bind_hrr",
    "bundle_hrr",
    "inverse_hrr",
    # CGR operations
    "bind_cgr",
    "bundle_cgr",
    "inverse_cgr",
    "matching_similarity",
    # MCR operations
    "bind_mcr",
    "bundle_mcr",
    "inverse_mcr",
    "phasor_similarity",
    # VTB operations
    "bind_vtb",
    "bundle_vtb",
    "inverse_vtb",
    # Universal operations
    "permute",
    "cleanup",
    # Multi-vector operations
    "multibind_map",
    "multibind_bsc",
    "cross_product",
    # Composite encodings
    "hash_table",
    "ngrams",
    "bundle_sequence",
    "bind_sequence",
    "graph_encode",
    "resonator",
    # Additional similarity metrics
    "jaccard_similarity",
    "tversky_similarity",
    # Selection and threshold
    "select_bsc",
    "select_map",
    "threshold",
    "window",
    # Noise injection
    "flip_fraction",
    "add_noise_map",
    # Power and quantisation
    "fractional_power",
    "soft_quantize",
    "hard_quantize",
    # Batch operations
    "batch_bind_bsc",
    "batch_bind_map",
    "batch_hamming_similarity",
    "batch_cosine_similarity",
]
