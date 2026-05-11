# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Symbolic data structures built on hypervectors.

Provides Multiset, HashTable, Sequence, and Graph structures that use
HDC operations internally for storage and retrieval.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from bayes_hdc import functional as F
from bayes_hdc._compat import register_dataclass


@register_dataclass
@dataclass
class Multiset:
    """Hypervector multiset (bag) structure.

    Supports adding and removing elements, membership testing, and
    creation from a batch of hypervectors.
    """

    value: jax.Array
    dimensions: int = field(metadata=dict(static=True))
    size: int = field(metadata=dict(static=True), default=0)

    @staticmethod
    def create(dimensions: int) -> "Multiset":
        return Multiset(
            value=jnp.zeros(dimensions),
            dimensions=dimensions,
            size=0,
        )

    def add(self, hv: jax.Array) -> "Multiset":
        """Add a hypervector to the multiset."""
        return Multiset(
            value=self.value + hv,
            dimensions=self.dimensions,
            size=self.size + 1,
        )

    def remove(self, hv: jax.Array) -> "Multiset":
        """Remove a hypervector from the multiset."""
        return Multiset(
            value=self.value - hv,
            dimensions=self.dimensions,
            size=max(self.size - 1, 0),
        )

    def contains(self, hv: jax.Array) -> jax.Array:
        """Return cosine similarity of *hv* against the multiset."""
        return F.cosine_similarity(hv, self.value)

    @staticmethod
    def from_vectors(vectors: jax.Array) -> "Multiset":
        """Create a multiset from a batch of hypervectors of shape (n, d)."""
        return Multiset(
            value=jnp.sum(vectors, axis=0),
            dimensions=vectors.shape[-1],
            size=vectors.shape[0],
        )


@register_dataclass
@dataclass
class HashTable:
    """Hypervector hash-table (key-value associative memory).

    Stores key-value pairs via binding (each pair is ``bind(key, value)``)
    and retrieves approximate values by unbinding the accumulated bundle
    with a query key. This is the canonical role-filler-bundle construction
    of Kanerva (2009) and the substrate of the "Dollar of Mexico" analogy
    pattern (Kanerva 2010).

    References:
    Kanerva, P. (2009). Hyperdimensional Computing: An Introduction.
    Cognitive Computation 1(2): 139-159.
    Kanerva, P. (2010). What We Mean When We Say "What's the Dollar of
    Mexico?": Prototypes and Mapping in Concept Space. AAAI Tech. Rep.
    FS-10-08, pp. 2-6.
    """

    value: jax.Array
    dimensions: int = field(metadata=dict(static=True))
    size: int = field(metadata=dict(static=True), default=0)

    @staticmethod
    def create(dimensions: int) -> "HashTable":
        return HashTable(
            value=jnp.zeros(dimensions),
            dimensions=dimensions,
            size=0,
        )

    def add(self, key: jax.Array, val: jax.Array) -> "HashTable":
        """Store a (key, value) pair."""
        pair = F.bind_map(key, val)
        return HashTable(
            value=self.value + pair,
            dimensions=self.dimensions,
            size=self.size + 1,
        )

    def remove(self, key: jax.Array, val: jax.Array) -> "HashTable":
        """Remove a (key, value) pair."""
        pair = F.bind_map(key, val)
        return HashTable(
            value=self.value - pair,
            dimensions=self.dimensions,
            size=max(self.size - 1, 0),
        )

    @jax.jit
    def get(self, key: jax.Array) -> jax.Array:
        """Retrieve the approximate value for *key*."""
        return F.bind_map(self.value, F.inverse_map(key))

    @staticmethod
    def from_pairs(keys: jax.Array, values: jax.Array) -> "HashTable":
        """Create from arrays of keys and values, each of shape (n, d)."""
        hv = F.hash_table(keys, values)
        return HashTable(
            value=hv,
            dimensions=keys.shape[-1],
            size=keys.shape[0],
        )


@register_dataclass
@dataclass
class Sequence:
    """Hypervector sequence structure using bundle-based encoding.

    Each element is permuted according to its position before bundling,
    preserving order information. The permute-then-bundle construction
    for sequences was introduced contemporaneously by Sahlgren et al.
    (2008) and Kanerva (2009).

    References:
    Sahlgren, M., Holst, A., Kanerva, P. (2008). Permutations as a
    Means to Encode Order in Word Space. Proc. 30th Annual Conference
    of the Cognitive Science Society, pp. 1300-1305.
    Kanerva, P. (2009). Hyperdimensional Computing: An Introduction.
    Cognitive Computation 1(2): 139-159.
    """

    value: jax.Array
    dimensions: int = field(metadata=dict(static=True))
    size: int = field(metadata=dict(static=True), default=0)

    @staticmethod
    def create(dimensions: int) -> "Sequence":
        return Sequence(
            value=jnp.zeros(dimensions),
            dimensions=dimensions,
            size=0,
        )

    def append(self, hv: jax.Array) -> "Sequence":
        """Append a hypervector to the right of the sequence."""
        rotated = F.permute(self.value, shifts=1)
        return Sequence(
            value=rotated + hv,
            dimensions=self.dimensions,
            size=self.size + 1,
        )

    def get(self, index: int) -> jax.Array:
        """Approximate retrieval of the element at *index*."""
        return F.permute(self.value, shifts=-(self.size - index - 1))

    @staticmethod
    def from_vectors(vectors: jax.Array) -> "Sequence":
        """Create a sequence from a batch of shape (m, d)."""
        return Sequence(
            value=F.bundle_sequence(vectors),
            dimensions=vectors.shape[-1],
            size=vectors.shape[0],
        )


@register_dataclass
@dataclass
class HierarchicalSequence:
    """Two-level chunked sequence for long-horizon HDC encoding.

    A flat :class:`Sequence` bundles all ``n`` items into one
    ``d``-vector, so per-item retrieval SNR degrades as
    :math:`O(1/\\sqrt{n})`. Even with a fixed item codebook for
    cleanup, capacity saturates well before the dimension limit:
    for example at ``d = 4096`` and ``n ≳ 200``, flat retrieval
    accuracy collapses below random.

    This class implements a two-level chunked construction
    inspired by Frady, Kleyko & Sommer (2018, *A Theory of Sequence
    Indexing and Working Memory in Recurrent Neural Networks*, Neural
    Computation 30(6): 1449–1513) and the hierarchical-binding
    discussion in Plate (2003) §10. The capacity gain has *two*
    sources, not one:

    1. **Structural** — partition the input into chunks of size
       ``C``; encode each chunk as a flat permute-bundle ``h_k``;
       encode the chunks as a higher-level flat permute-bundle
       ``s = Σ_k P^{K-1-k} h_k``.
    2. **Cleanup at the chunk level** — at retrieval time, after the
       outer un-permute returns a noisy chunk hypervector, project
       it back onto the *clean* chunk codebook
       ``{h_0, ..., h_{K-1}}`` (which we cache at construction). This
       step removes the cross-chunk noise *before* the inner
       un-permute, so the noise that survives to the item level is
       only the within-chunk noise from ``C - 1`` items rather than
       from all ``n - 1`` items.

    The chunk codebook is computed once at construction and stored
    on the dataclass; it is the algebraic prerequisite for the
    capacity gain. Without the intermediate cleanup, the noise from
    both layers sums to the same magnitude as the flat case and
    there is no improvement — a subtle point that is easy to miss
    when reading only the structural definition.

    The math (writing ``P`` for cyclic shift, ``v[i]`` for input
    items, ``i ∈ [0, n)``, ``C`` for chunk size, ``K = ⌈n / C⌉``):

    .. math::
        h_k = \\sum_{j=0}^{C-1} P^{C-1-j} \\cdot v[kC + j]
              \\quad \\text{(flat sequence within chunk } k\\text{)}

    .. math::
        s = \\sum_{k=0}^{K-1} P^{K-1-k} \\cdot h_k
              \\quad \\text{(flat sequence over chunks)}

    Retrieval at position ``i`` with ``chunk_id = i // C`` and
    ``pos = i % C``:

    .. math::
        \\tilde h = P^{-(K-1-chunk\\_id)} \\cdot s
              \\quad \\text{(noisy chunk_hv)}

    .. math::
        h^{\\star} = \\arg\\max_{h \\in \\{h_0,...,h_{K-1}\\}}
                      \\langle \\tilde h, h \\rangle
              \\quad \\text{(chunk-level cleanup)}

    .. math::
        \\hat v = P^{-(C-1-pos)} \\cdot h^{\\star}
              \\quad \\text{(item with noise from } C-1 \\text{ items only)}

    The caller is expected to clean ``\\hat v`` against the item
    codebook for symbolic recovery; the chunk-level cleanup is
    handled automatically by :meth:`get`.

    Attributes:
        value: Encoded hypervector ``s`` of shape ``(d,)``.
        chunk_codebook: Stacked clean chunk hypervectors of shape
            ``(K, d)``, used for intermediate cleanup. Computed at
            construction and not user-modifiable.
        n_items: Total number of items ``n``.
        chunk_size: Items per chunk ``C``.
        dimensions: Hypervector dimension ``d``.
    """

    value: jax.Array
    chunk_codebook: jax.Array  # (n_chunks, d)
    n_items: int = field(metadata=dict(static=True))
    chunk_size: int = field(metadata=dict(static=True))
    dimensions: int = field(metadata=dict(static=True))

    @staticmethod
    def from_vectors(vectors: jax.Array, chunk_size: int = 16) -> "HierarchicalSequence":
        """Encode a batch of items as a two-level chunked sequence.

        Args:
            vectors: Items of shape ``(n, d)``. The clean chunk
                hypervectors derived from this input are cached on
                the returned object as ``chunk_codebook`` and used by
                :meth:`get` for intermediate cleanup.
            chunk_size: Number of items per chunk. ``√n`` is the SNR-
                optimal choice; ``16`` is a reasonable default for
                ``n`` in the 64–256 range typical of trajectory
                encoding.

        Returns:
            A :class:`HierarchicalSequence` ready for ``get(i)``
            retrieval. The trailing chunk is zero-padded to
            ``chunk_size``; ``get(i)`` is restricted to
            ``i ∈ [0, n_items)``.
        """
        n, d = vectors.shape[0], vectors.shape[-1]
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if n == 0:
            return HierarchicalSequence(
                value=jnp.zeros(d),
                chunk_codebook=jnp.zeros((0, d)),
                n_items=0,
                chunk_size=chunk_size,
                dimensions=d,
            )

        n_chunks = (n + chunk_size - 1) // chunk_size
        pad = n_chunks * chunk_size - n
        if pad > 0:
            vectors = jnp.concatenate([vectors, jnp.zeros((pad, d))], axis=0)

        # (n_chunks, chunk_size, d) — group items by chunk.
        chunked = vectors.reshape(n_chunks, chunk_size, d)
        # Per-chunk flat permute-bundle.
        chunk_hvs = jax.vmap(F.bundle_sequence)(chunked)  # (n_chunks, d)
        # Outer-level flat permute-bundle over the chunks.
        value = F.bundle_sequence(chunk_hvs)
        return HierarchicalSequence(
            value=value,
            chunk_codebook=chunk_hvs,
            n_items=n,
            chunk_size=chunk_size,
            dimensions=d,
        )

    def get(self, index: int) -> jax.Array:
        """Retrieve the (noisy) item at position ``index``.

        Performs the outer un-permute, an intermediate cleanup
        against ``chunk_codebook``, the inner un-permute, and
        returns the resulting (still-noisy) item hypervector. The
        caller is expected to clean the result against the item
        codebook (e.g. via :func:`bayes_hdc.functional.cleanup`) for
        symbolic recovery.
        """
        if index < 0 or index >= self.n_items:
            raise IndexError(
                f"index {index} out of range for HierarchicalSequence of size {self.n_items}"
            )
        chunk_id = index // self.chunk_size
        pos = index % self.chunk_size
        n_chunks = (self.n_items + self.chunk_size - 1) // self.chunk_size

        # Outer un-permute → noisy chunk hypervector.
        chunk_noisy = F.permute(self.value, shifts=-(n_chunks - 1 - chunk_id))

        # Chunk-level cleanup: project onto the clean chunk codebook.
        # This step is what gives the hierarchical construction its
        # capacity advantage over the flat Sequence.
        sims = self.chunk_codebook @ chunk_noisy
        chunk_clean = self.chunk_codebook[jnp.argmax(sims)]

        # Inner un-permute → item with noise from C-1 items only.
        return F.permute(chunk_clean, shifts=-(self.chunk_size - 1 - pos))


@register_dataclass
@dataclass
class Graph:
    """Hypervector-based graph structure.

    Edges are encoded as bound node pairs and bundled into a single
    hypervector. Supports directed and undirected graphs.
    """

    value: jax.Array
    dimensions: int = field(metadata=dict(static=True))
    directed: bool = field(metadata=dict(static=True), default=False)

    @staticmethod
    def create(dimensions: int, *, directed: bool = False) -> "Graph":
        return Graph(
            value=jnp.zeros(dimensions),
            dimensions=dimensions,
            directed=directed,
        )

    def add_edge(self, u_hv: jax.Array, v_hv: jax.Array) -> "Graph":
        """Add an edge between two node hypervectors."""
        if self.directed:
            edge = F.bind_map(u_hv, F.permute(v_hv))
        else:
            edge = F.bind_map(u_hv, v_hv)
        return Graph(
            value=self.value + edge,
            dimensions=self.dimensions,
            directed=self.directed,
        )

    def neighbors(self, node_hv: jax.Array) -> jax.Array:
        """Return the approximate neighbor multiset of a node."""
        return F.bind_map(self.value, F.inverse_map(node_hv))

    def contains_edge(self, u_hv: jax.Array, v_hv: jax.Array) -> jax.Array:
        """Return dimension-normalised dot similarity of edge (u, v).

        Returns ``dot(edge_hv, self.value) / dimensions`` rather than
        cosine similarity. This convention differs from
        :meth:`Multiset.contains` (cosine) — chosen here because the
        graph's stored ``value`` is an *unnormalised* sum of edge
        hypervectors (its norm grows with the number of edges) and
        cosine would collapse to ~0 for dense graphs. The dot/d
        scale preserves relative magnitudes between edges-in-graph
        and not-in-graph. For thresholding, compare against a
        baseline computed on a random edge with the same scaling.
        """
        if self.directed:
            edge = F.bind_map(u_hv, F.permute(v_hv))
        else:
            edge = F.bind_map(u_hv, v_hv)
        return F.dot_similarity(edge, self.value) / self.dimensions


__all__ = [
    "Multiset",
    "HashTable",
    "Sequence",
    "HierarchicalSequence",
    "Graph",
]
