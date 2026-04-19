# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Symbolic data structures built on hypervectors.

Provides Multiset, HashTable, Sequence, and Graph structures that use
HDC operations internally for storage and retrieval.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from jax_hdc import functional as F
from jax_hdc._compat import register_dataclass


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

    Stores key-value pairs via binding and retrieves approximate values
    by unbinding with a query key.
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
    preserving order information.
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
        """Return dot similarity of edge (u, v) against the graph."""
        if self.directed:
            edge = F.bind_map(u_hv, F.permute(v_hv))
        else:
            edge = F.bind_map(u_hv, v_hv)
        return F.dot_similarity(edge, self.value) / self.dimensions


__all__ = [
    "Multiset",
    "HashTable",
    "Sequence",
    "Graph",
]
