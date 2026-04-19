# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Tests for symbolic data structures."""

import jax
import jax.numpy as jnp

from bayes_hdc import functional as F
from bayes_hdc.structures import Graph, HashTable, Multiset, Sequence


class TestMultiset:
    """Test Multiset structure."""

    def test_create_empty(self):
        ms = Multiset.create(100)
        assert ms.dimensions == 100
        assert ms.size == 0
        assert jnp.allclose(ms.value, 0.0)

    def test_add(self):
        ms = Multiset.create(100)
        hv = jax.random.normal(jax.random.PRNGKey(0), (100,))
        ms = ms.add(hv)
        assert ms.size == 1
        assert not jnp.allclose(ms.value, 0.0)

    def test_remove(self):
        ms = Multiset.create(100)
        hv = jax.random.normal(jax.random.PRNGKey(0), (100,))
        ms = ms.add(hv)
        ms = ms.remove(hv)
        assert ms.size == 0
        assert jnp.allclose(ms.value, 0.0, atol=1e-6)

    def test_contains(self):
        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)
        a = jax.random.normal(k1, (1000,))
        b = jax.random.normal(k2, (1000,))
        c = jax.random.normal(k3, (1000,))

        ms = Multiset.create(1000)
        ms = ms.add(a)
        ms = ms.add(b)

        sim_a = ms.contains(a)
        sim_c = ms.contains(c)
        assert sim_a > sim_c

    def test_from_vectors(self):
        vectors = jax.random.normal(jax.random.PRNGKey(0), (5, 100))
        ms = Multiset.from_vectors(vectors)
        assert ms.size == 5
        assert ms.dimensions == 100


class TestHashTable:
    """Test HashTable structure."""

    def test_create_empty(self):
        ht = HashTable.create(100)
        assert ht.dimensions == 100
        assert ht.size == 0

    def test_add_and_get(self):
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)

        k_hv = jax.random.normal(k1, (1000,))
        k_hv = k_hv / jnp.linalg.norm(k_hv)
        v_hv = jax.random.normal(k2, (1000,))

        ht = HashTable.create(1000)
        ht = ht.add(k_hv, v_hv)

        retrieved = ht.get(k_hv)
        retrieved_norm = retrieved / jnp.linalg.norm(retrieved)
        expected_norm = v_hv / jnp.linalg.norm(v_hv)
        sim = F.cosine_similarity(retrieved_norm, expected_norm)
        assert sim > 0.8

    def test_remove(self):
        key = jax.random.PRNGKey(42)
        k_hv = jax.random.normal(key, (100,))
        v_hv = jax.random.normal(jax.random.split(key)[1], (100,))

        ht = HashTable.create(100)
        ht = ht.add(k_hv, v_hv)
        ht = ht.remove(k_hv, v_hv)
        assert ht.size == 0
        assert jnp.allclose(ht.value, 0.0, atol=1e-6)

    def test_from_pairs(self):
        key = jax.random.PRNGKey(42)
        keys = jax.random.normal(key, (3, 200))
        values = jax.random.normal(jax.random.split(key)[1], (3, 200))
        ht = HashTable.from_pairs(keys, values)
        assert ht.size == 3
        assert ht.dimensions == 200


class TestSequence:
    """Test Sequence structure."""

    def test_create_empty(self):
        seq = Sequence.create(100)
        assert seq.dimensions == 100
        assert seq.size == 0

    def test_append(self):
        seq = Sequence.create(100)
        hv = jax.random.normal(jax.random.PRNGKey(0), (100,))
        seq = seq.append(hv)
        assert seq.size == 1

    def test_from_vectors(self):
        vectors = jax.random.normal(jax.random.PRNGKey(42), (5, 100))
        seq = Sequence.from_vectors(vectors)
        assert seq.size == 5
        assert seq.dimensions == 100

    def test_order_preserved(self):
        """Forward and reverse sequences produce different hypervectors."""
        vectors = jax.random.normal(jax.random.PRNGKey(42), (5, 100))
        fwd = Sequence.from_vectors(vectors)
        rev = Sequence.from_vectors(vectors[::-1])
        assert not jnp.allclose(fwd.value, rev.value)


class TestGraph:
    """Test Graph structure."""

    def test_create(self):
        g = Graph.create(100)
        assert g.dimensions == 100
        assert g.directed is False

    def test_add_edge(self):
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        u = jax.random.normal(k1, (100,))
        v = jax.random.normal(k2, (100,))

        g = Graph.create(100)
        g = g.add_edge(u, v)
        assert not jnp.allclose(g.value, 0.0)

    def test_contains_edge(self):
        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)
        u = jax.random.normal(k1, (1000,))
        v = jax.random.normal(k2, (1000,))
        w = jax.random.normal(k3, (1000,))

        g = Graph.create(1000)
        g = g.add_edge(u, v)

        edge_sim = g.contains_edge(u, v)
        non_edge_sim = g.contains_edge(u, w)
        assert edge_sim > non_edge_sim

    def test_directed_graph(self):
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        u = jax.random.normal(k1, (100,))
        v = jax.random.normal(k2, (100,))

        g = Graph.create(100, directed=True)
        g = g.add_edge(u, v)
        assert g.directed is True

    def test_neighbors(self):
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        u = jax.random.normal(k1, (1000,))
        v = jax.random.normal(k2, (1000,))

        g = Graph.create(1000)
        g = g.add_edge(u, v)

        nbrs = g.neighbors(u)
        assert nbrs.shape == (1000,)


class TestSequenceGet:
    """Cover Sequence.get method."""

    def test_get_retrieval(self):
        key = jax.random.PRNGKey(42)
        hvs = jax.random.normal(key, (3, 1000))
        seq = Sequence.from_vectors(hvs)
        retrieved = seq.get(0)
        assert retrieved.shape == (1000,)


class TestGraphDirectedContainsEdge:
    """Cover directed branch of Graph.contains_edge."""

    def test_directed_contains_edge(self):
        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)
        u = jax.random.normal(k1, (1000,))
        v = jax.random.normal(k2, (1000,))
        g = Graph.create(dimensions=1000, directed=True)
        g = g.add_edge(u, v)
        sim = g.contains_edge(u, v)
        assert float(sim) > 0
