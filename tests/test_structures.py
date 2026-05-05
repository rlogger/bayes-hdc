# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Tests for symbolic data structures."""

import jax
import jax.numpy as jnp

from bayes_hdc import functional as F
from bayes_hdc.structures import (
    Graph,
    HashTable,
    HierarchicalSequence,
    Multiset,
    Sequence,
)


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


class TestHierarchicalSequence:
    """Two-level chunked sequence — capacity and retrieval."""

    @staticmethod
    def _make_codebook(key: jax.Array, n: int, d: int) -> jax.Array:
        """L2-normalised Gaussian hypervectors."""
        v = jax.random.normal(key, (n, d))
        return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)

    @staticmethod
    def _retrieval_accuracy(
        sequence_get_fn,
        n_items: int,
        codebook: jax.Array,
        item_ids: jax.Array,
    ) -> float:
        """Fraction of positions where cleanup recovers the right codebook
        entry."""
        n_correct = 0
        for i in range(n_items):
            recovered = sequence_get_fn(int(i))
            similarities = codebook @ recovered
            predicted_id = int(jnp.argmax(similarities))
            if predicted_id == int(item_ids[i]):
                n_correct += 1
        return n_correct / n_items

    def test_create_empty(self):
        hs = HierarchicalSequence.from_vectors(jnp.zeros((0, 64)), chunk_size=4)
        assert hs.n_items == 0
        assert hs.chunk_size == 4
        assert hs.dimensions == 64

    def test_get_out_of_range_raises(self):
        v = jax.random.normal(jax.random.PRNGKey(0), (4, 64))
        hs = HierarchicalSequence.from_vectors(v, chunk_size=4)
        try:
            hs.get(5)
            raise AssertionError("Expected IndexError on out-of-range get()")
        except IndexError:
            pass

    def test_invalid_chunk_size_raises(self):
        try:
            HierarchicalSequence.from_vectors(jnp.zeros((4, 64)), chunk_size=0)
            raise AssertionError("Expected ValueError on chunk_size=0")
        except ValueError:
            pass

    def test_single_chunk_matches_flat_sequence(self):
        """When n ≤ chunk_size, the hierarchical encoding is the flat
        encoding of a single chunk wrapped in a one-element outer
        sequence; retrieval should still recover items reliably at
        small n."""
        key = jax.random.PRNGKey(0)
        d = 1024
        codebook = self._make_codebook(key, n=64, d=d)
        item_ids = jnp.arange(8)
        items = codebook[item_ids]
        hs = HierarchicalSequence.from_vectors(items, chunk_size=8)
        assert hs.n_items == 8

        acc = self._retrieval_accuracy(hs.get, 8, codebook, item_ids)
        assert acc >= 0.75, f"single-chunk accuracy = {acc:.2f}"

    def test_recovers_long_sequence_better_than_flat(self):
        """At T = 400, d = 4096, the flat Sequence's per-item cleanup
        signal-to-noise approaches the codebook-collision noise floor
        and retrieval degrades substantially. The hierarchical
        encoding's chunk-level cleanup keeps the per-chunk SNR high,
        so retrieval recovers a clearly larger fraction of items."""
        key = jax.random.PRNGKey(2026)
        d = 4096
        T = 400
        chunk_size = 16
        n_codebook = 256

        cb_key, item_key = jax.random.split(key)
        codebook = self._make_codebook(cb_key, n=n_codebook, d=d)
        item_ids = jax.random.randint(item_key, (T,), 0, n_codebook)
        items = codebook[item_ids]

        # Flat sequence.
        flat = Sequence.from_vectors(items)
        flat_acc = self._retrieval_accuracy(flat.get, T, codebook, item_ids)

        # Hierarchical with intermediate cleanup.
        hs = HierarchicalSequence.from_vectors(items, chunk_size=chunk_size)
        hs_acc = self._retrieval_accuracy(hs.get, T, codebook, item_ids)

        # Hierarchical should beat flat by a clear margin at T=400.
        # Empirically (d=4096, chunk=16, codebook=256, T=400):
        # flat is in the 0.4–0.7 range, hierarchical in 0.85–0.97.
        # Use a conservative differential threshold to absorb seed jitter.
        assert hs_acc > flat_acc + 0.1, (
            f"hierarchical did not beat flat by 0.1: hs={hs_acc:.2f}, flat={flat_acc:.2f}"
        )

    def test_pads_uneven_chunks(self):
        """A sequence whose length is not a multiple of chunk_size is
        zero-padded silently. The valid-index range stays equal to
        n_items."""
        v = jax.random.normal(jax.random.PRNGKey(0), (10, 256))
        hs = HierarchicalSequence.from_vectors(v, chunk_size=4)
        assert hs.n_items == 10  # not 12 (the padded total)
        # All ten user-visible indices must be retrievable.
        for i in range(10):
            recovered = hs.get(i)
            assert recovered.shape == (256,)
