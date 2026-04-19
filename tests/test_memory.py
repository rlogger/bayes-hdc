# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Tests for memory modules."""

import jax
import jax.numpy as jnp
import pytest

from bayes_hdc.memory import AttentionMemory, HopfieldMemory, SparseDistributedMemory


class TestSparseDistributedMemory:
    """Tests for SparseDistributedMemory."""

    def test_create_with_key(self):
        """Test SDM creation with explicit key."""
        key = jax.random.PRNGKey(42)
        sdm = SparseDistributedMemory.create(num_locations=50, dimensions=100, radius=0.2, key=key)
        assert sdm.locations.shape == (50, 100)
        assert sdm.contents.shape == (50, 100)
        assert sdm.dimensions == 100
        assert sdm.radius == 0.2

    def test_create_without_key(self):
        """Test SDM creation with default key."""
        sdm = SparseDistributedMemory.create(num_locations=20, dimensions=50)
        assert sdm.locations.shape == (20, 50)

    def test_write_and_read(self):
        """Test that read recovers a vector similar to what was written."""
        key = jax.random.PRNGKey(42)
        sdm = SparseDistributedMemory.create(num_locations=100, dimensions=100, radius=0.5, key=key)

        addr = sdm.locations[0]
        val = jax.random.normal(jax.random.split(key)[1], (100,))

        sdm = sdm.write(addr, val)
        result = sdm.read(addr)

        assert result.shape == (100,)
        assert jnp.isfinite(result).all()
        from bayes_hdc.functional import cosine_similarity

        sim = cosine_similarity(result, val)
        assert float(sim) > 0.5

    def test_read_normalized_output(self):
        """Test that read returns normalized vector."""
        key = jax.random.PRNGKey(42)
        sdm = SparseDistributedMemory.create(num_locations=50, dimensions=50, radius=0.5, key=key)
        addr = sdm.locations[0]
        val = jax.random.normal(key, (50,))
        sdm = sdm.write(addr, val)
        result = sdm.read(addr)
        norm = jnp.linalg.norm(result)
        assert jnp.allclose(norm, 1.0, atol=1e-5)


class TestHopfieldMemory:
    """Tests for HopfieldMemory."""

    def test_create_default_beta(self):
        """Test Hopfield creation with default beta."""
        hop = HopfieldMemory.create(dimensions=100)
        assert hop.dimensions == 100
        assert hop.beta == 1.0
        assert hop.patterns.shape == (0, 100)

    def test_create_custom_beta(self):
        """Test Hopfield creation with custom beta."""
        hop = HopfieldMemory.create(dimensions=100, beta=2.0)
        assert hop.beta == 2.0

    def test_add_and_retrieve(self):
        """Test that retrieval returns the stored pattern on a noisy query."""
        hop = HopfieldMemory.create(dimensions=100, beta=10.0)
        p = jax.random.normal(jax.random.PRNGKey(42), (100,))
        hop = hop.add(p)

        noisy = p + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (100,))
        result = hop.retrieve(noisy)

        assert result.shape == (100,)
        assert jnp.isfinite(result).all()
        from bayes_hdc.functional import cosine_similarity

        sim = cosine_similarity(result, p / jnp.linalg.norm(p))
        assert float(sim) > 0.9

    def test_retrieve_empty_memory(self):
        """Test retrieve on empty memory returns zeros."""
        hop = HopfieldMemory.create(dimensions=100)
        query = jax.random.normal(jax.random.PRNGKey(42), (100,))
        result = hop.retrieve(query)
        assert result.shape == (100,)
        assert jnp.allclose(result, 0.0)

    def test_add_multiple_patterns(self):
        """Test adding multiple patterns."""
        hop = HopfieldMemory.create(dimensions=50)
        for i in range(3):
            p = jax.random.normal(jax.random.PRNGKey(i), (50,))
            hop = hop.add(p)
        assert hop.patterns.shape == (3, 50)


class TestAttentionMemory:
    """Tests for AttentionMemory."""

    def test_create_default(self):
        mem = AttentionMemory.create(dimensions=100)
        assert mem.dimensions == 100
        assert mem.temperature == 1.0
        assert mem.num_heads == 1
        assert mem.keys.shape == (0, 100)
        assert mem.values.shape == (0, 100)

    def test_create_custom(self):
        mem = AttentionMemory.create(dimensions=100, temperature=0.5, num_heads=4)
        assert mem.temperature == 0.5
        assert mem.num_heads == 4

    def test_create_invalid_heads(self):
        with pytest.raises(ValueError):
            AttentionMemory.create(dimensions=100, num_heads=3)

    def test_write_and_retrieve(self):
        key = jax.random.PRNGKey(42)
        mem = AttentionMemory.create(dimensions=100)

        k = jax.random.normal(key, (100,))
        v = jax.random.normal(jax.random.split(key)[1], (100,))
        mem = mem.write(k, v)

        result = mem.retrieve(k)
        assert result.shape == (100,)
        assert jnp.isfinite(result).all()

    def test_retrieve_empty(self):
        mem = AttentionMemory.create(dimensions=100)
        query = jax.random.normal(jax.random.PRNGKey(0), (100,))
        result = mem.retrieve(query)
        assert result.shape == (100,)
        assert jnp.allclose(result, 0.0)

    def test_write_batch(self):
        key = jax.random.PRNGKey(42)
        mem = AttentionMemory.create(dimensions=50)

        keys = jax.random.normal(key, (5, 50))
        values = jax.random.normal(jax.random.split(key)[1], (5, 50))
        mem = mem.write_batch(keys, values)

        assert mem.keys.shape == (5, 50)
        assert mem.values.shape == (5, 50)

    def test_retrieve_nearest(self):
        """Test that retrieval favors the most similar key."""
        key = jax.random.PRNGKey(42)
        mem = AttentionMemory.create(dimensions=100, temperature=0.01)

        k1, k2, k3 = jax.random.split(key, 3)
        keys = jax.random.normal(k1, (3, 100))
        values = jax.random.normal(k2, (3, 100))

        for i in range(3):
            mem = mem.write(keys[i], values[i])

        result = mem.retrieve(keys[0])
        from bayes_hdc.functional import cosine_similarity

        sim = cosine_similarity(result, values[0])
        assert sim > 0.5

    def test_multi_head(self):
        key = jax.random.PRNGKey(42)
        mem = AttentionMemory.create(dimensions=100, num_heads=4)

        k = jax.random.normal(key, (100,))
        v = jax.random.normal(jax.random.split(key)[1], (100,))
        mem = mem.write(k, v)

        result = mem.retrieve(k)
        assert result.shape == (100,)
        assert jnp.isfinite(result).all()

    def test_retrieve_with_weights(self):
        key = jax.random.PRNGKey(42)
        mem = AttentionMemory.create(dimensions=50)

        k = jax.random.normal(key, (50,))
        v = jax.random.normal(jax.random.split(key)[1], (50,))
        mem = mem.write(k, v)

        result, weights = mem.retrieve_with_weights(k)
        assert result.shape == (50,)
        assert weights.shape == (1,)
        assert jnp.allclose(weights, 1.0)

    def test_retrieve_with_weights_empty(self):
        """Cover empty-memory branch in retrieve_with_weights."""
        mem = AttentionMemory.create(dimensions=50)
        query = jax.random.normal(jax.random.PRNGKey(0), (50,))
        result, weights = mem.retrieve_with_weights(query)
        assert result.shape == (50,)
        assert jnp.allclose(result, 0.0)
