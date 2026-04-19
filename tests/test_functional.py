# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Tests for functional operations."""

import jax
import jax.numpy as jnp
import pytest

from jax_hdc import functional as F


class TestBSCOperations:
    """Test Binary Spatter Code operations."""

    def test_bind_bsc_commutativity(self):
        k1, k2 = jax.random.split(jax.random.PRNGKey(42))
        x = jax.random.bernoulli(k1, 0.5, shape=(10000,))
        y = jax.random.bernoulli(k2, 0.5, shape=(10000,))
        assert jnp.array_equal(F.bind_bsc(x, y), F.bind_bsc(y, x))

    def test_bind_bsc_self_inverse(self):
        k1, k2 = jax.random.split(jax.random.PRNGKey(42))
        x = jax.random.bernoulli(k1, 0.5, shape=(10000,))
        y = jax.random.bernoulli(k2, 0.5, shape=(10000,))
        assert jnp.array_equal(F.bind_bsc(F.bind_bsc(x, y), y), x)

    def test_bind_bsc_dissimilarity(self):
        k1, k2 = jax.random.split(jax.random.PRNGKey(42))
        x = jax.random.bernoulli(k1, 0.5, shape=(10000,))
        y = jax.random.bernoulli(k2, 0.5, shape=(10000,))
        bound = F.bind_bsc(x, y)
        sim_x = F.hamming_similarity(bound, x)
        sim_y = F.hamming_similarity(bound, y)
        assert 0.45 < float(sim_x) < 0.55
        assert 0.45 < float(sim_y) < 0.55

    def test_bundle_bsc_similarity(self):
        keys = jax.random.split(jax.random.PRNGKey(42), 5)
        vectors = jnp.stack([jax.random.bernoulli(k, 0.5, (10000,)) for k in keys])
        bundled = F.bundle_bsc(vectors, axis=0)
        for v in vectors:
            sim = F.hamming_similarity(bundled, v)
            assert float(sim) > 0.55

    def test_hamming_similarity_range(self):
        k1, k2 = jax.random.split(jax.random.PRNGKey(42))
        x = jax.random.bernoulli(k1, 0.5, shape=(10000,))
        y = jax.random.bernoulli(k2, 0.5, shape=(10000,))
        sim = F.hamming_similarity(x, y)
        assert 0.45 < float(sim) < 0.55

    def test_hamming_similarity_identity(self):
        x = jax.random.bernoulli(jax.random.PRNGKey(42), 0.5, shape=(10000,))
        assert jnp.allclose(F.hamming_similarity(x, x), 1.0)


class TestMAPOperations:
    """Test Multiply-Add-Permute operations."""

    def test_bind_map_commutativity(self):
        k1, k2 = jax.random.split(jax.random.PRNGKey(42))
        x = jax.random.normal(k1, shape=(10000,))
        y = jax.random.normal(k2, shape=(10000,))
        assert jnp.allclose(F.bind_map(x, y), F.bind_map(y, x))

    def test_inverse_map_handles_zero(self):
        x = jnp.array([1.0, 0.0, -1.0, 2.0, 0.0])
        inv = F.inverse_map(x)
        assert jnp.isfinite(inv).all()
        assert inv[1] == 0.0
        assert inv[4] == 0.0
        assert jnp.allclose(inv[0], 1.0)
        assert jnp.allclose(inv[2], -1.0)
        assert jnp.allclose(inv[3], 0.5)

    def test_bind_map_inverse(self):
        k1, k2 = jax.random.split(jax.random.PRNGKey(42))
        x = jax.random.normal(k1, shape=(10000,))
        y = jax.random.normal(k2, shape=(10000,))
        bound = F.bind_map(x, y)
        unbound = F.bind_map(bound, F.inverse_map(y))
        sim = F.cosine_similarity(x, unbound)
        assert float(sim) > 0.95

    def test_bind_map_dissimilarity(self):
        k1, k2 = jax.random.split(jax.random.PRNGKey(42))
        x = jax.random.normal(k1, shape=(10000,))
        y = jax.random.normal(k2, shape=(10000,))
        bound = F.bind_map(x, y)
        assert abs(float(F.cosine_similarity(bound, x))) < 0.1
        assert abs(float(F.cosine_similarity(bound, y))) < 0.1

    def test_bundle_map_normalization(self):
        vectors = jax.random.normal(jax.random.PRNGKey(42), shape=(10, 10000))
        bundled = F.bundle_map(vectors, axis=0)
        assert jnp.allclose(jnp.linalg.norm(bundled), 1.0, atol=1e-6)

    def test_bundle_map_similarity(self):
        keys = jax.random.split(jax.random.PRNGKey(42), 5)
        vectors = jnp.stack([jax.random.normal(k, (10000,)) for k in keys])
        bundled = F.bundle_map(vectors, axis=0)
        for v in vectors:
            sim = F.cosine_similarity(bundled, v)
            assert float(sim) > 0.3

    def test_cosine_similarity_range(self):
        k1, k2 = jax.random.split(jax.random.PRNGKey(42))
        x = jax.random.normal(k1, shape=(10000,))
        y = jax.random.normal(k2, shape=(10000,))
        sim = F.cosine_similarity(x, y)
        assert -1 <= float(sim) <= 1
        assert abs(float(sim)) < 0.1

    def test_cosine_similarity_identity(self):
        x = jax.random.normal(jax.random.PRNGKey(42), shape=(10000,))
        assert jnp.allclose(F.cosine_similarity(x, x), 1.0, atol=1e-6)

    def test_cosine_similarity_orthogonal(self):
        x = jnp.zeros(10000).at[0].set(1.0)
        y = jnp.zeros(10000).at[1].set(1.0)
        assert jnp.allclose(F.cosine_similarity(x, y), 0.0, atol=1e-6)


class TestUniversalOperations:
    """Test operations that work with all VSA models."""

    def test_permute_invertibility(self):
        """Test that permutation is invertible."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, shape=(10000,))

        # Permute and unpermute
        permuted = F.permute(x, shifts=5)
        unpermuted = F.permute(permuted, shifts=-5)

        assert jnp.allclose(unpermuted, x)

    def test_permute_cyclic(self):
        """Test that permutation is cyclic."""
        x = jnp.arange(10)

        # Full cycle should return to original
        result = x
        for _ in range(10):
            result = F.permute(result, shifts=1)

        assert jnp.allclose(result, x)

    def test_cleanup_retrieval(self):
        """Test cleanup finds most similar vector."""
        key = jax.random.PRNGKey(42)

        # Create memory with 3 vectors
        memory = jax.random.normal(key, shape=(3, 10000))
        memory = memory / jnp.linalg.norm(memory, axis=-1, keepdims=True)

        # Query is a noisy version of first vector
        query = memory[0] + 0.1 * jax.random.normal(key, shape=(10000,))
        query = query / jnp.linalg.norm(query)

        # Cleanup should return first vector
        result = F.cleanup(query, memory)

        assert jnp.allclose(result, memory[0])

    def test_cleanup_return_similarity(self):
        """Test cleanup with return_similarity=True."""
        key = jax.random.PRNGKey(42)

        memory = jax.random.normal(key, shape=(3, 100))
        memory = memory / jnp.linalg.norm(memory, axis=-1, keepdims=True)

        query = memory[1].copy()

        vector, similarity = F.cleanup(query, memory, return_similarity=True)

        assert jnp.allclose(vector, memory[1])
        assert jnp.allclose(similarity, 1.0, atol=1e-5)


class TestBatchOperations:
    """Test batch operations using vmap."""

    def test_batch_bind_bsc(self):
        k1, k2 = jax.random.split(jax.random.PRNGKey(42))
        x = jax.random.bernoulli(k1, 0.5, shape=(10, 100))
        y = jax.random.bernoulli(k2, 0.5, shape=(10, 100))
        result = F.batch_bind_bsc(x, y)
        assert result.shape == (10, 100)
        for i in range(10):
            assert jnp.array_equal(result[i], F.bind_bsc(x[i], y[i]))

    def test_batch_bind_map(self):
        k1, k2 = jax.random.split(jax.random.PRNGKey(42))
        x = jax.random.normal(k1, shape=(10, 100))
        y = jax.random.normal(k2, shape=(10, 100))
        result = F.batch_bind_map(x, y)
        assert result.shape == (10, 100)
        for i in range(10):
            assert jnp.allclose(result[i], F.bind_map(x[i], y[i]))

    def test_batch_cosine_similarity(self):
        """Test batch cosine similarity computation."""
        key = jax.random.PRNGKey(42)
        queries = jax.random.normal(key, shape=(10, 10000))
        target = jax.random.normal(key, shape=(10000,))

        similarities = F.batch_cosine_similarity(queries, target)

        assert similarities.shape == (10,)
        assert jnp.all((similarities >= -1) & (similarities <= 1))


class TestHRROperations:
    """Test Holographic Reduced Representations operations."""

    def test_bind_hrr_inverse(self):
        """Test HRR binding inverse."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, shape=(10000,))
        y = jax.random.normal(key, shape=(10000,))

        # Normalize
        x = x / jnp.linalg.norm(x)
        y = y / jnp.linalg.norm(y)

        # bind(bind(x, y), inverse(y)) should be close to x
        bound = F.bind_hrr(x, y)
        y_inv = F.inverse_hrr(y)
        unbound = F.bind_hrr(bound, y_inv)

        # Normalize for comparison
        unbound = unbound / jnp.linalg.norm(unbound)

        similarity = jnp.sum(x * unbound)
        assert similarity > 0.8  # Should be quite similar

    def test_bundle_hrr_normalization(self):
        """Test that HRR bundling produces normalized vectors."""
        key = jax.random.PRNGKey(42)
        vectors = jax.random.normal(key, shape=(10, 10000))

        bundled = F.bundle_hrr(vectors, axis=0)
        norm = jnp.linalg.norm(bundled)

        assert jnp.allclose(norm, 1.0, atol=1e-6)


class TestDotSimilarity:
    """Test dot product similarity."""

    def test_dot_similarity_self(self):
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, shape=(1000,))
        x = x / jnp.linalg.norm(x)
        assert jnp.allclose(F.dot_similarity(x, x), 1.0, atol=1e-5)

    def test_dot_similarity_orthogonal(self):
        x = jnp.zeros(100).at[0].set(1.0)
        y = jnp.zeros(100).at[1].set(1.0)
        assert jnp.allclose(F.dot_similarity(x, y), 0.0)

    def test_dot_similarity_shape(self):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (100,))
        y = jax.random.normal(jax.random.split(key)[1], (100,))
        result = F.dot_similarity(x, y)
        assert result.shape == ()


class TestNegation:
    """Test bundling inverses."""

    def test_negative_bsc(self):
        x = jnp.array([True, False, True, False])
        neg = F.negative_bsc(x)
        assert jnp.array_equal(neg, jnp.array([False, True, False, True]))

    def test_negative_map(self):
        x = jnp.array([1.0, -1.0, 0.5])
        neg = F.negative_map(x)
        assert jnp.allclose(neg, jnp.array([-1.0, 1.0, -0.5]))

    def test_negative_map_cancels_bundle(self):
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (1000,))
        result = x + F.negative_map(x)
        assert jnp.allclose(result, 0.0)


class TestMultibind:
    """Test multi-vector binding."""

    def test_multibind_map_two_vectors(self):
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (1000,))
        y = jax.random.normal(jax.random.split(key)[1], (1000,))
        vectors = jnp.stack([x, y])
        result = F.multibind_map(vectors)
        expected = F.bind_map(x, y)
        assert jnp.allclose(result, expected)

    def test_multibind_map_three_vectors(self):
        keys = jax.random.split(jax.random.PRNGKey(0), 3)
        vecs = jnp.stack([jax.random.normal(k, (100,)) for k in keys])
        result = F.multibind_map(vecs)
        expected = F.bind_map(F.bind_map(vecs[0], vecs[1]), vecs[2])
        assert jnp.allclose(result, expected)

    def test_multibind_bsc(self):
        key = jax.random.PRNGKey(42)
        x = jax.random.bernoulli(key, 0.5, (1000,))
        y = jax.random.bernoulli(jax.random.split(key)[1], 0.5, (1000,))
        vectors = jnp.stack([x, y])
        result = F.multibind_bsc(vectors)
        expected = F.bind_bsc(x, y)
        assert jnp.array_equal(result, expected)


class TestCrossProduct:
    """Test cross product of hypervector sets."""

    def test_cross_product_shape(self):
        key = jax.random.PRNGKey(42)
        a = jax.random.normal(key, (3, 100))
        b = jax.random.normal(jax.random.split(key)[1], (4, 100))
        result = F.cross_product(a, b)
        assert result.shape == (3, 4, 100)

    def test_cross_product_values(self):
        key = jax.random.PRNGKey(42)
        a = jax.random.normal(key, (2, 50))
        b = jax.random.normal(jax.random.split(key)[1], (2, 50))
        result = F.cross_product(a, b)
        expected_00 = F.bind_map(a[0], b[0])
        assert jnp.allclose(result[0, 0], expected_00)


class TestHashTable:
    """Test functional hash table creation."""

    def test_hash_table_retrieval(self):
        key = jax.random.PRNGKey(42)
        d = 10000
        keys = jnp.sign(jax.random.normal(key, (3, d)))
        values = jnp.sign(jax.random.normal(jax.random.split(key)[1], (3, d)))

        ht = F.hash_table(keys, values)
        retrieved = F.bind_map(ht, F.inverse_map(keys[0]))
        retrieved = retrieved / (jnp.linalg.norm(retrieved) + 1e-8)
        expected = values[0] / (jnp.linalg.norm(values[0]) + 1e-8)
        sim = F.cosine_similarity(retrieved, expected)
        assert sim > 0.5

    def test_hash_table_shape(self):
        key = jax.random.PRNGKey(0)
        keys = jax.random.normal(key, (5, 200))
        values = jax.random.normal(jax.random.split(key)[1], (5, 200))
        ht = F.hash_table(keys, values)
        assert ht.shape == (200,)


class TestNgrams:
    """Test n-gram encoding."""

    def test_ngrams_shape(self):
        key = jax.random.PRNGKey(42)
        vectors = jax.random.normal(key, (10, 100))
        result = F.ngrams(vectors, n=3)
        assert result.shape == (100,)

    def test_ngrams_too_few_vectors(self):
        vectors = jax.random.normal(jax.random.PRNGKey(0), (2, 100))
        with pytest.raises(ValueError):
            F.ngrams(vectors, n=3)

    def test_ngrams_deterministic(self):
        key = jax.random.PRNGKey(42)
        vectors = jax.random.normal(key, (5, 100))
        r1 = F.ngrams(vectors, n=2)
        r2 = F.ngrams(vectors, n=2)
        assert jnp.allclose(r1, r2)


class TestBundleSequence:
    """Test bundle-based sequence encoding."""

    def test_bundle_sequence_shape(self):
        vectors = jax.random.normal(jax.random.PRNGKey(42), (5, 100))
        result = F.bundle_sequence(vectors)
        assert result.shape == (100,)

    def test_bundle_sequence_order_matters(self):
        key = jax.random.PRNGKey(42)
        vectors = jax.random.normal(key, (5, 100))
        fwd = F.bundle_sequence(vectors)
        rev = F.bundle_sequence(vectors[::-1])
        assert not jnp.allclose(fwd, rev)


class TestBindSequence:
    """Test bind-based sequence encoding."""

    def test_bind_sequence_shape(self):
        vectors = jax.random.normal(jax.random.PRNGKey(42), (5, 100))
        result = F.bind_sequence(vectors)
        assert result.shape == (100,)

    def test_bind_sequence_order_matters(self):
        key = jax.random.PRNGKey(42)
        vectors = jax.random.normal(key, (5, 100))
        fwd = F.bind_sequence(vectors)
        rev = F.bind_sequence(vectors[::-1])
        assert not jnp.allclose(fwd, rev)


class TestGraphEncode:
    """Test graph encoding."""

    def test_graph_encode_shape(self):
        key = jax.random.PRNGKey(42)
        nodes = jax.random.normal(key, (5, 100))
        edges = jnp.array([[0, 1], [1, 2], [2, 3]])
        result = F.graph_encode(edges, nodes)
        assert result.shape == (100,)

    def test_graph_encode_directed_vs_undirected(self):
        key = jax.random.PRNGKey(42)
        nodes = jax.random.normal(key, (5, 100))
        edges = jnp.array([[0, 1], [1, 2]])
        directed = F.graph_encode(edges, nodes, directed=True)
        undirected = F.graph_encode(edges, nodes, directed=False)
        assert not jnp.allclose(directed, undirected)


class TestJaccardSimilarity:
    """Test Jaccard similarity."""

    def test_jaccard_identical(self):
        x = jnp.array([True, True, False, True])
        assert jnp.allclose(F.jaccard_similarity(x, x), 1.0, atol=1e-5)

    def test_jaccard_disjoint(self):
        x = jnp.array([True, True, False, False])
        y = jnp.array([False, False, True, True])
        assert jnp.allclose(F.jaccard_similarity(x, y), 0.0, atol=1e-5)

    def test_jaccard_partial(self):
        x = jnp.array([True, True, False, False])
        y = jnp.array([True, False, True, False])
        sim = F.jaccard_similarity(x, y)
        assert 0.0 < sim < 1.0

    def test_jaccard_random_range(self):
        key = jax.random.PRNGKey(42)
        x = jax.random.bernoulli(key, 0.5, (10000,))
        y = jax.random.bernoulli(jax.random.split(key)[1], 0.5, (10000,))
        sim = F.jaccard_similarity(x, y)
        assert 0.2 < sim < 0.5


class TestTverskySimilarity:
    """Test Tversky similarity."""

    def test_tversky_is_jaccard_when_alpha_beta_1(self):
        key = jax.random.PRNGKey(42)
        x = jax.random.bernoulli(key, 0.5, (1000,))
        y = jax.random.bernoulli(jax.random.split(key)[1], 0.5, (1000,))
        jaccard = F.jaccard_similarity(x, y)
        tversky = F.tversky_similarity(x, y, alpha=1.0, beta=1.0)
        assert jnp.allclose(jaccard, tversky, atol=1e-5)

    def test_tversky_identical(self):
        x = jnp.array([True, True, False, True])
        assert jnp.allclose(F.tversky_similarity(x, x), 1.0, atol=1e-5)


class TestSelect:
    """Test MUX selection operations."""

    def test_select_bsc(self):
        cond = jnp.array([True, False, True, False])
        a = jnp.array([True, True, True, True])
        b = jnp.array([False, False, False, False])
        result = F.select_bsc(cond, a, b)
        expected = jnp.array([True, False, True, False])
        assert jnp.array_equal(result, expected)

    def test_select_map(self):
        cond = jnp.array([1.0, -1.0, 1.0, -1.0])
        a = jnp.array([10.0, 10.0, 10.0, 10.0])
        b = jnp.array([20.0, 20.0, 20.0, 20.0])
        result = F.select_map(cond, a, b)
        expected = jnp.array([10.0, 20.0, 10.0, 20.0])
        assert jnp.allclose(result, expected)


class TestThreshold:
    """Test generalised majority and window operations."""

    def test_threshold_majority(self):
        vectors = jnp.array(
            [
                [True, True, False],
                [True, False, False],
                [True, True, True],
            ]
        )
        result = F.threshold(vectors, t=2)
        expected = jnp.array([True, True, False])
        assert jnp.array_equal(result, expected)

    def test_threshold_unanimous(self):
        vectors = jnp.array(
            [
                [True, True, False],
                [True, False, False],
                [True, True, True],
            ]
        )
        result = F.threshold(vectors, t=3)
        expected = jnp.array([True, False, False])
        assert jnp.array_equal(result, expected)

    def test_window(self):
        vectors = jnp.array(
            [
                [True, True, False, True],
                [True, False, False, True],
                [True, True, False, True],
            ]
        )
        result = F.window(vectors, lo=2, hi=2)
        expected = jnp.array([False, True, False, False])
        assert jnp.array_equal(result, expected)


class TestNoiseInjection:
    """Test noise injection functions."""

    def test_flip_fraction_shape(self):
        key = jax.random.PRNGKey(42)
        x = jax.random.bernoulli(key, 0.5, (1000,))
        noisy = F.flip_fraction(jax.random.PRNGKey(0), x, 0.1)
        assert noisy.shape == x.shape

    def test_flip_fraction_approx_rate(self):
        key = jax.random.PRNGKey(42)
        x = jax.random.bernoulli(key, 0.5, (10000,))
        noisy = F.flip_fraction(jax.random.PRNGKey(0), x, 0.1)
        flipped = jnp.sum(x != noisy)
        assert 500 < flipped < 1500

    def test_flip_fraction_zero(self):
        key = jax.random.PRNGKey(42)
        x = jax.random.bernoulli(key, 0.5, (100,))
        noisy = F.flip_fraction(jax.random.PRNGKey(0), x, 0.0)
        assert jnp.array_equal(noisy, x)

    def test_add_noise_map_shape(self):
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (100,))
        x = x / jnp.linalg.norm(x)
        noisy = F.add_noise_map(jax.random.PRNGKey(0), x, 0.1)
        assert noisy.shape == x.shape

    def test_add_noise_map_normalised(self):
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (100,))
        x = x / jnp.linalg.norm(x)
        noisy = F.add_noise_map(jax.random.PRNGKey(0), x, 0.1)
        assert jnp.allclose(jnp.linalg.norm(noisy), 1.0, atol=1e-5)


class TestQuantisation:
    """Test soft and hard quantisation."""

    def test_soft_quantize(self):
        x = jnp.array([0.0, 10.0, -10.0])
        result = F.soft_quantize(x)
        assert jnp.allclose(result[0], 0.0, atol=1e-5)
        assert result[1] > 0.99
        assert result[2] < -0.99

    def test_hard_quantize(self):
        x = jnp.array([0.5, -0.3, 0.0, 2.0])
        result = F.hard_quantize(x)
        expected = jnp.array([1.0, -1.0, -1.0, 1.0])
        assert jnp.allclose(result, expected)


class TestResonator:
    """Test resonator network."""

    def test_resonator_simple(self):
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        d = 10000
        cb1 = jnp.sign(jax.random.normal(k1, (3, d)))
        cb2 = jnp.sign(jax.random.normal(k2, (3, d)))

        target = F.bind_map(cb1[1], cb2[2])
        estimates = F.resonator([cb1, cb2], target, max_iters=100)

        assert len(estimates) == 2
        sim1 = F.cosine_similarity(estimates[0], cb1[1])
        sim2 = F.cosine_similarity(estimates[1], cb2[2])
        assert sim1 > 0.8
        assert sim2 > 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
