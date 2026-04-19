# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Tests for VSA models."""

import jax
import jax.numpy as jnp
import pytest

from jax_hdc.vsa import BSBC, BSC, CGR, FHRR, HRR, MAP, MCR, VTB, VSAModel, create_vsa_model


class TestVSAModels:
    """Test VSA model creation and basic operations."""

    def test_create_bsc(self):
        """Test BSC model creation."""
        model = BSC.create(dimensions=10000)

        assert model.name == "bsc"
        assert model.dimensions == 10000

    def test_create_map(self):
        """Test MAP model creation."""
        model = MAP.create(dimensions=10000)

        assert model.name == "map"
        assert model.dimensions == 10000

    def test_create_hrr(self):
        """Test HRR model creation."""
        model = HRR.create(dimensions=10000)

        assert model.name == "hrr"
        assert model.dimensions == 10000

    def test_create_fhrr(self):
        """Test FHRR model creation."""
        model = FHRR.create(dimensions=10000)

        assert model.name == "fhrr"
        assert model.dimensions == 10000

    def test_create_bsbc(self):
        """Test BSBC model creation."""
        model = BSBC.create(dimensions=1000, block_size=100, k_active=5)

        assert model.name == "bsbc"
        assert model.dimensions == 1000
        assert model.block_size == 100
        assert model.k_active == 5

    def test_factory_function(self):
        """Test create_vsa_model factory function."""
        for model_type in ["bsc", "map", "hrr", "fhrr", "bsbc", "cgr", "mcr", "vtb"]:
            model = create_vsa_model(model_type, dimensions=10000)
            assert model.name == model_type
            assert model.dimensions == 10000

    def test_factory_invalid_type(self):
        """Test that invalid model type raises error."""
        with pytest.raises(ValueError):
            create_vsa_model("invalid_model", dimensions=10000)


class TestBSBCModel:
    """Test BSBC model operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = BSBC.create(dimensions=1000, block_size=100, k_active=5)
        self.key = jax.random.PRNGKey(42)

    def test_random_generation(self):
        """Test random block-sparse hypervector generation."""
        x = self.model.random(self.key, (1000,))

        assert x.shape == (1000,)
        assert x.dtype == jnp.bool_
        num_blocks = 1000 // 100
        expected_ones = num_blocks * 5
        assert jnp.sum(x) == expected_ones

    def test_bind_operation(self):
        """Test binding (XOR) operation."""
        x = self.model.random(self.key, (1000,))
        y = self.model.random(jax.random.split(self.key)[1], (1000,))

        bound = self.model.bind(x, y)

        assert bound.shape == (1000,)
        assert bound.dtype == jnp.bool_

    def test_similarity(self):
        """Test similarity computation."""
        x = self.model.random(self.key, (1000,))
        sim_self = self.model.similarity(x, x)
        assert jnp.allclose(sim_self, 1.0)


class TestBSCModel:
    """Test BSC model operations."""

    def setup_method(self):
        self.model = BSC.create(dimensions=10000)
        self.key = jax.random.PRNGKey(42)

    def test_random_generation(self):
        x = self.model.random(self.key, (10000,))
        assert x.shape == (10000,)
        assert x.dtype == jnp.bool_
        assert 0.45 < float(jnp.mean(x.astype(jnp.float32))) < 0.55

    def test_bind_operation(self):
        k1, k2 = jax.random.split(self.key)
        x = self.model.random(k1, (10000,))
        y = self.model.random(k2, (10000,))
        bound = self.model.bind(x, y)
        assert bound.shape == (10000,)
        assert bound.dtype == jnp.bool_
        sim = self.model.similarity(bound, x)
        assert 0.45 < float(sim) < 0.55

    def test_bundle_operation(self):
        vectors = self.model.random(self.key, (10, 10000))
        bundled = self.model.bundle(vectors, axis=0)
        assert bundled.shape == (10000,)
        assert bundled.dtype == jnp.bool_

    def test_inverse_operation(self):
        x = self.model.random(self.key, (10000,))
        assert jnp.array_equal(self.model.inverse(x), x)

    def test_similarity_computation(self):
        k1, k2 = jax.random.split(self.key)
        x = self.model.random(k1, (10000,))
        y = self.model.random(k2, (10000,))
        assert jnp.allclose(self.model.similarity(x, x), 1.0)
        sim_rand = self.model.similarity(x, y)
        assert 0.45 < float(sim_rand) < 0.55


class TestMAPModel:
    """Test MAP model operations."""

    def setup_method(self):
        self.model = MAP.create(dimensions=10000)
        self.key = jax.random.PRNGKey(42)

    def test_random_generation(self):
        x = self.model.random(self.key, (10000,))
        assert x.shape == (10000,)
        assert jnp.allclose(jnp.linalg.norm(x), 1.0)

    def test_bind_operation(self):
        k1, k2 = jax.random.split(self.key)
        x = self.model.random(k1, (10000,))
        y = self.model.random(k2, (10000,))
        bound = self.model.bind(x, y)
        assert bound.shape == (10000,)
        assert abs(float(self.model.similarity(bound, x))) < 0.1
        assert abs(float(self.model.similarity(bound, y))) < 0.1

    def test_bundle_operation(self):
        vectors = self.model.random(self.key, (10, 10000))
        bundled = self.model.bundle(vectors, axis=0)
        assert bundled.shape == (10000,)
        assert jnp.allclose(jnp.linalg.norm(bundled), 1.0, atol=1e-6)

    def test_inverse_operation(self):
        k1, k2 = jax.random.split(self.key)
        x = self.model.random(k1, (10000,))
        y = self.model.random(k2, (10000,))
        bound = self.model.bind(x, y)
        unbound = self.model.bind(bound, self.model.inverse(x))
        unbound = unbound / (jnp.linalg.norm(unbound) + 1e-8)
        sim = self.model.similarity(unbound, y)
        assert float(sim) > 0.9

    def test_similarity_computation(self):
        k1, k2 = jax.random.split(self.key)
        x = self.model.random(k1, (10000,))
        y = self.model.random(k2, (10000,))
        assert jnp.allclose(self.model.similarity(x, x), 1.0, atol=1e-6)
        assert abs(float(self.model.similarity(x, y))) < 0.1


class TestHRRModel:
    """Test HRR model operations."""

    def setup_method(self):
        self.model = HRR.create(dimensions=10000)
        self.key = jax.random.PRNGKey(42)

    def test_random_generation(self):
        x = self.model.random(self.key, (10000,))
        assert x.shape == (10000,)
        assert jnp.allclose(jnp.linalg.norm(x), 1.0)

    def test_bind_unbind_cycle(self):
        k1, k2 = jax.random.split(self.key)
        x = self.model.random(k1, (10000,))
        y = self.model.random(k2, (10000,))
        bound = self.model.bind(x, y)
        unbound = self.model.bind(bound, self.model.inverse(y))
        unbound = unbound / jnp.linalg.norm(unbound)
        sim = self.model.similarity(unbound, x)
        assert float(sim) > 0.6

    def test_bind_dissimilarity(self):
        k1, k2 = jax.random.split(self.key)
        x = self.model.random(k1, (10000,))
        y = self.model.random(k2, (10000,))
        bound = self.model.bind(x, y)
        bound = bound / (jnp.linalg.norm(bound) + 1e-8)
        assert abs(float(self.model.similarity(bound, x))) < 0.15
        assert abs(float(self.model.similarity(bound, y))) < 0.15


class TestFHRRModel:
    """Test FHRR model operations."""

    def setup_method(self):
        self.model = FHRR.create(dimensions=10000)
        self.key = jax.random.PRNGKey(42)

    def test_random_generation(self):
        x = self.model.random(self.key, (10000,))
        assert x.shape == (10000,)
        assert x.dtype == jnp.complex64 or x.dtype == jnp.complex128
        assert jnp.allclose(jnp.abs(x), 1.0, atol=1e-6)

    def test_bind_operation(self):
        k1, k2 = jax.random.split(self.key)
        x = self.model.random(k1, (10000,))
        y = self.model.random(k2, (10000,))
        bound = self.model.bind(x, y)
        assert bound.shape == (10000,)
        assert jnp.allclose(jnp.abs(bound), 1.0, atol=1e-6)

    def test_bind_unbind_cycle(self):
        k1, k2 = jax.random.split(self.key)
        x = self.model.random(k1, (10000,))
        y = self.model.random(k2, (10000,))
        bound = self.model.bind(x, y)
        unbound = self.model.bind(bound, self.model.inverse(y))
        sim = self.model.similarity(unbound, x)
        assert float(sim) > 0.99

    def test_inverse_conjugate(self):
        x = self.model.random(self.key, (10000,))
        assert jnp.allclose(self.model.inverse(x), jnp.conj(x))


class TestCGRModel:
    """Test CGR model operations."""

    def setup_method(self):
        self.model = CGR.create(dimensions=10000, q=8)
        self.key = jax.random.PRNGKey(42)

    def test_create(self):
        assert self.model.name == "cgr"
        assert self.model.dimensions == 10000
        assert self.model.q == 8

    def test_create_invalid_q(self):
        with pytest.raises(ValueError):
            CGR.create(dimensions=10000, q=1)

    def test_random_generation(self):
        x = self.model.random(self.key, (10000,))
        assert x.shape == (10000,)
        assert jnp.all(x >= 0) and jnp.all(x < 8)

    def test_bind_operation(self):
        k1, k2 = jax.random.split(self.key)
        x = self.model.random(k1, (10000,))
        y = self.model.random(k2, (10000,))
        bound = self.model.bind(x, y)
        assert bound.shape == (10000,)
        assert jnp.all(bound >= 0) and jnp.all(bound < 8)

    def test_inverse_operation(self):
        x = self.model.random(self.key, (10000,))
        x_inv = self.model.inverse(x)
        result = self.model.bind(x, x_inv)
        assert jnp.allclose(result, 0)

    def test_bundle_operation(self):
        vectors = jnp.stack([self.model.random(jax.random.PRNGKey(i), (10000,)) for i in range(5)])
        bundled = self.model.bundle(vectors, axis=0)
        assert bundled.shape == (10000,)
        assert jnp.all(bundled >= 0) and jnp.all(bundled < 8)

    def test_similarity_self(self):
        x = self.model.random(self.key, (10000,))
        sim = self.model.similarity(x, x)
        assert jnp.allclose(sim, 1.0)

    def test_similarity_random(self):
        k1, k2 = jax.random.split(self.key)
        x = self.model.random(k1, (10000,))
        y = self.model.random(k2, (10000,))
        sim = self.model.similarity(x, y)
        assert 0.0 < sim < 0.3


class TestMCRModel:
    """Test MCR model operations."""

    def setup_method(self):
        self.model = MCR.create(dimensions=10000, q=64)
        self.key = jax.random.PRNGKey(42)

    def test_create(self):
        assert self.model.name == "mcr"
        assert self.model.dimensions == 10000
        assert self.model.q == 64

    def test_create_invalid_q(self):
        with pytest.raises(ValueError):
            MCR.create(dimensions=10000, q=1)

    def test_random_generation(self):
        x = self.model.random(self.key, (10000,))
        assert x.shape == (10000,)
        assert jnp.all(x >= 0) and jnp.all(x < 64)

    def test_bind_operation(self):
        k1, k2 = jax.random.split(self.key)
        x = self.model.random(k1, (10000,))
        y = self.model.random(k2, (10000,))
        bound = self.model.bind(x, y)
        assert bound.shape == (10000,)
        assert jnp.all(bound >= 0) and jnp.all(bound < 64)

    def test_inverse_operation(self):
        x = self.model.random(self.key, (10000,))
        x_inv = self.model.inverse(x)
        result = self.model.bind(x, x_inv)
        assert jnp.allclose(result, 0)

    def test_bundle_operation(self):
        vectors = jnp.stack([self.model.random(jax.random.PRNGKey(i), (10000,)) for i in range(5)])
        bundled = self.model.bundle(vectors, axis=0)
        assert bundled.shape == (10000,)
        assert jnp.all(bundled >= 0) and jnp.all(bundled < 64)

    def test_similarity_self(self):
        x = self.model.random(self.key, (10000,))
        sim = self.model.similarity(x, x)
        assert jnp.allclose(sim, 1.0)

    def test_similarity_random(self):
        k1, k2 = jax.random.split(self.key)
        x = self.model.random(k1, (10000,))
        y = self.model.random(k2, (10000,))
        sim = self.model.similarity(x, y)
        assert -0.1 < sim < 0.1


class TestVTBModel:
    """Test VTB model operations."""

    def setup_method(self):
        self.model = VTB.create(dimensions=10000)
        self.key = jax.random.PRNGKey(42)

    def test_create(self):
        assert self.model.name == "vtb"
        assert self.model.dimensions == 10000

    def test_create_invalid_dimensions(self):
        with pytest.raises(ValueError):
            VTB.create(dimensions=10001)

    def test_random_generation(self):
        x = self.model.random(self.key, (10000,))
        assert x.shape == (10000,)
        norm = jnp.linalg.norm(x)
        assert jnp.allclose(norm, 1.0)

    def test_bind_operation(self):
        k1, k2 = jax.random.split(self.key)
        x = self.model.random(k1, (10000,))
        y = self.model.random(k2, (10000,))
        bound = self.model.bind(x, y)
        assert bound.shape == (10000,)

    def test_bind_not_commutative(self):
        """VTB binding is non-commutative (matrix multiplication)."""
        k1, k2 = jax.random.split(self.key)
        x = self.model.random(k1, (10000,))
        y = self.model.random(k2, (10000,))
        xy = self.model.bind(x, y)
        yx = self.model.bind(y, x)
        assert not jnp.allclose(xy, yx, atol=1e-3)

    def test_bundle_operation(self):
        vectors = self.model.random(self.key, (5, 10000))
        bundled = self.model.bundle(vectors, axis=0)
        assert bundled.shape == (10000,)
        norm = jnp.linalg.norm(bundled)
        assert jnp.allclose(norm, 1.0, atol=1e-5)

    def test_similarity_self(self):
        x = self.model.random(self.key, (10000,))
        sim = self.model.similarity(x, x)
        assert jnp.allclose(sim, 1.0, atol=1e-5)

    def test_inverse_operation(self):
        k1, k2 = jax.random.split(self.key)
        x = self.model.random(k1, (10000,))
        y = self.model.random(k2, (10000,))
        bound = self.model.bind(x, y)
        x_inv = self.model.inverse(x)
        unbound = self.model.bind(x_inv, bound)
        unbound_norm = unbound / (jnp.linalg.norm(unbound) + 1e-8)
        sim = self.model.similarity(unbound_norm, y)
        assert sim > 0.5


class TestVSAModelAbstract:
    """Test VSAModel base class raises NotImplementedError."""

    def test_bind_not_implemented(self):
        m = VSAModel(name="test", dimensions=100)
        with pytest.raises(NotImplementedError):
            m.bind(jnp.zeros(100), jnp.zeros(100))

    def test_bundle_not_implemented(self):
        m = VSAModel(name="test", dimensions=100)
        with pytest.raises(NotImplementedError):
            m.bundle(jnp.zeros((2, 100)))

    def test_inverse_not_implemented(self):
        m = VSAModel(name="test", dimensions=100)
        with pytest.raises(NotImplementedError):
            m.inverse(jnp.zeros(100))

    def test_similarity_not_implemented(self):
        m = VSAModel(name="test", dimensions=100)
        with pytest.raises(NotImplementedError):
            m.similarity(jnp.zeros(100), jnp.zeros(100))

    def test_random_not_implemented(self):
        m = VSAModel(name="test", dimensions=100)
        with pytest.raises(NotImplementedError):
            m.random(jax.random.PRNGKey(0), (100,))


class TestHRRCoverage:
    """Cover HRR bundle."""

    def test_bundle(self):
        m = HRR.create(dimensions=100)
        k = jax.random.PRNGKey(0)
        vecs = m.random(k, (3, 100))
        bundled = m.bundle(vecs, axis=0)
        assert bundled.shape == (100,)


class TestFHRRCoverage:
    """Cover FHRR bundle and similarity."""

    def test_bundle(self):
        m = FHRR.create(dimensions=100)
        k = jax.random.PRNGKey(0)
        vecs = m.random(k, (3, 100))
        bundled = m.bundle(vecs, axis=0)
        assert bundled.shape == (100,)

    def test_similarity(self):
        m = FHRR.create(dimensions=100)
        k = jax.random.PRNGKey(0)
        x = m.random(k, (100,))
        sim = m.similarity(x, x)
        assert float(sim) > 0.99


class TestBSBCCoverage:
    """Cover BSBC bundle, inverse, validation, and random batch shapes."""

    def test_bsbc_bad_dimensions(self):
        with pytest.raises(ValueError, match="divisible"):
            BSBC.create(dimensions=101, block_size=10, k_active=2)

    def test_bsbc_bad_k_active(self):
        with pytest.raises(ValueError, match="k_active"):
            BSBC.create(dimensions=100, block_size=10, k_active=0)
        with pytest.raises(ValueError, match="k_active"):
            BSBC.create(dimensions=100, block_size=10, k_active=11)

    def test_bundle(self):
        m = BSBC.create(dimensions=100, block_size=10, k_active=2)
        k = jax.random.PRNGKey(0)
        vecs = m.random(k, (3, 100))
        bundled = m.bundle(vecs, axis=0)
        assert bundled.shape == (100,)

    def test_inverse(self):
        m = BSBC.create(dimensions=100, block_size=10, k_active=2)
        k = jax.random.PRNGKey(0)
        x = m.random(k, (100,))
        inv = m.inverse(x)
        assert inv.shape == (100,)

    def test_random_batch_reshape(self):
        m = BSBC.create(dimensions=100, block_size=10, k_active=2)
        k = jax.random.PRNGKey(0)
        hvs = m.random(k, (5, 100))
        assert hvs.shape == (5, 100)

    def test_random_single_1d(self):
        m = BSBC.create(dimensions=100, block_size=10, k_active=2)
        k = jax.random.PRNGKey(0)
        hv = m.random(k, (100,))
        assert hv.shape == (100,)

    def test_random_short_1d_shape(self):
        """Cover the second 1D branch (shape != (dimensions,) but len==1)."""
        m = BSBC.create(dimensions=100, block_size=10, k_active=2)
        k = jax.random.PRNGKey(0)
        hv = m.random(k, (50,))
        assert hv.ndim == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
