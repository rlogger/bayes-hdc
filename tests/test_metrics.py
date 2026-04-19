# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Tests for statistical metrics and capacity analysis."""

import jax
import jax.numpy as jnp

from jax_hdc import functional as F
from jax_hdc.metrics import (
    bundle_capacity,
    bundle_snr,
    cosine_matrix,
    effective_dimensions,
    retrieval_confidence,
    saturation,
    signal_energy,
    sparsity,
)


class TestBundleSNR:
    def test_snr_increases_with_d(self):
        snr_low = bundle_snr(1000, 10)
        snr_high = bundle_snr(10000, 10)
        assert float(snr_high) > float(snr_low)

    def test_snr_decreases_with_n(self):
        snr_few = bundle_snr(10000, 5)
        snr_many = bundle_snr(10000, 100)
        assert float(snr_few) > float(snr_many)

    def test_snr_known_value(self):
        snr = bundle_snr(10000, 11)
        expected = jnp.sqrt(10000.0 / 10.0)
        assert jnp.allclose(snr, expected, atol=0.1)


class TestBundleCapacity:
    def test_capacity_increases_with_d(self):
        cap_low = bundle_capacity(1000)
        cap_high = bundle_capacity(10000)
        assert float(cap_high) > float(cap_low)

    def test_capacity_decreases_with_stricter_delta(self):
        cap_loose = bundle_capacity(10000, delta=0.1)
        cap_strict = bundle_capacity(10000, delta=0.001)
        assert float(cap_loose) > float(cap_strict)

    def test_capacity_positive(self):
        assert float(bundle_capacity(100)) > 0


class TestEffectiveDimensions:
    def test_uniform_vector(self):
        x = jnp.ones(100) / jnp.sqrt(100.0)
        pr = effective_dimensions(x)
        assert jnp.allclose(pr, 100.0, atol=1.0)

    def test_sparse_vector(self):
        x = jnp.zeros(100).at[0].set(1.0)
        pr = effective_dimensions(x)
        assert jnp.allclose(pr, 1.0, atol=0.1)

    def test_random_vector(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (10000,))
        pr = effective_dimensions(x)
        assert float(pr) > 1000


class TestSparsity:
    def test_dense_vector(self):
        x = jnp.ones(100)
        assert jnp.allclose(sparsity(x), 0.0)

    def test_sparse_vector(self):
        x = jnp.zeros(100)
        assert jnp.allclose(sparsity(x), 1.0)

    def test_partial_sparsity(self):
        x = jnp.zeros(100).at[:50].set(1.0)
        assert jnp.allclose(sparsity(x), 0.5)


class TestSignalEnergy:
    def test_unit_vector(self):
        x = jnp.ones(100) / jnp.sqrt(100.0)
        assert jnp.allclose(signal_energy(x), 1.0, atol=1e-5)

    def test_zero_vector(self):
        x = jnp.zeros(100)
        assert jnp.allclose(signal_energy(x), 0.0)


class TestSaturation:
    def test_fully_saturated(self):
        x = jnp.array([1.0, -1.0, 1.0, -1.0])
        assert jnp.allclose(saturation(x), 1.0)

    def test_unsaturated(self):
        x = jnp.array([0.1, -0.1, 0.2, -0.2])
        assert jnp.allclose(saturation(x), 0.0)


class TestCosineMatrix:
    def test_identity_on_orthonormal(self):
        eye = jnp.eye(3)
        cm = cosine_matrix(eye)
        assert jnp.allclose(cm, jnp.eye(3), atol=1e-5)

    def test_self_similarity(self):
        k1, k2 = jax.random.split(jax.random.PRNGKey(42))
        vecs = jax.random.normal(k1, (5, 1000))
        cm = cosine_matrix(vecs)
        diag = jnp.diag(cm)
        assert jnp.all(diag > 0.99)

    def test_random_near_orthogonal(self):
        vecs = jax.random.normal(jax.random.PRNGKey(42), (5, 10000))
        cm = cosine_matrix(vecs)
        off_diag = cm - jnp.diag(jnp.diag(cm))
        assert jnp.all(jnp.abs(off_diag) < 0.1)


class TestRetrievalConfidence:
    def test_exact_match_high_confidence(self):
        key = jax.random.PRNGKey(42)
        codebook = jax.random.normal(key, (5, 10000))
        gap = retrieval_confidence(codebook[0], codebook)
        assert float(gap) > 0.5

    def test_ambiguous_query_low_confidence(self):
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        a = jax.random.normal(k1, (10000,))
        b = jax.random.normal(k2, (10000,))
        query = a + b
        codebook = jnp.stack([a, b])
        gap = retrieval_confidence(query, codebook)
        assert float(gap) < 0.3


class TestFractionalPower:
    def test_power_one_is_identity(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (100,))
        result = F.fractional_power(x, 1.0)
        assert jnp.allclose(result, x, atol=1e-5)

    def test_power_zero_gives_sign(self):
        x = jnp.array([2.0, -3.0, 0.5, -0.5])
        result = F.fractional_power(x, 0.0)
        expected = jnp.array([1.0, -1.0, 1.0, -1.0])
        assert jnp.allclose(result, expected)

    def test_smooth_interpolation(self):
        k1, k2 = jax.random.split(jax.random.PRNGKey(42))
        x = jax.random.normal(k1, (10000,))
        x = x / jnp.linalg.norm(x)
        low = F.fractional_power(x, 0.1)
        mid = F.fractional_power(x, 0.5)
        high = F.fractional_power(x, 0.9)
        sim_low = F.cosine_similarity(x, low)
        sim_mid = F.cosine_similarity(x, mid)
        sim_high = F.cosine_similarity(x, high)
        assert float(sim_low) < float(sim_mid) < float(sim_high)
