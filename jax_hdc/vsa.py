# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Vector Symbolic Architecture (VSA) model implementations.

This module provides different VSA models, each with their own binding,
bundling, and similarity operations. All models follow a consistent API.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from jax_hdc import functional as F
from jax_hdc._compat import register_dataclass
from jax_hdc.constants import EPS


@register_dataclass
@dataclass
class VSAModel:
    """Base class for VSA models defining the interface."""

    name: str = field(metadata=dict(static=True))
    dimensions: int = field(metadata=dict(static=True))

    def bind(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Bind two hypervectors."""
        raise NotImplementedError

    def bundle(self, vectors: jax.Array, axis: int = 0) -> jax.Array:
        """Bundle multiple hypervectors."""
        raise NotImplementedError

    def inverse(self, x: jax.Array) -> jax.Array:
        """Compute the inverse of a hypervector."""
        raise NotImplementedError

    def similarity(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Compute similarity between hypervectors."""
        raise NotImplementedError

    def random(self, key: jax.Array, shape: tuple) -> jax.Array:
        """Generate random hypervectors."""
        raise NotImplementedError


@register_dataclass
@dataclass
class BSC(VSAModel):
    """Binary Spatter Codes (BSC).

    Binary hypervectors with XOR binding, majority bundling, Hamming similarity.
    """

    @staticmethod
    def create(dimensions: int = 10000) -> "BSC":
        """Create a BSC model.

        Args:
            dimensions: Dimensionality of hypervectors (default: 10000)

        Returns:
            Initialized BSC model
        """
        return BSC(name="bsc", dimensions=dimensions)

    @jax.jit
    def bind(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Bind using XOR."""
        return F.bind_bsc(x, y)

    def bundle(self, vectors: jax.Array, axis: int = 0) -> jax.Array:
        """Bundle using majority rule."""
        return F.bundle_bsc(vectors, axis=axis)

    @jax.jit
    def inverse(self, x: jax.Array) -> jax.Array:
        """Inverse is identity for XOR."""
        return F.inverse_bsc(x)

    @jax.jit
    def similarity(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Compute Hamming similarity."""
        return F.hamming_similarity(x, y)

    def random(self, key: jax.Array, shape: tuple) -> jax.Array:
        """Generate random binary hypervectors.

        Args:
            key: JAX random key
            shape: Shape of output array

        Returns:
            Random binary hypervectors with ~50% ones
        """
        return jax.random.bernoulli(key, 0.5, shape=shape)


@register_dataclass
@dataclass
class MAP(VSAModel):
    """Multiply-Add-Permute (MAP).

    Real-valued vectors with element-wise multiply binding, normalized sum bundling,
    cosine similarity.
    """

    @staticmethod
    def create(dimensions: int = 10000) -> "MAP":
        """Create a MAP model.

        Args:
            dimensions: Dimensionality of hypervectors (default: 10000)

        Returns:
            Initialized MAP model
        """
        return MAP(name="map", dimensions=dimensions)

    @jax.jit
    def bind(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Bind using element-wise multiplication."""
        return F.bind_map(x, y)

    def bundle(self, vectors: jax.Array, axis: int = 0) -> jax.Array:
        """Bundle using normalized sum."""
        return F.bundle_map(vectors, axis=axis)

    @jax.jit
    def inverse(self, x: jax.Array) -> jax.Array:
        """Inverse via element-wise reciprocal."""
        return F.inverse_map(x)

    @jax.jit
    def similarity(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Compute cosine similarity."""
        return F.cosine_similarity(x, y)

    def random(self, key: jax.Array, shape: tuple) -> jax.Array:
        """Generate random real-valued hypervectors.

        Args:
            key: JAX random key
            shape: Shape of output array

        Returns:
            Random normalized hypervectors sampled from normal distribution
        """
        vectors = jax.random.normal(key, shape=shape)
        # Normalize to unit length
        norm = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
        return vectors / (norm + EPS)


@register_dataclass
@dataclass
class HRR(VSAModel):
    """Holographic Reduced Representations (HRR).

    Real-valued vectors with circular convolution binding, normalized sum bundling,
    cosine similarity.
    """

    @staticmethod
    def create(dimensions: int = 10000) -> "HRR":
        """Create an HRR model.

        Args:
            dimensions: Dimensionality of hypervectors (default: 10000)

        Returns:
            Initialized HRR model
        """
        return HRR(name="hrr", dimensions=dimensions)

    @jax.jit
    def bind(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Bind using circular convolution."""
        return F.bind_hrr(x, y)

    def bundle(self, vectors: jax.Array, axis: int = 0) -> jax.Array:
        """Bundle using normalized sum."""
        return F.bundle_hrr(vectors, axis=axis)

    @jax.jit
    def inverse(self, x: jax.Array) -> jax.Array:
        """Inverse via element reversal."""
        return F.inverse_hrr(x)

    @jax.jit
    def similarity(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Compute cosine similarity."""
        return F.cosine_similarity(x, y)

    def random(self, key: jax.Array, shape: tuple) -> jax.Array:
        """Generate random real-valued hypervectors.

        Args:
            key: JAX random key
            shape: Shape of output array

        Returns:
            Random normalized hypervectors sampled from normal distribution
        """
        vectors = jax.random.normal(key, shape=shape)
        # Normalize to unit length
        norm = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
        return vectors / (norm + EPS)


@register_dataclass
@dataclass
class FHRR(VSAModel):
    """Fourier Holographic Reduced Representations (FHRR).

    Complex-valued vectors with element-wise multiply binding, normalized sum bundling.
    """

    @staticmethod
    def create(dimensions: int = 10000) -> "FHRR":
        """Create an FHRR model.

        Args:
            dimensions: Dimensionality of hypervectors (default: 10000)

        Returns:
            Initialized FHRR model
        """
        return FHRR(name="fhrr", dimensions=dimensions)

    @jax.jit
    def bind(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Bind using element-wise multiplication."""
        return x * y

    def bundle(self, vectors: jax.Array, axis: int = 0) -> jax.Array:
        """Bundle using normalized sum."""
        summed = jnp.sum(vectors, axis=axis)
        norm = jnp.linalg.norm(summed, axis=-1, keepdims=True)
        return summed / (norm + EPS)

    @jax.jit
    def inverse(self, x: jax.Array) -> jax.Array:
        """Inverse via complex conjugate."""
        return jnp.conj(x)

    @jax.jit
    def similarity(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Compute cosine similarity of complex vectors."""
        x_norm = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + EPS)
        y_norm = y / (jnp.linalg.norm(y, axis=-1, keepdims=True) + EPS)
        # Use real part of inner product, clip to handle floating point precision
        return jnp.clip(jnp.real(jnp.sum(x_norm * jnp.conj(y_norm), axis=-1)), -1.0, 1.0)

    def random(self, key: jax.Array, shape: tuple) -> jax.Array:
        """Generate random complex hypervectors on unit circle.

        Args:
            key: JAX random key
            shape: Shape of output array

        Returns:
            Random unit complex hypervectors
        """
        # Random phases on unit circle
        phases = jax.random.uniform(key, shape=shape, minval=0, maxval=2 * jnp.pi)
        return jnp.exp(1j * phases)


@register_dataclass
@dataclass
class BSBC(VSAModel):
    """Binary Sparse Block Codes (B-SBC).

    Block-sparse binary vectors with k_active ones per block, XOR binding,
    majority bundling.
    """

    block_size: int = field(metadata=dict(static=True), default=100)
    k_active: int = field(metadata=dict(static=True), default=5)

    @staticmethod
    def create(
        dimensions: int = 10000,
        block_size: int = 100,
        k_active: int = 5,
    ) -> "BSBC":
        """Create a B-SBC model.

        Args:
            dimensions: Total dimensionality (must be divisible by block_size)
            block_size: Size of each block
            k_active: Number of ones per block (sparsity)

        Returns:
            Initialized BSBC model
        """
        if dimensions % block_size != 0:
            raise ValueError(
                f"dimensions ({dimensions}) must be divisible by block_size ({block_size})"
            )
        if k_active > block_size or k_active < 1:
            raise ValueError(f"k_active must be in [1, block_size], got {k_active}")
        return BSBC(
            name="bsbc",
            dimensions=dimensions,
            block_size=block_size,
            k_active=k_active,
        )

    @jax.jit
    def bind(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Bind using XOR (same as BSC)."""
        return F.bind_bsc(x, y)

    def bundle(self, vectors: jax.Array, axis: int = 0) -> jax.Array:
        """Bundle using majority rule."""
        return F.bundle_bsc(vectors, axis=axis)

    @jax.jit
    def inverse(self, x: jax.Array) -> jax.Array:
        """Inverse is identity for XOR."""
        return F.inverse_bsc(x)

    @jax.jit
    def similarity(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Compute Hamming similarity."""
        return F.hamming_similarity(x, y)

    def random(self, key: jax.Array, shape: tuple) -> jax.Array:
        """Generate random block-sparse binary hypervectors."""
        num_blocks = self.dimensions // self.block_size

        def gen_block(key_b: jax.Array) -> jax.Array:
            perm = jax.random.permutation(key_b, self.block_size)
            block = jnp.zeros(self.block_size, dtype=jnp.bool_)
            return block.at[perm[: self.k_active]].set(True)

        batch_size = max(1, int(jnp.prod(jnp.array(shape))) // self.dimensions)
        keys = jax.random.split(key, batch_size * num_blocks + 1)[1:]
        keys_per_hv = jnp.reshape(
            jnp.stack(keys[: batch_size * num_blocks]), (batch_size, num_blocks, 2)
        )

        def make_hv(block_keys: jax.Array) -> jax.Array:
            blocks = jax.vmap(gen_block)(block_keys)
            return jnp.reshape(blocks, (self.dimensions,))

        hvs = jax.vmap(make_hv)(keys_per_hv)

        if batch_size == 1 and shape == (self.dimensions,):
            return hvs[0]
        if batch_size == 1 and len(shape) == 1:
            return hvs[0]
        return jnp.reshape(hvs, shape)


@register_dataclass
@dataclass
class CGR(VSAModel):
    """Cyclic Group Representation (CGR).

    Integer hypervectors in Z_q with modular addition binding,
    component-wise mode bundling.
    """

    q: int = field(metadata=dict(static=True), default=8)

    @staticmethod
    def create(dimensions: int = 10000, q: int = 8) -> "CGR":
        if q < 2:
            raise ValueError(f"q must be >= 2, got {q}")
        return CGR(name="cgr", dimensions=dimensions, q=q)

    @jax.jit
    def bind(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Bind using modular addition."""
        return F.bind_cgr(x, y, self.q)

    def bundle(self, vectors: jax.Array, axis: int = 0) -> jax.Array:
        """Bundle using component-wise mode."""
        return F.bundle_cgr(vectors, self.q, axis=axis)

    @jax.jit
    def inverse(self, x: jax.Array) -> jax.Array:
        """Inverse via modular negation."""
        return F.inverse_cgr(x, self.q)

    @jax.jit
    def similarity(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Compute fraction of matching elements."""
        return F.matching_similarity(x, y)

    def random(self, key: jax.Array, shape: tuple) -> jax.Array:
        """Generate random integer hypervectors in {0, ..., q-1}."""
        return jax.random.randint(key, shape=shape, minval=0, maxval=self.q)


@register_dataclass
@dataclass
class MCR(VSAModel):
    """Modular Composite Representation (MCR).

    Integer phase vectors with modular addition binding, phasor sum bundling.
    """

    q: int = field(metadata=dict(static=True), default=64)

    @staticmethod
    def create(dimensions: int = 10000, q: int = 64) -> "MCR":
        if q < 2:
            raise ValueError(f"q must be >= 2, got {q}")
        return MCR(name="mcr", dimensions=dimensions, q=q)

    @jax.jit
    def bind(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Bind using modular addition (phase addition)."""
        return F.bind_mcr(x, y, self.q)

    def bundle(self, vectors: jax.Array, axis: int = 0) -> jax.Array:
        """Bundle using phasor sum with snap-to-grid."""
        return F.bundle_mcr(vectors, self.q, axis=axis)

    @jax.jit
    def inverse(self, x: jax.Array) -> jax.Array:
        """Inverse via modular negation (phase conjugate)."""
        return F.inverse_mcr(x, self.q)

    @jax.jit
    def similarity(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Compute phasor similarity."""
        return F.phasor_similarity(x, y, self.q)

    def random(self, key: jax.Array, shape: tuple) -> jax.Array:
        """Generate random integer hypervectors in {0, ..., q-1}."""
        return jax.random.randint(key, shape=shape, minval=0, maxval=self.q)


@register_dataclass
@dataclass
class VTB(VSAModel):
    """Vector-Derived Transformation Binding (VTB).

    Real-valued vectors with matrix multiplication binding, normalized sum bundling.
    """

    @staticmethod
    def create(dimensions: int = 10000) -> "VTB":
        n = round(dimensions**0.5)
        if n * n != dimensions:
            raise ValueError(f"VTB requires dimensions to be a perfect square, got {dimensions}")
        return VTB(name="vtb", dimensions=dimensions)

    @jax.jit
    def bind(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Bind using matrix multiplication."""
        return F.bind_vtb(x, y)

    def bundle(self, vectors: jax.Array, axis: int = 0) -> jax.Array:
        """Bundle using normalized sum."""
        return F.bundle_vtb(vectors, axis=axis)

    @jax.jit
    def inverse(self, x: jax.Array) -> jax.Array:
        """Inverse via matrix pseudoinverse."""
        return F.inverse_vtb(x)

    @jax.jit
    def similarity(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Compute cosine similarity."""
        return F.cosine_similarity(x, y)

    def random(self, key: jax.Array, shape: tuple) -> jax.Array:
        """Generate random normalized real-valued hypervectors."""
        vectors = jax.random.normal(key, shape=shape)
        norm = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
        return vectors / (norm + EPS)


def create_vsa_model(model_type: str = "map", dimensions: int = 10000) -> VSAModel:
    """Factory function to create VSA models.

    Args:
        model_type: Type of VSA model ('bsc', 'map', 'hrr', 'fhrr', 'bsbc', 'cgr', 'mcr', 'vtb')
        dimensions: Dimensionality of hypervectors (default: 10000)

    Returns:
        Initialized VSA model
    """
    models = {
        "bsc": BSC,
        "map": MAP,
        "hrr": HRR,
        "fhrr": FHRR,
        "bsbc": BSBC,
        "cgr": CGR,
        "mcr": MCR,
        "vtb": VTB,
    }

    if model_type not in models:
        raise ValueError(
            f"Unknown VSA model: {model_type}. Available models: {list(models.keys())}"
        )

    return models[model_type].create(dimensions=dimensions)  # type: ignore[attr-defined]


__all__ = [
    "VSAModel",
    "BSC",
    "MAP",
    "HRR",
    "FHRR",
    "BSBC",
    "CGR",
    "MCR",
    "VTB",
    "create_vsa_model",
]
