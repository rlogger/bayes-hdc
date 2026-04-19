# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Memory modules for Hyperdimensional Computing.

Sparse Distributed Memory (SDM), Hopfield networks, and attention-based retrieval.
"""

from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from typing import Optional

import jax
import jax.numpy as jnp

from jax_hdc import functional as F
from jax_hdc._compat import register_dataclass


@register_dataclass
@dataclass
class SparseDistributedMemory:
    """Sparse Distributed Memory (SDM) for content-addressable storage."""

    locations: jax.Array  # (num_locations, dimensions)
    contents: jax.Array  # (num_locations, dimensions)
    dimensions: int = field(metadata=dict(static=True))
    radius: float = field(metadata=dict(static=True))

    @staticmethod
    def create(
        num_locations: int,
        dimensions: int,
        radius: float = 0.0,
        key: Optional[jax.Array] = None,
    ) -> "SparseDistributedMemory":
        if key is None:
            key = jax.random.PRNGKey(0)
        locs = jax.random.normal(key, (num_locations, dimensions))
        locs = locs / (jnp.linalg.norm(locs, axis=-1, keepdims=True) + 1e-8)
        contents = jnp.zeros((num_locations, dimensions))
        return SparseDistributedMemory(
            locations=locs,
            contents=contents,
            dimensions=dimensions,
            radius=radius,
        )

    def write(self, address: jax.Array, value: jax.Array) -> "SparseDistributedMemory":
        sims = jax.vmap(lambda loc: F.cosine_similarity(address, loc))(self.locations)
        mask = sims >= (1.0 - self.radius)
        delta = mask[:, None].astype(jnp.float32) * value
        return SparseDistributedMemory(
            locations=self.locations,
            contents=self.contents + delta,
            dimensions=self.dimensions,
            radius=self.radius,
        )

    @jax.jit
    def read(self, address: jax.Array) -> jax.Array:
        sims = jax.vmap(lambda loc: F.cosine_similarity(address, loc))(self.locations)
        mask = sims >= (1.0 - self.radius)
        summed = jnp.sum(self.contents * mask[:, None], axis=0)
        norm = jnp.linalg.norm(summed) + 1e-8
        return summed / norm


@register_dataclass
@dataclass
class HopfieldMemory:
    """Modern Hopfield network for associative memory."""

    patterns: jax.Array  # (num_patterns, dimensions)
    dimensions: int = field(metadata=dict(static=True))
    beta: float = field(metadata=dict(static=True), default=1.0)

    @staticmethod
    def create(
        dimensions: int,
        beta: float = 1.0,
    ) -> "HopfieldMemory":
        return HopfieldMemory(
            patterns=jnp.zeros((0, dimensions)),
            dimensions=dimensions,
            beta=beta,
        )

    def add(self, pattern: jax.Array) -> "HopfieldMemory":
        p = pattern.reshape(-1) / (jnp.linalg.norm(pattern) + 1e-8)
        new_patterns = jnp.concatenate([self.patterns, p[None, :]], axis=0)
        return dc_replace(self, patterns=new_patterns)

    @jax.jit
    def retrieve(self, query: jax.Array) -> jax.Array:
        q = query.reshape(-1)
        if self.patterns.shape[0] == 0:
            return jnp.zeros_like(q)
        sims = jax.vmap(lambda p: F.cosine_similarity(q, p))(self.patterns)
        weights = jax.nn.softmax(self.beta * sims)
        return jnp.sum(self.patterns * weights[:, None], axis=0)


@register_dataclass
@dataclass
class AttentionMemory:
    """Attention-based retrieval with key-value storage and multi-head support."""

    keys: jax.Array
    values: jax.Array
    dimensions: int = field(metadata=dict(static=True))
    temperature: float = field(metadata=dict(static=True), default=1.0)
    num_heads: int = field(metadata=dict(static=True), default=1)

    @staticmethod
    def create(
        dimensions: int,
        temperature: float = 1.0,
        num_heads: int = 1,
    ) -> "AttentionMemory":
        if num_heads > 1 and dimensions % num_heads != 0:
            raise ValueError(
                f"dimensions ({dimensions}) must be divisible by num_heads ({num_heads})"
            )
        return AttentionMemory(
            keys=jnp.zeros((0, dimensions)),
            values=jnp.zeros((0, dimensions)),
            dimensions=dimensions,
            temperature=temperature,
            num_heads=num_heads,
        )

    def write(self, key: jax.Array, value: jax.Array) -> "AttentionMemory":
        k = key.reshape(1, -1)
        v = value.reshape(1, -1)
        return dc_replace(
            self,
            keys=jnp.concatenate([self.keys, k], axis=0),
            values=jnp.concatenate([self.values, v], axis=0),
        )

    def write_batch(self, keys: jax.Array, values: jax.Array) -> "AttentionMemory":
        return dc_replace(
            self,
            keys=jnp.concatenate([self.keys, keys], axis=0),
            values=jnp.concatenate([self.values, values], axis=0),
        )

    @jax.jit
    def retrieve(self, query: jax.Array) -> jax.Array:
        q = query.reshape(-1)
        if self.keys.shape[0] == 0:
            return jnp.zeros(self.dimensions)

        if self.num_heads == 1:
            scale = (self.dimensions**0.5) * self.temperature
            scores = self.keys @ q / scale
            weights = jax.nn.softmax(scores)
            return jnp.sum(self.values * weights[:, None], axis=0)
        else:
            head_dim = self.dimensions // self.num_heads
            q_heads = q.reshape(self.num_heads, head_dim)
            k_heads = self.keys.reshape(-1, self.num_heads, head_dim)
            v_heads = self.values.reshape(-1, self.num_heads, head_dim)

            scale = (head_dim**0.5) * self.temperature
            scores = jnp.einsum("hd,nhd->hn", q_heads, k_heads) / scale
            weights = jax.nn.softmax(scores, axis=-1)
            result = jnp.einsum("hn,nhd->hd", weights, v_heads)
            return result.reshape(-1)

    @jax.jit
    def retrieve_with_weights(self, query: jax.Array) -> tuple:
        q = query.reshape(-1)
        if self.keys.shape[0] == 0:
            return jnp.zeros(self.dimensions), jnp.array([])

        scale = (self.dimensions**0.5) * self.temperature
        scores = self.keys @ q / scale
        weights = jax.nn.softmax(scores)
        result = jnp.sum(self.values * weights[:, None], axis=0)
        return result, weights


__all__ = ["SparseDistributedMemory", "HopfieldMemory", "AttentionMemory"]
