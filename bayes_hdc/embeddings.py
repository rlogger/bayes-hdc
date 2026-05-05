# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Encoders for transforming data into hypervectors.

This module provides various encoding strategies to transform different types
of data (discrete features, continuous values, images) into hypervectors.
"""

from dataclasses import dataclass, field
from typing import Optional, Union

import jax
import jax.numpy as jnp

from bayes_hdc import functional as F
from bayes_hdc._compat import register_dataclass
from bayes_hdc.constants import EPS
from bayes_hdc.vsa import VSAModel, create_vsa_model


@register_dataclass
@dataclass
class RandomEncoder:
    """Encoder using random hypervectors for discrete features.

    Each unique feature value is mapped to a random hypervector from a codebook.
    Multiple features are bundled together to form the final representation.
    """

    # Data fields (traced by JAX)
    codebook: jax.Array  # Shape: (num_features, num_values, dimensions)

    # Metadata fields (static, not traced)
    num_features: int = field(metadata=dict(static=True))
    num_values: int = field(metadata=dict(static=True))
    dimensions: int = field(metadata=dict(static=True))
    vsa_model_name: str = field(metadata=dict(static=True), default="map")

    @staticmethod
    def create(
        num_features: int,
        num_values: int,
        dimensions: int = 10000,
        vsa_model: Union[str, VSAModel] = "map",
        key: Optional[jax.Array] = None,
    ) -> "RandomEncoder":
        """Create a random encoder.

        Args:
            num_features: Number of features to encode
            num_values: Number of possible values per feature
            dimensions: Dimensionality of hypervectors (default: 10000)
            vsa_model: VSA model to use ('bsc', 'map', 'hrr', 'fhrr') or VSAModel instance
            key: JAX random key (default: PRNGKey(0))

        Returns:
            Initialized RandomEncoder
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        # Handle both string and VSAModel
        if isinstance(vsa_model, str):
            vsa_model_name = vsa_model
            vsa = create_vsa_model(vsa_model, dimensions)
        else:
            vsa_model_name = vsa_model.name
            vsa = vsa_model

        # Generate random codebook
        codebook = vsa.random(key, shape=(num_features, num_values, dimensions))

        return RandomEncoder(
            codebook=codebook,
            num_features=num_features,
            num_values=num_values,
            dimensions=dimensions,
            vsa_model_name=vsa_model_name,
        )

    @jax.jit
    def encode(self, indices: jax.Array) -> jax.Array:
        """Encode discrete features as hypervectors.

        Args:
            indices: Feature indices of shape (num_features,) with values in [0, num_values).
                     Out-of-bounds indices are clamped to valid range.

        Returns:
            Encoded hypervector of shape (dimensions,)
        """
        # Clamp indices to valid range to avoid out-of-bounds access
        indices = jnp.clip(indices.astype(jnp.int32), 0, self.num_values - 1)
        # Select hypervector for each feature
        # codebook[i, indices[i]] selects the hypervector for feature i with value indices[i]
        selected = jax.vmap(lambda i: self.codebook[i, indices[i]])(jnp.arange(self.num_features))

        # Bundle all feature hypervectors
        if self.vsa_model_name == "bsc":
            return F.bundle_bsc(selected, axis=0)
        else:
            return F.bundle_map(selected, axis=0)

    @jax.jit
    def encode_batch(self, indices: jax.Array) -> jax.Array:
        """Encode a batch of samples.

        Args:
            indices: Batch of feature indices of shape (batch_size, num_features)

        Returns:
            Encoded hypervectors of shape (batch_size, dimensions)
        """
        return jax.vmap(self.encode)(indices)


@register_dataclass
@dataclass
class LevelEncoder:
    """Encoder for continuous values using level hypervectors.

    Continuous values are encoded by interpolating between level hypervectors,
    creating a smooth representation where similar values map to similar hypervectors.
    """

    # Data fields
    level_hvs: jax.Array  # Shape: (num_levels, dimensions)

    # Metadata fields
    num_levels: int = field(metadata=dict(static=True))
    dimensions: int = field(metadata=dict(static=True))
    min_value: float = field(metadata=dict(static=True))
    max_value: float = field(metadata=dict(static=True))
    vsa_model_name: str = field(metadata=dict(static=True), default="map")
    encoding_type: str = field(metadata=dict(static=True), default="linear")

    @staticmethod
    def create(
        num_levels: int = 100,
        dimensions: int = 10000,
        min_value: float = 0.0,
        max_value: float = 1.0,
        vsa_model: Union[str, VSAModel] = "map",
        encoding_type: str = "linear",
        key: Optional[jax.Array] = None,
    ) -> "LevelEncoder":
        """Create a level encoder.

        Args:
            num_levels: Number of levels for discretization (default: 100)
            dimensions: Dimensionality of hypervectors (default: 10000)
            min_value: Minimum value of the range (default: 0.0)
            max_value: Maximum value of the range (default: 1.0)
            vsa_model: VSA model to use ('bsc', 'map', 'hrr', 'fhrr')
            encoding_type: 'linear' or 'circular' (default: 'linear')
            key: JAX random key

        Returns:
            Initialized LevelEncoder
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        # Handle both string and VSAModel
        if isinstance(vsa_model, str):
            vsa_model_name = vsa_model
            vsa = create_vsa_model(vsa_model, dimensions)
        else:
            vsa_model_name = vsa_model.name
            vsa = vsa_model

        if min_value >= max_value:
            raise ValueError(
                f"min_value ({min_value}) must be less than max_value ({max_value}). "
                "Use a non-empty range for level encoding."
            )

        # Generate random level hypervectors
        level_hvs = vsa.random(key, shape=(num_levels, dimensions))

        return LevelEncoder(
            level_hvs=level_hvs,
            num_levels=num_levels,
            dimensions=dimensions,
            min_value=min_value,
            max_value=max_value,
            vsa_model_name=vsa_model_name,
            encoding_type=encoding_type,
        )

    @jax.jit
    def encode(self, value: Union[float, jax.Array]) -> jax.Array:
        """Encode a continuous value as a hypervector.

        Args:
            value: Continuous value to encode (scalar or array)

        Returns:
            Encoded hypervector of shape (dimensions,) or batch shape + (dimensions,)
        """
        # Normalize value to [0, num_levels - 1] (range validated at create time)
        value_range = self.max_value - self.min_value
        normalized = (value - self.min_value) / jnp.maximum(value_range, EPS)
        normalized = jnp.clip(normalized, 0.0, 1.0)
        level_pos = normalized * (self.num_levels - 1)

        # Get lower and upper level indices
        lower_idx = jnp.floor(level_pos).astype(jnp.int32)
        upper_idx = jnp.ceil(level_pos).astype(jnp.int32)

        # Interpolation weight
        weight = level_pos - lower_idx

        # Get level hypervectors
        lower_hv = self.level_hvs[lower_idx]
        upper_hv = self.level_hvs[upper_idx]

        # Linear interpolation for real-valued models
        if self.vsa_model_name in ["map", "hrr", "fhrr"]:
            # Weighted combination
            encoded = (1 - weight[..., None]) * lower_hv + weight[..., None] * upper_hv
            # Normalize
            norm = jnp.linalg.norm(encoded, axis=-1, keepdims=True)
            return encoded / (norm + EPS)
        else:  # BSC
            # For binary, use threshold-based selection
            return jnp.where(weight[..., None] > 0.5, upper_hv, lower_hv)

    @jax.jit
    def encode_batch(self, values: jax.Array) -> jax.Array:
        """Encode a batch of continuous values.

        Args:
            values: Batch of values of shape (batch_size,) or (batch_size, num_features)

        Returns:
            Encoded hypervectors
        """
        return jax.vmap(self.encode)(values)


@register_dataclass
@dataclass
class ProjectionEncoder:
    """Encoder using random projection for high-dimensional data.

    Projects high-dimensional input data into hypervector space using a
    random projection matrix. Useful for images, text embeddings, etc.
    """

    # Data fields
    projection_matrix: jax.Array  # Shape: (input_dim, dimensions)

    # Metadata fields
    input_dim: int = field(metadata=dict(static=True))
    dimensions: int = field(metadata=dict(static=True))
    vsa_model_name: str = field(metadata=dict(static=True), default="map")

    @staticmethod
    def create(
        input_dim: int,
        dimensions: int = 10000,
        vsa_model: Union[str, VSAModel] = "map",
        key: Optional[jax.Array] = None,
    ) -> "ProjectionEncoder":
        """Create a projection encoder.

        Args:
            input_dim: Dimensionality of input data
            dimensions: Dimensionality of hypervectors (default: 10000)
            vsa_model: VSA model to use ('bsc', 'map', 'hrr', 'fhrr')
            key: JAX random key

        Returns:
            Initialized ProjectionEncoder
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        # Handle both string and VSAModel
        if isinstance(vsa_model, str):
            vsa_model_name = vsa_model
        else:
            vsa_model_name = vsa_model.name

        # Create random projection matrix (normalized)
        projection_matrix = jax.random.normal(key, shape=(input_dim, dimensions))
        projection_matrix = projection_matrix / jnp.sqrt(input_dim)

        return ProjectionEncoder(
            projection_matrix=projection_matrix,
            input_dim=input_dim,
            dimensions=dimensions,
            vsa_model_name=vsa_model_name,
        )

    @jax.jit
    def encode(self, x: jax.Array) -> jax.Array:
        """Encode input data as a hypervector.

        Args:
            x: Input data of shape (input_dim,)

        Returns:
            Encoded hypervector of shape (dimensions,)
        """
        # Random projection
        projected = jnp.dot(x, self.projection_matrix)

        # Apply activation based on VSA model
        if self.vsa_model_name == "bsc":
            # Threshold for binary
            return projected > 0
        else:
            # Normalize for real-valued
            norm = jnp.linalg.norm(projected)
            return projected / (norm + EPS)

    @jax.jit
    def encode_batch(self, x: jax.Array) -> jax.Array:
        """Encode a batch of inputs.

        Args:
            x: Batch of inputs of shape (batch_size, input_dim)

        Returns:
            Encoded hypervectors of shape (batch_size, dimensions)
        """
        return jax.vmap(self.encode)(x)


@register_dataclass
@dataclass
class KernelEncoder:
    """Encoder using RBF kernel approximation (Random Fourier Features).

    Approximates the RBF kernel k(x,y) = exp(-gamma ||x-y||^2) via random
    Fourier features, mapping input to a hypervector space that preserves
    kernel similarity.
    """

    omega: jax.Array  # Shape: (input_dim, n_features)
    bias: jax.Array  # Shape: (n_features,)

    input_dim: int = field(metadata=dict(static=True))
    dimensions: int = field(metadata=dict(static=True))
    gamma: float = field(metadata=dict(static=True))
    vsa_model_name: str = field(metadata=dict(static=True), default="map")

    @staticmethod
    def create(
        input_dim: int,
        dimensions: int = 10000,
        gamma: float = 1.0,
        vsa_model: Union[str, VSAModel] = "map",
        key: Optional[jax.Array] = None,
    ) -> "KernelEncoder":
        """Create a kernel encoder.

        Args:
            input_dim: Dimensionality of input data
            dimensions: Dimensionality of output hypervectors
            gamma: RBF kernel scale parameter (1 / 2*sigma^2)
            vsa_model: VSA model ('map', 'hrr', 'fhrr' for real-valued)
            key: JAX random key

        Returns:
            Initialized KernelEncoder
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        if isinstance(vsa_model, str):
            vsa_model_name = vsa_model
        else:
            vsa_model_name = vsa_model.name

        # Random Fourier features: omega ~ N(0, 2*gamma I), bias ~ U(0, 2*pi)
        key_omega, key_bias = jax.random.split(key)
        omega = jax.random.normal(key_omega, (input_dim, dimensions)) * jnp.sqrt(2.0 * gamma)
        bias = jax.random.uniform(key_bias, (dimensions,), minval=0.0, maxval=2.0 * jnp.pi)

        return KernelEncoder(
            omega=omega,
            bias=bias,
            input_dim=input_dim,
            dimensions=dimensions,
            gamma=gamma,
            vsa_model_name=vsa_model_name,
        )

    @jax.jit
    def encode(self, x: jax.Array) -> jax.Array:
        """Encode input using RBF kernel approximation."""
        proj = jnp.dot(x, self.omega) + self.bias
        features = jnp.cos(proj) * jnp.sqrt(2.0 / self.dimensions)

        if self.vsa_model_name == "bsc":
            return features > 0
        norm = jnp.linalg.norm(features) + EPS
        return features / norm

    @jax.jit
    def encode_batch(self, x: jax.Array) -> jax.Array:
        """Encode a batch of inputs."""
        return jax.vmap(self.encode)(x)


@register_dataclass
@dataclass
class GraphEncoder:
    """Encoder for graph structures (nodes and edges).

    Encodes a graph by assigning random hypervectors to nodes and
    bundling bound node pairs for edges. Graph = bundle of edge HVs.
    """

    node_embeddings: jax.Array  # Shape: (num_nodes, dimensions)

    num_nodes: int = field(metadata=dict(static=True))
    dimensions: int = field(metadata=dict(static=True))
    vsa_model_name: str = field(metadata=dict(static=True), default="map")

    @staticmethod
    def create(
        num_nodes: int,
        dimensions: int = 10000,
        vsa_model: Union[str, VSAModel] = "map",
        key: Optional[jax.Array] = None,
    ) -> "GraphEncoder":
        """Create a graph encoder.

        Args:
            num_nodes: Maximum number of nodes
            dimensions: Hypervector dimensionality
            vsa_model: VSA model for real-valued graphs
            key: JAX random key

        Returns:
            Initialized GraphEncoder
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        if isinstance(vsa_model, str):
            vsa_model_name = vsa_model
            vsa = create_vsa_model(vsa_model, dimensions)
        else:
            vsa_model_name = vsa_model.name
            vsa = vsa_model

        node_embeddings = vsa.random(key, (num_nodes, dimensions))

        return GraphEncoder(
            node_embeddings=node_embeddings,
            num_nodes=num_nodes,
            dimensions=dimensions,
            vsa_model_name=vsa_model_name,
        )

    def encode_edges(self, edges: jax.Array) -> jax.Array:
        """Encode graph as bundle of bound edge pairs.

        Args:
            edges: Array of shape (num_edges, 2) with node indices in [0, num_nodes).
                   Out-of-bounds indices are clamped to valid range.

        Returns:
            Graph hypervector of shape (dimensions,)
        """
        edge_hvs = []
        for i in range(edges.shape[0]):
            # Clamp node indices to valid range to avoid out-of-bounds access
            u = int(jnp.clip(edges[i, 0], 0, self.num_nodes - 1))
            v = int(jnp.clip(edges[i, 1], 0, self.num_nodes - 1))
            bound = F.bind_map(
                self.node_embeddings[u],
                F.permute(self.node_embeddings[v], 1),
            )
            edge_hvs.append(bound)
        return F.bundle_map(jnp.stack(edge_hvs), axis=0)


@register_dataclass
@dataclass
class TokenEncoder:
    r"""Encode token-ID sequences as positional hypervectors.

    A vocabulary-sized codebook of random unit-norm hypervectors:
    each token ID :math:`t \in [0, V)` is assigned a fixed random
    vector :math:`c_t \in \mathbb{R}^d`. A sequence of token IDs is
    encoded by looking up each token's vector and applying the
    standard permute-bundle sequence encoding (Sahlgren et al. 2008;
    Kanerva 2009) — flat for short sequences, hierarchical
    (:class:`~bayes_hdc.structures.HierarchicalSequence`) for long
    horizons where the per-position SNR of the flat construction
    becomes the bottleneck.

    Tokenizer-agnostic: takes integer token IDs from any source
    (HuggingFace tokenizers, SentencePiece, tiktoken, BPE, character
    indices). Tokenisation itself is the caller's responsibility::

        # HuggingFace example (the library is a runtime dep of the user, not us):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
        ids = tok.encode("the quick brown fox", return_tensors="jax")[0]
        encoder = TokenEncoder.create(vocab_size=tok.vocab_size, dimensions=10000)
        seq_hv = encoder.encode(ids)

    Attributes:
        codebook: Codebook of shape ``(vocab_size, dimensions)``.
            Stable across calls; persist or save with
            :func:`jax.numpy.save` for reproducible encodings.
        vocab_size: Token-vocabulary cardinality :math:`V`.
        dimensions: Hypervector dimension :math:`d`.
    """

    codebook: jax.Array  # (vocab_size, d)
    vocab_size: int = field(metadata=dict(static=True))
    dimensions: int = field(metadata=dict(static=True))

    @staticmethod
    def create(
        vocab_size: int,
        dimensions: int = 10_000,
        vsa_model: Union[str, VSAModel] = "map",
        key: Optional[jax.Array] = None,
    ) -> "TokenEncoder":
        """Build a fresh codebook of ``vocab_size`` random hypervectors.

        Args:
            vocab_size: Token-vocabulary cardinality.
            dimensions: Hypervector dimension. Defaults to 10 000;
                drop to ``4096`` for tighter memory at modest
                capacity loss.
            vsa_model: ``"map"`` (default — real-valued, unit-norm),
                ``"hrr"``, ``"fhrr"``, etc. For HDC text pipelines
                MAP is the conventional choice; the codebook is
                drawn from the chosen model and L2-normalised so
                ``permute_bundle_sequence`` retrieval works
                downstream.
            key: ``jax.random.PRNGKey``. Defaults to ``PRNGKey(0)``;
                pass an explicit key for reproducible runs.

        Returns:
            A frozen :class:`TokenEncoder` ready for ``encode()``.
        """
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
        if dimensions < 1:
            raise ValueError(f"dimensions must be >= 1, got {dimensions}")
        if key is None:
            key = jax.random.PRNGKey(0)

        if isinstance(vsa_model, str):
            vsa = create_vsa_model(vsa_model, dimensions)
        else:
            vsa = vsa_model
        codebook = vsa.random(key, shape=(vocab_size, dimensions))

        # L2-normalise per-row so each token's hypervector lives on
        # the unit sphere — required for the permute-bundle SNR
        # analysis the sequence encoders rely on.
        norms = jnp.linalg.norm(codebook, axis=-1, keepdims=True)
        codebook = codebook / (norms + EPS)

        return TokenEncoder(
            codebook=codebook,
            vocab_size=vocab_size,
            dimensions=dimensions,
        )

    @jax.jit
    def lookup(self, token_id: jax.Array) -> jax.Array:
        """Return the hypervector for a single token ID."""
        return self.codebook[token_id]

    @jax.jit
    def lookup_batch(self, token_ids: jax.Array) -> jax.Array:
        """Return the per-token hypervectors for a sequence of IDs.

        Args:
            token_ids: Integer IDs of shape ``(T,)``.

        Returns:
            Hypervectors of shape ``(T, d)``.
        """
        return self.codebook[token_ids]

    @jax.jit
    def encode(self, token_ids: jax.Array) -> jax.Array:
        """Flat permute-bundle encoding of a token sequence.

        Returns the flat :class:`~bayes_hdc.structures.Sequence`
        representation: ``Σ_i P^{T-1-i} c_{t_i}``. For long sequences
        (``T ≳ 200`` at ``d = 4 096``) prefer
        :meth:`encode_hierarchical`; see ``BENCHMARKS.md`` for the
        capacity table.
        """
        items = self.codebook[token_ids]
        return F.bundle_sequence(items)

    def encode_hierarchical(
        self,
        token_ids: jax.Array,
        chunk_size: int = 16,
    ):
        """Two-level chunked encoding for long-horizon sequences.

        Returns a :class:`~bayes_hdc.structures.HierarchicalSequence`
        rather than a raw hypervector, because retrieval requires the
        cached chunk codebook stored on that object. ``get(i)`` on
        the result returns the (still-noisy) item hypervector at
        position ``i``; clean against ``self.codebook`` for symbolic
        recovery.
        """
        # Local import to avoid the structures ↔ embeddings import cycle.
        from bayes_hdc.structures import HierarchicalSequence

        items = self.codebook[token_ids]
        return HierarchicalSequence.from_vectors(items, chunk_size=chunk_size)


__all__ = [
    "RandomEncoder",
    "LevelEncoder",
    "ProjectionEncoder",
    "KernelEncoder",
    "GraphEncoder",
    "TokenEncoder",
]
