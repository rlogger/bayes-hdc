# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Tests for embedding encoders."""

import jax
import jax.numpy as jnp
import pytest

from bayes_hdc.embeddings import (
    GraphEncoder,
    KernelEncoder,
    LevelEncoder,
    ProjectionEncoder,
    RandomEncoder,
    TokenEncoder,
)


class TestRandomEncoder:
    """Tests for RandomEncoder."""

    def test_creation_with_defaults(self):
        """Test RandomEncoder creation with default parameters."""
        encoder = RandomEncoder.create(
            num_features=5, num_values=10, dimensions=100, key=jax.random.PRNGKey(42)
        )

        assert encoder.num_features == 5
        assert encoder.num_values == 10
        assert encoder.dimensions == 100
        assert encoder.vsa_model_name == "map"
        assert encoder.codebook.shape == (5, 10, 100)

    def test_creation_with_bsc(self):
        """Test RandomEncoder creation with BSC model."""
        encoder = RandomEncoder.create(
            num_features=3, num_values=5, dimensions=50, vsa_model="bsc", key=jax.random.PRNGKey(42)
        )

        assert encoder.vsa_model_name == "bsc"
        assert encoder.codebook.dtype == jnp.bool_

    def test_creation_with_map(self):
        """Test RandomEncoder creation with MAP model."""
        encoder = RandomEncoder.create(
            num_features=3, num_values=5, dimensions=50, vsa_model="map", key=jax.random.PRNGKey(42)
        )

        assert encoder.vsa_model_name == "map"
        assert encoder.codebook.dtype == jnp.float32

    def test_encode_single_sample(self):
        """Test encoding a single sample."""
        encoder = RandomEncoder.create(
            num_features=3, num_values=5, dimensions=100, key=jax.random.PRNGKey(42)
        )

        indices = jnp.array([0, 2, 4])
        encoded = encoder.encode(indices)

        assert encoded.shape == (100,)
        assert jnp.isfinite(encoded).all()

    def test_encode_batch(self):
        """Test encoding a batch of samples."""
        encoder = RandomEncoder.create(
            num_features=3, num_values=5, dimensions=100, key=jax.random.PRNGKey(42)
        )

        batch = jnp.array([[0, 2, 4], [1, 3, 2], [4, 1, 0]])
        encoded = encoder.encode_batch(batch)

        assert encoded.shape == (3, 100)
        assert jnp.isfinite(encoded).all()

    def test_different_indices_produce_different_encodings(self):
        """Test that different feature indices produce different encodings."""
        encoder = RandomEncoder.create(
            num_features=3, num_values=5, dimensions=1000, key=jax.random.PRNGKey(42)
        )

        indices1 = jnp.array([0, 0, 0])
        indices2 = jnp.array([1, 1, 1])

        encoded1 = encoder.encode(indices1)
        encoded2 = encoder.encode(indices2)

        # Encoded vectors should be different
        similarity = jnp.dot(encoded1, encoded2) / (
            jnp.linalg.norm(encoded1) * jnp.linalg.norm(encoded2)
        )
        assert similarity < 0.5

    def test_encode_reproducibility(self):
        """Test that encoding is reproducible with same encoder."""
        encoder = RandomEncoder.create(
            num_features=3, num_values=5, dimensions=100, key=jax.random.PRNGKey(42)
        )

        indices = jnp.array([1, 2, 3])
        encoded1 = encoder.encode(indices)
        encoded2 = encoder.encode(indices)

        assert jnp.allclose(encoded1, encoded2)

    def test_encode_clamps_out_of_bounds_indices(self):
        """Test that out-of-bounds indices are clamped to valid range."""
        encoder = RandomEncoder.create(
            num_features=3, num_values=5, dimensions=100, key=jax.random.PRNGKey(42)
        )
        # Indices outside [0, 4] should be clamped
        indices = jnp.array([-1, 10, 100])
        encoded = encoder.encode(indices)
        assert encoded.shape == (100,)
        assert jnp.isfinite(encoded).all()

    def test_creation_with_vsa_model_instance(self):
        """Test RandomEncoder creation with VSAModel instance."""
        from bayes_hdc.vsa import MAP

        vsa = MAP.create(dimensions=100)
        encoder = RandomEncoder.create(
            num_features=5,
            num_values=10,
            dimensions=100,
            vsa_model=vsa,
            key=jax.random.PRNGKey(42),
        )

        assert encoder.vsa_model_name == "map"
        encoded = encoder.encode(jnp.array([0, 1, 2, 3, 4]))
        assert encoded.shape == (100,)


class TestLevelEncoder:
    """Tests for LevelEncoder."""

    def test_creation_with_defaults(self):
        """Test LevelEncoder creation with default parameters."""
        encoder = LevelEncoder.create(num_levels=50, dimensions=100, key=jax.random.PRNGKey(42))

        assert encoder.num_levels == 50
        assert encoder.dimensions == 100
        assert encoder.min_value == 0.0
        assert encoder.max_value == 1.0
        assert encoder.vsa_model_name == "map"
        assert encoder.encoding_type == "linear"
        assert encoder.level_hvs.shape == (50, 100)

    def test_creation_rejects_min_equals_max(self):
        """Test that LevelEncoder rejects min_value >= max_value."""
        with pytest.raises(ValueError, match="min_value.*must be less than max_value"):
            LevelEncoder.create(
                num_levels=50,
                dimensions=100,
                min_value=1.0,
                max_value=1.0,
                key=jax.random.PRNGKey(42),
            )
        with pytest.raises(ValueError, match="min_value.*must be less than max_value"):
            LevelEncoder.create(
                num_levels=50,
                dimensions=100,
                min_value=5.0,
                max_value=1.0,
                key=jax.random.PRNGKey(42),
            )

    def test_creation_with_custom_range(self):
        """Test LevelEncoder with custom value range."""
        encoder = LevelEncoder.create(
            num_levels=100,
            dimensions=100,
            min_value=-10.0,
            max_value=10.0,
            key=jax.random.PRNGKey(42),
        )

        assert encoder.min_value == -10.0
        assert encoder.max_value == 10.0

    def test_encode_scalar(self):
        """Test encoding a scalar value."""
        encoder = LevelEncoder.create(
            num_levels=100, dimensions=100, min_value=0.0, max_value=1.0, key=jax.random.PRNGKey(42)
        )

        value = 0.5
        encoded = encoder.encode(value)

        assert encoded.shape == (100,)
        assert jnp.isfinite(encoded).all()

    def test_encode_batch(self):
        """Test encoding a batch of values."""
        encoder = LevelEncoder.create(num_levels=100, dimensions=100, key=jax.random.PRNGKey(42))

        values = jnp.array([0.1, 0.5, 0.9])
        encoded = encoder.encode_batch(values)

        assert encoded.shape == (3, 100)
        assert jnp.isfinite(encoded).all()

    def test_similar_values_produce_similar_encodings(self):
        """Test that similar values produce similar encodings."""
        encoder = LevelEncoder.create(num_levels=100, dimensions=1000, key=jax.random.PRNGKey(42))

        value1 = 0.5
        value2 = 0.51
        value3 = 0.9

        encoded1 = encoder.encode(value1)
        encoded2 = encoder.encode(value2)
        encoded3 = encoder.encode(value3)

        # Similar values should have high similarity
        sim_12 = jnp.dot(encoded1, encoded2)
        sim_13 = jnp.dot(encoded1, encoded3)

        assert sim_12 > sim_13

    def test_value_clamping(self):
        """Test that values outside range are clamped."""
        encoder = LevelEncoder.create(
            num_levels=100, dimensions=100, min_value=0.0, max_value=1.0, key=jax.random.PRNGKey(42)
        )

        # Test values outside range
        encoded_below = encoder.encode(-0.5)
        encoded_above = encoder.encode(1.5)

        # Should clamp to min and max
        encoded_min = encoder.encode(0.0)
        encoded_max = encoder.encode(1.0)

        assert jnp.allclose(encoded_below, encoded_min)
        assert jnp.allclose(encoded_above, encoded_max)

    def test_encode_with_bsc(self):
        """Test LevelEncoder with BSC model."""
        encoder = LevelEncoder.create(
            num_levels=50, dimensions=100, vsa_model="bsc", key=jax.random.PRNGKey(42)
        )

        value = 0.7
        encoded = encoder.encode(value)

        assert encoder.vsa_model_name == "bsc"
        assert encoded.shape == (100,)
        # BSC output should be boolean
        assert encoded.dtype == jnp.bool_

    def test_encode_with_map(self):
        """Test LevelEncoder with MAP model."""
        encoder = LevelEncoder.create(
            num_levels=50, dimensions=100, vsa_model="map", key=jax.random.PRNGKey(42)
        )

        value = 0.7
        encoded = encoder.encode(value)

        assert encoder.vsa_model_name == "map"
        assert encoded.dtype == jnp.float32
        # Should be normalized
        norm = jnp.linalg.norm(encoded)
        assert jnp.allclose(norm, 1.0, atol=1e-5)

    def test_encoding_reproducibility(self):
        """Test that encoding is reproducible."""
        encoder = LevelEncoder.create(num_levels=100, dimensions=100, key=jax.random.PRNGKey(42))

        value = 0.75
        encoded1 = encoder.encode(value)
        encoded2 = encoder.encode(value)

        assert jnp.allclose(encoded1, encoded2)


class TestProjectionEncoder:
    """Tests for ProjectionEncoder."""

    def test_creation_with_defaults(self):
        """Test ProjectionEncoder creation with default parameters."""
        encoder = ProjectionEncoder.create(input_dim=50, dimensions=100, key=jax.random.PRNGKey(42))

        assert encoder.input_dim == 50
        assert encoder.dimensions == 100
        assert encoder.vsa_model_name == "map"
        assert encoder.projection_matrix.shape == (50, 100)

    def test_creation_with_bsc(self):
        """Test ProjectionEncoder with BSC model."""
        encoder = ProjectionEncoder.create(
            input_dim=50, dimensions=100, vsa_model="bsc", key=jax.random.PRNGKey(42)
        )

        assert encoder.vsa_model_name == "bsc"

    def test_encode_single_input(self):
        """Test encoding a single input."""
        encoder = ProjectionEncoder.create(input_dim=50, dimensions=100, key=jax.random.PRNGKey(42))

        x = jax.random.normal(jax.random.PRNGKey(0), (50,))
        encoded = encoder.encode(x)

        assert encoded.shape == (100,)
        assert jnp.isfinite(encoded).all()

    def test_encode_batch(self):
        """Test encoding a batch of inputs."""
        encoder = ProjectionEncoder.create(input_dim=50, dimensions=100, key=jax.random.PRNGKey(42))

        batch = jax.random.normal(jax.random.PRNGKey(0), (10, 50))
        encoded = encoder.encode_batch(batch)

        assert encoded.shape == (10, 100)
        assert jnp.isfinite(encoded).all()

    def test_encode_with_map_produces_normalized_output(self):
        """Test that MAP encoding produces normalized output."""
        encoder = ProjectionEncoder.create(
            input_dim=50, dimensions=100, vsa_model="map", key=jax.random.PRNGKey(42)
        )

        x = jax.random.normal(jax.random.PRNGKey(0), (50,))
        encoded = encoder.encode(x)

        norm = jnp.linalg.norm(encoded)
        assert jnp.allclose(norm, 1.0, atol=1e-5)

    def test_encode_with_bsc_produces_binary_output(self):
        """Test that BSC encoding produces binary output."""
        encoder = ProjectionEncoder.create(
            input_dim=50, dimensions=100, vsa_model="bsc", key=jax.random.PRNGKey(42)
        )

        x = jax.random.normal(jax.random.PRNGKey(0), (50,))
        encoded = encoder.encode(x)

        assert encoded.dtype == jnp.bool_

    def test_similar_inputs_produce_similar_encodings(self):
        """Test that similar inputs produce similar encodings."""
        encoder = ProjectionEncoder.create(
            input_dim=50, dimensions=1000, key=jax.random.PRNGKey(42)
        )

        x1 = jax.random.normal(jax.random.PRNGKey(0), (50,))
        x2 = x1 + 0.01 * jax.random.normal(jax.random.PRNGKey(1), (50,))
        x3 = jax.random.normal(jax.random.PRNGKey(2), (50,))

        encoded1 = encoder.encode(x1)
        encoded2 = encoder.encode(x2)
        encoded3 = encoder.encode(x3)

        sim_12 = jnp.dot(encoded1, encoded2)
        sim_13 = jnp.dot(encoded1, encoded3)

        # Similar inputs should have higher similarity
        assert sim_12 > sim_13

    def test_encoding_reproducibility(self):
        """Test that encoding is reproducible."""
        encoder = ProjectionEncoder.create(input_dim=50, dimensions=100, key=jax.random.PRNGKey(42))

        x = jax.random.normal(jax.random.PRNGKey(0), (50,))
        encoded1 = encoder.encode(x)
        encoded2 = encoder.encode(x)

        assert jnp.allclose(encoded1, encoded2)

    def test_projection_matrix_normalization(self):
        """Test that projection matrix is properly normalized."""
        encoder = ProjectionEncoder.create(input_dim=50, dimensions=100, key=jax.random.PRNGKey(42))

        # Check that columns have approximately equal variance
        col_norms = jnp.linalg.norm(encoder.projection_matrix, axis=0)
        # All column norms should be similar due to normalization
        assert jnp.std(col_norms) < 0.5


class TestKernelEncoder:
    """Tests for KernelEncoder."""

    def test_creation_with_defaults(self):
        """Test KernelEncoder creation."""
        encoder = KernelEncoder.create(input_dim=20, dimensions=100, key=jax.random.PRNGKey(42))
        assert encoder.input_dim == 20
        assert encoder.dimensions == 100
        assert encoder.gamma == 1.0
        assert encoder.omega.shape == (20, 100)
        assert encoder.bias.shape == (100,)

    def test_encode_single_input(self):
        """Test encoding a single input."""
        encoder = KernelEncoder.create(input_dim=10, dimensions=100, key=jax.random.PRNGKey(42))
        x = jax.random.normal(jax.random.PRNGKey(0), (10,))
        encoded = encoder.encode(x)
        assert encoded.shape == (100,)
        assert jnp.isfinite(encoded).all()

    def test_encode_batch(self):
        """Test batch encoding."""
        encoder = KernelEncoder.create(input_dim=10, dimensions=100, key=jax.random.PRNGKey(42))
        x = jax.random.normal(jax.random.PRNGKey(0), (5, 10))
        encoded = encoder.encode_batch(x)
        assert encoded.shape == (5, 100)

    def test_encode_with_bsc_produces_binary(self):
        """Test KernelEncoder with BSC produces binary output."""
        encoder = KernelEncoder.create(
            input_dim=10, dimensions=100, vsa_model="bsc", key=jax.random.PRNGKey(42)
        )
        x = jax.random.normal(jax.random.PRNGKey(0), (10,))
        encoded = encoder.encode(x)
        assert encoded.dtype == jnp.bool_

    def test_creation_with_vsa_model_instance(self):
        """Test KernelEncoder creation with VSAModel instance."""
        from bayes_hdc.vsa import MAP

        vsa = MAP.create(dimensions=100)
        encoder = KernelEncoder.create(
            input_dim=10, dimensions=100, vsa_model=vsa, key=jax.random.PRNGKey(42)
        )
        assert encoder.vsa_model_name == "map"


class TestGraphEncoder:
    """Tests for GraphEncoder."""

    def test_creation_with_defaults(self):
        """Test GraphEncoder creation."""
        encoder = GraphEncoder.create(num_nodes=10, dimensions=100, key=jax.random.PRNGKey(42))
        assert encoder.num_nodes == 10
        assert encoder.dimensions == 100
        assert encoder.node_embeddings.shape == (10, 100)

    def test_encode_edges(self):
        """Test encoding graph edges."""
        encoder = GraphEncoder.create(num_nodes=5, dimensions=100, key=jax.random.PRNGKey(42))
        edges = jnp.array([[0, 1], [1, 2], [2, 0]])
        hv = encoder.encode_edges(edges)
        assert hv.shape == (100,)

    def test_encode_edges_clamps_out_of_bounds_indices(self):
        """Test that out-of-bounds node indices in edges are clamped."""
        encoder = GraphEncoder.create(num_nodes=5, dimensions=100, key=jax.random.PRNGKey(42))
        edges = jnp.array([[0, 1], [10, 20], [-1, 100]])  # Out-of-bounds
        hv = encoder.encode_edges(edges)
        assert hv.shape == (100,)
        assert jnp.isfinite(hv).all()

    def test_creation_with_vsa_model_instance(self):
        """Test GraphEncoder creation with VSAModel instance."""
        from bayes_hdc.vsa import MAP

        vsa = MAP.create(dimensions=100)
        encoder = GraphEncoder.create(
            num_nodes=5, dimensions=100, vsa_model=vsa, key=jax.random.PRNGKey(42)
        )
        assert encoder.vsa_model_name == "map"


class TestEncoderDefaultKey:
    """Cover key=None default branches in each encoder."""

    def test_random_encoder_default_key(self):
        enc = RandomEncoder.create(num_features=5, num_values=3, dimensions=100)
        assert enc.codebook.shape == (5, 3, 100)

    def test_level_encoder_default_key(self):
        enc = LevelEncoder.create(num_levels=5, dimensions=100)
        assert enc.level_hvs.shape == (5, 100)

    def test_projection_encoder_default_key(self):
        enc = ProjectionEncoder.create(input_dim=10, dimensions=100)
        assert enc.projection_matrix.shape == (10, 100)

    def test_kernel_encoder_default_key(self):
        enc = KernelEncoder.create(input_dim=10, dimensions=100)
        assert enc.omega.shape == (10, 100)

    def test_graph_encoder_default_key(self):
        enc = GraphEncoder.create(num_nodes=5, dimensions=100)
        assert enc.node_embeddings.shape == (5, 100)


class TestTokenEncoder:
    """Vocabulary → hypervector codebook + permute-bundle sequence encoding."""

    def test_create_shapes_and_unit_norm_rows(self):
        enc = TokenEncoder.create(vocab_size=128, dimensions=512)
        assert enc.vocab_size == 128
        assert enc.dimensions == 512
        assert enc.codebook.shape == (128, 512)
        # Each row L2-normalised — required for the SNR analysis the
        # downstream sequence encoders rely on.
        norms = jnp.linalg.norm(enc.codebook, axis=-1)
        assert jnp.all(jnp.abs(norms - 1.0) < 1e-4)

    def test_invalid_construction_args(self):
        with pytest.raises(ValueError, match="vocab_size"):
            TokenEncoder.create(vocab_size=0, dimensions=64)
        with pytest.raises(ValueError, match="dimensions"):
            TokenEncoder.create(vocab_size=10, dimensions=0)

    def test_lookup_matches_codebook_indexing(self):
        enc = TokenEncoder.create(vocab_size=32, dimensions=128)
        single = enc.lookup(jnp.asarray(7, dtype=jnp.int32))
        assert jnp.allclose(single, enc.codebook[7])
        ids = jnp.asarray([3, 14, 7, 0], dtype=jnp.int32)
        batch = enc.lookup_batch(ids)
        assert batch.shape == (4, 128)
        assert jnp.allclose(batch[2], enc.codebook[7])

    def test_encode_recovers_token_ids_via_codebook_cleanup(self):
        """The flat permute-bundle plus argmax cleanup against the
        codebook should recover each token at d=2048 for a length-12
        sentence."""
        enc = TokenEncoder.create(vocab_size=64, dimensions=2048)
        token_ids = jnp.asarray([3, 17, 42, 5, 8, 21, 33, 9, 14, 50, 1, 60], dtype=jnp.int32)
        seq_hv = enc.encode(token_ids)
        from bayes_hdc.functional import permute  # local import: trace

        n = token_ids.shape[0]
        n_correct = 0
        for i in range(n):
            recovered = permute(seq_hv, shifts=-(n - 1 - i))
            sims = enc.codebook @ recovered
            if int(jnp.argmax(sims)) == int(token_ids[i]):
                n_correct += 1
        # All 12 positions should clean back to the right token.
        assert n_correct == n, f"{n_correct}/{n} positions recovered"

    def test_encode_hierarchical_returns_long_sequence_holder(self):
        """For T well past flat capacity, encode_hierarchical returns
        a HierarchicalSequence whose get(i) + cleanup recovers all
        tokens."""
        from bayes_hdc.structures import HierarchicalSequence

        enc = TokenEncoder.create(vocab_size=128, dimensions=4096)
        token_ids = jax.random.randint(jax.random.PRNGKey(0), (300,), 0, 128, dtype=jnp.int32)
        hs = enc.encode_hierarchical(token_ids, chunk_size=16)
        assert isinstance(hs, HierarchicalSequence)
        assert hs.n_items == 300

        # Spot-check 20 evenly spaced positions; all should clean back.
        positions = jnp.linspace(0, 299, 20).astype(jnp.int32)
        n_correct = 0
        for p in positions:
            recovered = hs.get(int(p))
            sims = enc.codebook @ recovered
            if int(jnp.argmax(sims)) == int(token_ids[p]):
                n_correct += 1
        assert n_correct >= 18, f"{n_correct}/20 positions recovered at T=300"

    def test_codebook_is_deterministic_given_seed(self):
        a = TokenEncoder.create(vocab_size=16, dimensions=128, key=jax.random.PRNGKey(1))
        b = TokenEncoder.create(vocab_size=16, dimensions=128, key=jax.random.PRNGKey(1))
        assert jnp.allclose(a.codebook, b.codebook)
        c = TokenEncoder.create(vocab_size=16, dimensions=128, key=jax.random.PRNGKey(2))
        assert not jnp.allclose(a.codebook, c.codebook)
