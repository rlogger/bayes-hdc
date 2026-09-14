# SPDX-License-Identifier: MIT
# Copyright (c) 2026 R.S.

"""Regression tests for memory retrieval, graph encoding, and mixture moments."""

import jax
import jax.numpy as jnp
import pytest

from bayes_hdc.distributions import GaussianHV, MixtureHV
from bayes_hdc.embeddings import GraphEncoder
from bayes_hdc.memory import AttentionMemory
from bayes_hdc.structures import Graph


@pytest.mark.parametrize("num_heads", [1, 2])
def test_attention_weights_reconstruct_retrieval(num_heads):
    keys = jnp.array([[8.0, 0.0, 0.0, 0.0], [0.0, 0.0, 8.0, 0.0]])
    values = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    memory = AttentionMemory.create(4, num_heads=num_heads).write_batch(keys, values)
    query = jnp.array([1.0, 0.0, 1.0, 0.0])

    result, weights = memory.retrieve_with_weights(query)
    assert jnp.allclose(result, memory.retrieve(query))
    assert jnp.allclose(jnp.sum(weights, axis=-1), 1.0)
    if num_heads == 1:
        assert weights.shape == (2,)
        reconstructed = weights @ values
    else:
        assert weights.shape == (2, 2)
        reconstructed = jnp.concatenate([weights[0] @ values[:, :2], weights[1] @ values[:, 2:]])
        # Both heads attend independently; their distributions differ.
        assert not jnp.allclose(weights[0], weights[1])
    assert jnp.allclose(result, reconstructed)


@pytest.mark.parametrize("num_heads", [1, 2])
def test_empty_attention_weight_shape(num_heads):
    memory = AttentionMemory.create(4, num_heads=num_heads)
    result, weights = memory.retrieve_with_weights(jnp.ones(4))
    assert jnp.allclose(result, jnp.zeros(4))
    assert weights.shape == ((0,) if num_heads == 1 else (num_heads, 0))


@pytest.mark.parametrize("num_heads", [0, -1])
def test_attention_rejects_nonpositive_head_count(num_heads):
    with pytest.raises(ValueError, match="num_heads must be >= 1"):
        AttentionMemory.create(4, num_heads=num_heads)


@pytest.mark.parametrize("directed", [False, True])
def test_graph_neighbors_recover_original_node_coordinates(directed):
    source = jnp.array([1.0, -1.0, 1.0, 1.0])
    first = jnp.array([1.0, 2.0, 3.0, 4.0])
    second = jnp.array([-2.0, 3.0, 0.0, 5.0])
    graph = Graph.create(4, directed=directed).add_edge(source, first)
    graph = graph.add_edge(source, second)

    retrieved = jax.jit(lambda g, node: g.neighbors(node))(graph, source)
    assert jnp.allclose(retrieved, first + second)


@pytest.mark.parametrize("offset", [0.0, 10000.0])
def test_mixture_variance_preserves_within_component_uncertainty(offset):
    means = jnp.array([[0.0, 2.0], [2.0, 4.0]]) + offset
    variances = jnp.array([[1.0, 3.0], [5.0, 7.0]])
    mixture = MixtureHV.from_components(
        [GaussianHV(mu=means[i], var=variances[i], dimensions=2) for i in range(2)],
        weights=jnp.array([0.25, 0.75]),
    )

    # E[conditional variance] + variance of the two component means.
    expected = jnp.array([4.75, 6.75])
    assert jnp.allclose(mixture.variance(), expected)
    assert jnp.allclose(mixture.collapse_to_gaussian().var, expected)
    assert jnp.allclose(jax.jit(lambda m: m.variance())(mixture), expected)


def test_single_component_mixture_preserves_small_variance_at_large_mean():
    component = GaussianHV.from_sample(jnp.array([10000.0]), var=1.0)
    mixture = MixtureHV.from_components([component])
    assert jnp.allclose(mixture.variance(), component.var)


@pytest.mark.parametrize(
    "edges",
    [
        jnp.array([[0, 1], [1, 2]]),
        jnp.array([[-1, 1], [1, 8]]),
        jnp.empty((0, 2), dtype=jnp.int32),
    ],
)
def test_graph_encoder_entry_points_match_normalized_edge_bundle(edges):
    nodes = jnp.array([[1.0, 2.0, 3.0, 4.0], [2.0, -1.0, 1.0, 3.0], [3.0, 1.0, -2.0, 2.0]])
    encoder = GraphEncoder(node_embeddings=nodes, num_nodes=3, dimensions=4)
    clipped = jnp.clip(edges, 0, 2)
    pairs = nodes[clipped[:, 0]] * jnp.roll(nodes[clipped[:, 1]], 1, axis=-1)
    expected = jnp.sum(pairs, axis=0)
    expected = expected / (jnp.linalg.norm(expected) + 1e-8)

    assert jnp.allclose(encoder.encode_edges(edges), expected)
    assert jnp.allclose(encoder.encode_batch(edges), expected)
    assert jnp.allclose(jax.jit(encoder.encode_edges)(edges), expected)
