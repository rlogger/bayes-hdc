# SPDX-License-Identifier: MIT
"""Regression checks for algebra, distribution, and encoder audit findings."""

import jax
import jax.numpy as jnp
import pytest

from bayes_hdc import functional as F
from bayes_hdc.distributions import DirichletHV, GaussianHV, MixtureHV, bind_dirichlet
from bayes_hdc.embeddings import (
    GraphEncoder,
    KernelEncoder,
    LevelEncoder,
    ProjectionEncoder,
    RandomEncoder,
)
from bayes_hdc.memory import AttentionMemory
from bayes_hdc.resonator import probabilistic_resonator
from bayes_hdc.vsa import BSBC, FHRR, create_vsa_model


def test_cleanup_handles_custom_similarity_and_multiple_query_axes():
    memory = jnp.eye(3)
    queries = memory[jnp.array([[2, 0], [1, 2]])]
    result, scores = F.cleanup(queries, memory, F.dot_similarity, return_similarity=True)
    assert jnp.array_equal(result, queries)
    assert jnp.array_equal(scores, jnp.ones((2, 2)))


def test_cleanup_binary_metric_and_empty_memory():
    memory = jnp.array([[True, False], [False, True]])
    assert jnp.array_equal(F.cleanup(memory[1], memory, F.hamming_similarity), memory[1])
    with pytest.raises(ValueError, match="non-empty"):
        F.cleanup(jnp.ones(2), jnp.empty((0, 2)))


def test_inverse_map_zero_has_finite_gradient():
    gradient = jax.grad(lambda x: jnp.sum(F.inverse_map(x)))(jnp.array([0.0, 2.0]))
    assert jnp.allclose(gradient, jnp.array([0.0, -0.25]))


def test_complex_cosine_uses_hermitian_inner_product():
    x = jnp.array([1j, 1.0 + 0j])
    assert jnp.isclose(F.cosine_similarity(x, x), 1.0)
    assert jnp.isclose(F.cosine_similarity(x, -x), -1.0)


def test_vtb_agrees_with_published_block_diagonal_definition():
    # Gosmann/Eliasmith: V_y = I_n kron (sqrt(n) * reshape(y)); B(x,y)=V_y x.
    x = jnp.arange(1.0, 10.0)
    y = jnp.array([2.0, -1.0, 4.0, 1.0, 3.0, 2.0, -2.0, 5.0, 1.0])
    transformation = jnp.kron(jnp.eye(3), jnp.sqrt(3.0) * y.reshape(3, 3))
    assert jnp.allclose(F.bind_vtb(x, y), transformation @ x)


def test_vtb_unitary_right_unbinding_and_nonassociativity():
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    y = jnp.array([0.0, -1.0, 1.0, 0.0]) / jnp.sqrt(2.0)
    assert jnp.allclose(F.bind_vtb(F.bind_vtb(x, y), F.inverse_vtb(y)), x)
    z = jnp.array([2.0, -1.0, 1.0, 3.0])
    assert not jnp.allclose(F.bind_vtb(F.bind_vtb(x, y), z), F.bind_vtb(x, F.bind_vtb(y, z)))


@pytest.mark.parametrize("key", [jax.random.PRNGKey(0), jax.random.key(0)])
def test_bsbc_supports_typed_and_legacy_keys_inside_jit(key):
    model = BSBC.create(20, block_size=5, k_active=2)
    result = jax.jit(lambda k: model.random(k, (2, 3, 20)))(key)
    assert result.shape == (2, 3, 20)
    assert jnp.all(result.reshape(2, 3, 4, 5).sum(-1) == 2)
    empty = model.random(key, (0, 20))
    assert empty.shape == (0, 20)


def test_fhrr_bundle_retains_unit_phasors_for_unbinding():
    model = FHRR.create(8)
    x = model.random(jax.random.key(1), (8,))
    y = model.bundle(model.random(jax.random.key(2), (3, 8)))
    assert jnp.allclose(jnp.abs(y), 1.0)
    assert jnp.allclose(model.bind(model.bind(x, y), model.inverse(y)), x, atol=1e-6)
    cancelled = model.bundle(jnp.array([[1j], [-1j]]))
    assert jnp.array_equal(cancelled, jnp.ones(1, dtype=jnp.complex64))


@pytest.mark.parametrize("name", ["bsc", "map", "hrr", "fhrr"])
def test_graph_encoder_uses_requested_binding_and_bundling(name):
    encoder = GraphEncoder.create(3, 16, vsa_model=name)
    vsa = create_vsa_model(name, 16)
    edges = jnp.array([[0, 1], [1, 2], [2, 0]])
    expected = vsa.bundle(
        vsa.bind(
            encoder.node_embeddings[edges[:, 0]], F.permute(encoder.node_embeddings[edges[:, 1]])
        )
    )
    assert jnp.allclose(encoder.encode_edges(edges), expected)
    assert jnp.all(encoder.encode_edges(jnp.empty((0, 2), dtype=jnp.int32)) == 0)


def test_random_fhrr_encoder_preserves_phasor_space():
    encoder = RandomEncoder.create(3, 4, 16, vsa_model="fhrr")
    assert jnp.allclose(jnp.abs(encoder.encode(jnp.array([0, 1, 2]))), 1.0)


@pytest.mark.parametrize("factory", [ProjectionEncoder, KernelEncoder])
def test_real_input_fhrr_encoders_return_phasors(factory):
    encoder = factory.create(3, 16, vsa_model="fhrr")
    encoded = encoder.encode(jnp.ones(3))
    assert jnp.iscomplexobj(encoded)
    assert jnp.allclose(jnp.abs(encoded), 1.0)


def test_circular_level_wraps_at_boundary_and_interpolates_across_it():
    encoder = LevelEncoder.create(8, 32, encoding_type="circular")
    values = jnp.array([-0.125, 0.0, 0.25, 1.0, 1.25])
    encoded = encoder.encode(values)
    assert jnp.allclose(encoded[0], encoder.encode(0.875))
    assert jnp.allclose(encoded[1], encoded[3])
    assert jnp.allclose(encoded[2], encoded[4])
    assert F.cosine_similarity(encoder.encode(0.9999), encoder.encode(0.0001)) > 0.99


def test_fhrr_level_interpolation_uses_short_phase_arc():
    encoder = LevelEncoder.create(2, 2, vsa_model="fhrr")
    encoder.level_hvs = jnp.exp(1j * jnp.array([[3.0, 0.0], [-3.0, 1.0]]))
    encoded = encoder.encode(0.5)
    assert jnp.allclose(jnp.abs(encoded), 1.0)
    assert jnp.allclose(encoded, jnp.array([-1.0 + 0j, jnp.exp(0.5j)]), atol=1e-6)


@pytest.mark.parametrize(
    "kwargs", [{"encoding_type": "unknown"}, {"num_levels": 0}, {"min_value": float("nan")}]
)
def test_level_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        LevelEncoder.create(dimensions=8, **kwargs)


@pytest.mark.parametrize("name", ["typo", "cgr", "mcr", "vtb"])
def test_projection_rejects_unimplemented_representations(name):
    with pytest.raises(ValueError):
        ProjectionEncoder.create(3, 16, vsa_model=name)


def test_small_dirichlet_concentration_preserves_simplex_and_total():
    x = DirichletHV(alpha=jnp.array([1e-12, 2e-12]), dimensions=2)
    y = DirichletHV(alpha=jnp.array([2e-12, 1e-12]), dimensions=2)
    assert jnp.allclose(x.mean(), jnp.array([1 / 3, 2 / 3]))
    bound = bind_dirichlet(x, y)
    assert jnp.allclose(bound.mean(), jnp.array([0.5, 0.5]))
    assert jnp.isclose(bound.concentration(), 6e-12, rtol=1e-5, atol=0)


def test_distribution_batch_sampling_preserves_existing_batch_shape():
    gaussian = GaussianHV(mu=jnp.zeros((2, 3)), var=jnp.ones((2, 3)), dimensions=3)
    assert gaussian.sample_batch(jax.random.key(1), 4).shape == (4, 2, 3)
    dirichlet = DirichletHV(alpha=jnp.ones((2, 3)), dimensions=3)
    samples = dirichlet.sample_batch(jax.random.key(2), 4)
    assert samples.shape == (4, 2, 3)
    assert jnp.allclose(samples.sum(-1), 1.0)


def test_mixture_small_weights_normalize_without_epsilon_bias():
    components = [
        GaussianHV.from_sample(jnp.array([1.0])),
        GaussianHV.from_sample(jnp.array([5.0])),
    ]
    mix = MixtureHV.from_components(components, jnp.array([1e-20, 3e-20]))
    assert jnp.allclose(mix.weights, jnp.array([0.25, 0.75]))
    assert jnp.allclose(mix.mean(), 4.0)


@pytest.mark.parametrize("weights", [[0.0, 0.0], [-1.0, 2.0], [float("nan"), 1.0], [1.0]])
def test_mixture_rejects_invalid_weights(weights):
    component = GaussianHV.from_sample(jnp.ones(2))
    with pytest.raises(ValueError, match="weights"):
        MixtureHV.from_components([component, component], jnp.array(weights))


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("inf"), float("nan")])
def test_attention_and_resonator_reject_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        AttentionMemory.create(4, temperature=temperature)
    cb = GaussianHV(mu=jnp.ones((2, 4)), var=jnp.zeros((2, 4)), dimensions=4)
    with pytest.raises(ValueError, match="temperature"):
        probabilistic_resonator(
            [cb], GaussianHV.from_sample(jnp.ones(4)), jax.random.key(0), temperature=temperature
        )


def test_jaccard_empty_sets_are_identical():
    empty = jnp.zeros(4, dtype=bool)
    assert F.jaccard_similarity(empty, empty) == 1
    assert F.tversky_similarity(empty, empty) == 1


def test_integer_component_means_do_not_truncate_mixture_weights():
    components = [GaussianHV.from_sample(jnp.array([1])), GaussianHV.from_sample(jnp.array([5]))]
    mix = MixtureHV.from_components(components, jnp.array([0.25, 0.75]))
    assert jnp.allclose(mix.mean(), 4.0)


def test_random_encoder_rejects_missing_features():
    encoder = RandomEncoder.create(3, 4, 16)
    with pytest.raises(ValueError, match="indices"):
        encoder.encode(jnp.array([0, 1]))
