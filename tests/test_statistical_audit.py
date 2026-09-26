# SPDX-License-Identifier: MIT
# Copyright (c) 2026 R.S.

"""Independent analytic regressions for statistical and prediction contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayes_hdc.anomaly import ConformalAnomalyDetector, HDCAnomalyScorer, fit_anomaly_pipeline
from bayes_hdc.bayesian_models import (
    BayesianAdaptiveHDC,
    BayesianCentroidClassifier,
    StreamingBayesianHDC,
)
from bayes_hdc.distributions import GaussianHV
from bayes_hdc.embeddings import ProjectionEncoder
from bayes_hdc.inference import gaussian_reconstruction_log_likelihood_mc
from bayes_hdc.metrics import (
    cosine_matrix,
    effective_dimensions,
    negative_log_likelihood,
    required_dimension,
    signal_energy,
)
from bayes_hdc.models import (
    AdaptiveHDC,
    CentroidClassifier,
    ClusteringModel,
    HDRegressor,
    LVQClassifier,
    RegularizedLSClassifier,
)
from bayes_hdc.training import adam_init, train_variational_codebook
from bayes_hdc.uncertainty import ConformalClassifier, ConformalRegressor, TemperatureCalibrator


def test_conformal_anomaly_rejects_calibration_training_leakage():
    detector = ConformalAnomalyDetector.create(HDCAnomalyScorer.create(dimensions=2))
    with pytest.raises(ValueError, match="separate training split"):
        detector.fit(jnp.ones((4, 2)))
    with pytest.raises(ValueError, match="separate training split"):
        jax.jit(detector.fit)(jnp.ones((4, 2)))


def test_preallocated_anomaly_buffer_is_not_fitted():
    scorer = HDCAnomalyScorer.create(dimensions=2).fit(jnp.ones((3, 2)))
    detector = ConformalAnomalyDetector.create(scorer, n_calibration=10)
    with pytest.raises(ValueError, match="fitted"):
        detector.pvalue(jnp.ones(2))


def test_conformal_anomaly_ties_are_conservative_and_empty_fdr_is_valid():
    scorer = HDCAnomalyScorer.create(dimensions=2).fit(jnp.ones((2, 2)))
    detector = ConformalAnomalyDetector.create(scorer).fit(jnp.ones((3, 2)))
    assert float(detector.pvalue(jnp.ones(2))) == 1.0
    assert float(detector.pvalue(-jnp.ones(2))) == 0.25
    assert detector.predict_fdr(jnp.empty((0, 2))).shape == (0,)
    assert bool(jax.jit(detector.predict)(-jnp.ones(2), jnp.array(0.3)))


def test_anomaly_pipeline_honors_supplied_default_alpha():
    encoder = ProjectionEncoder.create(input_dim=2, dimensions=16)
    detector = fit_anomaly_pipeline(encoder, jnp.ones((3, 2)), jnp.ones((3, 2)), alpha=0.3)
    query = encoder.encode(-jnp.ones(2))
    assert bool(detector.predict(query))
    assert not bool(detector.predict(query, alpha=0.1))


def test_aps_probability_ties_do_not_depend_on_class_order():
    calibration = jnp.tile(jnp.array([0.4, 0.3, 0.3]), (5, 1))
    labels = jnp.array([0, 0, 0, 1, 1])
    wrapper = ConformalClassifier.create(alpha=0.4).fit(calibration, labels)
    permutation = jnp.array([2, 0, 1])
    permuted_labels = jnp.argsort(permutation)[labels]
    permuted = ConformalClassifier.create(alpha=0.4).fit(
        calibration[:, permutation], permuted_labels
    )
    np.testing.assert_allclose(wrapper.threshold, 1.0)
    np.testing.assert_allclose(permuted.threshold, wrapper.threshold)
    probs = jnp.array([[0.4, 0.3, 0.3], [0.5, 0.25, 0.25]])
    np.testing.assert_array_equal(
        wrapper.predict_set(probs)[:, permutation], permuted.predict_set(probs[:, permutation])
    )
    np.testing.assert_array_equal(jax.vmap(wrapper.predict_set)(probs), wrapper.predict_set(probs))


def test_aps_all_tied_top_classes_are_included():
    wrapper = ConformalClassifier.create(alpha=0.5).fit(
        jnp.tile(jnp.array([0.8, 0.1, 0.1]), (3, 1)), jnp.zeros(3, dtype=int)
    )
    assert jnp.all(wrapper.predict_set(jnp.ones(3) / 3))


@pytest.mark.parametrize("labels", [jnp.array([-1, 0]), jnp.array([0, 2]), jnp.array([0.0, 1.0])])
def test_conformal_rejects_invalid_class_indices(labels):
    with pytest.raises(ValueError, match="labels"):
        ConformalClassifier.create().fit(jnp.array([[0.7, 0.3], [0.4, 0.6]]), labels)


def test_conformal_rejects_logits_and_unfitted_intervals():
    with pytest.raises(ValueError, match="probabilities|sum to one"):
        ConformalClassifier.create().fit(jnp.array([[2.0, 1.0], [1.0, 3.0]]), jnp.array([0, 1]))
    with pytest.raises(ValueError, match="fitted"):
        ConformalClassifier.create().predict_set(jnp.array([0.7, 0.3]))
    with pytest.raises(ValueError, match="fitted"):
        ConformalRegressor.create().predict_interval(jnp.zeros(3))


def test_streaming_variance_equals_weighted_distribution_variance():
    # Initial distribution: mean=0, var=4, weight decays by 1/2 per observation.
    model = StreamingBayesianHDC.create(1, 1, decay=0.5, prior_var=4.0)
    observations = jnp.array([[2.0], [6.0], [-1.0]])
    fitted = model.fit(observations, jnp.zeros(3, dtype=int))
    weights = np.array([1 / 8, 1 / 8, 1 / 4, 1 / 2])
    means = np.array([0.0, 2.0, 6.0, -1.0])
    variances = np.array([4.0, 0.0, 0.0, 0.0])
    expected_mean = np.dot(weights, means)
    expected_var = np.dot(weights, variances + (means - expected_mean) ** 2)
    np.testing.assert_allclose(fitted.mu, [[expected_mean]], atol=1e-6)
    np.testing.assert_allclose(fitted.var, [[expected_var]], atol=1e-6)
    eager = model
    for x in observations:
        eager = eager.update(x, 0)
    np.testing.assert_allclose(eager.var, fitted.var)


def test_kalman_matches_closed_form_precision_update():
    model = BayesianAdaptiveHDC.create(1, 2, prior_var=2.0, obs_var=0.5)
    x = jnp.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 6.0]])
    fitted = jax.jit(model.fit)(x, jnp.zeros(3, dtype=int))
    posterior_var = 1 / (1 / 2 + 3 / 0.5)
    np.testing.assert_allclose(fitted.var, posterior_var, rtol=1e-6)
    np.testing.assert_allclose(fitted.mu[0], posterior_var * np.sum(x, axis=0) / 0.5, rtol=1e-6)


@pytest.mark.parametrize("method", [BayesianAdaptiveHDC, StreamingBayesianHDC])
def test_online_gaussian_models_reject_invalid_label(method):
    model = method.create(2, 2)
    with pytest.raises(ValueError, match="label"):
        model.update(jnp.ones(2), -1)


@pytest.mark.parametrize("method", [BayesianAdaptiveHDC, StreamingBayesianHDC])
def test_gaussian_models_reject_negative_variance(method):
    with pytest.raises(ValueError, match="prior_var"):
        method.create(1, 2, prior_var=-1.0)


def test_batch_moment_model_empty_class_and_zero_prior_remain_finite():
    model = BayesianCentroidClassifier.create(2, 2)
    fitted = jax.jit(lambda x: model.fit(x, jnp.zeros(2, dtype=int), prior_strength=0.0))(
        jnp.ones((2, 2))
    )
    np.testing.assert_array_equal(fitted.var[1], model.var[1])
    assert jnp.all(jnp.isfinite(fitted.var))


@pytest.mark.parametrize("dtype", [jnp.bool_, jnp.int8, jnp.float32, jnp.complex64])
@pytest.mark.parametrize("shape", [(5, 2), (2, 5)])
def test_ridge_uses_numeric_hermitian_linear_algebra(dtype, shape):
    x = np.arange(np.prod(shape)).reshape(shape) % 3
    if dtype == jnp.bool_:
        x = x > 0
    if dtype == jnp.complex64:
        x = x + 1j * (np.arange(np.prod(shape)).reshape(shape) % 2)
    x = jnp.asarray(x, dtype=dtype)
    y = jnp.arange(shape[0], dtype=float)[:, None]
    model = HDRegressor.create(shape[1], 1, reg=0.3).fit(x, y)
    host_x = np.asarray(x).astype(np.complex128 if dtype == jnp.complex64 else np.float64)
    expected = np.linalg.solve(
        host_x.conj().T @ host_x + 0.3 * np.eye(shape[1]), host_x.conj().T @ np.asarray(y)
    )
    np.testing.assert_allclose(model.weights, expected, rtol=1e-4, atol=2e-5)


def test_unregularized_rank_deficient_regression_returns_minimum_norm_solution():
    x = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    model = HDRegressor.create(2, 1, reg=0.0).fit(x, jnp.array([2.0, 4.0, 6.0]))
    np.testing.assert_allclose(model.weights, [[1.0], [1.0]], atol=1e-5)
    assert float(model.score(x, jnp.array([2.0, 4.0, 6.0]))) > 0.999


def test_boolean_ridge_classifier_matches_float_encoding():
    x = jnp.array([[True, False], [True, True], [False, True]])
    y = jnp.array([0, 0, 1])
    model = RegularizedLSClassifier.create(2, 2)
    np.testing.assert_allclose(model.fit(x, y).weights, model.fit(x.astype(float), y).weights)


def test_centroid_fit_compiles_and_preserves_empty_class():
    model = CentroidClassifier.create(3, 2)
    fitted = jax.jit(model.fit)(jnp.array([[1.0, 0.0], [0.0, 1.0]]), jnp.array([0, 1]))
    np.testing.assert_allclose(fitted.prototypes[:2], jnp.eye(2), atol=1e-6)
    np.testing.assert_allclose(fitted.prototypes[2], model.prototypes[2])


def test_bsc_lvq_accumulates_subthreshold_updates():
    model = LVQClassifier.create(1, 2, vsa_model="bsc").replace(
        prototypes=jnp.array([[False, True]])
    )
    fitted = model.fit(jnp.array([[True, False]]), jnp.array([0]), epochs=10, lr=0.1)
    np.testing.assert_array_equal(fitted.prototypes, [[True, False]])


def test_adaptive_update_counts_are_recorded():
    model = AdaptiveHDC.create(2, 2)
    updated = model._update_prototypes(jnp.array([1.0, 0.0]), 0, 1, 0.1)
    np.testing.assert_array_equal(updated.num_updates, [1, 0])


def test_bsc_clustering_uses_hamming_and_binary_majority():
    model = ClusteringModel.create(2, 2, vsa_model="bsc").replace(
        centroids=jnp.array([[False, False], [True, True]])
    )
    x = jnp.array([[False, False], [False, False], [True, True]])
    np.testing.assert_array_equal(model.predict(x), [0, 0, 1])
    fitted = model.fit(x)
    assert fitted.centroids.dtype == jnp.bool_
    np.testing.assert_array_equal(fitted.predict(x), [0, 0, 1])


def test_complex_clustering_preserves_phase():
    model = ClusteringModel.create(2, 2, vsa_model="fhrr").replace(
        centroids=jnp.array([[1j, 1j], [-1j, -1j]]) / jnp.sqrt(2)
    )
    x = jnp.array([[1j, 1j], [-1j, -1j]])
    np.testing.assert_array_equal(model.fit(x).predict(x), [0, 1])


def test_metrics_support_complex_vectors_and_scale_invariance():
    x = jnp.array([[1j, 1j], [-1j, -1j]])
    np.testing.assert_allclose(signal_energy(x), [2.0, 2.0])
    np.testing.assert_allclose(cosine_matrix(x), [[1.0, -1.0], [-1.0, 1.0]], atol=1e-6)
    for scale in (1e-12, 1.0, 1e12):
        np.testing.assert_allclose(effective_dimensions(scale * x), [2.0, 2.0], atol=1e-6)


def test_required_dimension_uses_error_probability_from_primary_formula():
    # Stewart et al. example: k=7,M=100000,95% success -> about 700 dimensions.
    assert int(required_dimension(7, 100000, q=0.95)) == 697
    assert required_dimension(7, 100000, q=0.99) > required_dimension(7, 100000, q=0.95)
    assert int(required_dimension(1, 1, q=0.1)) >= 1


def test_nll_does_not_cap_impossible_event_penalty():
    assert jnp.isinf(negative_log_likelihood(jnp.array([[1.0, 0.0]]), jnp.array([1])))
    nll = negative_log_likelihood(jnp.array([[1.0, 1e-20]]), jnp.array([1]))
    np.testing.assert_allclose(nll, -np.log(1e-20), rtol=1e-6)


def test_gaussian_reconstruction_supports_noise_gradient_and_jit():
    posterior = GaussianHV.from_sample(jnp.array([1.0, 2.0]), var=0.0)
    target = GaussianHV.from_sample(jnp.array([0.0, 0.0]), var=0.0)
    key = jax.random.PRNGKey(0)

    def f(noise):
        return gaussian_reconstruction_log_likelihood_mc(
            posterior, target, key, n_samples=4, observation_noise=noise
        )

    # Exact deterministic residual squared norm=5, dimension=2.
    np.testing.assert_allclose(jax.grad(f)(jnp.array(2.0)), 5 / 8 - 2 / 2, atol=1e-5)
    assert jnp.isfinite(jax.jit(f)(jnp.array(2.0)))
    with pytest.raises(ValueError, match="observation_noise"):
        f(0.0)


def test_variational_training_rejects_empty_history_and_nonreal_parameters():
    with pytest.raises(ValueError, match="n_steps"):
        train_variational_codebook(
            jnp.ones(2), lambda p, k: jnp.sum(p * p), key=jax.random.PRNGKey(0), n_steps=0
        )
    with pytest.raises(ValueError, match="floating-point"):
        adam_init(jnp.ones(2, dtype=int))
    with pytest.raises(ValueError, match="floating-point"):
        adam_init(jnp.ones(2, dtype=complex))


def test_temperature_rejects_invalid_initial_value_and_bounds():
    with pytest.raises(ValueError, match="initial_temperature"):
        TemperatureCalibrator.create(-1.0)
    with pytest.raises(ValueError, match="t_max"):
        TemperatureCalibrator.create().fit(
            jnp.array([[1.0, 0.0]]), jnp.array([0]), t_min=2.0, t_max=1.0
        )


@pytest.mark.parametrize("label", [2**32, -(2**32)])
def test_labels_are_validated_before_jax_integer_narrowing(label):
    from bayes_hdc._validation import labels_array

    labels = np.array([label], dtype=np.int64)
    with pytest.raises(ValueError, match="labels"):
        labels_array(labels, 1, 2)
    with pytest.raises(ValueError, match="labels"):
        ConformalClassifier.create().fit(jnp.array([[0.7, 0.3]]), labels)
    for cls in (BayesianAdaptiveHDC, StreamingBayesianHDC):
        with pytest.raises(ValueError, match="label"):
            cls.create(2, 2).update(jnp.ones(2), np.int64(label))


@pytest.mark.parametrize("cls", [BayesianAdaptiveHDC, StreamingBayesianHDC])
def test_online_gaussian_models_reject_complex_input_eager_and_jit(cls):
    model = cls.create(2, 2)
    sample = jnp.array([10j, 10j])
    for update in (model.update, jax.jit(model.update)):
        with pytest.raises(ValueError, match="real-valued"):
            update(sample, 0)


@pytest.mark.parametrize("cls", [CentroidClassifier, AdaptiveHDC, LVQClassifier, ClusteringModel])
@pytest.mark.parametrize("vsa_name", ["cgr", "mcr", "bsbc"])
def test_prototype_models_reject_unsupported_vsa_geometries(cls, vsa_name):
    from bayes_hdc.vsa import create_vsa_model

    vsa = create_vsa_model(vsa_name, 1000)
    for argument in (vsa_name, vsa):
        with pytest.raises(ValueError, match="Unsupported classifier VSA"):
            cls.create(2, dimensions=1000, vsa_model=argument)


@pytest.mark.parametrize("cls", [CentroidClassifier, AdaptiveHDC, LVQClassifier, ClusteringModel])
def test_prototype_models_reject_mismatched_vsa_instance_dimensions(cls):
    from bayes_hdc.vsa import MAP

    with pytest.raises(ValueError, match="dimensions must match"):
        cls.create(2, dimensions=4, vsa_model=MAP.create(8))


def test_regression_r2_is_invariant_to_target_scale():
    model = HDRegressor.create(1, 1)
    x = jnp.ones((2, 1))
    for scale in (1.0, 1e-6, 1e6):
        np.testing.assert_allclose(model.score(x, scale * jnp.array([1.0, 2.0])), -9.0, rtol=1e-6)
    assert float(model.score(x, jnp.zeros(2))) == 1.0
    assert float(model.score(x, jnp.ones(2))) == 0.0
