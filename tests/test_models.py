# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Tests for HDC classification models."""

import jax
import jax.numpy as jnp
import pytest

from bayes_hdc.models import (
    AdaptiveHDC,
    CentroidClassifier,
    ClusteringModel,
    LVQClassifier,
    RegularizedLSClassifier,
)
from bayes_hdc.vsa import BSC, MAP


class TestCentroidClassifier:
    """Tests for CentroidClassifier."""

    def test_creation_with_defaults(self):
        """Test CentroidClassifier creation with default parameters."""
        classifier = CentroidClassifier.create(
            num_classes=5, dimensions=100, key=jax.random.PRNGKey(42)
        )

        assert classifier.num_classes == 5
        assert classifier.dimensions == 100
        assert classifier.vsa_model_name == "map"
        assert classifier.prototypes.shape == (5, 100)

    def test_creation_with_bsc(self):
        """Test CentroidClassifier with BSC model."""
        classifier = CentroidClassifier.create(
            num_classes=3, dimensions=50, vsa_model="bsc", key=jax.random.PRNGKey(42)
        )

        assert classifier.vsa_model_name == "bsc"
        assert classifier.prototypes.dtype == jnp.bool_

    def test_creation_with_vsa_model_instance(self):
        """Test creation with VSAModel instance."""
        vsa = MAP.create(dimensions=100)
        classifier = CentroidClassifier.create(
            num_classes=3, dimensions=100, vsa_model=vsa, key=jax.random.PRNGKey(42)
        )

        assert classifier.vsa_model_name == "map"

    def test_creation_with_initial_prototypes(self):
        """Test creation with custom initial prototypes."""
        initial = jax.random.normal(jax.random.PRNGKey(42), (3, 100))
        classifier = CentroidClassifier.create(
            num_classes=3, dimensions=100, initial_prototypes=initial
        )

        assert jnp.allclose(classifier.prototypes, initial)

    def test_similarity_computation(self):
        """Test similarity computation between query and prototypes."""
        classifier = CentroidClassifier.create(
            num_classes=3, dimensions=100, key=jax.random.PRNGKey(42)
        )

        query = jax.random.normal(jax.random.PRNGKey(0), (100,))
        similarities = classifier.similarity(query)

        assert similarities.shape == (3,)
        assert jnp.isfinite(similarities).all()

    def test_predict_single_query(self):
        """Test prediction on a single query."""
        classifier = CentroidClassifier.create(
            num_classes=3, dimensions=100, key=jax.random.PRNGKey(42)
        )

        query = jax.random.normal(jax.random.PRNGKey(0), (100,))
        prediction = classifier.predict(query)

        assert prediction.shape == ()
        assert 0 <= prediction < 3

    def test_predict_batch(self):
        """Test prediction on a batch of queries."""
        classifier = CentroidClassifier.create(
            num_classes=3, dimensions=100, key=jax.random.PRNGKey(42)
        )

        queries = jax.random.normal(jax.random.PRNGKey(0), (10, 100))
        predictions = classifier.predict(queries)

        assert predictions.shape == (10,)
        assert jnp.all((predictions >= 0) & (predictions < 3))

    def test_predict_proba_single_query(self):
        """Test probability prediction on a single query."""
        classifier = CentroidClassifier.create(
            num_classes=3, dimensions=100, key=jax.random.PRNGKey(42)
        )

        query = jax.random.normal(jax.random.PRNGKey(0), (100,))
        probs = classifier.predict_proba(query)

        assert probs.shape == (3,)
        assert jnp.allclose(jnp.sum(probs), 1.0, atol=1e-5)
        assert jnp.all(probs >= 0) and jnp.all(probs <= 1)

    def test_predict_proba_batch(self):
        """Test probability prediction on a batch."""
        classifier = CentroidClassifier.create(
            num_classes=3, dimensions=100, key=jax.random.PRNGKey(42)
        )

        queries = jax.random.normal(jax.random.PRNGKey(0), (10, 100))
        probs = classifier.predict_proba(queries)

        assert probs.shape == (10, 3)
        assert jnp.allclose(jnp.sum(probs, axis=-1), 1.0, atol=1e-5)
        assert jnp.all(probs >= 0) and jnp.all(probs <= 1)

    def test_fit_raises_on_empty_training_data(self):
        """Test that fit raises ValueError on empty training data."""
        classifier = CentroidClassifier.create(
            num_classes=3, dimensions=100, key=jax.random.PRNGKey(42)
        )
        empty_hvs = jnp.zeros((0, 100))
        empty_labels = jnp.array([], dtype=jnp.int32)
        with pytest.raises(ValueError, match="training data is empty"):
            classifier.fit(empty_hvs, empty_labels)

    def test_fit(self):
        """Test fitting the classifier on training data."""
        vsa = MAP.create(dimensions=100)
        classifier = CentroidClassifier.create(
            num_classes=3, dimensions=100, vsa_model=vsa, key=jax.random.PRNGKey(42)
        )

        # Create training data
        key = jax.random.PRNGKey(0)
        train_hvs = vsa.random(key, (30, 100))
        train_labels = jnp.array([0] * 10 + [1] * 10 + [2] * 10)

        # Fit classifier
        trained_classifier = classifier.fit(train_hvs, train_labels)

        assert trained_classifier.prototypes.shape == (3, 100)
        # Prototypes should be different from initial random ones
        assert not jnp.allclose(trained_classifier.prototypes, classifier.prototypes)

    def test_fit_improves_accuracy(self):
        """Test that fitting achieves high accuracy on separable data."""
        vsa = MAP.create(dimensions=1000)
        classifier = CentroidClassifier.create(
            num_classes=3, dimensions=1000, vsa_model=vsa, key=jax.random.PRNGKey(42)
        )

        key = jax.random.PRNGKey(0)
        proto_keys = jax.random.split(key, 3)
        base_vectors = [vsa.random(proto_keys[i], (1000,)) for i in range(3)]

        train_hvs = []
        train_labels = []
        noise_key = jax.random.PRNGKey(99)
        for class_idx in range(3):
            for j in range(10):
                noise_key, subkey = jax.random.split(noise_key)
                noise = jax.random.normal(subkey, (1000,)) * 0.1
                sample = base_vectors[class_idx] + noise
                sample = sample / jnp.linalg.norm(sample)
                train_hvs.append(sample)
                train_labels.append(class_idx)

        train_hvs = jnp.stack(train_hvs)
        train_labels = jnp.array(train_labels)

        trained_classifier = classifier.fit(train_hvs, train_labels)
        accuracy = trained_classifier.score(train_hvs, train_labels)
        assert float(accuracy) > 0.9

    def test_update_online(self):
        """Test online update with a single sample."""
        classifier = CentroidClassifier.create(
            num_classes=3, dimensions=100, key=jax.random.PRNGKey(42)
        )

        sample = jax.random.normal(jax.random.PRNGKey(0), (100,))
        old_prototype = classifier.prototypes[1].copy()

        updated_classifier = classifier.update_online(sample, label=1, learning_rate=0.1)

        # Prototype should have changed
        assert not jnp.allclose(updated_classifier.prototypes[1], old_prototype)

    def test_update_online_bsc(self):
        """Test online update with BSC model."""
        classifier = CentroidClassifier.create(
            num_classes=3, dimensions=100, vsa_model="bsc", key=jax.random.PRNGKey(42)
        )

        sample = jax.random.bernoulli(jax.random.PRNGKey(0), 0.5, (100,))
        old_prototype = classifier.prototypes[1].copy()

        updated_classifier = classifier.update_online(sample, label=1, learning_rate=0.1)

        assert not jnp.allclose(updated_classifier.prototypes[1], old_prototype)
        assert updated_classifier.prototypes.dtype == jnp.bool_
        # Other prototypes should be unchanged
        assert jnp.allclose(updated_classifier.prototypes[0], classifier.prototypes[0])
        assert jnp.allclose(updated_classifier.prototypes[2], classifier.prototypes[2])

    def test_score(self):
        """Test accuracy scoring."""
        vsa = MAP.create(dimensions=100)
        classifier = CentroidClassifier.create(
            num_classes=3, dimensions=100, vsa_model=vsa, key=jax.random.PRNGKey(42)
        )

        # Create test data
        test_hvs = vsa.random(jax.random.PRNGKey(0), (20, 100))
        test_labels = jax.random.randint(jax.random.PRNGKey(1), (20,), 0, 3)

        score = classifier.score(test_hvs, test_labels)

        assert 0.0 <= score <= 1.0
        assert isinstance(float(score), float)

    def test_fit_with_empty_class(self):
        """Test fitting when some classes have no samples."""
        classifier = CentroidClassifier.create(
            num_classes=5, dimensions=100, key=jax.random.PRNGKey(42)
        )

        # Create training data with only 3 classes
        vsa = MAP.create(dimensions=100)
        train_hvs = vsa.random(jax.random.PRNGKey(0), (30, 100))
        train_labels = jnp.array([0] * 10 + [1] * 10 + [2] * 10)  # Classes 3 and 4 are empty

        trained_classifier = classifier.fit(train_hvs, train_labels)

        # Should still produce valid prototypes
        assert trained_classifier.prototypes.shape == (5, 100)
        assert jnp.isfinite(trained_classifier.prototypes).all()

    def test_immutability(self):
        """Test that operations return new instances (immutability)."""
        classifier1 = CentroidClassifier.create(
            num_classes=3, dimensions=100, key=jax.random.PRNGKey(42)
        )

        vsa = MAP.create(dimensions=100)
        train_hvs = vsa.random(jax.random.PRNGKey(0), (30, 100))
        train_labels = jnp.array([0] * 10 + [1] * 10 + [2] * 10)

        classifier2 = classifier1.fit(train_hvs, train_labels)

        # Should be different instances
        assert classifier1 is not classifier2
        # Original should be unchanged
        assert not jnp.allclose(classifier1.prototypes, classifier2.prototypes)


class TestAdaptiveHDC:
    """Tests for AdaptiveHDC."""

    def test_creation(self):
        """Test AdaptiveHDC creation."""
        classifier = AdaptiveHDC.create(num_classes=5, dimensions=100, key=jax.random.PRNGKey(42))

        assert classifier.num_classes == 5
        assert classifier.dimensions == 100
        assert classifier.vsa_model_name == "map"
        assert classifier.prototypes.shape == (5, 100)
        assert classifier.num_updates.shape == (5,)
        assert jnp.all(classifier.num_updates == 0)

    def test_creation_with_bsc(self):
        """Test AdaptiveHDC with BSC model."""
        classifier = AdaptiveHDC.create(
            num_classes=3, dimensions=50, vsa_model="bsc", key=jax.random.PRNGKey(42)
        )

        assert classifier.vsa_model_name == "bsc"
        assert classifier.prototypes.dtype == jnp.bool_

    def test_predict_single_query(self):
        """Test prediction on a single query."""
        classifier = AdaptiveHDC.create(num_classes=3, dimensions=100, key=jax.random.PRNGKey(42))

        query = jax.random.normal(jax.random.PRNGKey(0), (100,))
        prediction = classifier.predict(query)

        assert prediction.shape == ()
        assert 0 <= prediction < 3

    def test_predict_batch(self):
        """Test prediction on a batch of queries."""
        classifier = AdaptiveHDC.create(num_classes=3, dimensions=100, key=jax.random.PRNGKey(42))

        queries = jax.random.normal(jax.random.PRNGKey(0), (10, 100))
        predictions = classifier.predict(queries)

        assert predictions.shape == (10,)
        assert jnp.all((predictions >= 0) & (predictions < 3))

    def test_fit_raises_on_empty_training_data(self):
        """Test that fit raises ValueError on empty training data."""
        classifier = AdaptiveHDC.create(num_classes=3, dimensions=100, key=jax.random.PRNGKey(42))
        empty_hvs = jnp.zeros((0, 100))
        empty_labels = jnp.array([], dtype=jnp.int32)
        with pytest.raises(ValueError, match="training data is empty"):
            classifier.fit(empty_hvs, empty_labels)

    def test_fit_single_epoch(self):
        """Test fitting with a single epoch."""
        vsa = MAP.create(dimensions=100)
        classifier = AdaptiveHDC.create(
            num_classes=3, dimensions=100, vsa_model=vsa, key=jax.random.PRNGKey(42)
        )

        # Create training data
        train_hvs = vsa.random(jax.random.PRNGKey(0), (30, 100))
        train_labels = jnp.array([0] * 10 + [1] * 10 + [2] * 10)

        trained_classifier = classifier.fit(train_hvs, train_labels, epochs=1)

        assert trained_classifier.prototypes.shape == (3, 100)
        # Prototypes should be different from initial
        assert not jnp.allclose(trained_classifier.prototypes, classifier.prototypes)

    def test_fit_multiple_epochs(self):
        """Test fitting with multiple epochs."""
        vsa = MAP.create(dimensions=100)
        classifier = AdaptiveHDC.create(
            num_classes=3, dimensions=100, vsa_model=vsa, key=jax.random.PRNGKey(42)
        )

        # Create training data
        train_hvs = vsa.random(jax.random.PRNGKey(0), (30, 100))
        train_labels = jnp.array([0] * 10 + [1] * 10 + [2] * 10)

        trained_classifier = classifier.fit(train_hvs, train_labels, epochs=5)

        assert trained_classifier.prototypes.shape == (3, 100)

    def test_score(self):
        """Test accuracy scoring."""
        vsa = MAP.create(dimensions=100)
        classifier = AdaptiveHDC.create(
            num_classes=3, dimensions=100, vsa_model=vsa, key=jax.random.PRNGKey(42)
        )

        # Create test data
        test_hvs = vsa.random(jax.random.PRNGKey(0), (20, 100))
        test_labels = jax.random.randint(jax.random.PRNGKey(1), (20,), 0, 3)

        score = classifier.score(test_hvs, test_labels)

        assert 0.0 <= score <= 1.0

    def test_adaptive_improves_with_epochs(self):
        """Test that multiple epochs achieve good accuracy on separable data."""
        vsa = MAP.create(dimensions=500)
        classifier = AdaptiveHDC.create(
            num_classes=3, dimensions=500, vsa_model=vsa, key=jax.random.PRNGKey(42)
        )

        key = jax.random.PRNGKey(0)
        proto_keys = jax.random.split(key, 3)
        base_vectors = [vsa.random(proto_keys[i], (500,)) for i in range(3)]

        train_hvs = []
        train_labels = []
        noise_key = jax.random.PRNGKey(99)
        for class_idx in range(3):
            for _ in range(10):
                noise_key, subkey = jax.random.split(noise_key)
                noise = jax.random.normal(subkey, (500,)) * 0.1
                sample = base_vectors[class_idx] + noise
                sample = sample / jnp.linalg.norm(sample)
                train_hvs.append(sample)
                train_labels.append(class_idx)

        train_hvs = jnp.stack(train_hvs)
        train_labels = jnp.array(train_labels)

        classifier_5epochs = classifier.fit(train_hvs, train_labels, epochs=5)
        acc = classifier_5epochs.score(train_hvs, train_labels)
        assert float(acc) > 0.8

    def test_immutability(self):
        """Test that operations return new instances."""
        classifier1 = AdaptiveHDC.create(num_classes=3, dimensions=100, key=jax.random.PRNGKey(42))

        vsa = MAP.create(dimensions=100)
        train_hvs = vsa.random(jax.random.PRNGKey(0), (30, 100))
        train_labels = jnp.array([0] * 10 + [1] * 10 + [2] * 10)

        classifier2 = classifier1.fit(train_hvs, train_labels, epochs=1)

        # Should be different instances
        assert classifier1 is not classifier2

    def test_update_prototypes_internal(self):
        """Test internal prototype update mechanism."""
        classifier = AdaptiveHDC.create(num_classes=3, dimensions=100, key=jax.random.PRNGKey(42))

        sample = jax.random.normal(jax.random.PRNGKey(0), (100,))
        old_prototype = classifier.prototypes[1].copy()

        updated_classifier = classifier._update_prototypes(
            sample, true_label=1, pred_label=2, learning_rate=0.1
        )

        # True class prototype should have changed
        assert not jnp.allclose(updated_classifier.prototypes[1], old_prototype)

    def test_fit_with_bsc(self):
        """Test AdaptiveHDC fit with BSC model."""
        vsa = BSC.create(dimensions=100)
        classifier = AdaptiveHDC.create(
            num_classes=3, dimensions=100, vsa_model=vsa, key=jax.random.PRNGKey(42)
        )

        train_hvs = vsa.random(jax.random.PRNGKey(0), (30, 100))
        train_labels = jnp.array([0] * 10 + [1] * 10 + [2] * 10)

        trained_classifier = classifier.fit(train_hvs, train_labels, epochs=2)

        assert trained_classifier.prototypes.shape == (3, 100)
        assert trained_classifier.prototypes.dtype == jnp.bool_


class TestLVQClassifier:
    """Tests for LVQClassifier."""

    def test_creation(self):
        """Test LVQClassifier creation."""
        clf = LVQClassifier.create(num_classes=3, dimensions=100, key=jax.random.PRNGKey(42))
        assert clf.prototypes.shape == (3, 100)
        assert clf.num_classes == 3

    def test_predict_single_and_batch(self):
        """Test prediction on single and batch."""
        clf = LVQClassifier.create(num_classes=3, dimensions=100, key=jax.random.PRNGKey(42))
        single = jax.random.normal(jax.random.PRNGKey(0), (100,))
        batch = jax.random.normal(jax.random.PRNGKey(1), (5, 100))
        pred_single = clf.predict(single)
        pred_batch = clf.predict(batch)
        assert pred_single.shape == ()
        assert pred_batch.shape == (5,)

    def test_fit_raises_on_empty_training_data(self):
        """Test that fit raises ValueError on empty training data."""
        clf = LVQClassifier.create(num_classes=3, dimensions=100, key=jax.random.PRNGKey(42))
        empty_hvs = jnp.zeros((0, 100))
        empty_labels = jnp.array([], dtype=jnp.int32)
        with pytest.raises(ValueError, match="training data is empty"):
            clf.fit(empty_hvs, empty_labels)

    def test_fit_and_score(self):
        """Test fit and score."""
        vsa = MAP.create(dimensions=100)
        clf = LVQClassifier.create(
            num_classes=3, dimensions=100, vsa_model=vsa, key=jax.random.PRNGKey(42)
        )
        train_hvs = vsa.random(jax.random.PRNGKey(0), (30, 100))
        train_labels = jnp.array([0] * 10 + [1] * 10 + [2] * 10)
        clf = clf.fit(train_hvs, train_labels, epochs=2)
        score = clf.score(train_hvs, train_labels)
        assert 0.0 <= float(score) <= 1.0

    def test_creation_default_key(self):
        """Cover key=None branch in LVQClassifier.create."""
        clf = LVQClassifier.create(num_classes=3, dimensions=100)
        assert clf.prototypes.shape == (3, 100)

    def test_fit_bsc_bundle_branch(self):
        """Cover BSC bundle branch in LVQ fit."""
        clf = LVQClassifier.create(
            num_classes=2, dimensions=100, vsa_model="bsc", key=jax.random.PRNGKey(42)
        )
        train_hvs = jax.random.bernoulli(jax.random.PRNGKey(0), 0.5, (20, 100)).astype(jnp.float32)
        train_labels = jnp.array([i % 2 for i in range(20)])
        clf = clf.fit(train_hvs, train_labels, epochs=1, lr=0.1)
        assert clf.prototypes.shape == (2, 100)

    def test_predict_with_bsc(self):
        """Test LVQ predict with BSC model."""
        clf = LVQClassifier.create(
            num_classes=3, dimensions=100, vsa_model="bsc", key=jax.random.PRNGKey(42)
        )
        x = jax.random.bernoulli(jax.random.PRNGKey(0), 0.5, (100,))
        pred = clf.predict(x)
        assert 0 <= pred < 3


class TestRegularizedLSClassifier:
    """Tests for RegularizedLSClassifier."""

    def test_creation(self):
        """Test RegularizedLSClassifier creation."""
        clf = RegularizedLSClassifier.create(dimensions=100, num_classes=5)
        assert clf.weights.shape == (100, 5)
        assert clf.reg == 1e-4

    def test_creation_custom_reg(self):
        """Test creation with custom regularization."""
        clf = RegularizedLSClassifier.create(dimensions=50, num_classes=4, reg=0.01)
        assert clf.reg == 0.01

    def test_fit_raises_on_empty_training_data(self):
        """Test that fit raises ValueError on empty training data."""
        clf = RegularizedLSClassifier.create(dimensions=100, num_classes=3)
        empty_hvs = jnp.zeros((0, 100))
        empty_labels = jnp.array([], dtype=jnp.int32)
        with pytest.raises(ValueError, match="training data is empty"):
            clf.fit(empty_hvs, empty_labels)

    def test_fit_and_predict(self):
        """Test fit and predict."""
        clf = RegularizedLSClassifier.create(dimensions=100, num_classes=3)
        train_hvs = jax.random.normal(jax.random.PRNGKey(42), (50, 100))
        train_labels = jnp.array([0] * 17 + [1] * 17 + [2] * 16)
        clf = clf.fit(train_hvs, train_labels)
        preds = clf.predict(train_hvs[:5])
        assert preds.shape == (5,)

    def test_score(self):
        """Test score computation."""
        clf = RegularizedLSClassifier.create(dimensions=100, num_classes=3)
        train_hvs = jax.random.normal(jax.random.PRNGKey(42), (60, 100))
        train_labels = jnp.array([i % 3 for i in range(60)])
        clf = clf.fit(train_hvs, train_labels)
        score = clf.score(train_hvs, train_labels)
        assert 0.0 <= float(score) <= 1.0

    def test_predict_single_query(self):
        """Test predict with single query (1D input)."""
        clf = RegularizedLSClassifier.create(dimensions=50, num_classes=4)
        train_hvs = jax.random.normal(jax.random.PRNGKey(42), (40, 50))
        train_labels = jnp.array([i % 4 for i in range(40)])
        clf = clf.fit(train_hvs, train_labels)
        single = jax.random.normal(jax.random.PRNGKey(0), (50,))
        pred = clf.predict(single)
        assert pred.shape == ()
        assert 0 <= int(pred) < 4


class TestClusteringModel:
    """Tests for ClusteringModel."""

    def test_creation(self):
        model = ClusteringModel.create(num_clusters=3, dimensions=100)
        assert model.num_clusters == 3
        assert model.dimensions == 100
        assert model.centroids.shape == (3, 100)

    def test_fit(self):
        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)
        cluster0 = jax.random.normal(k1, (20, 100)) + 5.0
        cluster1 = jax.random.normal(k2, (20, 100)) - 5.0
        hvs = jnp.concatenate([cluster0, cluster1], axis=0)
        norms = jnp.linalg.norm(hvs, axis=-1, keepdims=True)
        hvs = hvs / norms

        model = ClusteringModel.create(num_clusters=2, dimensions=100, key=k3)
        model = model.fit(hvs, max_iters=20)
        assert model.centroids.shape == (2, 100)

    def test_predict(self):
        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)
        cluster0 = jax.random.normal(k1, (20, 100)) + 5.0
        cluster1 = jax.random.normal(k2, (20, 100)) - 5.0
        hvs = jnp.concatenate([cluster0, cluster1], axis=0)
        norms = jnp.linalg.norm(hvs, axis=-1, keepdims=True)
        hvs = hvs / norms

        model = ClusteringModel.create(num_clusters=2, dimensions=100, key=k3)
        model = model.fit(hvs, max_iters=20)
        labels = model.predict(hvs)
        assert labels.shape == (40,)
        assert jnp.all((labels == 0) | (labels == 1))
        labels_0 = labels[:20]
        labels_1 = labels[20:]
        assert jnp.sum(labels_0 == labels_0[0]) >= 15
        assert jnp.sum(labels_1 == labels_1[0]) >= 15

    def test_predict_single(self):
        model = ClusteringModel.create(num_clusters=3, dimensions=50)
        query = jax.random.normal(jax.random.PRNGKey(0), (50,))
        pred = model.predict(query)
        assert 0 <= int(pred) < 3

    def test_create_with_vsa_model_instance(self):
        """Cover vsa_model.name branch in ClusteringModel.create."""
        vsa = MAP.create(dimensions=100)
        model = ClusteringModel.create(num_clusters=3, dimensions=100, vsa_model=vsa)
        assert model.vsa_model_name == "map"

    def test_replace(self):
        """Cover ClusteringModel.replace."""
        model = ClusteringModel.create(num_clusters=2, dimensions=50)
        new_centroids = jnp.ones((2, 50))
        updated = model.replace(centroids=new_centroids)
        assert jnp.allclose(updated.centroids, 1.0)
