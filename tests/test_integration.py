# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Integration tests: end-to-end encode -> train -> predict pipeline."""

import jax

from bayes_hdc import BSC, MAP
from bayes_hdc.embeddings import LevelEncoder, ProjectionEncoder, RandomEncoder
from bayes_hdc.models import AdaptiveHDC, CentroidClassifier


class TestIntegrationPipeline:
    """End-to-end pipeline with RandomEncoder + CentroidClassifier."""

    def test_discrete_features_map_centroid(self, prng_key, default_dimensions):
        """Encode discrete features -> train centroid -> predict."""
        n_features, n_values, n_classes = 10, 20, 5
        n_train, n_test = 100, 30

        model = MAP.create(dimensions=default_dimensions)
        encoder = RandomEncoder.create(
            num_features=n_features,
            num_values=n_values,
            dimensions=default_dimensions,
            vsa_model=model,
            key=prng_key,
        )
        classifier = CentroidClassifier.create(
            num_classes=n_classes,
            dimensions=default_dimensions,
            vsa_model=model,
        )

        key_train, key_test = jax.random.split(prng_key)
        train_data = jax.random.randint(key_train, (n_train, n_features), 0, n_values)
        train_labels = jax.random.randint(key_train, (n_train,), 0, n_classes)
        test_data = jax.random.randint(key_test, (n_test, n_features), 0, n_values)
        test_labels = jax.random.randint(key_test, (n_test,), 0, n_classes)

        train_hvs = encoder.encode_batch(train_data)
        test_hvs = encoder.encode_batch(test_data)

        classifier = classifier.fit(train_hvs, train_labels)
        predictions = classifier.predict(test_hvs)
        accuracy = classifier.score(test_hvs, test_labels)

        assert predictions.shape == (n_test,)
        assert 0.0 <= float(accuracy) <= 1.0

    def test_discrete_features_bsc_centroid(self, prng_key, default_dimensions):
        """BSC encoder + centroid classifier pipeline."""
        n_features, n_values, n_classes = 8, 10, 4
        n_train, n_test = 80, 20

        model = BSC.create(dimensions=default_dimensions)
        encoder = RandomEncoder.create(
            num_features=n_features,
            num_values=n_values,
            dimensions=default_dimensions,
            vsa_model=model,
            key=prng_key,
        )
        classifier = CentroidClassifier.create(
            num_classes=n_classes,
            dimensions=default_dimensions,
            vsa_model=model,
        )

        key_train, key_test = jax.random.split(prng_key)
        train_data = jax.random.randint(key_train, (n_train, n_features), 0, n_values)
        train_labels = jax.random.randint(key_train, (n_train,), 0, n_classes)
        test_data = jax.random.randint(key_test, (n_test, n_features), 0, n_values)

        train_hvs = encoder.encode_batch(train_data)
        test_hvs = encoder.encode_batch(test_data)

        classifier = classifier.fit(train_hvs, train_labels)
        predictions = classifier.predict(test_hvs)

        assert predictions.shape == (n_test,)

    def test_continuous_level_encoder_centroid(self, prng_key, default_dimensions):
        """LevelEncoder for continuous values -> centroid.

        LevelEncoder encodes single continuous values; use 1D per sample.
        """
        n_samples, n_classes = 60, 3

        model = MAP.create(dimensions=default_dimensions)
        encoder = LevelEncoder.create(
            num_levels=50,
            dimensions=default_dimensions,
            min_value=0.0,
            max_value=1.0,
            vsa_model=model,
            key=prng_key,
        )
        classifier = CentroidClassifier.create(
            num_classes=n_classes,
            dimensions=default_dimensions,
            vsa_model=model,
        )

        key_data, key_labels = jax.random.split(prng_key)
        train_data = jax.random.uniform(key_data, (n_samples,))
        train_labels = jax.random.randint(key_labels, (n_samples,), 0, n_classes)

        train_hvs = encoder.encode_batch(train_data)
        classifier = classifier.fit(train_hvs, train_labels)

        test_data = jax.random.uniform(key_data, (10,))
        test_hvs = encoder.encode_batch(test_data)
        predictions = classifier.predict(test_hvs)

        assert predictions.shape == (10,)

    def test_projection_encoder_centroid(self, prng_key, default_dimensions):
        """ProjectionEncoder for high-dim input -> centroid."""
        input_dim, n_samples, n_classes = 100, 50, 4

        model = MAP.create(dimensions=default_dimensions)
        encoder = ProjectionEncoder.create(
            input_dim=input_dim,
            dimensions=default_dimensions,
            vsa_model=model,
            key=prng_key,
        )
        classifier = CentroidClassifier.create(
            num_classes=n_classes,
            dimensions=default_dimensions,
            vsa_model=model,
        )

        key_data, key_labels = jax.random.split(prng_key)
        train_data = jax.random.normal(key_data, (n_samples, input_dim))
        train_labels = jax.random.randint(key_labels, (n_samples,), 0, n_classes)

        train_hvs = encoder.encode_batch(train_data)
        classifier = classifier.fit(train_hvs, train_labels)

        test_data = jax.random.normal(key_data, (15, input_dim))
        test_hvs = encoder.encode_batch(test_data)
        predictions = classifier.predict(test_hvs)

        assert predictions.shape == (15,)

    def test_full_pipeline_adaptive_hdc(self, prng_key, default_dimensions):
        """Full pipeline with AdaptiveHDC (multi-epoch)."""
        n_features, n_values, n_classes = 8, 12, 4
        n_train, n_test = 60, 20

        model = MAP.create(dimensions=default_dimensions)
        encoder = RandomEncoder.create(
            num_features=n_features,
            num_values=n_values,
            dimensions=default_dimensions,
            vsa_model=model,
            key=prng_key,
        )
        classifier = AdaptiveHDC.create(
            num_classes=n_classes,
            dimensions=default_dimensions,
            vsa_model=model,
        )

        key_train, key_test = jax.random.split(prng_key)
        train_data = jax.random.randint(key_train, (n_train, n_features), 0, n_values)
        train_labels = jax.random.randint(key_train, (n_train,), 0, n_classes)
        test_data = jax.random.randint(key_test, (n_test, n_features), 0, n_values)
        test_labels = jax.random.randint(key_test, (n_test,), 0, n_classes)

        train_hvs = encoder.encode_batch(train_data)
        test_hvs = encoder.encode_batch(test_data)

        classifier = classifier.fit(train_hvs, train_labels, epochs=3)
        predictions = classifier.predict(test_hvs)
        accuracy = classifier.score(test_hvs, test_labels)

        assert predictions.shape == (n_test,)
        assert 0.0 <= float(accuracy) <= 1.0
