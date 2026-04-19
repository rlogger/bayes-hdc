# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Classification and learning models for Hyperdimensional Computing."""

from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp

from bayes_hdc import functional as F
from bayes_hdc._compat import register_dataclass
from bayes_hdc.constants import EPS
from bayes_hdc.vsa import VSAModel, create_vsa_model


@register_dataclass
@dataclass
class CentroidClassifier:
    """Centroid-based classifier for HDC.

    Stores one prototype hypervector per class. Classification finds the
    most similar prototype to the query.
    """

    prototypes: jax.Array  # (num_classes, dimensions)
    num_classes: int = field(metadata=dict(static=True))
    dimensions: int = field(metadata=dict(static=True))
    vsa_model_name: str = field(metadata=dict(static=True), default="map")

    @staticmethod
    def create(
        num_classes: int,
        dimensions: int = 10000,
        vsa_model: Union[str, VSAModel] = "map",
        initial_prototypes: Optional[jax.Array] = None,
        key: Optional[jax.Array] = None,
    ) -> "CentroidClassifier":
        """Create a centroid classifier.

        Args:
            num_classes: Number of classes
            dimensions: Dimensionality of hypervectors
            vsa_model: VSA model name or instance
            initial_prototypes: Optional initial prototypes of shape (num_classes, dimensions)
            key: JAX random key for initialization
        """
        if isinstance(vsa_model, str):
            vsa_model_name = vsa_model
            vsa = create_vsa_model(vsa_model, dimensions)
        else:
            vsa_model_name = vsa_model.name
            vsa = vsa_model

        if initial_prototypes is not None:
            prototypes = initial_prototypes
        else:
            if key is None:
                key = jax.random.PRNGKey(0)
            prototypes = vsa.random(key, shape=(num_classes, dimensions))

        return CentroidClassifier(
            prototypes=prototypes,
            num_classes=num_classes,
            dimensions=dimensions,
            vsa_model_name=vsa_model_name,
        )

    @jax.jit
    def similarity(self, query: jax.Array) -> jax.Array:
        """Compute similarity between query and all class prototypes."""
        if self.vsa_model_name == "bsc":
            return jax.vmap(lambda p: F.hamming_similarity(query, p))(self.prototypes)
        else:
            return jax.vmap(lambda p: F.cosine_similarity(query, p))(self.prototypes)

    @jax.jit
    def predict(self, queries: jax.Array) -> jax.Array:
        """Predict class labels for queries.

        Args:
            queries: Shape (batch_size, dimensions) or (dimensions,)

        Returns:
            Predicted class indices
        """
        is_single = queries.ndim == 1
        if is_single:
            queries = queries[None, :]

        similarities = jax.vmap(self.similarity)(queries)
        predictions = jnp.argmax(similarities, axis=-1)

        if is_single:
            return predictions[0]
        return predictions

    @jax.jit
    def predict_proba(self, queries: jax.Array) -> jax.Array:
        """Predict class probabilities using softmax of similarities."""
        is_single = queries.ndim == 1
        if is_single:
            queries = queries[None, :]

        similarities = jax.vmap(self.similarity)(queries)
        probs = jax.nn.softmax(similarities, axis=-1)

        if is_single:
            return probs[0]
        return probs

    def fit(self, train_hvs: jax.Array, train_labels: jax.Array) -> "CentroidClassifier":
        """Train classifier by computing class centroids.

        Args:
            train_hvs: Training hypervectors of shape (n_samples, dimensions)
            train_labels: Training labels of shape (n_samples,)

        Returns:
            Trained CentroidClassifier (new instance)
        """
        if train_hvs.shape[0] == 0:
            raise ValueError("Cannot fit CentroidClassifier: training data is empty")

        new_prototypes_list = []
        for class_idx in range(self.num_classes):
            class_mask = train_labels == class_idx
            num_samples = jnp.sum(class_mask)

            if num_samples > 0:
                weights = jnp.where(class_mask[:, None], 1.0, 0.0)
                if self.vsa_model_name == "bsc":
                    weighted_hvs = train_hvs.astype(jnp.float32) * weights
                    summed = jnp.sum(weighted_hvs, axis=0)
                    centroid = summed > (num_samples / 2.0)
                else:
                    weighted_hvs = train_hvs * weights
                    summed = jnp.sum(weighted_hvs, axis=0)
                    centroid = summed / (jnp.linalg.norm(summed) + EPS)
                new_prototypes_list.append(centroid)
            else:
                new_prototypes_list.append(self.prototypes[class_idx])

        return self.replace(prototypes=jnp.stack(new_prototypes_list))

    def update_online(
        self, sample_hv: jax.Array, label: int, learning_rate: float = 0.1
    ) -> "CentroidClassifier":
        """Update classifier online with a single sample."""
        old_prototype = self.prototypes[label]

        if self.vsa_model_name == "bsc":
            combined = jnp.stack([old_prototype, sample_hv])
            new_prototype = F.bundle_bsc(combined, axis=0)
        else:
            new_prototype = (1 - learning_rate) * old_prototype + learning_rate * sample_hv
            new_prototype = new_prototype / (jnp.linalg.norm(new_prototype) + EPS)

        return self.replace(prototypes=self.prototypes.at[label].set(new_prototype))

    @jax.jit
    def score(self, test_hvs: jax.Array, test_labels: jax.Array) -> jax.Array:
        """Compute accuracy on test data."""
        predictions = self.predict(test_hvs)
        return jnp.mean(predictions == test_labels)

    def replace(self, **updates: Any) -> "CentroidClassifier":
        return dataclass_replace(self, **updates)


@register_dataclass
@dataclass
class AdaptiveHDC:
    """Adaptive HDC classifier with iterative prototype refinement."""

    prototypes: jax.Array
    num_updates: jax.Array
    num_classes: int = field(metadata=dict(static=True))
    dimensions: int = field(metadata=dict(static=True))
    vsa_model_name: str = field(metadata=dict(static=True), default="map")

    @staticmethod
    def create(
        num_classes: int,
        dimensions: int = 10000,
        vsa_model: Union[str, VSAModel] = "map",
        key: Optional[jax.Array] = None,
    ) -> "AdaptiveHDC":
        if isinstance(vsa_model, str):
            vsa_model_name = vsa_model
            vsa = create_vsa_model(vsa_model, dimensions)
        else:
            vsa_model_name = vsa_model.name
            vsa = vsa_model

        if key is None:
            key = jax.random.PRNGKey(0)

        return AdaptiveHDC(
            prototypes=vsa.random(key, shape=(num_classes, dimensions)),
            num_updates=jnp.zeros(num_classes, dtype=jnp.int32),
            num_classes=num_classes,
            dimensions=dimensions,
            vsa_model_name=vsa_model_name,
        )

    @jax.jit
    def predict(self, queries: jax.Array) -> jax.Array:
        """Predict class labels."""
        is_single = queries.ndim == 1
        if is_single:
            queries = queries[None, :]

        if self.vsa_model_name == "bsc":
            similarities = jax.vmap(
                lambda q: jax.vmap(lambda p: F.hamming_similarity(q, p))(self.prototypes)
            )(queries)
        else:
            similarities = jax.vmap(
                lambda q: jax.vmap(lambda p: F.cosine_similarity(q, p))(self.prototypes)
            )(queries)

        predictions = jnp.argmax(similarities, axis=-1)

        if is_single:
            return predictions[0]
        return predictions

    def fit(
        self,
        train_hvs: jax.Array,
        train_labels: jax.Array,
        epochs: int = 1,
        learning_rate: float = 0.1,
    ) -> "AdaptiveHDC":
        """Train with iterative refinement.

        Args:
            train_hvs: Training hypervectors
            train_labels: Training labels
            epochs: Number of training epochs
            learning_rate: Learning rate for updates
        """
        if train_hvs.shape[0] == 0:
            raise ValueError("Cannot fit AdaptiveHDC: training data is empty")

        classifier = self
        for class_idx in range(self.num_classes):
            class_mask = train_labels == class_idx
            num_samples = jnp.sum(class_mask)

            if num_samples > 0:
                weights = jnp.where(class_mask[:, None], 1.0, 0.0)
                if self.vsa_model_name == "bsc":
                    weighted_hvs = train_hvs.astype(jnp.float32) * weights
                    summed = jnp.sum(weighted_hvs, axis=0)
                    centroid = summed > (num_samples / 2.0)
                else:
                    weighted_hvs = train_hvs * weights
                    summed = jnp.sum(weighted_hvs, axis=0)
                    centroid = summed / (jnp.linalg.norm(summed) + EPS)

                classifier = classifier.replace(
                    prototypes=classifier.prototypes.at[class_idx].set(centroid)
                )

        for _epoch in range(epochs):
            for i in range(len(train_hvs)):
                pred = classifier.predict(train_hvs[i])
                true_label = train_labels[i]

                if pred != true_label:
                    classifier = classifier._update_prototypes(
                        train_hvs[i], true_label, pred, learning_rate
                    )

        return classifier

    def _update_prototypes(
        self,
        sample_hv: jax.Array,
        true_label: Union[int, jax.Array],
        pred_label: Union[int, jax.Array],
        learning_rate: float,
    ) -> "AdaptiveHDC":
        true_proto = self.prototypes[true_label]

        if self.vsa_model_name != "bsc":
            new_true_proto = true_proto + learning_rate * sample_hv
            new_true_proto = new_true_proto / (jnp.linalg.norm(new_true_proto) + EPS)
        else:
            new_true_proto = F.bundle_bsc(jnp.stack([true_proto, sample_hv]), axis=0)

        return self.replace(prototypes=self.prototypes.at[true_label].set(new_true_proto))

    @jax.jit
    def score(self, test_hvs: jax.Array, test_labels: jax.Array) -> jax.Array:
        """Compute accuracy."""
        predictions = self.predict(test_hvs)
        return jnp.mean(predictions == test_labels)

    def replace(self, **updates: Any) -> "AdaptiveHDC":
        return dataclass_replace(self, **updates)


@register_dataclass
@dataclass
class LVQClassifier:
    """Learning Vector Quantization classifier.

    Prototypes are updated: move winner toward sample if correct,
    away if wrong.
    """

    prototypes: jax.Array
    num_classes: int = field(metadata=dict(static=True))
    dimensions: int = field(metadata=dict(static=True))
    vsa_model_name: str = field(metadata=dict(static=True), default="map")

    @staticmethod
    def create(
        num_classes: int,
        dimensions: int = 10000,
        vsa_model: Union[str, VSAModel] = "map",
        key: Optional[jax.Array] = None,
    ) -> "LVQClassifier":
        if isinstance(vsa_model, str):
            vsa = create_vsa_model(vsa_model, dimensions)
        else:
            vsa = vsa_model
        if key is None:
            key = jax.random.PRNGKey(0)
        return LVQClassifier(
            prototypes=vsa.random(key, (num_classes, dimensions)),
            num_classes=num_classes,
            dimensions=dimensions,
            vsa_model_name=vsa.name,
        )

    @jax.jit
    def predict(self, queries: jax.Array) -> jax.Array:
        """Predict class labels by nearest prototype."""
        is_single = queries.ndim == 1
        if is_single:
            queries = queries[None, :]
        if self.vsa_model_name == "bsc":
            sims = jax.vmap(
                lambda q: jax.vmap(lambda p: F.hamming_similarity(q, p))(self.prototypes)
            )(queries)
        else:
            sims = jax.vmap(
                lambda q: jax.vmap(lambda p: F.cosine_similarity(q, p))(self.prototypes)
            )(queries)
        preds = jnp.argmax(sims, axis=-1)
        return preds[0] if is_single else preds

    def fit(
        self,
        train_hvs: jax.Array,
        train_labels: jax.Array,
        epochs: int = 10,
        lr: float = 0.1,
    ) -> "LVQClassifier":
        """Train with LVQ updates (winner-take-all, move toward/away)."""
        if train_hvs.shape[0] == 0:
            raise ValueError("Cannot fit LVQClassifier: training data is empty")
        clf = self
        for _ in range(epochs):
            for i in range(len(train_hvs)):
                x, y_true = train_hvs[i], int(train_labels[i])
                pred = int(clf.predict(x))
                if pred == y_true:
                    delta = lr * (x - clf.prototypes[pred])
                else:
                    delta = -lr * (x - clf.prototypes[pred])
                if self.vsa_model_name != "bsc":
                    new_p = clf.prototypes[pred] + delta
                    new_p = new_p / (jnp.linalg.norm(new_p) + EPS)
                else:
                    new_p = F.bundle_bsc(
                        jnp.stack([clf.prototypes[pred], (clf.prototypes[pred] + delta) > 0.5]),
                        axis=0,
                    )
                clf = clf.replace(prototypes=clf.prototypes.at[pred].set(new_p))
        return clf

    @jax.jit
    def score(self, test_hvs: jax.Array, test_labels: jax.Array) -> jax.Array:
        preds = self.predict(test_hvs)
        return jnp.mean(preds == test_labels)

    def replace(self, **updates: Any) -> "LVQClassifier":
        return dataclass_replace(self, **updates)


@register_dataclass
@dataclass
class RegularizedLSClassifier:
    """Regularized Least Squares classifier in HV space.

    Solves (X^T X + lambda I) W = X^T Y for weights W.
    """

    weights: jax.Array  # (dimensions, num_classes)
    dimensions: int = field(metadata=dict(static=True))
    num_classes: int = field(metadata=dict(static=True))
    reg: float = field(metadata=dict(static=True))

    @staticmethod
    def create(
        dimensions: int,
        num_classes: int,
        reg: float = 1e-4,
    ) -> "RegularizedLSClassifier":
        return RegularizedLSClassifier(
            weights=jnp.zeros((dimensions, num_classes)),
            dimensions=dimensions,
            num_classes=num_classes,
            reg=reg,
        )

    def fit(self, train_hvs: jax.Array, train_labels: jax.Array) -> "RegularizedLSClassifier":
        """Fit by solving regularized least squares."""
        n = train_hvs.shape[0]
        if n == 0:
            raise ValueError("Cannot fit RegularizedLSClassifier: training data is empty")

        Y = jax.nn.one_hot(train_labels, self.num_classes)
        XtX = train_hvs.T @ train_hvs + self.reg * jnp.eye(self.dimensions)
        XtY = train_hvs.T @ Y
        weights, *_ = jnp.linalg.lstsq(XtX, XtY, rcond=None)
        return self.replace(weights=weights)

    @jax.jit
    def predict(self, queries: jax.Array) -> jax.Array:
        logits = queries @ self.weights
        return jnp.argmax(logits, axis=-1)

    @jax.jit
    def score(self, test_hvs: jax.Array, test_labels: jax.Array) -> jax.Array:
        preds = self.predict(test_hvs)
        return jnp.mean(preds == test_labels)

    def replace(self, **updates: Any) -> "RegularizedLSClassifier":
        return dataclass_replace(self, **updates)


@register_dataclass
@dataclass
class ClusteringModel:
    """HDC-style k-means clustering.

    Encodes data into hypervectors, then iteratively assigns clusters
    by cosine similarity and updates centroids by bundling.

    Inspired by the ClusteringModel in hdlib (Cumbo et al., 2023).
    """

    centroids: jax.Array
    dimensions: int = field(metadata=dict(static=True))
    num_clusters: int = field(metadata=dict(static=True))
    vsa_model_name: str = field(metadata=dict(static=True), default="map")

    @staticmethod
    def create(
        num_clusters: int,
        dimensions: int = 10000,
        vsa_model: Union[str, "VSAModel"] = "map",
        key: Optional[jax.Array] = None,
    ) -> "ClusteringModel":
        if key is None:
            key = jax.random.PRNGKey(0)

        if isinstance(vsa_model, str):
            vsa_model_name = vsa_model
        else:
            vsa_model_name = vsa_model.name

        centroids = jax.random.normal(key, (num_clusters, dimensions))
        norms = jnp.linalg.norm(centroids, axis=-1, keepdims=True)
        centroids = centroids / (norms + EPS)

        return ClusteringModel(
            centroids=centroids,
            dimensions=dimensions,
            num_clusters=num_clusters,
            vsa_model_name=vsa_model_name,
        )

    def fit(
        self,
        hvs: jax.Array,
        max_iters: int = 50,
    ) -> "ClusteringModel":
        """Fit clusters by iterating assignment and centroid update.

        Args:
            hvs: Hypervectors of shape (n, d)
            max_iters: Maximum iterations (default: 50)

        Returns:
            Updated ClusteringModel with refined centroids
        """
        centroids = self.centroids

        for _ in range(max_iters):
            sims = hvs @ centroids.T
            assignments = jnp.argmax(sims, axis=-1)

            new_centroids = []
            for k in range(self.num_clusters):
                mask = assignments == k
                count = jnp.sum(mask)
                cluster_sum = jnp.sum(hvs * mask[:, None], axis=0)
                fallback = centroids[k]
                centroid = jnp.where(count > 0, cluster_sum / (count + EPS), fallback)
                norm = jnp.linalg.norm(centroid) + EPS
                new_centroids.append(centroid / norm)

            stacked_centroids: jax.Array = jnp.stack(new_centroids)

            if jnp.allclose(stacked_centroids, centroids, atol=1e-6):
                break
            centroids = stacked_centroids

        return dataclass_replace(self, centroids=centroids)

    @jax.jit
    def predict(self, hvs: jax.Array) -> jax.Array:
        """Assign each hypervector to the closest centroid.

        Args:
            hvs: Hypervectors of shape (n, d) or (d,)

        Returns:
            Cluster assignments (scalar for single query, array for batch)
        """
        single = hvs.ndim == 1
        if single:
            hvs = hvs[None, :]
        sims = hvs @ self.centroids.T
        result = jnp.argmax(sims, axis=-1)
        return result[0] if single else result

    def replace(self, **updates: Any) -> "ClusteringModel":
        return dataclass_replace(self, **updates)


__all__ = [
    "CentroidClassifier",
    "AdaptiveHDC",
    "LVQClassifier",
    "RegularizedLSClassifier",
    "ClusteringModel",
]
