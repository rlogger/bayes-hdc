# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Bayesian learning models — classifiers that report uncertainty natively.

These classifiers sit on top of the PVSA layer (``distributions.py``)
rather than the deterministic VSA layer (``models.py``). They store a
posterior distribution per class and expose three output modes:

- ``predict(x)`` — argmax prediction (MAP);
- ``predict_proba(x)`` — softmax over expected similarities;
- ``predict_uncertainty(x)`` — per-class variance of the similarity
  score. Returns a ``(batch, num_classes)`` array that quantifies
  how certain the classifier is about each possible class assignment.

No existing HDC library ships a classifier that stores per-class
posteriors and reports class-conditional similarity variance. This is
an original contribution of the ``bayes-hdc`` library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from typing import Any

import jax
import jax.numpy as jnp

from bayes_hdc._compat import register_dataclass
from bayes_hdc.constants import EPS
from bayes_hdc.distributions import GaussianHV


@register_dataclass
@dataclass
class BayesianCentroidClassifier:
    r"""Gaussian-posterior classifier — one :class:`GaussianHV` per class.

    The classifier stores a posterior :math:`p(w_c) = \mathcal{N}(\mu_c,
    \mathrm{diag}(\sigma_c^2))` for each class's prototype
    hypervector. Fitted by empirical-Bayes: for every class ``c`` we
    take :math:`\mu_c` as the sample mean and :math:`\sigma_c^2` as
    the sample diagonal variance of the in-class training
    hypervectors, regularised by a prior of strength ``prior_strength``
    to prevent variance collapse on small classes.

    At test time the classifier exposes three modes:

    - :meth:`predict` — plain ``argmax`` over cosine similarities to
      the class means. Accuracy-compatible with
      :class:`~bayes_hdc.models.CentroidClassifier`.
    - :meth:`predict_proba` — softmax over cosine similarities;
      calibratable with :class:`~bayes_hdc.uncertainty.TemperatureCalibrator`.
    - :meth:`predict_uncertainty` — per-class variance of the
      dot-product similarity, computed in closed form from the stored
      posterior variances. A PVSA-exclusive signal not available from
      deterministic classifiers.

    Attributes:
        mu: Class means of shape ``(num_classes, d)``.
        var: Class diagonal variances of shape ``(num_classes, d)``;
            non-negative.
        num_classes: Number of classes :math:`K`.
        dimensions: Hypervector dimensionality :math:`d`.
    """

    mu: jax.Array
    var: jax.Array
    num_classes: int = field(metadata=dict(static=True))
    dimensions: int = field(metadata=dict(static=True))

    @staticmethod
    def create(
        num_classes: int,
        dimensions: int,
    ) -> BayesianCentroidClassifier:
        """Construct an untrained classifier with standard-normal priors."""
        return BayesianCentroidClassifier(
            mu=jnp.zeros((num_classes, dimensions)),
            var=jnp.ones((num_classes, dimensions)),
            num_classes=num_classes,
            dimensions=dimensions,
        )

    def fit(
        self,
        train_hvs: jax.Array,
        train_labels: jax.Array,
        prior_strength: float = 1.0,
    ) -> BayesianCentroidClassifier:
        r"""Fit empirical-Bayes Gaussian posterior per class.

        For each class :math:`c`:

        .. math::
            \mu_c &= \frac{1}{n_c} \sum_{i: y_i = c} x_i \\
            \tilde\sigma_c^2 &= \frac{1}{n_c} \sum_{i: y_i = c} (x_i - \mu_c)^2 \\
            \sigma_c^2 &= \frac{n_c \tilde\sigma_c^2 + \lambda}{n_c + \lambda}

        The third line is a Bayesian-regularised variance with prior
        strength :math:`\lambda = ` ``prior_strength``, which prevents
        variance from collapsing to zero for small classes and gives
        sensible uncertainty even in the limit :math:`n_c = 0`.

        Args:
            train_hvs: Training hypervectors of shape ``(n, d)``.
            train_labels: Integer labels of shape ``(n,)``.
            prior_strength: Non-negative Bayesian prior strength
                (default 1.0). Higher values → wider posteriors.

        Returns:
            A new :class:`BayesianCentroidClassifier` with fitted moments.
        """
        if train_hvs.shape[0] == 0:
            raise ValueError("Cannot fit BayesianCentroidClassifier: training data is empty")

        new_mu = []
        new_var = []
        for c in range(self.num_classes):
            mask = train_labels == c
            n_c = jnp.sum(mask.astype(jnp.float32))
            safe_n = jnp.maximum(n_c, 1.0)

            class_hvs = jnp.where(mask[:, None], train_hvs, 0.0)
            mu_c = jnp.sum(class_hvs, axis=0) / safe_n

            centered_sq = jnp.where(mask[:, None], (train_hvs - mu_c) ** 2, 0.0)
            sample_var = jnp.sum(centered_sq, axis=0) / safe_n
            var_c = (n_c * sample_var + prior_strength) / (n_c + prior_strength)

            # Empty-class fallback: retain broad prior.
            mu_c = jnp.where(n_c > 0, mu_c, self.mu[c])
            var_c = jnp.where(n_c > 0, var_c, self.var[c])

            new_mu.append(mu_c)
            new_var.append(var_c)

        return self.replace(mu=jnp.stack(new_mu), var=jnp.stack(new_var))

    def class_posterior(self, class_idx: int) -> GaussianHV:
        """Return the stored posterior for class ``class_idx`` as a GaussianHV."""
        return GaussianHV(
            mu=self.mu[class_idx],
            var=self.var[class_idx],
            dimensions=self.dimensions,
        )

    @jax.jit
    def _similarity_row(self, query: jax.Array) -> jax.Array:
        """Cosine similarity from a single ``query`` to every class mean."""
        q_norm = query / (jnp.linalg.norm(query) + EPS)
        mu_norm = self.mu / (jnp.linalg.norm(self.mu, axis=-1, keepdims=True) + EPS)
        return jnp.clip(mu_norm @ q_norm, -1.0, 1.0)

    @jax.jit
    def predict(self, queries: jax.Array) -> jax.Array:
        """Argmax over cosine similarities."""
        single = queries.ndim == 1
        batched = queries[None, :] if single else queries
        sims = jax.vmap(self._similarity_row)(batched)
        preds = jnp.argmax(sims, axis=-1)
        return preds[0] if single else preds

    @jax.jit
    def predict_proba(self, queries: jax.Array) -> jax.Array:
        """Softmax over cosine similarities — class probabilities."""
        single = queries.ndim == 1
        batched = queries[None, :] if single else queries
        sims = jax.vmap(self._similarity_row)(batched)
        probs = jax.nn.softmax(sims, axis=-1)
        return probs[0] if single else probs

    @jax.jit
    def predict_uncertainty(self, queries: jax.Array) -> jax.Array:
        r"""Per-class variance of the dot-product similarity.

        For a query :math:`x` (treated as a zero-variance point) and a
        class posterior :math:`W_c \sim \mathcal{N}(\mu_c, \Sigma_c)`
        with diagonal :math:`\Sigma_c = \mathrm{diag}(\sigma_c^2)`:

        .. math::
            \mathrm{Var}[\langle x, W_c \rangle]
            = \sum_i x_i^2 \sigma_{c,i}^2

        High values for a given class indicate the classifier is
        uncertain about that assignment.

        Returns:
            Shape ``(num_classes,)`` for a single query, ``(batch,
            num_classes)`` for a batch.
        """
        single = queries.ndim == 1
        batched = queries[None, :] if single else queries
        q_sq = batched**2
        sim_vars = q_sq @ self.var.T
        return sim_vars[0] if single else sim_vars

    @jax.jit
    def predict_with_uncertainty(
        self, queries: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return ``(preds, probs, sim_variances)`` in one pass."""
        single = queries.ndim == 1
        batched = queries[None, :] if single else queries
        sims = jax.vmap(self._similarity_row)(batched)
        probs = jax.nn.softmax(sims, axis=-1)
        preds = jnp.argmax(sims, axis=-1)
        q_sq = batched**2
        sim_vars = q_sq @ self.var.T
        if single:
            return preds[0], probs[0], sim_vars[0]
        return preds, probs, sim_vars

    @jax.jit
    def score(self, test_hvs: jax.Array, test_labels: jax.Array) -> jax.Array:
        """Classification accuracy on a labelled test set."""
        preds = self.predict(test_hvs)
        return jnp.mean(preds == test_labels)

    def replace(self, **updates: Any) -> BayesianCentroidClassifier:
        return dataclass_replace(self, **updates)


__all__ = ["BayesianCentroidClassifier"]
