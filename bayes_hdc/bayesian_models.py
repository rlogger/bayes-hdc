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
    def logits(self, queries: jax.Array) -> jax.Array:
        """Pre-softmax cosine-similarity scores against each class posterior mean.

        This is the canonical input to :meth:`~bayes_hdc.TemperatureCalibrator.fit`
        and :meth:`~bayes_hdc.ConformalClassifier.fit`.

        Args:
            queries: Hypervector(s) of shape ``(D,)`` or ``(N, D)``.

        Returns:
            Cosine similarities of shape ``(num_classes,)`` for a single
            query or ``(N, num_classes)`` for a batch.
        """
        single = queries.ndim == 1
        batched = queries[None, :] if single else queries
        sims = jax.vmap(self._similarity_row)(batched)
        return sims[0] if single else sims

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


@register_dataclass
@dataclass
class BayesianAdaptiveHDC:
    r"""Streaming Gaussian-posterior classifier with Kalman updates.

    Starts from a broad prior per class and updates the class posterior
    one observation at a time via the conjugate-Gaussian (Kalman)
    update rule:

    .. math::
        \mu_{\text{new}} &= \frac{\sigma_\mathrm{obs}^2 \, \mu_{\text{old}}
                                  + \sigma_{\text{old}}^2 \, x}
                                 {\sigma_\mathrm{obs}^2 + \sigma_{\text{old}}^2} \\
        \sigma_{\text{new}}^2 &= \frac{\sigma_\mathrm{obs}^2 \, \sigma_{\text{old}}^2}
                                       {\sigma_\mathrm{obs}^2 + \sigma_{\text{old}}^2}

    The observation-noise variance :math:`\sigma_\mathrm{obs}^2` is a
    fixed hyperparameter controlling how aggressively each new sample
    is trusted vs. the current posterior.

    Compared to :class:`BayesianCentroidClassifier`, which does a
    single empirical-Bayes pass over the whole training set, this
    classifier supports streaming data, distribution shift, and
    anytime-valid uncertainty that depends on the number of
    observations per class seen so far.

    Attributes:
        mu: Class means, shape ``(num_classes, d)``.
        var: Class variances, shape ``(num_classes, d)``, non-negative.
        obs_var: Observation-noise variance (static hyperparameter).
        num_classes: Number of classes :math:`K` (static).
        dimensions: Hypervector dimensionality :math:`d` (static).
    """

    mu: jax.Array
    var: jax.Array
    obs_var: float = field(metadata=dict(static=True), default=0.1)
    num_classes: int = field(metadata=dict(static=True), default=2)
    dimensions: int = field(metadata=dict(static=True), default=10000)

    @staticmethod
    def create(
        num_classes: int,
        dimensions: int,
        prior_var: float = 1.0,
        obs_var: float = 0.1,
    ) -> BayesianAdaptiveHDC:
        """Construct an untrained classifier with an isotropic Gaussian prior."""
        return BayesianAdaptiveHDC(
            mu=jnp.zeros((num_classes, dimensions)),
            var=jnp.full((num_classes, dimensions), float(prior_var)),
            obs_var=float(obs_var),
            num_classes=num_classes,
            dimensions=dimensions,
        )

    def update(self, sample: jax.Array, label: int) -> BayesianAdaptiveHDC:
        """Single Kalman update for one (sample, label) pair."""
        mu_c = self.mu[label]
        var_c = self.var[label]
        denom = self.obs_var + var_c
        new_mu = (self.obs_var * mu_c + var_c * sample) / denom
        new_var = (self.obs_var * var_c) / denom
        return self.replace(
            mu=self.mu.at[label].set(new_mu),
            var=self.var.at[label].set(new_var),
        )

    def fit(
        self,
        train_hvs: jax.Array,
        train_labels: jax.Array,
        epochs: int = 1,
    ) -> BayesianAdaptiveHDC:
        """Apply Kalman updates sequentially over the training set.

        Implementation runs as a single :func:`jax.lax.scan` per epoch,
        so the whole pass JIT-compiles into one XLA computation and
        composes with ``vmap`` / ``pmap``. No Python-level loop over
        observations.

        Args:
            train_hvs: Training hypervectors of shape ``(n, d)``.
            train_labels: Integer labels of shape ``(n,)``.
            epochs: Number of passes through the data. More epochs
                tighten the posterior further (bounded below by the
                observation-noise floor).
        """
        if train_hvs.shape[0] == 0:
            raise ValueError("Cannot fit BayesianAdaptiveHDC: training data is empty")

        obs_var = self.obs_var

        def step(
            carry: tuple[jax.Array, jax.Array],
            sample_label: tuple[jax.Array, jax.Array],
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            mu, var = carry
            sample, label = sample_label
            mu_c = mu[label]
            var_c = var[label]
            denom = obs_var + var_c
            new_mu_c = (obs_var * mu_c + var_c * sample) / denom
            new_var_c = (obs_var * var_c) / denom
            return (mu.at[label].set(new_mu_c), var.at[label].set(new_var_c)), None

        labels = train_labels.astype(jnp.int32)
        mu, var = self.mu, self.var
        for _ in range(int(epochs)):
            (mu, var), _ = jax.lax.scan(step, (mu, var), (train_hvs, labels))
        return self.replace(mu=mu, var=var)

    @jax.jit
    def _similarity_row(self, query: jax.Array) -> jax.Array:
        q_norm = query / (jnp.linalg.norm(query) + EPS)
        mu_norm = self.mu / (jnp.linalg.norm(self.mu, axis=-1, keepdims=True) + EPS)
        return jnp.clip(mu_norm @ q_norm, -1.0, 1.0)

    @jax.jit
    def logits(self, queries: jax.Array) -> jax.Array:
        """Pre-softmax cosine-similarity scores. See :meth:`BayesianCentroidClassifier.logits`."""
        single = queries.ndim == 1
        batched = queries[None, :] if single else queries
        sims = jax.vmap(self._similarity_row)(batched)
        return sims[0] if single else sims

    @jax.jit
    def predict(self, queries: jax.Array) -> jax.Array:
        single = queries.ndim == 1
        batched = queries[None, :] if single else queries
        sims = jax.vmap(self._similarity_row)(batched)
        preds = jnp.argmax(sims, axis=-1)
        return preds[0] if single else preds

    @jax.jit
    def predict_proba(self, queries: jax.Array) -> jax.Array:
        single = queries.ndim == 1
        batched = queries[None, :] if single else queries
        sims = jax.vmap(self._similarity_row)(batched)
        probs = jax.nn.softmax(sims, axis=-1)
        return probs[0] if single else probs

    @jax.jit
    def predict_uncertainty(self, queries: jax.Array) -> jax.Array:
        """Per-class similarity variance (same semantics as BayesianCentroidClassifier)."""
        single = queries.ndim == 1
        batched = queries[None, :] if single else queries
        sim_vars = (batched**2) @ self.var.T
        return sim_vars[0] if single else sim_vars

    @jax.jit
    def score(self, test_hvs: jax.Array, test_labels: jax.Array) -> jax.Array:
        return jnp.mean(self.predict(test_hvs) == test_labels)

    def replace(self, **updates: Any) -> BayesianAdaptiveHDC:
        return dataclass_replace(self, **updates)


@register_dataclass
@dataclass
class StreamingBayesianHDC:
    r"""Bounded-memory streaming classifier with exponential decay.

    Maintains an exponentially-decayed running mean and variance per
    class, controlled by a ``decay`` factor :math:`\lambda \in (0, 1)`:

    .. math::
        \mu_\text{new} &= \lambda \mu_\text{old} + (1 - \lambda) x \\
        \sigma^2_\text{new} &= \lambda \sigma^2_\text{old}
                                + (1 - \lambda) (x - \mu_\text{new})^2

    This is the bounded-memory variant of :class:`BayesianAdaptiveHDC`:
    posterior memory is ``O(K × d)``, independent of stream length,
    and distribution shift is handled gracefully — old observations
    have exponentially decaying weight, so a changing data stream
    naturally re-fits the posterior without an explicit reset.

    Compared to :class:`BayesianAdaptiveHDC`, the Kalman-style updates
    there yield a strict posterior-narrowing sequence (variance can
    only shrink); the EMA here allows variance to grow if the
    incoming data is far from the current mean — the right behaviour
    for drifting streams.

    Attributes:
        mu: Running mean per class, shape ``(num_classes, d)``.
        var: Running variance per class, shape ``(num_classes, d)``.
        decay: EMA decay factor (static, :math:`\lambda`).
        num_classes: Number of classes :math:`K` (static).
        dimensions: Hypervector dimensionality :math:`d` (static).
    """

    mu: jax.Array
    var: jax.Array
    decay: float = field(metadata=dict(static=True), default=0.95)
    num_classes: int = field(metadata=dict(static=True), default=2)
    dimensions: int = field(metadata=dict(static=True), default=10000)

    @staticmethod
    def create(
        num_classes: int,
        dimensions: int,
        decay: float = 0.95,
        prior_var: float = 1.0,
    ) -> StreamingBayesianHDC:
        """Construct a streaming classifier with broad prior and specified decay."""
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1); got {decay}")
        return StreamingBayesianHDC(
            mu=jnp.zeros((num_classes, dimensions)),
            var=jnp.full((num_classes, dimensions), float(prior_var)),
            decay=float(decay),
            num_classes=num_classes,
            dimensions=dimensions,
        )

    def update(self, sample: jax.Array, label: int) -> StreamingBayesianHDC:
        """Single EMA update for one observation."""
        mu_old = self.mu[label]
        var_old = self.var[label]
        lam = self.decay

        new_mu = lam * mu_old + (1.0 - lam) * sample
        new_var = lam * var_old + (1.0 - lam) * (sample - new_mu) ** 2

        return self.replace(
            mu=self.mu.at[label].set(new_mu),
            var=self.var.at[label].set(new_var),
        )

    def fit(
        self,
        train_hvs: jax.Array,
        train_labels: jax.Array,
        epochs: int = 1,
    ) -> StreamingBayesianHDC:
        """Stream the training data through the classifier.

        Each epoch is a single :func:`jax.lax.scan`; the whole pass
        JIT-compiles into one XLA computation. Memory is ``O(K * d)``
        and stays constant across the stream.
        """
        if train_hvs.shape[0] == 0:
            raise ValueError("Cannot fit StreamingBayesianHDC: training data is empty")

        lam = self.decay

        def step(
            carry: tuple[jax.Array, jax.Array],
            sample_label: tuple[jax.Array, jax.Array],
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            mu, var = carry
            sample, label = sample_label
            mu_old = mu[label]
            var_old = var[label]
            new_mu = lam * mu_old + (1.0 - lam) * sample
            new_var = lam * var_old + (1.0 - lam) * (sample - new_mu) ** 2
            return (mu.at[label].set(new_mu), var.at[label].set(new_var)), None

        labels = train_labels.astype(jnp.int32)
        mu, var = self.mu, self.var
        for _ in range(int(epochs)):
            (mu, var), _ = jax.lax.scan(step, (mu, var), (train_hvs, labels))
        return self.replace(mu=mu, var=var)

    @jax.jit
    def _similarity_row(self, query: jax.Array) -> jax.Array:
        q_norm = query / (jnp.linalg.norm(query) + EPS)
        mu_norm = self.mu / (jnp.linalg.norm(self.mu, axis=-1, keepdims=True) + EPS)
        return jnp.clip(mu_norm @ q_norm, -1.0, 1.0)

    @jax.jit
    def logits(self, queries: jax.Array) -> jax.Array:
        """Pre-softmax cosine-similarity scores. See :meth:`BayesianCentroidClassifier.logits`."""
        single = queries.ndim == 1
        batched = queries[None, :] if single else queries
        sims = jax.vmap(self._similarity_row)(batched)
        return sims[0] if single else sims

    @jax.jit
    def predict(self, queries: jax.Array) -> jax.Array:
        single = queries.ndim == 1
        batched = queries[None, :] if single else queries
        sims = jax.vmap(self._similarity_row)(batched)
        preds = jnp.argmax(sims, axis=-1)
        return preds[0] if single else preds

    @jax.jit
    def predict_proba(self, queries: jax.Array) -> jax.Array:
        single = queries.ndim == 1
        batched = queries[None, :] if single else queries
        sims = jax.vmap(self._similarity_row)(batched)
        probs = jax.nn.softmax(sims, axis=-1)
        return probs[0] if single else probs

    @jax.jit
    def predict_uncertainty(self, queries: jax.Array) -> jax.Array:
        single = queries.ndim == 1
        batched = queries[None, :] if single else queries
        sim_vars = (batched**2) @ self.var.T
        return sim_vars[0] if single else sim_vars

    @jax.jit
    def score(self, test_hvs: jax.Array, test_labels: jax.Array) -> jax.Array:
        return jnp.mean(self.predict(test_hvs) == test_labels)

    def replace(self, **updates: Any) -> StreamingBayesianHDC:
        return dataclass_replace(self, **updates)


__all__ = [
    "BayesianCentroidClassifier",
    "BayesianAdaptiveHDC",
    "StreamingBayesianHDC",
]
