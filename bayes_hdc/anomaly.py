# SPDX-License-Identifier: MIT
# Copyright (c) 2026 R.S.

"""Calibrated anomaly detection for hyperdimensional computing.

This module ships a split-conformal one-class anomaly detector built on
top of HDC nonconformity scores. It composes two layers:

- :class:`HDCAnomalyScorer` — a *distance-metric-aware* nonconformity
  score. Given a learned reference set of "normal" hypervectors, the
  scorer maps a query hypervector to a non-negative scalar where larger
  values are more anomalous. Three score functions are supported:
  cosine-distance-to-centroid (the default for real-valued VSAs),
  Hamming-distance-to-centroid (for BSC), and max-similarity over the
  k nearest reference vectors (a kernel-style alternative that does not
  collapse the reference set into a single prototype).

- :class:`ConformalAnomalyDetector` — a split-conformal wrapper around
  any scorer. :meth:`fit` records the scorer's outputs on a held-out
  calibration set of normal data; :meth:`pvalue` returns a conformal
  *p*-value super-uniform under the exchangeability null,
  and :meth:`predict` thresholds the p-value at level :math:`\\alpha`
  with a finite-sample false-positive-rate (FPR) guarantee.

The scorer must be fitted on a training split independent of calibration
and test observations. Conservative conformal p-values are super-uniform
under exchangeability; ties make them more conservative. Calibration and
thresholding are discrete, while continuous cosine scores support gradients.

References:
    Bates, Candès, Lei, Romano & Sesia (2023), Testing for Outliers with
    Conformal p-Values, Annals of Statistics 51(1), 149–178.
    https://arxiv.org/abs/2104.08279
    Vovk, Gammerman & Shafer (2005), Algorithmic Learning in a Random
    World, Springer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp

from bayes_hdc._compat import register_dataclass
from bayes_hdc._validation import finite_array, positive_int
from bayes_hdc.constants import EPS

# ----------------------------------------------------------------------
# HDC nonconformity scorer
# ----------------------------------------------------------------------


@register_dataclass
@dataclass
class HDCAnomalyScorer:
    r"""Distance-metric-aware HDC nonconformity scorer.

    Given a learned reference set
    :math:`\mathcal{R} = \{r_1, \dots, r_n\} \subset \mathbb{R}^d` (or
    :math:`\{0,1\}^d` for BSC) of "normal" hypervectors, the scorer
    maps a query :math:`x \in \mathbb{R}^d` to a non-negative scalar

    .. math::
        s(x; \mathcal{R}) \;=\;
        \begin{cases}
            1 - \mathrm{sim}(x, \bar r) & k_{\text{nbrs}} = 1 \\
            1 - \displaystyle \frac{1}{k}\!\!\sum_{r \in \mathrm{topk}(x, k)}
                \mathrm{sim}(x, r) & k_{\text{nbrs}} = k > 1
        \end{cases}

    where :math:`\bar r` is the (normalised) centroid of
    :math:`\mathcal{R}` and :math:`\mathrm{sim}` is cosine or Hamming
    similarity depending on the VSA model. Larger :math:`s` means more
    anomalous. The score is bounded in :math:`[0, 2]` for cosine
    (typically in :math:`[0, 1]` after normalisation) and in
    :math:`[0, 1]` for Hamming.

    The k-nearest variant preserves multiple modes in the reference
    set instead of collapsing it into a single centroid. Detection power
    depends on the encoder, metric, and data distribution.

    Attributes:
        centroid: Bundled / mean reference hypervector of shape
            ``(dimensions,)``. For BSC it is a thresholded majority
            bundle; for real-valued VSAs it is the unit-normalised
            mean.
        reference: Full reference set of shape
            ``(n_reference, dimensions)``. Stored only when
            ``k_neighbors > 1``; otherwise filled with zeros so the
            pytree shape is static.
        dimensions: Hypervector dimension (static).
        distance_metric: Either ``"cosine"`` or ``"hamming"`` (static).
        k_neighbors: Number of nearest reference vectors to average
            over. ``k_neighbors == 1`` uses the centroid alone (static).
        vsa_model_name: Name of the underlying VSA model. Used only to
            select the default distance metric (static).
        fitted: Whether a reference set has been learned (static). A
            fitted centroid may legitimately be the zero vector.

    Example:
        >>> import jax, jax.numpy as jnp
        >>> from bayes_hdc.anomaly import HDCAnomalyScorer
        >>> key = jax.random.PRNGKey(0)
        >>> normal = jax.random.normal(key, (50, 1024))
        >>> scorer = HDCAnomalyScorer.create(dimensions=1024).fit(normal)
        >>> in_dist = float(scorer.score(normal[0]))
        >>> far_away = float(scorer.score(normal[0] + 100.0))
        >>> in_dist < far_away   # in-distribution scores lower than an outlier
        True
    """

    centroid: jax.Array  # (dimensions,)
    reference: jax.Array  # (n_reference, dimensions) — zeros if unused
    dimensions: int = field(metadata=dict(static=True))
    distance_metric: str = field(metadata=dict(static=True), default="cosine")
    k_neighbors: int = field(metadata=dict(static=True), default=1)
    vsa_model_name: str = field(metadata=dict(static=True), default="map")
    fitted: bool = field(metadata=dict(static=True), default=False)

    @staticmethod
    def create(
        dimensions: int = 10000,
        vsa_model: str | Any = "map",
        distance_metric: str | None = None,
        k_neighbors: int = 1,
        n_reference: int = 0,
    ) -> HDCAnomalyScorer:
        """Build an empty scorer. Call :meth:`fit` with normal hypervectors.

        Args:
            dimensions: Hypervector dimensionality.
            vsa_model: VSA model name (``"bsc"``, ``"map"``, ``"hrr"``, ...)
                or a :class:`~bayes_hdc.vsa.VSAModel` instance. Used
                only to pick the default distance metric.
            distance_metric: ``"cosine"`` or ``"hamming"``. Defaults to
                ``"hamming"`` for BSC and ``"cosine"`` otherwise.
            k_neighbors: ``1`` for centroid mode (cheap, default), or
                ``k > 1`` to score against the average similarity over
                the k nearest reference vectors. Larger ``k`` is more
                robust to multi-modal normal distributions.
            n_reference: Pre-allocate reference buffer to this many
                rows. Pass the size of the calibration set when
                ``k_neighbors > 1`` so :meth:`fit` does not need to
                reshape the pytree.

        Returns:
            An unfitted ``HDCAnomalyScorer``.
        """
        positive_int(dimensions, "dimensions")
        positive_int(k_neighbors, "k_neighbors")
        if isinstance(vsa_model, str):
            vsa_model_name = vsa_model
        else:
            vsa_model_name = getattr(vsa_model, "name", "map")

        if distance_metric is None:
            distance_metric = "hamming" if vsa_model_name == "bsc" else "cosine"
        if distance_metric not in ("cosine", "hamming"):
            raise ValueError(
                f"distance_metric must be 'cosine' or 'hamming', got {distance_metric!r}"
            )
        if k_neighbors < 1:
            raise ValueError(f"k_neighbors must be >= 1, got {k_neighbors}")
        if n_reference < 0:
            raise ValueError(f"n_reference must be >= 0, got {n_reference}")

        if distance_metric == "hamming":
            centroid = jnp.zeros((dimensions,), dtype=jnp.bool_)
        else:
            dtype = jnp.complex64 if vsa_model_name == "fhrr" else jnp.float32
            centroid = jnp.zeros((dimensions,), dtype=dtype)

        reference = jnp.zeros((max(n_reference, 0), dimensions), dtype=centroid.dtype)

        return HDCAnomalyScorer(
            centroid=centroid,
            reference=reference,
            dimensions=int(dimensions),
            distance_metric=distance_metric,
            k_neighbors=int(k_neighbors),
            vsa_model_name=vsa_model_name,
        )

    def fit(self, normal_hypervectors: jax.Array) -> HDCAnomalyScorer:
        """Learn the reference centroid (and reference set if ``k_neighbors > 1``).

        Args:
            normal_hypervectors: Array of shape ``(n, dimensions)`` of
                hypervectors drawn from the in-distribution / normal
                class.

        Returns:
            A new ``HDCAnomalyScorer`` with ``centroid`` (and
            ``reference``) populated.

        Raises:
            ValueError: If ``normal_hypervectors`` is empty.
            ValueError: If the second axis does not match
                ``self.dimensions``.
        """
        hvs = jnp.asarray(normal_hypervectors)
        if hvs.ndim != 2:
            raise ValueError(
                f"normal_hypervectors must be 2-D (n, dimensions); got shape {hvs.shape}"
            )
        n = int(hvs.shape[0])
        if n == 0:
            raise ValueError("Cannot fit HDCAnomalyScorer: normal_hypervectors is empty (n=0).")
        if int(hvs.shape[1]) != self.dimensions:
            raise ValueError(
                f"normal_hypervectors has dimension {hvs.shape[1]}, "
                f"but scorer was created with dimensions={self.dimensions}"
            )

        finite_array(hvs, "normal_hypervectors")
        if self.distance_metric == "hamming":
            # Majority bundle for BSC. Cast booleans to float for the sum.
            hvs_float = hvs.astype(jnp.float32)
            summed = jnp.sum(hvs_float, axis=0)
            centroid = summed > (n / 2.0)
        else:
            summed = jnp.sum(hvs.astype(jnp.result_type(hvs.dtype, jnp.float32)), axis=0)
            centroid = summed / (jnp.linalg.norm(summed) + EPS)

        if self.k_neighbors > 1:
            # Store the full reference set; cast to centroid dtype so the
            # pytree stays homogeneous across distance metrics.
            reference = hvs.astype(centroid.dtype)
        else:
            # Keep the pytree shape static by leaving reference as-is.
            reference = self.reference

        return HDCAnomalyScorer(
            centroid=centroid,
            reference=reference,
            dimensions=self.dimensions,
            distance_metric=self.distance_metric,
            k_neighbors=self.k_neighbors,
            vsa_model_name=self.vsa_model_name,
            fitted=True,
        )

    def _similarity(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Pairwise similarity used inside the score, dispatched by metric."""
        if self.distance_metric == "hamming":
            # Normalised Hamming similarity in [0, 1].
            matches = jnp.logical_not(jnp.logical_xor(x.astype(jnp.bool_), y.astype(jnp.bool_)))
            return jnp.mean(matches.astype(jnp.float32), axis=-1)
        # Cosine similarity in [-1, 1].
        dtype = jnp.result_type(x.dtype, y.dtype, jnp.float32)
        x_f = x.astype(dtype)
        y_f = y.astype(dtype)
        x_norm = x_f / (jnp.linalg.norm(x_f, axis=-1, keepdims=True) + EPS)
        y_norm = y_f / (jnp.linalg.norm(y_f, axis=-1, keepdims=True) + EPS)
        return jnp.clip(jnp.real(jnp.sum(jnp.conj(x_norm) * y_norm, axis=-1)), -1.0, 1.0)

    def score(self, query: jax.Array) -> jax.Array:
        r"""Compute the nonconformity score of a single ``query``.

        Returns:
            Scalar ``jax.Array`` :math:`s(x; \mathcal{R}) \geq 0`,
            larger = more anomalous.
        """
        if not self.fitted:
            raise ValueError("HDCAnomalyScorer must be fitted before scoring")
        query = jnp.asarray(query)
        if query.shape != (self.dimensions,):
            raise ValueError(f"query must have shape ({self.dimensions},)")
        if self.k_neighbors <= 1:
            sim = self._similarity(query, self.centroid)
            return 1.0 - sim
        # k-NN average over top-k similarities.
        sims = jax.vmap(lambda r: self._similarity(query, r))(self.reference)
        k = min(self.k_neighbors, int(self.reference.shape[0]))
        # `jax.lax.top_k` returns the largest k. We want the k *largest*
        # similarities (= k smallest distances = k nearest neighbours).
        top_sims, _ = jax.lax.top_k(sims, k)
        return 1.0 - jnp.mean(top_sims)

    def score_batch(self, queries: jax.Array) -> jax.Array:
        """Vectorised score over a batch of queries.

        Args:
            queries: Shape ``(batch, dimensions)``.

        Returns:
            Scores of shape ``(batch,)``.
        """
        return jax.vmap(self.score)(queries)

    def replace(self, **updates: Any) -> HDCAnomalyScorer:
        """Pytree-friendly functional update."""
        from dataclasses import replace as _replace

        return _replace(self, **updates)


# ----------------------------------------------------------------------
# Split-conformal one-class anomaly detector
# ----------------------------------------------------------------------


@register_dataclass
@dataclass
class ConformalAnomalyDetector:
    r"""Split-conformal one-class anomaly detector with calibrated FPR.

    Wraps an :class:`HDCAnomalyScorer` and a held-out calibration set
    of normal nonconformity scores
    :math:`\{\alpha_1, \dots, \alpha_n\}`. For a query :math:`x` with
    score :math:`\alpha(x)`, the conformal p-value is

    .. math::
        p(x) \;=\; \frac{1 + |\{i : \alpha_i \geq \alpha(x)\}|}{n + 1},

    which is super-uniform under exchangeability of calibration and normal
    test observations (Bates et al., 2023). Without ties its marginal
    distribution is uniform on the finite grid ``{1/(n+1), ..., 1}``,
    not continuous Uniform[0, 1]. The
    decision rule

    .. math::
        \widehat{\text{anomaly}}(x) \;=\; \mathbb{1}\{p(x) \leq \alpha\}

    has finite-sample false-positive rate at most :math:`\alpha` on
    exchangeable normal test data, with no assumption on the
    distribution or on the quality of the underlying score. The
    coverage guarantee is

    .. math::
        \mathbb{P}_{x \sim P_{\text{normal}}}\!\left(p(x) \leq \alpha\right)
        \;\leq\; \alpha.

    Attributes:
        scorer: The fitted :class:`HDCAnomalyScorer`.
        calibration_scores: Sorted (ascending) calibration scores of
            shape ``(n_calibration,)``.
        n_calibration: Calibration-set size (static, for the
            :math:`(n+1)` correction).

    Example:
        >>> import jax, jax.numpy as jnp
        >>> from bayes_hdc.anomaly import HDCAnomalyScorer, ConformalAnomalyDetector
        >>> key = jax.random.PRNGKey(0)
        >>> normal = jax.random.normal(key, (200, 1024))
        >>> scorer = HDCAnomalyScorer.create(dimensions=1024).fit(normal[:100])
        >>> detector = ConformalAnomalyDetector.create(scorer).fit(normal[100:])
        >>> p = float(detector.pvalue(normal[0]))    # in-distribution -> not extreme
        >>> p_out = float(detector.pvalue(-normal[0]))  # anti-correlated -> anomalous
        >>> p > p_out and p_out < 0.05
        True
    """

    scorer: HDCAnomalyScorer
    calibration_scores: jax.Array  # (n_calibration,), sorted ascending
    n_calibration: int = field(metadata=dict(static=True), default=0)
    fitted: bool = field(metadata=dict(static=True), default=False)
    alpha: float = field(metadata=dict(static=True), default=0.05)

    @staticmethod
    def create(
        scorer: HDCAnomalyScorer,
        n_calibration: int = 0,
    ) -> ConformalAnomalyDetector:
        """Build an unfitted detector around a (possibly unfitted) scorer.

        Args:
            scorer: An :class:`HDCAnomalyScorer`. Fit it on a separate
                training split before calling this detector's :meth:`fit`.
            n_calibration: Pre-allocate the calibration buffer to this
                many entries. Pass the size of the calibration set so
                the pytree shape is static under JIT.

        Returns:
            An unfitted ``ConformalAnomalyDetector``.
        """
        if not isinstance(scorer, HDCAnomalyScorer):
            raise TypeError(f"scorer must be an HDCAnomalyScorer, got {type(scorer).__name__}")
        if n_calibration < 0:
            raise ValueError(f"n_calibration must be >= 0, got {n_calibration}")

        calibration_scores = jnp.zeros((max(n_calibration, 0),), dtype=jnp.float32)
        return ConformalAnomalyDetector(
            scorer=scorer,
            calibration_scores=calibration_scores,
            n_calibration=int(n_calibration),
        )

    def fit(self, normal_data_hvs: jax.Array) -> ConformalAnomalyDetector:
        """Learn the calibration distribution of normal nonconformity scores.

        The scorer is left untouched (use :meth:`HDCAnomalyScorer.fit`
        on a separate "training" split, then call this :meth:`fit` on a
        held-out calibration split). An unfitted scorer is rejected:
        fitting it on calibration data would invalidate the finite-sample
        false-positive-rate guarantee.

        Args:
            normal_data_hvs: Calibration hypervectors of shape
                ``(n_calibration, dimensions)``. All assumed to be
                drawn i.i.d. (or at least exchangeably) from the
                normal class.

        Returns:
            A new ``ConformalAnomalyDetector`` with calibration scores
            stored.

        Raises:
            ValueError: If ``normal_data_hvs`` is empty.
        """
        hvs = jnp.asarray(normal_data_hvs)
        if hvs.ndim != 2:
            raise ValueError(f"normal_data_hvs must be 2-D (n, dimensions); got shape {hvs.shape}")
        n = int(hvs.shape[0])
        if n == 0:
            raise ValueError("Cannot fit ConformalAnomalyDetector: normal_data_hvs is empty (n=0).")

        if hvs.shape[1] != self.scorer.dimensions:
            raise ValueError(f"normal_data_hvs must have {self.scorer.dimensions} columns")
        finite_array(hvs, "normal_data_hvs")
        scorer = self.scorer
        if not scorer.fitted:
            raise ValueError("Fit the scorer on a separate training split before calibration")

        cal_scores = scorer.score_batch(hvs)
        cal_scores = jnp.sort(cal_scores)

        return ConformalAnomalyDetector(
            scorer=scorer,
            calibration_scores=cal_scores,
            n_calibration=n,
            fitted=True,
            alpha=self.alpha,
        )

    def score(self, query_hv: jax.Array) -> jax.Array:
        """Raw nonconformity score *before* conformalisation.

        Returns:
            Scalar ``jax.Array``; identical to
            ``scorer.score(query_hv)``.
        """
        return self.scorer.score(query_hv)

    def pvalue(self, query_hv: jax.Array) -> jax.Array:
        r"""Conformal p-value :math:`p(x) \in (0, 1]`.

        :math:`p(x) = \frac{1 + |\{i : \alpha_i \geq \alpha(x)\}|}{n + 1}`.
        Super-uniform under the exchangeability null;
        smaller values are stronger evidence against normality.

        Args:
            query_hv: Query hypervector of shape ``(dimensions,)``.

        Returns:
            Scalar p-value as ``jax.Array``.
        """
        if not self.fitted:
            raise ValueError("ConformalAnomalyDetector must be fitted before pvalue")
        score = self.score(query_hv)
        n = self.n_calibration
        # ``calibration_scores`` is stored sorted ascending (see ``fit``), so the
        # number of calibration scores >= ``score`` is an O(log n) binary search
        # rather than an O(n) scan. ``side="left"`` returns the first index whose
        # value is >= ``score``; everything from there to the end is the >= count.
        # This is exact (identical to the masked sum) and vmaps cleanly.
        ge_count = n - jnp.searchsorted(self.calibration_scores, score, side="left")
        return (1.0 + ge_count) / (n + 1.0)

    def pvalue_batch(self, queries: jax.Array) -> jax.Array:
        """Vectorised p-values over a batch of queries.

        Args:
            queries: Shape ``(batch, dimensions)``.

        Returns:
            P-values of shape ``(batch,)``.
        """
        return jax.vmap(self.pvalue)(queries)

    def predict(self, query_hv: jax.Array, alpha: float | None = None) -> jax.Array:
        r"""Boolean anomaly flag at miscoverage level :math:`\alpha`.

        Returns ``True`` iff the conformal p-value is at most
        :math:`\alpha`. The marginal false-positive rate of this rule
        is bounded by :math:`\alpha` under exchangeability
        (Bates et al., 2023).

        Args:
            query_hv: Query hypervector of shape ``(dimensions,)``.
            alpha: Target false-positive rate in :math:`(0, 1)`. The
                value is read at call time, so the same fitted detector
                can be queried at multiple :math:`\alpha`.

        Returns:
            Boolean ``jax.Array`` scalar.

        Raises:
            ValueError: If ``alpha`` is not strictly in :math:`(0, 1)`.
        """
        alpha = self.alpha if alpha is None else alpha
        if not isinstance(alpha, jax.core.Tracer) and not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        return self.pvalue(query_hv) <= alpha

    def predict_batch(self, queries: jax.Array, alpha: float | None = None) -> jax.Array:
        """Vectorised :meth:`predict` over a batch of queries.

        Args:
            queries: Shape ``(batch, dimensions)``.
            alpha: Target false-positive rate in :math:`(0, 1)`.

        Returns:
            Boolean array of shape ``(batch,)``.
        """
        alpha = self.alpha if alpha is None else alpha
        if not isinstance(alpha, jax.core.Tracer) and not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        return self.pvalue_batch(queries) <= alpha

    def predict_fdr(self, queries: jax.Array, q: float = 0.1) -> jax.Array:
        r"""Flag anomalies in a batch with false-discovery-rate control.

        Applies the Benjamini-Hochberg (BH) procedure to the conformal
        p-values of the batch. Under the independent-test-sample
        assumptions of Bates et al. (including independent normal test
        observations and an independent proper-training split), these
        p-values are positively dependent (PRDS), so BH controls the expected fraction
        of *false discoveries* among the flagged points at level
        :math:`q` (Bates, Candès, Lei, Romano & Sesia, 2023, *Testing for
        outliers with conformal p-values*, Annals of Statistics):

        .. math::
            \mathbb{E}\!\left[
              \frac{\#\{\text{true normal points flagged}\}}
                   {\#\{\text{points flagged}\} \vee 1}
            \right] \le q .

        This is the right control when you score *many* points at once
        and care about the purity of the alarm set, rather than the
        per-point false-positive rate that :meth:`predict_batch`
        controls. The two answer different questions; use FDR control
        for batch screening (scan logs / a wafer / a patient cohort),
        per-point :math:`\alpha` for an online single-point decision.

        Args:
            queries: Batch of query hypervectors, shape
                ``(batch, dimensions)``.
            q: Target false-discovery rate in :math:`(0, 1)`.

        Returns:
            Boolean array of shape ``(batch,)`` — ``True`` where the
            point is flagged as an anomaly under BH at level ``q``.

        Raises:
            ValueError: If ``q`` is not strictly in :math:`(0, 1)`.
        """
        if not isinstance(q, jax.core.Tracer) and not (0.0 < float(q) < 1.0):
            raise ValueError(f"q must be in (0, 1), got {q}")
        p = self.pvalue_batch(queries)
        m = p.shape[0]
        if m == 0:
            return jnp.zeros((0,), dtype=jnp.bool_)
        sorted_p = jnp.sort(p)
        # BH critical line (i/m) * q for i = 1..m.
        crit = (jnp.arange(1, m + 1) / m) * q
        below = sorted_p <= crit
        # Largest rank k (1-indexed) with p_(k) <= (k/m) q; 0 if none.
        ranks = jnp.where(below, jnp.arange(1, m + 1), 0)
        k = jnp.max(ranks)
        # Rejection threshold is p_(k); reject everything <= it. If k == 0,
        # use a negative threshold so nothing (p >= 0) is flagged.
        threshold = jnp.where(k > 0, sorted_p[jnp.maximum(k - 1, 0)], -1.0)
        return p <= threshold

    def replace(self, **updates: Any) -> ConformalAnomalyDetector:
        """Pytree-friendly functional update."""
        from dataclasses import replace as _replace

        return _replace(self, **updates)


# ----------------------------------------------------------------------
# Convenience pipeline
# ----------------------------------------------------------------------


def fit_anomaly_pipeline(
    encoder: Any,
    normal_data: jax.Array,
    calibration_data: jax.Array,
    alpha: float = 0.05,
    distance_metric: str | None = None,
    k_neighbors: int = 1,
) -> ConformalAnomalyDetector:
    r"""Encode raw features then fit a split-conformal anomaly detector.

    Convenience wrapper for split-conformal outlier testing
    (Bates et al., 2023), with an HDC scoring model:

    1. Encode raw features ``normal_data`` and ``calibration_data``
       with ``encoder.encode_batch`` to produce hypervectors.
    2. Fit an :class:`HDCAnomalyScorer` on the encoded normal data
       (the *proper-training* split).
    3. Fit a :class:`ConformalAnomalyDetector` on the encoded
       calibration data (the *calibration* split). The scorer is
       reused — the calibration data only updates the empirical
       distribution of nonconformity scores, not the centroid.

    This protocol gives the strict marginal FPR :math:`\leq \alpha`
    guarantee: ``normal_data`` and ``calibration_data`` must come from
    *disjoint* splits exchangeable with the test distribution. Calling
    this function with the same data for both arguments is in-sample
    and does not have this guarantee.

    Args:
        encoder: Any object exposing ``encode_batch(features) ->
            hypervectors`` and a ``dimensions`` attribute (e.g.
            :class:`~bayes_hdc.embeddings.ProjectionEncoder`).
        normal_data: Raw features of shape
            ``(n_train, *feature_shape)`` from the in-distribution /
            normal class.
        calibration_data: Held-out raw features of shape
            ``(n_cal, *feature_shape)``, also from the normal class.
        alpha: Target false-positive rate (kept on the returned
            detector for the user's convenience; the actual decision
            is made inside :meth:`ConformalAnomalyDetector.predict`).
        distance_metric: Override the scorer's distance metric.
            Defaults to ``"hamming"`` for BSC encoders, ``"cosine"``
            otherwise.
        k_neighbors: Forwarded to :meth:`HDCAnomalyScorer.create`.

    Returns:
        A fitted :class:`ConformalAnomalyDetector` ready for
        :meth:`~ConformalAnomalyDetector.pvalue` and
        :meth:`~ConformalAnomalyDetector.predict`.

    Raises:
        ValueError: If either ``normal_data`` or ``calibration_data``
            is empty, or if ``alpha`` is not in :math:`(0, 1)`.

    Example:
        >>> import jax, jax.numpy as jnp
        >>> from bayes_hdc.embeddings import ProjectionEncoder
        >>> from bayes_hdc.anomaly import fit_anomaly_pipeline
        >>> key = jax.random.PRNGKey(0)
        >>> enc = ProjectionEncoder.create(input_dim=16, dimensions=512, key=key)
        >>> normal = jax.random.normal(jax.random.PRNGKey(1), (200, 16))
        >>> det = fit_anomaly_pipeline(enc, normal[:100], normal[100:], alpha=0.05)
        >>> bool(det.predict(enc.encode(normal[0]), alpha=0.05))
        False
    """
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    normal_data = jnp.asarray(normal_data)
    calibration_data = jnp.asarray(calibration_data)
    if normal_data.shape[0] == 0:
        raise ValueError("normal_data must be non-empty.")
    if calibration_data.shape[0] == 0:
        raise ValueError("calibration_data must be non-empty.")

    if not hasattr(encoder, "encode_batch"):
        raise TypeError(
            "encoder must expose `encode_batch(features) -> hypervectors`; "
            f"got {type(encoder).__name__}"
        )
    dimensions = int(getattr(encoder, "dimensions"))
    vsa_model_name = getattr(encoder, "vsa_model_name", "map")

    normal_hvs = encoder.encode_batch(normal_data)
    calibration_hvs = encoder.encode_batch(calibration_data)

    scorer = HDCAnomalyScorer.create(
        dimensions=dimensions,
        vsa_model=vsa_model_name,
        distance_metric=distance_metric,
        k_neighbors=k_neighbors,
        n_reference=int(normal_hvs.shape[0]) if k_neighbors > 1 else 0,
    ).fit(normal_hvs)

    detector = ConformalAnomalyDetector.create(
        scorer=scorer,
        n_calibration=int(calibration_hvs.shape[0]),
    ).fit(calibration_hvs)

    return detector.replace(alpha=float(alpha))


__all__ = [
    "HDCAnomalyScorer",
    "ConformalAnomalyDetector",
    "fit_anomaly_pipeline",
]
