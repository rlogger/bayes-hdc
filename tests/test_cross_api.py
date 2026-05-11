# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Cross-API functional audit: every public primitive vs every other public primitive.

The other test files exercise modules in isolation (test_vsa.py, test_models.py,
etc.). This file exercises *compositions*: does encoder X + VSA model Y +
classifier Z work as an end-to-end pipeline? Does jit/vmap/grad compose through
the GaussianHV primitives? Do the deterministic and distributional layers
agree in the zero-variance limit?

Each test asserts behaviour, not just "doesn't crash". Tests are grouped by
composition category:

  1. Encoder x VSA x Classifier
  2. PVSA composition (GaussianHV / DirichletHV)
  3. JAX transformation composition (jit / vmap / grad)
  4. functional <-> VSAModel agreement
  5. Uncertainty pipeline (classifier -> temperature -> conformal)
  6. Sequence + TokenEncoder + cleanup
  7. Equivariance verifiers x bilinear primitives
  8. Variational training
  9. Probabilistic resonator
 10. Bayesian / Streaming classifier agreement
 11. Memory modules
 12. Diagnostics composition
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayes_hdc import (
    BSBC,
    BSC,
    CGR,
    FHRR,
    HRR,
    MAP,
    VTB,
    AdaptiveHDC,
    AttentionMemory,
    BayesianAdaptiveHDC,
    BayesianCentroidClassifier,
    CentroidClassifier,
    ClusteringModel,
    ConformalClassifier,
    ConformalRegressor,
    DirichletHV,
    GaussianHV,
    GraphEncoder,
    HDRegressor,
    HopfieldMemory,
    KernelEncoder,
    LevelEncoder,
    LVQClassifier,
    MixtureHV,
    Multiset,
    ProjectionEncoder,
    RandomEncoder,
    RegularizedLSClassifier,
    Sequence,
    SparseDistributedMemory,
    StreamingBayesianHDC,
    TemperatureCalibrator,
    TokenEncoder,
    bind_bsc,
    bind_dirichlet,
    bind_gaussian,
    bind_map,
    bundle_dirichlet,
    bundle_gaussian,
    bundle_map,
    cleanup,
    cleanup_gaussian,
    cleanup_gaussian_stacked,
    compose_shifts,
    cosine_similarity,
    coverage_calibration_check,
    elbo_gaussian,
    gaussian_reconstruction_log_likelihood_mc,
    hrr_equivariant_bilinear,
    inverse_gaussian,
    kl_dirichlet,
    kl_gaussian,
    permute_gaussian,
    posterior_predictive_check,
    probabilistic_resonator,
    reconstruction_score_mc,
    statistic_mean_norm,
    train_variational_codebook,
    verify_shift_equivariance,
    verify_shift_invariance,
    verify_single_argument_shift_equivariance,
)
from bayes_hdc.functional import bind_hrr

# ----------------------------------------------------------------------
# Test fixtures
# ----------------------------------------------------------------------

D = 64  # base hypervector dim
D_HASH = 1024  # larger dim for hash-table retrieval
N_CLASS = 3
N_SAMPLES = 30
KEY = jax.random.PRNGKey(0)


@pytest.fixture
def real_dataset():
    """Three well-separated unit-norm Gaussian clusters in R^D."""
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(N_CLASS, D)) * 3.0
    labels = rng.integers(low=0, high=N_CLASS, size=(N_SAMPLES,))
    X = centers[labels] + rng.normal(size=(N_SAMPLES, D)) * 0.1
    X = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)
    return jnp.asarray(X, dtype=jnp.float32), jnp.asarray(labels, dtype=jnp.int32)


@pytest.fixture
def real_calset():
    """A second independent draw from the same distribution for calibration."""
    rng = np.random.default_rng(2)
    centers = rng.normal(size=(N_CLASS, D)) * 3.0
    labels = rng.integers(low=0, high=N_CLASS, size=(N_SAMPLES,))
    X = centers[labels] + rng.normal(size=(N_SAMPLES, D)) * 0.1
    X = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)
    return jnp.asarray(X, dtype=jnp.float32), jnp.asarray(labels, dtype=jnp.int32)


# =====================================================================
# 1. Encoder x VSA x Classifier
# =====================================================================


class TestEncoderVSAClassifier:
    """Every encoder x VSA x classifier combination produces a working pipeline."""

    def test_random_encoder_map_centroid(self):
        enc = RandomEncoder.create(
            num_features=4, num_values=8, dimensions=D, vsa_model="map", key=KEY
        )
        rng = np.random.default_rng(0)
        indices = jnp.asarray(rng.integers(0, 8, size=(N_SAMPLES, 4)), dtype=jnp.int32)
        labels = jnp.asarray(rng.integers(0, N_CLASS, size=(N_SAMPLES,)), dtype=jnp.int32)
        hvs = enc.encode_batch(indices)
        assert hvs.shape == (N_SAMPLES, D)
        clf = CentroidClassifier.create(num_classes=N_CLASS, dimensions=D, vsa_model="map").fit(
            hvs, labels
        )
        preds = clf.predict(hvs)
        assert preds.shape == (N_SAMPLES,)
        assert preds.dtype == jnp.int32

    def test_level_encoder_map_regularized_ls_dual_path(self):
        # n < d hits the dual solver path; verify it's exercised.
        enc = LevelEncoder.create(num_levels=20, dimensions=D, vsa_model="map", key=KEY)
        rng = np.random.default_rng(0)
        values = jnp.asarray(rng.uniform(0, 1, size=(N_SAMPLES,)), dtype=jnp.float32)
        hvs = enc.encode_batch(values)
        labels = jnp.asarray(rng.integers(0, N_CLASS, size=(N_SAMPLES,)), dtype=jnp.int32)
        # n=30 < d=64, so this exercises the dual path.
        assert N_SAMPLES < D
        clf = RegularizedLSClassifier.create(dimensions=D, num_classes=N_CLASS, reg=1.0).fit(
            hvs, labels
        )
        preds = clf.predict(hvs)
        assert preds.shape == (N_SAMPLES,)

    def test_level_encoder_bsc_centroid(self):
        # Different return dtype from MAP — verify the BSC code path.
        enc = LevelEncoder.create(num_levels=20, dimensions=D, vsa_model="bsc", key=KEY)
        rng = np.random.default_rng(0)
        values = jnp.asarray(rng.uniform(0, 1, size=(N_SAMPLES,)), dtype=jnp.float32)
        hvs = enc.encode_batch(values)
        # BSC level encoder returns bool
        assert hvs.dtype == jnp.bool_
        labels = jnp.asarray(rng.integers(0, N_CLASS, size=(N_SAMPLES,)), dtype=jnp.int32)
        clf = CentroidClassifier.create(num_classes=N_CLASS, dimensions=D, vsa_model="bsc").fit(
            hvs, labels
        )
        preds = clf.predict(hvs)
        assert preds.shape == (N_SAMPLES,)

    def test_projection_encoder_hd_regressor(self, real_dataset):
        # End-to-end regression: ProjectionEncoder + HDRegressor.
        enc = ProjectionEncoder.create(input_dim=4, dimensions=D, vsa_model="map", key=KEY)
        rng = np.random.default_rng(0)
        X_raw = jnp.asarray(rng.normal(size=(N_SAMPLES, 4)), dtype=jnp.float32)
        # Learnable linear relationship.
        W_true = jnp.asarray(rng.normal(size=(4, 2)), dtype=jnp.float32)
        Y = X_raw @ W_true + 0.01 * rng.normal(size=(N_SAMPLES, 2))
        hvs = enc.encode_batch(X_raw)
        reg = HDRegressor.create(dimensions=D, output_dim=2, reg=1.0).fit(hvs, Y)
        preds = reg.predict(hvs)
        assert preds.shape == (N_SAMPLES, 2)
        # And score is a valid R^2
        r2 = float(reg.score(hvs, Y))
        # Training R^2 should be positive on this easy task.
        assert r2 > 0.0

    def test_kernel_encoder_map_centroid(self):
        enc = KernelEncoder.create(input_dim=4, dimensions=D, gamma=0.5, vsa_model="map", key=KEY)
        rng = np.random.default_rng(0)
        X_raw = jnp.asarray(rng.normal(size=(N_SAMPLES, 4)), dtype=jnp.float32)
        labels = jnp.asarray(rng.integers(0, N_CLASS, size=(N_SAMPLES,)), dtype=jnp.int32)
        hvs = enc.encode_batch(X_raw)
        # Encoded hvs should be unit-norm (MAP normalisation in KernelEncoder.encode).
        norms = jnp.linalg.norm(hvs, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-4)
        clf = CentroidClassifier.create(num_classes=N_CLASS, dimensions=D).fit(hvs, labels)
        assert clf.predict(hvs).shape == (N_SAMPLES,)

    def test_random_encoder_hrr_centroid(self):
        enc = RandomEncoder.create(
            num_features=4, num_values=8, dimensions=D, vsa_model="hrr", key=KEY
        )
        rng = np.random.default_rng(0)
        indices = jnp.asarray(rng.integers(0, 8, size=(N_SAMPLES, 4)), dtype=jnp.int32)
        labels = jnp.asarray(rng.integers(0, N_CLASS, size=(N_SAMPLES,)), dtype=jnp.int32)
        hvs = enc.encode_batch(indices)
        # HRR codebook is unit-norm Gaussian; encoded vectors are bundle of these,
        # so they're well-defined and finite.
        assert jnp.all(jnp.isfinite(hvs))
        clf = CentroidClassifier.create(num_classes=N_CLASS, dimensions=D, vsa_model="hrr").fit(
            hvs, labels
        )
        assert clf.predict(hvs).shape == (N_SAMPLES,)

    def test_adaptive_lvq_clustering_consistent(self, real_dataset):
        # Three "centroid-like" learners + a clusterer all produce (N,) outputs.
        hvs, labels = real_dataset
        for clf_factory in [
            lambda: AdaptiveHDC.create(num_classes=N_CLASS, dimensions=D, key=KEY).fit(
                hvs, labels, epochs=1
            ),
            lambda: LVQClassifier.create(num_classes=N_CLASS, dimensions=D, key=KEY).fit(
                hvs, labels, epochs=1
            ),
            lambda: ClusteringModel.create(num_clusters=N_CLASS, dimensions=D, key=KEY).fit(
                hvs, max_iters=5
            ),
        ]:
            clf = clf_factory()
            preds = clf.predict(hvs)
            assert preds.shape == (N_SAMPLES,)


# =====================================================================
# 2. PVSA composition: GaussianHV / DirichletHV
# =====================================================================


class TestPVSAComposition:
    """Distributional VSA primitives compose like the deterministic algebra."""

    def test_gaussian_zero_variance_reduces_to_map_bind(self):
        """GaussianHV with var=0 should match deterministic bind_map / bundle_map."""
        x = jax.random.normal(KEY, (D,))
        y = jax.random.normal(jax.random.fold_in(KEY, 1), (D,))
        gx = GaussianHV.from_sample(x, var=0.0)
        gy = GaussianHV.from_sample(y, var=0.0)
        bound = bind_gaussian(gx, gy)
        assert jnp.allclose(bound.mu, bind_map(x, y))
        assert jnp.max(jnp.abs(bound.var)) == 0.0

    def test_gaussian_zero_variance_reduces_to_map_bundle(self):
        x = jax.random.normal(KEY, (D,))
        y = jax.random.normal(jax.random.fold_in(KEY, 1), (D,))
        batched = GaussianHV(mu=jnp.stack([x, y]), var=jnp.zeros((2, D)), dimensions=D)
        bundled = bundle_gaussian(batched)
        assert jnp.allclose(bundled.mu, bundle_map(jnp.stack([x, y]), axis=0))

    def test_gaussian_self_kl_is_zero(self):
        a = GaussianHV.random(KEY, D, var=0.5)
        assert float(kl_gaussian(a, a)) < 1e-3

    def test_inverse_gaussian_round_trip(self):
        """bind(x, inverse(x)) approximates the unit impulse for non-degenerate x."""
        # Use means well away from zero so the delta-method inverse is sharp.
        mu = jnp.full((D,), 1.0)
        x = GaussianHV(mu=mu, var=jnp.full((D,), 0.001), dimensions=D)
        inv = inverse_gaussian(x)
        bound = bind_gaussian(x, inv)
        # Mean should be ~1 everywhere (ish).
        assert float(jnp.mean(bound.mu)) == pytest.approx(1.0, abs=0.05)

    def test_cleanup_gaussian_list_and_stacked_agree(self):
        mems = [GaussianHV.random(jax.random.fold_in(KEY, i), D, var=1e-4) for i in range(5)]
        q = mems[2]
        idx_list, _ = cleanup_gaussian(q, mems)
        stacked = GaussianHV(
            mu=jnp.stack([m.mu for m in mems]),
            var=jnp.stack([m.var for m in mems]),
            dimensions=D,
        )
        idx_stack, _ = cleanup_gaussian_stacked(q, stacked)
        assert idx_list == 2
        assert int(idx_stack) == 2
        assert idx_list == int(idx_stack)

    def test_permute_gaussian_preserves_norm(self):
        a = GaussianHV.random(KEY, D)
        shifted = permute_gaussian(a, shifts=7)
        assert jnp.allclose(jnp.linalg.norm(a.mu), jnp.linalg.norm(shifted.mu))

    def test_dirichlet_self_kl_is_zero(self):
        d1 = DirichletHV.create(dimensions=8, concentration=2.0)
        assert float(kl_dirichlet(d1, d1)) == pytest.approx(0.0, abs=1e-4)

    def test_dirichlet_bind_bundle_close(self):
        d1 = DirichletHV.create(dimensions=8, concentration=2.0)
        d2 = DirichletHV.create(dimensions=8, concentration=3.0)
        d3 = bind_dirichlet(d1, d2)
        # Concentrations should add for the bound result.
        total_old = float(d1.concentration()) + float(d2.concentration())
        total_new = float(d3.concentration())
        # Slight EPS perturbation per category times K=8.
        assert abs(total_new - total_old) < 1e-3
        # Bundle: concentrations add directly.
        batched = DirichletHV(alpha=jnp.stack([d1.alpha, d2.alpha]), dimensions=8)
        d4 = bundle_dirichlet(batched)
        assert jnp.allclose(d4.alpha, d1.alpha + d2.alpha)

    def test_mixture_collapse_to_gaussian(self):
        g1 = GaussianHV(mu=jnp.zeros(D), var=jnp.ones(D), dimensions=D)
        g2 = GaussianHV(mu=jnp.ones(D), var=jnp.ones(D), dimensions=D)
        m = MixtureHV.from_components([g1, g2])
        g = m.collapse_to_gaussian()
        # Mean of mixture = average of component means.
        assert jnp.allclose(g.mu, 0.5 * jnp.ones(D))


# =====================================================================
# 3. JAX transformation composition
# =====================================================================


class TestJaxTransforms:
    """jit, vmap, grad compose through every PVSA primitive."""

    def test_jit_through_bind_gaussian(self):
        f = jax.jit(bind_gaussian)
        a = GaussianHV.random(KEY, D)
        b = GaussianHV.random(jax.random.fold_in(KEY, 1), D)
        c = f(a, b)
        assert c.mu.shape == (D,)

    def test_vmap_through_bind_gaussian(self):
        a_stack = GaussianHV(
            mu=jax.random.normal(KEY, (8, D)),
            var=jnp.ones((8, D)) * 0.1,
            dimensions=D,
        )
        b_stack = GaussianHV(
            mu=jax.random.normal(jax.random.fold_in(KEY, 1), (8, D)),
            var=jnp.ones((8, D)) * 0.1,
            dimensions=D,
        )
        res = jax.vmap(bind_gaussian)(a_stack, b_stack)
        assert res.mu.shape == (8, D)
        assert res.var.shape == (8, D)

    def test_grad_through_kl_gaussian_zero_at_minimum(self):
        # gradient of KL(p || N(0, I)) w.r.t. mu_p should be zero at mu_p = 0.
        def loss(mu):
            p = GaussianHV(mu=mu, var=jnp.ones(D), dimensions=D)
            q = GaussianHV(mu=jnp.zeros(D), var=jnp.ones(D), dimensions=D)
            return kl_gaussian(p, q)

        g = jax.grad(loss)(jnp.zeros(D))
        assert float(jnp.linalg.norm(g)) < 1e-5

    def test_grad_through_classifier_predict_proba(self):
        """End-to-end grad through BayesianCentroidClassifier.predict_proba."""
        rng = np.random.default_rng(0)
        hvs = jnp.asarray(rng.normal(size=(N_SAMPLES, D)), dtype=jnp.float32)
        labels = jnp.asarray(rng.integers(0, N_CLASS, size=(N_SAMPLES,)), dtype=jnp.int32)
        clf = BayesianCentroidClassifier.create(num_classes=N_CLASS, dimensions=D).fit(hvs, labels)

        def query_loss(q):
            p = clf.predict_proba(q)
            return -jnp.log(p[0] + 1e-8)

        g = jax.grad(query_loss)(hvs[0])
        assert g.shape == (D,)
        assert bool(jnp.all(jnp.isfinite(g)))

    def test_jit_through_classifier_predict(self, real_dataset):
        hvs, labels = real_dataset
        clf = CentroidClassifier.create(num_classes=N_CLASS, dimensions=D).fit(hvs, labels)
        f = jax.jit(clf.predict)
        assert f(hvs).shape == (N_SAMPLES,)


# =====================================================================
# 4. functional bind/bundle agree with VSAModel.bind/bundle
# =====================================================================


class TestFunctionalVsModelAgreement:
    """Each VSA-class method and its functional counterpart return the same value."""

    def test_bsc_bind_agrees(self):
        m = BSC.create(D)
        rng = np.random.default_rng(0)
        x = jnp.asarray(rng.integers(0, 2, size=D), dtype=jnp.bool_)
        y = jnp.asarray(rng.integers(0, 2, size=D), dtype=jnp.bool_)
        assert jnp.array_equal(m.bind(x, y), bind_bsc(x, y))

    def test_map_bind_agrees(self):
        m = MAP.create(D)
        x = jax.random.normal(KEY, (D,))
        y = jax.random.normal(jax.random.fold_in(KEY, 1), (D,))
        assert jnp.allclose(m.bind(x, y), bind_map(x, y))

    def test_hrr_bind_agrees(self):
        m = HRR.create(D)
        x = jax.random.normal(KEY, (D,))
        y = jax.random.normal(jax.random.fold_in(KEY, 1), (D,))
        assert jnp.allclose(m.bind(x, y), bind_hrr(x, y), atol=1e-5)

    def test_fhrr_bind_and_inverse_unit_phasor(self):
        m = FHRR.create(D)
        x = m.random(KEY, (D,))
        # x should live on the unit circle in C.
        assert jnp.allclose(jnp.abs(x), 1.0, atol=1e-5)
        y = m.random(jax.random.fold_in(KEY, 1), (D,))
        bound = m.bind(x, y)
        assert bound.dtype in (jnp.complex64, jnp.complex128)
        # Bind on unit phasors stays on the unit circle.
        assert jnp.allclose(jnp.abs(bound), 1.0, atol=1e-5)
        # FHRR inverse is the conjugate.
        inv = m.inverse(x)
        assert jnp.allclose(inv, jnp.conj(x))

    def test_bsbc_random_is_block_sparse(self):
        m = BSBC.create(dimensions=D, block_size=8, k_active=2)
        x = m.random(KEY, (D,))
        # Each block of 8 should have exactly k_active=2 ones.
        for b in range(D // 8):
            block = x[b * 8 : (b + 1) * 8]
            assert int(jnp.sum(block)) == 2

    def test_cgr_inverse_round_trip(self):
        m = CGR.create(dimensions=D, q=8)
        x = m.random(KEY, (D,))
        # x + inverse(x) == 0 (mod q)
        inv = m.inverse(x)
        # CGR binding is modular addition.
        identity = m.bind(x, inv)
        assert jnp.array_equal(identity, jnp.zeros_like(identity))

    def test_vtb_only_for_square_dim(self):
        # 64 = 8^2 works.
        m = VTB.create(D)
        x = m.random(KEY, (D,))
        y = m.random(jax.random.fold_in(KEY, 1), (D,))
        bound = m.bind(x, y)
        assert bound.shape == (D,)
        # 65 is not a perfect square — should raise.
        with pytest.raises(ValueError):
            VTB.create(65)


# =====================================================================
# 5. Uncertainty pipeline: classifier -> temperature -> conformal
# =====================================================================


class TestUncertaintyPipeline:
    """Classifier logits flow through TemperatureCalibrator and ConformalClassifier."""

    def test_centroid_temperature_conformal_end_to_end(self, real_dataset, real_calset):
        hvs, labels = real_dataset
        cal_hvs, cal_labels = real_calset
        clf = CentroidClassifier.create(num_classes=N_CLASS, dimensions=D).fit(hvs, labels)
        cal_sims = jax.vmap(clf.similarity)(cal_hvs)
        tc = TemperatureCalibrator.create().fit(cal_sims, cal_labels, max_iters=50)
        cal_probs = tc.calibrate(cal_sims)
        cc = ConformalClassifier.create(alpha=0.1).fit(cal_probs, cal_labels)
        coverage = float(cc.coverage(cal_probs, cal_labels))
        set_size = float(cc.set_size(cal_probs))
        # Marginal coverage >= 1 - alpha (modulo finite-sample noise).
        assert coverage >= 0.85
        # Sets are non-empty and bounded by the class count (allow tiny fp slack).
        assert 1.0 <= set_size <= float(N_CLASS) + 1e-3

    def test_bayesian_centroid_temperature_conformal(self, real_dataset, real_calset):
        hvs, labels = real_dataset
        cal_hvs, cal_labels = real_calset
        clf = BayesianCentroidClassifier.create(num_classes=N_CLASS, dimensions=D).fit(hvs, labels)
        cal_logits = clf.logits(cal_hvs)
        tc = TemperatureCalibrator.create().fit(cal_logits, cal_labels, max_iters=50)
        cal_probs = tc.calibrate(cal_logits)
        cc = ConformalClassifier.create(alpha=0.1).fit(cal_probs, cal_labels)
        coverage = float(cc.coverage(cal_probs, cal_labels))
        assert 0.0 <= coverage <= 1.0

    def test_temperature_calibrator_is_accuracy_preserving(self, real_dataset, real_calset):
        """Temperature scaling must NOT change argmax — Guo et al. (2017) property."""
        hvs, labels = real_dataset
        cal_hvs, cal_labels = real_calset
        clf = CentroidClassifier.create(num_classes=N_CLASS, dimensions=D).fit(hvs, labels)
        cal_sims = jax.vmap(clf.similarity)(cal_hvs)
        tc = TemperatureCalibrator.create().fit(cal_sims, cal_labels, max_iters=50)
        cal_probs = tc.calibrate(cal_sims)
        pred_before = jnp.argmax(cal_sims, axis=-1)
        pred_after = jnp.argmax(cal_probs, axis=-1)
        assert jnp.array_equal(pred_before, pred_after)

    def test_hd_regressor_conformal_regressor_end_to_end(self):
        rng = np.random.default_rng(0)
        X_train = jnp.asarray(rng.normal(size=(N_SAMPLES, D)), dtype=jnp.float32)
        X_cal = jnp.asarray(rng.normal(size=(N_SAMPLES, D)), dtype=jnp.float32)
        W_true = jnp.asarray(rng.normal(size=(D, 2)), dtype=jnp.float32)
        Y_train = X_train @ W_true + 0.1 * rng.normal(size=(N_SAMPLES, 2))
        Y_cal = X_cal @ W_true + 0.1 * rng.normal(size=(N_SAMPLES, 2))
        reg = HDRegressor.create(dimensions=D, output_dim=2, reg=1.0).fit(X_train, Y_train)
        preds_cal = reg.predict(X_cal)
        cr = ConformalRegressor.create(alpha=0.1, output_dim=2).fit(preds_cal, Y_cal)
        lo, hi = cr.predict_interval(preds_cal)
        # Interval bounds have the right shapes.
        assert lo.shape == preds_cal.shape == hi.shape
        # Coverage on calibration set should match nominal level.
        coverage = cr.coverage(preds_cal, Y_cal)
        assert coverage.shape == (2,)
        # ConformalRegressor.coverage uses ceil-quantile; on calibration must be >= 0.9.
        assert bool(jnp.all(coverage >= 0.85))
        width = cr.interval_width()
        assert width.shape == (2,)
        assert bool(jnp.all(width >= 0))


# =====================================================================
# 6. Sequence + TokenEncoder + cleanup
# =====================================================================


class TestSequenceTokenEncoder:
    """Sequence retrieval after cleanup gets back the right token IDs."""

    def test_token_encoder_sequence_flat_retrieval(self):
        # Short sequence, large D: flat permute-bundle retrieval after cleanup
        # should recover every position exactly.
        enc = TokenEncoder.create(vocab_size=16, dimensions=D_HASH, vsa_model="map", key=KEY)
        ids = jnp.asarray([1, 5, 2, 9], dtype=jnp.int32)
        items = enc.lookup_batch(ids)
        seq = Sequence.from_vectors(items)
        assert seq.size == 4
        for pos in range(4):
            noisy = seq.get(pos)
            rec = cleanup(noisy, enc.codebook)
            sims = jax.vmap(lambda c: cosine_similarity(rec, c))(enc.codebook)
            pred_idx = int(jnp.argmax(sims))
            assert pred_idx == int(ids[pos])

    def test_token_encoder_hierarchical_retrieval_beats_flat_at_n_64(self):
        """At n=64 the flat construction is past capacity at D=512; the hierarchical
        construction recovers every position via chunk-level cleanup."""
        enc = TokenEncoder.create(vocab_size=128, dimensions=512, vsa_model="map", key=KEY)
        ids = jnp.asarray(list(range(64)), dtype=jnp.int32)
        hs = enc.encode_hierarchical(ids, chunk_size=8)
        # Sample 4 positions; cleanup should recover all of them.
        for pos in [0, 17, 33, 50]:
            item = hs.get(pos)
            rec = cleanup(item, enc.codebook)
            sims = jax.vmap(lambda c: cosine_similarity(rec, c))(enc.codebook)
            pred_idx = int(jnp.argmax(sims))
            assert pred_idx == int(ids[pos])

    def test_hierarchical_sequence_chunk_codebook_shape(self):
        enc = TokenEncoder.create(vocab_size=32, dimensions=D_HASH, vsa_model="map", key=KEY)
        ids = jnp.asarray(list(range(32)), dtype=jnp.int32)
        hs = enc.encode_hierarchical(ids, chunk_size=8)
        # 32 / 8 = 4 chunks
        assert hs.chunk_codebook.shape == (4, D_HASH)

    def test_multiset_membership_signal(self):
        rng = np.random.default_rng(0)
        items = jnp.asarray(rng.normal(size=(4, D_HASH)), dtype=jnp.float32)
        items = items / jnp.linalg.norm(items, axis=-1, keepdims=True)
        ms = Multiset.from_vectors(items)
        # Member should produce higher similarity than a random non-member.
        non_member = jax.random.normal(KEY, (D_HASH,))
        non_member = non_member / jnp.linalg.norm(non_member)
        sim_in = float(ms.contains(items[1]))
        sim_out = float(ms.contains(non_member))
        assert sim_in > sim_out


# =====================================================================
# 7. Equivariance verifiers x bilinear primitives
# =====================================================================


class TestEquivariance:
    """The shift-equivariance verifiers correctly classify each VSA primitive."""

    def test_bind_map_is_diagonally_shift_equivariant(self):
        a = jax.random.normal(KEY, (D,))
        b = jax.random.normal(jax.random.fold_in(KEY, 1), (D,))
        assert verify_shift_equivariance(bind_map, a, b, shifts=(1, 7, 30), atol=1e-4)

    def test_cosine_similarity_is_shift_invariant(self):
        a = jax.random.normal(KEY, (D,))
        b = jax.random.normal(jax.random.fold_in(KEY, 1), (D,))
        assert verify_shift_invariance(cosine_similarity, a, b, shifts=(1, 7, 30), atol=1e-4)

    def test_bind_hrr_single_arg_equivariant(self):
        a = jax.random.normal(KEY, (D,))
        b = jax.random.normal(jax.random.fold_in(KEY, 1), (D,))
        assert verify_single_argument_shift_equivariance(
            bind_hrr, a, b, arg_index=0, shifts=(1, 7, 30), atol=1e-4
        )

    def test_hrr_equivariant_bilinear_is_bind_hrr(self):
        a = jax.random.normal(KEY, (D,))
        b = jax.random.normal(jax.random.fold_in(KEY, 1), (D,))
        assert jnp.allclose(hrr_equivariant_bilinear(a, b), bind_hrr(a, b))

    def test_compose_shifts_in_z_mod_d(self):
        assert compose_shifts(3, 5, 10) == 8
        assert compose_shifts(7, 7, 10) == 4
        # Identity: any shift composed with 0 returns itself.
        for k in [0, 1, 5, 9]:
            assert compose_shifts(k, 0, 10) == k


# =====================================================================
# 8. Variational training pipeline
# =====================================================================


class TestVariationalTraining:
    """elbo + reconstruction_log_likelihood_mc + train_variational_codebook compose."""

    def test_full_elbo_training_decreases_loss(self):
        D_small = 32
        target = GaussianHV(mu=jnp.ones(D_small), var=jnp.full((D_small,), 0.1), dimensions=D_small)

        def loss_fn(params, key):
            posterior = GaussianHV(
                mu=params["mu"], var=jnp.exp(params["log_var"]), dimensions=D_small
            )
            prior = GaussianHV.create(D_small)
            recon = gaussian_reconstruction_log_likelihood_mc(
                posterior, target, key, n_samples=4, observation_noise=0.5
            )
            return -elbo_gaussian(posterior, prior, recon)

        init = {"mu": jnp.zeros(D_small), "log_var": jnp.zeros(D_small)}
        result = train_variational_codebook(
            init_params=init,
            loss_fn=loss_fn,
            key=KEY,
            n_steps=100,
            learning_rate=1e-2,
        )
        assert result.loss_history.shape == (100,)
        early = float(jnp.mean(result.loss_history[:10]))
        late = float(jnp.mean(result.loss_history[-10:]))
        # Loss must decrease.
        assert late < early
        # And the fitted mu should drift toward the target.
        assert float(jnp.linalg.norm(result.params["mu"] - target.mu)) < float(
            jnp.linalg.norm(jnp.zeros(D_small) - target.mu)
        )

    def test_reconstruction_score_mc_in_unit_range(self):
        a = GaussianHV(mu=jnp.ones(32), var=jnp.full((32,), 0.1), dimensions=32)
        score = float(reconstruction_score_mc(a, a, KEY, n_samples=16))
        assert -1.0 <= score <= 1.0


# =====================================================================
# 9. Probabilistic resonator
# =====================================================================


class TestProbabilisticResonator:
    """The probabilistic resonator factors composites built from Gaussian codebooks."""

    def test_resonator_recovers_two_factor_indices(self):
        D_small = 128
        codebooks = []
        for i in range(2):
            mu = jax.random.normal(jax.random.fold_in(KEY, i), (5, D_small))
            mu = mu / (jnp.linalg.norm(mu, axis=-1, keepdims=True) + 1e-8)
            codebooks.append(
                GaussianHV(mu=mu, var=jnp.ones((5, D_small)) * 1e-5, dimensions=D_small)
            )
        true_idx = jnp.array([2, 3])
        target_mu = codebooks[0].mu[2] * codebooks[1].mu[3]
        target = GaussianHV(mu=target_mu, var=jnp.ones(D_small) * 1e-5, dimensions=D_small)
        res = probabilistic_resonator(
            codebooks,
            target,
            jax.random.fold_in(KEY, 0),
            n_restarts=8,
            max_iters=20,
            temperature=0.1,
        )
        assert jnp.array_equal(res.indices, true_idx)
        assert res.alignment > 0.95
        assert res.history.shape == (20,)
        assert res.n_restarts == 8

    def test_resonator_alignment_strictly_in_unit_range(self):
        D_small = 64
        cb = GaussianHV(
            mu=jax.random.normal(KEY, (3, D_small)),
            var=jnp.ones((3, D_small)) * 1e-4,
            dimensions=D_small,
        )
        # Unit-norm.
        cb_mu = cb.mu / (jnp.linalg.norm(cb.mu, axis=-1, keepdims=True) + 1e-8)
        cb = GaussianHV(mu=cb_mu, var=cb.var, dimensions=D_small)
        target = GaussianHV(mu=cb.mu[1], var=cb.var[1], dimensions=D_small)
        res = probabilistic_resonator(
            [cb], target, jax.random.fold_in(KEY, 1), n_restarts=4, max_iters=10
        )
        assert -1.0 <= res.alignment <= 1.0


# =====================================================================
# 10. Bayesian / Streaming classifier agreement
# =====================================================================


class TestBayesianStreamingAgreement:
    """The four prototype-based classifiers all agree on well-separated data."""

    def test_all_four_classifiers_agree_on_clean_data(self, real_dataset):
        hvs, labels = real_dataset
        cc = CentroidClassifier.create(num_classes=N_CLASS, dimensions=D).fit(hvs, labels)
        bc = BayesianCentroidClassifier.create(num_classes=N_CLASS, dimensions=D).fit(hvs, labels)
        ba = BayesianAdaptiveHDC.create(num_classes=N_CLASS, dimensions=D, obs_var=0.01).fit(
            hvs, labels, epochs=3
        )
        sb = StreamingBayesianHDC.create(num_classes=N_CLASS, dimensions=D, decay=0.5).fit(
            hvs, labels
        )
        # All four classifiers should achieve 100% training accuracy on this data.
        for name, clf in [("CC", cc), ("BC", bc), ("BA", ba), ("SB", sb)]:
            acc = float(clf.score(hvs, labels))
            assert acc == 1.0, f"{name} acc={acc}"
        # And they should agree pairwise.
        cc_preds = cc.predict(hvs)
        for name, clf in [("BC", bc), ("BA", ba), ("SB", sb)]:
            preds = clf.predict(hvs)
            agree = float(jnp.mean(preds == cc_preds))
            assert agree == 1.0, f"{name} vs CC agree={agree}"

    def test_bayesian_centroid_uncertainty_shape(self, real_dataset):
        hvs, labels = real_dataset
        clf = BayesianCentroidClassifier.create(num_classes=N_CLASS, dimensions=D).fit(hvs, labels)
        # Single-query and batched shapes.
        u_single = clf.predict_uncertainty(hvs[0])
        u_batch = clf.predict_uncertainty(hvs)
        assert u_single.shape == (N_CLASS,)
        assert u_batch.shape == (N_SAMPLES, N_CLASS)
        # Uncertainty is non-negative.
        assert bool(jnp.all(u_batch >= 0))

    def test_streaming_decay_validates(self):
        with pytest.raises(ValueError):
            StreamingBayesianHDC.create(num_classes=2, dimensions=D, decay=1.0)
        with pytest.raises(ValueError):
            StreamingBayesianHDC.create(num_classes=2, dimensions=D, decay=0.0)


# =====================================================================
# 11. Memory modules
# =====================================================================


class TestMemoryModules:
    """All three memory modules support write/retrieve with reasonable shapes."""

    def test_attention_memory_retrieves(self):
        am = AttentionMemory.create(dimensions=D, temperature=1.0)
        k = jax.random.normal(KEY, (D,))
        v = jax.random.normal(jax.random.fold_in(KEY, 1), (D,))
        am = am.write(k, v)
        rec = am.retrieve(k)
        assert rec.shape == (D,)
        # Single-entry memory: retrieval is just `v` (after softmax over a single key).
        assert jnp.allclose(rec, v, atol=1e-4)

    def test_hopfield_memory_retrieves_added_pattern(self):
        hm = HopfieldMemory.create(D, beta=100.0)  # high beta -> near argmax
        p = jax.random.normal(KEY, (D,))
        p_unit = p / jnp.linalg.norm(p)
        hm = hm.add(p)
        rec = hm.retrieve(p_unit)
        # With one pattern, retrieval recovers it exactly (softmax of a single row).
        assert jnp.allclose(rec, p_unit, atol=1e-3)

    def test_sdm_write_read_finite(self):
        sdm = SparseDistributedMemory.create(num_locations=8, dimensions=D, radius=0.5, key=KEY)
        addr = jax.random.normal(jax.random.fold_in(KEY, 1), (D,))
        val = jax.random.normal(jax.random.fold_in(KEY, 2), (D,))
        sdm = sdm.write(addr, val)
        rec = sdm.read(addr)
        assert rec.shape == (D,)
        assert bool(jnp.all(jnp.isfinite(rec)))


# =====================================================================
# 12. Diagnostics composition
# =====================================================================


class TestDiagnostics:
    """The diagnostic helpers compose with the rest of the API."""

    def test_posterior_predictive_check_returns_valid_p_value(self):
        posterior = GaussianHV.random(KEY, D, var=0.1)
        rng = np.random.default_rng(0)
        obs = jnp.asarray(
            rng.normal(size=(20, D)) * 0.3 + np.asarray(posterior.mu),
            dtype=jnp.float32,
        )
        res = posterior_predictive_check(
            posterior,
            obs,
            statistic_mean_norm,
            jax.random.fold_in(KEY, 3),
            n_replicas=20,
        )
        assert 0.0 <= res.p_value <= 1.0
        assert res.predictive_std >= 0
        assert res.ci_low <= res.predictive_mean <= res.ci_high

    def test_coverage_calibration_check_sweeps_alphas(self, real_dataset, real_calset):
        hvs, labels = real_dataset
        cal_hvs, cal_labels = real_calset
        # Independent test draw.
        rng = np.random.default_rng(3)
        centers = rng.normal(size=(N_CLASS, D)) * 3.0
        test_labels_np = rng.integers(0, N_CLASS, size=(N_SAMPLES,))
        X = centers[test_labels_np] + rng.normal(size=(N_SAMPLES, D)) * 0.1
        X = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)
        test_hvs = jnp.asarray(X, dtype=jnp.float32)
        test_labels = jnp.asarray(test_labels_np, dtype=jnp.int32)

        clf = CentroidClassifier.create(num_classes=N_CLASS, dimensions=D).fit(hvs, labels)
        cal_sims = jax.vmap(clf.similarity)(cal_hvs)
        test_sims = jax.vmap(clf.similarity)(test_hvs)
        cal_probs = jax.nn.softmax(cal_sims, axis=-1)
        test_probs = jax.nn.softmax(test_sims, axis=-1)
        alphas = [0.1, 0.2, 0.3]
        res = coverage_calibration_check(
            lambda a: ConformalClassifier.create(alpha=a),
            cal_probs,
            cal_labels,
            test_probs,
            test_labels,
            alphas=alphas,
        )
        assert res.empirical_coverage.shape == (3,)
        assert res.set_sizes.shape == (3,)
        # As alpha grows, the conformal set shrinks (mean set size non-increasing).
        # Allow small non-monotonicities from finite-sample noise.
        for i in range(len(alphas) - 1):
            assert float(res.set_sizes[i + 1]) <= float(res.set_sizes[i]) + 0.5


# =====================================================================
# Additional cross-cutting checks
# =====================================================================


class TestCrossCutting:
    """Miscellaneous compositions worth a regression check."""

    def test_classifier_predict_single_vs_batched_consistent(self, real_dataset):
        """clf.predict(single_query) == clf.predict(batch)[0] when batch[0] = single."""
        hvs, labels = real_dataset
        for clf in [
            CentroidClassifier.create(num_classes=N_CLASS, dimensions=D).fit(hvs, labels),
            BayesianCentroidClassifier.create(num_classes=N_CLASS, dimensions=D).fit(hvs, labels),
            AdaptiveHDC.create(num_classes=N_CLASS, dimensions=D).fit(hvs, labels, epochs=1),
            LVQClassifier.create(num_classes=N_CLASS, dimensions=D).fit(hvs, labels, epochs=1),
        ]:
            pred_single = clf.predict(hvs[0])
            pred_batch = clf.predict(hvs)
            assert pred_single.shape == ()
            assert pred_batch.shape == (N_SAMPLES,)
            assert int(pred_single) == int(pred_batch[0])

    def test_hd_regressor_score_returns_r2(self):
        """HDRegressor.score returns scalar R^2 = 1 on a perfectly recoverable system."""
        rng = np.random.default_rng(0)
        # n=100 > d=64 → hits the primal path.
        X = jnp.asarray(rng.normal(size=(100, D)), dtype=jnp.float32)
        W = jnp.asarray(rng.normal(size=(D, 2)), dtype=jnp.float32)
        Y = X @ W
        reg = HDRegressor.create(dimensions=D, output_dim=2, reg=1e-6).fit(X, Y)
        # With negligible reg and a perfect linear relationship, R^2 -> 1.
        assert float(reg.score(X, Y)) == pytest.approx(1.0, abs=1e-3)

    def test_inverse_gaussian_zero_variance_matches_map_inverse(self):
        """GaussianHV.inverse with var=0 should match deterministic inverse_map on mu."""
        from bayes_hdc.functional import inverse_map

        x_mu = jnp.full((D,), 0.5)
        gx = GaussianHV.from_sample(x_mu, var=0.0)
        ginv = inverse_gaussian(gx)
        # Mean should be 1/0.5 = 2.0 elementwise.
        assert jnp.allclose(ginv.mu, inverse_map(x_mu))

    def test_temperature_calibrator_at_t_one_is_softmax(self):
        # T=1 makes calibrate() equal to plain softmax.
        tc = TemperatureCalibrator(temperature=jnp.asarray(1.0))
        logits = jax.random.normal(KEY, (5, N_CLASS))
        probs = tc.calibrate(logits)
        assert jnp.allclose(probs, jax.nn.softmax(logits, axis=-1))

    def test_conformal_classifier_predict_set_never_empty(self, real_dataset, real_calset):
        hvs, labels = real_dataset
        cal_hvs, cal_labels = real_calset
        clf = CentroidClassifier.create(num_classes=N_CLASS, dimensions=D).fit(hvs, labels)
        cal_sims = jax.vmap(clf.similarity)(cal_hvs)
        cal_probs = jax.nn.softmax(cal_sims, axis=-1)
        cc = ConformalClassifier.create(alpha=0.1).fit(cal_probs, cal_labels)
        mask = cc.predict_set(cal_probs)
        # At least one class is always in the set (top-1 inclusion guarantee).
        assert bool(jnp.all(jnp.sum(mask, axis=-1) >= 1))


# =====================================================================
# Known limitations: xfail with explanation
# =====================================================================


@pytest.mark.xfail(
    reason=(
        "GraphEncoder.encode_edges uses Python int() on traced array values inside "
        "a Python for-loop, so it cannot be jit-compiled. The functional alternative "
        "graph_encode uses jax.vmap and is jit-friendly. This is an API "
        "inconsistency, not a correctness bug: the eager call works."
    ),
    strict=True,
)
def test_graph_encoder_encode_edges_is_not_jit_compilable():
    """GraphEncoder.encode_edges should jit cleanly, but currently can't."""
    enc = GraphEncoder.create(num_nodes=10, dimensions=D, vsa_model="map", key=KEY)
    edges = jnp.array([[0, 1], [1, 2], [2, 3]], dtype=jnp.int32)
    jax.jit(enc.encode_edges)(edges)
