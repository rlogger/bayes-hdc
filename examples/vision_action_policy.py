# SPDX-License-Identifier: MIT
# Copyright (c) 2026 R.S.

"""Synthetic multimodal continuous-action regression with marginal intervals.

Independent Gaussian feature vectors stand in for vision embeddings, and a
second vector represents proprioception. Two normalised projections are
bundled and a ridge head predicts an additive noisy target. No images,
language instructions, trajectories, environment interaction or real robot
are used. Thus this is not a vision-language-action policy evaluation.

Independent calibration supplies marginal per-coordinate conformal intervals
under exchangeability. The magnitude-based abstention rule is a separate
heuristic; neither it nor the intervals establish control safety, simultaneous
multi-coordinate coverage, or error control conditional on executing actions.
Projection and normalisation can discard magnitude/information, so performance
with a real backbone must be evaluated rather than assumed.

Run: python examples/vision_action_policy.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bayes_hdc import (
    MAP,
    ConformalRegressor,
    HDRegressor,
    ProjectionEncoder,
    bundle_map,
)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

DIMS = 4096  # hypervector dimension
SEED = 2026

# Frozen vision backbone output dimension.
# 384 = DINOv2-S; 512 = CLIP-B/32; 768 = CLIP-L/14, SigLIP-base.
VISION_FEATURE_DIM = 384

# Robotic state.
PROPRIO_DIM = 7  # 7-DOF arm joint angles
ACTION_DIM = 7  # joint velocity command per DOF
ALPHA = 0.1  # 90 % target coverage on action intervals

# Dataset sizes.
N_SCENES = 6  # number of distinct scene "modes" in the synthetic data
N_TRAIN = 800
N_CAL = 400
N_TEST = 600


# ----------------------------------------------------------------------
# Synthetic VLA dataset
# ----------------------------------------------------------------------


def synthesise_scene_templates(key: jax.Array) -> jax.Array:
    """Generate ``N_SCENES`` random vision-feature centroids.

    In real data each centroid corresponds to a recurring scene
    configuration (e.g. "red cube on left of green cube"); a frozen
    vision backbone may map similar scenes to nearby feature vectors,
    but this synthetic construction does not evaluate that assumption.
    """
    return jax.random.normal(key, (N_SCENES, VISION_FEATURE_DIM))


def synthesise_dataset(
    key: jax.Array,
    scene_templates: jax.Array,
    n: int,
    vision_basis: jax.Array,
    proprio_basis: jax.Array,
):
    """Sample (vision_features, proprio, action) triples.

    Vision features are scene-conditional Gaussians (a clean centroid
    plus small noise); proprio is iid Gaussian; action is a fixed
    linear function of both modalities plus noise. The HDC pipeline
    has to recover this linear map through two random projections and
    a normalised bundle.
    """
    k_scene, k_v, k_p, k_eps = jax.random.split(key, 4)
    scene_ids = jax.random.randint(k_scene, (n,), 0, N_SCENES)
    vision_features = scene_templates[scene_ids] + 0.05 * jax.random.normal(
        k_v, (n, VISION_FEATURE_DIM)
    )
    proprio = jax.random.normal(k_p, (n, PROPRIO_DIM))
    # Linear ground truth: action = vision @ B_v + proprio @ B_p + noise.
    actions = vision_features @ vision_basis + proprio @ proprio_basis
    actions = actions + 0.1 * jax.random.normal(k_eps, (n, ACTION_DIM))
    return vision_features, proprio, actions


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------


def encode_state(
    vision_encoder: ProjectionEncoder,
    proprio_encoder: ProjectionEncoder,
    vision_features: jax.Array,
    proprio: jax.Array,
) -> jax.Array:
    """Normalise two projections, then sum and normalise the modality vectors.

    Normalisation discards magnitude, so additive target recovery is an
    empirical property of the fitted head rather than a linearity guarantee.
    """
    v_hv = vision_encoder.encode_batch(vision_features)
    p_hv = proprio_encoder.encode_batch(proprio)
    # Stack into (n, 2, d) and bundle along the modality axis.
    stacked = jnp.stack([v_hv, p_hv], axis=1)
    return jax.vmap(lambda s: bundle_map(s, axis=0))(stacked)


def main() -> None:
    print("Calibrated multi-modal action prediction (vision + proprio → action)")
    print(
        f"  d = {DIMS}    vision = {VISION_FEATURE_DIM}    proprio = {PROPRIO_DIM}"
        f"    action = {ACTION_DIM}    α = {ALPHA}\n"
    )

    key = jax.random.PRNGKey(SEED)
    (
        k_scene,
        k_vbasis,
        k_pbasis,
        k_train,
        k_cal,
        k_test,
        k_v_enc,
        k_p_enc,
    ) = jax.random.split(key, 8)

    # ----------------------------------------------------------------- 1.
    print("[1] Synthesise scene templates and the linear ground-truth map.")
    scene_templates = synthesise_scene_templates(k_scene)
    vision_basis = jax.random.normal(k_vbasis, (VISION_FEATURE_DIM, ACTION_DIM)) * 0.05
    proprio_basis = jax.random.normal(k_pbasis, (PROPRIO_DIM, ACTION_DIM)) * 0.5
    print(f"      scenes: {N_SCENES}")
    print(f"      vision_basis shape: {tuple(vision_basis.shape)}")
    print(f"      proprio_basis shape: {tuple(proprio_basis.shape)}")

    # ----------------------------------------------------------------- 2.
    print("\n[2] Build encoders for the two modalities.")
    vision_encoder = ProjectionEncoder.create(
        input_dim=VISION_FEATURE_DIM,
        dimensions=DIMS,
        vsa_model=MAP.create(dimensions=DIMS),
        key=k_v_enc,
    )
    proprio_encoder = ProjectionEncoder.create(
        input_dim=PROPRIO_DIM,
        dimensions=DIMS,
        vsa_model=MAP.create(dimensions=DIMS),
        key=k_p_enc,
    )
    print(f"      vision : {VISION_FEATURE_DIM}-d → {DIMS}-d (random projection)")
    print(f"      proprio: {PROPRIO_DIM}-d → {DIMS}-d (random projection)")

    # ----------------------------------------------------------------- 3.
    print("\n[3] Sample train / calibration / test sets.")
    train_v, train_p, train_a = synthesise_dataset(
        k_train, scene_templates, N_TRAIN, vision_basis, proprio_basis
    )
    cal_v, cal_p, cal_a = synthesise_dataset(
        k_cal, scene_templates, N_CAL, vision_basis, proprio_basis
    )
    test_v, test_p, test_a = synthesise_dataset(
        k_test, scene_templates, N_TEST, vision_basis, proprio_basis
    )
    print(f"      n_train = {N_TRAIN}    n_cal = {N_CAL}    n_test = {N_TEST}")
    print(f"      action stats — mean = {float(jnp.mean(train_a)):+.3f}   ", end="")
    print(f"std = {float(jnp.std(train_a)):.3f}")

    # ----------------------------------------------------------------- 4.
    print("\n[4] Encode each batch into bundled state hypervectors.")
    train_state = encode_state(vision_encoder, proprio_encoder, train_v, train_p)
    cal_state = encode_state(vision_encoder, proprio_encoder, cal_v, cal_p)
    test_state = encode_state(vision_encoder, proprio_encoder, test_v, test_p)
    print(f"      train_state shape: {tuple(train_state.shape)}")

    # ----------------------------------------------------------------- 5.
    print("\n[5] Fit the HDRegressor policy head (ridge on state hypervectors).")
    policy = HDRegressor.create(dimensions=DIMS, output_dim=ACTION_DIM, reg=0.5)
    policy = policy.fit(train_state, train_a)
    train_r2 = float(policy.score(train_state, train_a))
    test_r2 = float(policy.score(test_state, test_a))
    print(f"      R² (train) = {train_r2:+.4f}")
    print(f"      R² (test)  = {test_r2:+.4f}")

    test_preds = policy.predict(test_state)
    cal_preds = policy.predict(cal_state)
    rmse_per_dof = jnp.sqrt(jnp.mean((test_preds - test_a) ** 2, axis=0))
    print(f"      RMSE per DOF: {np.array2string(np.asarray(rmse_per_dof), precision=3)}")

    # ----------------------------------------------------------------- 6.
    print(f"\n[6] Calibrate ConformalRegressor at α = {ALPHA}.")
    cr = ConformalRegressor.create(alpha=ALPHA, output_dim=ACTION_DIM).fit(cal_preds, cal_a)
    width = np.asarray(cr.interval_width())
    print(f"      per-DOF interval width: {np.array2string(width, precision=3)}")
    print(f"      n_calibration: {cr.n_calibration}")

    # ----------------------------------------------------------------- 7.
    print("\n[7] Empirical coverage on held-out test set.")
    coverage = np.asarray(cr.coverage(test_preds, test_a))
    print(f"      target coverage:      1 - α = {1 - ALPHA:.2f}")
    print(f"      empirical (per DOF):  {np.array2string(coverage, precision=3)}")
    print(f"      empirical (mean):     {float(coverage.mean()):.3f}")
    print("      These are per-coordinate marginals, not simultaneous action coverage.")
    print("      One empirical coverage rate does not validate the theorem's assumptions.")

    # ----------------------------------------------------------------- 8.
    print("\n[8] Magnitude-based abstention heuristic (no safety guarantee).")
    # The threshold is fixed before test evaluation. Marginal conformal
    # coverage does not imply error control on the retained subset.
    pred_norm = jnp.linalg.norm(test_preds, axis=-1)
    interval_norm = jnp.linalg.norm(cr.quantile)  # constant scalar
    threshold_ratio = 0.6  # abstain when interval_norm > 0.6 * pred_norm
    abstain_mask = pred_norm < (interval_norm / threshold_ratio)
    n_abstain = int(jnp.sum(abstain_mask))
    n_act = N_TEST - n_abstain
    print(
        f"      rule: abstain when ‖ŷ‖ < ‖q‖ / {threshold_ratio} = "
        f"{float(interval_norm / threshold_ratio):.3f}"
    )
    print(f"      acted on:  {n_act} / {N_TEST} ({n_act / N_TEST:.1%})")
    print(f"      abstained: {n_abstain} / {N_TEST} ({n_abstain / N_TEST:.1%})")

    if 0 < n_act < N_TEST:
        kept = ~abstain_mask
        rmse_acted = float(jnp.sqrt(jnp.mean((test_preds[kept] - test_a[kept]) ** 2)))
        rmse_abstain = float(
            jnp.sqrt(jnp.mean((test_preds[abstain_mask] - test_a[abstain_mask]) ** 2))
        )
        rel_acted = rmse_acted / float(jnp.mean(pred_norm[kept]))
        rel_abstain = rmse_abstain / max(float(jnp.mean(pred_norm[abstain_mask])), 1e-8)
        print(f"      RMSE on acted points: {rmse_acted:.4f}")
        print(f"      RMSE on abstained:    {rmse_abstain:.4f}")
        print(f"      relative err (acted):     {rel_acted:.3f}")
        print(f"      relative err (abstained): {rel_abstain:.3f}")
        print("      Subset errors are descriptive; conditional risk is not controlled.")

    # ----------------------------------------------------------------- 9.
    print("\nThis synthetic regression demo measures no real perception or control outcome.")
    print("Real trajectories require suitable temporal/group splits and independent validation.")


if __name__ == "__main__":
    main()
