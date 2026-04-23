# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Weight-space as a PVSA posterior.

A short, self-contained demonstration that a classifier's "weights" are a
first-class object in this library — and specifically that
:class:`~bayes_hdc.BayesianCentroidClassifier` stores a distribution over
weight-space rather than a point estimate.

## What the weight-space research programme cares about

The recent programme of treating a trained network's weights as *data* — see
e.g. the literature on weight-space symmetries, equivariant neural
functionals (NFNs), and meta-learning over parameter tensors — asks three
things of a representation:

1. the weights must be a well-typed, inspectable object;
2. the symmetry group of weight-space (for us: the cyclic shift + channel
   permutations) must be first-class;
3. the representation should carry a distribution, not just a point.

This library gives all three for free on the HDC substrate. A
:class:`BayesianCentroidClassifier` is K class hypervectors
:math:`\\{\\mathbf{w}_c\\}_{c=1}^{K}`, each one a :class:`GaussianHV` with an
explicit posterior mean ``mu_c`` and per-dimension variance ``var_c``.

## What this example does

1. Draw 4 synthetic classes on the unit sphere.
2. Fit a :class:`BayesianCentroidClassifier` — the posterior over class
   centroids is :class:`GaussianHV`-valued.
3. Read off the posterior mean and variance per class — these *are* the
   weights and their uncertainty.
4. Sample several weight configurations from the posterior. Each sample is
   an entire alternate classifier.
5. Predict with each sampled classifier on a held-out query. Disagreement
   across samples is *epistemic uncertainty* — the query lies where the
   weight posterior is broad.
6. Verify that the whole pipeline respects the cyclic-shift symmetry of
   weight-space: shifting every training hypervector by the same ``k`` and
   refitting produces posterior centroids shifted by the same ``k``.

Run::

    python examples/weight_space_posterior.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bayes_hdc import (
    BayesianCentroidClassifier,
    GaussianHV,
    cosine_similarity,
    shift,
)

DIMS = 1024
NUM_CLASSES = 4
SAMPLES_PER_CLASS = 40
NUM_WEIGHT_SAMPLES = 8
SEED = 2026


def _make_clusters(
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Four well-separated unit-norm clusters on the sphere."""
    k_centres, k_noise = jax.random.split(key)
    centres = jax.random.normal(k_centres, (NUM_CLASSES, DIMS))
    centres = centres / jnp.linalg.norm(centres, axis=-1, keepdims=True)

    noise_keys = jax.random.split(k_noise, NUM_CLASSES)
    clusters = jnp.stack(
        [
            centres[c] + 0.08 * jax.random.normal(noise_keys[c], (SAMPLES_PER_CLASS, DIMS))
            for c in range(NUM_CLASSES)
        ]
    )
    # (K * N, D)
    hvs = clusters.reshape(-1, DIMS)
    labels = jnp.repeat(jnp.arange(NUM_CLASSES, dtype=jnp.int32), SAMPLES_PER_CLASS)
    return hvs, labels


def _sample_weight_configuration(clf: BayesianCentroidClassifier, key: jax.Array) -> jax.Array:
    """Draw one full weight matrix :math:`W \\in \\mathbb{R}^{K \\times d}`
    from the per-class Gaussian posteriors."""
    eps = jax.random.normal(key, clf.mu.shape)
    return clf.mu + eps * jnp.sqrt(clf.var + 1e-8)


def main() -> None:
    print("Weight-space as a PVSA posterior")
    print(f"  dimensions = {DIMS}")
    print(f"  classes    = {NUM_CLASSES}")
    print(f"  samples/cl = {SAMPLES_PER_CLASS}\n")

    key = jax.random.PRNGKey(SEED)
    k_data, k_samples = jax.random.split(key)

    hvs, labels = _make_clusters(k_data)
    clf = BayesianCentroidClassifier.create(
        num_classes=NUM_CLASSES,
        dimensions=DIMS,
    ).fit(hvs, labels, prior_strength=1.0)

    # --------------------------------------------------------------- 1.
    print("[1] The classifier's posterior is a distribution over weights.")
    mu = np.asarray(clf.mu)  # (K, D)
    var = np.asarray(clf.var)  # (K, D)
    mean_norm = float(np.linalg.norm(mu, axis=-1).mean())
    print(f"      posterior mean  shape = {mu.shape}   mean(||mu_c||) = {mean_norm:.3f}")
    print(f"      posterior var   shape = {var.shape}   mean(var_c)   = {var.mean():.5f}")

    # --------------------------------------------------------------- 2.
    print("\n[2] Each sample from the posterior is an alternate classifier.")
    sample_keys = jax.random.split(k_samples, NUM_WEIGHT_SAMPLES)
    sampled_ws = jnp.stack([_sample_weight_configuration(clf, k) for k in sample_keys])

    # Cosine similarity between the posterior mean and each sampled configuration.
    def _mean_cos(w_sample: jax.Array) -> float:
        return float(
            jnp.mean(
                jax.vmap(cosine_similarity)(
                    w_sample,
                    clf.mu,
                )
            )
        )

    print("      cos(sample, mean)  per draw:")
    for i, w_s in enumerate(sampled_ws):
        print(f"        draw {i}:  {_mean_cos(w_s):.3f}")

    # --------------------------------------------------------------- 3.
    print("\n[3] Epistemic uncertainty on a query = disagreement across draws.")
    # A single query vector that is ambiguously near the boundary of two classes.
    query = 0.5 * (clf.mu[0] + clf.mu[1])
    query = query / (jnp.linalg.norm(query) + 1e-8)

    def _predict_with(w: jax.Array, q: jax.Array) -> jax.Array:
        w_norm = w / (jnp.linalg.norm(w, axis=-1, keepdims=True) + 1e-8)
        return jnp.argmax(w_norm @ q)

    preds = jnp.asarray([int(_predict_with(w, query)) for w in sampled_ws])
    print(f"      query argmax across {NUM_WEIGHT_SAMPLES} posterior draws: {preds.tolist()}")
    _, counts = np.unique(np.asarray(preds), return_counts=True)
    entropy = -np.sum((counts / counts.sum()) * np.log(counts / counts.sum() + 1e-12))
    print(
        f"      predictive entropy = {entropy:.3f}  "
        "(0 = all draws agree; log(K) = maximum disagreement)"
    )

    # --------------------------------------------------------------- 4.
    print("\n[4] Symmetry check — the weight posterior is Z/d-equivariant.")
    k_shift = 17
    hvs_shifted = jax.vmap(lambda h: shift(h, k_shift))(hvs)
    clf_shifted = BayesianCentroidClassifier.create(
        num_classes=NUM_CLASSES,
        dimensions=DIMS,
    ).fit(hvs_shifted, labels, prior_strength=1.0)

    # The shifted classifier's mu should be the unshifted mu, shifted by k.
    diff = jnp.linalg.norm(clf_shifted.mu - jax.vmap(lambda w: shift(w, k_shift))(clf.mu))
    print(
        f"      ||mu_shifted − T_k(mu)|| = {float(diff):.2e}  "
        "(≈ 0 ⟹ the posterior commutes with the cyclic-shift action)"
    )

    # --------------------------------------------------------------- 5.
    print("\n[5] The posterior is a GaussianHV and composes like one.")
    posterior_class_0 = GaussianHV(mu=clf.mu[0], var=clf.var[0], dimensions=DIMS)
    print(
        f"      class-0 posterior: GaussianHV(mu.shape={tuple(posterior_class_0.mu.shape)}, "
        f"var.shape={tuple(posterior_class_0.var.shape)})"
    )
    print("      → bind_gaussian, bundle_gaussian, kl_gaussian, reparam. samplers")
    print("        all accept it directly; it is a first-class object.")


if __name__ == "__main__":
    main()
