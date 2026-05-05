# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Multi-device and streaming wrappers for PVSA primitives.

This module packages ``jax.pmap`` / ``jax.vmap`` parallelisations of
the PVSA bind/bundle primitives. On single-device hosts these degrade
to single-device execution (JAX treats the host as one device); on
multi-device hosts (CUDA GPU pods, TPU pods) they shard the work
across devices automatically.

Two modes:

- **Intra-op parallelism (``vmap``)** — the usual batched bind /
  bundle; call :func:`batch_bind_gaussian` to apply ``bind_gaussian``
  over a leading batch axis without writing a Python loop.
- **Multi-device parallelism (``pmap``)** — expects a leading
  ``num_devices`` axis. Use :func:`pmap_bind_gaussian` /
  :func:`pmap_bundle_gaussian` to spread a distributional VSA
  computation across an accelerator pod.

Nothing in this module loads JAX-pmap runtime state at import time,
so ``bayes_hdc.distributed`` is always safe to import even on
hosts without multiple devices.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from bayes_hdc.constants import EPS
from bayes_hdc.distributions import (
    GaussianHV,
    bind_gaussian,
    expected_cosine_similarity,
)

# ----------------------------------------------------------------------
# Intra-op parallelism (vmap)
# ----------------------------------------------------------------------


# ``vmap`` over the leading batch axis of both Gaussian HV operands.
batch_bind_gaussian = jax.vmap(bind_gaussian, in_axes=(0, 0))
"""Batched :func:`~bayes_hdc.distributions.bind_gaussian` — leading batch axis on both inputs."""


batch_similarity_gaussian = jax.vmap(expected_cosine_similarity, in_axes=(0, None))
"""Batched expected cosine similarity — batched query against a single target."""


# ----------------------------------------------------------------------
# Multi-device parallelism (pmap)
# ----------------------------------------------------------------------


def pmap_bind_gaussian(x: GaussianHV, y: GaussianHV) -> GaussianHV:
    """Bind two sharded Gaussian hypervector batches across devices.

    Both ``x`` and ``y`` must have a leading axis whose size matches
    the local device count (``jax.local_device_count()``). Returns a
    sharded :class:`GaussianHV` with the same leading shape.
    """
    return jax.pmap(bind_gaussian)(x, y)


def _sum_gaussian_hvs(hvs: GaussianHV) -> GaussianHV:
    """Sum-only reduction over the leading axis. No normalisation.

    Internal helper for :func:`pmap_bundle_gaussian`: each device sums
    its local batch of hypervectors, the cross-device sum and the final
    L2-norm normalisation happen exactly once on the host. This avoids
    the algebraic mismatch that would arise from composing
    ``bundle_gaussian`` twice (per-device, then over per-device
    normalised results).
    """
    return GaussianHV(
        mu=jnp.sum(hvs.mu, axis=0),
        var=jnp.sum(hvs.var, axis=0),
        dimensions=hvs.mu.shape[-1],
    )


def pmap_bundle_gaussian(hvs: GaussianHV) -> GaussianHV:
    """Bundle a sharded batch of Gaussian hypervectors across devices.

    Computes the *globally* normalised bundle: per-device partial sums
    are accumulated via ``pmap``, the cross-device reduction and the
    final ``mu / ||sum_mu||`` normalisation happen once on the host.
    This is algebraically identical to a single-device call to
    :func:`~bayes_hdc.distributions.bundle_gaussian` on the un-sharded
    batch — unlike a naive bundle-then-bundle composition, which would
    normalise twice and break the equivalence.

    ``hvs`` must carry a leading device-axis of size
    ``jax.local_device_count()``.
    """
    per_device = jax.pmap(_sum_gaussian_hvs)(hvs)
    summed_mu = jnp.sum(per_device.mu, axis=0)
    summed_var = jnp.sum(per_device.var, axis=0)
    norm = jnp.linalg.norm(summed_mu) + EPS
    return GaussianHV(
        mu=summed_mu / norm,
        var=summed_var / (norm**2),
        dimensions=summed_mu.shape[-1],
    )


def shard_map_bind_gaussian(x: GaussianHV, y: GaussianHV) -> GaussianHV:
    """``shard_map``-based bind for multi-accelerator pods (JAX ≥ 0.4.24).

    Uses explicit axis annotation via ``jax.experimental.shard_map``
    instead of the older ``pmap`` API. Inputs must carry a leading axis
    of size ``jax.local_device_count()``. Returns a sharded
    :class:`GaussianHV`.

    Falls back to a plain :func:`~bayes_hdc.distributions.bind_gaussian`
    call if the host has only one device (nothing to shard) or if the
    installed JAX version does not expose ``shard_map``.
    """
    try:
        from jax.experimental.shard_map import shard_map
        from jax.sharding import Mesh, PartitionSpec

        n_dev = jax.local_device_count()
        if n_dev < 2:
            return bind_gaussian(x, y)

        mesh = Mesh(jax.devices()[:n_dev], axis_names=("i",))
        spec = PartitionSpec("i", None)
        sharded = shard_map(
            bind_gaussian,
            mesh=mesh,
            in_specs=(spec, spec),
            out_specs=spec,
        )
        return sharded(x, y)
    except Exception:  # pragma: no cover — JAX version dependent
        # Fall through to pmap, then single-device.
        try:
            return jax.pmap(bind_gaussian)(x, y)
        except Exception:
            return bind_gaussian(x, y)


def shard_classifier_posteriors(
    mu: jax.Array,
    var: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Reshape ``(K, d)`` class posteriors into ``(n_devices, K // n_devices, d)``.

    Assumes ``K`` is divisible by ``jax.local_device_count()``. Returns
    the sharded mean and variance arrays; pass them to
    :func:`shard_map_bind_gaussian` or a user-defined sharded op. On a
    single-device host, returns the inputs unchanged with a leading
    axis of size 1.
    """
    n_dev = jax.local_device_count()
    k = mu.shape[0]
    if k % n_dev != 0:
        raise ValueError(
            f"num_classes ({k}) must be divisible by local_device_count ({n_dev}) "
            "for sharded posteriors."
        )
    reshape = (n_dev, k // n_dev, mu.shape[1])
    return mu.reshape(reshape), var.reshape(reshape)


__all__ = [
    "batch_bind_gaussian",
    "batch_similarity_gaussian",
    "pmap_bind_gaussian",
    "pmap_bundle_gaussian",
    "shard_map_bind_gaussian",
    "shard_classifier_posteriors",
]
