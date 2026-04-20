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

from bayes_hdc.distributions import (
    GaussianHV,
    bind_gaussian,
    bundle_gaussian,
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


def pmap_bundle_gaussian(hvs: GaussianHV) -> GaussianHV:
    """Bundle a sharded batch of Gaussian hypervectors across devices.

    ``hvs`` must carry a leading device-axis; the innermost bundle is
    performed per-device, then the per-device results are reduced on
    the host via a plain Python ``bundle_gaussian`` call.
    """
    per_device = jax.pmap(bundle_gaussian)(hvs)
    return bundle_gaussian(per_device)


__all__ = [
    "batch_bind_gaussian",
    "batch_similarity_gaussian",
    "pmap_bind_gaussian",
    "pmap_bundle_gaussian",
]
