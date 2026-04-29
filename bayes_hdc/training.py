# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""End-to-end variational training for PVSA codebooks and posteriors.

The library's core primitives — ``bind_gaussian``, ``bundle_gaussian``,
``permute_gaussian``, ``inverse_gaussian``, ``kl_gaussian`` — are all
pure JAX functions on registered pytrees. ``jax.grad`` therefore
composes through every PVSA operation without any user-side
flattening, which means a variational codebook (or any
:class:`~bayes_hdc.distributions.GaussianHV` parameterised by free
variational parameters) can be trained end-to-end with the same
optimiser idioms as a Flax / Equinox model.

This module ships a minimal, dependency-free Adam optimiser
(:func:`adam_init` / :func:`adam_update` / :class:`AdamState`) and a
high-level training loop, :func:`train_variational_codebook`, that
composes ``jit`` and ``lax.scan`` over user-supplied loss functions.

The intended pattern is::

    def loss_fn(params, key):
        posterior = GaussianHV(mu=params["mu"], var=jnp.exp(params["log_var"]),
                                dimensions=D)
        prior = GaussianHV.create(D)
        recon = reconstruction_log_likelihood_mc(posterior, target, key, 16)
        return -elbo_gaussian(posterior, prior, recon)

    result = train_variational_codebook(
        init_params={"mu": jnp.zeros(D), "log_var": jnp.zeros(D)},
        loss_fn=loss_fn,
        key=jax.random.PRNGKey(0),
        n_steps=1000,
        learning_rate=1e-2,
    )

The trainer makes no assumptions about the parameterisation; the user
is responsible for ensuring whatever pytree they pass in remains
compatible with ``jax.grad`` (avoid storing static fields on the
optimisation pytree). To our knowledge no other open-source HDC / VSA
library exposes a comparable end-to-end variational training API
[Heddes et al. 2023, J. Mach. Learn. Res. 24(255); Cumbo et al. 2023,
J. Open Source Softw. 8(89): 5704; Bekolay et al. 2014, Front.
Neuroinform. 7: 48].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

from bayes_hdc._compat import register_dataclass

# =============================================================================
# Adam optimiser — minimal, dependency-free, pytree-native.
# =============================================================================


@register_dataclass
@dataclass
class AdamState:
    """Diagonal-Adam optimiser state.

    Attributes:
        m: First-moment estimate, same pytree structure as ``params``.
        v: Second-moment estimate, same pytree structure as ``params``.
        step: Scalar update counter (used for bias correction).
    """

    m: Any
    v: Any
    step: jax.Array


def adam_init(params: Any) -> AdamState:
    """Initialise an :class:`AdamState` matched to ``params``."""
    zeros = jax.tree.map(jnp.zeros_like, params)
    return AdamState(m=zeros, v=zeros, step=jnp.asarray(0, dtype=jnp.int32))


def adam_update(
    state: AdamState,
    grads: Any,
    *,
    learning_rate: float = 1e-3,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[AdamState, Any]:
    """One Adam step. Returns ``(new_state, update)``.

    The caller applies the update by adding ``update`` to the params:
    ``new_params = jax.tree.map(jnp.add, params, update)``. Splitting
    state from update keeps :func:`jit` traces clean and the function
    purely functional (no in-place mutation).
    """
    step = state.step + 1
    new_m = jax.tree.map(lambda mi, gi: b1 * mi + (1 - b1) * gi, state.m, grads)
    new_v = jax.tree.map(lambda vi, gi: b2 * vi + (1 - b2) * gi**2, state.v, grads)

    bc1 = 1.0 - b1**step
    bc2 = 1.0 - b2**step

    update = jax.tree.map(
        lambda mi, vi: -learning_rate * (mi / bc1) / (jnp.sqrt(vi / bc2) + eps),
        new_m,
        new_v,
    )
    return AdamState(m=new_m, v=new_v, step=step), update


# =============================================================================
# Training loop.
# =============================================================================


@register_dataclass
@dataclass
class TrainResult:
    """Output of :func:`train_variational_codebook`.

    Registered as a JAX pytree so the entire trainer can be wrapped in
    :func:`jax.jit`.

    Attributes:
        params: Fitted parameters, same pytree shape as ``init_params``.
        loss_history: Per-step loss as a 1-D array of length ``n_steps``.
    """

    params: Any
    loss_history: jax.Array

    @property
    def final_loss(self) -> jax.Array:
        """Scalar loss at the last training step."""
        return self.loss_history[-1]


def train_variational_codebook(
    init_params: Any,
    loss_fn: Callable[[Any, jax.Array], jax.Array],
    *,
    key: jax.Array,
    n_steps: int = 1_000,
    learning_rate: float = 1e-2,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> TrainResult:
    """End-to-end Adam training loop for variational PVSA objectives.

    Composes :func:`jax.grad` through ``loss_fn`` and runs ``n_steps``
    Adam updates inside a single :func:`jax.lax.scan`, so the entire
    training trajectory compiles to one XLA program. Works for any
    pytree of parameters that ``loss_fn`` is differentiable in.

    Args:
        init_params: Initial parameters as a pytree (typically a dict
            of arrays — e.g. ``{"mu": ..., "log_var": ...}``). Every
            leaf must be a ``jax.Array`` of a floating-point dtype.
        loss_fn: Scalar loss as ``(params, key) -> Array(())``. The
            ``key`` argument is fresh per step and may be split inside
            for Monte-Carlo reconstruction terms.
        key: A ``jax.random.PRNGKey``. Will be split internally into
            one fold-in per step so that loss evaluations are
            decorrelated across steps (as required for unbiased
            stochastic-gradient ELBO optimisation).
        n_steps: Number of Adam updates.
        learning_rate: Adam learning rate.
        b1, b2, eps: Standard Adam hyperparameters.

    Returns:
        A :class:`TrainResult` with the fitted parameters and the
        per-step loss trajectory.

    Notes:
        For numerical stability the user should parameterise variances
        on a log scale (e.g. store ``log_var`` and reconstruct
        ``var = jnp.exp(log_var)`` inside ``loss_fn``). The trainer
        itself imposes no constraints on the parameter pytree.
    """
    state0 = adam_init(init_params)

    def step_fn(carry, _):
        params, opt_state, key = carry
        key, subkey = jax.random.split(key)
        loss, grads = jax.value_and_grad(loss_fn)(params, subkey)
        opt_state, update = adam_update(
            opt_state,
            grads,
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            eps=eps,
        )
        params = jax.tree.map(jnp.add, params, update)
        return (params, opt_state, key), loss

    (final_params, _, _), losses = jax.lax.scan(
        step_fn,
        (init_params, state0, key),
        xs=None,
        length=n_steps,
    )

    return TrainResult(
        params=final_params,
        loss_history=losses,
    )


__all__ = [
    "AdamState",
    "TrainResult",
    "adam_init",
    "adam_update",
    "train_variational_codebook",
]
