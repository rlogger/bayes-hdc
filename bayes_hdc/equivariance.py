# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Group actions and equivariance on hypervectors.

The core HDC primitives have a clean group-theoretic structure. This module
makes that structure first-class — exposing the cyclic-shift action, the
equivariance/invariance laws of the primitives, and property-based checkers
for verifying that custom ops respect them.

## The group action

For a fixed dimension :math:`d`, cyclic shift by :math:`k` defines an action
of :math:`\\mathbb{Z}/d` on :math:`\\mathbb{R}^d`:

    T_k : R^d → R^d,   T_k(x)_i = x_{(i - k) mod d}

The :func:`~bayes_hdc.functional.permute` primitive *is* this action. It is:

* **faithful** — ``T_k(x) = x`` for all ``x`` iff ``k ≡ 0 (mod d)``;
* **additive in the shift** — ``T_j ∘ T_k = T_{j+k}``;
* **an isometry** — ``||T_k(x)|| = ||x||`` and ``⟨T_k(x), T_k(y)⟩ = ⟨x, y⟩``.

## Equivariances of the HDC primitives

Two flavours of shift equivariance matter here.

**Diagonal equivariance** — shifting *every* argument by the same ``k`` also
shifts the output by ``k``. Let :math:`\\star` denote element-wise product
binding (``bind_map``, ``bind_bsc``) and :math:`\\oplus` denote normalised sum
bundling (``bundle_map``, ``bundle_bsc``). Element-wise operations are
pointwise, so they commute with the diagonal shift:

    T_k(x ⋆ y)   = T_k(x) ⋆ T_k(y)                      (diagonal)
    T_k(⨁_i xi)  = ⨁_i T_k(xi)                          (diagonal)
    sim(T_k(x), T_k(y)) = sim(x, y)                     (diagonally invariant)

**Single-argument equivariance** — shifting *one* argument shifts the
output. This is the equivariance of circular convolution (``bind_hrr``) with
respect to its inputs:

    T_k(x * y) = T_k(x) * y = x * T_k(y)                (single-argument)

Under the *diagonal* action, circular convolution picks up a double shift:
``T_k(x) * T_k(y) = T_{2k}(x * y)``. That is not a defect — it is a
feature, and it is why shift *covariance* rather than shift invariance is
the right frame for HRR. Convolution is the canonical bilinear operator on
:math:`\\mathbb{R}^d` that respects the cyclic structure, which is the reason
Holographic Reduced Representations are a natural binding when positional
semantics matter.

## Why this matters

* **Equivariant neural functionals (NFNs).** Layers that respect the
  symmetries of weight-space benefit from a substrate whose symmetries are
  explicit. Hypervectors carry the :math:`\\mathbb{Z}/d` action out of the
  box; equivariant pipelines compose from primitives already here.

* **Weight-space representations.** A classifier's class centroids are a
  point in :math:`\\mathbb{R}^{K \\times d}`. Their posterior
  (:class:`~bayes_hdc.BayesianCentroidClassifier`) is a
  :class:`~bayes_hdc.GaussianHV` — a distribution over weight vectors whose
  symmetries are inherited from the hypervector algebra.

* **Structured representations.** ``bind(task, state)`` encodes a (task,
  state) pair whose symmetries follow from those of its operands. Useful for
  meta-RL and any setting where task or context conditioning is needed.

## What this module provides

* :func:`shift` — an alias for :func:`~bayes_hdc.functional.permute` that
  signals the group-action intent.
* :func:`verify_shift_equivariance` — property-based check that a function
  commutes with the action on all its hypervector arguments.
* :func:`verify_shift_invariance` — property-based check that a function is
  invariant under the diagonal action.
* :func:`hrr_equivariant_bilinear` — the canonical shift-equivariant
  bilinear operator (circular convolution) re-exported with an explicit
  equivariance-aware docstring.
* :func:`compose_shifts` — explicit group composition in :math:`\\mathbb{Z}/d`.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from bayes_hdc.functional import bind_hrr, permute


def shift(x: jax.Array, k: int) -> jax.Array:
    """Apply the :math:`\\mathbb{Z}/d` cyclic-shift action ``T_k`` to ``x``.

    This is an alias for :func:`~bayes_hdc.functional.permute` with the
    group-action name instead of the VSA-primitive name. Prefer this symbol
    when reasoning about equivariance; prefer ``permute`` when reasoning
    about the VSA algebra. They are literally the same function.

    Args:
        x: Hypervector of shape ``(..., d)``.
        k: Shift amount. Negative values apply the inverse action.

    Returns:
        ``T_k(x)``, the cyclically shifted hypervector.
    """
    return permute(x, k)


def compose_shifts(j: int, k: int, d: int) -> int:
    """Compose two shifts in :math:`\\mathbb{Z}/d`.

    ``T_j ∘ T_k = T_{(j+k) mod d}``. Useful for building equivariant layers
    that compose shifts symbolically before materialising them on arrays.

    Args:
        j: First shift amount.
        k: Second shift amount.
        d: Dimension of the hypervector space (the modulus).

    Returns:
        The shift amount ``(j + k) mod d`` in the canonical range ``[0, d)``.
    """
    return (j + k) % d


def verify_shift_equivariance(
    fn: Callable[..., jax.Array],
    *args: jax.Array,
    shifts: tuple[int, ...] = (1, 7, 123),
    atol: float = 1e-4,
) -> bool:
    """Check whether ``fn`` commutes with the diagonal :math:`\\mathbb{Z}/d` action.

    Verifies that for every shift ``k`` in ``shifts``:

    ``fn(T_k(args[0]), ..., T_k(args[-1])) ≈ T_k(fn(args))``

    to within ``atol``. All positional arguments must be hypervectors with a
    trailing dimension of the same size, since the action is applied along
    the last axis.

    Args:
        fn: Function under test. Must return a JAX array.
        *args: Hypervector arguments; the action is applied to each.
        shifts: Shift amounts to test. Default covers a small, medium, and
            moderately-large shift.
        atol: Absolute tolerance for the equality check.

    Returns:
        ``True`` iff ``fn`` is shift-equivariant on ``args`` at every shift
        in ``shifts``.
    """
    baseline = fn(*args)
    for k in shifts:
        shifted_args = tuple(shift(a, k) for a in args)
        shifted_out = fn(*shifted_args)
        expected = shift(baseline, k)
        if not bool(jnp.allclose(shifted_out, expected, atol=atol)):
            return False
    return True


def verify_single_argument_shift_equivariance(
    fn: Callable[..., jax.Array],
    *args: jax.Array,
    arg_index: int = 0,
    shifts: tuple[int, ...] = (1, 7, 123),
    atol: float = 1e-4,
) -> bool:
    """Check whether ``fn`` is shift-equivariant in a single argument.

    Verifies that for every shift ``k`` in ``shifts``, shifting only
    ``args[arg_index]`` shifts the output by the same ``k``:

    ``fn(..., T_k(args[arg_index]), ...) ≈ T_k(fn(args))``

    while the other arguments are held fixed. This is the equivariance
    structure of linear and bilinear operators whose kernel has the cyclic
    symmetry — circular convolution being the canonical example.

    Args:
        fn: Function under test.
        *args: Hypervector arguments.
        arg_index: Which positional argument to shift.
        shifts: Shift amounts to test.
        atol: Absolute tolerance for the equality check.

    Returns:
        ``True`` iff ``fn`` is single-argument shift-equivariant in the
        chosen position, at every shift in ``shifts``.
    """
    baseline = fn(*args)
    for k in shifts:
        perturbed = list(args)
        perturbed[arg_index] = shift(args[arg_index], k)
        shifted_out = fn(*perturbed)
        expected = shift(baseline, k)
        if not bool(jnp.allclose(shifted_out, expected, atol=atol)):
            return False
    return True


def verify_shift_invariance(
    fn: Callable[..., jax.Array],
    *args: jax.Array,
    shifts: tuple[int, ...] = (1, 7, 123),
    atol: float = 1e-4,
) -> bool:
    """Check whether ``fn`` is invariant under the diagonal :math:`\\mathbb{Z}/d` action.

    Verifies that for every shift ``k`` in ``shifts``:

    ``fn(T_k(args[0]), ..., T_k(args[-1])) ≈ fn(args)``

    to within ``atol``. Typical examples: :func:`~bayes_hdc.cosine_similarity`
    is invariant under the diagonal action because cyclic shift is an
    isometry of the inner product.

    Args:
        fn: Function under test. Must return a JAX array.
        *args: Hypervector arguments; the action is applied to each.
        shifts: Shift amounts to test.
        atol: Absolute tolerance for the equality check.

    Returns:
        ``True`` iff ``fn`` is invariant on ``args`` at every shift in
        ``shifts``.
    """
    baseline = fn(*args)
    for k in shifts:
        shifted_args = tuple(shift(a, k) for a in args)
        shifted_out = fn(*shifted_args)
        if not bool(jnp.allclose(shifted_out, baseline, atol=atol)):
            return False
    return True


def hrr_equivariant_bilinear(x: jax.Array, filter_hv: jax.Array) -> jax.Array:
    """The canonical single-argument shift-equivariant bilinear operator.

    Circular convolution (``bind_hrr``) is the bilinear operator on
    :math:`\\mathbb{R}^d` whose kernel has cyclic symmetry. It is equivariant
    in each argument separately:

    ``T_k(conv(x, f)) = conv(T_k(x), f) = conv(x, T_k(f))``

    Under the *diagonal* action it picks up a double shift:
    ``conv(T_k(x), T_k(f)) = T_{2k}(conv(x, f))``. This is the standard
    property of convolution and is why the right frame for HRR is shift
    covariance rather than shift invariance.

    Use this as a building block when composing permutation-equivariant
    layers over hypervectors — for example, in a neural functional whose
    inputs are hypervector-valued and whose symmetries include the cyclic
    group action.

    Args:
        x: Input hypervector of shape ``(..., d)``.
        filter_hv: Filter hypervector of shape ``(..., d)``.

    Returns:
        The convolved hypervector, of shape ``(..., d)``.
    """
    return bind_hrr(x, filter_hv)
