# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Property-based tests for the group-theoretic structure on hypervectors.

These tests verify, on pseudo-random hypervectors, the algebraic laws that
the module docstring of :mod:`bayes_hdc.equivariance` claims to hold by
construction:

* the cyclic-shift action is faithful, additive, and isometric;
* element-wise product binding is :math:`\\mathbb{Z}/d`-equivariant;
* normalised-sum bundling is :math:`\\mathbb{Z}/d`-equivariant;
* cosine similarity is :math:`\\mathbb{Z}/d`-invariant (diagonal action);
* circular-convolution binding (HRR) is :math:`\\mathbb{Z}/d`-equivariant;
* ``verify_shift_equivariance`` / ``verify_shift_invariance`` are
  themselves correct detectors — they reject obvious counterexamples.
"""

import jax
import jax.numpy as jnp
import pytest

from bayes_hdc import (
    bind_map,
    bundle_map,
    cosine_similarity,
    permute,
)
from bayes_hdc.equivariance import (
    compose_shifts,
    hrr_equivariant_bilinear,
    shift,
    verify_shift_equivariance,
    verify_shift_invariance,
    verify_single_argument_shift_equivariance,
)
from bayes_hdc.functional import bind_hrr

DIMS = 1024
SEED = 2026


@pytest.fixture
def hv_pair() -> tuple[jax.Array, jax.Array]:
    k1, k2 = jax.random.split(jax.random.PRNGKey(SEED))
    x = jax.random.normal(k1, (DIMS,))
    y = jax.random.normal(k2, (DIMS,))
    return x, y


class TestShiftAction:
    """The cyclic-shift action T_k on R^d."""

    def test_shift_equals_permute(self, hv_pair: tuple[jax.Array, jax.Array]) -> None:
        x, _ = hv_pair
        assert jnp.allclose(shift(x, 7), permute(x, 7))

    def test_shift_faithful(self, hv_pair: tuple[jax.Array, jax.Array]) -> None:
        # T_k(x) = x iff k ≡ 0 (mod d) — test the forward direction.
        x, _ = hv_pair
        assert not jnp.allclose(shift(x, 1), x)
        assert jnp.allclose(shift(x, DIMS), x)  # full period

    def test_shift_additive(self, hv_pair: tuple[jax.Array, jax.Array]) -> None:
        # T_j ∘ T_k = T_{j+k}
        x, _ = hv_pair
        j, k = 5, 11
        assert jnp.allclose(shift(shift(x, k), j), shift(x, j + k))

    def test_shift_isometry_norm(self, hv_pair: tuple[jax.Array, jax.Array]) -> None:
        x, _ = hv_pair
        assert jnp.allclose(jnp.linalg.norm(shift(x, 13)), jnp.linalg.norm(x))

    def test_shift_isometry_inner_product(self, hv_pair: tuple[jax.Array, jax.Array]) -> None:
        x, y = hv_pair
        k = 42
        original = jnp.dot(x, y)
        shifted = jnp.dot(shift(x, k), shift(y, k))
        assert jnp.allclose(original, shifted, atol=1e-4)


class TestComposeShifts:
    def test_modular_composition(self) -> None:
        assert compose_shifts(3, 5, 10) == 8
        assert compose_shifts(7, 9, 10) == 6  # wraps
        assert compose_shifts(-2, 3, 10) == 1  # negative wraps
        assert compose_shifts(0, 0, 10) == 0  # identity


class TestPrimitiveEquivariances:
    """bind / bundle / cosine under the diagonal Z/d action."""

    def test_bind_map_equivariant(self, hv_pair: tuple[jax.Array, jax.Array]) -> None:
        x, y = hv_pair
        assert verify_shift_equivariance(bind_map, x, y)

    def test_bind_hrr_single_argument_equivariant(
        self, hv_pair: tuple[jax.Array, jax.Array]
    ) -> None:
        # Circular convolution is equivariant in each argument separately:
        # T_k(conv(x, y)) = conv(T_k(x), y) = conv(x, T_k(y)).
        x, y = hv_pair
        assert verify_single_argument_shift_equivariance(bind_hrr, x, y, arg_index=0)
        assert verify_single_argument_shift_equivariance(bind_hrr, x, y, arg_index=1)

    def test_bind_hrr_double_shift_under_diagonal(
        self, hv_pair: tuple[jax.Array, jax.Array]
    ) -> None:
        # conv(T_k(x), T_k(y)) = T_{2k}(conv(x, y)) — the standard
        # convolution-under-diagonal-shift identity.
        x, y = hv_pair
        k = 17
        lhs = bind_hrr(shift(x, k), shift(y, k))
        rhs = shift(bind_hrr(x, y), 2 * k)
        assert jnp.allclose(lhs, rhs, atol=1e-4)

    def test_hrr_equivariant_bilinear_alias(self, hv_pair: tuple[jax.Array, jax.Array]) -> None:
        x, y = hv_pair
        assert jnp.allclose(hrr_equivariant_bilinear(x, y), bind_hrr(x, y), atol=1e-5)

    def test_bundle_map_equivariant(self) -> None:
        keys = jax.random.split(jax.random.PRNGKey(SEED), 4)
        xs = jnp.stack([jax.random.normal(k, (DIMS,)) for k in keys])

        def bundle_fn(stacked: jax.Array) -> jax.Array:
            return bundle_map(stacked, axis=0)

        # Shift each row, bundle, compare to bundle-then-shift.
        baseline = bundle_fn(xs)
        shifted_rows = jax.vmap(lambda row: shift(row, 17))(xs)
        shifted_bundle = bundle_fn(shifted_rows)
        assert jnp.allclose(shifted_bundle, shift(baseline, 17), atol=1e-4)

    def test_cosine_invariant(self, hv_pair: tuple[jax.Array, jax.Array]) -> None:
        x, y = hv_pair
        assert verify_shift_invariance(cosine_similarity, x, y)


class TestVerifiersDetectViolations:
    """The verifiers must reject obvious counterexamples."""

    def test_non_equivariant_detected(self, hv_pair: tuple[jax.Array, jax.Array]) -> None:
        # Zeroing the first component is not shift-equivariant:
        # the first component is the "fixed point" for shift, not a structural property.
        def bad_fn(x: jax.Array) -> jax.Array:
            return x.at[0].set(0.0)

        x, _ = hv_pair
        assert not verify_shift_equivariance(bad_fn, x)

    def test_non_invariant_detected(self, hv_pair: tuple[jax.Array, jax.Array]) -> None:
        # Returning the first component is not shift-invariant.
        def bad_fn(x: jax.Array) -> jax.Array:
            return x[0]

        x, _ = hv_pair
        assert not verify_shift_invariance(bad_fn, x)
