# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Property-based tests for the Fodor-Pylyshyn 1988 cognitive-architecture
challenges that VSAs answer.

* **Systematicity** — if the architecture can represent ``aRb`` it can
  also represent ``bRa``. In a VSA this is automatic: role-filler bundles
  with the role/filler order swapped are equally constructible and decode
  symmetrically. The tests below witness this property across the
  algebraically-clean VSA models in the library.

* **Productivity** — finitely many primitives generate infinitely many
  structures. The fixed-dimensional closure of ``bind`` (``R^d × R^d → R^d``
  rather than ``R^{d^2}``) is exactly the technical fact Gayler 2003
  invokes to answer this challenge. The tests construct deeply nested
  binding chains and verify that retrieval-by-cue still works.

References
----------
Fodor, J. A., Pylyshyn, Z. W. (1988). Connectionism and Cognitive
Architecture: A Critical Analysis. Cognition 28(1-2): 3-71.
Gayler, R. W. (2003). Vector Symbolic Architectures answer Jackendoff's
challenges for cognitive neuroscience. arXiv:cs/0412059.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from bayes_hdc import (
    BSC,
    HRR,
    MAP,
)
from bayes_hdc.functional import cosine_similarity, hamming_similarity

DIMS = 10000
SEED = 2026


# Models for which bind / inverse / bundle round-trip cleanly enough to
# witness systematicity and productivity at d = 10000. CGR/MCR/VTB use
# domain-specific similarity metrics that are tested elsewhere.
MODELS_AND_SIM = [
    ("BSC", BSC, hamming_similarity),
    ("MAP", MAP, cosine_similarity),
    ("HRR", HRR, cosine_similarity),
]


def _make_atoms(model, key, n: int):
    keys = jax.random.split(key, n)
    return [model.random(k, (DIMS,)) for k in keys]


# ----------------------------------------------------------------------
# Systematicity (Fodor & Pylyshyn 1988): if a system can represent aRb it
# can represent bRa.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name,model_cls,sim_fn", MODELS_AND_SIM)
def test_systematicity_role_filler_swap(name: str, model_cls, sim_fn) -> None:
    """Swapping the *fillers* across two role bindings reverses retrieval.

    Encode ``aRb`` as ``bind(role_a, A) + bind(role_b, B)`` and ``bRa``
    as ``bind(role_a, B) + bind(role_b, A)``. Unbinding each with the
    appropriate role recovers the right filler in both directions.
    """
    model = model_cls.create(dimensions=DIMS)
    key = jax.random.PRNGKey(SEED)
    role_a, role_b, atom_a, atom_b = _make_atoms(model, key, 4)

    # Forward: aRb
    forward = model.bundle(
        jnp.stack([model.bind(role_a, atom_a), model.bind(role_b, atom_b)]),
        axis=0,
    )
    # Swap: bRa (same algebra, fillers in the other slots)
    swap = model.bundle(
        jnp.stack([model.bind(role_a, atom_b), model.bind(role_b, atom_a)]),
        axis=0,
    )

    # Decode forward: forward * inverse(role_a) ≈ atom_a; ...inverse(role_b) ≈ atom_b
    fwd_a = model.bind(forward, model.inverse(role_a))
    fwd_b = model.bind(forward, model.inverse(role_b))
    swp_a = model.bind(swap, model.inverse(role_a))
    swp_b = model.bind(swap, model.inverse(role_b))

    # Each retrieval should align more with the correct filler than the wrong one.
    assert float(sim_fn(fwd_a, atom_a)) > float(sim_fn(fwd_a, atom_b)), name
    assert float(sim_fn(fwd_b, atom_b)) > float(sim_fn(fwd_b, atom_a)), name
    assert float(sim_fn(swp_a, atom_b)) > float(sim_fn(swp_a, atom_a)), name
    assert float(sim_fn(swp_b, atom_a)) > float(sim_fn(swp_b, atom_b)), name


@pytest.mark.parametrize("name,model_cls,sim_fn", MODELS_AND_SIM)
def test_systematicity_bind_commutativity_under_swap(name: str, model_cls, sim_fn) -> None:
    """``bind(a, b) ≈ bind(b, a)`` — order does not change the bound vector.

    This is the operator-level systematicity fact: the bind operator
    treats its two arguments symmetrically, so swapping operand order
    is invisible to downstream consumers. Holds exactly for BSC and MAP
    (commutative bind) and approximately for HRR (circular convolution
    is commutative on the unit sphere).
    """
    model = model_cls.create(dimensions=DIMS)
    keys = jax.random.split(jax.random.PRNGKey(SEED + 1), 2)
    a = model.random(keys[0], (DIMS,))
    b = model.random(keys[1], (DIMS,))

    ab = model.bind(a, b)
    ba = model.bind(b, a)

    # For BSC/MAP this is exact (commutative bind); for HRR it is also
    # exact in the FFT domain.
    sim = float(sim_fn(ab, ba))
    assert sim > 0.99, f"{name}: bind not commutative (sim={sim})"


# ----------------------------------------------------------------------
# Productivity (Fodor & Pylyshyn 1988): finitely many primitives generate
# infinitely many structures, at fixed dimension d. The closure of bind
# under R^d × R^d → R^d is the technical fact.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name,model_cls,sim_fn", MODELS_AND_SIM)
@pytest.mark.parametrize("depth", [2, 4, 8])
def test_productivity_nested_binding_round_trip(name: str, model_cls, sim_fn, depth: int) -> None:
    """Depth-N nested ``bind`` returns a vector of the same dimension and
    composes back to the original under sequential unbinding.

    Constructs ``c = bind(... bind(bind(a_1, a_2), a_3) ..., a_N)`` and
    verifies (a) the result has shape ``(d,)`` (closure at fixed
    dimension), and (b) sequentially unbinding by ``inverse(a_N), ...,
    inverse(a_3), inverse(a_2)`` recovers something more similar to
    ``a_1`` than to any unrelated atom.
    """
    model = model_cls.create(dimensions=DIMS)
    atoms = _make_atoms(model, jax.random.PRNGKey(SEED + depth), depth + 1)

    # Build c = bind(... bind(bind(a_0, a_1), a_2) ..., a_{depth-1})
    c = atoms[0]
    for i in range(1, depth):
        c = model.bind(c, atoms[i])

    # Closure: still in R^d.
    assert c.shape == (DIMS,), f"{name}@depth={depth}: closure broken"

    # Sequential unbind back to a_0.
    recovered = c
    for i in range(depth - 1, 0, -1):
        recovered = model.bind(recovered, model.inverse(atoms[i]))

    # The recovered vector should align more with a_0 than with the
    # control atom atoms[depth] (which was never bound in).
    sim_target = float(sim_fn(recovered, atoms[0]))
    sim_control = float(sim_fn(recovered, atoms[depth]))
    assert sim_target > sim_control, (
        f"{name}@depth={depth}: target sim {sim_target:.3f} not greater "
        f"than control sim {sim_control:.3f}"
    )


@pytest.mark.parametrize("name,model_cls,sim_fn", MODELS_AND_SIM)
def test_productivity_closure_at_fixed_dimension(name: str, model_cls, sim_fn) -> None:
    """Bind closes under (R^d, R^d) → R^d for every supported model."""
    model = model_cls.create(dimensions=DIMS)
    keys = jax.random.split(jax.random.PRNGKey(SEED + 99), 5)
    atoms = [model.random(k, (DIMS,)) for k in keys]

    bound = atoms[0]
    for atom in atoms[1:]:
        bound = model.bind(bound, atom)
        assert bound.shape == (DIMS,), f"{name}: bind expanded the dimension"
