# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Kanerva's 'Dollar of Mexico' example.

This example implements the classic HDC demonstration from:
"What We Mean When We Say 'What's the Dollar of Mexico?':
Prototypes and Mapping in Concept Space"
by Pentti Kanerva (2010).

Paper: https://redwood.berkeley.edu/wp-content/uploads/2020/05/kanerva2010what.pdf
Citation: P. Kanerva, "What We Mean When We Say 'What's the Dollar of
          Mexico?': Prototypes and Mapping in Concept Space,"
          *Quantum Informatics for Cognitive, Social, and Semantic
          Processes: Papers from the AAAI Fall Symposium*, AAAI
          Technical Report FS-10-08, AAAI Press, 2010, pp. 2-6.

The role-filler binding mechanism itself was introduced for HDC in
Kanerva (2009) — see *Hyperdimensional Computing: An Introduction*,
Cognitive Computation 1(2): 139-159 — which is the operation this
example demonstrates. The "Dollar of Mexico" 2010 paper is the
worked-example presentation. The graph-isomorphism / replicator-
iteration generalisation of this pattern is in Gayler & Levy (2009),
*A Distributed Basis for Analogical Mapping*, ANALOGY-2009.

Demonstrates structured knowledge representation and analogical reasoning.

The example encodes country information (country name, capital, currency) as
hypervectors, then uses binding and bundling to create relational representations.
The key insight: mappings between structured representations can answer analogical
queries like "What's the dollar of Mexico?" (answer: Mexican Peso).

Concepts demonstrated:
- Role-filler binding to create structured representations (Kanerva 2009)
- Bundle to combine multiple role-filler pairs
- Inverse binding to create mappings between structures
- Similarity search over memory to answer queries
"""

import jax
import jax.numpy as jnp

from bayes_hdc import BSC
from bayes_hdc.functional import hamming_similarity


def main():
    """Run the Kanerva Dollar of Mexico example."""
    print("=" * 70)
    print("Kanerva's 'Dollar of Mexico' Example")
    print("=" * 70)

    # Initialize. Kanerva (2010) demonstrates the analogy in the Binary
    # Spatter Code (BSC) substrate, where XOR binding is self-inverse and
    # the algebra `F = X * Y, then F * X ≈ Y` is exact. We use BSC here
    # to match the paper faithfully; the same construction expressed in
    # MAP works in principle but is noisier because real-valued binding
    # is not self-inverse.
    d = 10000  # number of dimensions
    key = jax.random.PRNGKey(42)
    model = BSC.create(dimensions=d)

    print(f"\nUsing {d}-dimensional hypervectors")

    # Create role hypervectors (slots in the structure)
    print("\nCreating role hypervectors (keys)...")
    print("Roles define the structure: each country has a country, capital, and currency")
    keys_key, values_key = jax.random.split(key)
    role_keys = jax.random.split(keys_key, 3)
    country_key = model.random(role_keys[0], (d,))
    capital_key = model.random(role_keys[1], (d,))
    currency_key = model.random(role_keys[2], (d,))
    keys = jnp.stack([country_key, capital_key, currency_key])

    print("  - country (role)")
    print("  - capital (role)")
    print("  - currency (role)")

    # Create filler hypervectors for USA
    print("\nCreating filler hypervectors for United States...")
    print("Fillers are the actual values that fill the roles")
    us_keys = jax.random.split(values_key, 6)
    usa = model.random(us_keys[0], (d,))
    wdc = model.random(us_keys[1], (d,))
    usd = model.random(us_keys[2], (d,))

    print("  - USA (country filler)")
    print("  - Washington D.C. (capital filler)")
    print("  - US Dollar (currency filler)")

    # Create filler hypervectors for Mexico
    print("\nCreating filler hypervectors for Mexico...")
    mex = model.random(us_keys[3], (d,))
    mxc = model.random(us_keys[4], (d,))
    mxn = model.random(us_keys[5], (d,))

    print("  - Mexico (country filler)")
    print("  - Mexico City (capital filler)")
    print("  - Mexican Peso (currency filler)")

    # Create country representations by binding roles with fillers
    print("\nCreating country representations...")
    print("  US = bind(country, USA) + bind(capital, WDC) + bind(currency, USD)")

    us_values = jnp.stack([usa, wdc, usd])
    us_bound = jax.vmap(model.bind)(keys, us_values)
    us = model.bundle(us_bound, axis=0)

    print("  MX = bind(country, MEX) + bind(capital, MXC) + bind(currency, MXN)")

    mx_values = jnp.stack([mex, mxc, mxn])
    mx_bound = jax.vmap(model.bind)(keys, mx_values)
    mx = model.bundle(mx_bound, axis=0)

    # Create mapping from US to Mexico
    print("\nCreating mapping: US → Mexico")
    print("  Mapping = bind(US, MX)   (BSC: XOR is self-inverse, so no")
    print("                            explicit inverse is needed)")

    # In BSC, inverse(x) == x (XOR is self-inverse), so the canonical
    # Kanerva-2010 mapping `F_UM = USTATES * MEXICO` is just `bind(us, mx)`.
    # We retain the more general `bind(inverse(us), mx)` form so the same
    # code works for MAP when the model is swapped, and BSC's identity
    # inverse simplifies it back to bind(us, mx).
    us_to_mx = model.bind(model.inverse(us), mx)

    # Query: What's the dollar of Mexico?
    print("\n" + "=" * 70)
    print("Query: What's the Dollar of Mexico?")
    print("=" * 70)

    print("\nApplying mapping to US Dollar:")
    print("  Result = bind(Mapping, USD)")

    usd_of_mex = model.bind(us_to_mx, usd)

    # Create memory of all known concepts
    memory = jnp.concatenate(
        [keys, us_values, mx_values],
        axis=0,  # Role vectors  # US fillers  # Mexico fillers
    )

    memory_labels = [
        "country (role)",
        "capital (role)",
        "currency (role)",
        "USA",
        "Washington D.C.",
        "US Dollar",
        "Mexico",
        "Mexico City",
        "Mexican Peso",
    ]

    # Find most similar concept
    print("\nComputing similarity with all known concepts:")
    print("-" * 70)

    similarities = jax.vmap(lambda m: hamming_similarity(usd_of_mex, m))(memory)

    # Sort by similarity
    sorted_indices = jnp.argsort(similarities)[::-1]

    for idx in sorted_indices:
        sim = similarities[idx]
        label = memory_labels[idx]
        bar = "█" * int(sim * 50)
        print(f"  {label:20s} | {bar} {sim:.4f}")

    best_match_idx = sorted_indices[0]
    best_match = memory_labels[best_match_idx]
    best_similarity = similarities[best_match_idx]

    print("-" * 70)
    print(f"\nBest match: {best_match} (similarity: {best_similarity:.4f})")

    if best_match == "Mexican Peso":
        print("\nResult: Correct identification of Mexican Peso")
        print("  Currency mapping Mexico -> Peso analogous to USA -> Dollar")
    else:
        print(f"\nResult: Unexpected match '{best_match}' (expected 'Mexican Peso')")

    # Additional queries
    print("\n" + "=" * 70)
    print("Additional Queries")
    print("=" * 70)

    # Query: What's the capital of Mexico?
    #
    # Direct unbinding: bind the bundle with the inverse of the role-key, and
    # the matching filler (the capital) emerges as the dominant component.
    # In Kanerva-2009 / 2010 notation: given H = bind(country, MEX) +
    # bind(capital, MXC) + bind(currency, MXN), the recovery of the capital
    # is bind(H, inverse(capital)) ≈ MXC. This is the textbook one-step form;
    # the "double-binding" idiom that previously appeared here was algebraically
    # incorrect for real-valued MAP (it left an extra capital^2 / country
    # factor) even though it printed the right answer empirically at d=10000.
    print("\nQuery: What's the capital of Mexico?")
    capital_query = model.bind(mx, model.inverse(capital_key))

    capital_sims = jax.vmap(lambda m: hamming_similarity(capital_query, m))(memory)
    capital_match = jnp.argmax(capital_sims)

    print(f"Answer: {memory_labels[capital_match]} (similarity: {capital_sims[capital_match]:.4f})")

    # ------------------------------------------------------------------
    # Gayler 2003 frame and substitution recipes (one-liners).
    #
    # Gayler's response to Jackendoff's "problem of 2" — multiple instances
    # of the same type — uses a *frame* construction:
    #
    #     make_frame(a) = bind(permute(a), a)
    #
    # Each frame is unique even when its internal structure is shared.
    # His response to the "problem of variables" (productivity) treats
    # substitution as a binding-of-structures:
    #
    #     apply_substitution(sub, struct) = bind(sub, struct)
    #
    # where ``sub = transformation_vector(x, y)`` (i.e. ``bind(inverse(x), y)``)
    # encodes the rule "replace x with y". In Kanerva-2010 / BSC notation,
    # ``us_to_mx`` constructed above is exactly such a substitution: applying
    # it to ``usd`` substitutes USA→MX and yields the Mexican Peso.
    #
    # We do not expose `make_frame` or `apply_substitution` as named
    # primitives because the one-line recipes above are clearer than a
    # renamed wrapper would be. See ``bayes_hdc.transformation_vector`` if
    # you want a named helper for the substitution case.
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("Example Complete")
    print("=" * 70)
    print("\nDemonstrates structured knowledge representation and analogical reasoning")
    print("using hyperdimensional vector operations: bind, bundle, and similarity.")


if __name__ == "__main__":
    main()
