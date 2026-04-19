# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Basic operations example for Bayes-HDC.

Demonstrates fundamental operations of Hyperdimensional Computing:
- Random hypervector generation
- Binding (dissimilar vector combination)
- Bundling (similar vector aggregation)
- Similarity computation
- Permutation for sequence encoding

References:
- Kanerva, P. (2009). "Hyperdimensional Computing: An Introduction to Computing
  in Distributed Representation with High-Dimensional Random Vectors."
  Cognitive Computation, 1(2), 139-159.

- Plate, T. A. (1995). "Holographic Reduced Representations."
  IEEE Transactions on Neural Networks, 6(3), 623-641.
"""

import time

import jax
import jax.numpy as jnp

from bayes_hdc import BSC, MAP
from bayes_hdc.utils import normalize


def demo_binding_and_unbinding():
    """Demonstrate binding and unbinding operations.

    Binding creates a dissimilar vector from two inputs. Unbinding with the
    inverse of one input recovers the other input, enabling information storage
    and retrieval in distributed representations.
    """
    print("=" * 60)
    print("Binding and Unbinding Demo")
    print("=" * 60)

    # Create MAP model
    model = MAP.create(dimensions=10000)
    key = jax.random.PRNGKey(42)

    # Generate random hypervectors
    key_x, key_y = jax.random.split(key)
    x = model.random(key_x, (10000,))
    y = model.random(key_y, (10000,))

    print(f"\nGenerated two random hypervectors of dimension {x.shape[0]}")

    # Check initial similarity
    sim_xy = model.similarity(x, y)
    print(f"Similarity between x and y: {sim_xy:.4f}")
    print("Expected: ~0.0 (random vectors are nearly orthogonal in high dimensions)")

    # Bind x and y
    start_time = time.perf_counter()
    bound = model.bind(x, y)
    bound.block_until_ready()
    bind_time = (time.perf_counter() - start_time) * 1000
    print(f"\nBound x and y to create new hypervector (time: {bind_time:.3f}ms)")

    # Check similarity with original vectors
    sim_bound_x = model.similarity(bound, x)
    sim_bound_y = model.similarity(bound, y)
    print(f"Similarity between bound and x: {sim_bound_x:.4f}")
    print(f"Similarity between bound and y: {sim_bound_y:.4f}")
    print("Expected: ~0.0 (bound vector is dissimilar to both inputs)")

    # Unbind using inverse
    y_inv = model.inverse(y)
    unbound = model.bind(bound, y_inv)
    unbound = normalize(unbound)

    sim_unbound_x = model.similarity(unbound, x)
    print("\nAfter unbinding (binding with inverse of y):")
    print(f"Similarity with original x: {sim_unbound_x:.4f}")
    print("Expected: >0.9 (should recover the original x with high similarity)")

    # Demonstrate commutativity of binding
    bound_reversed = model.bind(y, x)
    sim_commutative = model.similarity(bound, bound_reversed)
    print("\nCommutativity check:")
    print(f"Similarity bind(x,y) vs bind(y,x): {sim_commutative:.4f}")
    print("Expected: ~1.0 (binding is commutative for MAP model)")


def demo_bundling():
    """Demonstrate bundling operations.

    Bundling aggregates multiple hypervectors into a single vector that
    remains similar to all inputs. This is the superposition principle
    fundamental to distributed representations.
    """
    print("\n" + "=" * 60)
    print("Bundling Demo")
    print("=" * 60)

    model = MAP.create(dimensions=10000)
    key = jax.random.PRNGKey(42)

    # Generate multiple random hypervectors
    num_vectors = 5
    vectors = model.random(key, (num_vectors, 10000))

    print(f"\nGenerated {num_vectors} random hypervectors")

    # Bundle them together
    start_time = time.perf_counter()
    bundled = model.bundle(vectors, axis=0)
    bundled.block_until_ready()
    bundle_time = (time.perf_counter() - start_time) * 1000
    print(f"Bundled all vectors together (time: {bundle_time:.3f}ms)")

    # Check similarity with each input
    print("\nSimilarity of bundled vector with each input:")
    similarities = []
    for i in range(num_vectors):
        sim = model.similarity(bundled, vectors[i])
        similarities.append(sim)
        print(f"  Vector {i + 1}: {sim:.4f}")

    avg_sim = jnp.mean(jnp.array(similarities))
    print(f"\nAverage similarity: {avg_sim:.4f}")
    print("Expected: ~0.4-0.6 (bundled vector is similar to all inputs)")

    # Test capacity: bundle more vectors and observe degradation
    print("\nCapacity test (bundling varying numbers of vectors):")
    for n in [5, 10, 20, 50, 100]:
        test_vectors = model.random(jax.random.split(key)[0], (n, 10000))
        test_bundled = model.bundle(test_vectors, axis=0)
        test_sim = jnp.mean(jax.vmap(lambda v: model.similarity(test_bundled, v))(test_vectors))
        print(f"  {n:3d} vectors: avg similarity = {test_sim:.4f}")
    print("Note: Similarity decreases as more vectors are bundled (capacity limit)")


def demo_sequence_encoding():
    """Demonstrate sequence encoding using permutation.

    Permutation creates position-dependent representations, allowing sequences
    to be encoded while preserving order information.
    """
    print("\n" + "=" * 60)
    print("Sequence Encoding Demo")
    print("=" * 60)

    from bayes_hdc.functional import permute

    model = MAP.create(dimensions=10000)
    key = jax.random.PRNGKey(42)

    # Create symbols for a sequence
    keys = jax.random.split(key, 3)
    a = model.random(keys[0], (10000,))
    b = model.random(keys[1], (10000,))
    c = model.random(keys[2], (10000,))

    print("Created three symbols: A, B, C")

    # Encode sequence [A, B, C] using position-dependent permutation
    # Position 0: permute by 2, Position 1: permute by 1, Position 2: no permutation
    seq_abc = permute(a, 2) + permute(b, 1) + c
    seq_abc = normalize(seq_abc)

    # Encode different sequence [C, B, A]
    seq_cba = permute(c, 2) + permute(b, 1) + a
    seq_cba = normalize(seq_cba)

    # Encode same symbols different order [A, C, B]
    seq_acb = permute(a, 2) + permute(c, 1) + b
    seq_acb = normalize(seq_acb)

    print("\nEncoded three sequences:")
    print("  Sequence 1: [A, B, C]")
    print("  Sequence 2: [C, B, A]")
    print("  Sequence 3: [A, C, B]")

    # Check similarities between sequences
    sim_abc_cba = model.similarity(seq_abc, seq_cba)
    sim_abc_acb = model.similarity(seq_abc, seq_acb)
    print("\nSequence similarities:")
    print(f"  [A,B,C] vs [C,B,A]: {sim_abc_cba:.4f}")
    print(f"  [A,B,C] vs [A,C,B]: {sim_abc_acb:.4f}")
    print("Expected: Low similarity (different orders create different representations)")

    # Query for symbol at position 0 in seq_abc
    print("\nQuerying for symbol at position 0 in sequence [A,B,C]:")
    query = permute(seq_abc, -2)  # Unpermute position 0
    sim_a = model.similarity(query, a)
    sim_b = model.similarity(query, b)
    sim_c = model.similarity(query, c)

    print(f"  Similarity with A: {sim_a:.4f}")
    print(f"  Similarity with B: {sim_b:.4f}")
    print(f"  Similarity with C: {sim_c:.4f}")

    best_match = ["A", "B", "C"][jnp.argmax(jnp.array([sim_a, sim_b, sim_c]))]
    print(f"  Best match: {best_match} (expected: A)")

    # Query for symbol at position 1
    print("\nQuerying for symbol at position 1 in sequence [A,B,C]:")
    query_pos1 = permute(seq_abc, -1)
    sim_a_pos1 = model.similarity(query_pos1, a)
    sim_b_pos1 = model.similarity(query_pos1, b)
    sim_c_pos1 = model.similarity(query_pos1, c)

    print(f"  Similarity with A: {sim_a_pos1:.4f}")
    print(f"  Similarity with B: {sim_b_pos1:.4f}")
    print(f"  Similarity with C: {sim_c_pos1:.4f}")

    best_match_pos1 = ["A", "B", "C"][jnp.argmax(jnp.array([sim_a_pos1, sim_b_pos1, sim_c_pos1]))]
    print(f"  Best match: {best_match_pos1} (expected: B)")


def demo_bsc_vs_map():
    """Compare BSC and MAP models.

    BSC uses binary vectors with XOR binding, while MAP uses real-valued vectors
    with element-wise multiplication. Each has different trade-offs.
    """
    print("\n" + "=" * 60)
    print("Comparing BSC vs MAP Models")
    print("=" * 60)

    key = jax.random.PRNGKey(42)
    num_ops = 1000

    # BSC model (binary)
    print("\nBSC (Binary Spatter Codes):")
    bsc = BSC.create(dimensions=10000)
    x_bsc = bsc.random(key, (10000,))
    y_bsc = bsc.random(jax.random.split(key)[0], (10000,))

    print(f"  Vector dtype: {x_bsc.dtype}")
    print(f"  Memory per vector: {x_bsc.nbytes / 1024:.2f} KB")
    print("  Binding operation: XOR (reversible, lossless)")

    # Benchmark binding
    start_time = time.perf_counter()
    for _ in range(num_ops):
        _ = bsc.bind(x_bsc, y_bsc)
    jax.block_until_ready(bsc.bind(x_bsc, y_bsc))
    bsc_time = (time.perf_counter() - start_time) * 1000 / num_ops

    bound_bsc = bsc.bind(x_bsc, y_bsc)
    sim_bsc = bsc.similarity(bound_bsc, x_bsc)
    print(f"  Similarity after binding: {sim_bsc:.4f}")
    print(f"  Binding time (avg): {bsc_time:.4f}ms")

    # MAP model (real-valued)
    print("\nMAP (Multiply-Add-Permute):")
    map_model = MAP.create(dimensions=10000)
    x_map = map_model.random(key, (10000,))
    y_map = map_model.random(jax.random.split(key)[0], (10000,))

    print(f"  Vector dtype: {x_map.dtype}")
    print(f"  Memory per vector: {x_map.nbytes / 1024:.2f} KB")
    print("  Binding operation: Element-wise multiplication")

    # Benchmark binding
    start_time = time.perf_counter()
    for _ in range(num_ops):
        _ = map_model.bind(x_map, y_map)
    jax.block_until_ready(map_model.bind(x_map, y_map))
    map_time = (time.perf_counter() - start_time) * 1000 / num_ops

    bound_map = map_model.bind(x_map, y_map)
    sim_map = map_model.similarity(bound_map, x_map)
    print(f"  Similarity after binding: {sim_map:.4f}")
    print(f"  Binding time (avg): {map_time:.4f}ms")

    # Comparison
    print("\nComparison:")
    print(f"  Memory efficiency: BSC is {x_map.nbytes / x_bsc.nbytes:.1f}x smaller")
    print(
        f"  Speed: {'BSC' if bsc_time < map_time else 'MAP'} is "
        f"{max(bsc_time, map_time) / min(bsc_time, map_time):.1f}x faster"
    )
    print("\nTrade-offs:")
    print("  BSC: Memory-efficient, discrete operations, XOR binding")
    print("  MAP: Gradient-friendly, smooth similarity, real-valued optimization")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("Bayes-HDC: Basic Operations Examples")
    print("=" * 60)

    demo_binding_and_unbinding()
    demo_bundling()
    demo_sequence_encoding()
    demo_bsc_vs_map()

    print("\n" + "=" * 60)
    print("Examples Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
