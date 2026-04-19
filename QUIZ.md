# jax-hdc self-quiz

58 questions to test your grasp of the library. Answers at the bottom.
File references point to `jax_hdc/*.py` unless noted.

Suggested use: read `SLIDES.md` first, then attempt each section without
opening source. Anything you miss, read the file and try again.

---

## A. HDC / VSA fundamentals (10)

1. What does the **bind** operation produce relative to its two inputs — similar, dissimilar, or orthogonal?
2. What does the **bundle** operation produce relative to its inputs?
3. Why does HDC typically use dimensions of 10 000 or more?
4. For a MAP hypervector of dimension *d*, what is the expected cosine similarity of two independent random vectors?
5. What is **capacity** in HDC, in one sentence?
6. Write the identity that defines the **inverse** operation.
7. Which operation encodes **order** in a sequence, and why does it work?
8. What is **cleanup**, and why is it needed?
9. Explain in one sentence why bundling many vectors eventually causes retrieval failure.
10. What is the expected similarity of random binary (BSC) vectors?

---

## B. The eight VSA models (10)

11. Which VSA model uses **XOR** as its binding operation?
12. Which model uses **circular convolution** for binding, and why is it efficient to compute?
13. What is the element type of FHRR vectors, and what is their binding operation?
14. What constraint does **VTB** impose on the dimensionality, and why?
15. What does the `q` parameter mean in **CGR** and **MCR**?
16. What is the structural difference between **BSC** and **BSBC**?
17. What is the similarity metric for MCR, in words?
18. Which two models share the same `bundle` function in `functional.py`?
19. What is the inverse of an HRR vector?
20. Which models support exact, lossless unbinding?

---

## C. Encoders — `embeddings.py` (8)

21. What is the shape of `RandomEncoder.codebook`?
22. What does `LevelEncoder` do differently for BSC vs. MAP?
23. What classical approximation does `KernelEncoder` implement?
24. How does `GraphEncoder` encode a directed edge `(u, v)`?
25. Which encoder would you use for high-dimensional image embeddings (hundreds of real-valued features)?
26. Which encoder would you use for 20 categorical features with 10 values each?
27. What does `encode_batch` do in terms of JAX primitives?
28. Name one reason `RandomEncoder` clips `indices` with `jnp.clip` before lookup.

---

## D. Classifiers — `models.py` (8)

29. What does `CentroidClassifier.fit` compute for each class?
30. How does `AdaptiveHDC.fit` differ from `CentroidClassifier.fit`?
31. What matrix equation does `RegularizedLSClassifier` solve?
32. What does `LVQClassifier` do when a prototype misclassifies a sample?
33. What does `ClusteringModel` implement, in one phrase?
34. Why do `fit` methods return a new instance instead of mutating `self`?
35. What similarity function do classifiers use when `vsa_model_name == "bsc"`?
36. Which classifier would you reach for first for a closed-form baseline?

---

## E. Memory modules — `memory.py` (5)

37. What does the `radius` parameter control in `SparseDistributedMemory`?
38. How does `HopfieldMemory.retrieve` weight stored patterns?
39. What does a larger `beta` do in `HopfieldMemory`?
40. What is the role of `temperature` in `AttentionMemory`?
41. What extra functionality does `retrieve_with_weights` offer over `retrieve`?

---

## F. Symbolic structures — `structures.py` (5)

42. How is a `Multiset.value` computed from a batch of hypervectors?
43. How does `HashTable.add(key, val)` combine them into the accumulator?
44. Why does `Sequence.append` permute before adding?
45. How does `Graph.contains_edge(u, v)` check membership?
46. Which structure uses `permute` internally for ordering?

---

## G. Metrics — `metrics.py` (6)

47. What is `bundle_snr(d, n)` approximately equal to?
48. What is the formula for `effective_dimensions(x)` (participation ratio)?
49. What does `sparsity(x, τ)` measure?
50. What does `retrieval_confidence(query, codebook)` return?
51. What would `saturation(x)` close to 1.0 indicate?
52. Why is `cosine_matrix(codebook)` useful to look at?

---

## H. Code reading (4)

53. In `vsa.py`, `BSC.bind` wraps `F.bind_bsc(x, y)`. What is `F.bind_bsc` in one line of NumPy-esque code?
54. In `functional.py`, `bind_hrr` uses `jnp.fft.fft`. Why multiply in the FFT domain rather than convolve in the time domain?
55. In `models.py`, `ClusteringModel.fit` stops iterating when what condition holds?
56. In `structures.py`, `HashTable.get(key)` returns `F.bind_map(self.value, F.inverse_map(key))`. Why does that approximately recover the value stored under `key`?

---

## I. Differentiation vs. TorchHD & roadmap (4)

57. Name two modules or operations already in `jax-hdc` that are not standard in TorchHD.
58. What single feature in the v0.3 roadmap is genuinely novel relative to every public HDC library today?

---

## Answer key

**A. Fundamentals**

1. Dissimilar to both.
2. Similar to all inputs.
3. Random high-dim vectors are quasi-orthogonal and noise-robust; the curse of dimensionality becomes a *feature* (concentration of measure).
4. ≈ 0 (quasi-orthogonal for large *d*).
5. The maximum number of items that can be bundled or bound into a single hypervector while still being reliably retrievable.
6. `bind(bind(x, y), inverse(y)) ≈ x`.
7. `permute` — cyclic shift makes `permute(x, i) ≠ permute(x, j)` for `i ≠ j`, so position is encoded by which shift you apply.
8. Nearest-neighbour lookup in a codebook; unbinding is lossy so the recovered hypervector is noisy and must be snapped back to a clean symbol.
9. Each added vector contributes independent noise to every component; the signal-to-noise ratio falls as √(d/(n-1)) and eventually the target stops being the argmax.
10. 0.5 (random bits match half the time).

**B. VSA models**

11. BSC (and BSBC inherits it).
12. HRR. Circular convolution equals element-wise multiplication in the Fourier domain, so it's O(d log d) via FFT.
13. `complex64` (points on the unit circle), bound by element-wise complex multiplication.
14. VTB requires `dimensions` to be a perfect square because it reshapes the vector to a √d × √d matrix and binds by matrix multiplication.
15. The modulus of the cyclic group: CGR vectors have components in ℤ_q, and MCR uses q discrete phase levels.
16. BSBC is block-sparse: `dimensions` is split into blocks of `block_size`, each block has exactly `k_active` ones. BSC has no such constraint.
17. Real part of the normalised phasor inner product: treat integers as phases on the unit circle and take the mean inner product.
18. HRR and VTB both reuse `bundle_map` (normalised sum).
19. Reverse-flip: `[x[0], x[d-1], x[d-2], ..., x[1]]` — i.e. first element fixed, rest reversed.
20. BSC and BSBC (XOR is self-inverse, no information loss). MAP / HRR / FHRR / VTB are lossy due to normalisation or pseudoinverse. CGR / MCR are exact in integer arithmetic.

**C. Encoders**

21. `(num_features, num_values, dimensions)`.
22. For MAP / HRR / FHRR, it linearly interpolates between the two nearest level hypervectors. For BSC, it picks the lower or upper level based on whether the interpolation weight is < 0.5.
23. Random Fourier Features — approximates the RBF kernel `k(x,y) = exp(-γ‖x-y‖²)`.
24. `bind_map(node_hvs[u], permute(node_hvs[v]))` — permutation on one endpoint makes the edge direction-sensitive.
25. `ProjectionEncoder` (dense random projection) or `KernelEncoder` (if you want RBF similarity preserved).
26. `RandomEncoder` with `num_features=20, num_values=10`.
27. `jax.vmap(self.encode)` — the single-sample `encode` is automatically vectorised over the leading batch axis.
28. To avoid out-of-bounds codebook lookups at trace time; JAX requires statically-shaped indexing.

**D. Classifiers**

29. The (normalised) sum of training hypervectors belonging to that class — one prototype per class.
30. Starts from the centroids, then iteratively nudges prototypes toward misclassified samples over multiple epochs.
31. `(XᵀX + λI) W = XᵀY` where `X` is the training hypervector matrix and `Y` is one-hot labels.
32. Moves the wrongly-chosen prototype *away* from the sample; if prediction was correct it moves *toward* it.
33. k-means clustering in hypervector space, using cosine-similarity assignment and norm-sum centroid updates.
34. Dataclasses are registered as JAX pytrees, so they are treated as immutable; returning a new instance keeps traces clean and lets the object cross `jit` / `vmap` boundaries.
35. `hamming_similarity`; for all other models it's `cosine_similarity`.
36. `RegularizedLSClassifier` — single linear solve, no hyperparameters beyond regularisation.

**E. Memory**

37. The cosine-similarity radius around each location within which a write / read "hits" — larger radius means more locations participate.
38. Softmax over similarities with inverse temperature `beta`; retrieve is a weighted sum of stored patterns.
39. Sharpens retrieval toward the single most-similar pattern (approaches nearest-neighbour as β → ∞).
40. Divides the pre-softmax attention scores — larger temperature = flatter attention distribution.
41. Returns both the retrieved value *and* the attention weight vector so you can inspect which memory entries contributed.

**F. Structures**

42. `jnp.sum(vectors, axis=0)` — the straight sum (no normalisation). `size` tracks how many were added.
43. `self.value + bind_map(key, val)` — accumulates the bound pair.
44. Permutation shifts the stored bundle so the newest element sits at shift 0; position i ends up at shift `size-1-i`, which is what `get(i)` reverses.
45. Recomputes the edge hypervector from `u`, `v` and compares via `dot_similarity(edge, self.value) / dimensions`; high similarity ⇒ probably contained.
46. `Sequence` — each append rotates the running bundle by one position.

**G. Metrics**

47. `√(d / (n-1))` — the expected signal-to-noise ratio.
48. `PR(x) = (Σ xᵢ²)² / Σ xᵢ⁴`.
49. Fraction of components with `|xᵢ| < τ` (near-zero), treating τ as the zero threshold.
50. The gap `best_sim − second_best_sim` when the query is compared against every codebook row — a small gap flags ambiguity.
51. Representation is fully committed to ±1 — useful signal that a bipolar encoding is saturated.
52. Off-diagonal entries should be near 0; if they aren't, the codebook is not quasi-orthogonal and capacity will degrade.

**H. Code reading**

53. `jnp.logical_xor(x, y)`.
54. Circular convolution in the spatial domain equals pointwise multiplication in the Fourier domain. FFT + pointwise-mul + inverse FFT is `O(d log d)` instead of `O(d²)`.
55. `jnp.allclose(stacked_centroids, centroids, atol=1e-6)` — the centroids moved less than 1e-6 since the previous iteration.
56. `HashTable.value ≈ Σᵢ bind(kᵢ, vᵢ)`. Binding with `inverse(key)` distributes: `bind(Σᵢ bind(kᵢ, vᵢ), inverse(key)) = Σᵢ bind(kᵢ, vᵢ, inverse(key))`. The term where `kᵢ = key` simplifies to `vᵢ`; the other terms are pseudo-random noise that cancel out — so `cleanup` against a value codebook recovers the stored value.

**I. Differentiation & roadmap**

57. `metrics.py` (capacity, SNR, participation ratio, retrieval-confidence APIs) and the `resonator` skeleton in `functional.py` — neither ships with TorchHD. Tversky / Jaccard similarities and fractional-power binding are also library-level primitives we expose that TorchHD does not.
58. The full factorisation / resonator-network toolkit in v0.3 — no public HDC library ships accelerated, sparse, noise-tolerant factorisers with convergence diagnostics as a first-class API. TorchHD has none of this; even `hd-computing.com/software` lists zero libraries where it is the headline feature.
