# Paper [4]: Plate (2003) — Holographic Reduced Representations

## Bibliographic
- **Cited as (in repo bibliography):** T. A. Plate, *Holographic Reduced Representations: Distributed Representation for Cognitive Structures*, CSLI / Stanford, 2003 (book; supersedes Plate 1995 IEEE TNN paper).
- **Verified metadata (Semantic Scholar paperId `c3577312cb178cc93459bda92e37076e1fa9af88`):** T. Plate, *Holographic Reduced Representation: Distributed Representation for Cognitive Structures*, 2003. Citations ≈ 255. The book is published by CSLI Publications (Stanford), ISBN 978-1-57586-430-9. (Note: the canonical title in the published volume is the **singular** form *"Holographic Reduced Representation"*; both forms are widely used in citing literature.)
- **Free preprint of equivalent material:** T. A. Plate, "Holographic reduced representations," *IEEE Trans. Neural Networks*, 6(3):623–641, 1995. DOI `10.1109/72.377968`. Citations ≈ 720. Open PDF mirror: `redwood.berkeley.edu/wp-content/uploads/2020/08/Plate-HRR-IEEE-TransNN.pdf`.
- **Source consulted for this audit:** the IEEE TNN 1995 version (pages 623–628) — Plate's 2003 book is an expanded restatement of the same algebra; the binding/inverse equations are identical. **All quoted equations below are from Plate 1995 verbatim.**
- **`examples/basic_operations.py:18` already cites Plate 1995** by full reference, which is consistent with using the 1995 paper as a citation surrogate when the 2003 book is paywalled.

## Summary (200 words)
Plate proposes **Holographic Reduced Representations (HRRs)** as a fixed-dimensionality alternative to Smolensky's tensor product. Items are real-valued vectors in `ℝ^n` whose elements are i.i.d. with mean 0 and variance `1/n` (so expected `‖x‖ = 1`). Two operations are defined: **circular convolution** as binding, `t_j = (c ⊛ x)_j = Σ_{k=0}^{n-1} c_k · x_{(j-k) mod n}` (Plate 1995 Fig. 4, p. 625), and **circular correlation** as the (approximate) decoder. Crucially, Plate proves correlation = convolution-with-the-involution: "The correlation of c̃ and t̃ is equivalent to the convolution of t̃ with the involution of c̃. The involution of c̃ is the vector d̃ = c̃* such that **d_i = c_{-i}**, where subscripts are modulo-n. For example, if c̃ = (c_0, c_1, c_2, c_3), then **c̃* = (c_0, c_3, c_2, c_1)**" (p. 627). The binding is implementable via FFT (`F^{-1}(F(c) ⊙ F(x))`). Plate gives capacity / SNR analysis (`η_i ~ N(0,(n-1)/n²)`), demonstrates representations of pairs, sequences, stacks, and frame-like structures, and pairs the convolution memory with a clean-up associative item memory (Fig. 6). Successors: FHRR (Plate), MAP (Gayler), VTB (Gosmann–Eliasmith), Hadamard / Voicu, and modern HD-classifier work — see Kleyko et al. 2023 (paper [1]) for the lineage.

## Paper → code map

### Bind: circular convolution

**Plate (1995) p. 625, Fig. 4:**
> `t_j = (c ⊛ x)_j = Σ_{k=0}^{n-1} c_k · x_{(j-k) mod n}` ,  for `j = 0..n-1`.

Plate explicitly notes (p. 625, last paragraph) that circular convolution can be computed via the FFT.

**Code (`bayes_hdc/functional.py:211–227`, `bind_hrr`):**
```python
x_fft = jnp.fft.fft(x, axis=-1)
y_fft = jnp.fft.fft(y, axis=-1)
result_fft = x_fft * y_fft
return jnp.fft.ifft(result_fft, axis=-1).real
```

**Verdict: exact match.** The convolution theorem `(c ⊛ x) = F^{-1}(F(c) ⊙ F(x))` is the standard FFT realisation Plate himself recommends. Casting to `.real` is correct since `c` and `x` are real (the imaginary part is numerical noise). Cost is O(n log n).

### Inverse: element-reversal involution

**Plate (1995) p. 627, §II.F:**
> "The involution of c̃ is the vector d̃ = c̃* such that **d_i = c_{-i}**, where subscripts are modulo-n. For example, if c̃ = (c_0, c_1, c_2, c_3), then **c̃* = (c_0, c_3, c_2, c_1)**."

So for a vector indexed `0..n-1`:
`x_inv = (x_0, x_{n-1}, x_{n-2}, ..., x_2, x_1)`.

**Code (`bayes_hdc/functional.py:231–242`, `inverse_hrr`):**
```python
return jnp.concatenate([x[..., :1], jnp.flip(x[..., 1:], axis=-1)], axis=-1)
```

Trace on a length-4 vector `[c_0, c_1, c_2, c_3]`:
- `x[..., :1]` → `[c_0]`
- `jnp.flip(x[..., 1:])` → flip of `[c_1, c_2, c_3]` → `[c_3, c_2, c_1]`
- concatenated → `[c_0, c_3, c_2, c_1]` ✓

**Verdict: exact match to Plate's involution definition (p. 627).** This is the *approximate* inverse Plate calls `c̃*`, not the *exact* inverse `c̃^{-1}` (which exists only when no Fourier coefficient of `c` vanishes; Plate discusses this in §VIII.C of the 1995 paper). The library's docstring at line 232 ("reverse the circular convolution") is fine but slightly informal; see substantive findings below.

### Bundle: normalised sum

`bundle_hrr = bundle_map` (`bayes_hdc/functional.py:246`) — i.e., element-wise sum followed by L2 normalisation. This matches Plate's standard usage of additive trace composition (Plate 1995 p. 624, "trace composition operation `⊞`" implemented as addition for HRR; explicit normalisation onto the unit sphere is a routine engineering choice the library makes consistently across MAP/HRR/FHRR).

### Random codebook

`HRR.random` (`vsa.py:197–210`) draws `~ N(0, 1)` and L2-normalises. Plate (1995, p. 626 §II.D) gives "the elements of each vector ... independently and identically distributed with mean zero and variance `1/n`. This results in the expected Euclidean length of a vector being one." Drawing from `N(0, 1)` and then dividing by the norm yields a vector on the unit sphere whose components are *not* literally `N(0, 1/n)` but which is statistically indistinguishable from Plate's prescription for the purposes of HRR algebra (the marginal distribution becomes a scaled Beta, but for large `n` the cosine-similarity behaviour is identical, and unit-norm is what `cosine_similarity` actually wants). **Acceptable; a minor cosmetic note in substantive findings.**

### Similarity: cosine

`HRR.similarity` uses `cosine_similarity`. Plate uses dot-products on (approximately) unit vectors throughout, which is the same quantity. **Match.**

### DESIGN.md §1, §2

- Line 10: "HRR uses circular convolution" — **correct**.
- Line 18: "Distributivity: `x ⋆ (y ⊕ z) ≈ (x ⋆ y) ⊕ (x ⋆ z)` (exact in HRR, ...)" — **correct**: circular convolution is bilinear, hence exactly distributive over addition (the "≈" comes from the *normalisation* step of `bundle_hrr`, not from convolution itself; the parenthetical is therefore precisely right).
- Line 48: "Circular-convolution binding satisfies [single-argument equivariance]" — **correct**; this is verified in `tests/test_equivariance.py:97–117`.

### Tests
- `tests/test_functional.py:215–247` (`TestHRROperations`) verifies `bind ∘ inverse ≈ identity` (similarity > 0.8 at d=10000) and that `bundle_hrr` produces unit-norm vectors. Both are consistent with Plate's claimed properties.
- `tests/test_equivariance.py:97–119` verifies single-argument shift equivariance and the diagonal-shift double-shift identity for `bind_hrr`. Both follow from the convolution theorem and are claims Plate makes in the 1995 paper.

## Trivial fixes proposed

These are docstring / wording-only suggestions; the user can apply or ignore. **No code, API, or test changes.**

1. **(Trivial, docstring polish.) `bayes_hdc/functional.py:231–242`, `inverse_hrr`.** The current docstring says "the inverse reverses the order of elements (except the first)." This is correct but could optionally name the operation by its Plate-given name: *involution*, denoted `x*`, with the explicit formula `(x*)_i = x_{(-i) mod d}`. A one-line parenthetical citation `(Plate 1995, §II.F)` would make the implementation self-documenting. Not required for correctness.

2. **(Trivial, docstring polish.) `bayes_hdc/vsa.py:159–164`, `class HRR` docstring.** Says "Real-valued vectors with circular convolution binding, normalized sum bundling, cosine similarity." A one-sentence reference such as "Plate (1995, IEEE TNN; book version 2003)" would mirror what `bayes_hdc/vsa.py:215` (the FHRR class) ought to say as well.

3. **(Trivial, citation hygiene.) `examples/basic_operations.py:18`** already cites Plate 1995 — that's good. If the project bibliography prefers the 2003 book as the canonical citation (as paper [4] in the master list), consider adding the 2003 book as the *primary* citation and Plate 1995 as the *open-access preprint* in a "see also" line. This is purely a stylistic choice.

## Substantive findings (for user review)

1. **HRR algebra in this library matches Plate exactly.** Both `bind_hrr` (FFT-based circular convolution) and `inverse_hrr` (element-reversal involution `x_inv = [x_0, x_{d-1}, ..., x_1]`) are character-for-character correct against the formulas on Plate (1995) pp. 625 and 627. There is **no algebraic defect** to report.

2. **Approximate vs exact inverse — wording, not behaviour.** Plate distinguishes the *involution* `x*` (cheap, always defined, approximate inverse) from the *exact inverse* `x^{-1}` (defined when all Fourier coefficients of `x` are nonzero; computable as `F^{-1}(1 / F(x))`). The library implements only the involution. This is the standard choice in every HRR library I know of, and it's the right default — but a user who reads the docstring "Inverse via element reversal" without context might believe `bind(bind(x, y), inverse_hrr(y)) == x` exactly, when in fact it's only approximate (this is what `tests/test_functional.py:237` checks with `similarity > 0.8`). Recommendation: leave the implementation alone; consider a one-line note in the docstring clarifying "approximate inverse (involution)". *Report-only, no fix in this audit.*

3. **`HRR.random` distribution is `N(0,1)`-then-normalise rather than `N(0, 1/d)`-then-leave-be.** Both produce vectors on the unit sphere; the marginal-component distributions differ slightly, but for HRR algebra the relevant property is `‖x‖ ≈ 1`, which both satisfy. Plate's analytic capacity bounds (e.g., `η_i ~ N(0, (n-1)/n²)`) assume the i.i.d. construction; under the unit-sphere construction the same bounds hold up to vanishing O(1/d) corrections. **Acceptable for a library, worth noting in DESIGN.md if/when capacity claims are made quantitatively.**

4. **Plate 1995 vs Plate 2003.** The 2003 CSLI book is the named citation in the master bibliography but is paywalled / not openly available; the 1995 IEEE TNN paper covers the same algebra and is open-access via the Berkeley Redwood mirror. **For the audit, I verified equations against Plate 1995** — this is explicitly noted at the top of this report and is consistent with the brief's instruction to fall back to Plate 1995 with a flag.

5. **Distributivity remark in DESIGN.md is technically tighter than it advertises.** Line 18 says distributivity is "exact in HRR." This is true for *unnormalised* HRR — circular convolution is bilinear over `ℝ^d`. Once `bundle_hrr` normalises, it becomes only approximate. The current wording is fine because the parenthetical "(exact in HRR, approximate after normalisation in MAP)" implicitly applies to MAP normalisation; if a future reader complains, a one-word clarification ("exact in unnormalised HRR") would suffice. **Minor, report-only.**

## Recommended canonical citation

Primary (matches master bibliography):
```bibtex
@book{plate2003hrr,
  author    = {Plate, Tony A.},
  title     = {Holographic Reduced Representation: Distributed Representation
               for Cognitive Structures},
  publisher = {CSLI Publications},
  address   = {Stanford, CA},
  year      = {2003},
  isbn      = {978-1-57586-430-9},
  series    = {CSLI Lecture Notes}
}
```

Open-access preprint (recommended as a "see also"):
```bibtex
@article{plate1995hrr,
  author  = {Plate, Tony A.},
  title   = {Holographic Reduced Representations},
  journal = {IEEE Transactions on Neural Networks},
  volume  = {6},
  number  = {3},
  pages   = {623--641},
  year    = {1995},
  doi     = {10.1109/72.377968}
}
```

The current entry in `examples/basic_operations.py:18` ("Plate, T. A. (1995). 'Holographic Reduced Representations.' IEEE Transactions on Neural Networks, 6(3), 623-641.") is correct and may be retained verbatim.
