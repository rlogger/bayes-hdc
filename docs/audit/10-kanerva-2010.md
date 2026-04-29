# Paper 10: Kanerva (2010) — "What's the Dollar of Mexico?"

## Bibliographic
- **Full citation.** P. Kanerva, "What We Mean When We Say 'What's the Dollar of Mexico?': Prototypes and Mapping in Concept Space," in *Quantum Informatics for Cognitive, Social, and Semantic Processes: Papers from the AAAI Fall Symposium*, Technical Report FS-10-08, AAAI Press, 2010, pp. 2–6.
- **Author affiliation at time of writing.** Center for the Study of Language and Information, Stanford University.
- **Open-access copy.** Berkeley Redwood Center mirror — `https://redwood.berkeley.edu/wp-content/uploads/2020/05/kanerva2010what.pdf` (the URL already cited inside `examples/kanerva_example.py`).
- **Series note.** AAAI Fall Symposium, FS-10-08, not "AAAI Fall Symposium Series" generically. The full proceedings volume's editors are Bruza, Sofge, Lawless, van Rijsbergen, Klusch (commonly cited).

## Summary (200 words)
Kanerva argues that brains compute on very wide (≈10,000-bit) hyperdimensional vectors, and that the figurative/analogical use of language is naturally captured by simple algebra in such a space. He works through a Binary Spatter Code (BSC) version, where binding is element-wise XOR (`*`), bundling is the majority-rule mean, and binding is its own inverse. A record with roles `X, Y, Z` and fillers `A, B, C` is encoded holistically as `H = [(X*A)+(Y*B)+(Z*C)]`. Concept retrieval works because XOR distributes over the bundle: `X*H ≈ A`. The central worked example builds country records `USTATES = [(NAM*USA)+(CAP*WDC)+(MON*DOL)]` and `MEXICO = [(NAM*MEX)+(CAP*MXC)+(MON*PES)]`, then forms a *prototype-based* mapping vector `F_UM = USTATES * MEXICO` whose noisy expansion contains the pairs `(USA*MEX)+(WDC*MXC)+(DOL*PES)`. Applying the map to DOL gives `DOL * F_UM ≈ PES` — "the Dollar of Mexico is the Peso." Two extensions: composing maps (`F_SU * F_UM = F_SM`, Sweden→Mexico through US) and the variable-free IQ-puzzle form (`MEXICO * USTATES * DOL = PES`). Successor citations: Plate's HRR thesis (1994/2003), Smolensky's tensor products (1990), Gayler's MAP (1998), Aerts–Czachor–De Moor geometric algebra (2009), and Kanerva's own 2009 *Cognitive Computation* article are all explicitly invoked. The paper is the canonical *worked example* for analogical mapping in VSA.

## Paper → code map

| Paper notation | Code in `examples/kanerva_example.py` |
|---|---|
| `NAM, CAP, MON` (role vectors) | `country_key, capital_key, currency_key` (lines 54–56) |
| `USA, WDC, DOL`; `MEX, MXC, PES` (filler vectors) | `usa, wdc, usd`; `mex, mxc, mxn` (lines 67–79) |
| `USTATES = [(NAM*USA)+(CAP*WDC)+(MON*DOL)]` | `us = bundle(vmap(bind)(keys, us_values))` (lines 89–91) |
| `MEXICO = [(NAM*MEX)+(CAP*MXC)+(MON*PES)]` | `mx = bundle(vmap(bind)(keys, mx_values))` (lines 95–97) |
| `F_UM = USTATES * MEXICO` (BSC; XOR is self-inverse) | `us_to_mx = bind(inverse(us), mx)` (lines 103–104) — the MAP-correct generalization, since for real-valued MAP `inverse(x) = 1/x` rather than `x` itself |
| `DOL * F_UM ≈ PES` | `usd_of_mex = bind(us_to_mx, usd)` (line 114) |
| Clean-up memory of "all known concepts" | `memory = jnp.concatenate([keys, us_values, mx_values])` and cosine similarity scan (lines 117–138) |

The `HashTable.get` retrieval pattern (`bayes_hdc/structures.py:110–112`) is exactly the binding-then-unbinding move used to extract `A` from `H` in Kanerva's `X*H ≈ A` derivation:

```python
@jax.jit
def get(self, key: jax.Array) -> jax.Array:
    return F.bind_map(self.value, F.inverse_map(key))
```

For BSC, `inverse(key) = key`, so this reduces to `bind(value, key)` — Kanerva's exact formula. For MAP it generalizes to `bind(H, 1/key)` — the correct extension.

The composite encoding helper `F.hash_table` (`bayes_hdc/functional.py:487–505`) is the formal counterpart of the `H = Σ bind(k_i, v_i)` definition introduced on page 3 of the paper.

## Trivial fixes proposed
- **Citation venue precision.** The example docstring (line 13) reads "2010 AAAI Fall Symposium Series." Strictly the venue is *Quantum Informatics for Cognitive, Social, and Semantic Processes: Papers from the AAAI Fall Symposium*, Technical Report FS-10-08, page range 2–6. A one-line edit would bring this to a publication-grade citation.
- **Role-label naming.** Kanerva uses `NAM, CAP, MON`. The example uses `country_key, capital_key, currency_key`. The semantics are identical and the code's labels are arguably clearer for a modern reader, but a parenthetical "(NAM, CAP, MON in Kanerva's notation)" comment near line 53 would let a careful reader cross-reference without ambiguity.
- **Mapping-construction comment.** Line 101 reads `Mapping = bind(inverse(US), MX)`. This is the MAP-architecture analogue of Kanerva's `F_UM = USTATES * MEXICO` (which works in BSC because XOR is self-inverse). A one-line note that the inverse is required only because MAP is not self-inverse would head off reader confusion.

## Substantive findings (for user review)

### S1 — Second-query unbinding formula appears incorrect for real-valued MAP. *Medium confidence, low-medium severity.*
The "What's the capital of Mexico?" block (lines 167–175) computes:
```python
capital_query = model.bind(mx, model.inverse(country_key))
capital_query = model.bind(capital_query, capital_key)
```
Expanded for MAP (real-valued, where `inverse(x) = 1/x` and `x*x ≠ identity`):
```
mx * (1/country) * capital
  = (country*MEX) * (1/country) * capital   →  MEX * capital
  + (capital*MXC) * (1/country) * capital   →  MXC * capital² / country
  + (currency*MXN) * (1/country) * capital  →  MXN * capital * currency / country
```
None of these terms approximates `MXC` cleanly: the wanted term has an extra `capital²/country` factor, which is non-trivial for normalized-Gaussian MAP vectors. The standard one-step retrieval is `bind(mx, inverse(capital_key)) ≈ MXC`. The current double-bind formula would only collapse cleanly for bipolar MAP (vectors in `{-1, +1}` so `x*x = 1`), but `MAP.random` in `bayes_hdc/vsa.py:141–154` returns Gaussian samples, not bipolar.

**Status caveat.** The example's `print(...)` shows `Mexico City` as the answer in the user's expected output, suggesting the example currently passes empirically — likely because at d=10000 the leading `MXC * capital²/country` term still has higher cosine similarity to `MXC` than to any other memory item, *even though* it is not the clean BSC-style reading the comment implies. Worth either (a) replacing the formula with the standard `bind(mx, inverse(capital_key))`, or (b) explicitly flagging that this is a "double-binding" demonstration distinct from the textbook unbinding.

### S2 — Bundling normalisation departs from paper. *Low severity, documentation only.*
Kanerva's bundle is the unnormalized superposition `[U+V+W]` (with majority rule for binary). The code uses `model.bundle = F.bundle_map`, which is the L2-normalized sum. For the geometry of the worked example this rescaling is invisible to cosine similarity, but the paper's algebra is stated in unnormalized form, and the `Σ` in the `H = Σ bind(k_i,v_i)` rendering of `F.hash_table` (functional.py:494) is consistent with the paper while `bundle_map` is not. Worth a one-line comment that `MAP.bundle` introduces an L2 normalization that has no effect on cosine queries.

### S3 — Role of `inverse` in the mapping is more than cosmetic. *Documentation gap.*
The code path `bind(inverse(us), mx)` is the *only* place in the example where the difference between BSC and MAP genuinely matters for correctness (S1 aside). A reader who only knows BSC ("XOR is self-inverse") could miss why `inverse` is needed at all. Currently the docstring only says "Inverse binding to create mappings between structures" (line 25). Two extra sentences would close the gap.

## Recommended canonical citation
```bibtex
@inproceedings{kanerva2010dollar,
  author    = {Pentti Kanerva},
  title     = {What We Mean When We Say ``What's the Dollar of Mexico?'':
               Prototypes and Mapping in Concept Space},
  booktitle = {Quantum Informatics for Cognitive, Social, and Semantic Processes:
               Papers from the AAAI Fall Symposium},
  series    = {AAAI Technical Report FS-10-08},
  publisher = {AAAI Press},
  year      = {2010},
  pages     = {2--6},
  url       = {https://redwood.berkeley.edu/wp-content/uploads/2020/05/kanerva2010what.pdf}
}
```
