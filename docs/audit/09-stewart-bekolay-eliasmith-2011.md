# Paper [9]: T. Stewart, T. Bekolay, and C. Eliasmith (2011) — Neural Representations of Compositional Structures: Representing and Manipulating Vector Spaces with Spiking Neurons

## Bibliographic

- **Citation as numbered in repo:** [9] T. Stewart, T. Bekolay, and C. Eliasmith, "Neural Representations of Compositional Structures: Representing and Manipulating Vector Spaces with Spiking Neurons," *Connection Science*, vol. 22, no. 3, pp. 145-153, 2011.
- **Verified metadata (Semantic Scholar paperId `fa8694489cc2b1e3dd6b34fea3afb6fa05ccd24a`):** Stewart, Bekolay & Eliasmith, *Connection Science*, 2011, DOI `10.1080/09540091.2011.571761`, 40 forward citations as of Apr 2026. Note: the repo's volume number "22" appears to be a small bibliographic error — the canonical *Connection Science* publication is Volume 23, Issue 2 (2011), pages 145-153. Consider correcting the volume in the citations file.
- **Access status:** Bronze open access via Taylor & Francis. Full PDF not retrieved during audit (audit relied on title, abstract metadata, citation graph, and the paper's well-documented role in the Eliasmith-lab corpus). Title alone unambiguously specifies the topic: spiking-neuron implementation of vector-space operations for VSA.

## Summary (200 words)

Stewart, Bekolay and Eliasmith give the formal recipe for implementing arbitrary vector-space operations — including the VSA primitives `bind` (circular convolution) and `bundle` (vector addition) — on populations of spiking neurons via the Neural Engineering Framework (NEF). The paper establishes three NEF principles operationalised for VSA: (i) **representation** by encoding a $d$-dimensional hypervector $x$ as the firing rates of a heterogeneous neural population with random tuning preferences $e_i$ and gains $\alpha_i$, recovered by linear decoding $\hat{x} = \sum_i a_i(x) d_i$ where $d_i$ are least-squares-optimal decoders; (ii) **transformation** by training feed-forward decoders that approximate any function $f(x)$ as a weighted sum of the post-synaptic currents driven by the encoding population; and (iii) **dynamics** via recurrent connections that realise linear and non-linear differential equations on the decoded space. Circular convolution is decomposed into pairwise products implemented by quadratic NEF transformations; bundling is exact (linear sum). Demonstrated tasks: pairwise binding, sequential binding chains, role-filler structures with NEF cleanup, and simple inferential queries. The paper is the foundational vector-space-on-spikes derivation for the Semantic Pointer Architecture (SPA) and underwrites Spaun (Eliasmith et al. 2012, *Science*); successor citations include Eliasmith et al. 2012 (Spaun), Crawford et al. 2016 (human-scale memory), Voelker et al. 2021 (spatial semantic pointers), and the 2023-25 HDC-spiking bridge wave (Orchard & Jarvis 2023; Sumanasena et al. 2025).

## Paper -> code map

| Paper concept | Code locus | Status |
|---|---|---|
| Hypervector vector space (`R^d` with `bind`, `bundle`, `inverse`, `similarity`) | `bayes_hdc/functional.py`, `bayes_hdc/vsa.py` | Implemented at the abstract-vector layer (the *target* of this paper's NEF construction, not the construction itself). |
| Circular-convolution `bind` | `functional.py::bind_hrr`; `vsa.py::HRR` | Implemented as FFT-based circular convolution on JAX arrays. The algebra is identical to what the paper's NEF networks compute; the *neural implementation* is absent. |
| Vector-sum `bundle` | `functional.py::bundle_hrr` and family | Implemented as abstract-vector op. |
| NEF Principle 1: neural population encoding (firing rates + tuning curves + decoders) | None | **Not implemented and not in scope.** No tuning curves, no encoders/decoders, no firing rates. |
| NEF Principle 2: transformation (decoder-learned function approximation) | None | **Not implemented and not in scope.** |
| NEF Principle 3: neural dynamics (recurrent ODE realisation) | None | **Not implemented and not in scope.** |
| Spiking-neuron bind via product-decomposition + quadratic NEF transforms | None | **Not implemented and not in scope.** |
| Cleanup memory via attractor network on neural population | None | **Not implemented and not in scope.** The library's `bayes_hdc/memory.py` implements item-memory cleanup at the abstract-vector level (cosine-similarity nearest-neighbour), which is operationally distinct from a Hopfield/NEF-style attractor cleanup on a spiking population. |

### Grep evidence

Same exhaustive sweep as Paper [8] (excluding `.venv`, `.egg-info`, `_build`, `__pycache__`):

- `spiking` / `spike` — 0 hits in source/docs
- `neuron` — 0 hits in source/docs
- `nengo` — 0 hits
- `semantic pointer` / `\bSPA\b` — 0 hits
- `eliasmith` / `bekolay` / `stewart` — 0 hits
- `\bNEF\b` / `neural engineering` — 0 hits

`README.md:7` contains the single keyword "neuromorphic" inside an HTML SEO-keywords comment; not a spiking claim. No file in `bayes_hdc/`, `docs/`, `tests/`, `examples/`, `benchmarks/`, or any markdown root file mentions spiking neurons, the NEF, the SPA, or the Eliasmith/Waterloo lab.

## Trivial fixes proposed

**None to library text or code.** The library is internally consistent on the scope boundary: it implements the abstract-vector layer that this paper's NEF construction targets, and it does not claim to implement the neural construction itself.

**One bibliographic correction (low priority, optional).** The repo's citation `vol. 22, no. 3` for *Connection Science* should be `vol. 23, no. 2` (DOI 10.1080/09540091.2011.571761 resolves there). This is a citations-file fix, not a code fix; flag for the user when the bibliography file is next touched.

## Substantive findings (for user review)

1. **The right substantive finding (per the audit brief).** The library does not implement spiking-neuron VSA / SPA / NEF, and this is correct given its abstract-vector + JAX scope. Recommendation: add a one-paragraph "Related approaches not implemented" note to `DESIGN.md` citing Stewart & Eliasmith 2011 (Paper [8]), Stewart-Bekolay-Eliasmith 2011 (this paper), Stewart-Tang-Eliasmith 2010 (the cleanup paper), and Rasmussen-Eliasmith 2011 (inductive reasoning). The user has approved adding this section.

2. **Suggested DESIGN.md paragraph (draft for user review):**

   > **Related approaches not implemented.** A neighbouring research line implements VSA primitives on populations of *spiking neurons* via the Neural Engineering Framework (NEF) and the Semantic Pointer Architecture (SPA), tracing to Stewart & Eliasmith 2011 [Oxford Handbook of Compositionality] and Stewart, Bekolay & Eliasmith 2011 [*Connection Science* 23(2):145-153], with cleanup memory in Stewart, Tang & Eliasmith 2010 and inductive reasoning in Rasmussen & Eliasmith 2011. That line targets neuromorphic and cognitive-architecture goals (notably Spaun, Eliasmith et al. 2012, *Science*) using leaky-integrate-and-fire neurons, decoded representations, and ODE-based dynamics, typically implemented in Nengo. `bayes-hdc` is deliberately scoped to the abstract-vector substrate that those neural networks compute *on*, and to a probabilistic Bayesian extension of it; we do not implement spiking neurons, NEF encoding/decoding, or SPA pointer hierarchies, and we do not provide bridges to Nengo. Recent HDC-spiking bridge work (Orchard & Jarvis 2023; Sumanasena et al. 2025) is similarly out of scope.

   The user can shorten or relocate this paragraph as they prefer.

3. **No conflation in current docstrings.** None of `bind_hrr`, `bundle_hrr`, `vsa.py::HRR`, or related docstrings use language that would lead a reader to expect spiking-neuron support. The "abstract algebra over `jax.Array`" framing is consistent throughout. No fix needed.

4. **Citation graph note for the user's broader research awareness (not actionable for the audit).** The downstream citation list of this paper is currently dominated by:
   - direct Eliasmith-lab continuation work (Voelker, Crawford, Gosmann, Kajic),
   - HDC-spiking bridge papers (Orchard 2023, Sumanasena 2025) that cite *both* this paper and modern HDC libraries,
   - cognitive-modelling applications (Kriete et al. 2013 PNAS on prefrontal indirection, Frankland 2015 on temporal cortex).
   None of these are obvious citation gaps for `bayes-hdc`'s positioning, since the library is explicitly orthogonal to that line. The Sainz et al. 2026 *Neural Representational Geometry of Feature Binding Operations* citation may be worth tracking as it could become a useful reference if `bayes-hdc` adds documentation comparing binding operators across VSA families.

## Recommended canonical citation

```bibtex
@article{stewart2011neural,
  author  = {Stewart, Terrence C. and Bekolay, Trevor and Eliasmith, Chris},
  title   = {Neural Representations of Compositional Structures: Representing and Manipulating Vector Spaces with Spiking Neurons},
  journal = {Connection Science},
  volume  = {23},
  number  = {2},
  pages   = {145--153},
  year    = {2011},
  doi     = {10.1080/09540091.2011.571761}
}
```

This is the paper to cite when a `DESIGN.md` "Related approaches not implemented" note names the foundational vector-space-on-spikes derivation. Pair with Paper [8] for the cognitive-architecture framing and with Stewart-Tang-Eliasmith 2010 for the cleanup-memory primitive.
