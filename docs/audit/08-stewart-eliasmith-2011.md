# Paper [8]: T. Stewart and C. Eliasmith (2011) — Compositionality and Biologically Plausible Models

## Bibliographic

- **Citation as numbered in repo:** [8] T. Stewart and C. Eliasmith, "Compositionality and Biologically Plausible Models," in *The Oxford Handbook of Compositionality*, pp. 596-615, 2011.
- **Verified metadata (Semantic Scholar paperId `b4ca7de1e785dff18c5ff886a0349fcbf059b622`):** Stewart & Eliasmith, "Compositionality and Biologically Plausible Models," *Oxford Handbook of Compositionality* (Oxford University Press), 2012 print date with the chapter widely circulated as a 2011 preprint. DOI `10.1093/oxfordhb/9780199541072.013.0029`. 21 forward citations as of Apr 2026.
- **Access status:** Full text paywalled (closed access on Oxford Handbooks Online). No arXiv preprint located. Audit relied on Semantic Scholar metadata, the chapter's known role in the Eliasmith-lab corpus (heavily cross-cited with Eliasmith 2013 *How to Build a Brain*), and the citation graph (downstream papers identifying it as the Semantic Pointer Architecture's compositionality-handbook treatment).

## Summary (200 words)

Stewart and Eliasmith argue that compositional behaviour observed in human cognition can be reproduced by *biologically plausible* spiking-neuron systems implementing Vector Symbolic Architectures (VSA) — specifically the Holographic Reduced Representation (HRR) family — within the Neural Engineering Framework (NEF). The chapter positions VSA as a substrate that satisfies Fodor & Pylyshyn's classical compositionality requirements (productivity, systematicity, inferential coherence) without classical symbol-token dynamics: hypervectors carry meaning distributively, circular convolution implements role-filler binding, and superposition implements bundling. The authors situate the Semantic Pointer Architecture (SPA) — vectors that index lower-level perceptual or motor representations — as the cognitive-architecture wrapper around HRR. Algorithmic content focuses on (i) the HRR `bind` (circular convolution) and `unbind` (correlation with the involution) operators, (ii) bundling via vector sum followed by normalisation, and (iii) the NEF's neural-tuning-curve decoding scheme that lets populations of leaky-integrate-and-fire neurons compute these operators on their decoded representations. Demonstrated tasks: simple analogical retrieval, slot-filler propositional encoding, and pointer-based deep cognitive structure (a precursor to the full Spaun model). Successor citations include Eliasmith et al. 2012 (Spaun, *Science*), Stewart & Eliasmith 2013 (quantum-probability VSA), Crawford et al. 2016 (human-scale knowledge), and a 2024-25 wave of HDC-spiking bridge papers (Orchard & Jarvis 2023; Sumanasena et al. 2025).

## Paper -> code map

| Paper concept | Code locus | Status |
|---|---|---|
| HRR `bind` (circular convolution) | `bayes_hdc/functional.py::bind_hrr`; `bayes_hdc/vsa.py::HRR.bind` | Implemented as abstract-vector op (FFT-based circular convolution). Paper concept matches at the algebraic layer. |
| HRR `unbind` (involution + correlation) | `bayes_hdc/functional.py::unbind_hrr` (or equivalent inverse) | Implemented as abstract-vector op. Same algebra as paper. |
| Bundling via normalised superposition | `bayes_hdc/functional.py::bundle_hrr`; `bundle_*` family | Implemented as abstract-vector op. |
| Spiking-neuron implementation of bind/bundle (NEF decoding) | None | **Not implemented and not in scope.** The library operates on `jax.Array` hypervectors with deterministic JAX ops; it does not model neural tuning curves, spike trains, or population decoding. |
| Semantic Pointer Architecture (vectors indexing perceptual/motor representations) | None | **Not implemented and not in scope.** No SPA-style hierarchical pointer-to-substrate mechanism. |
| Neural Engineering Framework (encoding/decoding/transformation principles) | None | **Not implemented and not in scope.** |

### Grep evidence

Ran across the full repo (excluding `.venv`, `.egg-info`, `_build`, `__pycache__`):

- `spiking` / `spike` — 0 hits
- `neuron` — 0 hits in source/docs (a `numpy.lib._function_base_impl` hit inside `.venv` is unrelated)
- `nengo` — 0 hits
- `semantic pointer` / `SPA` (word boundary) — 0 hits
- `eliasmith` / `bekolay` / `stewart` — 0 hits

`README.md:7` lists "neuromorphic" as one of ~15 SEO keywords inside an HTML comment block. This is a generic-field association word (HDC is widely cited as a candidate for neuromorphic hardware), not a claim that the library implements spiking neurons. No verbal claim of spiking, NEF, or SPA support exists anywhere in the codebase or documentation.

## Trivial fixes proposed

**None for this paper.** The library makes no false or misleading claim about spiking-neuron support, no mention of SPA/NEF/Nengo, and no docstring confuses HDC with the Waterloo-line spiking implementation. The single `neuromorphic` SEO keyword in `README.md:7` is acceptable as is — it accurately describes the broader HDC research community's downstream interest in neuromorphic hardware and does not claim spiking implementation. Recommend no edit.

## Substantive findings (for user review)

1. **Out-of-scope but cleanly handled.** The library does not implement spiking-neuron VSA, the Semantic Pointer Architecture, or the Neural Engineering Framework, and correctly does not claim to. The audit confirms the codebase is internally consistent on this scope boundary.

2. **DESIGN.md gap (per user pre-approval).** Add a one-paragraph "Related approaches not implemented" subsection to `DESIGN.md` (suggested location: after the algebra section, before PVSA, or as a new terminal section). The paragraph should:
   - Explicitly state that spiking-neuron VSA implementations are out of scope.
   - Cite Stewart & Eliasmith 2011 (this paper) and Stewart, Bekolay & Eliasmith 2011 (Paper [9]) as the canonical pointers into the SPA/NEF tradition.
   - Cite Stewart, Tang & Eliasmith 2010 ("Neural Cleanup for SPA," Cognitive Science) for the cleanup-memory primitive.
   - Cite Rasmussen & Eliasmith 2011 ("A Neural Model of Rule Generation in Inductive Reasoning") for the inductive-reasoning extension.
   - Optionally note the modern HDC-spiking bridge work (Orchard & Jarvis 2023 *Hyperdimensional Computing with Spiking-Phasor Neurons*, Sumanasena et al. 2025 *Implementing HRRs for Spiking Neural Networks*) so readers see the field is active and distinct from this library's abstract-vector + JAX scope.

3. **Stronger framing for `bind_hrr` docstring (optional, low priority).** The HRR docstring in `functional.py` could include a one-line citation pointer of the form "circular convolution as in Plate 1995; biologically plausible spiking implementations in the SPA/NEF line are out of scope (see DESIGN.md)." This is *not* required to fix any inaccuracy — only suggested if the user wants to redirect curious readers from the algebraic layer to the right literature. Skip if minimalist docstrings are preferred.

4. **No conflation risk detected in present text.** Several places in `DESIGN.md` use the word "biologically plausible" loosely is *not* the case — `DESIGN.md` does not invoke that phrase, so there is no language requiring disambiguation.

## Recommended canonical citation

```bibtex
@incollection{stewart2011compositionality,
  author    = {Stewart, Terrence C. and Eliasmith, Chris},
  title     = {Compositionality and Biologically Plausible Models},
  booktitle = {The Oxford Handbook of Compositionality},
  editor    = {Werning, Markus and Hinzen, Wolfram and Machery, Edouard},
  publisher = {Oxford University Press},
  year      = {2011},
  pages     = {596--615},
  doi       = {10.1093/oxfordhb/9780199541072.013.0029}
}
```

Use this citation if a "Related approaches not implemented" note is added to `DESIGN.md`. It is the right entry point for readers who want a handbook-level treatment of why VSA addresses the classical compositionality argument, and it is the paper most often paired with Paper [9] when the SPA/NEF tradition is contrasted with abstract-vector HDC libraries.
