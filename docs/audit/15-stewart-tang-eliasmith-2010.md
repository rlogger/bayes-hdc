# Paper [15]: Stewart, Tang & Eliasmith (2010/2011) — A Biologically Realistic Cleanup Memory

## Bibliographic

- **Authors:** Terrence C. Stewart, Yichuan Tang, Chris Eliasmith
- **Title:** "A Biologically Realistic Cleanup Memory: Autoassociation in Spiking Neurons"
- **Venue:** *Cognitive Systems Research*, vol. 12, pp. 84–92
- **Year:** Bibliography lists 2010; the published volume is dated 2011 (the journal page lists year 2011, DBLP `journals/cogsr/StewartTE11`). The reference in the README/CITATION.cff matches the 2010 cite line.
- **DOI:** 10.1016/j.cogsys.2010.06.006
- **Affiliation:** Centre for Theoretical Neuroscience, University of Waterloo
- **Open access:** Author preprint at <https://compneuro.uwaterloo.ca/files/Stewart.BioCleanup.pdf> (consulted)
- **Citation count (Semantic Scholar):** 56

## Summary (200 words)

**Claim.** Cleanup (autoassociative recognition of a noisy, post-binding hypervector) can be implemented as a single feed-forward population of spiking LIF neurons, scaling **linearly in the number of stored symbols M** in neuron count and clean-up latency of 5–10 ms — the first such result for Vector Symbolic Architectures (VSAs).

**Algorithms / equations.** Vectors x are encoded into spike trains via Neural Engineering Framework (NEF) tuning curves: each neuron i has preferred direction φ̃_i and current J_i = α_i φ̃_i · x + J^bias (Eq. 2). Decoding uses linearly-optimal vectors φ from a Γ-matrix pseudoinverse (Eq. 4). Connection weights between populations are derived as ω_ij = α_j φ̃_j W φ_i (Eq. 6). The cleanup memory itself is a *single* hidden layer where each "middle" neuron's preferred direction is set to one of the stored clean vectors and J^bias is tuned slightly negative so the unit only fires when its dot-product with input exceeds ~0.2; output weights then project the firing pattern back to the clean vector. Authors compare against (i) least-squares linear associator (Hinton & Anderson 1989), (ii) MLP cleanup, (iii) ideal Plate (2003) cleanup — the threshold-and-project model is the only one that scales to M ≈ 100,000 with D ≈ 1000.

**Terminology.** Autoassociative memory; cleanup memory; Holographic Reduced Representation (HRR); Neural Engineering Framework (NEF); preferred direction vectors; LIF spiking neurons; chunk size k.

**Demonstrated tasks.** (1) Cleanup of HRR-bound role/filler structures (e.g. `chase⊗verb + dog⊗subj + cat⊗obj`); (2) capacity sweeps over (D, M, k) up to D=1000, M=100,000, k=8 (Figs 5, 6); (3) temporal dynamics — five noisy queries cleaned at 50 ms each (Fig 7).

**Successor citations.** Cited in subsequent Compneuro Waterloo work (Crawford et al. 2016 "Biologically Plausible, Human-Scale Knowledge Representation"), in NEF/SPA architecture papers (Eliasmith 2013 *How to Build a Brain*), and in Kleyko et al. 2023 VSA survey (arXiv:2112.15424) as the canonical neural cleanup reference.

## Paper → code map

This paper is **out of scope** for `bayes_hdc` (the library is non-spiking JAX; no LIF neurons, no NEF tuning curves). Nothing in the codebase claims to implement Stewart-Tang-Eliasmith. The relevance is **conceptual**: the paper is the canonical "cleanup memory" reference cited in HDC/VSA literature.

| Paper construct | Codebase touch-point | Relationship |
|---|---|---|
| Generic *cleanup* operation (find argmax-similar stored vector for a noisy query) | `bayes_hdc/functional.py::cleanup` (line 174) | **Same operation, different substrate.** Stewart et al. implement cleanup in spikes; we implement it as a vmapped argmax over a similarity function (cosine by default). Our docstring at line 180–193 is correctly framed as a generic non-spiking cleanup; it does not claim biological plausibility, and so does not need to cite Stewart et al. as the implementation source. |
| Threshold-and-project autoassociator (each "middle" neuron is a stored prototype, fires only when dot-product > θ) | `bayes_hdc/memory.py::HopfieldMemory` (line 71) | **Architecturally similar but distinct.** Stewart's middle layer = stored prototype directions with hard sub-threshold cutoff, single feed-forward pass. `HopfieldMemory.retrieve` uses a *softmax* over cosine similarities — i.e. the modern continuous Hopfield network of Ramsauer et al. 2020 (one-step attention). Both differ from the original Hopfield 1982 sign-update network. |
| Capacity scaling argument (D = 4.5(k + 0.7) ln(M / 30q⁴), Eq. 1, from Plate 2003) | None | The library does not surface a capacity-bound utility; this is not flagged as a gap, just noted. |

The cleanup CONCEPT is central; the spiking IMPLEMENTATION is not what we provide. The library is correct to expose `functional.cleanup` as substrate-agnostic.

## Trivial fixes proposed

1. **`HopfieldMemory` docstring is ambiguous (line 72).** Current text — `"""Modern Hopfield network for associative memory."""` — names the right architecture but does not disambiguate it from (a) the classical Hopfield 1982 sign-thresholded recurrent network, nor (b) the spiking-neuron cleanup memories of Stewart-Tang-Eliasmith 2010 (which are also colloquially called "Hopfield-like" in the VSA literature). Phase 1 audit flagged this as a conflation risk. **Proposed exact rewrite:**

   ```python
   class HopfieldMemory:
       """Modern continuous Hopfield network (Ramsauer et al. 2020) for
       associative memory.

       This is the one-step softmax-attention formulation, distinct from
       the classical sign-thresholded Hopfield network (Hopfield 1982)
       and from the spiking-neuron cleanup memories of Stewart, Tang &
       Eliasmith (2010). Retrieval is a single feed-forward softmax over
       cosine similarities to stored patterns; no recurrent settling.
       """
   ```

   This both (i) correctly attributes the algorithm and (ii) heads off the readerly assumption that the class implements Hopfield's 1982 dynamics. The `retrieve` method at line 94 already does softmax-over-similarities, so the rewrite matches the implementation.

2. **`functional.cleanup` docstring (line 180).** Currently writes "Cleanup (or resonator) is used to retrieve…" — the parenthetical "or resonator" risks confusion with the Resonator Networks of Frady et al. 2020, which are a different (factorisation-by-iterative-cleanup) algorithm. *Optional* trivial fix: drop the "(or resonator)" clause and keep the rest. (Listing this only because the audit asked for trivial fixes; the user can decline if they prefer the broader idiom.)

## Substantive findings (for user review)

- **No code change recommended for the spiking aspect.** `bayes_hdc` is explicitly a JAX-on-accelerators library; reproducing Stewart-Tang-Eliasmith 2010 would require an LIF neuron model and the NEF tuning-curve infrastructure, which is well out of scope. A "Related approaches not implemented" subsection in the README (or in DESIGN.md) should mention spiking-neuron cleanup memories (Stewart-Tang-Eliasmith 2010; the Nengo / NEF tradition) as a deliberate non-goal.

- **Reference framing.** When the bibliography cites this paper in the context of the cleanup operation, the citation is essentially decorative — we use the *concept* of cleanup but not the algorithm. A short footnote in the README ("we do not implement the spiking variant of Stewart-Tang-Eliasmith 2010; `functional.cleanup` is a generic argmax-over-similarity") would prevent reviewers from expecting NEF-style biological plausibility.

- **Capacity formula.** Plate (2003) Eq. 1 — `D ≈ 4.5(k + 0.7) ln(M / 30q⁴)` — quoted in this paper, could be a useful one-liner utility (`bayes_hdc.functional.required_dimension(k, M, q)`) if the user wants a capacity-planning helper. Flagged as a *new feature* candidate, not an audit fix.

## Recommended canonical citation

```bibtex
@article{stewart2011biologically,
  title   = {A biologically realistic cleanup memory: Autoassociation in spiking neurons},
  author  = {Stewart, Terrence C. and Tang, Yichuan and Eliasmith, Chris},
  journal = {Cognitive Systems Research},
  volume  = {12},
  number  = {2},
  pages   = {84--92},
  year    = {2011},
  doi     = {10.1016/j.cogsys.2010.06.006}
}
```

(The README/bibliography currently dates this as 2010, which matches the journal-issue cover date some indices use; the published volume is 2011 per Elsevier and DBLP. Either is defensible — the user may want to standardise on 2011 to match the DOI, but this is a *bibliographic* preference, not a substantive fix.)
