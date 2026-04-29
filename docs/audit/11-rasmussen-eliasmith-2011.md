# Paper [11]: Rasmussen & Eliasmith (2011) — A Neural Model of Rule Generation in Inductive Reasoning

## Bibliographic

- **Authors:** Daniel Rasmussen, Chris Eliasmith.
- **Title:** A Neural Model of Rule Generation in Inductive Reasoning.
- **Venue:** *Topics in Cognitive Science*, vol. 3, no. 1, pp. 140–153, 2011.
- **DOI:** 10.1111/j.1756-8765.2010.01127.x.
- **Open access:** Wiley Online Library (BRONZE OA); author copy at compneuro.uwaterloo.ca/files/publications/rasmussen.2011.pdf.
- **Bibliographic note:** the bibliography entry calls the journal "Cognitive Science"; the actual venue is *Topics in Cognitive Science* (TopiCS), the affiliated short-form journal of the Cognitive Science Society. Page range and year are correct.

## Summary (200 words)

Rasmussen and Eliasmith implement a spiking-neuron model that *induces* the rules required to solve Raven's Advanced Progressive Matrices (RPM), in contrast to prior RPM models (Carpenter et al. 1990; Lovett et al. 2010) where rules were hand-supplied by the modeller. Cells of an RPM matrix are encoded as Holographic Reduced Representations (Plate 2003): role-filler vectors over `shape`, `number`, `color`, `orientation` and so on, composed by circular convolution `C = A ⊗ B` (eq. 1) and superposition. A *transformation vector* `T = A' ⊗ B` (eq. 2; `'` is HRR's approximate inverse) captures the rule that maps cell `A` to cell `B`. Generalising across pairs uses Neumann's average `T = (1/n) Σ A'ᵢ ⊗ Bᵢ`, realised online as the synaptic update `Tᵢ₊₁ = Tᵢ − wᵢ(Tᵢ − A'ᵢ ⊗ Bᵢ)`. Vectors are realised in spiking LIF populations via the Neural Engineering Framework (Eliasmith & Anderson 2003), with sub-modules of ~800–11 000 neurons each (input/inverse/integrator/cleanup/solution generator/checker; Fig. 4). A spiking-attractor cleanup memory (Stewart, Tang & Eliasmith 2009) recovers learned rules and yields practice-effects matching human data. The model reproduces qualitative human RPM phenomena: improvement with practice, non-deterministic responses, and individual-difference effects mapped onto vector dimensionality and a strategy threshold.

**Successor citations:** Eliasmith's *How to Build a Brain* (2013) extends the SPA approach to a unified cognitive architecture; Eliasmith et al. *Spaun* (Science 2012) embeds rule-induction-style components in a larger spiking model. The Kleyko et al. *VSA Survey* (arXiv:2112.15424) cites this work as the canonical neural-VSA RPM model.

## Paper → code map

| Paper concept | bayes-hdc primitive (if any) | Status |
|---|---|---|
| HRR circular convolution `A ⊗ B` (eq. 1) | `bayes_hdc.vsa.HRR.bind` (and `bind_gaussian` lifted form) | **Present** as a generic VSA primitive — but used here for a different purpose. |
| Approximate inverse `A'` for HRR | `HRR.inverse` (involution / circular correlation) | **Present**. |
| Transformation vector `T = A' ⊗ B` | none as a named operation | **Missing as a named primitive.** Trivially expressible (`bind(inverse(a), b)`), e.g. used unnamed inside `examples/kanerva_example.py` lines 99–104 to build the US→MX mapping, but not exposed as `transformation_vector(a, b)` or similar. |
| Averaged transformation `T = (1/n) Σ A'ᵢ ⊗ Bᵢ` for rule induction | none | **Missing.** No example or helper averages binds across example pairs to extract a general rule. |
| Online learning rule `Tᵢ₊₁ = Tᵢ − wᵢ(Tᵢ − A'ᵢ ⊗ Bᵢ)` | none | **Missing.** `AdaptiveHDC` updates centroids, not transformation vectors. |
| Cleanup memory (spiking attractor) | `bayes_hdc.memory` (Hopfield-style) and `cleanup_gaussian` | **Adjacent.** A non-spiking analytic cleanup exists; spiking-attractor implementation is out of scope. |
| Spiking neuron simulation (NEF / LIF) | none | **Out of scope.** bayes-hdc operates on continuous hypervectors and does not target spiking simulators. |
| RPM-style rule induction example | none | **Missing.** No example, benchmark, or test attempts induction-from-examples. |
| Per-cell role-filler encoding (`shape ⊗ circle + number ⊗ three + …`) | demonstrated for analogy in `examples/kanerva_example.py` | Present for *structural mapping* (one-to-one country↔country), absent for *rule induction* (general transformation across many pairs). |

The repository's only explicit nod to "analogical reasoning" is `examples/kanerva_example.py`, which implements Kanerva's "Dollar of Mexico" structural mapping, not Rasmussen-Eliasmith RPM rule induction. Searches for `induct`, `Raven`, `Rasmussen`, `Eliasmith`, `Nengo`, `spiking`, `semantic pointer` returned no false claims (`grep` over `bayes_hdc/`, `examples/`, `docs/`, top-level `*.md` is silent on all of these).

## Trivial fixes proposed

None. No text in the repository falsely claims to demonstrate inductive rule generation, RPM, semantic pointers, or spiking-neural HDC. The library's existing claims (HRR/MAP/BSC binding, Gaussian moment propagation, conformal calibration) are accurate.

Optional polish: a single bibliographic correction in any forthcoming citation list — venue is *Topics in Cognitive Science*, not "Cognitive Science". Not in scope of any current source file.

## Substantive findings (for user review)

1. **Add a "Related approaches not implemented" subsection to `DESIGN.md` § 6** ("When to reach for this library"). The current section says when to *not* reach for the library by example (ImageNet-scale deep learning, irreducible aleatoric uncertainty), but is silent on related-but-out-of-scope VSA application areas. A one-paragraph note would set expectations honestly and credit the prior art:

   > Rule induction in spiking neurons (Rasmussen & Eliasmith, *Topics in Cognitive Science* 2011) and behaviour-hierarchy learning for robot control (Levy, Bajracharya, & Gayler, AAAI Workshop on Learning Rich Representations from Low-Level Sensors, 2013) are well-known applied-VSA settings that this library does not target. The HRR / MAP primitives here can serve as building blocks for either, but spiking-neuron simulation, online robot control, and Raven-style rule induction are out of scope.

2. **Consider exposing `transformation_vector(a, b) = bind(inverse(a), b)` as a named primitive** if a future user does try to follow Rasmussen-Eliasmith's recipe. The math is one line, but a named binding clarifies intent and parallels Plate's standard nomenclature. Not urgent; flag only because the audit surfaced it.

3. **Library scope decision is consistent with stated goals.** The README architecture diagram lists EMG, activity recognition, language identification, sequence memory, and weight-space posteriors — all classification/sequence/uncertainty workloads — which are well-served by the closed-form Gaussian/Dirichlet moment propagation that is the library's distinctive contribution. Rule induction needs a *learning* loop over transformation vectors and (in the canonical model) a spiking substrate; neither is on the bayes-hdc roadmap. No action item here beyond the DESIGN.md note above.

## Recommended canonical citation

```bibtex
@article{rasmussen2011neural,
  author  = {Rasmussen, Daniel and Eliasmith, Chris},
  title   = {A Neural Model of Rule Generation in Inductive Reasoning},
  journal = {Topics in Cognitive Science},
  volume  = {3},
  number  = {1},
  pages   = {140--153},
  year    = {2011},
  doi     = {10.1111/j.1756-8765.2010.01127.x}
}
```
