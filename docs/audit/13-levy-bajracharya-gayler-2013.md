# Paper [13]: Levy, Bajracharya, & Gayler (2013) — Learning Behavior Hierarchies via High-Dimensional Sensor Projection

## Bibliographic

- **Authors:** Simon D. Levy, Suraj Bajracharya, Ross W. Gayler.
- **Title:** Learning Behavior Hierarchies via High-Dimensional Sensor Projection.
- **Venue:** *Learning Rich Representations from Low-Level Sensors: Papers from the AAAI 2013 Workshop*, Technical Report WS-13-15, pp. 25–27 (4 pages including title block).
- **PDF:** https://cdn.aaai.org/ocs/ws/ws1006/7075-30615-1-PB.pdf
- **Bibliographic note:** the bibliography entry attributes the paper to the "Twenty-Seventh AAAI Conference on Artificial Intelligence (AAAI)" main proceedings. The paper actually appeared in the **co-located AAAI 2013 workshop track** (workshop on *Learning Rich Representations from Low-Level Sensors*), not the main conference. The page range "1–4" in the bibliography is also a relative count — actual pages in the workshop volume are 25–27.

## Summary (200 words)

Levy, Bajracharya & Gayler argue that non-trivial sensor-actuator policies can be encoded entirely in a Vector Symbolic Architecture rather than as an explicit subsumption finite-state machine (Brooks). Sensor and actuator labels become bipolar `{−1, +1}^N` random hypervectors with `N > 1000`. Binding is element-wise multiplication `⊗` (self-inverse: `X ⊗ X ⊗ Y = Y`); bundling is element-wise addition. A behaviour hierarchy is a single bundle of rule terms, each rule being a sensor-state vector bound to an action vector. For the corral-escape task (V-REP simulator, two wheels, one touch sensor, two light sensors), the controller is hand-coded as one ~3 000-element vector summing eight terms (light-seeking variants + obstacle-avoidance + cruise). At runtime the live sensor-bundle is bound element-wise into the controller, and dot-products against the actuator basis vectors recover `(±1, ±1)` wheel commands; orthogonal cross-terms collapse to noise. The work is explicitly framed as a *position paper*: the authors note "our network weights are hand-coded rather than learned" and propose evolutionary learning as future work — so despite the title, no learning is demonstrated. The contribution is the encoding scheme (a "high-dimensional sensor projection") and the proof-of-concept that a subsumption-equivalent policy fits inside one VSA bundle.

**Successor citations:** Neubert, Schubert & Protzel, *A Vector Symbolic Architecture for Robotics* (Frontiers in Neurorobotics 2019); the Kleyko et al. VSA survey (arXiv:2112.15424). Levy himself extended this thread in subsequent W&L undergraduate-research papers on VSA-controlled robots.

## Paper → code map

| Paper concept | bayes-hdc primitive (if any) | Status |
|---|---|---|
| Bipolar `{−1, +1}^N` random hypervectors | `bayes_hdc.vsa.MAP.random`, `BSC.random` | **Present.** MAP uses bipolar; BSC uses binary. |
| Element-wise multiplication binding `⊗` | `MAP.bind`, `bind_map` | **Present** (this is exactly the MAP/Gayler bipolar binding). |
| Element-wise addition bundling | `MAP.bundle` | **Present.** |
| Self-inverse `X ⊗ X ⊗ Y = Y` | property of MAP and BSC; checked in `tests/test_functional.py` | **Present.** |
| Cleanup via Hopfield-style stored vectors | `bayes_hdc.memory` (Hopfield-style associative cleanup) | **Present** as a generic primitive; not used in any robot-control example. |
| Sensor-bundle binding into a controller bundle, then dot-product against actuator basis | none | **Missing as an example.** The pattern is one bind + one dot-product, both already supported, but no example wires it up for a behaviour-policy. |
| Subsumption-style behaviour hierarchy | none | **Out of scope.** bayes-hdc has no robot-control or behaviour-hierarchy module. |
| V-REP / robotics integration | none | **Out of scope.** No simulator hooks, no real-time loop, no actuator API. |
| Online or evolutionary *learning* of the controller | none | The paper itself does not actually learn — weights are hand-coded. So there is no "learning" to compare against. |

`grep -ni -E "robot\|behavior\|behaviour\|sensorimotor\|subsumption\|Levy\|Bajracharya\|Gayler\|V-REP\|actuator"` over `bayes_hdc/`, `examples/`, `docs/`, and top-level `*.md` finds:
- `COMMUNITY.md:41` — "sensor-fusion robotics" listed as a *desired* future application example (aspirational, not a claim).
- `COMMUNITY.md:101` — "Application examples — biosignals, robotics, time-series, edge ML" as roadmap items.
- `CODE_OF_CONDUCT.md` — only the word "behaviour" used in its ordinary English sense.
- Several `# majority rule` / `# update rule` doc strings, none related to subsumption or behaviour hierarchies.

No file claims that bayes-hdc demonstrates robot control, behaviour hierarchies, or sensorimotor learning. The COMMUNITY.md mentions are correctly framed as wished-for contributions, not delivered features.

## Trivial fixes proposed

None. The repository neither claims robot-control support nor falsely cites Levy-Bajracharya-Gayler. The COMMUNITY.md "robotics" mentions are honest aspiration.

Optional bibliographic polish if/when this reference is used in a citation list: the venue is the **AAAI 2013 Workshop on Learning Rich Representations from Low-Level Sensors** (Technical Report WS-13-15), not the AAAI main conference, and pages are 25–27.

## Substantive findings (for user review)

1. **The DESIGN.md "Related approaches not implemented" paragraph proposed in the Rasmussen-Eliasmith audit (paper [11]) should also cite this paper.** The two together cover the two highest-profile applied-VSA settings the library does not target — neural rule induction and robot behaviour hierarchies — and adding both in one paragraph is cleaner than two separate notes. Suggested wording (single paragraph for `DESIGN.md` § 6, after the existing "Reach for something else when …"):

   > Two well-known applied-VSA settings sit outside this library's scope. Rasmussen & Eliasmith (*Topics in Cognitive Science* 3:140–153, 2011) build a spiking-neuron rule-induction model for Raven's Progressive Matrices using HRR + the Neural Engineering Framework; the binding and cleanup primitives here are compatible building blocks, but spiking simulation and online transformation-vector learning are not provided. Levy, Bajracharya & Gayler (AAAI 2013 Workshop on Learning Rich Representations from Low-Level Sensors, pp. 25–27) implement a VSA behaviour hierarchy for a corral-escape robot in V-REP using bipolar MAP binding; bayes-hdc supplies the same MAP primitives but no robot-control loop, simulator integration, or behaviour-hierarchy module. Reach for Nengo, NengoSPA, or a dedicated robotics-VSA codebase if either of those is the actual target.

2. **The paper is a useful template if a contributor ever does pick up "sensor-fusion robotics" from `COMMUNITY.md`'s wishlist.** The full controller is one bundle of eight bind-terms — small enough to fit inside an `examples/` script. If such a contribution lands, the Levy-Bajracharya-Gayler citation belongs in that example's docstring; until then, no citation is needed.

3. **Naming nit (no action).** The paper title says "Learning" but the body explicitly defers learning to future work. If the DESIGN.md note above is added, mirror the paper's own framing: cite it as a *behaviour-hierarchy encoding scheme* rather than a learning result, to avoid implying more than the paper claims.

## Recommended canonical citation

```bibtex
@inproceedings{levy2013learning,
  author    = {Levy, Simon D. and Bajracharya, Suraj and Gayler, Ross W.},
  title     = {Learning Behavior Hierarchies via High-Dimensional Sensor Projection},
  booktitle = {Learning Rich Representations from Low-Level Sensors:
               Papers from the AAAI 2013 Workshop},
  series    = {AAAI Technical Report WS-13-15},
  pages     = {25--27},
  year      = {2013},
  publisher = {AAAI Press},
  url       = {https://cdn.aaai.org/ocs/ws/ws1006/7075-30615-1-PB.pdf}
}
```
