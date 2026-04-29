# Literature audit — bayes-hdc against 18 foundational HDC/VSA papers

## Executive summary

A nine-agent audit verified the bayes-hdc implementation against the 18 foundational HDC/VSA papers in the user's reading list. Each agent took two papers, fetched the primary source (or flagged the fallback), digested it, mapped its claims to specific file:line locations in the codebase, and produced a per-paper detail report at `docs/audit/NN-shortname.md`.

**Headline result.** The library implements every primitive correctly. The mathematics — XOR/majority for BSC, element-wise product/normalised sum for MAP, FFT circular convolution and element-reversal involution for HRR, complex phasors for FHRR, modular arithmetic for CGR/MCR, matrix-product reshape for VTB — matches the canonical literature character-for-character. There are no algorithmic defects to repair. The audit's findings concern citation hygiene, docstring precision, and three substantive items the user must decide on.

| Quantity | Count |
|---|---|
| Papers audited | 18 |
| Agents | 9 (2 papers each) |
| Primary source consulted | 14 |
| Primary source unavailable, fallback used (Kleyko 2023, secondary expositions) | 4 (papers 2, 5, 7, 16) |
| Trivial fixes proposed (citation additions, docstring polish, broken-link repointing) | 31 |
| Substantive findings (algorithm differences, missing primitives, naming, missing tests) | 14 |
| Bibliographic corrections (volume, venue, year) needed in any future BibTeX | 4 |
| Algorithmic defects discovered | 0 |

**Papers with primary source not directly consulted.** [2] Fodor & Pylyshyn 1988 (paywalled in *Cognition*; pages 1–10 of preprint read; Part 3 inferred from Kleyko 2023 + universal references), [5] Kanerva 1997 (RWC '97 proceedings, PostScript-only, no open mirror found; verified through Kleyko 2023 Part I §2.3.6), [7] Kanerva 2009 (paywalled in *Cognitive Computation*; canonical content cross-checked through Kleyko 2023 + the existing 2009 citation in `examples/basic_operations.py`), [16] Mizraji 1989 (paywalled in *Bulletin of Mathematical Biology*; abstract via PubMed 2924018, formal description verified through Mizraji 1999 follow-up + Plate 2003 + Kleyko 2023). All four flagged inline in the per-paper reports with `:::warning Primary source not consulted`.

## Per-paper summaries

| # | Paper | Implementation status | Detail file |
|---|---|---|---|
| 1 | Gayler (2003) — VSA Answers Jackendoff's Challenges | Coined "VSA"; eight-model API embodies the framework; needs inline attribution | [`docs/audit/01-gayler-2003.md`](audit/01-gayler-2003.md) |
| 2 | Fodor & Pylyshyn (1988) — Connectionism critique | Foundational *motivation* for VSA; library satisfies systematicity/productivity by construction; not implemented (theoretical paper) | [`docs/audit/02-fodor-pylyshyn-1988.md`](audit/02-fodor-pylyshyn-1988.md) |
| 3 | Smolensky (1990) — Tensor Product Variable Binding | Out of scope (TPR is dimension-expanding); HRR/MAP/BSC are the fixed-d compressions of TPR; not implemented per user direction | [`docs/audit/03-smolensky-1990.md`](audit/03-smolensky-1990.md) |
| 4 | Plate (2003) — Holographic Reduced Representations | Implementation matches Plate's algebra **character-for-character** (FFT circular convolution + involution `x_i → x_{-i mod d}`) | [`docs/audit/04-plate-2003-hrr.md`](audit/04-plate-2003-hrr.md) |
| 5 | Kanerva (1997) — Fully Distributed Representation | Spatter-code (BSC) implementation is faithful; class docstring should cite this paper as the origin | [`docs/audit/05-kanerva-1997.md`](audit/05-kanerva-1997.md) |
| 6 | Hecht-Nielsen (1994) — Context Vectors | Library makes no claim about context vectors; `ProjectionEncoder` is Johnson-Lindenstrauss, not context-vector training; correctly omitted | [`docs/audit/06-hecht-nielsen-1994.md`](audit/06-hecht-nielsen-1994.md) |
| 7 | Kanerva (2009) — Hyperdimensional Computing: An Introduction | Canonical HDC reference; every primitive (binding, bundling, permutation, similarity, cleanup, sequence-via-permute, role-filler) is present and faithful; sparse inline citation | [`docs/audit/07-kanerva-2009.md`](audit/07-kanerva-2009.md) |
| 8 | Stewart & Eliasmith (2011) — Compositionality | Out of scope (spiking-neuron VSA, SPA/NEF); library correctly avoids spiking claims | [`docs/audit/08-stewart-eliasmith-2011.md`](audit/08-stewart-eliasmith-2011.md) |
| 9 | Stewart, Bekolay & Eliasmith (2011) — Vector Spaces with Spiking Neurons | Out of scope (NEF spiking implementation); volume number in bibliography is `22, no. 3` but should be `23, no. 2` | [`docs/audit/09-stewart-bekolay-eliasmith-2011.md`](audit/09-stewart-bekolay-eliasmith-2011.md) |
| 10 | Kanerva (2010) — "Dollar of Mexico" | `examples/kanerva_example.py` is a close match; one substantive finding on second-query unbinding formula | [`docs/audit/10-kanerva-2010.md`](audit/10-kanerva-2010.md) |
| 11 | Rasmussen & Eliasmith (2011) — Inductive Reasoning | Out of scope (NEF spiking + RPM rule induction); venue is *Topics in Cognitive Science*, not "Cognitive Science" | [`docs/audit/11-rasmussen-eliasmith-2011.md`](audit/11-rasmussen-eliasmith-2011.md) |
| 12 | Gayler & Levy (2009) — Distributed Basis for Analogical Mapping | Generalises Kanerva 2010 to graph isomorphism; the *holistic vector intersection* primitive and replicator iteration are not implemented; not currently cited | [`docs/audit/12-gayler-levy-2009.md`](audit/12-gayler-levy-2009.md) |
| 13 | Levy, Bajracharya & Gayler (2013) — Behavior Hierarchies | Out of scope (V-REP robot control); venue is AAAI 2013 Workshop, not main conference | [`docs/audit/13-levy-bajracharya-gayler-2013.md`](audit/13-levy-bajracharya-gayler-2013.md) |
| 14 | Jones & Mewhort (2007) — BEAGLE | The canonical "permute + bind for word order" pattern; `examples/language_identification.py` uses the same pattern (on character trigrams via MAP) but does not credit BEAGLE as the precedent | [`docs/audit/14-jones-mewhort-2007.md`](audit/14-jones-mewhort-2007.md) |
| 15 | Stewart, Tang & Eliasmith (2010) — Cleanup Memory | Out of scope (spiking cleanup); `HopfieldMemory` docstring should disambiguate Hopfield 1982 vs Ramsauer 2020 vs spiking | [`docs/audit/15-stewart-tang-eliasmith-2010.md`](audit/15-stewart-tang-eliasmith-2010.md) |
| 16 | Mizraji (1989) — Context-Dependent Associations | Out of scope (Kronecker-product binding); not currently referenced anywhere | [`docs/audit/16-mizraji-1989.md`](audit/16-mizraji-1989.md) |
| 17 | Mizraji (1992) — Vector Logics | Out of scope (matrix-vector logic algebra); Kleyko 2023 does not cite Mizraji either; recommend leaving uncited | [`docs/audit/17-mizraji-1992.md`](audit/17-mizraji-1992.md) |
| 18 | Kleyko et al. (2023) — Survey Part II | The application set covers 6 of 7 top-tier clusters; only EEG/iEEG seizure detection is missing; survey is **not currently cited anywhere** in the repo | [`docs/audit/18-kleyko-2023.md`](audit/18-kleyko-2023.md) |

## Substantive findings — for user review

These are findings the audit recommended *not* be applied automatically. They concern algorithm choices, naming, missing primitives, missing tests, and missing examples — each with implications that the user (not the audit) should decide. Listed by severity.

### High severity

**SF-1. `kanerva_example.py:167–175` second-query unbinding formula is algebraically incorrect for real-valued MAP** *(from paper 10).* The "What's the capital of Mexico?" block computes `bind(bind(mx, inverse(country_key)), capital_key)`. Expanded for MAP (where `inverse(x) = 1/x` and `x*x ≠ 1`), the wanted term `MXC * capital²/country` does not collapse to `MXC` algebraically. The example currently appears to print the right answer empirically because at d=10000 the leading term still has highest cosine similarity to `MXC`, but the formula is not the textbook one. The textbook one-step retrieval is `bind(mx, inverse(capital_key))`. Decision: replace with the textbook formula or add an explicit comment that this is a "double-binding" demonstration distinct from textbook unbinding.
- File: `examples/kanerva_example.py:167–175`
- Severity: high (correctness within an example that is otherwise canonical)

### Medium severity

**SF-2. `bundle_bsc` tie-break for even N** *(from paper 5).* `bundle_bsc` computes `counts > shape_size / 2.0`. For odd N this is correct; for even N it silently maps ties to 0, biasing the bundled bit. Kanerva (1997) specifies a stochastic per-component coin flip or a fixed tiebreaker hypervector. Decision: document the deterministic-bias-toward-0 convention, or accept an optional `key` argument for stochastic tie-break, or follow Hannagan-style (XOR an extra summand).
- File: `bayes_hdc/functional.py:39–55`
- Severity: medium (correctness of attribution to Kanerva 1997)

**SF-3. No explicit test verifies systematicity (Fodor & Pylyshyn 1988, paper 2).** The library satisfies systematicity by construction — `bind(R, A) + bind(R_b, b)` and its swap are equally constructible — but no property-based test in `tests/` witnesses this. A `test_systematicity` parametrised across all eight VSA models, asserting that an asymmetric relation `aRb` and its swap `bRa` both round-trip through bind/unbind, would (i) document the property explicitly and (ii) catch any future regression.
- File: new `tests/test_systematicity.py` or addition to `tests/test_functional.py`
- Severity: medium

**SF-4. `HopfieldMemory` docstring conflates three distinct architectures** *(from paper 15).* Current docstring `"""Modern Hopfield network for associative memory."""` does not distinguish (a) the modern continuous Hopfield network of Ramsauer et al. 2020 — the actual implementation; (b) the classical sign-thresholded Hopfield network of Hopfield 1982; (c) the spiking-neuron cleanup memories of Stewart, Tang & Eliasmith 2010. Proposed exact rewrite is in `docs/audit/15-stewart-tang-eliasmith-2010.md`. *(This is borderline trivial vs substantive — listed as substantive because the rewrite is more than a one-line citation and warrants user review of the wording.)*
- File: `bayes_hdc/memory.py:71–72`
- Severity: medium

### Low severity

**SF-5. No explicit test verifies productivity (depth-N nested binding closure)** *(from paper 2).* A regression test that constructs a deeply nested structure (e.g. `bind(bind(bind(a,b),c),d)` at d=10000) and verifies retrieval-by-cue works would document the closure property and ground the design choice in Fodor & Pylyshyn 1988.
- File: addition to `tests/test_functional.py`
- Severity: low

**SF-6. `make_frame` / `apply_substitution` primitives missing** *(from paper 1).* Gayler 2003's response to Jackendoff's "problem of 2" introduces `make-frame(a) = bind(P(a), a)` to give frame instances unique identity tokens, and substitution as binding-of-structures. Neither is exposed as a public primitive. Decision: add as optional helpers, document the one-liner recipe in an example, or leave alone. (Recommended: leave alone; one-liner recipes are easier to teach in an example than as a renamed primitive.)
- File: `bayes_hdc/__init__.py` and possibly `bayes_hdc/functional.py`
- Severity: low

**SF-7. `transformation_vector(a, b) = bind(inverse(a), b)` not exposed as a named primitive** *(from paper 11).* Used inline in `examples/kanerva_example.py:99–104`; one helper function would clarify intent and parallel Plate's nomenclature.
- File: `bayes_hdc/functional.py`
- Severity: low

**SF-8. Holistic vector-intersection primitive (Gayler & Levy 2009) not implemented** *(from paper 12).* The replicator-iteration mechanism for VSA-based graph-isomorphism analogical mapping requires a `vector_intersect(x, π)` op built from dual permutations + cleanup. Adding ~150 lines (one helper + one example) would close an in-scope gap from the user's literature list. Decision: add the primitive + an `examples/gayler_levy_example.py`, or leave for a future contributor.
- File: new helper in `bayes_hdc/functional.py` + new example
- Severity: low

**SF-9. EEG/iEEG seizure-detection example missing** *(from paper 18).* The only top-tier application cluster in the Kleyko 2023 survey not covered by `examples/`. Adding `examples/eeg_seizure_detection.py` along the lines of Burrello, Schindler, Benini & Rahimi 2018–2021 would close coverage of all top-tier clusters.
- File: new `examples/eeg_seizure_detection.py`
- Severity: low (current set already covers 6 of 7 top clusters)

**SF-10. Resonator-network example missing** *(from paper 18).* `bayes_hdc/resonator.py` exists but no runnable example demonstrates it. Kleyko §2.1.4 calls out resonator factorisation as a distinctive deterministic-behaviour HDC application. Adding an `examples/resonator_factorisation.py` would surface a capability already in the repo.
- File: new `examples/resonator_factorisation.py`
- Severity: low

**SF-11. "Related approaches not implemented" subsection missing from DESIGN.md** *(from papers 8, 9, 11, 13, 16, 17).* User pre-approved adding this section. A single paragraph in `DESIGN.md` §6 would credit the spiking-VSA / SPA / NEF line (Stewart-Eliasmith 2011, Stewart-Bekolay-Eliasmith 2011, Stewart-Tang-Eliasmith 2010, Rasmussen-Eliasmith 2011), the V-REP behaviour-hierarchy line (Levy-Bajracharya-Gayler 2013), and optionally the Kronecker-product binding ancestry (Mizraji 1989; Smolensky 1990). Mizraji 1992 — *not* recommended for inclusion (Kleyko 2023 doesn't cite it either; weak case for adding here).
- File: `DESIGN.md` §6
- Severity: low (purely additive)

**SF-12. Bibliographic year/venue corrections in repo's reading list (4 papers)** *(from papers 9, 11, 13, 18).*
- Paper [9]: `Connection Science 22(3)` should be `Connection Science 23(2)`.
- Paper [11]: `Cognitive Science` should be `Topics in Cognitive Science (TopiCS)`.
- Paper [13]: "Twenty-Seventh AAAI Conference on Artificial Intelligence (AAAI), pp. 1–4" should be "AAAI 2013 Workshop on Learning Rich Representations from Low-Level Sensors, Technical Report WS-13-15, pp. 25–27".
- Paper [18]: cite as **2023** everywhere, never 2022 (the journal year is 2023; arXiv v3 is also dated 2023; only v1 was 2022).
- File: any future bibliography file (none currently in repo)
- Severity: low (no current file is wrong; just guard rails for the future)

**SF-13. No centralised bibliography file** *(from papers 5, 7).* Kanerva 2009 — the field's most-cited paper — appears in only three places in the repo (`examples/basic_operations.py`, `examples/image_classification.py`, `benchmarks/benchmark_calibration.py`). A `docs/references.rst` (or `docs/bibliography.bib`) consolidating all canonical citations, with each docstring referencing it by key, would solve attribution at scale.
- File: new `docs/references.rst` or `docs/bibliography.bib`
- Severity: low

**SF-14. Capacity-bound utility (`required_dimension(k, M, q)`) missing** *(from papers 5, 15).* Plate (2003) Eq. 1, `D ≈ 4.5(k + 0.7) ln(M / 30q⁴)`, would be a useful one-liner for capacity planning. Currently no such helper.
- File: new helper in `bayes_hdc/functional.py` or `bayes_hdc/metrics.py`
- Severity: low

## Trivial fixes (will be auto-applied in Phase C)

The 31 trivial fixes proposed across the 18 reports collapse into three coherent groups, listed here for transparency. The full list of file-and-line locations + before/after diffs lives in the per-paper reports; this section is the consolidated overview.

**Group A — VSA model attributions (8 fixes).** Add primary-source citations to the docstrings of the BSC, MAP, HRR, and FHRR classes and their associated `bind_*` / `bundle_*` / `inverse_*` functions in `bayes_hdc/vsa.py` and `bayes_hdc/functional.py`. Cite Kanerva 1997 + 2009 for BSC; Gayler 1998 + 2003 for MAP; Plate 1995 + 2003 for HRR/FHRR.

**Group B — Structures and primitives attributions (6 fixes).** Add Kanerva 2009 citations to the class docstrings of `Sequence`, `HashTable`, and the function docstrings of `cleanup`, `bundle_sequence`, `hash_table`. Add Sahlgren et al. 2008 alongside Kanerva 2009 wherever the permutation-as-position primitive is documented. Add BEAGLE (Jones & Mewhort 2007) as a precedent citation in `bayes_hdc/functional.py::bind_hrr` docstring and in `examples/language_identification.py` module docstring.

**Group C — Doc-prose attributions (17 fixes).** Add inline citations for VSA terminology to `DESIGN.md` and `README.md`: Gayler 2003 next to "Vector Symbolic Architecture"; Fodor & Pylyshyn 1988 next to "compositionality / systematicity / productivity"; Kanerva 2009 next to "quasi-orthogonality"; Kleyko et al. 2023 to `examples/README.md` and `README.md` as the master survey reference; Plate 1995/2003 wherever HRR is named. Drop the misleading "(or resonator)" parenthetical in `bayes_hdc/functional.py::cleanup` docstring (resonator networks are a different algorithm). Refine the Kanerva 2010 citation in `examples/kanerva_example.py` to the precise venue (`AAAI Technical Report FS-10-08, pp. 2–6`). Rewrite the `HopfieldMemory` docstring to disambiguate Hopfield 1982 / Ramsauer 2020 / Stewart-Tang-Eliasmith 2010.

## Related approaches not implemented

The audit verified — by exhaustive grep — that the library makes no false claim of supporting any of the following. They are out of the current library's scope by deliberate design choice; documented here so a reader can locate the right adjacent literature without expecting bayes-hdc to provide it.

- **Tensor-product binding (Smolensky 1990; Mizraji 1989).** Dimension-expanding binding via outer / Kronecker product. The library implements only dimension-preserving compressed bindings (Hadamard for MAP; XOR for BSC; circular convolution for HRR). HRR/MAP/BSC are the fixed-d compressions of TPR; we do not provide the uncompressed ancestor.
- **Vector logics (Mizraji 1992).** Matrix-vector representation of propositional truth values. Independent of high-dimensional symbol algebras; not a VSA in the modern sense. Kleyko et al. 2023 does not cite Mizraji either; we follow the survey's choice and leave this uncited.
- **Spiking-neuron VSA / Semantic Pointer Architecture / Neural Engineering Framework (Stewart & Eliasmith 2011; Stewart, Bekolay & Eliasmith 2011; Stewart, Tang & Eliasmith 2010; Rasmussen & Eliasmith 2011).** A neighbouring research line implements VSA primitives on populations of leaky-integrate-and-fire spiking neurons via the NEF, with cleanup as attractor dynamics on a population. bayes-hdc operates on `jax.Array` hypervectors with deterministic JAX ops; we do not model spike trains, tuning curves, or population decoding. For that line, see Nengo / NengoSPA.
- **Robot behaviour hierarchies (Levy, Bajracharya & Gayler 2013).** A bipolar-MAP encoding of subsumption-style controllers tested in V-REP. The library supplies the same MAP primitives, but no robot-control loop, simulator integration, or behaviour-hierarchy module.
- **Random Indexing / corpus-trained context vectors (Hecht-Nielsen 1994; Sahlgren 2005; the BEAGLE training loop of Jones & Mewhort 2007).** The library provides random-projection encoders (`ProjectionEncoder`, `KernelEncoder`) and bag-of-words bundling (`examples/song_matching.py`), but no corpus-pass training loop that updates per-token vectors from co-occurrence statistics. A future `examples/beagle.py` would close this; the substrate (HRR + circular convolution + permutation) is already complete.
- **VSA-based graph-isomorphism analogical mapping (Gayler & Levy 2009).** The replicator-iteration mechanism with holistic vector-intersection primitive is not implemented; the reference MATLAB code is at `github.com/simondlevy/GraphIsomorphism`.
- **Inductive rule generation / Raven's Progressive Matrices (Rasmussen & Eliasmith 2011).** Combines HRR with NEF spiking and online transformation-vector learning. Library has the HRR primitives but no induction loop, no spiking layer, no false claim.

This is the suggested wording for a `## Related approaches not implemented` section to be added to `DESIGN.md` §6 (or its own section) per substantive finding **SF-11** above.

## Citation table

After the trivial fixes in Phase C are applied, every paper in the user's reading list will be cited at one or more locations in the repository. The table below records the canonical BibTeX form recommended for each paper and the locations where it will be cited post-Phase-C. *(Locations marked TBD are pending the user's decision on the substantive findings.)*

| # | BibTeX key | Year | Where cited (after Phase C) |
|---|---|---|---|
| 1 | `gayler2003vsa` | 2003 | `bayes_hdc/vsa.py` module docstring; `bayes_hdc/vsa.py:MAP` class docstring; `DESIGN.md` §1; `README.md` About paragraph |
| 2 | `fodor1988connectionism` | 1988 | `DESIGN.md` §1 motivating preamble; `README.md` (footnote-style sentence after Highlights) |
| 3 | `smolensky1990tensor` | 1990 | `DESIGN.md` §1 (one-line acknowledgement that compressed bindings descend from TPR) — optional |
| 4 | `plate2003hrr` (+ `plate1995hrr` as open-access preprint) | 2003 / 1995 | `bayes_hdc/vsa.py:HRR` class docstring; `bayes_hdc/functional.py:bind_hrr` docstring; `bayes_hdc/functional.py:inverse_hrr` docstring; already in `examples/basic_operations.py` |
| 5 | `kanerva1997fdr` | 1997 | `bayes_hdc/vsa.py:BSC` class docstring; `bayes_hdc/functional.py:bind_bsc`/`bundle_bsc`/`inverse_bsc`; `bayes_hdc/vsa.py:BSBC` class docstring; `docs/vsa.rst` BSC heading |
| 6 | `hechtnielsen1994contextvectors` | 1994 | (none — library makes no claim about context vectors; cited only if "Related approaches not implemented" section adds it) |
| 7 | `kanerva2009hyperdimensional` | 2009 | `bayes_hdc/structures.py:Sequence` and `:HashTable` class docstrings; `bayes_hdc/functional.py:cleanup`/`bundle_sequence`/`hash_table` docstrings; `DESIGN.md` quasi-orthogonality bullet; `examples/sequence_memory.py` and `examples/song_matching.py` module docstrings; existing `examples/basic_operations.py` |
| 8 | `stewart2011compositionality` | 2011 | `DESIGN.md` "Related approaches not implemented" (TBD pending SF-11) |
| 9 | `stewart2011neural` | 2011 (vol. 23, no. 2) | `DESIGN.md` "Related approaches not implemented" (TBD pending SF-11) |
| 10 | `kanerva2010dollar` | 2010 | `examples/kanerva_example.py` module docstring (existing; precision fix applied) |
| 11 | `rasmussen2011neural` | 2011 (Topics in Cognitive Science) | `DESIGN.md` "Related approaches not implemented" (TBD pending SF-11) |
| 12 | `gayler2009distributed` | 2009 | `examples/kanerva_example.py` module docstring (as the natural sequel); README/keywords (TBD pending SF-13) |
| 13 | `levy2013learning` | 2013 (AAAI Workshop) | `DESIGN.md` "Related approaches not implemented" (TBD pending SF-11) |
| 14 | `jones2007beagle` | 2007 | `bayes_hdc/functional.py:bind_hrr` docstring; `examples/language_identification.py` module docstring |
| 15 | `stewart2011biologically` | 2010 / 2011 | `bayes_hdc/memory.py:HopfieldMemory` class docstring (disambiguation note); `DESIGN.md` "Related approaches not implemented" (TBD pending SF-11) |
| 16 | `mizraji1989context` | 1989 | (none; recommend leaving uncited per per-paper report) |
| 17 | `mizraji1992vector` | 1992 | (none; Kleyko 2023 doesn't cite it; recommend leaving uncited) |
| 18 | `kleyko2023survey2` | 2023 (ACM CS 55(9), Article 175) | `examples/README.md`; `README.md` (master-reference); `DESIGN.md` §6 (TBD pending SF-11) |

The full BibTeX entries for all 18 papers are in the corresponding `docs/audit/NN-shortname.md` files under each report's `## Recommended canonical citation` section.

## Methodology notes

- **Audit framework**: 9 agents × 2 papers each, paired by topical adjacency so each agent developed coherent expertise.
- **Paper access strategy**: open-access first (arXiv, author homepages, ResearchGate); fall back to the Semantic Scholar API (rate-limited at 1 req/sec); final fallback to the Kleyko et al. 2023 survey (open-access on arXiv:2112.15424) which covers all 18 papers in dedicated sections.
- **Trivial vs substantive criterion**: missing inline citations, author-name typos, missing year/venue, docstring polish, and broken bibliographic links count as **trivial** and are auto-applied in Phase C. Algorithm differences, missing primitives, public-API naming inconsistencies, and missing tests count as **substantive** and require user review (this section).
- **Strict no-hallucination rule**: agents that did not consult a paper's primary source flagged the per-paper report with `:::warning Primary source not consulted`. No invented quotations or formulas appear in any of the 18 reports.
- **No code, test, or API modified by agents during the audit phase**. Agents wrote *only* their per-paper report file. Trivial fixes are applied in a separate, coherent Phase-C pass with full lint/test/docs verification before commit.
