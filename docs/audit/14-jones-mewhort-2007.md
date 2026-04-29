# Paper [14]: M. N. Jones and D. J. K. Mewhort (2007) — BEAGLE: Composite Holographic Lexicon

## Bibliographic

- **Citation as numbered in repo:** [14] M. N. Jones and D. J. Mewhort, "Representing Word Meaning and Order Information in a Composite Holographic Lexicon," *Psychological Review*, vol. 114, no. 1, pp. 1-37, 2007.
- **Verified metadata (Semantic Scholar paperId `d3cf28ab36ff7f7601a55c1e832736b2473a07f0`):** Jones (University of Colorado at Boulder, then Indiana University) & Mewhort (Queen's University), *Psychological Review* 114(1), 1-37, 2007. DOI `10.1037/0033-295X.114.1.1`. PubMed PMID `17227180`. **627 forward citations** as of April 2026 — one of the most-cited cognitive-science papers using HRR / circular convolution.
- **Access status:** Closed-access via APA (`Psychological Review` is paywalled), but a preprint-style PDF was obtained via UC San Diego (`https://cseweb.ucsd.edu//~gary/PAPER-SUGGESTIONS/jones-mewhort-psych-rev-2007.pdf`); contents verified directly. Pagination, equations, and BEAGLE acronym confirmed against the original. The paper's abstract and the model description in the Kleyko 2023 *ACM Computing Surveys* HDC/VSA Part II survey (their reference [172]) match.

## Summary (200 words)

BEAGLE — *Bound Encoding of the Aggregate Language Environment* — is a holographic lexicon that learns distributed word representations from raw text by superposing two information sources into a single composite memory vector per word. Each word `i` carries an environmental vector `e_i` (random Gaussian, `μ=0`, `σ=1/sqrt(D)`, fixed across learning, `D=2048` in their simulations) and a memory vector `m_i` (initially zero, accumulated across sentences). For each sentence containing word `i`, BEAGLE forms a context vector `c_i = sum_{j != i} e_j` (Eq. 1) and an order vector `o_i` built from circular convolutions of n-gram chunks containing `i`, with a fixed placeholder vector `Φ` standing in for the focus word and noncommutative permutations distinguishing left/right neighbours (Eq. 3, λ=7 chunk size). The update rule is `m_i ← m_i + c_i + o_i` (Eq. 2). Training was on the 90,000-word TASA corpus. Demonstrated tasks: nearest-neighbour semantic structure (Table 1), order-space neighbours that recover lexical class (Table 2 — verbs cluster, locatives cluster, numerals cluster), TOEFL synonym detection, semantic priming, typicality, semantic constraint in sentence completion, and recovery of word transitions without built-in grammar rules. Successor citations: Recchia & Jones 2009; Sahlgren et al. 2008 (permutation-based BEAGLE variant); Cox et al. 2011 *Behav. Res. Methods* (orthography); Rutledge-Taylor et al. 2014 *BICA*; Kachergis et al. 2011 OrBEAGLE.

## Paper -> code map

| Paper concept | Code locus | Status |
|---|---|---|
| Random Gaussian environment vectors `e_i`, μ=0, σ=1/√D | `bayes_hdc/embeddings.py::RandomEncoder` (line 22-115) generates one random HV per token via `vsa.random(...)`. For HRR-family models this draws from a unit-norm normal distribution scaled like Plate. Used identically in `examples/song_matching.py` (line 131-138) for words and in `examples/language_identification.py` (line 205-211) for characters. | **Implemented (substrate)** at the level of "atomic random vector per token." The exact normalisation is delegated to `VSAModel.random` (per-model: `MAP`, `HRR`, etc.); BEAGLE uses HRR-with-Gaussian-elements, which `bayes_hdc.vsa.HRR` supports. |
| Context vector `c_i = sum_{j≠i} e_j` (Eq. 1) | `examples/song_matching.py::encode_song` (line 141-144): "song HV = normalised sum of word HVs." This is the Eq. 1 operation applied to a *song* rather than a *sentence*; the focus-word-exclusion (`j != i`) is not implemented because the example builds song HVs, not per-word memory updates. | **Spiritually adjacent.** The bundle-of-word-HVs operation is identical algebraically; the missing pieces are (i) per-word memory accumulation across many sentences and (ii) the focus-word exclusion. The example is a single-document context HV, not a corpus-trained per-word memory. |
| Memory update `m_i ← m_i + c_i + o_i` (Eq. 2) | None | **Not implemented.** No corpus-driven word-level memory accumulation loop exists in `bayes_hdc`. |
| Order vector via circular convolution + placeholder Φ + position permutations (Eq. 3) | `examples/language_identification.py::encode_text` (line 168-194): `tri_i = c0 * permute(c1, 1) * permute(c2, 2)`, then bundle. Uses `bind_map` (MAP elementwise multiplication) and `permute` (cyclic shift). The pattern is **permute + bind to encode position**, which is the BEAGLE family's trick — but applied to **character trigrams**, not to **word n-grams** with a Φ placeholder. | **Structurally similar, semantically different — see Substantive findings.** This is the Joshi/Halseth/Kanerva (2016) language-ID encoder, not BEAGLE. Both schemes use "permute by position, then bind, then bundle" to compress order-of-elements into a fixed-D HV, and BEAGLE is the canonical antecedent of that pattern in the HDC/VSA literature. The differences: (a) BEAGLE encodes word-level n-grams with circular convolution (HRR), not MAP element-wise binding; (b) BEAGLE uses a single fixed placeholder `Φ` for the focus word so that the convolutional product is associative-but-noncommutative under permutation, whereas the language-ID encoder permutes each trigram element by its position and binds; (c) BEAGLE uses chunk size up to λ=7, the language-ID encoder uses fixed n=3 (trigrams). |
| HRR / circular convolution binding | `bayes_hdc/functional.py::bind_hrr` (line 212-228) — FFT-based circular convolution; `bayes_hdc/vsa.py::HRR.bind` (line 180+) | **Implemented.** Available substrate; not currently wired into a BEAGLE example. |
| TASA corpus (90,000 words) | None | **Not implemented.** The repo's word-text examples use small in-file corpora: 8 pseudo-songs in `song_matching.py`, 5×20 phrases in `language_identification.py`. No corpus-scale training loop. |
| Cosine similarity over normalised composite vectors | `examples/song_matching.py:150` (`song_hvs @ song_hvs.T` after normalisation) and `examples/language_identification.py:194` (post-bundle normalisation) | **Implemented (idiom).** Same retrieval-by-cosine pattern BEAGLE uses for nearest-neighbour tables (their Table 1 / Table 2). |

### Grep evidence

Searched the full repo (excluding `.venv`, `.egg-info`, `_build`, `__pycache__`):

- `BEAGLE` / `Beagle` / `beagle` — 0 hits
- `Jones` / `Mewhort` — 0 hits
- `holographic lexicon` / `composite holographic` / `Murdock 1982` — 0 hits
- `TASA` / `aggregate language environment` — 0 hits
- `circular convolution` — 14 hits (all in `functional.py`, `vsa.py`, `equivariance.py`, `embeddings.py`); none cite BEAGLE.
- `Plate 1995` — 1 hit in `examples/basic_operations.py:18` (citing Plate for HRR, the binding op BEAGLE uses).

The library implements the substrate BEAGLE depends on (HRR, circular convolution, bundling, permutation) and the related-but-distinct character-n-gram order encoder (`language_identification.py`), but does not credit Jones & Mewhort 2007 anywhere — neither in the example that *most resembles* their order-encoding pattern (the language-ID file) nor in any `functional.py` / `vsa.py` docstring that introduces circular convolution.

## Trivial fixes proposed

1. **Add a one-line citation in `examples/language_identification.py` clarifying the BEAGLE precedent for "permute + bind to encode position."** Suggested location: after the existing Joshi/Halseth/Kanerva 2016 attribution (line 6-8 in the module docstring or line 169-179 in the `encode_text` docstring). Suggested wording (user's call on phrasing):

   > "The 'permute-by-position, bind, bundle' family of order-encoding schemes goes back to BEAGLE (Jones & Mewhort 2007, *Psychol. Rev.* 114(1):1-37), where it was applied to word n-grams with circular convolution and a fixed placeholder vector. The Joshi-Halseth-Kanerva (2016) encoder used here applies the same idea to character n-grams with MAP-style elementwise binding."

   This is a **trivial fix** because the citation is unambiguously due — BEAGLE is the canonical HDC/VSA paper for this pattern, has 627 citations, and predates the Joshi/Halseth/Kanerva 2016 application by nine years. Adding it does not change semantics; it only restores attribution.

2. **Add Jones & Mewhort 2007 to the references in `bayes_hdc/functional.py::bind_hrr` docstring (line 212-228).** Currently the docstring describes circular convolution mechanically; it cites no paper. Adding "(Plate 1995 for the HRR algebra; Jones & Mewhort 2007 for the canonical cognitive-science application to word semantics and order)" would mirror the existing citation pattern in `examples/basic_operations.py:18` and give downstream readers the right entry point. This is **trivial** because the citation is factual and uncontested.

## Substantive findings (for user review)

1. **`song_matching.py` is BEAGLE-adjacent in spirit but does not implement BEAGLE.** The song HV is a sum of word HVs, normalised — exactly BEAGLE's Eq. 1 context vector for a single sentence-equivalent. What is missing for full BEAGLE: (a) per-word memory accumulation across multiple "sentences" (the example treats songs as documents, not as training instances that update word vectors); (b) the focus-word exclusion (`j != i`); (c) order information. The example's docstring is honest about this — it explicitly disclaims learning ("Zero training. Zero backprop. Add a new song = bundle its words.") so there is no inaccurate claim. **No fix recommended.** A discretionary one-line note ("the bundle-then-cosine pattern is the same operation BEAGLE (Jones & Mewhort 2007) uses to form per-sentence context vectors before accumulating them into per-word memory — we stop at the first step") would be informative but is not required.

2. **`language_identification.py` uses a permute-and-bind order encoder that is structurally similar to BEAGLE but is *character* trigrams, not *word* n-grams, and uses MAP not HRR.** This is a NOTE, not a fix: the example credits Joshi/Halseth/Kanerva (2016) — the correct primary citation for *character-trigram language ID* — and that attribution should stay. The case for *also* mentioning BEAGLE rests on lineage: Jones & Mewhort 2007 is where the "fixed placeholder + permutations + circular-convolution-binding to encode position into a fixed-D HV" idea is established for HDC/VSA; Joshi/Halseth/Kanerva is the character-level adaptation nine years later. The Trivial-fixes section above proposes adding BEAGLE as a precedent citation without removing the Joshi citation; this finding flags it as a deliberate choice rather than a bug.

3. **`functional.py::bind_hrr` and the HRR module describe circular convolution without naming BEAGLE.** Plate 1995 is the right primary citation for the algebra (and is already cited in `basic_operations.py`), but BEAGLE is the empirically dominant downstream use of HRR in cognitive science. If the docstring philosophy is "name the algorithm + the canonical application," adding BEAGLE alongside Plate is appropriate. If the philosophy is "name the inventor only," skip. No `bind_hrr` text currently makes a misleading claim — this is purely an attribution-completeness call.

4. **A BEAGLE-style example would be a high-value addition (out of scope for this audit).** The `bayes_hdc` substrate (HRR with Gaussian elements, `bind_hrr`, `permute`, `bundle_hrr`, FFT-fast convolution) supports BEAGLE *exactly* — only a corpus-pass training loop and a placeholder vector `Φ` are missing. A 100-line `examples/beagle.py` that trains on a small text corpus and recovers Jones & Mewhort's Table 1 / Table 2 nearest-neighbour patterns would be a natural addition that (a) showcases HRR (currently underused in examples) and (b) gives the cognitive-science audience an entry point into the library. Out of scope for the present audit — flagged for the user's roadmap, not for this PR.

5. **No mention of the permutation-based BEAGLE variant (Sahlgren, Holst, Kanerva 2008; Recchia, Jones, Sahlgren, Kanerva 2010).** This permutation-only variant (no circular convolution) is what subsequent HDC papers often mean by "BEAGLE-style" because it is faster and scales better. The repo does not implement it either; if a BEAGLE example is added per finding 4, this variant is the more natural target for an HDC library and should be cited alongside Jones & Mewhort 2007 as the two-paper canon for "permutation-based order encoding for word semantics."

## Recommended canonical citation

```bibtex
@article{jones2007beagle,
  author    = {Jones, Michael N. and Mewhort, Douglas J. K.},
  title     = {Representing Word Meaning and Order Information in a
               Composite Holographic Lexicon},
  journal   = {Psychological Review},
  volume    = {114},
  number    = {1},
  pages     = {1--37},
  year      = {2007},
  doi       = {10.1037/0033-295X.114.1.1},
  pmid      = {17227180}
}
```

This is the right citation if the user adopts trivial fix 1 (BEAGLE precedent in `language_identification.py`) and/or trivial fix 2 (BEAGLE in `bind_hrr` docstring). If a future BEAGLE example is added (substantive finding 4), pair this with Sahlgren, Holst, Kanerva 2008 ("Permutations as a Means to Encode Order in Word Space") for the permutation-only variant.
