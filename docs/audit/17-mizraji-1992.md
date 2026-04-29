# Paper [17]: E. Mizraji (1992) — Vector Logics: The Matrix-Vector Representation of Logical Calculus

## Bibliographic

- **Authors:** Eduardo Mizraji
- **Title:** "Vector logics: the matrix-vector representation of logical calculus"
- **Journal:** *Fuzzy Sets and Systems*, vol. 50, no. 2, pp. 179–185, 1992
- **DOI:** `10.1016/0165-0114(92)90216-Q`
- **Affiliation:** Sección Biofísica, Facultad de Ciencias, Universidad de la República, Montevideo, Uruguay
- **Citation count (Semantic Scholar, retrieved 2026-04-28):** 62
- **Open access PDF:** Closed at the publisher; original 1992 paper not freely available. The 2008 follow-up "Vector Logic: A Natural Algebraic Representation of the Fundamental Logical Gates" (Mizraji, *J. Logic & Computation*, vol. 18, pp. 97–121) is openly hosted by the author and re-states the framework with a self-contained introduction; we used it to verify the framework's content.
- **Listed in our codebase?** No. `grep -ri "mizraji"` over `bayes_hdc/`, `examples/`, `benchmarks/`, all `*.md` and `*.rst` documentation, `CITATION.cff` and `pyproject.toml` returns zero hits. The paper is not currently a reference in this project.

## Summary (200 words)

Mizraji's "vector logics" is a *matrix–vector* encoding of two-valued (later three-valued) propositional logic. The truth values **true** and **false** are mapped to a pair of orthonormal vectors `s`, `n` ∈ ℝ^Q (Q ≥ 2). Each *monadic* logical operator (e.g. negation) is realised as a square matrix that, applied to a truth vector, returns the truth vector of the operator's output: e.g. NOT is the matrix that swaps `s` and `n`. *Dyadic* operators (AND, OR, XOR, IMPLIES, …) are realised as rectangular matrices acting on the **Kronecker product** `t₁ ⊗ t₂` of the two argument truth vectors. Classical theorems and tautologies of propositional logic are then re-derived using ordinary matrix algebra. The 2008 follow-up extends the construction to a three-valued vector logic that admits modal "possibility"/"necessity" operators as plain square matrices, and to a complex-valued √NOT (relating the framework to reversible/quantum logic gates).

This is a *truth-value algebra in vector form*, not a high-dimensional symbol algebra. There is no random-coding scheme, no `bind/bundle/permute` triple, no quasi-orthogonality, and no holographic recovery: the dimension Q is small (Mizraji works in ℝ² and ℝ³), the vectors are deterministic, and the focus is the algebra of logical operators rather than the algebra of compositional symbol structures.

## Paper → code map

The audit instruction explicitly flagged Mizraji 1992 as **out of scope** for this codebase. We confirm that designation.

| Codebase concern | Relation to Mizraji 1992 |
|---|---|
| `bayes_hdc/vsa.py` (BSC, MAP, HRR, FHRR, BSBC, CGR, MCR, VTB binding/bundling) | Independent of Mizraji's framework. None of the eight VSA models in this library is a "vector logic" in Mizraji's sense — they are high-dimensional, randomly initialised, and rely on quasi-orthogonality and approximate inverses, none of which appear in Mizraji's small-Q exact-truth-value construction. |
| `bayes_hdc/equivariance.py` (group actions and equivariance verifiers) | Independent. Mizraji 1992 does not consider any group action; logical operators are unconstrained linear maps. |
| `examples/kanerva_example.py` ("Dollar of Mexico" role-filler binding) | Independent. The closest thematic neighbour in our codebase is symbolic role-filler binding, but the underlying mechanism (random HRR/MAP HVs + circular-convolution/element-wise binding) is unrelated to Mizraji's truth-vector matrices. |
| `DESIGN.md` §1 ("the algebra") | Could optionally cite Mizraji 1992 in a "Related approaches that we deliberately do not implement" / historical-context note, alongside Plate (HRR), Kanerva (BSC/SDM), and Gallant. Currently no such note exists. |

There is **no place in the current codebase that references, depends on, or is interfered-with-by** Mizraji's vector logics. The only citation question is *whether to add one* in a related-approaches section.

## Trivial fixes proposed

None. There is no existing Mizraji citation to date-check, year-check, or de-duplicate.

## Substantive findings (for user review)

1. **Is a "Related approaches" mention warranted at all?** Mizraji is occasionally name-checked in HDC retrospective surveys as a *pre-VSA* attempt to give logical operators an algebraic vector representation, but Kleyko, Rachkovskij, Osipov & Rahimi's two-part HDC/VSA survey (2022/2023, refs [3] of Part II and the Part II PDF we read for paper [18]) does **not** cite Mizraji in either part — neither in the model overview nor in the cognitive-modelling / logic-and-inference subsection (Kleyko Part II §3.1.3.6 "General-purpose rule-based and logic-based inference with HVs"). That is the most authoritative recent landscape map of the field, and it leaves Mizraji out. So the historiographic case for adding Mizraji to our `DESIGN.md` is *weak*: he is not part of the lineage the field itself acknowledges.
2. **The user direction "out of scope" matches the technical content.** The 1992 paper does not propose a high-dimensional symbol algebra, does not contemplate random codes, does not use binding-as-circular-convolution or binding-as-elementwise-product, and does not engage with similarity-based retrieval. It is a separate research programme — algebraic representation of *logical truth*, not algebraic representation of *symbolic structure*. Marking it out-of-scope is correct.
3. **If a future revision wants to cite it,** the natural home is a sentence in `DESIGN.md` §1 along the lines of *"Earlier vector-as-logic constructions (Mizraji 1992; Plate 1995; Gallant 2013) are not VSAs in the modern sense; bayes-hdc follows the high-dimensional, random-coding lineage initiated by Plate, Kanerva, and Gayler."* This costs one citation and demarcates scope explicitly. The user should decide whether the demarcation is worth the citation budget; our recommendation is **no — leave Mizraji unmentioned**, because the surrounding survey literature (Kleyko Part II) does not cite it either.

## Recommended canonical citation

If, against the recommendation above, the project chooses to cite Mizraji 1992:

> E. Mizraji, "Vector logics: the matrix-vector representation of logical calculus," *Fuzzy Sets and Systems*, vol. 50, no. 2, pp. 179–185, 1992. doi:10.1016/0165-0114(92)90216-Q.

(Use exactly this form. Note "logics" — plural — in the title; that matches the journal record.)
