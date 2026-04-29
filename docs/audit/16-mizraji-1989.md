# Paper [16]: Mizraji (1989) — Context-Dependent Associations in Linear Distributed Memories

## Bibliographic

- **Author:** Eduardo Mizraji
- **Title:** "Context-dependent associations in linear distributed memories"
- **Venue:** *Bulletin of Mathematical Biology*, vol. 51, no. 2, pp. 195–205
- **Year:** 1989
- **DOI:** 10.1007/BF02458441 (Springer) / 10.1016/S0092-8240(89)80067-9 (ScienceDirect)
- **PubMed ID:** 2924018
- **Affiliation:** Sección Biofísica, Facultad de Ciencias, Universidad de la República, Montevideo, Uruguay
- **Open access:** Paywalled at Springer / ScienceDirect; full PDF not consulted directly. Audit relies on the published abstract (PubMed 2924018), the formal description in subsequent Mizraji papers (e.g. *Memories in context*, 1999, BioSystems), and standard secondary references in Plate (2003) and Kleyko et al. 2023 (arXiv:2112.15424). **Flagged as not consulted in primary form.**
- **Citation count (Semantic Scholar):** 46

## Summary (200 words)

**Claim.** Linear (matrix-vector) associative memories can be made *context-dependent* by preprocessing the input with a Kronecker (tensor) product against a context vector before the matrix-vector recall step. This two-stage architecture vastly enlarges what a linear distributed memory can represent and is offered as a biologically plausible mechanism for context modulation in cognition.

**Algorithms / equations (per the abstract and Mizraji's later expositions).** Given training pairs (key xₖ, context cₖ, response yₖ), build a memory matrix
  M = Σₖ yₖ (xₖ ⊗ cₖ)ᵀ
where ⊗ is the Kronecker product. Recall with new (x', c') is then ŷ = M (x' ⊗ c'). Network 1 computes the Kronecker preprocessing; Network 2 is the linear associator. With orthonormal x's (or c's), this reduces to a context-conditional projection: choosing a different c selects a different sub-mapping from the same matrix.

**Terminology.** Linear distributed memory; contextualization; Kronecker (tensor) preprocessing; multiplicative context; conditional feature extraction.

**Demonstrated tasks.** Three classes per the abstract: (i) conditional feature extraction from complex perceptual inputs, (ii) quasi-logical operations including XOR, (iii) context-dependent access to temporal sequences. The XOR demonstration is conceptually load-bearing — XOR is the canonical "linearly inseparable" task, and contextual Kronecker preprocessing solves it without hidden units.

**Successor citations.** Smolensky's tensor-product representations (1990) develop the same outer-product-binding idea into a full role-filler theory; Plate's HRR (1995, 2003) compresses Smolensky-style binding via circular convolution; Mizraji's own 1999 *Memories in context* paper extends the formalism. The work is regularly cited in the VSA/HDC lineage as an early matrix-vector ancestor of role-filler binding.

## Paper → code map

This paper is **out of scope** for `bayes_hdc`. The library implements *binding* via element-wise multiplication (`bind_map`, `bind_bsc`) and circular convolution (`bind_hrr`) — i.e. compressed/holographic bindings that produce a vector of the **same** dimension as the operands. Mizraji's Kronecker product produces a vector of dimension `d² ` (or matrix), which is what HRR/MAP-C are explicitly designed to *avoid*.

Verification:

```
$ grep -r -i "mizraji\|kronecker\|context-dependent" bayes_hdc/ docs/
no matches
```

Nothing in the codebase references Mizraji or Kronecker-product binding, which is correct — it is a deliberate non-goal of compressed VSAs.

| Paper construct | Codebase touch-point | Relationship |
|---|---|---|
| Kronecker-product binding x ⊗ c | None | Not implemented. `bayes_hdc.functional` provides `bind_map` (Hadamard), `bind_bsc` (XOR), `bind_hrr` (circular convolution) — all dimension-preserving compressed alternatives. Mizraji's uncompressed tensor-product is a conceptual ancestor that VSAs explicitly compress. |
| Linear associative memory M = Σ y(x ⊗ c)ᵀ | `bayes_hdc/memory.py::AttentionMemory` (line 106) is the closest cousin — a key→value linear retrieval — but uses softmax weighting, not a sum of outer products. `HopfieldMemory` (line 71) is also outer-product-style storage but autoassociative, not heteroassociative-with-context. | Conceptually adjacent; algorithmically different. No fix needed. |
| Context-dependent retrieval | Could be expressed in our `AttentionMemory` by binding the query with a context vector before retrieval — but this would be a *user-level* recipe, not a library primitive. | Out of scope. |

## Trivial fixes proposed

**None for code.** Mizraji is not referenced and should not be (the algorithm is not implemented).

**One documentation suggestion (optional):** add a short "Related approaches not implemented" subsection to `DESIGN.md` or the README listing the paper as a deliberate non-goal. Suggested text:

> **Tensor-product / Kronecker binding (Mizraji 1989; Smolensky 1990).** `bayes_hdc` implements binding via dimension-preserving compressed operators (Hadamard, XOR, circular convolution). The uncompressed Kronecker-product binding of Mizraji 1989 and Smolensky's tensor-product representations (1990) is the conceptual ancestor we explicitly compress; storing structures as full outer products grows memory as O(d²) per binding and is incompatible with the fixed-D hypervector contract.

This is a *new doc paragraph*, not a code change; flagged as recommended, not required.

## Substantive findings (for user review)

- **Mizraji 1989 should remain uncited in code.** It is a historical/conceptual citation appropriate for a literature-review section but not for any docstring, since the algorithm is not present.

- **Bibliography hygiene.** If the README's full reference list cites Mizraji [16], the citation is *background*, not *implementation* — and should be grouped with Smolensky 1990 / Plate 1995 in a "compositional structures: ancestry and compression" cluster, separate from the citations that document operators actually exported by the library.

- **Primary-source not consulted.** I was unable to retrieve the 1989 PDF directly (Bulletin of Mathematical Biology paywall, no open-access preprint located). The summary above relies on the published PubMed abstract, Mizraji's 1999 follow-up, and standard secondary expositions in Plate (2003) and Kleyko et al. (2023). For final-publication audit notes, a colleague with Springer access should verify that the matrix equation M = Σ y(x ⊗ c)ᵀ is stated in those exact terms — Mizraji's 1989 notation may differ slightly. None of the conclusions here depend on that detail.

## Recommended canonical citation

```bibtex
@article{mizraji1989context,
  title   = {Context-dependent associations in linear distributed memories},
  author  = {Mizraji, Eduardo},
  journal = {Bulletin of Mathematical Biology},
  volume  = {51},
  number  = {2},
  pages   = {195--205},
  year    = {1989},
  doi     = {10.1007/BF02458441},
  pmid    = {2924018}
}
```
