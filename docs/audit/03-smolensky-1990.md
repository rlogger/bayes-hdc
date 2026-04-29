# Paper [3]: Smolensky (1990) — Tensor Product Variable Binding

## Bibliographic
- **Cited as (in repo):** P. Smolensky, "Tensor Product Variable Binding and the Representation of Symbolic Structures in Connectionist Systems," *Artificial Intelligence*, vol. 46, pp. 159–216, 1990.
- **Verified metadata (Semantic Scholar paperId `24484e1105bd28acbf0184c94ac9833511328087`):** P. Smolensky, *Artificial Intelligence*, 46:159–216, 1990. DOI `10.1016/0004-3702(90)90007-M`. DBLP `journals/ai/Smolensky90`. Citations ≈ 443. Author affiliation on title page: Department of Computer Science and Institute of Cognitive Science, University of Colorado, Boulder.
- **Citation as printed in repo is correct.** Title, journal, volume, page range, and year all match the canonical record.
- **Source consulted:** Free PDF mirror of the *Artificial Intelligence* version (Smolensky_1990_TensorProductVariableBinding.AI.pdf, lscp.net mirror); pages 159–161 (intro/abstract) and 172–175 (Definitions 2.9 and 2.10, the formal binding definition).

## Summary (200 words)
Smolensky introduces the **tensor product representation (TPR)** as a formal connectionist binding scheme for filler/role pairs. The construction is unambiguous (Definition 2.9, p. 174): a filler `f ∈ V_F` and a role `r ∈ V_R` bind to the **outer (tensor) product** `f/r = f ⊗ r ∈ V_F ⊗ V_R`, with components `b_{φρ} = f_φ · r_ρ`. The binding space `V_B = V_F ⊗ V_R` therefore has dimension `dim(V_F) · dim(V_R)` — i.e., binding is **dimensionality-expanding** (O(d²) for `d_F = d_R = d`). A symbolic structure `S` with role decomposition `F/R` is represented additively (Def. 2.10): `Ψ_b(s) = Σ_{(f,r) ∈ β(s)} Ψ_F(f) ⊗ Ψ_R(r)`. Unbinding is achieved by inner product with the (dual of the) role vector. The paper shows the construction (i) recovers prior fully- and partially-localist schemes as special cases, (ii) supports **recursive** structures (e.g., trees, strings), (iii) admits Boolean and continuous variants, and (iv) allows formal capacity / interference analysis. Demonstrated structures include strings indexed by position roles and tree structures with car/cdr roles. Successor work — Plate (1995/2003) HRR, FHRR, MAP, BSC, VTB — explicitly frames itself as **compressing** TPR to keep dimensionality fixed.

## Paper → code map

| Paper concept | Codebase status |
|---|---|
| Tensor product binding `f ⊗ r` (Def. 2.9) | **Not implemented.** Per user direction this is **out of scope** — the library deliberately ships only fixed-dimension VSAs. |
| Outer-product binding space `V_F ⊗ V_R` of dimension `d_F · d_R` | Not implemented. |
| Recursive TPR for trees | Not implemented (the recursive primitive in this library is HRR circular convolution, not TPR). |
| Smolensky as the **historical predecessor** of HRR / VSA | Implicitly motivates the existence of HRR (`bind_hrr`, `bayes_hdc/functional.py:211`) and the FFT trick that makes circular convolution a fixed-`d` substitute for the outer product. **Currently uncited** anywhere in the repo. |

The conceptual lineage *Smolensky 1990 (TPR, O(d²)) → Plate 1995/2003 (HRR via circular convolution, O(d log d) compute, fixed `d`)* is the standard framing in the VSA literature (e.g., Kleyko et al. 2023 §II–III). The library implements the second step but not the first, and currently does not name the first step anywhere in DESIGN.md or README.md.

## Trivial fixes proposed
None. Per the user directive, do **not** propose adding a TPR implementation, a TPR file, or a TPR test. The single soft suggestion below is left to the user's discretion (report-only):

- *(Optional, one-liner only)* In `DESIGN.md §1` ("The algebra"), the sentence that introduces the family of bindings ("MAP uses element-wise product, HRR uses circular convolution, BSC uses XOR") could optionally acknowledge that all three are **fixed-`d` compressions of Smolensky's tensor-product binding**. This is a one-sentence parenthetical citation, not a code change. If added, the canonical citation is the one already in the bibliography (Smolensky 1990).

## Substantive findings (for user review)

1. **TPR is genuinely out of scope and the framing is correct.** The repo does not claim to implement tensor-product binding, and there is no false advertising. No fix needed in code.

2. **The citation, where it would naturally appear, is missing.** The library's whole reason for using *circular convolution* (rather than outer product) is to escape the O(d²) blow-up that TPR suffers. That is precisely the contribution of Smolensky 1990 → Plate 1995. Currently, neither `DESIGN.md` nor `README.md` nor the docstring of `bind_hrr` cites Smolensky 1990. This is a minor scholarly gap, not a code defect. The user may choose to add a one-line parenthetical citation in `DESIGN.md §1` (as suggested above) or leave the survey-style framing for `Kleyko et al. 2023` (paper [1]) to handle.

3. **No invariant or law in the codebase needs to change.** All algebraic claims in `DESIGN.md §1` (commutativity of bind, associativity of bundle, distributivity, etc.) are about HRR/MAP/BSC and remain correct independently of Smolensky.

## Recommended canonical citation

```bibtex
@article{smolensky1990tensor,
  author  = {Smolensky, Paul},
  title   = {Tensor Product Variable Binding and the Representation of Symbolic
             Structures in Connectionist Systems},
  journal = {Artificial Intelligence},
  volume  = {46},
  number  = {1--2},
  pages   = {159--216},
  year    = {1990},
  doi     = {10.1016/0004-3702(90)90007-M}
}
```

The form already used in the repo's intended bibliography matches this exactly; no change to the citation string is required if/when it is added.
