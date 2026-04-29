# Paper 12: Gayler & Levy (2009) — A Distributed Basis for Analogical Mapping

## Bibliographic
- **Full citation.** R. W. Gayler and S. D. Levy, "A Distributed Basis for Analogical Mapping," in *New Frontiers in Analogy Research: Proceedings of the Second International Conference on Analogy (ANALOGY-2009)*, B. Kokinov, K. J. Holyoak, and D. Gentner, Eds., Sofia, Bulgaria: New Bulgarian University Press, 2009, pp. 165–174.
- **Author affiliations at time of writing.** Ross W. Gayler — School of Communication, Arts and Critical Enquiry, La Trobe University, Australia. Simon D. Levy — Department of Computer Science, Washington and Lee University, USA.
- **Open-access copy.** Berkeley Redwood Center mirror — `https://redwood.berkeley.edu/wp-content/uploads/2021/08/Gayler2009.pdf`. Reference MATLAB implementation: `tinyurl.com/gidemo` (now `github.com/simondlevy/GraphIsomorphism`).

## Summary (200 words)
Gayler and Levy generalise Kanerva's "Dollar of Mexico" pattern to *full analogical mapping as approximate graph isomorphism*. The setup: source and target situations are graphs; the mapping problem is to find a vertex-correspondence that maximally preserves edges. They build on Pelillo's (1999) reformulation of (sub)graph isomorphism as a max-clique problem in the *association graph*, solved by replicator-equation dynamics — a population-game iteration `x_i(t+1) = x_i(t) π_i(t) / Σ x_j π_j` with payoff `π_i = Σ w_ij x_j`. Their contribution is a distributed (VSA) re-implementation of Pelillo's localist circuit using MAP coding (Gayler 1998). Vertex sets are bundled (`A+B+C+D`), edge sets are bundled binds (`Σ bind(u,v)`), the candidate-mapping vector is the cross-product `x = (A+B+C+D)*(P+Q+R+S) = ΣΣ X*Y`, and the weight vector `w` is a four-fold bind of edge-sets. Iteration computes `π = x*w`, then a holistic *vector-intersection* operator `x ∧ π` (built from dual permutations and cleanup memory — sigma-pi units), and renormalises. They reproduce Pelillo's principal example (Figure 1: graphs ABCD vs PQRS, 16 candidate mappings) with d=10000 MAP vectors and recover the same isomorphism dynamics, breaking saddle-point symmetry. Successor citations: Levy & Gayler "lateral inhibition" (2009 ICCM), Eliasmith & Thagard's DRAMA (2001), and the entire VSA-for-analogy line (Plate, Kanerva, Smolensky). The paper is the canonical *graph-level* generalisation of the Dollar-of-Mexico micro-example.

## Paper → code map

The repository implements the *role-filler* tier of analogical reasoning (Kanerva 2010) but **does not** implement the *graph-isomorphism / replicator-equation* tier of Gayler & Levy 2009.

| Paper construct | Codebase counterpart |
|---|---|
| Vertex-set bundle `A + B + C + D` | `Multiset.from_vectors` (`structures.py:60–67`); also any `model.bundle` call. |
| Edge-set bundle `Σ bind(u_i, v_i)` (undirected) | `Graph.add_edge` / `Graph` value (`structures.py:190–200`). The graph structure exists. |
| Vertex-mapping vector `x = (A+B+C+D)*(P+Q+R+S)` | Implementable as `bind(bundle(src_nodes), bundle(tgt_nodes))`, but no example or helper does this. |
| Edge-mapping weight `w` (Equation 8 in the paper) | Implementable as `bind(graph_src.value, graph_tgt.value)`, but again no example. |
| Payoff `π = x*w` (Equation 9) | One `bind_map` call away. Not exposed. |
| Holistic intersection `x ∧ π` via dual permutations + cleanup (Figure 2) | **Not implemented.** This is the substantive novel mechanism of the paper. The cleanup memory is the closest analogue (`memory.py` exists; not inspected for this audit beyond the file listing). |
| Replicator iteration `x ← x∧π / ‖x∧π‖` (Equations 3–4) | **Not implemented.** No iterated-dynamics / fixed-point analogical mapper appears in `examples/`, `bayes_hdc/`, or `tests/`. |
| Pelillo's Figure-1 graph-isomorphism demo | **Not reproduced.** No example files demonstrate analogical mapping over multiple roles or graphs. |

A repository-wide `grep` for `Gayler`, `Levy`, `graph isomorphism`, `analogical map`, and `replicator` returns *zero* hits across `bayes_hdc/`, `examples/`, `tests/`, `docs/`, and `README.md`. The paper is not currently cited or referenced anywhere in the codebase.

## Trivial fixes proposed
- *None.* There is no existing code to correct against this paper.

## Substantive findings (for user review)

### S1 — Missing example for graph-level analogical mapping. *Low severity; in-scope per literature list.*
`kanerva_example.py` covers the simplest analogical-mapping pattern (single mapping vector between two role-filler records). The natural next-step example is *the* Gayler–Levy demo: Figure 1 of the paper (two 4-vertex graphs ABCD and PQRS with one structural ambiguity) with 10000-dim MAP vectors, showing the replicator iteration converging on `{A=P, B=Q, C=R, D=S}` (or the symmetric swap, broken by injected noise). The repository already provides every primitive needed:
- Vertex/edge bundling — `Multiset`, `Graph.add_edge`.
- MAP bind/inverse — `MAP.bind`, `F.inverse_map`.
- Cleanup memory — `bayes_hdc/memory.py` (contents not inspected in detail; functionality should be verifiable).
- Permutations — `F.permute` (used in `Sequence` and `Graph` directed mode).

The only missing primitive is the **holistic vector intersection** of Gayler-Levy 2009 §"Distributed Implementation" (their Figure 2): `intersect(x, π)` defined as `(P1(x) ∗ P2(π)) ∗ cleanup_memory[Σ_i X_i ∗ P1(X_i) ∗ P2(X_i)]`, summed over multiple permutation pairs. Adding this as a single helper in `functional.py` (perhaps `vector_intersect`) plus an `examples/gayler_levy_example.py` reproducing Figure 1 would close the in-scope gap from the user's literature list with ~150 lines of code. The MATLAB reference (`github.com/simondlevy/GraphIsomorphism`) provides a working numerical baseline.

The user has stated they are not adding *new* examples for out-of-scope items; Gayler-Levy 2009 is in the literature list, so this gap is reportable, but I am flagging severity as *low* because (a) the paper's main contribution is the algorithmic mechanism, not a particular dataset, and (b) Kanerva 2010's example already conveys 80% of the analogical-mapping intuition.

### S2 — README and docstrings under-credit Gayler. *Low severity, documentation.*
The README's tagline (`README.md:7`) and Sphinx meta-keywords (`docs/conf.py:183`) list "Kanerva, HRR, BSC, MAP, Hopfield, sparse distributed memory" but do not credit Gayler or VSA's analogical-mapping line. Given that MAP — the codebase's *default* VSA model — is Gayler's own 1998 contribution and that Gayler also coined the term "Vector Symbolic Architecture" (Gayler 2003, cited in this paper at p. 169), at minimum a `Gayler 1998` reference next to MAP and a `Gayler & Levy 2009` reference somewhere in `examples/kanerva_example.py` (as the natural sequel) would tighten the citation graph. The current `examples/kanerva_example.py` mentions "analogical reasoning" in the docstring (line 15) without pointing forward to the graph-level extension that the same authors built.

### S3 — `Graph` class is closer to Gayler-Levy than the audit-touchpoints suggested. *Informational.*
The brief specified `examples/kanerva_example.py`, `HashTable.get`, and `F.hash_table` as the expected touch-points for Kanerva 2010. The `Graph` class in `structures.py:170–212` is in fact the closer match for *Gayler-Levy 2009*: edge representation `bind(u, v)`, summed (their Equation 8 without the four-fold bind), and `Graph.neighbors(node)` (line 202–204) implements `bind(graph_value, inverse(node))`, which corresponds to "given a vertex, retrieve its neighbour-multiset" — a primitive Gayler-Levy use, though their replicator iteration goes further. Worth noting that the Graph class is already a partial implementation of Gayler-Levy primitives even though no example exercises this connection.

## Recommended canonical citation
```bibtex
@inproceedings{gayler2009distributed,
  author    = {Ross W. Gayler and Simon D. Levy},
  title     = {A Distributed Basis for Analogical Mapping},
  booktitle = {New Frontiers in Analogy Research: Proceedings of the Second
               International Conference on Analogy (ANALOGY-2009)},
  editor    = {Boicho Kokinov and Keith J. Holyoak and Dedre Gentner},
  publisher = {New Bulgarian University Press},
  address   = {Sofia, Bulgaria},
  year      = {2009},
  pages     = {165--174},
  url       = {https://redwood.berkeley.edu/wp-content/uploads/2021/08/Gayler2009.pdf}
}
```
