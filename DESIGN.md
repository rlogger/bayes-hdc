# Design notes

The long-form companion to the README, covering the algebraic structure
of the library, the functional-programming commitments, and the JAX
idioms it relies on. Read this if you want to extend the library or
understand the trade-offs behind the API.

## 1. The algebra

Why an algebra at all? Fodor & Pylyshyn (1988) argued that any architecture aspiring to model cognition must support combinatorial syntax-and-semantics and structure-sensitive processes — the two commitments that distinguish "Classical" symbol systems from spreading-activation networks. VSAs answer that critique with a connectionist substrate that is nonetheless a closed algebra over a fixed-dimensional vector space.

A *Vector Symbolic Architecture* (Gayler 2003) is a compact algebraic object on $\mathbb{R}^d$: a commutative binding $\star$, an associative bundling $\oplus$, a cyclic group action $T_k$, and a similarity measure (cosine). The choice of binding selects the VSA family — MAP uses element-wise product, HRR uses circular convolution, BSC uses XOR — but the interface is uniform. All three are fixed-dimensional compressions of Smolensky's (1990) tensor-product binding, which produces a $d^2$-dimensional output and which the modern VSA family deliberately avoids.

### What the laws say

For any hypervectors $x, y, z \in \mathbb{R}^d$:

- **Commutative bind:** $x \star y = y \star x$.
- **Associative bundle:** $(x \oplus y) \oplus z = x \oplus (y \oplus z)$.
- **Distributivity:** $x \star (y \oplus z) \approx (x \star y) \oplus (x \star z)$ (exact in HRR, approximate after normalisation in MAP).
- **Self-inverse bind (MAP/BSC):** $x \star x \approx \mathbf{1}$ up to the codebook.
- **Quasi-orthogonality** (Kanerva 2009): for random $x, y$, $\mathbb{E}[\cos(x, y)] \approx 0$ with variance $1/d$.

`tests/test_functional.py` checks these at realistic dimensions. The ones
that hold exactly are checked with `jnp.allclose`; the ones that hold up
to dimension-dependent concentration are checked with tolerances chosen to
flag violations without being flaky.

### The group action

For fixed $d$, cyclic shift by $k$ defines an action of $\mathbb{Z}/d$ on $\mathbb{R}^d$:

$$
T_k : \mathbb{R}^d \to \mathbb{R}^d, \qquad T_k(x)_i = x_{(i - k) \bmod d}.
$$

The action is faithful, additive, and isometric:

- $T_k(x) = x \ \forall x \iff k \equiv 0 \pmod{d}$ (faithful).
- $T_j \circ T_k = T_{j+k}$ (additive).
- $\|T_k(x)\| = \|x\|$ and $\langle T_k(x), T_k(y)\rangle = \langle x, y\rangle$ (isometric).

The HDC primitives have two flavours of equivariance with respect to this
action, and conflating them is a common mistake:

- **Diagonal equivariance** — shifting *every* argument shifts the output.
  Element-wise binding and bundling satisfy this.
- **Single-argument equivariance** — shifting *one* argument shifts the
  output; under the diagonal action the output picks up a double shift.
  Circular-convolution binding satisfies this.

Both are correct, and both are useful. The module `bayes_hdc.equivariance`
documents both, names them separately, and provides verifiers for each.
Test file: `tests/test_equivariance.py`.

## 2. PVSA: lifting to measures

The probabilistic layer replaces each hypervector $x \in \mathbb{R}^d$ with a posterior distribution $X$. A `GaussianHV` carries a mean $\mu \in \mathbb{R}^d$ and a per-dimension variance $\sigma^2 \in \mathbb{R}_{\ge 0}^d$.

### Closed-form moments

For independent $X \sim \mathcal{N}(\mu_x, \mathrm{diag}(\sigma_x^2))$ and $Y \sim \mathcal{N}(\mu_y, \mathrm{diag}(\sigma_y^2))$, the first and second moments of the element-wise product $Z = X \cdot Y$ are exact:

$$
\begin{aligned}
\mathbb{E}[Z]   &= \mu_x \cdot \mu_y, \\
\mathrm{Var}[Z] &= \mu_x^2 \sigma_y^2 + \mu_y^2 \sigma_x^2 + \sigma_x^2 \sigma_y^2.
\end{aligned}
$$

`bind_gaussian` returns a `GaussianHV` with exactly these moments. It is not a Monte Carlo estimate. It is not a delta-method approximation. It is the analytic answer.

The sum (bundle) is trivial under independence:

$$
\begin{aligned}
\mathbb{E}\!\left[\textstyle\sum_i X_i\right]   &= \textstyle\sum_i \mu_i, \\
\mathrm{Var}\!\left[\textstyle\sum_i X_i\right] &= \textstyle\sum_i \sigma_i^2.
\end{aligned}
$$

Normalisation onto the unit sphere uses the delta method for the variance term, which is the dominant source of approximation in `bundle_gaussian` — the cost of insisting the library always returns objects on the manifold classical HDC uses.

### KL divergences

Gaussian–Gaussian and Dirichlet–Dirichlet KL divergences have closed forms:

$$
D_{\mathrm{KL}}\!\bigl(\mathcal{N}(\mu_0, \Sigma_0) \,\|\, \mathcal{N}(\mu_1, \Sigma_1)\bigr)
= \tfrac{1}{2}\!\left[\mathrm{tr}(\Sigma_1^{-1}\Sigma_0) + (\mu_1 - \mu_0)^\top \Sigma_1^{-1} (\mu_1 - \mu_0) - d + \ln\tfrac{|\Sigma_1|}{|\Sigma_0|}\right].
$$

`kl_gaussian` and `kl_dirichlet` return this analytically. They are differentiable end-to-end under `jax.grad`, which is what makes them useful in a variational objective.

### Reparameterisation

Every `GaussianHV` has a `.sample(key)` method that uses the standard
reparameterisation trick. `jax.grad` composes through every distributional
op by construction; there is no hidden non-differentiable step.

## 3. Functional programming commitments

The library is deliberately FP-shaped.

- **Immutable values.** Every type is a frozen dataclass. There is no
  :code:`__setattr__`, no hidden mutable state. Updates return new objects.
  The exception — `AdaptiveHDC` — uses JAX's `lax.scan` under the hood so
  the appearance of mutation is an illusion over pure folds.

- **Pure functions.** Core ops — `bind_gaussian`, `bundle_gaussian`,
  `permute_gaussian`, `cleanup_gaussian`, `inverse_gaussian`, `kl_gaussian` —
  have no side effects, no global state, no dependency on time.

- **Small, typed API.** Every public function has explicit argument types
  and a return type. `Any` is avoided. Shape contracts are stated in the
  docstring and checked by tests.

- **Pytree-native.** Every dataclass is registered via
  `jax.tree_util.register_dataclass`. `jit`, `vmap`, `grad`, `pmap`,
  `shard_map` compose unconditionally without user-side `flatten_util`
  boilerplate.

- **No custom VJPs where the default works.** The library prefers to lean
  on JAX's autodiff. Custom `jvp`/`vjp` rules appear only when there is a
  measurable speedup and they never change the numerical output.

## 4. JAX idioms

- **`jit` at the boundary, not the interior.** Individual primitives are
  cheap; composite operations (e.g. `BayesianCentroidClassifier.fit`)
  `jit` their inner workers but leave the constructor and the dispatch
  un-jitted so traceable error messages survive.
- **`vmap` for batching, not manual loops.** Every op has a natural batched
  form via `vmap`. `encode_batch`, `bind_batch`, etc. are thin wrappers.
- **`shard_map` at scale.** `pmap_bind_gaussian` and `shard_map_bind_gaussian`
  fall back to single-device when only one device is visible, so the same
  code runs on a laptop and on a pod.
- **`jax.random.PRNGKey` splitting is explicit.** Keys are never reused.
  Every function that consumes randomness takes a key as an argument; there
  is no hidden global key.
- **Float32 default, float64 available.** The library is careful about
  dtypes; `jnp.astype` is used where promotion matters; no accidental
  upcasts.

## 5. Versioning and stability

The library is at `0.4.0a0`. The public API (everything in
`bayes_hdc/__init__.py`) follows semver loosely — breaking changes are
called out in `CHANGELOG.md`. Before `1.0`, names may be renamed or
reorganised; behaviours will not silently change.

The internal layout is pytree-first and module-local: new distributional
types belong in `bayes_hdc/distributions.py`, new inference primitives in
`bayes_hdc/inference.py`, new group-theoretic helpers in
`bayes_hdc/equivariance.py`. Public names are re-exported from
`bayes_hdc/__init__.py` with an explicit `__all__`.

## 6. When to reach for this library

Reach for it when you need one or more of:

- a well-typed hypervector algebra with pytree-native composition;
- closed-form moment propagation for a probabilistic HDC pipeline;
- coverage-guaranteed or calibrated predictions on top of an HDC classifier;
- a distribution-valued representation of classifier weights;
- structured representations for task-conditioned agents;
- **bounded-memory streaming inference under distribution shift.** `StreamingBayesianHDC` maintains exponential-moving-average posteriors per class in `O(K·d)` memory regardless of stream length — useful for non-stationary biomedical-signal streams (drifting EMG calibration, EEG montage shifts, wearable-sensor recalibration) where a Kalman-style fixed-shrinkage filter (`BayesianAdaptiveHDC`) is too rigid and a full posterior fit is too expensive.

Reach for something else when the task calls for deep end-to-end learning
on natural images at ImageNet scale, or when the uncertainty in your model
is irreducibly aleatoric and no amount of Bayesian machinery will surface
it.

## 7. Related approaches not implemented

Several well-known research lines sit alongside this library; we cite them so that readers know the right adjacent literature without expecting bayes-hdc to provide it.

- **Tensor-product binding** (Smolensky 1990; Mizraji 1989). Dimension-expanding binding via outer / Kronecker product. The library implements only fixed-dimensional compressed bindings (Hadamard for MAP; XOR for BSC; circular convolution for HRR). HRR/MAP/BSC are the compressions of TPR; we do not provide the uncompressed ancestor.
- **Spiking-neuron VSA / Semantic Pointer Architecture / Neural Engineering Framework** (Stewart & Eliasmith 2011, *Oxford Handbook of Compositionality*; Stewart, Bekolay & Eliasmith 2011, *Connection Science* 23(2): 145-153; Stewart, Tang & Eliasmith 2010, *Cognitive Systems Research* 12: 84-92; Rasmussen & Eliasmith 2011, *Topics in Cognitive Science* 3: 140-153). A neighbouring research line implements the same VSA primitives on populations of leaky-integrate-and-fire spiking neurons via the NEF, with cleanup as attractor dynamics on a population. bayes-hdc operates on `jax.Array` hypervectors with deterministic JAX ops; spike trains, tuning curves, population decoding, and Nengo integration are out of scope. The HRR/MAP primitives here can serve as building blocks for that line, but the library does not provide the neural-implementation layer.
- **Robot behaviour hierarchies via VSA** (Levy, Bajracharya & Gayler 2013, AAAI 2013 Workshop on Learning Rich Representations from Low-Level Sensors). The library supplies the same MAP primitives but no robot-control loop, simulator integration, or behaviour-hierarchy module.
- **VSA-based graph-isomorphism analogical mapping** (Gayler & Levy 2009). The replicator-iteration mechanism with the holistic vector-intersection primitive is not implemented; the reference MATLAB code is at <https://github.com/simondlevy/GraphIsomorphism>.
- **Random Indexing / corpus-trained context vectors** (Hecht-Nielsen 1994; Sahlgren 2005; the BEAGLE training loop of Jones & Mewhort 2007). The library provides random-projection encoders and bag-of-words bundling but no corpus-pass training loop that updates per-token vectors from co-occurrence statistics.

For the broader application landscape, see Kleyko, Rachkovskij, Osipov & Rahimi (2023), *A Survey on HDC aka VSA, Part II: Applications, Cognitive Models, and Challenges*, ACM Computing Surveys 55(9), Article 175.
