# Design notes

This document is the long-form companion to the README. It exists to make the
design choices of the library legible — the algebraic structure, the
functional-programming commitments, the JAX idioms, and the research
programmes the design is in conversation with.

## 1. The algebra

A Vector Symbolic Architecture (VSA) is a compact algebraic object on
:math:`\mathbb{R}^d`: a commutative binding :math:`\star`, an associative
bundling :math:`\oplus`, a cyclic group action :math:`T_k`, and a similarity
measure (cosine). The choice of binding selects the VSA family — MAP uses
element-wise product, HRR uses circular convolution, BSC uses XOR — but the
interface is uniform.

### What the laws say

For any hypervectors :math:`x, y, z \in \mathbb{R}^d`:

- **Commutative bind:** :math:`x \star y = y \star x`.
- **Associative bundle:** :math:`(x \oplus y) \oplus z = x \oplus (y \oplus z)`.
- **Distributivity:** :math:`x \star (y \oplus z) \approx (x \star y) \oplus (x \star z)`
  (exact in HRR, approximate after normalisation in MAP).
- **Self-inverse bind (MAP/BSC):** :math:`x \star x \approx \mathbf{1}` up to
  the codebook.
- **Quasi-orthogonality:** For random :math:`x, y`, :math:`\mathbb{E}[\cos(x, y)] \approx 0`
  with variance :math:`1/d`.

`tests/test_functional.py` checks these at realistic dimensions. The ones
that hold exactly are checked with `jnp.allclose`; the ones that hold up
to dimension-dependent concentration are checked with tolerances chosen to
flag violations without being flaky.

### The group action

For fixed :math:`d`, cyclic shift by :math:`k` defines an action of
:math:`\mathbb{Z}/d` on :math:`\mathbb{R}^d`:

.. math::

    T_k : \mathbb{R}^d \to \mathbb{R}^d, \qquad T_k(x)_i = x_{(i - k) \bmod d}.

The action is faithful, additive, and isometric:

- :math:`T_k(x) = x \ \forall x \iff k \equiv 0 \pmod{d}` (faithful).
- :math:`T_j \circ T_k = T_{j+k}` (additive).
- :math:`\|T_k(x)\| = \|x\|` and :math:`\langle T_k(x), T_k(y)\rangle = \langle x, y\rangle`
  (isometric).

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

The probabilistic layer replaces each hypervector :math:`x \in \mathbb{R}^d`
with a posterior distribution :math:`X`. A `GaussianHV` carries a mean
:math:`\mu \in \mathbb{R}^d` and a per-dimension variance
:math:`\sigma^2 \in \mathbb{R}_{\ge 0}^d`.

### Closed-form moments

For independent :math:`X \sim \mathcal{N}(\mu_x, \mathrm{diag}(\sigma_x^2))`
and :math:`Y \sim \mathcal{N}(\mu_y, \mathrm{diag}(\sigma_y^2))`, the
first and second moments of the element-wise product :math:`Z = X \cdot Y`
are exact:

.. math::

    \mathbb{E}[Z]   &= \mu_x \cdot \mu_y, \\
    \mathrm{Var}[Z] &= \mu_x^2 \sigma_y^2 + \mu_y^2 \sigma_x^2 + \sigma_x^2 \sigma_y^2.

`bind_gaussian` returns a `GaussianHV` with exactly these moments. It is
not a Monte Carlo estimate. It is not a delta-method approximation. It is
the analytic answer.

The sum (bundle) is trivial under independence:

.. math::

    \mathbb{E}\big[\textstyle\sum_i X_i\big] &= \textstyle\sum_i \mu_i, \\
    \mathrm{Var}\big[\textstyle\sum_i X_i\big] &= \textstyle\sum_i \sigma_i^2.

Normalisation onto the unit sphere uses the delta method for the variance
term, which is the dominant source of approximation in `bundle_gaussian` —
the cost of insisting the library always returns objects on the manifold
classical HDC uses.

### KL divergences

Gaussian-Gaussian and Dirichlet-Dirichlet KL divergences have closed forms:

.. math::

    D_{\mathrm{KL}}(\mathcal{N}(\mu_0, \Sigma_0) \,\|\, \mathcal{N}(\mu_1, \Sigma_1))
        = \tfrac{1}{2}\Big[\mathrm{tr}(\Sigma_1^{-1}\Sigma_0) +
          (\mu_1 - \mu_0)^\top \Sigma_1^{-1} (\mu_1 - \mu_0) - d +
          \ln\tfrac{|\Sigma_1|}{|\Sigma_0|}\Big].

`kl_gaussian` and `kl_dirichlet` return this analytically. They are
differentiable end-to-end under `jax.grad`, which is what makes them
useful in a variational objective.

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

## 5. Research programmes the design serves

### Weight-space learning

The shift in the literature from "networks train on data" to "networks
operate on networks" creates demand for weight-space representations that
are well-typed, symmetric-by-construction, and distribution-valued. A
`BayesianCentroidClassifier` here satisfies all three simultaneously:

- **Well-typed**: the weights are K hypervectors. You can read them, write
  them, sample from them, bind them, bundle them, KL them.
- **Symmetric**: the posterior is `Z/d`-equivariant, as verified
  numerically in `examples/weight_space_posterior.py`.
- **Distribution-valued**: each class-centroid is a `GaussianHV`, not a
  point estimate. Epistemic uncertainty is a straight read-off from
  `clf.var`; sampled weight configurations are alternate classifiers.

This is exactly the representation the weight-space research programme
assumes, offered on a substrate where the primitives are algebraic rather
than MLP-based.

### Equivariant neural functionals (NFNs)

NFN-style architectures build layers that are equivariant under the
symmetries of weight-space. Two of those symmetries are already first-class
here: the cyclic shift action of `Z/d`, and the channel permutation that
acts on :math:`\mathbb{R}^{K \times d}`.

- `hrr_equivariant_bilinear` is the canonical single-argument
  shift-equivariant bilinear layer — the operator you want at the bottom
  of an NFN block that operates on hypervectors.
- `verify_shift_equivariance` / `verify_single_argument_shift_equivariance`
  are property-based checkers; you can attach them to any user-defined
  layer as a unit test.

### Meta-RL with structured representations

Meta-RL agents that generalise across tasks need a composable
representation of "(task, state)". HDC gives one by construction:
`bind(task_hv, state_hv)` is a structured pair with an approximate inverse
via similarity. Task and state live on the same manifold; the task-state
symmetry group is inherited.

The library's classifier line-up maps onto standard meta-RL primitives:

- **Posterior sampling / Thompson exploration** —
  `BayesianCentroidClassifier` stores a Gaussian posterior per action;
  sampling a weight configuration is one line.
- **UCB** — `similarity_variance` gives `Var[⟨x, W_a⟩]` directly from the
  posterior; no ensembles needed.
- **Novelty / intrinsic reward** — per-class posterior Mahalanobis
  distance (the OOD signal in `anomaly_detection.py`).
- **Safe RL** — conformal prediction sets over action outcomes; abstain
  when the set is not a singleton.
- **Non-stationary streams** — `StreamingBayesianHDC` maintains an EMA
  posterior with bounded memory.

## 6. What this library is not

- It is not a deep-net framework. It does not compete with flax, equinox,
  or haiku. It composes with them.
- It is not a drop-in replacement for classical HDC on large vision tasks.
  The universal approximation of a transformer is not on the menu.
- It is not a port. Every component is implemented from the primary paper;
  see [`ORIGINALITY.md`](ORIGINALITY.md).
- It is not a research platform where correctness is aspirational. Every
  theorem in the docs is a property-based test in `tests/`.

## 7. Versioning and stability

The library is at `0.4.0a0`. The public API (everything in
`bayes_hdc/__init__.py`) follows semver loosely — breaking changes are
called out in `CHANGELOG.md`. Before `1.0`, names may be renamed or
reorganised; behaviours will not silently change.

The internal layout is pytree-first and module-local: new distributional
types belong in `bayes_hdc/distributions.py`, new inference primitives in
`bayes_hdc/inference.py`, new group-theoretic helpers in
`bayes_hdc/equivariance.py`. Public names are re-exported from
`bayes_hdc/__init__.py` with an explicit `__all__`.

## 8. When to reach for this library

Reach for it when you need one or more of:

- a well-typed hypervector algebra with pytree-native composition;
- closed-form moment propagation for a probabilistic HDC pipeline;
- coverage-guaranteed or calibrated predictions on top of an HDC classifier;
- an equivariance-respecting substrate for NFN-style experiments;
- a distribution-valued representation of classifier weights;
- structured representations for meta-RL or task-conditioned agents.

Reach for something else when the task calls for deep end-to-end learning
on natural images at ImageNet scale, or when the uncertainty in your model
is irreducibly aleatoric and no amount of Bayesian machinery will surface
it.
