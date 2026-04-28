:hide-toc:

bayes-hdc
=========

**Probabilistic hyperdimensional computing in JAX.**

bayes-hdc is a JAX library for hyperdimensional computing (HDC) with a
probabilistic layer on top — *PVSA*, Probabilistic Vector Symbolic
Architectures. Every hypervector is an element of a small, well-designed
algebra: a commutative binding, an associative bundling, a cyclic group
action, a cosine measure, and a posterior distribution over the whole
thing. Every type is a JAX pytree, so ``jit``, ``vmap``, ``grad``,
``pmap``, and ``shard_map`` compose with every operation.

.. grid:: 1 2 2 3
   :gutter: 3
   :margin: 3 3 0 0

   .. grid-item-card:: :octicon:`zap` 30-second tour
      :link: quickstart
      :link-type: doc

      Construct a ``GaussianHV``, propagate moments through ``bind`` and
      ``bundle``, and read off expected cosine similarity — closed form,
      no Monte Carlo.

   .. grid-item-card:: :octicon:`graph` PVSA primitives
      :link: api
      :link-type: doc

      ``GaussianHV``, ``DirichletHV``, ``MixtureHV`` with closed-form
      moments and KL divergences. Reparameterisation gradients
      everywhere.

   .. grid-item-card:: :octicon:`shield-check` Coverage guarantees
      :link: api
      :link-type: doc

      ``ConformalClassifier`` returns prediction sets with marginal
      coverage ≥ 1 − α on exchangeable data. ``TemperatureCalibrator``
      fits the convex MLE temperature.

   .. grid-item-card:: :octicon:`workflow` Group structure
      :link: api
      :link-type: doc

      The cyclic-shift action of :math:`\mathbb{Z}/d` is first-class.
      Property-based verifiers reject ops claiming a symmetry they do
      not have.

   .. grid-item-card:: :octicon:`stack` 8 VSA models
      :link: vsa
      :link-type: doc

      BSC, MAP, HRR, FHRR, BSBC, CGR, MCR, VTB — uniform ``bind`` /
      ``bundle`` / ``inverse`` / ``similarity`` / ``random`` API.

   .. grid-item-card:: :octicon:`server` Scales out
      :link: api
      :link-type: doc

      ``pmap_bind_gaussian``, ``shard_map_bind_gaussian``,
      ``shard_classifier_posteriors`` for pod-scale training.

Install
-------

.. code-block:: bash

   pip install -e .                # core
   pip install -e ".[examples]"    # + matplotlib + scikit-learn
   pip install -e ".[dev]"         # + pytest, ruff, mypy

Status
------

Alpha. Versions 0.2 through 1.0 shipped — Gaussian and Dirichlet
posteriors, conformal prediction, temperature calibration, probabilistic
resonator, posterior predictive checks, streaming Bayesian updates,
multi-device sharding, 11 standard HDC datasets, and equivariance
verifiers. **480 tests, 97 % line coverage**, Ubuntu + macOS × Python
3.9–3.13 on every push.

The public API may shift before 1.0; behaviour changes are called out in
``CHANGELOG.md``.

----

User guide
----------

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   classification
   examples

API reference
-------------

.. toctree::
   :maxdepth: 2

   api
   functional
   vsa
   embeddings
   models
   memory
   utils

Project
-------

.. toctree::
   :maxdepth: 1

   contributing

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
