Bayes-HDC Documentation
=====================

Bayes-HDC is a high-performance library for Hyperdimensional Computing (HDC) and Vector Symbolic Architectures (VSA) built on JAX.

Features
--------

- XLA compilation and automatic kernel fusion
- Native GPU/TPU support through JAX backend
- Pure functional design enabling JAX transformations (jit, vmap, grad, pmap)
- Eight VSA model implementations: BSC, MAP, HRR, FHRR, BSBC, CGR, MCR, VTB
- Feature encoders for discrete, continuous, kernel, and graph data
- Classification models with iterative refinement
- Memory modules: SDM, Hopfield, attention-based retrieval

Quick Start
-----------

Installation::

   git clone https://github.com/rlogger/bayes-hdc.git
   cd bayes-hdc && pip install -e .

Basic usage::

   import jax
   from bayes_hdc import MAP

   model = MAP.create(dimensions=10000)
   key = jax.random.PRNGKey(42)

   x = model.random(key, (10000,))
   y = model.random(jax.random.split(key)[1], (10000,))
   bound = model.bind(x, y)
   similarity = model.similarity(x, y)

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   classification
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api
   functional
   vsa
   embeddings
   models
   memory
   utils

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
