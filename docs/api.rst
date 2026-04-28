API reference
=============

This page is the canonical entry point for the public API. Detailed
signatures live in the per-module pages (:doc:`functional`, :doc:`vsa`,
:doc:`embeddings`, :doc:`models`, :doc:`memory`, :doc:`utils`); this page
is the high-level index.

PVSA — probabilistic primitives
-------------------------------

.. autosummary::

   bayes_hdc.GaussianHV
   bayes_hdc.DirichletHV
   bayes_hdc.MixtureHV
   bayes_hdc.bind_gaussian
   bayes_hdc.bundle_gaussian
   bayes_hdc.permute_gaussian
   bayes_hdc.cleanup_gaussian
   bayes_hdc.cleanup_gaussian_stacked
   bayes_hdc.inverse_gaussian
   bayes_hdc.expected_cosine_similarity
   bayes_hdc.similarity_variance
   bayes_hdc.kl_gaussian
   bayes_hdc.bind_dirichlet
   bayes_hdc.bundle_dirichlet
   bayes_hdc.kl_dirichlet

Bayesian classifiers
--------------------

.. autosummary::

   bayes_hdc.BayesianCentroidClassifier
   bayes_hdc.BayesianAdaptiveHDC
   bayes_hdc.StreamingBayesianHDC

Uncertainty quantification
--------------------------

.. autosummary::

   bayes_hdc.TemperatureCalibrator
   bayes_hdc.ConformalClassifier
   bayes_hdc.posterior_predictive_check
   bayes_hdc.coverage_calibration_check

Group-theoretic structure
-------------------------

.. autosummary::

   bayes_hdc.shift
   bayes_hdc.compose_shifts
   bayes_hdc.hrr_equivariant_bilinear
   bayes_hdc.verify_shift_equivariance
   bayes_hdc.verify_single_argument_shift_equivariance
   bayes_hdc.verify_shift_invariance

Inference helpers
-----------------

.. autosummary::

   bayes_hdc.elbo_gaussian
   bayes_hdc.reconstruction_log_likelihood_mc
   bayes_hdc.probabilistic_resonator

Multi-device
------------

.. autosummary::

   bayes_hdc.distributed.pmap_bind_gaussian
   bayes_hdc.distributed.pmap_bundle_gaussian
   bayes_hdc.distributed.shard_map_bind_gaussian
   bayes_hdc.distributed.shard_classifier_posteriors

Classical VSA models
--------------------

.. autosummary::

   bayes_hdc.MAP
   bayes_hdc.BSC
   bayes_hdc.HRR
   bayes_hdc.FHRR
   bayes_hdc.BSBC
   bayes_hdc.CGR
   bayes_hdc.MCR
   bayes_hdc.VTB

See :doc:`functional`, :doc:`vsa`, :doc:`embeddings`, :doc:`models`,
:doc:`memory`, and :doc:`utils` for full per-module documentation
including method signatures, return types, and docstrings.
