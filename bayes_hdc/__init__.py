# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Bayes-HDC: Probabilistic Vector Symbolic Architectures (PVSA) on JAX.

PVSA is an HDC algebra in which every hypervector is a posterior
distribution, and every VSA primitive propagates that posterior's
moments in closed form. Bayes-HDC is the first library implementing
PVSA: it ships Gaussian and Dirichlet hypervector types, a
temperature calibrator, a conformal classifier with coverage
guarantees, and calibration metrics (ECE / MCE / Brier / reliability).

On top of PVSA, the library provides a complete deterministic VSA
foundation (eight classical models, five encoders, five classifiers,
three memory modules, four symbolic data structures, capacity analysis)
as a baseline and substrate.
"""

__version__ = "0.4.0a0"

from bayes_hdc import (
    bayesian_models,
    datasets,
    distributed,
    distributions,
    embeddings,
    functional,
    inference,
    memory,
    metrics,
    models,
    structures,
    uncertainty,
    utils,
    vsa,
)
from bayes_hdc.bayesian_models import BayesianAdaptiveHDC, BayesianCentroidClassifier
from bayes_hdc.distributions import (
    DirichletHV,
    GaussianHV,
    MixtureHV,
    bind_dirichlet,
    bind_gaussian,
    bundle_dirichlet,
    bundle_gaussian,
    cleanup_gaussian,
    expected_cosine_similarity,
    inverse_gaussian,
    kl_dirichlet,
    kl_gaussian,
    permute_gaussian,
    similarity_variance,
)
from bayes_hdc.embeddings import (
    GraphEncoder,
    KernelEncoder,
    LevelEncoder,
    ProjectionEncoder,
    RandomEncoder,
)
from bayes_hdc.functional import (
    add_noise_map,
    bind_bsc,
    bind_map,
    bind_sequence,
    bundle_bsc,
    bundle_map,
    bundle_sequence,
    cleanup,
    cosine_similarity,
    cross_product,
    dot_similarity,
    flip_fraction,
    fractional_power,
    graph_encode,
    hamming_similarity,
    hard_quantize,
    hash_table,
    inverse_bsc,
    inverse_map,
    jaccard_similarity,
    multibind_bsc,
    multibind_map,
    negative_bsc,
    negative_map,
    ngrams,
    permute,
    resonator,
    select_bsc,
    select_map,
    soft_quantize,
    threshold,
    tversky_similarity,
    window,
)
from bayes_hdc.inference import elbo_gaussian, reconstruction_log_likelihood_mc
from bayes_hdc.memory import AttentionMemory, HopfieldMemory, SparseDistributedMemory
from bayes_hdc.metrics import (
    brier_score,
    bundle_capacity,
    bundle_snr,
    cosine_matrix,
    effective_dimensions,
    expected_calibration_error,
    maximum_calibration_error,
    negative_log_likelihood,
    reliability_curve,
    retrieval_confidence,
    saturation,
    sharpness,
    signal_energy,
    sparsity,
)
from bayes_hdc.models import (
    AdaptiveHDC,
    CentroidClassifier,
    ClusteringModel,
    LVQClassifier,
    RegularizedLSClassifier,
)
from bayes_hdc.structures import Graph, HashTable, Multiset, Sequence
from bayes_hdc.uncertainty import ConformalClassifier, TemperatureCalibrator
from bayes_hdc.vsa import BSBC, BSC, CGR, FHRR, HRR, MAP, MCR, VTB

__all__ = [
    # Modules
    "functional",
    "vsa",
    "embeddings",
    "models",
    "utils",
    "memory",
    "metrics",
    "structures",
    "distributions",
    "uncertainty",
    "datasets",
    "bayesian_models",
    "inference",
    "distributed",
    # Bayesian classifiers
    "BayesianCentroidClassifier",
    "BayesianAdaptiveHDC",
    # Variational inference
    "elbo_gaussian",
    "reconstruction_log_likelihood_mc",
    # Bayesian hypervectors — Gaussian
    "GaussianHV",
    "bind_gaussian",
    "bundle_gaussian",
    "expected_cosine_similarity",
    "similarity_variance",
    "kl_gaussian",
    "permute_gaussian",
    "cleanup_gaussian",
    "inverse_gaussian",
    # Bayesian hypervectors — Dirichlet
    "DirichletHV",
    "bind_dirichlet",
    "bundle_dirichlet",
    "kl_dirichlet",
    # Bayesian hypervectors — Mixture
    "MixtureHV",
    # Uncertainty quantification
    "TemperatureCalibrator",
    "ConformalClassifier",
    # Core operations
    "bind_bsc",
    "bundle_bsc",
    "inverse_bsc",
    "negative_bsc",
    "hamming_similarity",
    "bind_map",
    "bundle_map",
    "inverse_map",
    "negative_map",
    "cosine_similarity",
    "dot_similarity",
    "permute",
    "cleanup",
    # Multi-vector operations
    "multibind_map",
    "multibind_bsc",
    "cross_product",
    # Composite encodings
    "hash_table",
    "ngrams",
    "bundle_sequence",
    "bind_sequence",
    "graph_encode",
    "resonator",
    # Additional similarity metrics
    "jaccard_similarity",
    "tversky_similarity",
    # Selection and threshold
    "select_bsc",
    "select_map",
    "threshold",
    "window",
    # Noise injection
    "flip_fraction",
    "add_noise_map",
    # Power and quantisation
    "fractional_power",
    "soft_quantize",
    "hard_quantize",
    # VSA models
    "BSC",
    "BSBC",
    "MAP",
    "HRR",
    "FHRR",
    "CGR",
    "MCR",
    "VTB",
    # Encoders
    "RandomEncoder",
    "LevelEncoder",
    "ProjectionEncoder",
    "KernelEncoder",
    "GraphEncoder",
    # Classifiers
    "CentroidClassifier",
    "AdaptiveHDC",
    "LVQClassifier",
    "RegularizedLSClassifier",
    "ClusteringModel",
    # Memory
    "SparseDistributedMemory",
    "HopfieldMemory",
    "AttentionMemory",
    # Metrics — capacity / diagnostics
    "bundle_snr",
    "bundle_capacity",
    "effective_dimensions",
    "sparsity",
    "signal_energy",
    "saturation",
    "cosine_matrix",
    "retrieval_confidence",
    # Metrics — calibration
    "expected_calibration_error",
    "maximum_calibration_error",
    "brier_score",
    "sharpness",
    "negative_log_likelihood",
    "reliability_curve",
    # Structures
    "Multiset",
    "HashTable",
    "Sequence",
    "Graph",
]
