# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""JAX-HDC: Hyperdimensional Computing with JAX."""

__version__ = "0.1.0a0"

from jax_hdc import embeddings, functional, memory, metrics, models, structures, utils, vsa
from jax_hdc.embeddings import (
    GraphEncoder,
    KernelEncoder,
    LevelEncoder,
    ProjectionEncoder,
    RandomEncoder,
)
from jax_hdc.functional import (
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
from jax_hdc.memory import AttentionMemory, HopfieldMemory, SparseDistributedMemory
from jax_hdc.metrics import (
    bundle_capacity,
    bundle_snr,
    cosine_matrix,
    effective_dimensions,
    retrieval_confidence,
    saturation,
    signal_energy,
    sparsity,
)
from jax_hdc.models import (
    AdaptiveHDC,
    CentroidClassifier,
    ClusteringModel,
    LVQClassifier,
    RegularizedLSClassifier,
)
from jax_hdc.structures import Graph, HashTable, Multiset, Sequence
from jax_hdc.vsa import BSBC, BSC, CGR, FHRR, HRR, MAP, MCR, VTB

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
    # Metrics
    "bundle_snr",
    "bundle_capacity",
    "effective_dimensions",
    "sparsity",
    "signal_energy",
    "saturation",
    "cosine_matrix",
    "retrieval_confidence",
    # Structures
    "Multiset",
    "HashTable",
    "Sequence",
    "Graph",
]
