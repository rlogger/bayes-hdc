# SPDX-License-Identifier: MIT
# Copyright (c) 2026 bayes-hdc contributors
"""Reproducibility helpers shared by executable benchmark scripts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
from sklearn.preprocessing import StandardScaler


def standardize_train(train, *heldout):
    """Fit preprocessing only on proper training observations."""
    scaler = StandardScaler().fit(train)
    return tuple(scaler.transform(x).astype(np.float32) for x in (train, *heldout))


def provenance():
    """Record the checkout, its dirty diff, runtime, and resolved dependencies."""
    root = Path(__file__).resolve().parents[1]

    def git(*args):
        try:
            result = subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)
            return result.stdout if result.returncode == 0 else b""
        except OSError:
            return b""

    # Hash every source file, including untracked helpers, so a dirty checkout
    # cannot be mistaken for the unchanged base commit.
    digest = hashlib.sha256()
    for folder in (root / "bayes_hdc", root / "benchmarks"):
        for path in sorted(folder.rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    versions = {}
    for name in (
        "bayes-hdc",
        "jax",
        "jaxlib",
        "numpy",
        "scipy",
        "scikit-learn",
        "torch",
        "torch-hd",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "commit": git("rev-parse", "HEAD").decode().strip() or "unknown",
        "dirty": bool(git("status", "--porcelain")),
        "source_sha256": digest.hexdigest(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "packages": versions,
    }


def write_json(path, value):
    """Write portable JSON; undefined statistics are null, never NaN tokens."""

    def clean(obj):
        if isinstance(obj, dict):
            return {key: clean(val) for key, val in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(val) for val in obj]
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
