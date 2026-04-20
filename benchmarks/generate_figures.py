#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Generate paper-ready figures from benchmark results.

Reads ``benchmark_calibration_results.json`` and produces:

- ``figures/reliability_<dataset>.pdf`` — Guo-et-al. 2017 reliability
  diagram for each benchmark dataset, using the Bayes-HDC
  temperature-scaled probabilities.
- ``figures/coverage_<dataset>.pdf`` — empirical conformal coverage
  and mean set size vs target coverage, for each dataset.
- ``figures/accuracy_comparison.pdf`` — bar chart comparing Bayes-HDC
  accuracy to the baseline (TorchHD) across datasets.
- ``figures/ece_reduction.pdf`` — ECE before and after temperature
  scaling, for each library and dataset.

Because the library does not store the full (probs, labels) arrays
in the JSON results (it stores summaries), this script also reruns
the benchmark pipeline at lower dimensionality to collect
reliability-diagram-ready data for the four offline sklearn datasets.
MNIST reliability is generated from a fresh minimal pipeline too; the
goal is reproducible figures, not paper-final numbers.

Run with::

    python benchmarks/generate_figures.py

Requires the ``examples`` extras (matplotlib, scikit-learn).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

# Use headless backend before importing pyplot.
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import KBinsDiscretizer  # noqa: E402

from bayes_hdc import (  # noqa: E402
    MAP,
    ConformalClassifier,
    RandomEncoder,
    RegularizedLSClassifier,
    TemperatureCalibrator,
)
from bayes_hdc.datasets import (  # noqa: E402
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)
from bayes_hdc.plots import plot_coverage_curve, plot_reliability_diagram  # noqa: E402

OUT_DIR = Path(__file__).parent / "figures"
DIMS = 4096  # smaller than the headline benchmark; figures converge at this D
LEVELS = 32
SEED = 42
ALPHA = 0.1


def _reencode_and_classify(
    loader,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, int, str]:
    """Rerun the PVSA pipeline on one dataset.

    Returns ``(probs_cal, y_cal, probs_te, y_te, num_classes, name)``.
    """
    ds = loader()
    Xtr, Xte, ytr, yte = train_test_split(
        ds.X,
        ds.y,
        test_size=0.3,
        random_state=SEED,
        stratify=ds.y,
    )
    Xtr, Xca, ytr, yca = train_test_split(
        Xtr,
        ytr,
        test_size=0.3,
        random_state=SEED,
        stratify=ytr,
    )

    disc = KBinsDiscretizer(n_bins=LEVELS, encode="ordinal", strategy="quantile")
    X_tr_idx = disc.fit_transform(Xtr).astype(np.int32)
    X_ca_idx = np.clip(disc.transform(Xca), 0, LEVELS - 1).astype(np.int32)
    X_te_idx = np.clip(disc.transform(Xte), 0, LEVELS - 1).astype(np.int32)

    vsa = MAP.create(dimensions=DIMS)
    enc = RandomEncoder.create(
        num_features=Xtr.shape[1],
        num_values=LEVELS,
        dimensions=DIMS,
        vsa_model=vsa,
        key=jax.random.PRNGKey(SEED),
    )
    hv_tr = enc.encode_batch(jnp.asarray(X_tr_idx))
    hv_ca = enc.encode_batch(jnp.asarray(X_ca_idx))
    hv_te = enc.encode_batch(jnp.asarray(X_te_idx))

    clf = RegularizedLSClassifier.create(
        dimensions=DIMS,
        num_classes=ds.n_classes,
        reg=1.0,
    ).fit(hv_tr, jnp.asarray(ytr))

    logits_ca = hv_ca @ clf.weights
    logits_te = hv_te @ clf.weights

    calibrator = TemperatureCalibrator.create().fit(
        logits_ca,
        jnp.asarray(yca),
        max_iters=300,
    )
    probs_ca = calibrator.calibrate(logits_ca)
    probs_te = calibrator.calibrate(logits_te)

    return (
        probs_ca,
        jnp.asarray(yca),
        probs_te,
        jnp.asarray(yte),
        ds.n_classes,
        ds.name,
    )


def _generate_reliability_figures() -> None:
    """One reliability diagram per offline dataset."""
    loaders = [load_iris, load_wine, load_breast_cancer, load_digits]
    for loader in loaders:
        _, _, probs_te, yte, _, name = _reencode_and_classify(loader)
        fig, _ = plot_reliability_diagram(
            probs_te,
            yte,
            n_bins=12,
            title=f"Reliability — {name}",
        )
        fig.savefig(OUT_DIR / f"reliability_{name}.pdf", bbox_inches="tight", dpi=150)
        fig.savefig(OUT_DIR / f"reliability_{name}.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  wrote figures/reliability_{name}.{{pdf,png}}")


def _generate_coverage_figures() -> None:
    """One conformal-coverage curve per offline dataset."""
    loaders = [load_iris, load_wine, load_breast_cancer, load_digits]
    for loader in loaders:
        probs_ca, yca, probs_te, yte, _, name = _reencode_and_classify(loader)
        fig, _ = plot_coverage_curve(
            lambda a: ConformalClassifier.create(alpha=a),
            probs_ca,
            yca,
            probs_te,
            yte,
            alphas=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
            title=f"Conformal coverage — {name}",
        )
        fig.savefig(OUT_DIR / f"coverage_{name}.pdf", bbox_inches="tight", dpi=150)
        fig.savefig(OUT_DIR / f"coverage_{name}.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  wrote figures/coverage_{name}.{{pdf,png}}")


def _generate_accuracy_comparison(results_json: Path) -> None:
    """Grouped bar chart: Bayes-HDC vs TorchHD across datasets."""
    if not results_json.exists():
        print(f"  [skipped] no results file at {results_json}")
        return
    results = json.loads(results_json.read_text())

    names = []
    acc_bh = []
    acc_th = []
    for entry in results:
        names.append(entry["dataset"]["name"])
        acc_bh.append(entry["bayes_hdc"]["raw"]["accuracy"])
        if entry["torchhd"] is not None:
            acc_th.append(entry["torchhd"]["raw"]["accuracy"])
        else:
            acc_th.append(np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, acc_bh, width, label="Bayes-HDC", color="#2e75b6")
    ax.bar(x + width / 2, acc_th, width, label="TorchHD", color="#c00000")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy: Bayes-HDC vs TorchHD (identical pipeline)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(OUT_DIR / "accuracy_comparison.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(OUT_DIR / "accuracy_comparison.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  wrote figures/accuracy_comparison.{pdf,png}")


def _generate_ece_reduction(results_json: Path) -> None:
    """Grouped bar chart: ECE before and after temperature scaling."""
    if not results_json.exists():
        print(f"  [skipped] no results file at {results_json}")
        return
    results = json.loads(results_json.read_text())

    names = []
    ece_raw = []
    ece_cal = []
    for entry in results:
        names.append(entry["dataset"]["name"])
        ece_raw.append(entry["bayes_hdc"]["raw"]["ece"])
        ece_cal.append(entry["bayes_hdc"]["calibrated"]["ece"])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, ece_raw, width, label="ECE raw", color="#f08080")
    ax.bar(x + width / 2, ece_cal, width, label="ECE + T", color="#2e75b6")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Expected Calibration Error")
    ax.set_title("ECE reduction via temperature scaling (Bayes-HDC)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(OUT_DIR / "ece_reduction.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(OUT_DIR / "ece_reduction.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  wrote figures/ece_reduction.{pdf,png}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).parent / "benchmark_calibration_results.json",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to {OUT_DIR}/")
    print("---")

    print("Reliability diagrams:")
    _generate_reliability_figures()
    print("Coverage curves:")
    _generate_coverage_figures()
    print("Accuracy comparison:")
    _generate_accuracy_comparison(args.results)
    print("ECE reduction:")
    _generate_ece_reduction(args.results)
    print("---")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
