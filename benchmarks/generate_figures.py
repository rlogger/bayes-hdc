#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 R.S.

"""Generate figures only from the predictions saved by the matched benchmark.

Run benchmark_calibration.py first. Legacy summary-only artifacts are rejected:
re-running another model to manufacture missing predictions would produce
figures for a different experiment. --output-dir controls all generated files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bayes_hdc import ConformalClassifier  # noqa: E402
from bayes_hdc.plots import plot_coverage_curve, plot_reliability_diagram  # noqa: E402

OUT_DIR = Path(__file__).parent / "figures"


def _read_results(results_json):
    results = json.loads(results_json.read_text())
    if not results or any(
        row.get("config", {}).get("protocol") != "matched-centroid-v2"
        or "predictions" not in row.get("bayes_hdc", {})
        for row in results
    ):
        raise ValueError(
            "Rerun benchmark_calibration.py: figures require matched-centroid-v2 predictions"
        )
    return results


def _generate_prediction_figures(results_json):
    import jax.numpy as jnp

    for row in _read_results(results_json):
        data = row["bayes_hdc"]["predictions"]
        name = row["dataset"]["name"]
        probs_te, yte = jnp.asarray(data["probs_test"]), jnp.asarray(data["y_test"])
        probs_ca, yca = jnp.asarray(data["probs_cal"]), jnp.asarray(data["y_cal"])
        for kind in ("reliability", "coverage"):
            if kind == "reliability":
                fig, _ = plot_reliability_diagram(
                    probs_te, yte, n_bins=12, title=f"Reliability — {name}"
                )
            else:
                fig, _ = plot_coverage_curve(
                    lambda a: ConformalClassifier.create(alpha=a),
                    probs_ca,
                    yca,
                    probs_te,
                    yte,
                    alphas=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
                    title=f"Conformal coverage — {name}",
                )
            for extension in ("pdf", "png"):
                fig.savefig(OUT_DIR / f"{kind}_{name}.{extension}", bbox_inches="tight", dpi=150)
            plt.close(fig)


def _generate_accuracy_comparison(results_json: Path) -> None:
    """Grouped bar chart: Bayes-HDC vs TorchHD across datasets."""
    if not results_json.exists():
        print(f"  [skipped] no results file at {results_json}")
        return
    results = _read_results(results_json)

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
    ax.set_title("Accuracy: Bayes-HDC vs TorchHD (matched cosine centroids)")
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
    results = _read_results(results_json)

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
    ax.set_title("ECE before and after temperature scaling (Bayes-HDC)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(OUT_DIR / "ece_reduction.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(OUT_DIR / "ece_reduction.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  wrote figures/ece_reduction.{pdf,png}")


def _set_output_dir(path):
    global OUT_DIR
    OUT_DIR = path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).parent / "benchmark_calibration_results.json",
    )
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    _read_results(args.results)
    _set_output_dir(args.output_dir)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to {OUT_DIR}/")
    print("---")

    _generate_prediction_figures(args.results)
    print("Accuracy comparison:")
    _generate_accuracy_comparison(args.results)
    print("ECE reduction:")
    _generate_ece_reduction(args.results)
    print("---")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
