# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Plotting helpers for calibration and coverage diagnostics.

This module is optional. It requires :mod:`matplotlib` — install with
``pip install bayes-hdc[examples]`` or ``pip install matplotlib``. All
functions raise a friendly :class:`ImportError` if matplotlib is missing.

Two helpers are provided:

- :func:`plot_reliability_diagram` — the Guo et al. (2017) reliability
  diagram showing per-bin accuracy vs confidence, with the ideal
  :math:`y = x` calibration line and the ECE gap bars.
- :func:`plot_coverage_curve` — for conformal predictors, the
  empirical coverage / mean set size as a function of the miscoverage
  level :math:`\\alpha`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from bayes_hdc.metrics import expected_calibration_error, reliability_curve

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover — user-environment dependent
        raise ImportError(
            "bayes_hdc.plots requires matplotlib. Install with "
            "`pip install bayes-hdc[examples]` or `pip install matplotlib`."
        ) from e
    return plt


def plot_reliability_diagram(
    probs: Any,
    labels: Any,
    *,
    n_bins: int = 15,
    ax: Axes | None = None,
    show_ece: bool = True,
    show_gap: bool = True,
    title: str = "Reliability Diagram",
) -> tuple[Figure, Axes]:
    """Plot a reliability diagram (Guo et al. 2017).

    Args:
        probs: Predicted class probabilities of shape ``(n, k)``.
        labels: Integer class labels of shape ``(n,)``.
        n_bins: Number of equal-width confidence bins on ``[0, 1]``.
        ax: Existing matplotlib :class:`Axes` to plot into. A new figure
            is created if ``None``.
        show_ece: Whether to render the computed ECE as an inset label.
        show_gap: Whether to draw the accuracy–confidence gap bars (red
            overlay on the blue accuracy bars).
        title: Plot title.

    Returns:
        ``(figure, axes)`` — the matplotlib figure and axes. The caller
        can save, further customise, or embed into a larger layout.
    """
    plt = _require_matplotlib()

    centers, accs, confs, counts = reliability_curve(probs, labels, n_bins=n_bins)
    centers = np.asarray(centers)
    accs = np.asarray(accs)
    confs = np.asarray(confs)
    counts = np.asarray(counts)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    width = 1.0 / n_bins
    nonempty = counts > 0

    # Blue accuracy bars.
    ax.bar(
        centers[nonempty],
        accs[nonempty],
        width=width * 0.95,
        alpha=0.75,
        edgecolor="#1f4e79",
        color="#2e75b6",
        label="accuracy",
    )

    # Red "gap" bars sitting on top of accuracy up to confidence.
    if show_gap:
        gaps = confs - accs
        ax.bar(
            centers[nonempty],
            gaps[nonempty],
            bottom=accs[nonempty],
            width=width * 0.95,
            alpha=0.4,
            edgecolor="#c00000",
            color="#f08080",
            label="gap (conf − acc)",
        )

    # Ideal calibration line.
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")

    if show_ece:
        ece = float(expected_calibration_error(probs, labels, n_bins=n_bins))
        ax.text(
            0.05,
            0.95,
            f"ECE = {ece:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "gray"},
        )

    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")

    return fig, ax


def plot_coverage_curve(
    conformal_factory: Any,
    probs_cal: Any,
    labels_cal: Any,
    probs_test: Any,
    labels_test: Any,
    *,
    alphas: Any = None,
    ax: Axes | None = None,
    title: str = "Conformal coverage curve",
) -> tuple[Figure, Axes]:
    """Plot empirical coverage and mean set size vs target coverage.

    For each :math:`\\alpha` in ``alphas``, fits a fresh
    :class:`~bayes_hdc.uncertainty.ConformalClassifier` on
    ``(probs_cal, labels_cal)`` and reports the empirical coverage and
    mean set size on ``(probs_test, labels_test)``. The ideal line is
    :math:`1 - \\alpha`.

    Args:
        conformal_factory: A callable ``alpha -> ConformalClassifier``.
            In practice ``lambda a: ConformalClassifier.create(alpha=a)``.
        probs_cal, labels_cal: Calibration set.
        probs_test, labels_test: Test set.
        alphas: Iterable of miscoverage targets in ``(0, 1)``. Defaults
            to ``[0.05, 0.1, 0.15, 0.2, 0.25, 0.3]``.
        ax: Optional pre-existing axes.
        title: Plot title.
    """
    plt = _require_matplotlib()

    if alphas is None:
        alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    alphas_arr = np.asarray(alphas, dtype=np.float64)

    coverages = []
    set_sizes = []
    for alpha in alphas_arr:
        conformal = conformal_factory(float(alpha)).fit(probs_cal, labels_cal)
        coverages.append(float(conformal.coverage(probs_test, labels_test)))
        set_sizes.append(float(conformal.set_size(probs_test)))

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    target = 1.0 - alphas_arr

    ax.plot(target, coverages, marker="o", label="empirical coverage", color="#2e75b6")
    ax.plot(target, target, linestyle="--", color="gray", label=r"target $1 - \alpha$")
    ax.set_xlabel(r"target coverage $1 - \alpha$")
    ax.set_ylabel("empirical coverage")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(
        target,
        set_sizes,
        marker="s",
        color="#c00000",
        label="mean set size",
    )
    ax2.set_ylabel("mean set size", color="#c00000")
    ax2.tick_params(axis="y", colors="#c00000")

    # Combine legends.
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right")

    return fig, ax


__all__ = [
    "plot_reliability_diagram",
    "plot_coverage_curve",
]
