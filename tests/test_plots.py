# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Tests for bayes_hdc.plots.

Matplotlib is an optional dependency — these tests skip cleanly if
it is not installed in the test environment.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")  # headless backend for CI

from bayes_hdc.plots import plot_coverage_curve, plot_reliability_diagram  # noqa: E402
from bayes_hdc.uncertainty import ConformalClassifier  # noqa: E402


def _random_probs(key: jax.Array, n: int, k: int) -> jax.Array:
    alphas = jnp.ones(k)
    return jax.random.dirichlet(key, alphas, shape=(n,))


# ----------------------------------------------------------------------
# plot_reliability_diagram
# ----------------------------------------------------------------------


def test_reliability_returns_figure_and_axes() -> None:
    probs = _random_probs(jax.random.PRNGKey(0), 100, 5)
    labels = jax.random.randint(jax.random.PRNGKey(1), (100,), 0, 5)
    fig, ax = plot_reliability_diagram(probs, labels, n_bins=10)
    assert fig is not None
    assert ax is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_reliability_with_custom_axes() -> None:
    import matplotlib.pyplot as plt

    probs = _random_probs(jax.random.PRNGKey(2), 100, 3)
    labels = jax.random.randint(jax.random.PRNGKey(3), (100,), 0, 3)
    fig, ax = plt.subplots()
    out_fig, out_ax = plot_reliability_diagram(probs, labels, ax=ax)
    assert out_ax is ax
    assert out_fig is fig
    plt.close(fig)


def test_reliability_without_ece_label() -> None:
    import matplotlib.pyplot as plt

    probs = _random_probs(jax.random.PRNGKey(4), 50, 2)
    labels = jax.random.randint(jax.random.PRNGKey(5), (50,), 0, 2)
    fig, ax = plot_reliability_diagram(probs, labels, show_ece=False)
    # No text widgets with "ECE =" prefix should exist.
    texts = [t.get_text() for t in ax.texts]
    assert not any("ECE" in t for t in texts)
    plt.close(fig)


def test_reliability_has_perfect_calibration_line() -> None:
    import matplotlib.pyplot as plt

    probs = _random_probs(jax.random.PRNGKey(6), 50, 3)
    labels = jax.random.randint(jax.random.PRNGKey(7), (50,), 0, 3)
    fig, ax = plot_reliability_diagram(probs, labels)
    # Legend should include the perfect-calibration entry.
    labels_legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("perfect" in lbl for lbl in labels_legend)
    plt.close(fig)


# ----------------------------------------------------------------------
# plot_coverage_curve
# ----------------------------------------------------------------------


def test_coverage_curve_runs_end_to_end() -> None:
    import matplotlib.pyplot as plt

    cal_key, test_key = jax.random.split(jax.random.PRNGKey(10))
    probs_cal = _random_probs(cal_key, 200, 4)
    labels_cal = jax.random.randint(jax.random.PRNGKey(11), (200,), 0, 4)
    probs_te = _random_probs(test_key, 100, 4)
    labels_te = jax.random.randint(jax.random.PRNGKey(12), (100,), 0, 4)

    def factory(alpha: float) -> ConformalClassifier:
        return ConformalClassifier.create(alpha=alpha)

    fig, ax = plot_coverage_curve(
        factory,
        probs_cal,
        labels_cal,
        probs_te,
        labels_te,
    )
    assert fig is not None
    assert ax is not None
    plt.close(fig)


def test_coverage_curve_custom_alphas() -> None:
    import matplotlib.pyplot as plt

    cal_key, test_key = jax.random.split(jax.random.PRNGKey(20))
    probs_cal = _random_probs(cal_key, 300, 3)
    labels_cal = jax.random.randint(jax.random.PRNGKey(21), (300,), 0, 3)
    probs_te = _random_probs(test_key, 150, 3)
    labels_te = jax.random.randint(jax.random.PRNGKey(22), (150,), 0, 3)

    fig, ax = plot_coverage_curve(
        lambda a: ConformalClassifier.create(alpha=a),
        probs_cal,
        labels_cal,
        probs_te,
        labels_te,
        alphas=[0.1, 0.2],
    )
    # Custom alphas respected — the primary axis should have 2 x-points.
    # (Direct inspection of the marker positions.)
    line = ax.get_lines()[0]  # empirical coverage
    assert len(line.get_xdata()) == 2
    plt.close(fig)
