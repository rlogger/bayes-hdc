# SPDX-License-Identifier: MIT
# Copyright (c) 2026 bayes-hdc contributors
"""Audit the experimental protocol, independently of headline metric values."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def calibration(monkeypatch):
    # Scripts are intentionally importable without installing benchmarks in the wheel.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("benchmark_calibration")


def test_fit_preprocessing_is_invariant_to_heldout_outliers(calibration):
    train = np.array([[0.0], [2.0]])
    fit, held = calibration.standardize_train(train, np.array([[1000.0]]))
    np.testing.assert_allclose(fit[:, 0], [-1, 1])
    np.testing.assert_allclose(held[:, 0], [999])


def test_calibration_and_temperature_are_disjoint_and_test_label_blind(calibration):
    labels = np.arange(100) % 2
    ids = calibration.split_indices(labels, 4)
    assert len(set(np.concatenate(ids))) == 100
    changed = labels.copy()
    changed[ids[-1]] = 1 - changed[ids[-1]]
    for first, second in zip(ids, calibration.split_indices(changed, 4)):
        np.testing.assert_array_equal(first, second)


def test_saved_predictions_reconstruct_metrics_and_match_torch(calibration):
    import jax.numpy as jnp

    from bayes_hdc import ConformalClassifier

    rng = np.random.default_rng(4)
    X = rng.normal(size=(100, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int32)
    spec = calibration.DatasetSpec("test", 100, 3, 2)
    row = calibration.run_dataset(spec, X, y, dimensions=32, levels=4)
    result = row["bayes_hdc"]
    data = result["predictions"]
    conf = ConformalClassifier.create(alpha=0.1).fit(
        jnp.array(data["probs_cal"]), jnp.array(data["y_cal"])
    )
    covered = float(conf.coverage(jnp.array(data["probs_test"]), jnp.array(data["y_test"])))
    assert covered == result["conformal"]["coverage"]
    if row["torchhd"] is not None:
        assert row["torchhd"]["raw"]["accuracy"] == result["raw"]["accuracy"]


def test_figures_reject_historical_results_instead_of_rerunning(calibration, tmp_path):
    figures = importlib.import_module("generate_figures")
    source = tmp_path / "old.json"
    source.write_text(json.dumps([{"bayes_hdc": {"raw": {"accuracy": 0.99}}}]))
    with pytest.raises(ValueError, match="Rerun"):
        figures._read_results(source)


def test_undefined_statistics_are_portable_json(calibration, tmp_path):
    path = tmp_path / "result.json"
    calibration.write_json(path, {"accuracy": float("nan"), "score": float("inf")})
    assert json.loads(path.read_text()) == {"accuracy": None, "score": None}
