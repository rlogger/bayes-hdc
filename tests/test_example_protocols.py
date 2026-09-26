"""Guard data-generation and preprocessing boundaries in runnable examples."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _example(name):
    path = Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"example_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "name", ["activity_recognition", "eeg_seizure_detection", "emg_gesture_recognition"]
)
def test_example_quantisation_is_independent_of_other_test_rows(name):
    module = _example(name)
    training = np.arange(24, dtype=float).reshape(12, 2)
    query = np.array([[4.0, 9.0], [18.0, 3.0]])
    # Extreme future values must not move previously fitted bin boundaries.
    augmented = np.vstack([query, [[-1e6, 1e6], [1e6, -1e6]]])
    alone = module._discretise(query, 4, reference=training)
    together = module._discretise(augmented, 4, reference=training)
    np.testing.assert_array_equal(alone, together[: len(query)])
    assert np.all((together >= 0) & (together < 4))


def test_intrusion_normal_splits_share_the_same_population_covariance(monkeypatch):
    module = _example("anomaly_detection_intrusion")
    for name in ("N_TRAIN", "N_CAL", "N_TEST_NORMAL"):
        monkeypatch.setattr(module, name, 6000)
    train, calibration, test, labels, _ = module.synthesise_dataset(2026)
    covariance = np.cov(train, rowvar=False)
    for normal_split in (calibration, test[labels == 0]):
        relative_difference = np.linalg.norm(np.cov(normal_split, rowvar=False) - covariance)
        relative_difference /= np.linalg.norm(covariance)
        # Independent Monte Carlo samples from a fixed ten-dimensional mixture.
        # Redrawing its loading matrix per split creates a much larger mismatch.
        assert relative_difference < 0.12
