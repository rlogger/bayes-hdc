# SPDX-License-Identifier: MIT
# Copyright (c) 2026 bayes-hdc contributors
"""Regression tests for lossless labels and safe dataset cache behavior."""

import urllib.request

import numpy as np
import pytest

from bayes_hdc.datasets import loaders


def test_labels_do_not_merge_after_int32_overflow_or_float_truncation():
    for labels in (np.array([0, 2**32, 0, 2**32]), np.array([0.1, 0.9, 0.1, 0.9])):
        np.testing.assert_array_equal(loaders._normalise_labels(labels), [0, 1, 0, 1])


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
@pytest.mark.parametrize("loader", [loaders.load_mnist, loaders.load_fashion_mnist])
def test_invalid_subsample_rejected_before_network(monkeypatch, value, loader):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid input attempted a download")

    monkeypatch.setattr(loaders, "_fetch_openml_cached", unexpected)
    with pytest.raises(ValueError, match="subsample"):
        loader(subsample=value)


@pytest.mark.parametrize(
    "kwargs",
    [{"window": 0}, {"window": -1}, {"subjects": ()}, {"subjects": (0,)}, {"subjects": (1, 1)}],
)
def test_invalid_emg_request_never_downloads(monkeypatch, kwargs):
    monkeypatch.setattr(loaders, "_cache_dir", lambda: pytest.fail("touched cache"))
    with pytest.raises(ValueError):
        loaders.load_emg(**kwargs)


def test_interrupted_download_does_not_poison_cache(monkeypatch, tmp_path):
    target = tmp_path / "data.zip"

    def partial(url, path):
        from pathlib import Path

        Path(path).write_bytes(b"truncated")
        raise OSError("connection dropped")

    monkeypatch.setattr(urllib.request, "urlretrieve", partial)
    with pytest.raises(OSError, match="connection dropped"):
        loaders._download_atomic("https://example.invalid/data.zip", target)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_openml_numeric_id_is_not_used_as_name(monkeypatch):
    from types import SimpleNamespace

    seen = {}

    def fetch(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(data=np.zeros((4, 2)), target=np.zeros(4))

    monkeypatch.setattr(
        loaders, "_import_sklearn", lambda: (SimpleNamespace(fetch_openml=fetch), None)
    )
    loaders._fetch_openml_cached(554)
    assert seen["data_id"] == 554
    assert "name" not in seen
    assert "version" not in seen


def test_failed_derived_archive_preserves_existing_cache(monkeypatch, tmp_path):
    target = tmp_path / "data.npz"
    target.write_bytes(b"existing")

    def partial(output, **arrays):
        output.write(b"truncated")
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", partial)
    with pytest.raises(OSError, match="disk full"):
        loaders._savez_atomic(target, X=np.zeros((3, 2)))
    assert target.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [target]
