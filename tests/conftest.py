# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Shared pytest fixtures for JAX-HDC tests."""

import jax
import pytest


@pytest.fixture
def prng_key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def default_dimensions():
    return 100
