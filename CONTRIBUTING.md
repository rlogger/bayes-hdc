# Contributing to JAX-HDC

Thank you for your interest in contributing to JAX-HDC.

## Development Setup

```bash
git clone https://github.com/rlogger/jax-hdc.git
cd jax-hdc
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Code Style

We use `ruff` for linting and formatting (line length: 100, target: Python 3.9+):

```bash
ruff check jax_hdc/ tests/
ruff format jax_hdc/ tests/
mypy jax_hdc/
```

### Docstrings

Use Google-style docstrings:

```python
def function_name(arg1: type, arg2: type) -> return_type:
    """Brief description.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value
    """
```

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=jax_hdc --cov-report=term-missing
```

Add tests for all new functionality in the `tests/` directory.

## Functional Programming Principles

JAX-HDC follows functional programming principles:

1. **Pure functions**: No side effects
2. **Immutability**: Use `.at[]` syntax for updates, return new instances
3. **Explicit state**: Pass state as arguments
4. **JIT-compatible**: Avoid Python control flow in JIT functions

## Pull Request Process

1. Create a feature branch from `main`.
2. Add tests for new functionality. Target **≥ 99% line coverage** on touched files.
3. Ensure `ruff check`, `ruff format --check`, `mypy`, and `pytest` all pass locally.
4. Update `CHANGELOG.md` under the `[Unreleased]` section with a one-line entry.
5. If adding a new primitive, confirm it is `jit`- and `vmap`-compatible and returns a `jax.Array`.
6. Open a PR referencing any related issues. The PR template is pre-filled.

CI enforces the same checks across Ubuntu, macOS, and Windows on Python 3.9–3.13.

## Review expectations

- A maintainer will respond within 7 days.
- Tests must pass on all CI jobs before merge.
- PRs that add a public API require a docstring, a unit test, and a `CHANGELOG.md` entry.
- PRs that change a public API require a deprecation path unless the release is a pre-1.0 breaking change, in which case call it out explicitly in the PR description.

## Release process

Releases follow [Semantic Versioning](https://semver.org).

1. Update `CHANGELOG.md`: move `[Unreleased]` entries under a new `[X.Y.Z] - YYYY-MM-DD` heading.
2. Bump `__version__` in `jax_hdc/__init__.py` and `version` in `pyproject.toml`.
3. Bump `version` and `date-released` in `CITATION.cff`.
4. Tag: `git tag -s vX.Y.Z -m "Release vX.Y.Z"`.
5. Push: `git push origin main --tags`. The `publish.yml` workflow builds the wheel and uploads to TestPyPI, then PyPI.
6. Create a GitHub release referencing the changelog.

## Governance

The project is currently maintained by a single maintainer (see `CITATION.cff`).
Active contributors who land three substantive merged pull requests
will be invited to become maintainers with commit access.

## Security

See [`SECURITY.md`](SECURITY.md) for vulnerability-reporting procedure.
Please do **not** file security issues in the public tracker.

## Code of Conduct

Participation is governed by the [Contributor Covenant](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
Each new source file must carry the SPDX header:

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh
```
