<!-- Thanks for contributing to bayes-hdc. Fill in the sections below. -->

## Summary

<!-- One paragraph: what does this PR do and why? -->

## Type of change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that changes existing public API)
- [ ] Documentation only
- [ ] Tooling / CI / refactor

## Checklist

- [ ] `pytest tests/` passes locally.
- [ ] `ruff check bayes_hdc/ tests/ examples/ benchmarks/` is clean.
- [ ] `ruff format --check` is clean.
- [ ] `mypy bayes_hdc/` is clean.
- [ ] New code has unit tests; coverage stays at >= 99%.
- [ ] New public API has docstrings in Google style.
- [ ] `CHANGELOG.md` updated under `[Unreleased]`.
- [ ] If a new primitive: it is `jit`- and `vmap`-compatible and returns a JAX array.

## Related issues

<!-- e.g. Closes #123, refs #45 -->
