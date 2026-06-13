# AGENTS.md

## Cursor Cloud specific instructions

`bayes-hdc` is a pure-Python JAX library (no service, server, database, or GUI). "Running the
app" means installing it editable and exercising it via tests, examples, or a short script.
Standard commands live in the `Makefile`, `pyproject.toml`, and `README.md`; prefer those.

Non-obvious gotchas for this environment:

- **Console scripts are not on `PATH`.** `pip install` places `pytest`, `ruff`, and `mypy` in
  `~/.local/bin`, which is not on `PATH`. Invoke them as modules instead:
  `python3 -m pytest`, `python3 -m ruff`, `python3 -m mypy`. The `Makefile` `lint`/`format`/
  `typecheck` targets call `ruff`/`mypy` directly and will fail with "command not found"; run
  the module form or add `~/.local/bin` to `PATH` first. `make test` works as-is because it
  uses `python3 -m pytest`.
- **Default test run is offline and slow.** `make test` (i.e.
  `python3 -m pytest tests/ -k "not benchmark" -m "not network"`) skips network/benchmark
  tests and takes ~3 minutes (665+ tests). Network-gated dataset tests are skipped by default;
  run them with `-m network` only when internet + scikit-learn are available.
- **JAX is CPU-only here** and emits a harmless "no GPU/TPU found" message on import; ignore it.
- **Real-data sanity check:** `bayes_hdc.sklearn.HDClassifier(encoder="kernel")` on iris gives
  ~0.9 test accuracy. Some committed `examples/` (e.g. `classification_simple.py`) use random
  synthetic data, so their low accuracy is expected and not a setup failure.
