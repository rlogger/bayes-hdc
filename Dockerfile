# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh
#
# Multi-stage Dockerfile for bayes-hdc.
#
# Stages:
#   base       — minimal runtime image with the library installed
#   dev        — base + dev tools (pytest, ruff, mypy)
#   benchmark  — base + benchmark extras (sklearn, torchhd, matplotlib)
#                with the benchmark scripts pre-bundled
#   runtime    — thin default image; prints library version on run

FROM python:3.12-slim AS base

WORKDIR /app

# System deps for JAX + scientific Python.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ORIGINALITY.md CITATION.cff ./
COPY bayes_hdc/ bayes_hdc/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# --------- dev image: pytest / ruff / mypy ---------
FROM base AS dev

COPY tests/ tests/
COPY examples/ examples/
COPY docs/ docs/
COPY Makefile ./

RUN pip install --no-cache-dir ".[dev,docs,examples]"

CMD ["pytest", "tests/", "-v", "-k", "not benchmark", "-m", "not network"]

# --------- benchmark image: reproducible head-to-head vs baselines ---------
FROM base AS benchmark

COPY benchmarks/ benchmarks/
COPY tests/ tests/

# scikit-learn + torchhd + matplotlib for plots, plus every optional extras group.
RUN pip install --no-cache-dir ".[benchmark,examples,datasets]"

# Mount a volume here to persist benchmark_*_results.json and figures/
VOLUME ["/app/benchmarks/results"]

ENV PYTHONUNBUFFERED=1

# Default: run every benchmark end-to-end and dump to /app/benchmarks/results.
CMD ["bash", "-c", "\
    set -e && \
    mkdir -p benchmarks/results && \
    python benchmarks/benchmark_calibration.py \
        --output benchmarks/results/calibration.json && \
    python benchmarks/benchmark_selective.py \
        --output benchmarks/results/selective.json && \
    python benchmarks/benchmark_ood.py \
        --output benchmarks/results/ood.json && \
    python benchmarks/generate_figures.py \
        --results benchmarks/results/calibration.json && \
    cp -r benchmarks/figures benchmarks/results/figures || true && \
    echo 'All benchmarks complete — results in benchmarks/results/'"]

# --------- minimal runtime image ---------
FROM base AS runtime

CMD ["python", "-c", "import bayes_hdc; print(f'bayes-hdc {bayes_hdc.__version__} ready')"]
