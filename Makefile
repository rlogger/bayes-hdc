.PHONY: help install install-dev install-all test test-all lint format typecheck \
        docs docs-serve build publish publish-test clean \
        bench bench-docker docker-build docker-test docker-bench figures

PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	$(PIP) install -e .

install-dev: ## Install with dev dependencies
	$(PIP) install -e ".[dev]"

install-all: ## Install with all optional dependencies
	$(PIP) install -e ".[dev,docs,examples,benchmark,datasets]"

test: ## Run tests with coverage (skips network-gated tests)
	$(PYTHON) -m pytest tests/ -v --cov=bayes_hdc --cov-report=term-missing \
		-k "not benchmark" -m "not network"

test-all: ## Run all tests including benchmarks and network
	$(PYTHON) -m pytest tests/ -v --cov=bayes_hdc --cov-report=term-missing

lint: ## Run linter
	ruff check bayes_hdc/ tests/ examples/ benchmarks/

format: ## Auto-format code
	ruff format bayes_hdc/ tests/ examples/ benchmarks/
	ruff check --fix bayes_hdc/ tests/ examples/ benchmarks/

typecheck: ## Run type checker
	mypy bayes_hdc/ --ignore-missing-imports

docs: ## Build Sphinx documentation
	$(PYTHON) -m sphinx -b html docs/ docs/_build/html

docs-serve: docs ## Build and open docs in browser
	$(PYTHON) -m webbrowser docs/_build/html/index.html

bench: ## Run all calibration / selective / OOD benchmarks locally
	$(PYTHON) benchmarks/benchmark_calibration.py
	$(PYTHON) benchmarks/benchmark_selective.py
	$(PYTHON) benchmarks/benchmark_ood.py

figures: bench ## Generate paper figures from benchmark results
	$(PYTHON) benchmarks/generate_figures.py

docker-build: ## Build the benchmark Docker image
	docker build --target benchmark -t bayes-hdc:benchmark .

docker-test: ## Run the dev Docker image (= pytest)
	docker build --target dev -t bayes-hdc:dev .
	docker run --rm bayes-hdc:dev

docker-bench: docker-build ## Run the benchmark Docker image with persisted results volume
	mkdir -p $$PWD/benchmarks/results
	docker run --rm \
		-v $$PWD/benchmarks/results:/app/benchmarks/results \
		bayes-hdc:benchmark

build: clean ## Build sdist and wheel
	$(PIP) install --upgrade build
	$(PYTHON) -m build

publish-test: build ## Publish to TestPyPI
	$(PIP) install --upgrade twine
	$(PYTHON) -m twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	$(PIP) install --upgrade twine
	$(PYTHON) -m twine upload dist/*

clean: ## Remove build artifacts
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
