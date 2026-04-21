# Setup the repo .venv via uv
setup:
    uv sync

# Run static analysis and automatically fix issues where possible
check:
    uvx ruff check . --fix

# Format code according to project style
format:
    uvx ruff format .

# Run formatting and linting (CI-style target)
clean: format check

# Generate or Update all readme on HF (Update yaml)
readme:
    uv run scripts/generate_readme.py

# Just encode a dataset for timm models
timm-encode DATASET:
    uv run scripts/encode_dataset_all_timm.py dataset={{DATASET}} hf.push=false

# Push models of a dataset to HF & update readme
push DATASET:
    uv run scripts/push_to_hf.py dataset={{DATASET}}
    just readme

# Push timm model metadata registry to HuggingFace
model-registry:
    uv run scripts/push_model_registry.py

# Run the tda signature extraction for all the timm models
tda-extraction:
    uv run scripts/tda_extraction.py

# Phase 1: Download all model latents (no computation)
compute-download:
    uv run scripts/compute_metrics.py download_only=true

# Phase 2: Compute metrics (stat + TDA) from cached latents
compute-compute:
    uv run scripts/compute_metrics.py download_only=false

# Compute stat metrics only (default)
compute-metrics DATASET:
    uv run scripts/compute_metrics.py dataset={{DATASET}}

# Compute stat + TDA metrics
compute-metrics-tda DATASET:
    uv run scripts/compute_metrics.py dataset={{DATASET}} compute_tda=true

# Compute metrics with limited models (for testing)
compute-metrics-test DATASET:
    uv run scripts/compute_metrics.py dataset={{DATASET}} limit_models=5

# Phase 1: Download all model latents (no computation)
stat-download:
    uv run scripts/stat_analysis.py download_only=true

# Phase 2: Compute entropy metrics from cached latents
stat-compute:
    uv run scripts/stat_analysis.py download_only=false

# Phase 3: Run regression on existing metrics
stat-regression:
    uv run scripts/stat_analysis.py regression_only=true

# Run stat analysis (full pipeline)
stat:
    just stat-download
    just stat-compute
    just stat-regression

# Install test dependencies
test-install:
    uv sync --extra test

# Run all tests
test:
    uv run pytest

# Run tests with coverage
test-coverage:
    uv run pytest --cov
    uv run coverage report

# Run tests matching a pattern
test-match PATTERN:
    uv run pytest -k {{PATTERN}}

# Run tests fast (no coverage)
test-fast:
    uv run pytest -v -m "not slow"

# Run only slow tests
test-slow:
    uv run pytest -v -m "slow"
