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

# Run the TSP (graph-based) signature extraction for all timm models
tsp-extraction:
    uv run scripts/tsp_extraction.py

# Compute pairwise TDA distances for a dataset (bottleneck by default)
tda-compare DATASET:
    uv run scripts/tda_comparison.py dataset={{DATASET}}

# Compute pairwise TDA distances with a specific distance metric (bottleneck | wasserstein | hausdorff | betti_curve)
tda-compare-distance DATASET DISTANCE:
    uv run scripts/tda_comparison.py dataset={{DATASET}} comparison.distance={{DISTANCE}}

# Plot persistence diagrams and images for all models
tda-diagrams:
    uv run scripts/tda.py

# Plot persistence diagrams and images for a specific dataset
tda-diagrams-dataset DATASET:
    uv run scripts/tda.py dataset={{DATASET}}

# Plot TDA model distance graph (uses tda_plot config defaults)
tda-graph:
    uv run scripts/tda_plot.py

# Plot TDA distance graph for a specific dataset
tda-graph-dataset DATASET:
    uv run scripts/tda_plot.py dataset={{DATASET}}

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

# Run stat analysis (loads precomputed metrics, runs regression)
stat:
    uv run scripts/stat_analysis.py

# Plot statistical regression results
stat-plot:
    uv run scripts/plot_stat.py

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

# Run marimo notebooks
marimo:
    uv run marimo edit .

# Run prototype comparison between two models
proto-compare:
    uv run scripts/prototype_comparison.py

# Run prototype comparison with custom dataset
proto-compare-dataset DATASET:
    uv run scripts/prototype_comparison.py dataset={{DATASET}}

# Run prototype comparison with custom models
proto-compare-models MODEL_A MODEL_B:
    uv run scripts/prototype_comparison.py model_a={{MODEL_A}} model_b={{MODEL_B}}

# Run prototype alignment (lstsq probing on transmitted A → B raw space)
proto-alignment:
    uv run scripts/proto_alignment.py

# Run prototype alignment on a specific dataset
proto-alignment-dataset DATASET:
    uv run scripts/proto_alignment.py dataset={{DATASET}}

# Run prototype alignment with custom models
proto-alignment-models MODEL_A MODEL_B:
    uv run scripts/proto_alignment.py model_a={{MODEL_A}} model_b={{MODEL_B}}

# Run semantic alignment evaluation across model pairs and datasets
alignment:
    uv run scripts/alignment.py

# Plot alignment metrics vs compression ratio (k) for all datasets and methods
alignment-plot:
    uv run scripts/plot_alignment.py

# Plot PC correlation heatmap between two models on a dataset
plot-heatmap MODEL_A MODEL_B DATASET:
    uv run scripts/plot_correlation_heatmap.py {{MODEL_A}} {{MODEL_B}} {{DATASET}}
