# Run static analysis and automatically fix issues where possible
check:
    uvx ruff check . --fix

# Format code according to project style
format:
    uvx ruff format .

# Run formatting and linting (CI-style target)
all: format check

# Generate or Update all readme on HF (Update yaml)
readme:
    uv run scripts/generate_readme.py

# Push models of a dataset to HF & update readme
push DATASET:
    uv run scripts/push_to_hf.py dataset={{DATASET}}
    just readme
