.DEFAULT_GOAL := help

PYTHON_DIRS := nanochat scripts tasks dev tests

.PHONY: help format check test

help: # Show available commands
	@grep -E '^[a-zA-Z0-9_-]+:.*#' Makefile | sort | awk 'BEGIN {FS = ":.*#"} {printf "%-10s %s\n", $$1, $$2}'

format: # Format Python code with Ruff
	uv run --group dev ruff format $(PYTHON_DIRS)

check: # Lint and type check Python code
	uv run --group dev ruff check $(PYTHON_DIRS)
	uv run --group dev ty check $(PYTHON_DIRS)

test: # Run the test suite
	uv run --group dev python -m pytest
