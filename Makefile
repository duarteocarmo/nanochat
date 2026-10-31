.DEFAULT_GOAL := help

.PHONY: help install-cpu install-gpu format check test

help: # Show available commands
	@grep -E '^[a-zA-Z0-9_-]+:.*#' Makefile | sort | awk 'BEGIN {FS = ":.*#"} {printf "%-10s %s\n", $$1, $$2}'

install-cpu: # Install CPU dependencies
	uv sync --locked --extra cpu

install-gpu: # Install CUDA GPU dependencies
	uv sync --locked --extra gpu

format: # Format Python code with Ruff
	uv run --group dev ruff format .

check: # Lint and type check Python code
	uv run --group dev ruff check .
	uv run --group dev ty check .

test: # Run the test suite
	uv run --group dev python -m pytest
