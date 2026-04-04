set shell := ["bash", "-uc"]

default: lint test

# ── Formatting / linting ─────────────────────────────────────────────

# Auto-format all Python sources.
format:
	uvx ruff format --preview

# Lint-fix all Python sources.
check:
	uvx ruff check --preview --fix

# Check formatting + lint without modifying files.
lint:
	uvx ruff format --preview --check
	uvx ruff check --preview --no-fix

# ── Tests ────────────────────────────────────────────────────────────

# Run the test suite.
test:
	uv run pytest

# Run tests with coverage report.
coverage:
	uv run --extra test pytest --cov --cov-report=term-missing --cov-fail-under=80

# ── Type checking ────────────────────────────────────────────────────

# Type-check library code (strict mode).
mypy:
	uv run --extra dev mypy --strict src

# ── CI aggregate ─────────────────────────────────────────────────────

# Run lint, typecheck, and tests with coverage.
ci: lint mypy coverage

# ── Build & publish ──────────────────────────────────────────────────

# Build sdist + wheel and check with twine.
build-check:
	rm -rf dist
	uv build
	uvx twine check --strict dist/*

# Clean-room install test: fresh venv, install wheel, run tests.
build-test:
	#!/usr/bin/env bash
	set -euo pipefail
	CLEANROOM=$(mktemp -d)
	trap 'rm -rf "$CLEANROOM"' EXIT
	uv venv "$CLEANROOM/venv"
	source "$CLEANROOM/venv/bin/activate"
	uv build --wheel --out-dir "$CLEANROOM/dist"
	uv pip install --no-cache "$CLEANROOM"/dist/*.whl
	uv pip install pytest
	python -m pytest tests -q -x
	echo "Clean-room test passed."

# Full pre-publish dry run.
build-all: build-check build-test

# ── Utilities ────────────────────────────────────────────────────────

# Remove generated/cached artifacts.
clean:
	rm -f uv.lock
	rm -rf .venv dist .*_cache
