set shell := ["bash", "-uc"]

default: lint test

# ── Formatting / linting ─────────────────────────────────────────────

# Auto-format all Python sources.
format:
	uv run ruff format --preview src tests examples/replicated_study.py

# Lint-fix all Python sources.
check:
	uv run ruff check --preview --fix src tests examples/replicated_study.py

# Check formatting + lint without modifying files (matches CI).
lint:
	uv run ruff format --preview --check src tests examples/replicated_study.py
	uv run ruff check --preview src tests examples/replicated_study.py

# ── Tests ────────────────────────────────────────────────────────────

# Run the test suite.
test:
	uv run pytest

# Run tests with coverage report.
coverage:
	uv run --extra test pytest --cov --cov-report=term-missing --cov-fail-under=40

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
	uv build --wheel --out-dir "$CLEANROOM/dist"
	uv pip install --python "$CLEANROOM/venv/bin/python" --no-cache "$CLEANROOM/dist/$(ls "$CLEANROOM/dist/" | grep '\.whl$')[test]"
	cp -R tests examples "$CLEANROOM/"
	cp README.md "$CLEANROOM/README.md"
	cd "$CLEANROOM"
	unset PYTHONPATH
	"$CLEANROOM/venv/bin/python" -c 'import trade_study; assert "site-packages" in trade_study.__file__'
	"$CLEANROOM/venv/bin/python" -m pytest --doctest-modules --pyargs trade_study tests -q -x
	echo "Clean-room test passed."

# Full pre-publish dry run.
build-all: build-check build-test

# ── Utilities ────────────────────────────────────────────────────────

# Remove generated/cached artifacts.
clean:
	rm -f uv.lock
	rm -rf .venv dist .*_cache

# API examples plus the actual complete README and standalone study.
doctest:
	uv run pytest --doctest-modules src/trade_study tests/test_documentation.py

example-replicates:
	uv run python examples/replicated_study.py
