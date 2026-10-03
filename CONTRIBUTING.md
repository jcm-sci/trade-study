# Contributing

Small, focused contributions are welcome. Check the open issues before starting;
for a substantial API change, discuss the intended behavior in an issue first.

## Set up

Install Python 3.10 or newer, [uv](https://docs.astral.sh/uv/), and
[just](https://just.systems/). Clone the repository and install the development
extras:

```bash
git clone https://github.com/jcm-sci/trade-study.git
cd trade-study
uv sync --extra dev
```

`uv` creates a project-local `.venv`; you do not need to activate it when using
`uv run` or the `just` recipes. If working from conda, keep the project dependencies
in this uv environment rather than installing them into the conda base environment.
`uv.lock` is currently ignored, so CI resolves dependency versions from
`pyproject.toml` rather than from a committed lockfile.

## Make a change

Create a branch from current `main` and keep each PR focused on one issue or
coherent behavior change. Add regression tests for bug fixes and meaningful tests
for new behavior. Update the relevant documentation and the `Unreleased` section
of `CHANGELOG.md` for user-visible changes.

```bash
git switch -c fix/issue-number-short-description
just format
just check
just ci
```

`just format` formats source and tests; `just check` applies available lint fixes.
Review the resulting diff. **Run `just ci` before every commit.** It checks
formatting and lint, strict typing, and the full test suite with coverage. The
current coverage gate is 40%; avoid reducing the existing coverage.

For focused iteration:

```bash
uv run pytest tests/test_runner.py
uv run mypy --strict src
uv run mkdocs build --strict
```

The complete `just ci` gate still needs to pass before committing. For release or
packaging changes, also run `just build-all`, which builds and checks the artifacts
and runs tests against a wheel installed in a fresh environment.

## Code standards

- Ruff uses `select = ["ALL"]` and preview rules with the minimal exceptions in
  `pyproject.toml`. Fix a lint issue rather than suppressing it; any unavoidable
  exception needs an architectural explanation.
- Library code must pass `mypy --strict`. Use precise types and
  `from __future__ import annotations`; move annotation-only imports under
  `if TYPE_CHECKING:` where appropriate.
- Use Google-style docstrings. Public APIs document parameters, return values,
  and exceptions, and explain assumptions that affect scientific interpretation.
- Keep NumPy as the only core dependency. Put integrations in optional extras and
  defer their imports so users can install only the features they need.
- Tests live in `tests/test_*.py`. Prefer deterministic seeds and behavioral
  assertions that would fail for the original bug.

## Submit a PR

Use descriptive commit messages. Explain the concrete problem, resulting behavior,
and validation in the PR description. Link the issue with `Closes #NUMBER` when
it is fully addressed. List compatibility changes or limitations that reviewers
need to assess. Keep unrelated cleanup and new features in separate PRs.

All applicable GitHub checks must pass before merge. Branch protection may also
require an approving review. After merge, remove the feature branch locally and
on the remote; preserve any unmerged work.
