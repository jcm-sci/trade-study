# jcm-sci Python standards

## Workflow

Run `just ci` before every commit. CI runs three gates in order:

1. **Lint** — `ruff format --preview --check` then `ruff check --preview --no-fix`
2. **Type-check** — `mypy --strict src`
3. **Test** — `pytest --cov --cov-report=term-missing --cov-fail-under=<threshold>`

All three must pass with zero errors. Do not commit with warnings or suppressions unless architecturally justified (see below).

Use `just format` to auto-format and `just check` to auto-fix lint issues before running `just ci`.

## Ruff

All repos use `select = ["ALL"]` with a minimal ignore list:

```toml
[tool.ruff.lint]
select = ["ALL"]
ignore = ["COM812", "CPY001", "D203", "D212", "PLR2004"]
```

### Fix rules, do not suppress them

- **Do not add `noqa` comments** unless the rule is structurally impossible to satisfy.
- Acceptable project-level ignores: `PLC0415` (deferred imports for optional dependencies), `ANN401` on Protocol files where `Any` is semantically correct.
- Acceptable per-file ignores: `tests/**/*` gets `INP001`, `S101`.

### Common fixes

| Rule | Fix |
|------|-----|
| TC001/TC002/TC003 | Move import into `if TYPE_CHECKING:` block |
| N806 | Rename uppercase variable to lowercase |
| PLR0911 | Refactor to dispatch dict or early-return pattern |
| PLR6104 | Use augmented assignment (`x /= y`) |
| PLR6201 | Use set literal for membership tests (`in {a, b}`) |
| B905 | Add explicit `strict=True` or `strict=False` to `zip()` |
| DOC201/DOC501 | Add `Returns:` / `Raises:` sections to docstring |
| ANN401 | Replace `Any` with a concrete type or `Protocol` |
| E501 | Break long lines; refactor complex conditionals |

## mypy

All repos use `strict = true`. Guidelines:

- Use `TYPE_CHECKING` blocks for annotation-only imports (numpy.typing, Protocol types, etc.).
- Place `# type: ignore[import-untyped]` on the `from` line of untyped third-party imports, not on the imported name. mypy only reports the error once per module — only the first import needs the comment.
- Do not add `# type: ignore` for errors that can be fixed with proper typing.

## Docstrings

Google convention (`[tool.ruff.lint.pydocstyle] convention = "google"`).

Every public function and class needs a docstring with:
- Summary line
- `Args:` section (if parameters exist)
- `Returns:` section (if non-None return)
- `Raises:` section (if exceptions are raised)

Magic methods (`__post_init__`, etc.) also need docstrings under `select = ["ALL"]`.

## Dependencies

- Use `uv` for dependency management and virtual environments.
- Optional dependencies go in `[project.optional-dependencies]` extras.
- Deferred imports (`if TYPE_CHECKING` or inside functions) are the pattern for optional deps — this is why `PLC0415` is project-level ignored.

## Testing

- Tests live in `tests/` with `test_` prefix.
- Use `pytest` with `--cov` for coverage.
- Coverage threshold is set per-project in the justfile — raise it as test coverage improves.
