"""Public API surface audit.

Verifies that ``__all__`` is complete, sorted, and matches the actual
public symbols defined across all submodules.
"""

from __future__ import annotations

import importlib
import pkgutil

import trade_study


def _public_names(module_name: str) -> set[str]:
    """Return names defined in *module_name* that don't start with ``_``."""
    mod = importlib.import_module(module_name)
    names: set[str] = set()
    for name in dir(mod):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name)
        obj_module = getattr(obj, "__module__", None)
        if obj_module == module_name:
            names.add(name)
    return names


def _all_public_symbols() -> set[str]:
    """Collect every public symbol across trade_study submodules.

    Returns:
        Set of public symbol names found in all submodules.
    """
    symbols: set[str] = set()
    pkg_path = trade_study.__path__
    for info in pkgutil.walk_packages(pkg_path, prefix="trade_study."):
        if info.name.endswith(".egg-info"):
            continue
        symbols |= _public_names(info.name)
    return symbols


def test_all_is_sorted() -> None:
    """``__all__`` must be in sorted order."""
    assert trade_study.__all__ == sorted(trade_study.__all__)


def test_all_no_duplicates() -> None:
    """``__all__`` must not contain duplicates."""
    assert len(trade_study.__all__) == len(set(trade_study.__all__))


def test_all_covers_public_api() -> None:
    """Every public symbol in submodules must appear in ``__all__``."""
    expected = _all_public_symbols()
    exported = {name for name in trade_study.__all__ if name != "__version__"}
    missing = expected - exported
    assert not missing, f"Public symbols missing from __all__: {sorted(missing)}"


def test_all_contains_no_extras() -> None:
    """``__all__`` must not export names absent from submodules."""
    expected = _all_public_symbols()
    exported = {name for name in trade_study.__all__ if name != "__version__"}
    extra = exported - expected
    assert not extra, f"Symbols in __all__ not found in submodules: {sorted(extra)}"
