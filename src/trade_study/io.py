"""Save and load study results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .artifacts import load_result_bundle

if TYPE_CHECKING:
    from .protocols import ResultsTable


def save_results(results: ResultsTable, path: str | Path) -> None:
    """Save a ResultsTable to disk.

    Uses .npz for score arrays and .json for configs/metadata.
    Saving into an existing directory replaces the result files and removes
    previously saved annotations when the new table has none. Unrelated files
    are retained. The multi-file write is not an atomic publication operation.

    Args:
        results: The ResultsTable to save.
        path: Directory to write into (created if needed).

    Raises:
        ValueError: If the destination contains a versioned bundle.
    """
    p = Path(path)
    existing_meta = p / "meta.json"
    if existing_meta.exists():
        try:
            existing = json.loads(existing_meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Legacy exports can have interrupted metadata writes; retain
            # their established ability to replace an invalid legacy export.
            existing = None
        if isinstance(existing, dict) and "schema_version" in existing:
            msg = "Legacy export cannot overwrite a versioned result bundle"
            raise ValueError(msg)
    p.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(p / "scores.npz", scores=results.scores)
    if results.annotations is not None:
        np.savez_compressed(p / "annotations.npz", annotations=results.annotations)
    else:
        (p / "annotations.npz").unlink(missing_ok=True)

    meta = {
        "observable_names": results.observable_names,
        "annotation_names": results.annotation_names,
        "configs": results.configs,
        "metadata": results.metadata,
    }
    (p / "meta.json").write_text(json.dumps(meta, default=str, indent=2))


def load_results(path: str | Path) -> ResultsTable:
    """Load a ResultsTable from disk.

    Args:
        path: Directory previously written by ``save_results``.

    Returns:
        Reconstructed ResultsTable.

    Raises:
        ValueError: If parallel arrays/metadata or a versioned bundle are inconsistent.
    """
    try:
        return load_result_bundle(path).results
    except ValueError as error:
        msg = f"Cannot load study results: {error}"
        raise ValueError(msg) from error
