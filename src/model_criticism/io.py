"""Save and load study results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .protocols import ResultsTable


def save_results(results: ResultsTable, path: str | Path) -> None:
    """Save a ResultsTable to disk.

    Uses .npz for score arrays and .json for configs/metadata.

    Args:
        results: The ResultsTable to save.
        path: Directory to write into (created if needed).
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(p / "scores.npz", scores=results.scores)
    if results.annotations is not None:
        np.savez_compressed(p / "annotations.npz", annotations=results.annotations)

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
    """
    p = Path(path)

    scores = np.load(p / "scores.npz")["scores"]

    annotations = None
    ann_path = p / "annotations.npz"
    if ann_path.exists():
        annotations = np.load(ann_path)["annotations"]

    with (p / "meta.json").open() as f:
        meta: dict[str, Any] = json.load(f)

    return ResultsTable(
        configs=meta["configs"],
        scores=scores,
        observable_names=meta["observable_names"],
        annotations=annotations,
        annotation_names=meta.get("annotation_names", []),
        metadata=meta.get("metadata", []),
    )
