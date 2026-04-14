"""Tests for io module — save_results / load_results roundtrip (issue #24)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from trade_study.io import load_results, save_results
from trade_study.protocols import ResultsTable


@pytest.fixture
def results_with_annotations() -> ResultsTable:
    """ResultsTable with scores, annotations, and metadata.

    Returns:
        A populated ResultsTable.
    """
    return ResultsTable(
        configs=[{"alpha": 0.1, "method": "a"}, {"alpha": 0.9, "method": "b"}],
        scores=np.array([[0.4, 1.0], [0.1, 9.0]]),
        observable_names=["error", "cost"],
        annotations=np.array([[10.0], [20.0]]),
        annotation_names=["method_cost"],
        metadata=[{"wall_seconds": 0.01}, {"wall_seconds": 0.02}],
    )


@pytest.fixture
def results_without_annotations() -> ResultsTable:
    """ResultsTable with no annotations.

    Returns:
        A minimal ResultsTable.
    """
    return ResultsTable(
        configs=[{"x": 1}, {"x": 2}, {"x": 3}],
        scores=np.array([[0.5], [0.3], [0.7]]),
        observable_names=["metric"],
    )


# ---------------------------------------------------------------------------
# Roundtrip tests
# ---------------------------------------------------------------------------


def test_roundtrip_scores(
    tmp_path: Path,
    results_with_annotations: ResultsTable,
) -> None:
    save_results(results_with_annotations, tmp_path / "out")
    loaded = load_results(tmp_path / "out")
    np.testing.assert_allclose(loaded.scores, results_with_annotations.scores)


def test_roundtrip_configs(
    tmp_path: Path,
    results_with_annotations: ResultsTable,
) -> None:
    save_results(results_with_annotations, tmp_path / "out")
    loaded = load_results(tmp_path / "out")
    assert loaded.configs == results_with_annotations.configs


def test_roundtrip_observable_names(
    tmp_path: Path,
    results_with_annotations: ResultsTable,
) -> None:
    save_results(results_with_annotations, tmp_path / "out")
    loaded = load_results(tmp_path / "out")
    assert loaded.observable_names == ["error", "cost"]


def test_roundtrip_annotations(
    tmp_path: Path,
    results_with_annotations: ResultsTable,
) -> None:
    save_results(results_with_annotations, tmp_path / "out")
    loaded = load_results(tmp_path / "out")
    assert loaded.annotations is not None
    np.testing.assert_allclose(loaded.annotations, results_with_annotations.annotations)


def test_roundtrip_annotation_names(
    tmp_path: Path,
    results_with_annotations: ResultsTable,
) -> None:
    save_results(results_with_annotations, tmp_path / "out")
    loaded = load_results(tmp_path / "out")
    assert loaded.annotation_names == ["method_cost"]


def test_roundtrip_metadata(
    tmp_path: Path,
    results_with_annotations: ResultsTable,
) -> None:
    save_results(results_with_annotations, tmp_path / "out")
    loaded = load_results(tmp_path / "out")
    assert loaded.metadata == results_with_annotations.metadata


# ---------------------------------------------------------------------------
# No-annotations path
# ---------------------------------------------------------------------------


def test_roundtrip_no_annotations(
    tmp_path: Path,
    results_without_annotations: ResultsTable,
) -> None:
    save_results(results_without_annotations, tmp_path / "out")
    loaded = load_results(tmp_path / "out")
    assert loaded.annotations is None
    assert loaded.annotation_names == []


def test_no_annotations_scores(
    tmp_path: Path,
    results_without_annotations: ResultsTable,
) -> None:
    save_results(results_without_annotations, tmp_path / "out")
    loaded = load_results(tmp_path / "out")
    np.testing.assert_allclose(loaded.scores, results_without_annotations.scores)


def test_no_annotations_file_absent(
    tmp_path: Path,
    results_without_annotations: ResultsTable,
) -> None:
    save_results(results_without_annotations, tmp_path / "out")
    assert not (tmp_path / "out" / "annotations.npz").exists()


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------


def test_save_creates_directory(tmp_path: Path) -> None:
    rt = ResultsTable(
        configs=[{"a": 1}],
        scores=np.array([[1.0]]),
        observable_names=["m"],
    )
    nested = tmp_path / "deep" / "nested" / "dir"
    save_results(rt, nested)
    assert (nested / "scores.npz").exists()
    assert (nested / "meta.json").exists()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_roundtrip_empty_metadata(tmp_path: Path) -> None:
    rt = ResultsTable(
        configs=[{"a": 1}],
        scores=np.array([[2.0]]),
        observable_names=["m"],
        metadata=[],
    )
    save_results(rt, tmp_path / "out")
    loaded = load_results(tmp_path / "out")
    assert loaded.metadata == []


def test_roundtrip_non_serialisable_config_values(tmp_path: Path) -> None:
    """Config values that are not JSON-native use default=str."""
    from pathlib import PurePosixPath

    rt = ResultsTable(
        configs=[{"path": PurePosixPath("/a/b")}],
        scores=np.array([[1.0]]),
        observable_names=["m"],
    )
    save_results(rt, tmp_path / "out")
    loaded = load_results(tmp_path / "out")
    assert loaded.configs[0]["path"] == "/a/b"


def test_roundtrip_many_observables(tmp_path: Path) -> None:
    n_obs = 10
    rt = ResultsTable(
        configs=[{"x": i} for i in range(5)],
        scores=np.arange(50, dtype=float).reshape(5, n_obs),
        observable_names=[f"obs_{i}" for i in range(n_obs)],
    )
    save_results(rt, tmp_path / "out")
    loaded = load_results(tmp_path / "out")
    np.testing.assert_allclose(loaded.scores, rt.scores)
    assert loaded.observable_names == rt.observable_names


def test_file_structure(
    tmp_path: Path,
    results_with_annotations: ResultsTable,
) -> None:
    """Verify expected files are written."""
    save_results(results_with_annotations, tmp_path / "out")
    written = {f.name for f in (tmp_path / "out").iterdir()}
    assert written == {"scores.npz", "annotations.npz", "meta.json"}
