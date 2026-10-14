"""Versioned result schemas, type preservation and interrupted-write behavior."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from trade_study import (
    Direction,
    Observable,
    ResultsTable,
    load_result_bundle,
    load_results,
    save_result_bundle,
    save_results,
)

if TYPE_CHECKING:
    from pathlib import Path


def _results() -> ResultsTable:
    return ResultsTable(
        configs=[
            {"value": 1, "flag": True, "nested": [None, 0.125, {"name": "x"}]},
            {"value": 2},
        ],
        scores=np.array([[0.25, np.nan], [0.75, np.inf]], dtype=np.float32),
        observable_names=["error", "coverage"],
        annotations=np.array([[3.0], [4.0]]),
        annotation_names=["cost"],
        metadata=[{"replicate": 0, "seed": 3}, {"replicate": 1, "seed": 4}],
    )


def _observables() -> list[Observable]:
    return [
        Observable("error", Direction.MINIMIZE),
        Observable("coverage", Direction.MAXIMIZE, weight=0.5),
    ]


def _write(path: Path, results: ResultsTable | None = None) -> None:
    save_result_bundle(
        _results() if results is None else results,
        path,
        observables=_observables(),
        provenance={
            "model_id": "synthetic-v1",
            "score_transforms": {"error": "identity", "coverage": "indicator"},
            "root_seed": 3,
        },
    )


def _manifest(path: Path) -> dict[str, Any]:
    return json.loads((path / "meta.json").read_text(encoding="utf-8"))


def _replace_manifest(path: Path, meta: dict[str, Any]) -> None:
    (path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_versioned_roundtrip_preserves_data_and_interpretation(tmp_path: Path) -> None:
    destination = tmp_path / "versioned"
    _write(destination)
    loaded = load_result_bundle(destination)
    assert loaded.schema_version == 1
    assert loaded.results.configs == _results().configs
    assert type(loaded.results.configs[0]["value"]) is int
    assert type(loaded.results.configs[0]["flag"]) is bool
    assert loaded.results.metadata == _results().metadata
    np.testing.assert_array_equal(loaded.results.scores, _results().scores)
    assert loaded.results.scores.dtype == np.float32
    np.testing.assert_array_equal(loaded.results.annotations, [[3.0], [4.0]])
    assert loaded.observables == tuple(_observables())
    assert loaded.provenance["caller"]["model_id"] == "synthetic-v1"
    assert loaded.provenance["runtime"]["numpy"] == np.__version__
    assert set(_manifest(destination)) == {
        "schema_version",
        "encoding",
        "observable_names",
        "annotation_names",
        "configs",
        "metadata",
        "observables",
        "provenance",
        "arrays",
    }
    assert {item.name for item in destination.iterdir()} == {
        "scores.npz",
        "annotations.npz",
        "meta.json",
    }
    np.testing.assert_array_equal(
        load_results(destination).scores, loaded.results.scores
    )


def test_empty_rows_metadata_and_optional_annotations(tmp_path: Path) -> None:
    for index, results in enumerate((
        replace(_results(), metadata=[], annotations=None, annotation_names=[]),
        ResultsTable(
            configs=[], scores=np.empty((0, 2)), observable_names=["error", "coverage"]
        ),
    )):
        destination = tmp_path / str(index)
        _write(destination, results)
        loaded = load_result_bundle(destination)
        assert loaded.results.metadata == []
        assert loaded.results.annotations is None
        assert not (destination / "annotations.npz").exists()
        assert loaded.results.scores.shape == results.scores.shape


@pytest.mark.parametrize(
    "value",
    [
        PurePosixPath("/data"),
        (1, 2),
        np.int64(1),
        np.float64(1.0),
        np.array([1.0]),
        object(),
        float("nan"),
        float("inf"),
        {1: "bad key"},
    ],
)
def test_unsupported_inputs_fail_before_destination_creation(
    tmp_path: Path, value: object
) -> None:
    destination = tmp_path / "unsupported"
    results = replace(_results(), configs=[{"x": value}, {}])
    with pytest.raises(ValueError, match="replayable result inputs"):
        _write(destination, results)
    assert not destination.exists()


@pytest.mark.parametrize(
    "change",
    [
        {"configs": [{}]},
        {"scores": np.ones(2)},
        {"scores": np.ones((2, 1))},
        {"scores": np.ones((2, 2), dtype=complex)},
        {"observable_names": ["error", "error"]},
        {"metadata": [{}]},
        {"metadata": ["invalid", {}]},
        {"annotations": np.ones((1, 1))},
        {"annotation_names": []},
        {"annotations": None},
    ],
)
def test_inconsistent_tables_fail_before_writing(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="Cannot save replayable result inputs"):
        _write(tmp_path / "invalid", replace(_results(), **change))
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize(
    "observables",
    [
        [
            Observable("coverage", Direction.MAXIMIZE),
            Observable("error", Direction.MINIMIZE),
        ],
        [
            Observable("error", Direction.MINIMIZE, weight=-1),
            Observable("coverage", Direction.MAXIMIZE),
        ],
        [
            Observable("error", Direction.MINIMIZE, weight=float("inf")),
            Observable("coverage", Direction.MAXIMIZE),
        ],
    ],
)
def test_observable_order_and_weights_are_checked(
    tmp_path: Path, observables: list[Observable]
) -> None:
    with pytest.raises(ValueError, match="observable"):
        save_result_bundle(_results(), tmp_path / "invalid", observables=observables)
    assert not (tmp_path / "invalid").exists()


def test_replaced_same_shape_scores_fail_digest_check(tmp_path: Path) -> None:
    destination = tmp_path / "versioned"
    _write(destination)
    np.savez_compressed(
        destination / "scores.npz", scores=np.zeros((2, 2), dtype=np.float32)
    )
    with pytest.raises(ValueError, match="does not match its manifest"):
        load_result_bundle(destination)


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"encoding": "unknown"},
        {"configs": [{}]},
        {"observable_names": ["error", "error"]},
        {"metadata": [{}]},
        {"provenance": {}},
        {"observables": []},
    ],
)
def test_manifest_inconsistency_is_rejected(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    destination = tmp_path / "versioned"
    _write(destination)
    meta = _manifest(destination)
    meta.update(change)
    _replace_manifest(destination, meta)
    with pytest.raises(ValueError, match="Invalid result bundle"):
        load_result_bundle(destination)


@pytest.mark.parametrize(
    ("shape", "dtype"), [([1, 2], "<f4"), ([True, 2], "<f4"), ([2, 2], "<f8")]
)
def test_array_shape_and_dtype_metadata_must_match(
    tmp_path: Path, shape: object, dtype: str
) -> None:
    destination = tmp_path / "versioned"
    _write(destination)
    meta = _manifest(destination)
    meta["arrays"]["scores"].update(shape=shape, dtype=dtype)
    _replace_manifest(destination, meta)
    with pytest.raises(ValueError, match=r"shape or dtype|two nonnegative integers"):
        load_result_bundle(destination)


def test_missing_schema_version_does_not_downgrade_a_versioned_bundle(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "versioned"
    _write(destination)
    meta = _manifest(destination)
    del meta["schema_version"]
    _replace_manifest(destination, meta)
    with pytest.raises(ValueError, match="schema_version may be missing"):
        load_result_bundle(destination)


def test_undeclared_annotation_file_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "versioned"
    _write(destination)
    meta = _manifest(destination)
    meta["arrays"]["annotations"] = None
    _replace_manifest(destination, meta)
    with pytest.raises(ValueError, match="Undeclared annotation file"):
        load_result_bundle(destination)


def test_interrupted_initial_write_has_no_committed_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "partial"
    original = np.savez_compressed
    calls = 0

    def fail_second_write(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            msg = "synthetic interrupted annotation write"
            raise OSError(msg)
        original(*args, **kwargs)

    monkeypatch.setattr(np, "savez_compressed", fail_second_write)
    with pytest.raises(OSError, match="interrupted"):
        _write(destination)
    assert (destination / "scores.npz").exists()
    assert not (destination / "meta.json").exists()
    with pytest.raises(FileNotFoundError):
        load_result_bundle(destination)
    with pytest.raises(FileExistsError):
        _write(destination)


def test_existing_bundle_is_never_overwritten(tmp_path: Path) -> None:
    destination = tmp_path / "existing"
    _write(destination)
    before = {path.name: path.read_bytes() for path in destination.iterdir()}
    with pytest.raises(FileExistsError):
        _write(destination, replace(_results(), scores=np.zeros((2, 2))))
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == before


def test_legacy_export_cannot_downgrade_a_committed_bundle(tmp_path: Path) -> None:
    destination = tmp_path / "existing"
    _write(destination)
    before = {path.name: path.read_bytes() for path in destination.iterdir()}
    with pytest.raises(ValueError, match="cannot overwrite a versioned"):
        save_results(_results(), destination)
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == before


def test_cyclic_configs_fail_before_creating_destination(tmp_path: Path) -> None:
    cyclic: dict[str, object] = {}
    cyclic["loop"] = cyclic
    with pytest.raises(ValueError, match="replayable result inputs"):
        _write(tmp_path / "cyclic", replace(_results(), configs=[cyclic, {}]))
    assert not (tmp_path / "cyclic").exists()


def test_legacy_export_retains_its_explicit_missing_interpretation(
    tmp_path: Path,
) -> None:
    results = replace(_results(), configs=[{"path": PurePosixPath("/a/b")}, {}])
    save_results(results, tmp_path / "legacy")
    loaded = load_result_bundle(tmp_path / "legacy")
    assert loaded.schema_version == 0
    assert loaded.observables == ()
    assert loaded.provenance == {}
    assert loaded.results.configs[0]["path"] == "/a/b"
    np.testing.assert_array_equal(loaded.results.scores, results.scores)


def test_legacy_parallel_row_inconsistency_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "legacy"
    save_results(_results(), destination)
    np.savez_compressed(destination / "annotations.npz", annotations=np.ones((3, 1)))
    with pytest.raises(ValueError, match="annotations must"):
        load_results(destination)


def test_legacy_writer_can_repair_interrupted_legacy_metadata(tmp_path: Path) -> None:
    destination = tmp_path / "legacy"
    save_results(_results(), destination)
    (destination / "meta.json").write_text("{", encoding="utf-8")
    replacement = replace(_results(), scores=np.ones((2, 2)))
    save_results(replacement, destination)
    np.testing.assert_array_equal(load_results(destination).scores, replacement.scores)
